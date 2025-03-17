import math
import datetime
import logging
from time import perf_counter_ns
import numpy as np
import os

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp

from omnilearn_pytorch.helper import data_partitioner as dp
from omnilearn_pytorch.helper import models
from omnilearn_pytorch.helper import miscellaneous as misc
from omnilearn_pytorch.helper.dynamicbatching import DynamicHeterogeneityEmulator

class BSPOmniLearn(object):
    def __init__(self, args):
        self.args = args
        self.train_bsz = args.bsz
        self.test_bsz = args.test_bsz
        self.logdir = args.dir
        self.model_name = args.model
        self.rank = args.rank
        self.worldsize = args.world_size
        self.backend = args.backend
        self.init_method = args.init_method
        self.globalB = args.globalB
        self.seed = args.seed
        self.train_freq = args.trainfreq
        self.test_freq = args.testfreq
        self.top1acc, self.top5acc = None, None
        self.deadband_threshold = args.deadband_threshold
        self.min_bsz, self.max_bsz = args.min_bsz, args.max_bsz
        self.global_step = 0
        self.cpu_logfile = os.path.join(self.logdir, 'cpu-' + str(self.rank) + '.log')
        self.sync_mode = 'BSP' if args.bsp else 'ASP'
        self.dynamicHL = DynamicHeterogeneityEmulator(model_name=self.model_name, sync_mode=self.sync_mode,
                                                      cpulog_file=self.cpu_logfile, id=int(self.rank))

        if self.init_method == 'sharedfile':
            sharedfile = 'file://' + args.shared_file
            dist.init_process_group(backend=self.backend, init_method=sharedfile, rank=self.rank, world_size=self.worldsize)
        elif self.init_method == 'tcp':
            timeout = datetime.timedelta(seconds=60 * 60)
            tcp_addr = 'tcp://' + str(args.master_addr) + ':' + str(args.master_port)
            dist.init_process_group(backend=self.backend, init_method=tcp_addr, rank=self.rank, world_size=self.worldsize, timeout=timeout)

        logging.basicConfig(filename=self.logdir + '/g' + str(self.rank) + '/' + self.model_name + '-'
                                     + str(self.rank) + '.log', level=logging.INFO)

        self.dataset_name = args.dataset
        self.determinism = args.determinism
        self.async_op = args.asyncop
        self.measured_computetime = []
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(self.rank))
        else:
            self.device = torch.device("cpu")

        self.model_obj = models.get_model(model_name=self.model_name, determinism=self.determinism, args=args)
        self.model = self.model_obj.get_model().to(self.device)
        self.loss = self.model_obj.get_loss()
        self.opt = self.model_obj.get_optim()
        self.lr_scheduler = self.model_obj.get_lrscheduler()
        self.lr_milestones = self.model_obj.get_milestones()
        self.lr_gamma = self.model_obj.get_lrgamma()
        self.gradient_scaling = self.train_bsz / self.globalB

        # get both training and test data initially
        self.dataset_obj = dp.TrainingTestingDataset(bsz=self.train_bsz, dataset_name=self.dataset_name,
                                                     args=args, fetchtestdata=True)
        self.trainloader = self.dataset_obj.getTrainloader()
        self.testloader = self.dataset_obj.getTestloader()
        logging.info(f'model configuration {args}')

    def train(self):
        train_reloader = None
        previous_epoch, lr_trigger_epoch = 0, 0
        for _, param in self.model.named_parameters():
            dist.broadcast(tensor=param.data, src=0, async_op=self.async_op)

        self.top1acc, self.top5acc, self.train_loss = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()

        while True:
            if train_reloader is not None:
                self.trainloader = train_reloader
                train_reloader = None

            for input, label in self.trainloader:
                input, label = input.to(self.device), label.to(self.device)
                self.global_step += 1
                epoch = int(math.floor((self.global_step * self.globalB) / self.dataset_obj.getTrainsize()))
                begin = perf_counter_ns()
                output = self.model(input)
                loss = self.loss(output, label)
                loss.backward()
                # measure compute time in milliseconds
                compute_time = (perf_counter_ns() - begin) / 1e6
                self.measured_computetime.append(compute_time)

                for param in self.model.parameters():
                    param.grad *= self.gradient_scaling

                begin = perf_counter_ns()
                for param in self.model.parameters():
                    dist.all_reduce(tensor=param.grad, op=ReduceOp.SUM, async_op=self.async_op)
                sync_time = (perf_counter_ns() - begin) / 1e6

                self.opt.step()
                self.opt.zero_grad()

                logging.info(f'bsp training at step {self.global_step} epoch {epoch} compute_time {compute_time} ms '
                             f'sync_time {sync_time} ms and grad_scaling factor {self.gradient_scaling}')

                self.train_accuracy(input=input, label=label, output=output, loss=loss, epoch=epoch)
                self.test_accuracy(epoch=epoch)
                del input, label

                if epoch in self.lr_milestones and lr_trigger_epoch != epoch:
                    self.opt.param_groups[0]['lr'] *= self.lr_gamma
                    lr_trigger_epoch = epoch

                if previous_epoch != epoch:
                    self.top1acc, self.top5acc = misc.AverageMeter(), misc.AverageMeter()
                    previous_epoch = epoch
                    # trigger batch-size adjustments based on workers' computation times
                    train_reloader = self.evaluate_bsz(epoch=epoch)
                    # trigger heterogeneity change at different epochs
                    self.dynamicHL.triggerHLadjustment(curr_epoch=epoch)
                    break

    def train_accuracy(self, input, label, output, loss, epoch):
        if self.global_step >= self.train_freq and self.global_step % self.train_freq == 0:
            with torch.no_grad():
                train_acc = misc.compute_accuracy(output=output, target=label, topk=(1, 5))
                self.top1acc.update(train_acc[0], input.size(0))
                self.top5acc.update(train_acc[1], input.size(0))
                self.train_loss.update(loss.item(), input.size(0))
                logging.info(f'training metrics at step {self.global_step} epoch {epoch} avg_loss '
                             f'{self.train_loss.avg} avg_top1acc {self.top1acc.avg.cpu().numpy().item()} '
                             f'top5avg {self.top5acc.avg.cpu().numpy().item()}')

    def test_accuracy(self, epoch):
        if self.global_step >= self.test_freq and self.global_step % self.test_freq == 0:
            self.model.eval()
            with torch.no_grad():
                top1acc, top5acc, test_loss = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()
                for input, label in self.testloader:
                    input, label = input.to(self.device), label.to(self.device)
                    output = self.model(input)
                    loss = self.loss(output, label)
                    test_acc = misc.compute_accuracy(output=output, target=label, topk=(1, 5))
                    top1acc.update(test_acc[0], input.size(0))
                    top5acc.update(test_acc[1], input.size(0))
                    test_loss.update(loss.item(), input.size(0))
                    logging.info(f'test metrics at step {self.global_step} epoch {epoch} test_loss '
                                 f'{test_loss.avg} top1avg {top1acc.avg.cpu().numpy().item()} '
                                 f'top5avg {top5acc.avg.cpu().numpy().item()}')

            self.model.train()

    def evaluate_bsz(self, epoch):
        avg_compute_time = torch.FloatTensor([np.mean(self.measured_computetime)])
        self.measured_computetime = []
        avg_compute_time = avg_compute_time.to(torch.device("cpu"))
        worker_compute_times = [torch.FloatTensor([0.]) for _ in range(self.worldsize)]
        dist.all_gather(tensor_list=worker_compute_times, tensor=avg_compute_time, async_op=self.async_op)
        worker_compute_times = [t.item() for t in worker_compute_times]
        avg_cluster_time = np.mean(worker_compute_times)
        delta_b = - (self.train_bsz / avg_compute_time.item()) * (avg_compute_time.item() - avg_cluster_time)
        candidate_bsz = self.train_bsz + delta_b
        bsz_change = abs((candidate_bsz - self.train_bsz) / candidate_bsz)

        if bsz_change >= self.deadband_threshold:
            logging.info(f'deadbanding with candidate_bsz to {candidate_bsz} from old bsz {self.train_bsz} '
                         f'at step {self.global_step} epoch {epoch} delta_b {delta_b} bsz_change {bsz_change}')
        else:
            candidate_bsz = self.train_bsz
            logging.info(f'not updating bsz at step {self.global_step} epoch {epoch} delta_b {delta_b} '
                         f'bsz_change {bsz_change}')

        updated_bsz, bsz_list = self.adjust_bsz_bounds(candidate_bsz=candidate_bsz)
        self.train_bsz = math.ceil(updated_bsz)
        bsz_tensor_list = [torch.zeros(size=(self.worldsize,), dtype=torch.float32, device=torch.device("cpu")) for _ in
                           range(self.worldsize)]
        dist.all_gather(tensor_list=bsz_tensor_list, tensor=bsz_list, async_op=self.async_op)
        bsz_adjustments = [bslist.tolist()[self.rank] for bslist in bsz_tensor_list]
        self.train_bsz += np.sum(bsz_adjustments)
        adjusted_bsz = torch.FloatTensor([float(self.train_bsz)]).to(torch.device("cpu"))
        dist.all_reduce(tensor=adjusted_bsz, op=ReduceOp.SUM, async_op=self.async_op)
        updated_globalB = math.ceil(adjusted_bsz.item())
        globalB_adjustment = (updated_globalB - self.globalB) / self.worldsize
        self.train_bsz = math.ceil(self.train_bsz - globalB_adjustment)
        updated_globalB = torch.FloatTensor([float(self.train_bsz)]).to(torch.device("cpu"))
        dist.all_reduce(tensor=updated_globalB, op=ReduceOp.SUM, async_op=self.async_op)
        updated_globalB = math.ceil(updated_globalB.item())
        self.gradient_scaling = self.train_bsz / updated_globalB
        dataset_obj = dp.TrainingTestingDataset(bsz=self.train_bsz, dataset_name=self.dataset_name,
                                                args=self.args, fetchtestdata=False)

        logging.info(f'bsz adjustment at step {self.global_step} epoch {epoch} for avg_compute_time '
                     f'{avg_compute_time} ms updated_globalB {updated_globalB} avg_cluster_time '
                     f'{avg_cluster_time} ms worker_avg_compute_times {worker_compute_times} train_bsz '
                     f'{self.train_bsz} and grad_scaling {self.gradient_scaling} for worker rank {self.rank}')

        return dataset_obj.getTrainloader()

    def adjust_bsz_bounds(self, candidate_bsz):
        if candidate_bsz < self.min_bsz:
            bs_diff = candidate_bsz - self.min_bsz
            bounded_bsz = self.min_bsz
            logging.info(f'bsz smaller than minimum allowed bsz {self.global_step} by {bs_diff}')

        elif candidate_bsz > self.max_bsz:
            bs_diff = candidate_bsz - self.max_bsz
            bounded_bsz = self.max_bsz
            logging.info(f'bsz greater than maximum allowed bsz {self.global_step} by {bs_diff}')

        else:
            bs_diff = 0
            bounded_bsz = candidate_bsz

        bsz_list = torch.FloatTensor([math.ceil(bs_diff / self.worldsize) for _ in range(self.worldsize)])
        bsz_list = bsz_list.to(torch.device("cpu"))

        return bounded_bsz, bsz_list