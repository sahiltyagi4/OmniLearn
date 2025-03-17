import logging
import threading
import math
from time import perf_counter_ns
import os

import torch
import numpy as np
import torch.distributed.rpc as rpc

import pytorch.helper.miscellaneous as misc
import pytorch.helper.data_partitioner as dp
from pytorch.helper import models
from pytorch.helper.dynamicbatching import DynamicHeterogeneityEmulator

class ASPOmniLearnPS(object):
    def __init__(self, args):
        self.args = args
        self.logdir = args.dir
        self.train_bsz = args.bsz
        self.model_name = args.model
        self.rank = args.rank
        self.worldsize = args.world_size
        # world-size comprises 1 PS and multiple workers
        self.num_workers = self.worldsize - 1
        self.datadir = args.data_dir
        self.dataset_name = args.dataset
        self.determinism = args.determinism
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(self.rank))
        else:
            self.device = torch.device("cpu")

        logging.basicConfig(filename=self.logdir + '/g' + str(self.rank) + '/' + self.model_name
                                     + '-' + str(self.rank) + '.log', level=logging.INFO)

        # initial worker batches supplied as comma-separated integers
        self.bszlist = [int(bsz) for bsz in self.args.bszlist.split(',')]
        self.dataset_obj = dp.TrainingTestingDataset(bsz=self.train_bsz, dataset_name=self.dataset_name,
                                                     args=args, fetchtestdata=True)
        self.model_obj = models.get_model(model_name=self.model_name, determinism=self.determinism, args=args)
        self.model = self.model_obj.get_model().to(self.device)

        self.model_lock = threading.Lock()
        self.model_future = torch.futures.Future()
        self.step_lock = threading.Lock()
        self.step_future = torch.futures.Future()
        self.epoch_lock = threading.Lock()
        self.epoch_future = torch.futures.Future()

        self.computetime_lock = threading.Lock()
        self.computetime_future = torch.futures.Future()
        self.worker_compute_times, self.worker_bszs = {}, {}
        self.new_worker_bszs = {}
        self.compute_time_count = 0
        self.deadband_threshold = args.deadband_threshold
        self.min_bsz, self.max_bsz = args.min_bsz, args.max_bsz

        self.global_step = 0
        self.curr_epoch = 0
        self.total_samples = 0
        logging.info(f'model configuration {args}')

    def push_model_parameters(self, model_params, rank):
        self.model_lock.acquire()
        model_params = model_params.to(self.device)
        for param in self.model.parameters():
            p_shape = param.size()
            p_length = param.numel()
            worker_param = model_params[0:p_length]
            model_params = worker_param[p_length:]
            param.data = worker_param.reshape(p_shape)
            param.grad = torch.zeros_like(param.grad)

        self.global_step += 1
        self.total_samples += self.bszlist[rank]
        self.curr_epoch = math.ceil(self.total_samples / self.dataset_obj.getTrainsize())
        logging.info(f'pushing model update at step {self.global_step} from rank {rank}')
        self.model_lock.release()

    @staticmethod
    @rpc.functions.async_execution
    def get_global_step(ps_rref):
        self = ps_rref.local_value()
        self.step_lock.acquire()
        step_future = self.step_future
        step_future.set_result(self.global_step)
        self.step_future = torch.futures.Future()
        self.step_lock.release()
        return step_future

    @staticmethod
    @rpc.functions.async_execution
    def get_epoch(ps_rref):
        self = ps_rref.local_value()
        self.epoch_lock.acquire()
        epoch_future = self.epoch_future
        epoch_future.set_result(self.curr_epoch)
        self.epoch_future = torch.futures.Future()
        self.epoch_lock.release()
        return epoch_future

    @staticmethod
    @rpc.functions.async_execution
    def pull_model_parameters(ps_rref):
        self = ps_rref.local_value()
        self.model_lock.acquire()
        model_params = torch.cat([p.data.view(-1) for p in self.model.parameters()]).reshape(-1).to(torch.device("cpu"))
        model_future = self.model_future
        model_future.set_result(model_params)
        self.model_future = torch.futures.Future()
        self.model_lock.release()
        return model_future

    @staticmethod
    @rpc.functions.async_execution
    def compute_bsz(ps_rref, worker_rank, worker_bsz, avg_compute_time):
        self = ps_rref.local_value()
        self.worker_compute_times[worker_rank] = avg_compute_time
        self.worker_bszs[worker_rank] = worker_bsz
        with self.computetime_lock:
            self.compute_time_count += 1
            compute_time_fut = self.computetime_future

            if self.compute_time_count >= self.num_workers:
                mean_compute_times = list(self.worker_compute_times.values())
                mean_time = np.mean(mean_compute_times)
                for rank, time in self.worker_compute_times.items():
                    delta_b = - (self.worker_bszs[rank] / time) * (time - mean_time)
                    potential_bsz = self.worker_bszs[rank] + delta_b
                    bsz_change = abs((potential_bsz - self.worker_bszs[rank]) / potential_bsz)
                    if bsz_change >= self.deadband_threshold:
                        candidate_bsz = self.worker_bszs[rank] + delta_b
                    else:
                        candidate_bsz = self.worker_bszs[rank]

                    if candidate_bsz < self.min_bsz:
                        bounded_bsz = self.min_bsz_bound
                    elif candidate_bsz > self.max_bsz:
                        bounded_bsz = self.max_bsz
                    else:
                        bounded_bsz = candidate_bsz

                    self.new_worker_bszs[rank] = math.ceil(bounded_bsz)

                compute_time_fut.set_result(self.new_worker_bszs)
                self.computetime_future = torch.futures.Future()
                self.compute_time_count = 0

        return compute_time_fut


class ASPOmniLearnWorker(object):
    def __init__(self, ps_rref, rank, args):
        self.args = args
        self.ps_rref = ps_rref
        self.bszlist = args.bszlist
        self.train_bsz = args.bsz
        self.test_bsz = args.test_bsz
        self.logdir = args.dir
        self.model_name = args.model
        self.seed = args.seed
        self.train_dir = args.train_dir
        self.rank = rank
        self.worldsize = args.world_size
        self.datadir = args.data_dir
        self.dataset_name = args.dataset
        self.determinism = args.determinism
        self.train_freq = args.trainfreq
        self.test_freq = args.testfreq
        self.comptime_list = []
        self.deadband_threshold = args.deadband_threshold
        self.min_bsz, self.max_bsz = args.min_bsz, args.max_bsz
        self.global_step = 0
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(self.rank))
        else:
            self.device = torch.device("cpu")

        logging.basicConfig(filename=self.logdir + '/g' + str(self.rank) + '/' + self.model_name
                                     + '-' + str(self.rank) + '.log', level=logging.INFO)

        self.cpu_logfile = os.path.join(self.logdir, 'cpu-' + str(self.rank - 1) + '.log')
        self.sync_mode = 'BSP' if args.bsp else 'ASP'
        self.dynamicHL = DynamicHeterogeneityEmulator(model_name=self.model_name, sync_mode=self.sync_mode,
                                                      cpulog_file=self.cpu_logfile, id=int(self.rank))

        self.dataset_obj = dp.TrainingTestingDataset(bsz=self.train_bsz, dataset_name=self.dataset_name,
                                                     args=args, fetchtestdata=True)
        self.trainloader = self.dataset_obj.getTrainloader()
        self.testloader = self.dataset_obj.getTestloader()

        self.model_obj = models.get_model(model_name=self.model_name, determinism=self.determinism, args=args)
        self.model = self.model_obj.get_model().to(self.device)
        self.loss = self.model_obj.get_loss()
        self.opt = self.model_obj.get_optim()
        self.lr_scheduler = self.model_obj.get_lrscheduler()
        self.lr_milestones = self.model_obj.get_milestones()
        self.lr_gamma = self.model_obj.get_lrgamma()
        logging.info(f'model configuration {args}')

    def train(self):
        self.top1acc, self.top5acc, self.train_loss = misc.AverageMeter(), misc.AverageMeter(), misc.AverageMeter()
        train_reloader, updated_bsz = None, None
        curr_epoch, prev_epoch, lrtrigger_epoch = 0, 0, 0

        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)

        while True:
            pull_model = rpc.rpc_sync(self.ps_rref.owner(), ASPOmniLearnPS.pull_model_parameters,
                                         args=(self.ps_rref,))
            pull_model = pull_model.to(self.device)

            for param in self.model.parameters():
                p_shape = param.size()
                p_length = param.numel()
                p = pull_model[0:p_length]
                pull_model = pull_model[p_length:]
                param.data = p.reshape(p_shape)
                param.grad = torch.zeros_like(param.data)

            if train_reloader is not None:
                self.trainloader = train_reloader
                train_reloader = None

            for input, label in self.trainloader:
                input, label = input.to(self.device), label.to(self.device)
                begin = perf_counter_ns()
                output = self.model(input)
                loss = self.loss(output, label)
                loss.backward()
                compute_time = (perf_counter_ns() - begin) / 1e6
                self.comptime_list.append(compute_time)

                self.opt.step()
                self.opt.zero_grad()

                begin = perf_counter_ns()
                self.ps_rref.rpc_sync().push_model_parameters(
                    concat_params=torch.cat([p.data.view(-1) for p in self.model.parameters()]).reshape(-1).to(torch.device("cpu")),
                    rank=self.rank)

                self.global_step = rpc.rpc_sync(self.ps_rref.owner(), ASPOmniLearnPS.get_global_step,
                                                args=(self.ps_rref,))
                epoch = rpc.rpc_sync(self.ps_rref.owner(), ASPOmniLearnPS.get_epoch, args=(self.ps_rref,))
                comm_overhead = (perf_counter_ns() - begin) / 1e6

                logging.info(f'asp traininig at step {self.global_step} epoch {epoch} compute_time {compute_time} ms '
                             f'comm_time {comm_overhead} ms')

                self.train_accuracy(epoch=epoch, input=input, label=label, output=output, loss=loss)
                self.test_accuracy(epoch=epoch)

                if epoch in self.lr_milestones and lrtrigger_epoch != epoch:
                    self.opt.param_groups[0]['lr'] *= self.lr_gamma
                    lrtrigger_epoch = epoch

                if prev_epoch != epoch:
                    self.top1accs, self.top5accs = misc.AverageMeter(), misc.AverageMeter()
                    self.train_loss = misc.AverageMeter()
                    prev_epoch = epoch
                    train_reloader = self.evaluate_bsz(epoch=epoch)
                    self.dynamicHL.triggerHLadjustment(curr_epoch=epoch)
                    break

    def train_accuracy(self, epoch, input, label, output, loss):
        if self.global_step >= self.train_freq and self.global_step % self.train_freq == 0:
            with torch.no_grad():
                trainaccs = misc.compute_accuracy(output=output, target=label, topk=(1, 5))
                self.top1acc.update(trainaccs[0], input.size(0))
                self.top5acc.update(trainaccs[1], input.size(0))
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
                    topaccs = misc.compute_accuracy(output=output, target=label, topk=(1, 5))
                    top1acc.update(topaccs[0], input.size(0))
                    top5acc.update(topaccs[1], input.size(0))
                    test_loss.update(loss.item(), input.size(0))
                    logging.info(f'test metrics at step {self.global_step} epoch {epoch} test_loss '
                                 f'{test_loss.avg} top1avg {top1acc.avg.cpu().numpy().item()} '
                                 f'top5avg {top5acc.avg.cpu().numpy().item()}')

            self.model.train()

    def evaluate_bsz(self, epoch):
        avg_compute_time = np.mean(self.comptime_list)
        self.comptime_list = []
        worker_batches = rpc.rpc_sync(self.ps_rref.owner(), ASPOmniLearnPS.compute_bsz,
                                   args=(self.ps_rref, self.rank, self.train_bsz, avg_compute_time,))
        self.train_bsz = worker_batches[self.rank]

        dataset_obj = dp.TrainingTestingDataset(bsz=self.train_bsz, dataset_name=self.dataset_name,
                                                     args=self.args, fetchtestdata=False)
        logging.info(f'OmniLearn ASP batch-size at step {self.global_step} epoch {epoch} is {self.train_bsz}')

        return dataset_obj.getTrainloader()