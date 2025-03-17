import datetime
import logging
import os
from time import perf_counter_ns
import math

import requests
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import torch.distributed as dist
from transformers import GPT2Tokenizer
from torch._C._distributed_c10d import ReduceOp

from omnilearn_pytorch.helper import models
from omnilearn_pytorch.helper import DynamicHeterogeneityEmulator

# Implements GPT-2 BSP training over OmniLearn

# Dataset and DataLoader
class CharDataset(Dataset):
    def __init__(self, encoded_text, seq_length):
        self.data = encoded_text
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_length]
        y = self.data[idx + 1: idx + self.seq_length + 1]
        return x, y


class GPT2OmniLearnBSP(object):
    def __init__(self, args):
        self.model_name = "gpt2"
        self.args = args
        self.rank = args.rank
        self.world_size = args.world_size
        self.master_addr = args.master_addr
        self.master_port = args.master_port
        self.backend = args.backend
        self.train_bsz = args.bsz
        self.seq_length = args.seq_length
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.logdir = args.dir
        self.determinism = args.determinism
        self.async_op = args.asyncop
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(self.rank))
        else:
            self.device = torch.device("cpu")

        timeout = datetime.timedelta(seconds=60 * 60)
        tcp_addr = 'tcp://' + str(self.master_addr) + ':' + str(self.master_port)
        dist.init_process_group(backend=self.backend, init_method=tcp_addr, rank=self.rank,
                                world_size=self.world_size, timeout=timeout)
        logging.basicConfig(filename=self.logdir + '/g' + str(self.rank) + '/' + self.model_name + '-'
                                     + str(self.rank) + '.log', level=logging.INFO)

        self.model_obj = models.get_model(model_name=self.model_name, determinism=self.determinism, args=args)
        self.model = self.model_obj.get_model().to(self.device)
        self.opt = self.model_obj.get_optim()
        self.cpu_logfile = os.path.join(self.logdir, 'cpu-' + str(self.rank) + '.log')
        self.sync_mode = 'BSP' if args.bsp else 'ASP'
        self.dynamicHL = DynamicHeterogeneityEmulator(model_name=self.model_name, sync_mode=self.sync_mode,
                                                      cpulog_file=self.cpu_logfile, id=int(self.rank) + 1)

        self.global_step = 0
        self.measured_computetime = []
        self.globalB = self.train_bsz * self.world_size
        self.gradient_scaling = self.train_bsz / self.globalB
        self.deadband_threshold = args.deadband_threshold
        self.min_bsz = args.min_bsz
        self.max_bsz = args.max_bsz
        logging.info(f'model configuration {args}')

    def train(self):
        # Load GPT-2 model and tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)

        # Adjust tokenizer to work at character level
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        vocab_size = tokenizer.vocab_size

        # Load Shakespeare's text data
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        text = requests.get(url).text

        # Tokenize text for character-level modeling
        encoded_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)["input_ids"].squeeze()

        self.dataset = CharDataset(encoded_text, self.seq_length)
        dataloader = DataLoader(self.dataset, batch_size=self.train_bsz, shuffle=True)
        dataset_size = len(dataloader.dataset)
        train_reloader = None
        prev_epoch = 0
        while True:
            if train_reloader is not None:
                dataloader = train_reloader
                train_reloader = None

            for input, target in dataloader:
                input, target = input.to(self.device), target.to(self.device)
                self.global_step += 1
                epoch = int(math.floor((self.global_step * self.globalB) / dataset_size))
                begin = perf_counter_ns()
                outputs = self.model(input, labels=target)
                loss = outputs.loss
                loss.backward()
                compute_time = (perf_counter_ns() - begin) / 1e6
                self.measured_computetime.append(compute_time)

                for param in self.model.parameters():
                    param.grad *= self.gradient_scaling

                begin = perf_counter_ns()
                for param in self.model.parameters():
                    dist.all_reduce(tensor=param.grad, op=ReduceOp.SUM, async_op=False)
                sync_time = (perf_counter_ns() - begin) / 1e6

                self.opt.step()
                self.opt.zero_grad()
                logging.info(f'bsp gpt2 training step {self.global_step} epoch {epoch} compute_time {compute_time} ms '
                             f'sync_time {sync_time} ms and grad_scaling factor {self.gradient_scaling}')

                if prev_epoch != epoch:
                    train_reloader = self.evaluate_bsz(epoch=epoch)
                    # uncomment the following to trigger heterogeneity change at different epochs
                    self.dynamicHL.triggerHLadjustment(curr_epoch=epoch)
                    break

    def evaluate_bsz(self, epoch):
        avg_compute_time = torch.FloatTensor([np.mean(self.measured_computetime)])
        self.measured_computetime = []
        avg_compute_time = avg_compute_time.to(torch.device("cpu"))
        worker_compute_times = [torch.FloatTensor([0.]) for _ in range(self.world_size)]
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
        bsz_tensor_list = [torch.zeros(size=(self.world_size,), dtype=torch.float32, device=torch.device("cpu")) for _ in
                           range(self.world_size)]
        dist.all_gather(tensor_list=bsz_tensor_list, tensor=bsz_list, async_op=self.async_op)
        bsz_adjustments = [bslist.tolist()[self.rank] for bslist in bsz_tensor_list]
        self.train_bsz += np.sum(bsz_adjustments)
        adjusted_bsz = torch.FloatTensor([float(self.train_bsz)]).to(torch.device("cpu"))
        dist.all_reduce(tensor=adjusted_bsz, op=ReduceOp.SUM, async_op=self.async_op)
        updated_globalB = math.ceil(adjusted_bsz.item())
        globalB_adjustment = (updated_globalB - self.globalB) / self.world_size
        self.train_bsz = math.ceil(self.train_bsz - globalB_adjustment)
        updated_globalB = torch.FloatTensor([float(self.train_bsz)]).to(torch.device("cpu"))
        dist.all_reduce(tensor=updated_globalB, op=ReduceOp.SUM, async_op=self.async_op)
        updated_globalB = math.ceil(updated_globalB.item())
        self.gradient_scaling = self.train_bsz / updated_globalB
        dataloader = DataLoader(self.dataset, batch_size=self.train_bsz, shuffle=True)

        logging.info(f'bsz adjustment at step {self.global_step} epoch {epoch} for avg_compute_time '
                     f'{avg_compute_time} ms updated_globalB {updated_globalB} avg_cluster_time '
                     f'{avg_cluster_time} ms worker_avg_compute_times {worker_compute_times} train_bsz '
                     f'{self.train_bsz} and grad_scaling {self.gradient_scaling} for worker rank {self.rank}')

        return dataloader

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

        bsz_list = torch.FloatTensor([math.ceil(bs_diff / self.world_size) for _ in range(self.world_size)])
        bsz_list = bsz_list.to(torch.device("cpu"))

        return bounded_bsz, bsz_list