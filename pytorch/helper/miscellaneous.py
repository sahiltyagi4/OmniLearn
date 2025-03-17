import os
import random
from pathlib import Path
import subprocess

import numpy as np

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp

def set_seed(seed, determinism=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(determinism)

def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class AverageMeter(object):
    """Computes and stores the average and current value for model loss, accuracy etc."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CollectiveOps(object):

    def __init__(self, world_size, async_op):
        self.world_size = world_size
        self.async_op = async_op

    def broadcast(self, model, rank=0):
        for _, param in model.named_parameters():
            if not param.requires_grad: continue
            dist.broadcast(tensor=param.data, src=rank, async_op=self.async_op)

        return model

    def simpleAggregation(self, model):
        for param in model.parameters():
            dist.all_reduce(tensor=param.grad, op=ReduceOp.SUM, async_op=self.async_op)
            param.grad /= self.world_size

    def weightedAggregation(self, model):
        for param in model.parameters():
            dist.all_reduce(tensor=param.grad, op=ReduceOp.SUM, async_op=self.async_op)
            param.grad = param.grad


def get_container_network_info(container_name, script_name='get_container_network.sh'):
    pth = Path(os.getcwd())
    pth = Path(pth.parent.absolute())
    script_pth = os.path.join(os.path.join(pth.parent.absolute(), 'scripts'), script_name)
    try:
        result = subprocess.run([script_pth, container_name], shell=True, check=True, text=True, capture_output=True)
        output_lines = result.stdout.strip().split("\n")
        container_ip, container_iface = None, None
        for line in output_lines:
            if "Container IP Address:" in line:
                container_ip = line.split(":")[1].strip()
            elif "Container Network Interface:" in line:
                container_iface = line.split(":")[1].strip()

        return  container_ip, container_iface

    except subprocess.CalledProcessError as e:
        print("Error executing script:", e)
        print("Script output (stderr):", e.stderr)