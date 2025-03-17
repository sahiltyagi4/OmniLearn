import argparse
import os

import torch.distributed.rpc as rpc
import torch

import omnilearn_pytorch.helper.miscellaneous as misc
from omnilearn_pytorch.dynamic_controller.asp_omnilearn import ASPOmniLearnPS, ASPOmniLearnWorker

def launch_worker(ps_rref, worker_rank, args):
    ASPOmniLearnWorker(ps_rref=ps_rref, rank=worker_rank, args=args).train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help='seed value for result replication')
    parser.add_argument('--dir', type=str, default='/logs', help='dir where data (training, logs) is stored')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--master-addr', type=str, default='127.0.0.1')
    parser.add_argument('--master-port', type=str, default='28564')
    parser.add_argument('--bszlist', type=str, default="32,32,32,32")
    parser.add_argument('--test-bsz', type=int, default=32)
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--trainfreq', type=int, default=200, help='compute train acc after this iterations')
    parser.add_argument('--testfreq', type=int, default=400, help='compute test acc after this iterations')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--deadband-threshold', type=float, default=0.2)
    parser.add_argument('--min-bsz', type=int, default=16, help='minimum batch-size to be allotted on a worker')
    parser.add_argument('--max-bsz', type=int, default=96, help='maximum batch-size to be allotted on a worker')
    parser.add_argument("--determinism", action="store_true", help="to use determinism or not")
    parser.add_argument("--asyncop", action="store_true", help="communication ops to use async mode if on gpu or not")
    parser.add_argument("--bsp", action="store_true", help="to use bsp or asp for training. use --no-bsp as argument in ASP training")
    args = parser.parse_args()

    # get IP address of parameter server container 'aspworker1'
    container_name = 'aspworker1'
    container_ip, container_iface = misc.get_container_network_info(container_name=container_name)

    os.environ['MASTER_ADDR'] = container_ip
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['TP_SOCKET_IFNAME'] = container_iface

    misc.set_seed(seed=args.seed, determinism=args.determinism)

    opts = rpc.TensorPipeRpcBackendOptions(num_worker_threads=4, rpc_timeout=0)
    print('FINISHED THIS...')

    # here rank 0 corresponds to the container where the parameter server is deployed.
    # all other containers are workers training the models
    if args.rank == 0:
        rpc.init_rpc('ps', rank=args.rank, world_size=args.world_size, rpc_backend_options=opts)
        print('STARTING NOW...')
        ps_rref = rpc.RRef(ASPOmniLearnPS(args=args))
        futures_list = []
        for i in range(1, args.world_size):
            worker = 'aspworker-' + str(i)
            futures_list.append(rpc.rpc_async(worker, launch_worker, args=(ps_rref, i, args)))

        torch.futures.wait_all(futures_list)

    elif args.rank > 0:
        print('IN HERE NOW....')
        rpc.init_rpc(f'aspworker-{args.rank}', rank=args.rank, world_size=args.world_size,
                     rpc_backend_options=opts)

    rpc.shutdown()