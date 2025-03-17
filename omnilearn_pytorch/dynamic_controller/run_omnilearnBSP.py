import argparse

import omnilearn_pytorch.helper.miscellaneous as misc
from omnilearn_pytorch.dynamic_controller.bsp_omnilearn import BSPOmniLearn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help='seed value for result replication')
    parser.add_argument('--dir', type=str, default='/logs', help='dir where data (training, logs) is stored')
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--backend', type=str, default='gloo')
    parser.add_argument('--master-addr', type=str, default='127.0.0.1')
    parser.add_argument('--master-port', type=str, default='27208')
    parser.add_argument('--init-method', type=str, default='tcp')
    parser.add_argument('--test-bsz', type=int, default=32)
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--trainfreq', type=int, default=200, help='compute train acc after this iterations')
    parser.add_argument('--testfreq', type=int, default=400, help='compute test acc after this iterations')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--globalB', type=int, default=32)
    parser.add_argument('--deadband-threshold', type=float, default=0.2)
    parser.add_argument('--min-bsz', type=int, default=16, help='minimum batch-size to be allotted on a worker')
    parser.add_argument('--max-bsz', type=int, default=96, help='maximum batch-size to be allotted on a worker')
    parser.add_argument("--determinism", action="store_true", help="to use determinism or not")
    parser.add_argument("--asyncop", action="store_false", help="communication ops to use async mode if on gpu or not")
    parser.add_argument("--bsp", action="store_true", help="to use bsp or asp for training")
    args = parser.parse_args()

    print('going to deploy containers for bsp!!!')
    BSPOmniLearn(args=args).train()

    # set rank-0 worker IP as the master address
    container_name = 'worker1'
    container_ip, _ = misc.get_container_network_info(container_name=container_name)
    args.master_addr = container_ip

    misc.set_seed(seed=args.seed, determinism=args.determinism)
    BSPOmniLearn(args=args).train()