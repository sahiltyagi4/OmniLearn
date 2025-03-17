import argparse

import omnilearn_pytorch.helper.miscellaneous as misc
from omnilearn_pytorch.llm_training.gpt2_omnilearn import GPT2OmniLearnBSP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--master-addr', type=str, default='127.0.0.1')
    parser.add_argument('--master-port', type=str, default='28564')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--backend', type=str, default='gloo')
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--seq-length', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--dir', type=str, default='/')
    parser.add_argument('--deadband-threshold', type=float, default=0.1)
    parser.add_argument('--min_bsz', type=int, default=4)
    parser.add_argument('--max_bsz', type=int, default=64)
    parser.add_argument("--determinism", action="store_true", help="to use determinism or not")
    parser.add_argument("--asyncop", action="store_true", help="communication ops to use async mode if on gpu or not")
    parser.add_argument("--bsp", action="store_true", help="to use bsp or asp for training")
    args = parser.parse_args()

    misc.set_seed(seed=args.seed, determinism=args.determinism)
    GPT2OmniLearnBSP(args=args).train()