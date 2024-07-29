
"""
# rank 0
python launch_distributed.py \
    --world_size 2 \
    --rank 0 \
    --master_address 127.0.0.1 \
    --master_port 12345 \
    --distributed
"""

import argparse
import logging
import os

from distributed_launcher import DistributedLauncher



parser = argparse.ArgumentParser(description="SVM_train")
parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)
parser.add_argument(
    "--rank",
    type=int,
    help="The rank of the current party. Each party acts as its own process",
)
parser.add_argument(
    "--master_address",
    type=str,
    help="master IP Address",
)
parser.add_argument(
    "--master_port",
    type=int,
    help="master port",
)
parser.add_argument(
    "--backend",
    type=str,
    default="gloo",
    help="backend for torhc.distributed, 'NCCL' or 'gloo'.",
)
parser.add_argument(
    "--distributed",
    default=True,
    action="store_true",
    help="Run example in distributed mode",
)
parser.add_argument(
    "--epoch", 
    default=100, 
    type=int, 
    help="epoch"
)
parser.add_argument(
    "--lr", 
    default=0.5, 
    type=float, 
    help="learning rate"
)


def _run_experiment(args):
    # only import here to initialize crypten within the subprocesses
    from test import run_mpc_linear_svm
    # Only Rank 0 will display logs.
    level = logging.INFO
    #if "RANK" in os.environ and os.environ["RANK"] != "0":
    #    level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    run_mpc_linear_svm(
        args.epoch,
        args.lr
    )


def main(run_experiment):
    args = parser.parse_args()
    if args.distributed:
        launcher = DistributedLauncher(args.world_size, args.rank, args.master_address, args.master_port, args.backend, run_experiment, args)
        launcher.start()
        #launcher.join()
        #launcher.terminate()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main(_run_experiment)
