# File: launcher.py
# Time: 2024/ 06/ 17 16: 48: 12
# Desc : SVM训练
"""
To run mpc_linear_svm example in multiprocess mode:

$ python3 examples/mpc_linear_svm/launcher.py --multiprocess
"""

import argparse
import logging
import os

from examples.multiprocess_launcher import MultiProcessLauncher


parser = argparse.ArgumentParser(description="feature selection based on one-hot matrix")
parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)
parser.add_argument(
    "--multiprocess",
    default=False,
    action="store_true",
    help="Run example in multiprocess mode",
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
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
    )
    from test import run_mpc_linear_svm

    run_mpc_linear_svm(
        args.epoch, 
        args.lr
    )


def main(run_experiment):
    args = parser.parse_args()
    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, run_experiment, args)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main(_run_experiment)
