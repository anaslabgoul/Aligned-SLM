"""Train a character-level language model from scratch on JSONL data.

To continue training from an existing checkpoint instead, use train_model.py.
The training machinery itself lives in training_common.py.
"""

from __future__ import annotations

import argparse

import training_common
import wandb_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a character-level language model on JSONL data."
    )
    training_common.add_common_args(parser)
    args = parser.parse_args()
    training_common.check_data_args(parser, args)
    return args


def main():
    args = parse_args()
    setup = training_common.setup(args)

    run = wandb_logging.init_run(args, config=setup.wandb_config(), job_type="train")
    training_common.print_setup(
        args, setup, args.output_dir / "best_model.pt", run=run
    )
    training_common.train(args, setup, run=run)


if __name__ == "__main__":
    main()
