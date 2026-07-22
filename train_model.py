"""Continue training a character-level language model from a checkpoint.

Only the model weights are restored; the optimizer and learning-rate schedule
start fresh. The training machinery itself lives in training_common.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import training_common
import wandb_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continue training a character-level language model from checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a checkpoint .pt file containing model_state_dict.",
    )
    training_common.add_common_args(parser)
    return parser.parse_args()


def load_pretrained_checkpoint(model, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} does not contain 'model_state_dict'."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def main():
    args = parse_args()
    setup = training_common.setup(args)

    checkpoint_path = training_common.resolve_path(args.checkpoint)
    checkpoint = load_pretrained_checkpoint(setup.model, checkpoint_path, setup.device)

    config = setup.wandb_config()
    # Record where this run continued from, so the dashboard shows the lineage.
    config["resumed_from"] = str(checkpoint_path)
    config["resumed_epoch"] = checkpoint.get("epoch")
    run = wandb_logging.init_run(args, config=config, job_type="continue-train")

    print(f"Loaded pretrained checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    training_common.print_setup(
        args, setup, args.output_dir / "best_model.pt", run=run
    )
    training_common.train(args, setup, run=run)


if __name__ == "__main__":
    main()
