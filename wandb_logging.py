"""Optional Weights & Biases experiment tracking.

Tracking is opt-in: every helper here takes a `run` that is `None` when the user
did not pass `--wandb`, and does nothing in that case. That keeps train.py,
train_model.py and evaluate_model.py runnable without wandb installed or
configured, and keeps the existing commands in USAGE.md working unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path


DEFAULT_PROJECT = "aligned-slm"


def add_wandb_args(parser, default_project: str = DEFAULT_PROJECT, artifacts: bool = False):
    """Attach the shared --wandb* flags to an argparse parser.

    `artifacts=True` adds --wandb-artifacts, which only makes sense for the
    training scripts (they are the ones that produce checkpoints).
    """
    group = parser.add_argument_group("weights & biases")
    group.add_argument(
        "--wandb",
        action="store_true",
        help="Log this run to Weights & Biases (off by default).",
    )
    group.add_argument("--wandb-project", default=default_project, help="W&B project name.")
    group.add_argument(
        "--wandb-entity",
        default=None,
        help="W&B team or user (default: your personal entity).",
    )
    group.add_argument(
        "--wandb-run-name",
        default=None,
        help="Run name shown in the dashboard (default: wandb picks one).",
    )
    group.add_argument(
        "--wandb-tags",
        nargs="*",
        default=[],
        help="Tags to attach to the run, e.g. --wandb-tags level1 baseline",
    )
    group.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="'offline' logs to ./wandb without network access; sync later with `wandb sync`.",
    )
    if artifacts:
        group.add_argument(
            "--wandb-artifacts",
            action="store_true",
            help="Upload the best checkpoint to W&B as a versioned model artifact.",
        )
    return parser


def _jsonable(value):
    """argparse gives us Path objects; wandb's config wants plain values."""
    if isinstance(value, Path):
        return str(value)
    return value


def init_run(args, config: dict | None = None, job_type: str = "train", run_id=None, resume=None):
    """Start a run, or return None when --wandb was not passed.

    Every non-wandb CLI argument lands in the run config automatically, so runs
    stay filterable by lr, batch size, seed, dataset and so on without having to
    list them here.
    """
    if not getattr(args, "wandb", False):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "--wandb requires the wandb package.\n"
            "Install it with:  pip install wandb\n"
            "Then authenticate once with:  wandb login"
        ) from exc

    merged = {
        key: _jsonable(value)
        for key, value in vars(args).items()
        if not key.startswith("wandb")
    }
    merged.update(config or {})

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=args.wandb_tags or None,
        mode=args.wandb_mode,
        job_type=job_type,
        id=run_id,
        resume=resume,
        config=merged,
    )


def run_config(model_module, model, device, data_path, train_size, test_size, total_steps):
    """Architecture and dataset facts to record next to the CLI arguments.

    Hyperparameters are read off the model module rather than hardcoded, so the
    config follows whatever module was passed to --model.
    """
    config = {
        "vocab_size": model.vocab_size,
        "max_seq_len": model.max_seq_len,
        "n_params": sum(param.numel() for param in model.parameters()),
        "device": str(device),
        "dataset": str(data_path),
        "train_samples": train_size,
        "test_samples": test_size,
        "total_steps": total_steps,
    }
    for key in ("d_model", "n_heads", "n_layers", "dropout"):
        if hasattr(model_module, key):
            config[key] = getattr(model_module, key)
    return config


def perplexity(loss: float) -> float:
    """exp(loss), clamped so an early garbage loss cannot overflow the chart."""
    return math.exp(min(loss, 20.0))


def run_location(run) -> str:
    """Where to find this run: a dashboard URL, or a local path when offline."""
    if run is None:
        return ""
    return run.url or f"offline at {Path(run.dir).parent} (upload later with `wandb sync`)"


def log(run, metrics: dict, step: int | None = None):
    if run is not None:
        run.log(metrics, step=step)


def summary(run, values: dict):
    """Write run-level summary values (what the dashboard's run table sorts on)."""
    if run is not None:
        run.summary.update(values)


def log_table(run, key: str, columns, rows):
    if run is None:
        return
    import wandb

    run.log({key: wandb.Table(columns=list(columns), data=[list(row) for row in rows])})


def log_checkpoint(run, path: Path, metadata: dict | None = None):
    """Upload a checkpoint as a versioned model artifact.

    Only called at the end of training: the checkpoint is a few hundred MB for
    the default architecture, so uploading on every improvement is wasteful.
    """
    if run is None:
        return
    import wandb

    artifact = wandb.Artifact(f"model-{run.id}", type="model", metadata=metadata or {})
    artifact.add_file(str(path))
    run.log_artifact(artifact)


def finish(run):
    if run is not None:
        run.finish()
