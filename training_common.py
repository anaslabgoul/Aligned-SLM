"""Shared training machinery for train.py and train_model.py.

Both entry points do the same thing — build a model from a module, tokenize a
JSONL dataset, and run an AdamW + warmup/cosine training loop — and differ only
in whether they start from random weights or from a checkpoint. Everything they
share lives here so the two scripts cannot drift apart.
"""

from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

import dataset_mix
import wandb_logging


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def add_common_args(parser):
    """Add every argument shared by train.py and train_model.py."""
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the model module, e.g. models/model.py or models.model",
    )
    parser.add_argument(
        "--data",
        required=True,
        action="append",
        metavar="PATH",
        help=(
            "Path to a JSON or JSONL dataset file. Repeat it to train on a mix "
            "of datasets, and pair it with --weight to set each one's share."
        ),
    )
    parser.add_argument(
        "--weight",
        type=float,
        action="append",
        metavar="W",
        help=(
            "Share of the mix taken from the matching --data file, in the same "
            "order. Weights are normalized, so '--weight 3 --weight 1' means "
            "75%%/25%%. Omit for an equal split."
        ),
    )
    parser.add_argument(
        "--mix-total",
        type=int,
        default=None,
        help=(
            "Total number of samples to draw across all --data files. Without "
            "it, the mix is the largest one that respects the weights without "
            "reusing any sample."
        ),
    )
    parser.add_argument(
        "--mix-replace",
        action="store_true",
        help=(
            "Allow sampling with replacement, so a source smaller than its "
            "share is oversampled instead of raising an error."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of training epochs.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate.")
    parser.add_argument(
        "--min-lr",
        type=float,
        default=3e-5,
        help="Final learning rate the cosine schedule decays to.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Linear warmup steps before cosine decay.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="AdamW weight decay (applied to matmul weights only).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Max gradient norm (0 disables clipping).",
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2.")
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evals-per-epoch",
        type=int,
        default=0,
        help=(
            "Evaluate this many times per epoch instead of only at epoch end. "
            "Use it for short runs over large datasets, where one point per "
            "epoch is too coarse to see anything or to catch the best model."
        ),
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=0,
        help=(
            "Cap mid-epoch evaluations to this many test batches (0 = full test "
            "set). Keeps --evals-per-epoch cheap; the epoch-end evaluation always "
            "uses the full test set."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the best checkpoint is saved.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, or cuda.",
    )
    wandb_logging.add_wandb_args(parser, artifacts=True)
    return parser


# --------------------------------------------------------------------------
# Model and data loading
# --------------------------------------------------------------------------


def resolve_path(path_str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_model_module(model_arg: str):
    model_path = resolve_path(model_arg)
    if model_path.suffix != ".py":
        model_path = model_path.with_suffix(".py")

    if not model_path.exists():
        dotted = model_arg.replace("\\", ".").replace("/", ".").removesuffix(".py")
        try:
            return importlib.import_module(dotted)
        except ImportError as exc:
            raise FileNotFoundError(
                f"Could not find model module at {model_path} or import {dotted}"
            ) from exc

    module_name = f"custom_model_{model_path.stem}"
    module_dir = str(model_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load model module from {model_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_model_class(module):
    if hasattr(module, "model") and issubclass(module.model, nn.Module):
        return module.model

    candidates = []
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, nn.Module)
            and obj is not nn.Module
            and name != "CharTokenizer"
            and name != "CharEmbedding"
            and name != "TransformerBlock"
        ):
            candidates.append(obj)

    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        for candidate in candidates:
            if candidate.__name__ == "model":
                return candidate
        return candidates[0]

    raise ValueError(
        "No nn.Module subclass found in the model file. "
        "Expected a class named 'model' or another torch module."
    )


def build_model(module):
    model_cls = get_model_class(module)
    init_kwargs = {}
    for key in ("vocab_size", "d_model", "n_heads", "n_layers", "max_seq_len", "dropout"):
        if hasattr(module, key):
            init_kwargs[key] = getattr(module, key)
    return model_cls(**init_kwargs)


def check_data_args(parser, args):
    """Report a mismatched --data / --weight pairing as a CLI error, not a traceback."""
    try:
        dataset_mix.resolve_source_weights(args.data, args.weight)
    except ValueError as error:
        parser.error(str(error))


def resolve_data_sources(args) -> list[tuple[Path, float]]:
    """Turn the --data / --weight arguments into (path, normalized weight) pairs."""
    pairs = dataset_mix.resolve_source_weights(args.data, args.weight)
    return [(resolve_path(path_str), weight) for path_str, weight in pairs]


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer):
        self.samples = []
        for text in texts:
            tokens = tokenizer.encode(text)
            if tokens.numel() > 1:
                self.samples.append(tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_batch(batch: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(sample.size(0) for sample in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for index, sample in enumerate(batch):
        padded[index, : sample.size(0)] = sample
    return padded


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_arg == "cuda" and not torch.cuda.is_available():
        hint = (
            "Install a CUDA-enabled PyTorch build, e.g.\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu130"
        )
        if torch.version.cuda is None:
            raise RuntimeError(
                "CUDA was requested but this PyTorch install is CPU-only.\n" + hint
            )
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False.\n" + hint
        )

    return torch.device(device_arg)


# --------------------------------------------------------------------------
# Optimizer and schedule
# --------------------------------------------------------------------------


def configure_optimizer(model, lr: float, weight_decay: float, betas):
    """AdamW with weight decay applied only to matmul weights.

    LayerNorm weights, biases, and embeddings (all tensors with < 2 dims, plus
    the embedding tables) are excluded from decay, following common LLM practice.
    """
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "embedding" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas)


def build_scheduler(optimizer, warmup_steps: int, total_steps: int, peak_lr: float, min_lr: float):
    """Linear warmup followed by cosine decay from peak_lr down to min_lr."""
    min_ratio = min_lr / peak_lr if peak_lr > 0 else 0.0
    total_steps = max(total_steps, 1)
    warmup_steps = max(min(warmup_steps, total_steps - 1), 0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------


class TrainingSetup:
    """Everything a training loop needs, built once from the parsed arguments."""

    def __init__(self, args):
        self.model_module = load_model_module(args.model)
        self.model = build_model(self.model_module)
        self.device = select_device(args.device)
        self.model.to(self.device)

        self.data_sources = resolve_data_sources(args)
        self.data_paths = [path for path, _ in self.data_sources]
        texts, self.mix_report = dataset_mix.load_mixed_texts(
            self.data_sources,
            mix_total=args.mix_total,
            seed=args.seed,
            replace=args.mix_replace,
        )
        full_dataset = TextDataset(texts, self.model.tokenizer)

        if len(full_dataset) < 2:
            raise ValueError("Need at least 2 valid samples to create train and test splits.")

        self.test_size = max(1, int(len(full_dataset) * args.test_split))
        self.train_size = len(full_dataset) - self.test_size
        if self.train_size < 1:
            raise ValueError("test-split is too large for this dataset.")

        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, test_dataset = random_split(
            full_dataset, [self.train_size, self.test_size], generator=generator
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )

        self.optimizer = configure_optimizer(
            self.model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
        self.total_steps = args.epochs * max(len(self.train_loader), 1)
        self.scheduler = build_scheduler(
            self.optimizer,
            warmup_steps=args.warmup_steps,
            total_steps=self.total_steps,
            peak_lr=args.lr,
            min_lr=args.min_lr,
        )

    def wandb_config(self) -> dict:
        config = wandb_logging.run_config(
            self.model_module,
            self.model,
            self.device,
            ", ".join(str(path) for path in self.data_paths),
            self.train_size,
            self.test_size,
            self.total_steps,
        )
        if len(self.mix_report) > 1:
            # Record the realized mix, not just the requested weights, so runs
            # stay comparable when a source was too small to fill its share.
            config["dataset_mix"] = {
                item["path"]: item["sampled"] for item in self.mix_report
            }
        return config


def setup(args) -> TrainingSetup:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    return TrainingSetup(args)


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, data_loader, device, max_batches: int = 0) -> float:
    """Mean token-weighted loss over the test set.

    Leaves the model in eval mode; the caller restores training mode. Pass
    `max_batches` to score only a prefix of the test set — enough signal for a
    mid-epoch curve at a fraction of the cost.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for batch_index, batch in enumerate(data_loader, start=1):
        batch = batch.to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        token_count = targets.numel()
        total_loss += loss.item() * token_count
        total_tokens += token_count

        if max_batches and batch_index >= max_batches:
            break

    return total_loss / max(total_tokens, 1)


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    epoch: int,
    total_epochs: int,
    scheduler=None,
    grad_clip: float = 0.0,
    progress_updates: int = 10,
    run=None,
    global_step: int = 0,
    eval_every: int = 0,
    eval_hook=None,
) -> tuple[float, int]:
    """One pass over the training data, returning (mean loss, new global step).

    `eval_hook(global_step, epoch, epoch_progress, running_train_loss)` is called
    every `eval_every` batches, which is how mid-epoch test loss and
    best-checkpoint saving happen.
    """
    model.train()

    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()
    num_batches = len(data_loader)

    milestone_batches = set()
    if progress_updates > 0 and num_batches > 0:
        # Evenly spaced batch milestones; print once per milestone.
        milestone_batches = {
            max(1, (num_batches * idx) // progress_updates)
            for idx in range(1, progress_updates + 1)
        }

    for batch_index, batch in enumerate(data_loader, start=1):
        batch = batch.to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        grad_norm = None
        if grad_clip and grad_clip > 0:
            # clip_grad_norm_ returns the pre-clip norm: a free signal for
            # spotting exploding or vanishing gradients.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        global_step += 1
        current_lr = (
            scheduler.get_last_lr()[0]
            if scheduler is not None
            else optimizer.param_groups[0]["lr"]
        )
        epoch_progress = epoch - 1 + batch_index / max(num_batches, 1)

        token_count = targets.numel()
        total_loss += loss.item() * token_count
        total_tokens += token_count

        if run is not None:
            metrics = {
                "train/loss_step": loss.item(),
                "train/lr": current_lr,
                "train/epoch_progress": epoch_progress,
            }
            if grad_norm is not None:
                metrics["train/grad_norm"] = grad_norm.item()
            wandb_logging.log(run, metrics, step=global_step)

        if batch_index in milestone_batches:
            running_loss = total_loss / max(total_tokens, 1)
            print(
                f"Epoch {epoch}/{total_epochs} | "
                f"batch {batch_index}/{num_batches} | "
                f"train loss: {running_loss:.4f} | lr: {current_lr:.2e}"
            )

        # Mid-epoch evaluation. Skipped on the final batch, where the epoch-end
        # evaluation over the full test set is about to run anyway.
        if (
            eval_hook is not None
            and eval_every > 0
            and batch_index % eval_every == 0
            and batch_index < num_batches
        ):
            eval_hook(
                global_step, epoch, epoch_progress, total_loss / max(total_tokens, 1)
            )
            model.train()

    return total_loss / max(total_tokens, 1), global_step


def save_checkpoint(path: Path, model, epoch: int, train_loss: float, test_loss: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "model_state_dict": model.state_dict(),
            "vocab_size": model.vocab_size,
            "max_seq_len": model.max_seq_len,
        },
        path,
    )


def print_setup(args, setup: TrainingSetup, best_checkpoint: Path, run=None):
    print(f"Device: {setup.device}")
    if len(setup.mix_report) > 1 or args.mix_total is not None:
        print(dataset_mix.format_mix_report(setup.mix_report))
    print(f"Training samples: {setup.train_size}")
    print(f"Test samples: {setup.test_size}")
    print(
        f"Optimizer: AdamW(betas=({args.beta1}, {args.beta2}), wd={args.weight_decay}) | "
        f"peak lr: {args.lr:.2e} -> min lr: {args.min_lr:.2e} | "
        f"warmup: {args.warmup_steps} steps | total: {setup.total_steps} steps | "
        f"grad clip: {args.grad_clip}"
    )
    if args.evals_per_epoch > 0:
        scope = (
            f"first {args.eval_batches} test batches"
            if args.eval_batches
            else "full test set"
        )
        print(f"Evaluating {args.evals_per_epoch}x per epoch ({scope})")
    print(f"Saving best checkpoint to: {best_checkpoint}")
    if run is not None:
        print(f"Tracking run on W&B: {wandb_logging.run_location(run)}")


def train(args, setup: TrainingSetup, run=None) -> float:
    """Run the training loop, saving the best checkpoint. Returns the best loss."""
    model = setup.model
    best_checkpoint = args.output_dir / "best_model.pt"
    num_batches = max(len(setup.train_loader), 1)

    # Spread evaluations evenly across the epoch's batches.
    eval_every = (
        max(1, num_batches // args.evals_per_epoch) if args.evals_per_epoch > 0 else 0
    )

    best = {"loss": float("inf"), "epoch": 0, "step": 0}
    state = {"global_step": 0}

    def record_evaluation(test_loss, epoch, global_step, epoch_progress, partial, train_loss):
        """Log a test-loss point and save the checkpoint if it is the best yet."""
        is_best = test_loss < best["loss"]
        if is_best:
            best["loss"] = test_loss
            best["epoch"] = epoch
            best["step"] = global_step
            save_checkpoint(best_checkpoint, model, epoch, train_loss, test_loss)

        wandb_logging.log(
            run,
            {
                "epoch": epoch_progress,
                "test/loss": test_loss,
                "test/perplexity": wandb_logging.perplexity(test_loss),
                "test/best_loss": best["loss"],
                # 1 when the point came from a capped mid-epoch pass, so a noisy
                # point is never mistaken for a full-test-set measurement.
                "test/partial": int(partial),
            },
            step=global_step,
        )
        return is_best

    def eval_hook(global_step, epoch, epoch_progress, running_train_loss):
        test_loss = evaluate(
            model, setup.test_loader, setup.device, max_batches=args.eval_batches
        )
        is_best = record_evaluation(
            test_loss,
            epoch,
            global_step,
            epoch_progress,
            partial=bool(args.eval_batches),
            # Mean over the epoch so far — the closest thing to a training loss
            # for a checkpoint saved before the epoch finished.
            train_loss=running_train_loss,
        )
        marker = "  -> New best model saved" if is_best else ""
        print(
            f"Epoch {epoch}/{args.epochs} | step {global_step} | "
            f"test loss: {test_loss:.4f}{marker}"
        )

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, global_step = train_one_epoch(
                model,
                setup.train_loader,
                setup.optimizer,
                setup.device,
                epoch=epoch,
                total_epochs=args.epochs,
                scheduler=setup.scheduler,
                grad_clip=args.grad_clip,
                run=run,
                global_step=state["global_step"],
                eval_every=eval_every,
                eval_hook=eval_hook,
            )
            state["global_step"] = global_step

            # Epoch end always scores the full test set, never a capped prefix.
            test_loss = evaluate(model, setup.test_loader, setup.device)
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train loss: {train_loss:.4f} | test loss: {test_loss:.4f}"
            )

            wandb_logging.log(
                run, {"train/loss_epoch": train_loss}, step=global_step
            )
            if record_evaluation(
                test_loss,
                epoch,
                global_step,
                float(epoch),
                partial=False,
                train_loss=train_loss,
            ):
                print(f"  -> New best model saved (test loss: {test_loss:.4f})")

        print(f"Training complete. Best test loss: {best['loss']:.4f}")
        print(f"Best checkpoint: {best_checkpoint}")

        wandb_logging.summary(
            run,
            {
                "best_test_loss": best["loss"],
                "best_epoch": best["epoch"],
                "best_step": best["step"],
            },
        )
        if getattr(args, "wandb_artifacts", False) and best_checkpoint.exists():
            print("Uploading best checkpoint to W&B...")
            wandb_logging.log_checkpoint(
                run,
                best_checkpoint,
                metadata={"test_loss": best["loss"], "epoch": best["epoch"]},
            )
    finally:
        # Without this, an interrupted run stays stuck as "running" in the UI.
        wandb_logging.finish(run)

    return best["loss"]
