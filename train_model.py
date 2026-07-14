from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continue training a character-level language model from checkpoint."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the model module, e.g. models/model.py or models.model",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a checkpoint .pt file containing model_state_dict.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to a JSON or JSONL dataset file.",
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
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
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
    for key in (
        "vocab_size",
        "d_model",
        "n_heads",
        "n_layers",
        "max_seq_len",
        "dropout",
    ):
        if hasattr(module, key):
            init_kwargs[key] = getattr(module, key)
    return model_cls(**init_kwargs)


def load_dataset_records(data_path: Path) -> list[str]:
    suffix = data_path.suffix.lower()
    records = []

    if suffix == ".jsonl":
        with data_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = extract_text(record, line_number)
                records.append(text)
    elif suffix == ".json":
        with data_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            for index, record in enumerate(payload, start=1):
                records.append(extract_text(record, index))
        elif isinstance(payload, dict) and "text" in payload:
            records.append(extract_text(payload, 1))
        else:
            raise ValueError(
                "JSON file must be a list of records or a single object with a 'text' field."
            )
    else:
        raise ValueError("Data file must be .json or .jsonl")

    if not records:
        raise ValueError(f"No training samples found in {data_path}")

    return records


def extract_text(record, line_number: int) -> str:
    if isinstance(record, str):
        return record
    if isinstance(record, dict) and "text" in record:
        return record["text"]
    raise ValueError(
        f"Record at line {line_number} must contain a 'text' field or be a string."
    )


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


def configure_optimizer(model, lr: float, weight_decay: float, betas):
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


def build_scheduler(
    optimizer, warmup_steps: int, total_steps: int, peak_lr: float, min_lr: float
):
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


def run_epoch(
    model,
    data_loader,
    optimizer,
    device,
    train: bool,
    epoch: int | None = None,
    total_epochs: int | None = None,
    progress_updates: int = 0,
    scheduler=None,
    grad_clip: float = 0.0,
) -> float:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()
    num_batches = len(data_loader)
    milestone_batches = set()
    if train and progress_updates > 0 and num_batches > 0:
        milestone_batches = {
            max(1, (num_batches * idx) // progress_updates)
            for idx in range(1, progress_updates + 1)
        }

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(data_loader, start=1):
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            logits = model(inputs)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            token_count = targets.numel()
            total_loss += loss.item() * token_count
            total_tokens += token_count

            if train and batch_index in milestone_batches:
                running_loss = total_loss / max(total_tokens, 1)
                current_lr = (
                    scheduler.get_last_lr()[0]
                    if scheduler is not None
                    else optimizer.param_groups[0]["lr"]
                )
                if epoch is not None and total_epochs is not None:
                    print(
                        f"Epoch {epoch}/{total_epochs} | "
                        f"batch {batch_index}/{num_batches} | "
                        f"train loss: {running_loss:.4f} | lr: {current_lr:.2e}"
                    )
                else:
                    print(
                        f"Batch {batch_index}/{num_batches} | "
                        f"train loss: {running_loss:.4f} | lr: {current_lr:.2e}"
                    )

    return total_loss / max(total_tokens, 1)


def load_pretrained_checkpoint(model, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} does not contain 'model_state_dict'."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


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


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_module = load_model_module(args.model)
    model = build_model(model_module)
    device = select_device(args.device)
    model.to(device)

    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = load_pretrained_checkpoint(model, checkpoint_path, device)

    data_path = resolve_path(args.data)
    texts = load_dataset_records(data_path)
    full_dataset = TextDataset(texts, model.tokenizer)

    if len(full_dataset) < 2:
        raise ValueError("Need at least 2 valid samples to create train and test splits.")

    test_size = max(1, int(len(full_dataset) * args.test_split))
    train_size = len(full_dataset) - test_size
    if train_size < 1:
        raise ValueError("test-split is too large for this dataset.")

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    optimizer = configure_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2)
    )
    total_steps = args.epochs * max(len(train_loader), 1)
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        peak_lr=args.lr,
        min_lr=args.min_lr,
    )
    best_test_loss = float("inf")
    best_checkpoint = args.output_dir / "best_model.pt"

    print(f"Loaded pretrained checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Device: {device}")
    print(f"Training samples: {train_size}")
    print(f"Test samples: {test_size}")
    print(
        f"Optimizer: AdamW(betas=({args.beta1}, {args.beta2}), wd={args.weight_decay}) | "
        f"peak lr: {args.lr:.2e} -> min lr: {args.min_lr:.2e} | "
        f"warmup: {args.warmup_steps} steps | total: {total_steps} steps | "
        f"grad clip: {args.grad_clip}"
    )
    print(f"Saving best checkpoint to: {best_checkpoint}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            epoch=epoch,
            total_epochs=args.epochs,
            progress_updates=10,
            scheduler=scheduler,
            grad_clip=args.grad_clip,
        )
        test_loss = run_epoch(model, test_loader, optimizer, device, train=False)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss: {train_loss:.4f} | test loss: {test_loss:.4f}"
        )

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_checkpoint(best_checkpoint, model, epoch, train_loss, test_loss)
            print(f"  -> New best model saved (test loss: {test_loss:.4f})")

    print(f"Training complete. Best test loss: {best_test_loss:.4f}")
    print(f"Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
