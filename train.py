import argparse
import importlib.util
import json
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
        description="Train a character-level language model on JSONL data."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the model module, e.g. models/model.py or models.model",
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
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
    for key in ("vocab_size", "d_model", "n_heads", "n_layers", "max_seq_len", "dropout"):
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
            tokens = tokenizer.encode(text, add_bos=False, add_eos=False)
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
    return torch.device(device_arg)


def run_epoch(model, data_loader, optimizer, device, train: bool) -> float:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in data_loader:
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
                optimizer.step()

            token_count = targets.numel()
            total_loss += loss.item() * token_count
            total_tokens += token_count

    return total_loss / max(total_tokens, 1)


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
        full_dataset,
        [train_size, test_size],
        generator=generator,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_test_loss = float("inf")
    best_checkpoint = args.output_dir / "best_model.pt"

    print(f"Device: {device}")
    print(f"Training samples: {train_size}")
    print(f"Test samples: {test_size}")
    print(f"Saving best checkpoint to: {best_checkpoint}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
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
