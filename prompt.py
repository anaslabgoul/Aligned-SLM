import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "model.py"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a trained checkpoint and generate text from a prompt."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to model module, e.g. models/model.py or models.model",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to checkpoint file (.pt).",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt text to feed into the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=120,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (use 0 for greedy decoding).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top-k sampling cutoff.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--print-full-output",
        action="store_true",
        help="Print full generated text including the original prompt.",
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

    module_name = f"prompt_model_{model_path.stem}"
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
            and name not in {"CharTokenizer", "CharEmbedding", "TransformerBlock"}
        ):
            candidates.append(obj)

    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        for candidate in candidates:
            if candidate.__name__ == "model":
                return candidate
        return candidates[0]

    raise ValueError("No nn.Module subclass found in the model file.")


def build_model(module):
    model_cls = get_model_class(module)
    init_kwargs = {}
    for key in ("vocab_size", "d_model", "n_heads", "n_layers", "max_seq_len", "dropout"):
        if hasattr(module, key):
            init_kwargs[key] = getattr(module, key)
    return model_cls(**init_kwargs)


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def strip_prompt_prefix(full_text: str, prompt: str) -> str:
    lower_full = full_text.lower()
    lower_prompt = prompt.lower()
    if lower_full.startswith(lower_prompt):
        return full_text[len(prompt) :]
    return full_text


def normalize_prompt(prompt: str) -> str:
    return prompt.strip()


def trim_at_eos(text: str) -> str:
    eos_index = text.find("<eos>")
    if eos_index != -1:
        return text[:eos_index]
    return text


def main():
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_module = load_model_module(args.model)
    model = build_model(model_module)
    device = select_device(args.device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"'model_state_dict' key not found in checkpoint: {checkpoint_path}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    normalized_prompt = normalize_prompt(args.prompt)
    if not normalized_prompt:
        raise ValueError("Prompt is empty.")

    with torch.no_grad():
        generated = model.generate(
            prompt=normalized_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    generated = trim_at_eos(generated)
    generated = generated.replace("<bos>", "")

    if args.print_full_output:
        output = generated
    else:
        output = strip_prompt_prefix(generated, normalized_prompt)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print("\n=== Prompt ===")
    print(normalized_prompt)
    print("\n=== Output ===")
    print(output)


if __name__ == "__main__":
    main()
