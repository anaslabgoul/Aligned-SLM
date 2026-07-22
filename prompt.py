from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def normalize_prompt(prompt: str) -> str:
    return prompt.strip()


def prompt_content(tokenizer, prompt: str) -> str:
    """Return the lowercased prompt text after stripping literal <bos>/<eos> markers."""
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    return tokenizer.decode(prompt_ids[1:].tolist())


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
) -> int:
    if temperature == 0:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / temperature
    if top_k is not None:
        values, _ = torch.topk(logits, top_k)
        logits = torch.where(logits < values[-1], torch.tensor(float("-inf")), logits)

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_until_eos(
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    use_cache: bool = True,
) -> list[int]:
    """Generate token ids until <eos>, the token budget, or the context limit.

    The KV cache makes this O(n) forward passes over one token each, instead of
    O(n) passes over a prefix that grows every step. Models without a cache-aware
    forward (use_cache unsupported) fall back to re-encoding the whole prefix.
    """
    tokenizer = model.tokenizer
    token_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False).tolist()

    supports_cache = use_cache and "use_cache" in inspect.signature(model.forward).parameters
    past_kvs = None
    step_ids = token_ids

    for _ in range(max_new_tokens):
        x = torch.tensor([step_ids], dtype=torch.long, device=model.device)
        if supports_cache:
            logits, past_kvs = model(x, past_kvs=past_kvs, use_cache=True)
        else:
            logits = model(x)
        next_token_id = sample_next_token(logits[:, -1, :].squeeze(0), temperature, top_k)
        token_ids.append(next_token_id)
        # With a cache only the new token is fed back; without one the model
        # needs the full prefix again.
        step_ids = [next_token_id] if supports_cache else token_ids

        if next_token_id == tokenizer.eos_token_id:
            break
        if len(token_ids) >= model.max_seq_len:
            break

    return token_ids


def format_token_ids(tokenizer, token_ids: list[int]) -> str:
    eos_id = tokenizer.eos_token_id
    if eos_id in token_ids:
        token_ids = token_ids[: token_ids.index(eos_id)]

    text = tokenizer.decode(token_ids)
    if text.startswith("<bos>"):
        text = text[len("<bos>") :]
    return text


def strip_prompt_prefix(full_text: str, prompt_content_text: str) -> str:
    if full_text.startswith(prompt_content_text):
        return full_text[len(prompt_content_text) :]
    return full_text


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

    prompt_text = prompt_content(model.tokenizer, normalized_prompt)

    with torch.no_grad():
        token_ids = generate_until_eos(
            model=model,
            prompt=normalized_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    generated = format_token_ids(model.tokenizer, token_ids)

    if args.print_full_output:
        output = generated
    else:
        output = strip_prompt_prefix(generated, prompt_text)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print("\n=== Prompt ===")
    print(normalized_prompt)
    print("\n=== Output ===")
    print(output)


if __name__ == "__main__":
    main()
