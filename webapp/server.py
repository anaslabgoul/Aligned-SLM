"""
Local web server for the Aligned-SLM math reasoning playground.

This is a *thin UI layer* on top of the project's existing inference code. It
does not change the model, the tokenizer, the checkpoints, or the decoding — it
imports and reuses the exact functions from ``prompt.py`` (``load_model_module``,
``build_model``, ``select_device``, ``generate_until_eos`` and the shared
``sample_next_token`` sampler) so what you see in the browser is byte-for-byte
what ``python prompt.py`` would produce.

Only Python's standard library is used (plus the ``torch`` that the project
already needs), so there is nothing extra to install.

Run it with::

    python webapp/server.py                 # then open http://127.0.0.1:8000
    python webapp/server.py --port 9000

and stop it with Ctrl+C. Pick CPU or GPU per-request from the UI's device menu.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# Make the project importable no matter where the server is launched from.
# prompt.py lives one directory up from this file; importing it also pulls the
# models/ package onto sys.path via its own loader helpers.
# ---------------------------------------------------------------------------
WEBAPP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEBAPP_DIR.parent
STATIC_DIR = WEBAPP_DIR / "static"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402  (imported after sys.path fix)

from prompt import (  # noqa: E402
    build_model,
    format_token_ids,
    generate_until_eos,
    load_model_module,
    normalize_prompt,
    resolve_path,
    sample_next_token,
    select_device,
)

DEFAULT_MODEL_ARG = "models/model.py"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


# ---------------------------------------------------------------------------
# Model registry: load checkpoints once, keep them warm, reuse across requests.
# A single lock serialises generation so concurrent browser tabs can't run two
# forward passes through the same module at the same time.
# ---------------------------------------------------------------------------
class ModelRegistry:
    def __init__(self):
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(model_arg: str, checkpoint_arg: str) -> str:
        model_path = resolve_path(model_arg)
        ckpt_path = resolve_path(checkpoint_arg)
        return f"{model_path}::{ckpt_path}"

    def load(self, model_arg: str, checkpoint_arg: str, device_arg: str) -> dict:
        """Return a cached entry {model, device, info}, loading it if needed."""
        key = self._key(model_arg, checkpoint_arg)
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None and entry["device_arg"] == device_arg:
                return entry

            started = time.time()
            checkpoint_path = resolve_path(checkpoint_arg)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            module = load_model_module(model_arg)
            model = build_model(module)
            device = select_device(device_arg)

            checkpoint = torch.load(checkpoint_path, map_location=device)
            if "model_state_dict" not in checkpoint:
                raise KeyError(
                    f"'model_state_dict' key not found in checkpoint: {checkpoint_path}"
                )
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError as exc:
                raise RuntimeError(
                    "This checkpoint does not match the architecture in "
                    f"'{model_arg}'. It was trained with different hyper-parameters "
                    "(d_model / n_heads / n_layers). Pick a matching model file or "
                    "the checkpoint's original architecture.\n\n"
                    f"Original error: {exc}"
                ) from exc

            model.to(device)
            model.eval()

            info = self._describe(module, model, device, checkpoint, checkpoint_path)
            info["load_seconds"] = round(time.time() - started, 3)

            entry = {
                "model": model,
                "device": device,
                "device_arg": device_arg,
                "info": info,
                "lock": threading.Lock(),
            }
            self._cache[key] = entry
            return entry

    @staticmethod
    def _describe(module, model, device, checkpoint, checkpoint_path) -> dict:
        def attr(name, default=None):
            return getattr(module, name, default)

        n_params = sum(p.numel() for p in model.parameters())
        meta = {
            k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in checkpoint.items()
            if k in ("epoch", "train_loss", "test_loss", "vocab_size", "max_seq_len")
        }
        return {
            "checkpoint": str(checkpoint_path),
            "checkpoint_name": checkpoint_path.name,
            "device": str(device),
            "parameters": n_params,
            "parameters_millions": round(n_params / 1e6, 2),
            "d_model": attr("d_model"),
            "n_heads": attr("n_heads"),
            "n_layers": attr("n_layers"),
            "max_seq_len": model.max_seq_len,
            "vocab_size": model.tokenizer.vocab_size,
            "checkpoint_meta": meta,
        }


REGISTRY = ModelRegistry()


# ---------------------------------------------------------------------------
# Prompt construction and output parsing (mirrors the training surface form and
# evaluate_model.py's answer extraction).
# ---------------------------------------------------------------------------
def build_prompt(problem: str) -> str:
    """Prime the model exactly like training: '<bos>Problem: ...; step:'."""
    return f"<bos>Problem: {problem}; step:"


def parse_generated(full_text: str) -> dict:
    """Split '<problem>; step: ...; step: <final>' into structured pieces.

    Answers and steps never contain ';' in the project's data format, so ';' is
    a safe separator (the same assumption evaluate_model.py relies on).
    """
    problem = None
    steps: list[str] = []
    for segment in full_text.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        lowered = segment.lower()
        if lowered.startswith("problem:"):
            problem = segment.split(":", 1)[1].strip()
        elif lowered.startswith("step:"):
            steps.append(segment.split(":", 1)[1].strip())
        else:
            steps.append(segment)
    return {
        "problem": problem,
        "steps": steps,
        "final": steps[-1] if steps else "",
    }


def _apply_seed(seed):
    if seed is not None:
        torch.manual_seed(int(seed))


def solve(entry: dict, problem: str, params: dict) -> dict:
    """Non-streaming solve. Returns structured steps + timing stats."""
    model = entry["model"]
    prompt = build_prompt(normalize_prompt(problem))

    with entry["lock"]:
        _apply_seed(params.get("seed"))
        started = time.time()
        with torch.no_grad():
            token_ids = generate_until_eos(
                model=model,
                prompt=prompt,
                max_new_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                top_k=params["top_k"],
            )
        elapsed = time.time() - started

    prompt_len = len(model.tokenizer.encode(prompt, add_bos=True, add_eos=False))
    generated_tokens = max(len(token_ids) - prompt_len, 0)
    full_text = format_token_ids(model.tokenizer, token_ids)

    parsed = parse_generated(full_text)
    parsed["raw"] = full_text
    parsed["stats"] = {
        "seconds": round(elapsed, 3),
        "generated_tokens": generated_tokens,
        "tokens_per_second": round(generated_tokens / elapsed, 1) if elapsed > 0 else None,
        "device": str(entry["device"]),
        "hit_eos": token_ids[-1] == model.tokenizer.eos_token_id if token_ids else False,
    }
    return parsed


def stream_solve(entry: dict, problem: str, params: dict):
    """Yield generated characters one at a time as the model decodes them.

    This mirrors prompt.generate_until_eos exactly (same KV-cache loop, same
    shared sample_next_token sampler) but surfaces each token as it is produced
    so the UI can type out the reasoning live.
    """
    model = entry["model"]
    tokenizer = model.tokenizer
    prompt = build_prompt(normalize_prompt(problem))
    token_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False).tolist()

    supports_cache = "use_cache" in inspect.signature(model.forward).parameters
    past_kvs = None
    step_ids = token_ids
    generated = 0
    hit_eos = False
    started = time.time()

    with entry["lock"]:
        _apply_seed(params.get("seed"))
        with torch.no_grad():
            for _ in range(params["max_new_tokens"]):
                x = torch.tensor([step_ids], dtype=torch.long, device=model.device)
                if supports_cache:
                    logits, past_kvs = model(x, past_kvs=past_kvs, use_cache=True)
                else:
                    logits = model(x)
                next_id = sample_next_token(
                    logits[:, -1, :].squeeze(0),
                    params["temperature"],
                    params["top_k"],
                )
                if next_id == tokenizer.eos_token_id:
                    hit_eos = True
                    break

                token_ids.append(next_id)
                generated += 1
                step_ids = [next_id] if supports_cache else token_ids

                yield {"type": "token", "text": tokenizer.itos.get(next_id, "")}

                if len(token_ids) >= model.max_seq_len:
                    break

    elapsed = time.time() - started
    full_text = format_token_ids(model.tokenizer, token_ids)
    parsed = parse_generated(full_text)
    yield {
        "type": "done",
        "problem": parsed["problem"],
        "steps": parsed["steps"],
        "final": parsed["final"],
        "raw": full_text,
        "stats": {
            "seconds": round(elapsed, 3),
            "generated_tokens": generated,
            "tokens_per_second": round(generated / elapsed, 1) if elapsed > 0 else None,
            "device": str(entry["device"]),
            "hit_eos": hit_eos,
        },
    }


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
def list_checkpoints() -> list[dict]:
    if not CHECKPOINTS_DIR.exists():
        return []
    items = []
    for path in sorted(CHECKPOINTS_DIR.glob("*.pt")):
        stat = path.stat()
        items.append(
            {
                "name": path.name,
                # Path relative to the project root, which is what resolve_path
                # in prompt.py expects for a non-absolute argument.
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "size_mb": round(stat.st_size / 1e6, 1),
                "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
            }
        )
    return items


# ---------------------------------------------------------------------------
# Request parameter parsing / validation
# ---------------------------------------------------------------------------
def clean_params(src: dict) -> dict:
    """Coerce and clamp decoding parameters from a request payload."""

    def as_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def as_int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    temperature = max(0.0, as_float(src.get("temperature"), 0.0))

    top_k_raw = src.get("top_k")
    top_k = as_int(top_k_raw, 0)
    top_k = top_k if top_k and top_k > 0 else None

    max_new_tokens = as_int(src.get("max_new_tokens"), 200)
    max_new_tokens = min(max(max_new_tokens, 1), 1024)

    seed_raw = src.get("seed")
    seed = as_int(seed_raw, None) if seed_raw not in (None, "", "null") else None

    return {
        "temperature": temperature,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "AlignedSLM/1.0"
    protocol_version = "HTTP/1.1"

    # Quieter logging: one concise line per request.
    def log_message(self, fmt, *args):  # noqa: D401
        sys.stderr.write("  %s - %s\n" % (self.address_string(), fmt % args))

    # -- helpers ------------------------------------------------------------
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _resolve_model_args(self, src: dict):
        model_arg = src.get("model") or DEFAULT_MODEL_ARG
        checkpoint_arg = src.get("checkpoint")
        device_arg = src.get("device") or "auto"
        return model_arg, checkpoint_arg, device_arg

    # -- routing ------------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path

        if route in ("/", "/index.html"):
            return self._serve_static("index.html")
        if route.startswith("/static/"):
            return self._serve_static(route[len("/static/"):])
        if route == "/api/checkpoints":
            return self._send_json(
                {
                    "checkpoints": list_checkpoints(),
                    "default_model": DEFAULT_MODEL_ARG,
                    "cuda_available": torch.cuda.is_available(),
                }
            )
        if route == "/api/stream":
            return self._handle_stream(parse_qs(parsed.query))

        return self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        route = urlparse(self.path).path
        body = self._read_json_body()

        if route == "/api/load":
            return self._handle_load(body)
        if route == "/api/solve":
            return self._handle_solve(body)

        return self._send_json({"error": "Not found"}, status=404)

    # -- static -------------------------------------------------------------
    def _serve_static(self, rel_path: str):
        # Prevent path traversal; only serve from STATIC_DIR.
        target = (STATIC_DIR / rel_path).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            return self._send_json({"error": "Not found"}, status=404)

        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".ico": "image/x-icon",
        }.get(target.suffix, "application/octet-stream")
        self._send_bytes(target.read_bytes(), content_type)

    # -- API: load ----------------------------------------------------------
    def _handle_load(self, body: dict):
        model_arg, checkpoint_arg, device_arg = self._resolve_model_args(body)
        if not checkpoint_arg:
            return self._send_json({"error": "No checkpoint specified."}, status=400)
        try:
            entry = REGISTRY.load(model_arg, checkpoint_arg, device_arg)
            return self._send_json({"ok": True, "info": entry["info"]})
        except Exception as exc:  # surfaced to the user as a friendly message
            traceback.print_exc()
            return self._send_json({"error": str(exc)}, status=400)

    # -- API: solve (non-streaming) ----------------------------------------
    def _handle_solve(self, body: dict):
        model_arg, checkpoint_arg, device_arg = self._resolve_model_args(body)
        problem = (body.get("problem") or "").strip()
        if not checkpoint_arg:
            return self._send_json({"error": "No checkpoint specified."}, status=400)
        if not problem:
            return self._send_json({"error": "Please enter a math problem."}, status=400)
        try:
            entry = REGISTRY.load(model_arg, checkpoint_arg, device_arg)
            result = solve(entry, problem, clean_params(body))
            return self._send_json({"ok": True, **result})
        except Exception as exc:
            traceback.print_exc()
            return self._send_json({"error": str(exc)}, status=400)

    # -- API: stream (Server-Sent Events) ----------------------------------
    def _handle_stream(self, query: dict):
        def first(name, default=None):
            values = query.get(name)
            return values[0] if values else default

        checkpoint_arg = first("checkpoint")
        model_arg = first("model") or DEFAULT_MODEL_ARG
        device_arg = first("device") or "auto"
        problem = (first("problem") or "").strip()

        if not checkpoint_arg or not problem:
            return self._send_json(
                {"error": "Missing checkpoint or problem."}, status=400
            )

        params = clean_params(
            {
                "temperature": first("temperature"),
                "top_k": first("top_k"),
                "max_new_tokens": first("max_new_tokens"),
                "seed": first("seed"),
            }
        )

        # SSE headers.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def emit(event: dict):
            chunk = f"data: {json.dumps(event)}\n\n".encode("utf-8")
            self.wfile.write(chunk)
            self.wfile.flush()

        try:
            entry = REGISTRY.load(model_arg, checkpoint_arg, device_arg)
            emit({"type": "start", "info": entry["info"]})
            for event in stream_solve(entry, problem, params):
                emit(event)
        except BrokenPipeError:
            # Client navigated away / closed the tab mid-stream.
            return
        except Exception as exc:
            traceback.print_exc()
            try:
                emit({"type": "error", "message": str(exc)})
            except OSError:
                pass


def parse_args():
    parser = argparse.ArgumentParser(description="Aligned-SLM math reasoning web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default 8000).")
    return parser.parse_args()


def main():
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"

    print("=" * 60)
    print("  Aligned-SLM  ·  Math Reasoning Playground")
    print("=" * 60)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Checkpoints  : {CHECKPOINTS_DIR}  ({len(list_checkpoints())} found)")
    print(f"  CUDA         : {'available' if torch.cuda.is_available() else 'not available (CPU)'}")
    print(f"  Serving at   : {url}")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
