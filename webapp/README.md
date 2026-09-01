# Math Reasoning Playground

A small, beautiful **web app** for the Aligned-SLM math model. Load any trained
checkpoint, type a math problem, and watch the character-level transformer
**reason step by step** before giving its final answer.

Model &amp; decoding controls live on the left; you type a problem on the right and
the reasoning streams in below it, step by step, ending with a highlighted answer.

> This app is a **UI layer only**. It imports and reuses the project's existing
> inference code from [`prompt.py`](../prompt.py) (the same `load_model_module`,
> `build_model`, `select_device`, `generate_until_eos` and `sample_next_token`
> helpers) and does **not** change the model, tokenizer, checkpoints, or the way
> decoding works. What you see in the browser is byte-for-byte what
> `python prompt.py` produces.

---

## 1. Requirements

Nothing new to install. The server uses only the **Python standard library**
plus the **`torch`** the project already depends on (see
[`../requirements.txt`](../requirements.txt)). If you can already run
`python prompt.py` or `python evaluate_model.py`, you can run this.

- Python 3.10+
- `torch` (CPU is fine; a CUDA GPU is used automatically if available)
- At least one checkpoint in [`../checkpoints/`](../checkpoints) (e.g. `model_1.pt`,
  `model_2.pt`, `model_3.pt`)

## 2. Start the app

From the **project root** (`Aligned-SLM/`):

```bash
python webapp/server.py
```

Then open **http://127.0.0.1:8000** in your browser.

Options:

```bash
python webapp/server.py --port 9000        # use a different port
python webapp/server.py --host 0.0.0.0     # expose on your local network
```

Stop the server with **Ctrl+C**.

> If you use the project's virtual environment, call its Python directly, e.g.
> on Windows: `.\venv\Scripts\python.exe webapp\server.py`

## 3. Using the interface

The screen has a **control sidebar** on the left and the **problem + reasoning**
area on the right.

### ① Model & checkpoint
- **Checkpoint** — pick any `.pt` file found in `checkpoints/`, or choose
  *Custom path…* to point at a checkpoint elsewhere (path is relative to the
  project root, e.g. `checkpoints/my_run.pt`).
- **Model architecture** — the model definition to load the weights into.
  Defaults to `models/model.py`. Only change this if a checkpoint was trained
  with a different architecture file.
- **Device** — `Auto` uses the GPU when one is available, otherwise CPU. You can
  force `CPU` or `CUDA`.
- **Load model** — loads the checkpoint and shows its stats: parameter count,
  device, number of layers, `d_model`, the checkpoint's test loss, and how long
  it took to load. Loading is optional — pressing **Solve** loads the selected
  model automatically if needed.

### ② Decoding settings
- **Temperature** — `0` is greedy/deterministic and is **recommended for correct
  answers**. Higher values add randomness (and more mistakes).
- **Top-k** — optional cap on the sampling pool (only matters when temperature > 0).
- **Max tokens** — safety limit on how much the model may generate (1–1024).
- **Seed** — set an integer for reproducible sampling.

These map exactly to the flags on `prompt.py` / `evaluate_model.py`.

### ③ Your math problem
- Type a problem, or click one of the **example chips** (grouped by curriculum
  level L1–L3).
- **Live reasoning** (on by default) streams the model's output token by token so
  you watch each reasoning step appear. Turn it off for a single, instant result.
- Press **Solve & show reasoning** or hit **Ctrl/Cmd + Enter**.

### Reasoning & answer
- Each `step:` the model produces is rendered as a numbered card, with light
  syntax highlighting for numbers, variables, and operators.
- The **final answer** (the model's last step) is highlighted separately and can
  be copied with one click.
- A stats row shows total time, tokens generated, tokens/second, the device used,
  and whether the model finished cleanly (`✓ complete`) or ran into the token
  limit (`⚠ hit token limit`).
- **Show raw model output** reveals the exact untouched text the model generated.

### Extras
- **History** — your recent problems and answers are saved in the browser
  (`localStorage`) so you can re-run them with a click. Nothing leaves your machine.
- **Theme** — toggle light/dark with the ◐ button (top right).

## 4. How a problem is fed to the model

The app builds the exact training-time prompt and reads the answer the same way
the evaluation script does:

```
prompt      →  <bos>Problem: <your problem>; step:
model output→  problem: ...; step: <intermediate>; step: <final answer>
```

Everything after the last `step:` is treated as the final answer. Answers and
steps never contain `;`, which is why the app can split cleanly on it — the same
assumption [`evaluate_model.py`](../evaluate_model.py) relies on.

## 5. Which checkpoint should I use?

Later checkpoints were trained on more of the curriculum, so they handle harder
problems. For example, `model_1.pt` was trained mainly on Level 1 and will give
wrong answers on Level 2/3 expansions, while `model_3.pt` handles all three
levels. Load different checkpoints and compare — that is what the app is for.

## 6. Endpoints (for reference)

The frontend talks to a tiny JSON/SSE API, in case you want to script it:

| Method | Path               | Purpose                                             |
|--------|--------------------|-----------------------------------------------------|
| GET    | `/api/checkpoints` | List `.pt` files in `checkpoints/` + GPU availability |
| POST   | `/api/load`        | Load a checkpoint, return its stats                 |
| POST   | `/api/solve`       | Solve a problem, return structured steps + answer   |
| GET    | `/api/stream`      | Same as solve, streamed token-by-token (SSE)        |

## 7. Troubleshooting

- **"Could not reach the server."** — the server isn't running, or the browser is
  pointed at the wrong port. Start it with `python webapp/server.py` and use the
  URL it prints.
- **"This checkpoint does not match the architecture…"** — the `.pt` was trained
  with different hyper-parameters than `models/model.py` declares. Select the
  matching model file, or a checkpoint trained with the current architecture.
  (This is the same guard `evaluate_model.py` shows.)
- **CUDA option is greyed out** — no GPU was detected; the app runs on CPU, which
  is perfectly fine for this model.
- **Answer looks wrong** — try a stronger checkpoint (e.g. `model_3.pt`) and keep
  temperature at `0`. Small models still make mistakes on harder problems.

## 8. Files

```
webapp/
├── server.py            # stdlib HTTP server; wraps the existing inference code
├── static/
│   └── index.html       # the single-page UI (self-contained, no external deps)
└── README.md            # this file
```
