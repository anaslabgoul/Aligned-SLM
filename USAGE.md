# Data generation and training

Terminal commands for generating math curriculum data and training the character-level language model.

Run all commands from the project root:

```text
c:\Users\DELL\Desktop\Aligned-SLM
```

## Setup

Activate the virtual environment (recommended):

**Windows (PowerShell)**

```powershell
.\venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install the CUDA build of PyTorch **first** — plain PyPI serves
the CPU-only wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Experiment tracking is optional; everything runs without it. To use it,
authenticate once:

```bash
wandb login
```

See [Track experiments with Weights & Biases](#track-experiments-with-weights--biases).

---

## Generate data

Data is produced as JSON Lines (`.jsonl`). Each line has this shape:

```json
{"text": "<bos>Problem: ...; step: ...; step: ...<eos>"}
```

The final answer is the text after the **last** `step:`.



### Default generation (100 samples)

`level_1_generating_data.py` writes `math_curriculum.jsonl` in the current working directory.

From the project root:

```powershell
python Generating_data/level_1_generating_data.py
```

From inside `Generating_data/`:

```powershell
cd Generating_data
python level_1_generating_data.py
```



### Custom sample count or output file

Use Python to call `build_dataset` directly.

**1000 samples →** `Generating_data/math_curriculum.jsonl`

```powershell
python -c "from Generating_data.level_1_generating_data import build_dataset; build_dataset(1000, 'Generating_data/math_curriculum.jsonl')"
```

**5000 samples → custom path**

```powershell
python -c "from Generating_data.level_1_generating_data import build_dataset; build_dataset(5000, 'Generating_data/my_dataset.jsonl')"
```



### What gets generated

`level_1_generating_data.py` creates Level 1 math problems:

- Integer arithmetic (`+`, `-`, `*`)
- Multi-step integer expressions
- Fraction arithmetic
- Linear expression simplification (variables `x` and `y`)

---



## Train a model

Training is launched with `train.py`. You must pass:


| Argument   | Description                                       |
| ---------- | ------------------------------------------------- |
| `--model`  | Path to the model module (e.g. `models/model.py`) |
| `--data`   | Path to a `.jsonl` or `.json` dataset             |
| `--epochs` | Number of training epochs                         |




### Basic training

```powershell
python train.py --model models/model.py --data Generating_data/math_curriculum.jsonl --epochs 10
```



### Training with extra options

```powershell
python train.py `
  --model models/model.py `
  --data Generating_data/math_curriculum.jsonl `
  --epochs 20 `
  --batch-size 8 `
  --lr 3e-4 `
  --test-split 0.1 `
  --output-dir checkpoints `
  --device auto
```

**Linux / macOS** (single line):

```bash
python train.py --model models/model.py --data Generating_data/math_curriculum.jsonl --epochs 20 --batch-size 8 --lr 3e-4 --test-split 0.1 --output-dir checkpoints --device auto
```



### Optional flags


| Flag                | Default       | Description                                        |
| ------------------- | ------------- | -------------------------------------------------- |
| `--batch-size`      | `32`          | Batch size                                         |
| `--lr`              | `3e-4`        | Learning rate                                      |
| `--test-split`      | `0.1`         | Fraction of data used for test loss                |
| `--seed`            | `42`          | Random seed                                        |
| `--evals-per-epoch` | `0`           | Evaluate N times per epoch instead of only at epoch end |
| `--eval-batches`    | `0`           | Cap mid-epoch evaluations to N test batches (`0` = full test set) |
| `--output-dir`      | `checkpoints` | Directory for the best checkpoint                  |
| `--device`          | `auto`        | `auto`, `cpu`, or `cuda`                           |
| `--wandb`           | off           | Log the run to Weights & Biases                    |

`train.py` also accepts the full set of `--wandb-*` flags — see
[Track experiments with Weights & Biases](#track-experiments-with-weights--biases).




### Training output

Each epoch prints training and test loss:

```text
Epoch 1/10 | train loss: 3.6570 | test loss: 2.6327
  -> New best model saved (test loss: 2.6327)
```

The best checkpoint (lowest test loss) is saved to:

```text
checkpoints/best_model.pt
```

### Short runs over large datasets

By default the test loss is only measured **once per epoch**. If you stop after
one or two epochs — because the dataset is large and the loss has already
flattened — that gives you only one or two test-loss measurements for the whole
run. Two consequences:

- there is no test-loss curve to look at, only a couple of isolated points;
- `best_model.pt` is only considered for saving at those same one or two moments,
  so the best model within an epoch can be passed over and never saved.

`--evals-per-epoch` fixes both. It measures the test loss at evenly spaced points
inside the epoch, and each measurement is a chance to save a new best checkpoint:

```powershell
python train.py `
  --model models/model.py `
  --data level_1_data.jsonl `
  --epochs 2 `
  --evals-per-epoch 10 `
  --eval-batches 20
```

```text
Epoch 1/2 | batch 120/1200 | train loss: 1.8421 | lr: 2.94e-04
Epoch 1/2 | step 120 | test loss: 1.7233  -> New best model saved
Epoch 1/2 | batch 240/1200 | train loss: 1.6104 | lr: 2.81e-04
Epoch 1/2 | step 240 | test loss: 1.5502  -> New best model saved
```

Two epochs at `--evals-per-epoch 10` gives 20 test-loss points instead of 2.

**Use `--eval-batches` to keep it cheap.** A full test-set pass ten times per
epoch is real overhead: with the default `--test-split 0.1` it adds roughly
another epoch's worth of forward passes. `--eval-batches 20` scores only the
first 20 test batches at each mid-epoch checkpoint, which is plenty to see the
trend. The epoch-end evaluation always uses the **full** test set regardless, so
your headline numbers stay exact.

In W&B, mid-epoch points carry `test/partial = 1` and full-test-set points carry
`test/partial = 0`, so a noisy sampled point is never mistaken for an exact one.

> **Note:** the training loss needs none of this — `train/loss_step` is logged on
> every optimizer step, so even a single epoch produces a dense training curve.

---

## Continue training from a checkpoint

Use `train_model.py` to keep training a model that was already saved with `train.py` (or a previous run of `train_model.py`). It loads the weights from a `.pt` checkpoint, then trains on a new dataset for more epochs.

Required arguments (everything from `train.py`, plus `--checkpoint`):

| Argument       | Description                                              |
| -------------- | -------------------------------------------------------- |
| `--model`      | Path to the model module (e.g. `models/model.py`)        |
| `--checkpoint` | Path to a saved checkpoint (e.g. `checkpoints/model_1.pt`) |
| `--data`       | Path to a `.jsonl` or `.json` dataset                    |
| `--epochs`     | Number of additional training epochs                     |

The checkpoint must contain a `model_state_dict` key (same format as `checkpoints/best_model.pt`).

### Basic continued training

**Windows (PowerShell)**

```powershell
python train_model.py `
  --model models/model.py `
  --checkpoint checkpoints/model_1.pt `
  --data level_1_data.jsonl `
  --epochs 10
```

**Linux / macOS**

```bash
python train_model.py \
  --model models/model.py \
  --checkpoint checkpoints/model_1.pt \
  --data level_1_data.jsonl \
  --epochs 10
```

### Continued training with extra options

```bash
python train_model.py \
  --model models/model.py \
  --checkpoint checkpoints/model_1.pt \
  --data level_1_data.jsonl \
  --epochs 20 \
  --batch-size 16 \
  --lr 3e-4 \
  --test-split 0.04 \
  --output-dir checkpoints \
  --device cuda
```

To avoid overwriting an earlier best checkpoint, set a different output directory or filename:

```bash
python train_model.py \
  --model models/model.py \
  --checkpoint checkpoints/model_1.pt \
  --data level_1_1_data.jsonl \
  --epochs 20 \
  --output-dir checkpoints
```

Then rename the saved file if needed, e.g. `checkpoints/best_model.pt` → `checkpoints/model_2.pt`.

### Optional flags

Same as `train.py`:

| Flag             | Default       | Description                         |
| ---------------- | ------------- | ----------------------------------- |
| `--batch-size`   | `32`          | Batch size                          |
| `--lr`           | `3e-4`        | Peak learning rate                  |
| `--min-lr`       | `3e-5`        | Final learning rate after cosine decay |
| `--warmup-steps` | `500`         | Linear warmup steps                 |
| `--weight-decay` | `0.1`         | AdamW weight decay                  |
| `--grad-clip`    | `1.0`         | Max gradient norm (`0` disables)    |
| `--test-split`      | `0.1`      | Fraction of data used for test loss |
| `--seed`            | `42`       | Random seed                         |
| `--evals-per-epoch` | `0`        | Evaluate N times per epoch          |
| `--eval-batches`    | `0`        | Cap mid-epoch evaluations to N test batches |
| `--output-dir`   | `checkpoints` | Directory for the best checkpoint   |
| `--device`       | `auto`        | `auto`, `cpu`, or `cuda`            |
| `--wandb`        | off           | Log the run to Weights & Biases     |

### Notes

- The optimizer and learning-rate schedule **start fresh**; only model weights are loaded from `--checkpoint`.
- Use the **same model architecture** as when the checkpoint was created (`models/model.py` must match).
- At startup, the script prints the loaded checkpoint path and its saved epoch (if present).
- With `--wandb`, the run config records `resumed_from` and `resumed_epoch`, so you can trace which checkpoint a run continued from.

---

## Track experiments with Weights & Biases

Tracking is **opt-in**. Without `--wandb`, nothing changes: no network calls, no
extra files, and `wandb` does not even need to be installed. Pass `--wandb` and
the run streams to your dashboard instead.

### One-time setup

```powershell
pip install wandb
wandb login
```

`wandb login` asks for the API key from <https://wandb.ai/authorize> and caches
it, so later runs need no further authentication.

### Track a training run

```powershell
python train.py `
  --model models/model.py `
  --data level_1_data.jsonl `
  --epochs 20 `
  --device cuda `
  --wandb `
  --wandb-run-name level1-baseline `
  --wandb-tags level1 baseline
```

The script prints the run URL at startup. The same flags work on `train_model.py`.

### What gets logged

**Per optimizer step**

| Metric                 | Meaning                                                     |
| ---------------------- | ----------------------------------------------------------- |
| `train/loss_step`      | Loss for that batch                                          |
| `train/lr`             | Current learning rate (warmup, then cosine decay)            |
| `train/grad_norm`      | Gradient norm **before** clipping — spikes reveal instability |
| `train/epoch_progress` | Fractional epoch, for aligning runs of different lengths      |

**At every evaluation** — epoch end, plus each mid-epoch point when
`--evals-per-epoch` is set

| Metric              | Meaning                                                       |
| ------------------- | ------------------------------------------------------------- |
| `test/loss`         | Held-out loss                                                  |
| `test/perplexity`   | `exp(test/loss)`                                               |
| `test/best_loss`    | Best held-out loss so far                                      |
| `test/partial`      | `1` if scored on a capped `--eval-batches` subset, `0` if on the full test set |
| `epoch`             | Fractional epoch (e.g. `1.4`), so it works as an x-axis         |
| `train/loss_epoch`  | Token-weighted mean training loss (epoch end only)             |

**Run config** — every CLI argument (lr, batch size, seed, warmup, weight decay,
grad clip, dataset path, …) plus the architecture read from `models/model.py`
(`d_model`, `n_heads`, `n_layers`, `dropout`, `max_seq_len`, `vocab_size`),
parameter count, device, and train/test split sizes. This is what makes runs
comparable and filterable later.

**Run summary** — `best_test_loss`, `best_epoch` and `best_step`, the columns
worth sorting the run table by.

If you stop training after one or two epochs, set `--evals-per-epoch` so the
test-loss curve has more than a couple of points — see
[Short runs over large datasets](#short-runs-over-large-datasets).

Note that `--grad-clip 0` disables clipping, and `train/grad_norm` is then not
logged — the norm is a by-product of clipping and is not computed separately.

### W&B flags

| Flag                 | Default       | Description                                                     |
| -------------------- | ------------- | --------------------------------------------------------------- |
| `--wandb`            | off           | Enable tracking; all other `--wandb-*` flags are ignored without it |
| `--wandb-project`    | `aligned-slm` | Project name in the dashboard                                    |
| `--wandb-entity`     | your account  | Team or user to log under                                        |
| `--wandb-run-name`   | auto          | Run name (W&B generates one if omitted)                          |
| `--wandb-tags`       | none          | Space-separated tags, e.g. `--wandb-tags level2 finetune`         |
| `--wandb-mode`       | `online`      | `online`, `offline`, or `disabled`                               |
| `--wandb-artifacts`  | off           | Upload the best checkpoint as a versioned model artifact (training scripts only) |

### Try it without uploading anything

`--wandb-mode offline` writes everything to a local `wandb/` directory and makes
no network calls — useful for a first check:

```powershell
python train.py --model models/model.py --data level_1_data.jsonl --epochs 1 --wandb --wandb-mode offline
```

Upload it afterwards if the run looks right:

```powershell
wandb sync wandb/offline-run-<timestamp>-<id>
```

### Versioned checkpoints

`--wandb-artifacts` uploads `best_model.pt` at the end of training, tagged with
its test loss and epoch:

```powershell
python train.py --model models/model.py --data level_1_data.jsonl --epochs 20 --wandb --wandb-artifacts
```

> **Note:** with the current architecture the checkpoint is a few hundred MB, so
> this is off by default. It uploads once at the end of training, not on every
> improvement.

### Track evaluation results

`evaluate_model.py` accepts the same flags and logs per-operation accuracy as
metrics, a table, and a bar chart:

```powershell
python evaluate_model.py --level 1 --num-samples 100 --show-errors 3 --wandb
```

To attach the results to the **training run** that produced the checkpoint —
so accuracy sits alongside that run's loss curves instead of in a separate run —
pass its run id (the short id in the run URL):

```powershell
python evaluate_model.py --level 1 --num-samples 100 --wandb --wandb-run-id abc123xy
```

With `--show-errors N`, the example failures are also uploaded as a table you can
sort and filter in the dashboard.

---



## Test a checkpoint with a prompt

Use `prompt.py` to load a trained checkpoint and generate model output.

### Basic prompt test

```powershell
python prompt.py --prompt "<bos>Problem: 7 + 5; Answer:"
```

By default, this uses:

- Model file: `models/model.py`
- Checkpoint: `checkpoints/best_model.pt`
- Device: `auto` (CUDA if available, otherwise CPU)

Notes:

- `prompt.py` now keeps your prompt exactly as typed (including `<bos>`, `:`, and `;`).
- If a character is outside the tokenizer vocabulary, the model may still emit `<unk>` in output.
- Generation now stops early when `<eos>` is produced; otherwise it stops at `--max-new-tokens`.



### Use a specific checkpoint

```powershell
python prompt.py --checkpoint checkpoints/best_model.pt --prompt "<bos>Problem: simplify (3*x + 2) - (x - 5); Answer:"
```



### Control generation behavior

```powershell
python prompt.py `
  --prompt "<bos>Problem: 4/5 - 1/2; Answer:" `
  --max-new-tokens 80 `
  --temperature 0.7 `
  --top-k 20 `
  --print-full-output
```



### Prompt script arguments


| Flag                  | Default                     | Description                         |
| --------------------- | --------------------------- | ----------------------------------- |
| `--model`             | `models/model.py`           | Model module path                   |
| `--checkpoint`        | `checkpoints/best_model.pt` | Checkpoint file to load             |
| `--prompt`            | required                    | Input text prompt                   |
| `--max-new-tokens`    | `120`                       | Max generated tokens                |
| `--temperature`       | `0.8`                       | Sampling temperature (`0` = greedy) |
| `--top-k`             | none                        | Optional top-k sampling             |
| `--device`            | `auto`                      | `auto`, `cpu`, or `cuda`            |
| `--print-full-output` | off                         | Print full text including prompt    |


---



## Evaluate a model per-operation

`evaluate_model.py` measures how accurately a checkpoint answers each **operation
type** within a level, so you can see exactly where the model gets things wrong.
For a chosen level it generates fresh problems separately per operation, feeds each
to the model, reads the model's final answer (the text after the last `step:`),
compares it to the ground truth with **SymPy** (so `7 + 2*x` and `2*x + 7` count as
equal), and reports the percentage correct per operation.



### Basic evaluation

```powershell
python evaluate_model.py --level 1 --num-samples 50
```

This generates 50 problems for **each** Level 1 operation and prints a table:

```text
=== Evaluation: level 1 | 50 samples/operation | greedy ===
operation               correct   total   accuracy
--------------------------------------------------
integer_arithmetic           47      50      94.0%
integer_chain                41      50      82.0%
fraction_arithmetic          38      50      76.0%
fraction_simplify            44      50      88.0%
linear_simplify              36      50      72.0%
--------------------------------------------------
overall                     206     250      82.4%
```



### Operations per level

| Level | Operations |
| ----- | ---------- |
| 1     | `integer_arithmetic`, `integer_chain`, `fraction_arithmetic`, `fraction_simplify`, `linear_simplify` |
| 2     | `polynomial_expand`, `polynomial_sum`, `distributive`, `mixed_chain`, `linear_solve` |



### See where the model fails

Print example wrong answers per operation with `--show-errors`:

```powershell
python evaluate_model.py --level 1 --num-samples 100 --show-errors 3
```

```text
--- Example errors: fraction_arithmetic ---
  Problem  : Find the result of 17/9 * (-69/7)
  Expected : -391/21
  Predicted: -1213/63
```



### Evaluate specific operations only

```powershell
python evaluate_model.py --level 1 --num-samples 200 --operations integer_chain fraction_simplify
```



### Save a JSON report

```powershell
python evaluate_model.py --level 2 --num-samples 100 --save-report report.json
```



### Evaluation arguments


| Flag                  | Default                     | Description                                        |
| --------------------- | --------------------------- | -------------------------------------------------- |
| `--model`             | `models/model.py`           | Model module path                                  |
| `--checkpoint`        | `checkpoints/best_model.pt` | Checkpoint to evaluate                             |
| `--level`             | `1`                         | Curriculum level (`1` or `2`)                      |
| `-n`, `--num-samples` | `50`                        | Problems generated and evaluated **per operation** |
| `--operations`        | all in the level            | Subset of operation names to evaluate              |
| `--max-new-tokens`    | `200`                       | Max tokens generated per problem                   |
| `--temperature`       | `0.0`                       | Sampling temperature (`0` = greedy, recommended)   |
| `--top-k`             | none                        | Top-k cutoff (only used when temperature > 0)      |
| `--device`            | `auto`                      | `auto`, `cpu`, or `cuda`                           |
| `--seed`              | `42`                        | Seed for reproducible problem generation           |
| `--show-errors`       | `0`                         | Print up to N example wrong answers per operation  |
| `--save-report`       | none                        | Write a JSON report to this path                   |
| `--wandb`             | off                         | Log per-operation accuracy to Weights & Biases     |
| `--wandb-run-id`      | none                        | Attach results to an existing W&B run instead of creating one |


> **Note:** the checkpoint must match the architecture currently declared in
> `models/model.py` (same `d_model` / `n_heads` / `n_layers`). If it doesn't, the
> script stops with a clear message — retrain with the current `model.py`, or
> restore `model.py` to the checkpoint's architecture.

---



## Full workflow example

```powershell
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Generate 1000 training samples
python -c "from Generating_data.level_1_generating_data import build_dataset; build_dataset(1000, 'Generating_data/math_curriculum.jsonl')"

# 3. Train for 20 epochs, tracking the run on W&B
python train.py --model models/model.py --data Generating_data/math_curriculum.jsonl --epochs 20 --batch-size 8 --evals-per-epoch 10 --eval-batches 20 --wandb --wandb-run-name level1-baseline

# 4. Run a prompt from the best checkpoint
python prompt.py --prompt "<bos>Problem: 9 * 6; step:"

# 5. Evaluate accuracy per operation, attaching the results to the training run
python evaluate_model.py --level 1 --num-samples 100 --show-errors 3 --wandb --wandb-run-id <run-id-from-step-3>
```

Drop the `--wandb` flags from steps 3 and 5 to run without any tracking.

---



## File reference


| File                                         | Role                                 |
| -------------------------------------------- | ------------------------------------ |
| `Generating_data/level_1_generating_data.py` | Generates Level 1 math JSONL data    |
| `Generating_data/math_curriculum.jsonl`      | Example generated dataset            |
| `train.py`                                   | Training script (train from scratch) |
| `train_model.py`                             | Continue training from a checkpoint  |
| `training_common.py`                         | Shared data loading, model building, and training loop |
| `models/model.py`                            | Model definition and hyperparameters |
| `checkpoints/best_model.pt`                  | Saved best model after training      |
| `prompt.py`                                  | Generate text from a trained checkpoint |
| `evaluate_model.py`                          | Per-operation accuracy evaluation    |
| `wandb_logging.py`                           | Optional Weights & Biases experiment tracking |
| `requirements.txt`                           | Python dependencies                  |


