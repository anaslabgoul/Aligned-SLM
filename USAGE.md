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

Install dependencies if needed:

```bash
pip install torch sympy
```

---

## Generate data

Data is produced as JSON Lines (`.jsonl`). Each line has this shape:

```json
{"text": "<bos>Problem: ...; Answer: ...<eos>"}
```

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

**1000 samples → `Generating_data/math_curriculum.jsonl`**

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

| Argument | Description |
|----------|-------------|
| `--model` | Path to the model module (e.g. `models/model.py`) |
| `--data` | Path to a `.jsonl` or `.json` dataset |
| `--epochs` | Number of training epochs |

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

| Flag | Default | Description |
|------|---------|-------------|
| `--batch-size` | `16` | Batch size |
| `--lr` | `3e-4` | Learning rate |
| `--test-split` | `0.1` | Fraction of data used for test loss |
| `--seed` | `42` | Random seed |
| `--output-dir` | `checkpoints` | Directory for the best checkpoint |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |

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

---

## Full workflow example

```powershell
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Generate 1000 training samples
python -c "from Generating_data.level_1_generating_data import build_dataset; build_dataset(1000, 'Generating_data/math_curriculum.jsonl')"

# 3. Train for 20 epochs
python train.py --model models/model.py --data Generating_data/math_curriculum.jsonl --epochs 20 --batch-size 8
```

---

## File reference

| File | Role |
|------|------|
| `Generating_data/level_1_generating_data.py` | Generates Level 1 math JSONL data |
| `Generating_data/math_curriculum.jsonl` | Example generated dataset |
| `train.py` | Training script |
| `models/model.py` | Model definition and hyperparameters |
| `checkpoints/best_model.pt` | Saved best model after training |
