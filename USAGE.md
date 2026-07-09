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
python train.py --model models/model.py --data Generating_data/math_curriculum.jsonl --epochs 10
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


| Flag           | Default       | Description                         |
| -------------- | ------------- | ----------------------------------- |
| `--batch-size` | `16`          | Batch size                          |
| `--lr`         | `3e-4`        | Learning rate                       |
| `--test-split` | `0.1`         | Fraction of data used for test loss |
| `--seed`       | `42`          | Random seed                         |
| `--output-dir` | `checkpoints` | Directory for the best checkpoint   |
| `--device`     | `auto`        | `auto`, `cpu`, or `cuda`            |




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



## Full workflow example

```powershell
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Generate 1000 training samples
python -c "from Generating_data.level_1_generating_data import build_dataset; build_dataset(1000, 'Generating_data/math_curriculum.jsonl')"

# 3. Train for 20 epochs
python train.py --model models/model.py --data Generating_data/math_curriculum.jsonl --epochs 20 --batch-size 8

# 4. Run a prompt from the best checkpoint
python prompt.py --prompt "<bos>Problem: 9 * 6; Answer:"
```

---



## File reference


| File                                         | Role                                 |
| -------------------------------------------- | ------------------------------------ |
| `Generating_data/level_1_generating_data.py` | Generates Level 1 math JSONL data    |
| `Generating_data/math_curriculum.jsonl`      | Example generated dataset            |
| `train.py`                                   | Training script                      |
| `models/model.py`                            | Model definition and hyperparameters |
| `checkpoints/best_model.pt`                  | Saved best model after training      |


