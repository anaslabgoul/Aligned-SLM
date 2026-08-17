# Aligned-SLM

A character-level language model trained **from scratch** to do step-by-step
mathematics, then progressively taught harder material through a curriculum — and
next, **aligned with DPO** so it prefers correct reasoning over incorrect
reasoning.

The project has one property that makes it unusually clean to study: **every
answer is verifiable with SymPy**. That single fact drives the whole pipeline —
data is generated and checked automatically, models are scored per-operation
automatically, and preference data for alignment is labelled automatically with no
human annotation.

![Accuracy evolution across the three curriculum stages](assets/accuracy_evolution.png)

*Left: overall accuracy on each level as training progresses through the three
stages. Right: Level 1 accuracy broken down by operation — every operation
improves even as Level 1's share of the data shrinks. See [Results](#results).*

---

## What the model is

A small GPT-style decoder, implemented from scratch in [`models/`](models/):

| Component        | Choice                                                        |
| ---------------- | ------------------------------------------------------------- |
| Tokenizer        | Character-level, vocab **56** (`a–z 0–9 * + - / ^ = ( ) …` + `<bos>/<eos>/<unk>`) |
| Dimensions       | `d_model=384`, `n_layers=12`, `n_heads=6`, context `1024`     |
| Block            | Pre-LayerNorm, causal multi-head self-attention, 4× ReLU FFN  |
| Positions        | Learned positional embeddings                                 |
| Inference        | KV-cache for O(n) decoding                                    |
| Size             | **~21.7M parameters**                                         |

Everything the model reads and writes is a string of characters in one format:

```
<bos>Problem: <question>; step: <step_1>; step: <step_2>; step: <final answer><eos>
```

The final answer is the text after the **last** `step:`. There is no separate
"answer" token — reasoning and answer are one continuous character stream, and the
model learns chain-of-thought by imitation.

---

## The data: a verifiable math curriculum

Problems are generated procedurally in [`Generating_data/`](Generating_data/), and
**every generated sample is checked with SymPy** — each intermediate step and the
final answer must be algebraically equal to the original expression, or the sample
is re-rolled. Difficulty is organized into three levels:

| Level | Focus                                                              | Example operations |
| ----- | ----------------------------------------------------------------- | ------------------ |
| **1** | Core syntax & arithmetic                                          | integer arithmetic, integer chains, fraction arithmetic, fraction simplification, linear simplification |
| **2** | Polynomials, distribution, linear solving                        | polynomial expand/sum, distributive, mixed chains, linear solve |
| **3** | Mixed polynomial simplification & fractional distribution        | mixed polynomial, fraction distribute |

---

## Training: curriculum by continued fine-tuning

Three checkpoints were trained, each **initialized from the previous one** and
trained on a progressively harder mixture (mixtures are built by
[`dataset_mix.py`](dataset_mix.py), which draws a weighted blend of the level
datasets):

| Model       | Initialized from | Data mixture              | LR     | Epochs |
| ----------- | ---------------- | ------------------------- | ------ | ------ |
| **Model 1** | scratch          | 100% Level 1              | 3e-4   | 20     |
| **Model 2** | Model 1          | 50% Level 1 · 50% Level 2 | 2e-4   | 20     |
| **Model 3** | Model 2          | 40% Level 1 · 40% Level 2 · 20% Level 3 | 1e-4 | 20 |

The learning rate is lowered at each stage, since each model starts from an
already-competent checkpoint rather than from noise.

---

## Results

All numbers below are from the Weights & Biases runs: **greedy decoding
(temperature 0), 400 freshly-generated problems per operation**, with answers
checked by SymPy (so `2*x + 7` and `7 + 2*x` both count as correct). Each model was
evaluated on **the levels it was trained on**.

### Level 1 — accuracy (%) as the curriculum grows

| Operation            | Model 1 | Model 2 | Model 3 |
| -------------------- | :-----: | :-----: | :-----: |
| integer_arithmetic   |  70.50  |  80.75  | **83.00** |
| integer_chain        |  47.50  |  60.75  | **63.50** |
| fraction_arithmetic  |  36.25  |  57.00  | **66.50** |
| fraction_simplify    |  88.00  |  94.00  | **96.50** |
| linear_simplify      |  45.00  |  53.00  | **54.75** |
| **Overall**          | **57.45** | **69.10** | **72.85** |

> **Positive transfer, no forgetting.** Level 1's share of the training data fell
> from 100% → 50% → 40%, yet Level 1 accuracy *rose on every single operation*.
> Learning harder material made the model better at the easy material.

### Level 2 — accuracy (%)

| Operation          | Model 2 | Model 3 |
| ------------------ | :-----: | :-----: |
| polynomial_expand  |  100.0  |  100.0  |
| polynomial_sum     |  92.00  |  93.00  |
| distributive       |  100.0  |  100.0  |
| mixed_chain        |  94.50  |  94.50  |
| linear_solve       |  99.75  |  100.0  |
| **Overall**        | **97.25** | **97.50** |

> **Mastered and retained.** Level 2 was essentially solved by Model 2, and
> introducing Level 3 did not regress it — Model 3 held 97.5%.

### Level 3 — accuracy (%) (Model 3 only)

| Operation           | Model 3 |
| ------------------- | :-----: |
| mixed_polynomial    |  26.00  |
| fraction_distribute |  23.25  |
| **Overall**         | **24.62** |

> **The frontier.** Level 3 — multi-step polynomial simplification and fractional
> distribution — sits at ~25%. This is the hardest, most compositional material,
> and it's where the next step is aimed.

### The story in one line

| Level tested | Model 1 | Model 2 | Model 3 |
| ------------ | :-----: | :-----: | :-----: |
| Level 1      | 57.45   | 69.10   | 72.85   |
| Level 2      | —       | 97.25   | 97.50   |
| Level 3      | —       | —       | 24.62   |

The single-step operations (integer arithmetic, fraction simplify, polynomial
expand, distributive) are essentially solved. The **multi-step / compositional**
operations lag at every level — `integer_chain` and `linear_simplify` in Level 1,
`mixed_chain` in Level 2, and both Level 3 operations. Chained reasoning is the
consistent weak point, which is exactly what alignment targets next.

---

## Next: DPO alignment

The plan is to align **Model 3** with **Direct Preference Optimization (DPO)** —
teaching it to *prefer* correct chains over incorrect ones — using the tooling in
this repo:

- [`Generating_data/generate_dpo_data.py`](Generating_data/generate_dpo_data.py) —
  samples several chains per problem, checks each with the same SymPy verifier used
  for evaluation, and pairs a **correct** chain (`chosen`) against a **wrong** one
  (`rejected`). No human labels.
- [`train_dpo.py`](train_dpo.py) — optimizes the DPO loss against a frozen copy of
  Model 3 as the reference.

**Why this should work here.** DPO needs problems where the model produces *both* a
correct and an incorrect chain — a "contested" problem. Level 3's ~25% pass-rate is
close to ideal for this: pairs form easily, unlike a near-solved level where the
model is almost always right. The weak multi-step operations in Levels 1–2
(`integer_chain`, `linear_simplify`, `mixed_chain`) are contested too and can be
mixed in.

**What to watch during DPO** — reward accuracy climbing toward 1.0 and a positive,
growing reward margin (the chosen chain pulling ahead of the rejected one) — and
then the real test, downstream evaluation accuracy.

### Model 3 → Model 3 + DPO (to be filled in)

| Level tested | Model 3 (baseline) | Model 3 + DPO |
| ------------ | :----------------: | :-----------: |
| Level 1      | 72.85              | _pending_     |
| Level 2      | 97.50              | _pending_     |
| Level 3      | 24.62              | _pending_     |

The key questions: **how much does Level 3 improve**, and does alignment hold Levels
1–2 steady (no regression)? That before/after delta is the headline result the
alignment step is meant to produce.

---

## Repository layout

| Path                                   | Role                                                        |
| -------------------------------------- | ----------------------------------------------------------- |
| `models/`                              | Model, attention, FFN, and character tokenizer (from scratch) |
| `Generating_data/`                     | Per-level procedural data generators (SymPy-verified)       |
| `Generating_data/generate_dpo_data.py` | Build DPO preference pairs from a trained checkpoint         |
| `train.py` / `train_model.py`          | Train from scratch / continue from a checkpoint             |
| `training_common.py`                   | Shared data loading, optimizer, schedule, and training loop |
| `dataset_mix.py`                       | Weighted mixing of several level datasets                   |
| `train_dpo.py`                         | Align a checkpoint with DPO                                 |
| `prompt.py`                            | Generate from a trained checkpoint                          |
| `evaluate_model.py`                    | Per-operation accuracy evaluation (SymPy-checked)           |
| `wandb_logging.py`                     | Optional Weights & Biases experiment tracking               |

The naming convention for W&B runs is:

- **Training** — `train_model_<n>_data_<L1%>[_<L2%>][_<L3%>]` (the model and its
  training mixture, e.g. `train_model_1_data_100`, `..._40_40_20`).
- **Evaluation** — `eval_model_<n>_level_<k>` (which checkpoint, evaluated on which
  level).

For exact commands — generating data, training, evaluating, and running the full
DPO workflow — see [`USAGE.md`](USAGE.md).
