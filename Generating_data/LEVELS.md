# Data generation levels

This folder contains scripts for generating curriculum-style math data.

Each example is written as JSON Lines (`.jsonl`) with the shape:

```json
{"text": "<bos>Problem: ...; Answer: ...<eos>"}
```

## Level 1 — Core Syntax and Arithmetic

Source: `Generating_data/level_1_generating_data.py`

**What it generates**

- Coefficients and constants are sampled in **[-10, 10]**
- Problems are **forward-generated** and answers are computed with **SymPy** (then simplified)
- Linear expressions are **degree 1** and use a random variable from **`x` or `y`**
- One sample is produced by randomly choosing one of the 4 problem types below

### 1) Integer arithmetic (single-step)

**Description**

- A single operation (`+`, `-`, or `*`) on two integers

**Example**

- `Problem: Calculate 7 - 5; Answer: 2`

**Keywords (prefixes)**

- `Calculate`
- `Find the result of`
- `Evaluate the expression`
- `What is the value of`
- `Solve`
- `Compute`
- `Determine the value of`
- `Work out`
- `What does this equal`
- `Give the result for`
- `Perform the calculation`
- *(empty string)*

### 2) Integer chains (multi-step)

**Description**

- A multi-step integer expression (2–4 terms) using a mix of `+`, `-`, `*`
- Intended to exercise parsing and order of operations

**Example**

- `Problem: Evaluate the expression 1 - 5 + 7; Answer: 3`

**Keywords (prefixes)**

- `Calculate`
- `Find the result of`
- `Evaluate the expression`
- `What is the value of`
- `Solve`
- `Compute`
- `Determine the value of`
- `Work out`
- `What does this equal`
- `Give the result for`
- `Perform the calculation`
- *(empty string)*

### 3) Fraction arithmetic (single-step)

**Description**

- A single operation (`+`, `-`, or `*`) on two rational numbers written as `num/den`

**Example**

- `Problem: What is the value of 3/9 - 2/3; Answer: -1/3`

**Keywords (prefixes)**

- `Calculate`
- `Find the result of`
- `Evaluate the expression`
- `What is the value of`
- `Solve`
- `Compute`
- `Determine the value of`
- `Work out`
- `What does this equal`
- `Give the result for`
- `Perform the calculation`
- *(empty string)*

### 4) Linear simplify (degree-1 simplify)

**Description**

- Builds 2–3 parenthesized linear groups like `(a*x + b)` and combines them with `+` or `-`
- The answer is the simplified linear expression (e.g. `2*x + 7`)

**Example**

- `Problem: Simplify (3*x + 2) - (x - 5); Answer: 2*x + 7`

**Keywords (prefixes)**

- `Simplify`
- `Simplify the expression`
- `Reduce`
- `Reduce the expression`
- `Expand and simplify`
- `Rewrite in simplest form`
- `Combine like terms`
- `Simplify the following:`
- `Put in simplest form`
- `Solve for the simplified version`
- *(empty string)*

