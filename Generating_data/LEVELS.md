# Data generation levels

This folder contains scripts for generating curriculum-style math data.

Each example is written as JSON Lines (`.jsonl`) with the shape:

```json
{"text": "<bos>Problem: ...; step: ...; step: ...<eos>"}
```

The number of intermediate `step:` segments is problem-dependent — the **last
`step:` is always the final answer**, and a problem that needs no working out
(e.g. a single integer product) has that step alone. Powers are written with `^`,
never `**`.

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

## Level 2 — Polynomials, Distribution, Mixed Arithmetic and Linear Solving

Source: `Generating_data/level_2_generating_data.py`

**What it generates**

- Coefficients and constants are sampled in **[-10, 10]** (`MIN_INT` / `MAX_INT`)
- Polynomials go up to **degree 2** (`POLY_MAX_DEGREE`)
- Fraction denominators are sampled in **[1, 10]** (`FRACTION_DEN_MAX`)
- Problems are **forward-generated** with **SymPy**, and every sample is verified
  before it is returned: each step and the answer must be equal to the original
  expression (or, for equations, to the unique solution)
- One sample is produced by randomly choosing one of the 5 problem types below

### 1) Polynomial expand

**Description**

- Raises a linear binomial `(a*x + b)` to a power (2, up to `POLY_MAX_DEGREE`)
- The first step writes the power out as a repeated product, the last step gives
  the expanded polynomial

**Example**

- `Problem: Write the expansion of (4*y + 6)^2; step: (4*y + 6)*(4*y + 6); step: 16*y^2 + 48*y + 36`

**Keywords (prefixes)**

- `Expand`
- `Expand the expression`
- `Expand and simplify`
- `Multiply out`
- `Write the expansion of`
- *(empty string)*

### 2) Polynomial sum

**Description**

- Adds or subtracts two random polynomials of degree up to `POLY_MAX_DEGREE`
- The first step drops the parentheses (flipping signs on a subtraction), the
  last step combines like terms

**Example**

- `Problem: Add (4*x^2 + 8*x - 4) - (10*x - 1); step: 4*x^2 + 8*x - 4 - 10*x + 1; step: 4*x^2 - 2*x - 3`

**Keywords (prefixes)**

- `Add`
- `Sum`
- `Find the sum of`
- `Combine`
- `Add together`
- *(empty string)*

### 3) Distributive property

**Description**

- Multiplies an integer scalar through a parenthesized polynomial, written
  either as `k*(...)` or `(...)*k`
- The first step writes out the term-by-term products without evaluating them,
  the last step performs the multiplications

**Example**

- `Problem: Multiply out -9*(-7*x - 10); step: 9*7*x + 9*10; step: 63*x + 90`

**Keywords (prefixes)**

- `Expand`
- `Use the distributive property on`
- `Distribute`
- `Multiply out`
- `Expand the expression`
- *(empty string)*

### 4) Mixed chains (integers and fractions)

**Description**

- A 2–3 term chain over `+`, `-`, `*` mixing integers with rational literals
- At least one term is always a fraction
- Multiplications are resolved first (one step each), then the additive terms
  are put over a common denominator and combined, mirroring Level 1's fraction
  walkthrough

**Example**

- `Problem: Perform the calculation 1/3 - 5 * 2/5; step: 1/3 - 2; step: 1/3 - 6/3; step: -5/3`

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

### 5) Linear solve (one-step equations)

**Description**

- Solves a one-step linear equation for `x` or `y`, drawn from four templates:
  `x + b = c`, `x - b = c`, `a*x = c`, `x/a = c`
- The step shows the inverse operation applied to the right-hand side, the last
  step gives the solved value
- Both the step and the answer are written as `var = ...`

**Example**

- `Problem: Determine y if y/10 = -4/5; step: y = -4/5*10; step: y = -8`

**Keywords (prefixes)** — `{var}` is replaced with the variable name

- `Solve for {var}:`
- `Find {var} in`
- `What is {var} in`
- `Determine {var} if`
- `Solve the equation`
- *(empty string)*

## Level 3 — Mixed Polynomial Simplification and Fractional Distribution

Source: `Generating_data/level_3_generating_data.py`

**What it generates**

- Coefficients and constants are sampled in **[-10, 10]** (`MIN_INT` / `MAX_INT`)
- Polynomials go up to **degree 2** (`POLY_MAX_DEGREE`)
- Each problem chains **2–3 operands** with `+` / `-` (`MAX_TERMS`)
- Fractional coefficients are built from numerators in **[-9, 9]**
  (`FRACTION_NUM_MAX`) over denominators in **[2, 10]** (`FRACTION_DEN_MAX`),
  kept smaller than the integer range because distribution multiplies them
- Degenerate samples are re-rolled (up to `MAX_ATTEMPTS = 20`): a result of `0`,
  or one whose variable terms all cancel to a bare constant, is rejected
- Problems are **forward-generated** with **SymPy** and every step is verified
  against the original expression
- One sample is produced by randomly choosing one of the 2 problem types below

Both types follow the same chain-of-thought shape as Levels 1 and 2: open up the
constructs first, then drop parentheses, then combine like terms.

### 1) Mixed polynomial

**Description**

- Combines Level 2's expand, sum and distributive types into one problem
- Each operand is randomly one of:
  - a plain polynomial — `(a*x^2 + b*x + c)`
  - a scalar times a polynomial — `k*(...)` or `(...)*k`
  - a product of two linear factors — `(a*x + b)*(c*x + d)`
  - a squared binomial — `(a*x + b)^2`
- At least one operand is guaranteed to be a construct that must be opened up,
  so the problem never collapses to a plain Level 2 polynomial sum
- Steps: open every construct (products written term-by-term, powers written as
  repeated products) → drop the parentheses → combine like terms

**Example**

- `Problem: Write in simplest form (7*x - 5)^2 + (x - 7); step: (7*x - 5)*(7*x - 5) + (x - 7); step: 49*x^2 - 70*x + 25 + x - 7; step: 49*x^2 - 69*x + 18`
- `Problem: Expand and simplify (10*y - 9)*(-8*y + 7) + (8*y^2 - 9*y + 6); step: (-10*y*8*y + 10*y*7 + 9*8*y - 9*7) + (8*y^2 - 9*y + 6); step: -80*y^2 + 142*y - 63 + 8*y^2 - 9*y + 6; step: -72*y^2 + 133*y - 57`

**Keywords (prefixes)**

- `Expand and simplify`
- `Simplify`
- `Expand`
- `Multiply out and simplify`
- `Write in simplest form`
- `Combine like terms in`
- *(empty string)*

### 2) Fraction distribute

**Description**

- A sum/difference of **first-degree** operands whose coefficients may be
  fractions, with at least one operand distributed
- The fraction usually rides on the outer scalar (the textbook `1/2*(4*x - 6)`
  shape, `SCALAR_FRACTION_PROB = 0.75`); the coefficients inside the parentheses
  are fractional less often (`INNER_FRACTION_PROB = 0.25`) to stop denominators
  from exploding once three operands are combined
- At least one fractional coefficient is guaranteed to appear
- Steps follow Level 1's fraction walkthrough, one stage per step: distribute →
  multiply straight across (unreduced) → reduce → put like terms over a common
  denominator → combine → reduce. A stage that would just repeat the previous
  line or the answer is dropped

**Example**

- `Problem: Multiply out (-1/2)*(-6*y + 1) - 1*(10*y - 9); step: 1/2*6*y - 1/2*1 - 1*10*y + 1*9; step: 6/2*y - 1/2 - 10*y + 9; step: 3*y - 1/2 - 10*y + 9; step: 3*y - 1/2 - 10*y + 18/2; step: -7*y + 17/2`

**Keywords (prefixes)**

- `Expand`
- `Use the distributive property on`
- `Distribute`
- `Multiply out`
- `Expand and simplify`
- `Simplify`
- *(empty string)*

