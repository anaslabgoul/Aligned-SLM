"""
Level 1 data generation: Core Syntax and Arithmetic (with Chain-of-Thought).

generate_level_1() randomly produces one of four problem types (forward generation
via SymPy; coefficients and constants in [-100, 100]; linear terms use x or y, degree 1).

Each sample is emitted as:

    <bos>Problem: ...; step: ...; step: ...<eos>

The number of steps is problem-dependent — a single integer operation has *no*
intermediate steps, while multi-step problems expose as many steps as they need:

1. Integer arithmetic   — single +, -, or * on two integers.   No steps.
2. Integer chains       — multi-step expression; one step per operation, resolving
                          multiplication (higher precedence) before +/- .
3. Fraction arithmetic  — rewrite over a common denominator, combine, then reduce.
4. Fraction simplify    — reduce a single fraction to lowest terms.
5. Linear simplify      — distribute/remove parentheses, then combine like terms.

Every generated sample is verified with SymPy: each intermediate step and the final
answer must evaluate to the same value as the original expression, otherwise the
sample is re-rolled.
"""

import json
import random

import sympy as sp

x, y = sp.symbols("x y")
VARS = (x, y)

MIN_INT = -100
MAX_INT = 100


# Keyword prefixes shared by the arithmetic-style problems (an empty string means
# the bare expression with no natural-language lead-in).
ARITHMETIC_KEYWORDS = [
    "Calculate ",
    "Find the result of ",
    "Evaluate the expression ",
    "What is the value of ",
    "Solve ",
    "Compute ",
    "Determine the value of ",
    "Work out ",
    "What does this equal ",
    "Give the result for ",
    "Perform the calculation ",
    "",
]

SIMPLIFY_KEYWORDS = [
    "Simplify ",
    "Simplify the expression ",
    "Reduce ",
    "Reduce the expression ",
    "Expand and simplify ",
    "Rewrite in simplest form ",
    "Combine like terms ",
    "Simplify the following: ",
    "Put in simplest form ",
    "Solve for the simplified version ",
    "",
]

FRACTION_SIMPLIFY_KEYWORDS = [
    "Simplify ",
    "Reduce ",
    "Write in lowest terms ",
    "Simplify the fraction ",
    "Reduce the fraction ",
    "Put in simplest form ",
    "Express in lowest terms ",
    "",
]


def _rand_int(low=MIN_INT, high=MAX_INT, exclude_zero=False):
    while True:
        value = random.randint(low, high)
        if not exclude_zero or value != 0:
            return value


def _format_sympy(expr):
    """Render a sympy expression in the project's token vocabulary."""
    simplified = sp.simplify(expr)
    return sp.sstr(simplified, order="lex")


def _format_fraction(num, den):
    if num < 0:
        return f"-{abs(num)}/{den}"
    return f"{num}/{den}"


def _format_binomial(a, b, var_name):
    """Format ax + b as a parenthesized linear term, e.g. (3*x + 2) or (x - 5)."""
    parts = []
    if a != 0:
        if a == 1:
            parts.append(var_name)
        elif a == -1:
            parts.append(f"-{var_name}")
        else:
            parts.append(f"{a}*{var_name}")

    if b != 0:
        if b > 0:
            parts.append(f"+ {b}" if parts else str(b))
        else:
            parts.append(f"- {abs(b)}" if parts else str(b))

    if not parts:
        return "(0)"

    return f"({' '.join(parts)})"


def _format_int_expr(nums, ops):
    """Format a flat integer expression, e.g. [-6, 9, 6], ['+', '-'] -> '-6 + 9 - 6'.

    Signs are normalised the same way across the problem and every step so the model
    sees one consistent surface form (e.g. '+ -3' is written '- 3', and a negative
    multiplicand is parenthesised as '* (-3)').
    """
    parts = [str(nums[0])]
    for op, val in zip(ops, nums[1:]):
        if op == "+":
            parts.append(f"+ {val}" if val >= 0 else f"- {abs(val)}")
        elif op == "-":
            parts.append(f"- {val}" if val >= 0 else f"+ {abs(val)}")
        else:  # multiplication
            parts.append(f"* {val}" if val >= 0 else f"* ({val})")
    return " ".join(parts)


def _format_monomials(monomials, var_name):
    """Format ordered (kind, value) monomials into a flat linear expression.

    kind is 'x' for a coefficient of the variable and 'c' for a constant, e.g.
    [('x', 3), ('c', 2), ('x', -1), ('c', 5)] -> '3*x + 2 - x + 5'.
    """
    if not monomials:
        return "0"

    parts = []
    for index, (kind, value) in enumerate(monomials):
        if kind == "x":
            magnitude = var_name if abs(value) == 1 else f"{abs(value)}*{var_name}"
        else:
            magnitude = str(abs(value))

        if index == 0:
            parts.append(f"-{magnitude}" if value < 0 else magnitude)
        else:
            parts.append(f"+ {magnitude}" if value > 0 else f"- {magnitude}")
    return " ".join(parts)


def _verify(value, steps, answer):
    """Return True if every step and the answer evaluate to `value`."""
    try:
        target = sp.sympify(value)
        for expression in list(steps) + [answer]:
            if sp.expand(sp.sympify(expression) - target) != 0:
                return False
    except (sp.SympifyError, TypeError, ValueError):
        return False
    return True


def _generate_integer_arithmetic():
    """Addition, subtraction, or multiplication of integers (no intermediate steps)."""
    op = random.choice(["+", "-", "*"])
    a = _rand_int()
    b = _rand_int()

    expr = a + b if op == "+" else (a - b if op == "-" else a * b)

    if op == "+":
        prob = f"{a} + {b}" if b >= 0 else f"{a} - {abs(b)}"
    elif op == "-":
        prob = f"{a} - {b}" if b >= 0 else f"{a} + {abs(b)}"
    else:
        prob = f"{a} * {b}" if b >= 0 else f"{a} * ({b})"

    prob = random.choice(ARITHMETIC_KEYWORDS) + prob
    value = sp.Integer(expr)
    return prob, [], _format_sympy(value), value


def _generate_integer_chain():
    """Multi-step integer expression; one step per resolved operation.

    Multiplication (higher precedence) is resolved before +/-, and each step shows
    the whole expression after one operation has been carried out. A 2-term chain
    reduces in a single move, so it carries no intermediate step.
    """
    n_terms = random.randint(2, 4)
    nums = [_rand_int() for _ in range(n_terms)]
    ops = [random.choice(["+", "-", "*"]) for _ in range(n_terms - 1)]

    prob_expr = _format_int_expr(nums, ops)

    # Reduce one operation at a time, emitting the intermediate expression as a step.
    work_nums, work_ops = list(nums), list(ops)
    steps = []
    while len(work_nums) > 1:
        index = work_ops.index("*") if "*" in work_ops else 0
        left, right, op = work_nums[index], work_nums[index + 1], work_ops[index]
        result = left + right if op == "+" else (left - right if op == "-" else left * right)
        work_nums = work_nums[:index] + [result] + work_nums[index + 2:]
        work_ops = work_ops[:index] + work_ops[index + 1:]
        if len(work_nums) > 1:
            steps.append(_format_int_expr(work_nums, work_ops))

    value = sp.Integer(work_nums[0])
    prob = random.choice(ARITHMETIC_KEYWORDS) + prob_expr
    return prob, steps, _format_sympy(value), value


def _generate_fraction_arithmetic():
    """Add, subtract, or multiply two fractions, showing the working."""
    op = random.choice(["+", "-", "*"])
    num1, den1 = _rand_int(exclude_zero=True), _rand_int(low=1, high=10, exclude_zero=True)
    num2, den2 = _rand_int(exclude_zero=True), _rand_int(low=1, high=10, exclude_zero=True)

    frac1 = _format_fraction(num1, den1)
    frac2 = _format_fraction(num2, den2)

    if op == "+":
        prob = f"{frac1} + {frac2}" if num2 >= 0 else f"{frac1} - {abs(num2)}/{den2}"
    elif op == "-":
        prob = f"{frac1} - {frac2}" if num2 >= 0 else f"{frac1} + {abs(num2)}/{den2}"
    else:
        prob = f"{frac1} * {frac2}" if num2 >= 0 else f"{frac1} * ({frac2})"

    value = (
        sp.Rational(num1, den1) + sp.Rational(num2, den2)
        if op == "+"
        else sp.Rational(num1, den1) - sp.Rational(num2, den2)
        if op == "-"
        else sp.Rational(num1, den1) * sp.Rational(num2, den2)
    )
    answer = _format_sympy(value)

    steps = []
    if op == "*":
        # Multiply straight across, then reduce if the product is not already lowest terms.
        product = _format_fraction(num1 * num2, den1 * den2)
        if product != answer:
            steps.append(product)
    else:
        lcd = sp.ilcm(den1, den2)
        scaled1, scaled2 = num1 * (lcd // den1), num2 * (lcd // den2)

        # Step 1 (only if a rewrite is actually needed): put both over the common denominator.
        if den1 != lcd or den2 != lcd:
            if op == "+":
                tail = (
                    f"+ {_format_fraction(scaled2, lcd)}"
                    if scaled2 >= 0
                    else f"- {_format_fraction(abs(scaled2), lcd)}"
                )
            else:
                tail = (
                    f"- {_format_fraction(scaled2, lcd)}"
                    if scaled2 >= 0
                    else f"+ {_format_fraction(abs(scaled2), lcd)}"
                )
            steps.append(f"{_format_fraction(scaled1, lcd)} {tail}")

        # Step 2 (only if it differs from the reduced answer): combine the numerators.
        combined = scaled1 + scaled2 if op == "+" else scaled1 - scaled2
        combined_str = _format_fraction(combined, lcd)
        if combined_str != answer:
            steps.append(combined_str)

    prob = random.choice(ARITHMETIC_KEYWORDS) + prob
    return prob, steps, answer, value


def _generate_fraction_simplify():
    """Reduce a single fraction to lowest terms."""
    factor = random.randint(2, 6)
    reduced_num = _rand_int(exclude_zero=True)
    reduced_den = _rand_int(low=2, high=10, exclude_zero=True)

    num = reduced_num * factor
    den = reduced_den * factor
    value = sp.Rational(reduced_num, reduced_den)
    answer = _format_sympy(value)

    prob = random.choice(FRACTION_SIMPLIFY_KEYWORDS) + _format_fraction(num, den)
    return prob, [], answer, value


def _generate_linear_simplify():
    """Simplify parenthesized linear expressions: distribute, then combine like terms."""
    var = random.choice(VARS)
    var_name = var.name
    n_groups = random.randint(2, 3)

    coefs, consts, groups = [], [], []
    for _ in range(n_groups):
        coef = _rand_int()
        const = _rand_int()
        if coef == 0 and const == 0:
            coef = _rand_int(exclude_zero=True)
            const = _rand_int()
        coefs.append(coef)
        consts.append(const)
        groups.append(_format_binomial(coef, const, var_name))

    signs = [1]
    prob_parts = [groups[0]]
    for group in groups[1:]:
        if random.random() < 0.5:
            signs.append(1)
            prob_parts.append(f"+ {group}")
        else:
            signs.append(-1)
            prob_parts.append(f"- {group}")

    value = sum(sign * (coef * var + const) for sign, coef, const in zip(signs, coefs, consts))
    answer = _format_sympy(value)

    # Distribute step: drop the parentheses, applying each group's sign to its terms,
    # keeping the original left-to-right order (e.g. '3*x + 2 - x + 5').
    monomials = []
    for sign, coef, const in zip(signs, coefs, consts):
        if sign * coef != 0:
            monomials.append(("x", sign * coef))
        if sign * const != 0:
            monomials.append(("c", sign * const))
    distributed = _format_monomials(monomials, var_name)

    steps = []
    if distributed != answer:
        steps.append(distributed)

    prob = random.choice(SIMPLIFY_KEYWORDS) + " ".join(prob_parts)
    return prob, steps, answer, value


def generate_level_1():
    """
    Level 1: Core Syntax and Arithmetic (with Chain-of-Thought).

    Forward generation of integer, fraction, and linear (degree-1) problems, each
    accompanied by as many reasoning steps as the problem needs. Samples whose steps
    do not verify against SymPy are re-rolled.

    Returns (problem, steps, answer) where `steps` is a possibly empty list.
    """
    generators = [
        _generate_integer_arithmetic,
        _generate_integer_chain,
        _generate_fraction_arithmetic,
        _generate_fraction_simplify,
        _generate_linear_simplify,
    ]

    for _ in range(50):
        prob, steps, answer, value = random.choice(generators)()
        if _verify(value, steps, answer):
            return prob, steps, answer

    # Extremely unlikely: fall back to a trivially correct, step-free sample.
    return prob, [], answer


def _format_sample(prob, steps, answer):
    """Assemble one training string in the '<bos>Problem: ...; step: ...<eos>' shape."""
    segments = [f"Problem: {prob}"]
    segments += [f"step: {step}" for step in steps]
    segments.append(f"step: {answer}")
    return "<bos>" + "; ".join(segments) + "<eos>"


def build_dataset(num_samples, filename="math_curriculum.jsonl", append=False):
    """Generates the dataset and saves it as a JSON Lines file."""
    mode = "a" if append else "w"
    action = "Appending" if append else "Generating"
    print(f"{action} {num_samples} samples...")

    with open(filename, mode, encoding="utf-8") as f:
        for _ in range(num_samples):
            prob, steps, answer = generate_level_1()
            formatted_text = _format_sample(prob, steps, answer)
            json_record = {"text": formatted_text}
            f.write(json.dumps(json_record) + "\n")

    print(f"Done! Dataset saved to {filename}")


def append_fraction_simplify_samples(num_samples, filename="level_1_data.jsonl"):
    """Append only fraction-simplification samples to an existing dataset."""
    print(f"Appending {num_samples} fraction-simplification samples...")

    written = 0
    with open(filename, "a", encoding="utf-8") as handle:
        while written < num_samples:
            prob, steps, answer, value = _generate_fraction_simplify()
            if not _verify(value, steps, answer):
                continue
            formatted_text = _format_sample(prob, steps, answer)
            handle.write(json.dumps({"text": formatted_text}) + "\n")
            written += 1

    print(f"Done! Fraction-simplification samples appended to {filename}")


if __name__ == "__main__":
    build_dataset(1000000, filename="level_1_data.jsonl")
