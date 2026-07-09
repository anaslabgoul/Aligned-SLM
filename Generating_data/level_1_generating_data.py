"""
Level 1 data generation: Core Syntax and Arithmetic.

generate_level_1() randomly produces one of four problem types (forward generation
via SymPy; coefficients and constants in [-10, 10]; linear terms use x or y, degree 1):

1. Integer arithmetic   — single +, -, or * on two integers (e.g. 7 - 5 -> 2)
2. Integer chains       — multi-step expressions with order of operations (e.g. 1 - 5 + 7 -> 3)
3. Fraction arithmetic  — +, -, or * on two fractions (e.g. 3/9 - 2/3 -> -1/3)
4. Linear simplify      — simplify parenthesized linear expressions
                          (e.g. Simplify (3*x + 2) - (x - 5) -> 2*x + 7)
"""

import json
import random

import sympy as sp

x, y = sp.symbols("x y")
VARS = (x, y)

MIN_INT = -10
MAX_INT = 10


def _rand_int(low=MIN_INT, high=MAX_INT, exclude_zero=False):
    while True:
        value = random.randint(low, high)
        if not exclude_zero or value != 0:
            return value


def _format_sympy(expr):
    """Render a sympy expression in the project's token vocabulary."""
    simplified = sp.simplify(expr)
    return sp.sstr(simplified, order="lex")


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


def _format_signed_term(value, is_first=False):
    """Format an integer as a term in an infix expression."""
    if is_first:
        return str(value)
    if value >= 0:
        return f"+ {value}"
    return f"- {abs(value)}"


def _build_int_chain():
    """Build a multi-step integer expression and its sympy value."""
    n_terms = random.randint(2, 4) # number of terms in the expression
    values = [_rand_int() for _ in range(n_terms)]
    ops = [random.choice(["+", "-", "*"]) for _ in range(n_terms - 1)]

    parts = [_format_signed_term(values[0], is_first=True)]
    for op, val in zip(ops, values[1:]):
        if op == "+":
            parts.append(_format_signed_term(val))
        elif op == "-":
            parts.append(f"- {val}" if val >= 0 else f"+ {abs(val)}")
        else:
            parts.append(f"* {val}" if val >= 0 else f"* ({val})")

    prob = " ".join(parts)
    return prob, sp.sympify(prob)


def _generate_integer_arithmetic():
    """Addition, subtraction, or multiplication of integers."""
    op = random.choice(["+", "-", "*"])
    a = _rand_int()
    b = _rand_int()

    expr = a + b if op == "+" else (a - b if op == "-" else a * b)

    if op == "+":
        if b >= 0:
            prob = f"{a} + {b}"
        else:
            prob = f"{a} - {abs(b)}"
    elif op == "-":
        if b >= 0:
            prob = f"{a} - {b}"
        else:
            prob = f"{a} + {abs(b)}"
    else:
        if b >= 0:
            prob = f"{a} * {b}"
        else:
            prob = f"{a} * ({b})"

    key_words = [
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
    ""
]
    
    prob = random.choice(key_words) + prob

    return prob, _format_sympy(expr)


def _generate_integer_chain():
    """Multi-step integer expression with order of operations."""
    prob, expr = _build_int_chain()
    key_words = [
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
    ""
]
    
    prob = random.choice(key_words) + prob
    return prob, _format_sympy(expr)


def _generate_fraction_arithmetic():
    """Addition, subtraction, or multiplication of fractions."""
    op = random.choice(["+", "-", "*"])
    num1, den1 = _rand_int(exclude_zero=True), _rand_int(low=1, high=10, exclude_zero=True)
    num2, den2 = _rand_int(exclude_zero=True), _rand_int(low=1, high=10, exclude_zero=True)

    frac1 = sp.Rational(num1, den1)
    frac2 = sp.Rational(num2, den2)

    if op == "+":
        result = frac1 + frac2
    elif op == "-":
        result = frac1 - frac2
    else:
        result = frac1 * frac2

    if op == "+":
        if num2 >= 0:
            prob = f"{num1}/{den1} + {num2}/{den2}"
        else:
            prob = f"{num1}/{den1} - {abs(num2)}/{den2}"
    elif op == "-":
        if num2 >= 0:
            prob = f"{num1}/{den1} - {num2}/{den2}"
        else:
            prob = f"{num1}/{den1} + {abs(num2)}/{den2}"
    else:
        if num2 >= 0:
            prob = f"{num1}/{den1} * {num2}/{den2}"
        else:
            prob = f"{num1}/{den1} * ({num2}/{den2})"

    key_words = [
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
    ""
]
    
    prob = random.choice(key_words) + prob

    return prob, _format_sympy(result)


def _generate_linear_simplify():
    """Simplify parenthesized linear expressions (max degree 1)."""
    var = random.choice(VARS)
    var_name = var.name
    n_groups = random.randint(2, 3)

    groups = []
    sympy_terms = []
    for _ in range(n_groups):
        coef = _rand_int()
        const = _rand_int()
        if coef == 0 and const == 0:
            coef = _rand_int(exclude_zero=True)
            const = _rand_int()
        groups.append(_format_binomial(coef, const, var_name))
        sympy_terms.append(coef * var + const)

    expr = sympy_terms[0]
    prob_parts = [groups[0]]
    for group, term in zip(groups[1:], sympy_terms[1:]):
        if random.random() < 0.5:
            expr = expr + term
            prob_parts.append(f"+ {group}")
        else:
            expr = expr - term
            prob_parts.append(f"- {group}")

    problem = f"{' '.join(prob_parts)}"
    key_words = [
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
    ""
]

    problem = random.choice(key_words) + problem
    return problem, _format_sympy(expr)


def generate_level_1():
    """
    Level 1: Core Syntax and Arithmetic.

    Forward generation of integer, fraction, and linear (degree-1) problems.
    Coefficients and constants are in [-10, 10].
    """
    generators = [
        _generate_integer_arithmetic,
        _generate_integer_chain,
        _generate_fraction_arithmetic,
        _generate_linear_simplify,
    ]
    return random.choice(generators)()


def build_dataset(num_samples, filename="math_curriculum.jsonl"):
    """Generates the dataset and saves it as a JSON Lines file."""
    print(f"Generating {num_samples} samples...")

    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            prob, ans = generate_level_1()
            formatted_text = f"<bos>Problem: {prob}; Answer: {ans}<eos>"
            json_record = {"text": formatted_text}
            f.write(json.dumps(json_record) + "\n")

    print(f"Done! Dataset saved to {filename}")


if __name__ == "__main__":
    build_dataset(1000000, filename="level_1_data.jsonl")
