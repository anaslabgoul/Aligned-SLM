"""
Level 1.1 data generation: Integer arithmetic only.

generate_level_1_1() produces single-operation integer problems (+, -, or *)
on two integers (e.g. 7 - 5 -> 2).
"""

import json
import random

import sympy as sp

MIN_INT = -100
MAX_INT = 100


def _rand_int(low=MIN_INT, high=MAX_INT, exclude_zero=False):
    while True:
        value = random.randint(low, high)
        if not exclude_zero or value != 0:
            return value


def _format_sympy(expr):
    """Render a sympy expression in the project's token vocabulary."""
    simplified = sp.simplify(expr)
    return sp.sstr(simplified, order="lex")


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
        "",
    ]

    prob = random.choice(key_words) + prob

    return prob, _format_sympy(expr)


def generate_level_1_1():
    """
    Level 1.1: Integer arithmetic only.

    Forward generation of single-operation integer problems (+, -, or *).
    Operands are in [-100, 100].
    """
    return _generate_integer_arithmetic()


def build_dataset(num_samples, filename="level_1_1_data.jsonl"):
    """Generates the dataset and saves it as a JSON Lines file."""
    print(f"Generating {num_samples} samples...")

    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            prob, ans = generate_level_1_1()
            formatted_text = f"<bos>Problem: {prob}; Answer: {ans}<eos>"
            json_record = {"text": formatted_text}
            f.write(json.dumps(json_record) + "\n")

    print(f"Done! Dataset saved to {filename}")


if __name__ == "__main__":
    build_dataset(100000, filename="level_1_1_data.jsonl")
