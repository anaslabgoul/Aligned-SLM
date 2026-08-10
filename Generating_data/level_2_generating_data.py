"""
Level 2 data generation: Polynomials, Distribution, Mixed Arithmetic, and Linear Solving.

generate_level_2() randomly produces one of four problem types (forward generation
via SymPy; coefficients and constants in [MIN_INT, MAX_INT]; polynomial degree up to
POLY_MAX_DEGREE; powers are written with ^, not **):

Each sample is emitted as:

    <bos>Problem: ...; step: ...; step: ...<eos>

1. Polynomial expand / sum — expand (ax + b)^n or add/subtract two polynomials.
2. Distributive property   — multiply a scalar through a parenthesized polynomial.
3. Mixed chains            — multi-step expressions mixing integers and fractions.
4. Linear solve            — solve one-step linear equations for x or y.

Polynomial problems follow the same chain-of-thought pattern as Level 1: drop
parentheses first, then calculate/combine terms.

Every generated sample is verified with SymPy before being returned.
"""

import json
import random

import sympy as sp

# --- Configuration -----------------------------------------------------------
MIN_INT = -10
MAX_INT = 10
POLY_MAX_DEGREE = 2
# Denominators for fraction literals are sampled from [1, FRACTION_DEN_MAX].
FRACTION_DEN_MAX = 10
# -----------------------------------------------------------------------------

x, y = sp.symbols("x y")
VARS = (x, y)

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

EXPAND_KEYWORDS = [
    "Expand ",
    "Expand the expression ",
    "Expand and simplify ",
    "Multiply out ",
    "Write the expansion of ",
    "",
]

SUM_KEYWORDS = [
    "Add ",
    "Sum ",
    "Find the sum of ",
    "Combine ",
    "Add together ",
    "",
]

DISTRIBUTE_KEYWORDS = [
    "Expand ",
    "Use the distributive property on ",
    "Distribute ",
    "Multiply out ",
    "Expand the expression ",
    "",
]

SOLVE_KEYWORDS = [
    "Solve for {var}: ",
    "Find {var} in ",
    "What is {var} in ",
    "Determine {var} if ",
    "Solve the equation ",
    "",
]


def _rand_int(low=MIN_INT, high=MAX_INT, exclude_zero=False):
    while True:
        value = random.randint(low, high)
        if not exclude_zero or value != 0:
            return value


def _sympify(text):
    return sp.sympify(str(text).replace("^", "**"))


def _format_sympy(expr):
    """Render a sympy expression using the project's token vocabulary."""
    simplified = sp.expand(sp.simplify(expr))
    return sp.sstr(simplified, order="lex").replace("**", "^")


def _format_signed_value(value, is_first=False):
    if is_first:
        return str(value)
    if value >= 0:
        return f"+ {value}"
    return f"- {abs(value)}"


def _format_fraction(num, den):
    if num < 0:
        return f"-{abs(num)}/{den}"
    return f"{num}/{den}"


def _format_monomial(coef, degree, var_name):
    if coef == 0:
        return None

    if degree == 0:
        magnitude = str(abs(coef))
    elif degree == 1:
        if abs(coef) == 1:
            magnitude = var_name
        else:
            magnitude = f"{abs(coef)}*{var_name}"
    else:
        if abs(coef) == 1:
            magnitude = f"{var_name}^{degree}"
        else:
            magnitude = f"{abs(coef)}*{var_name}^{degree}"

    if coef < 0:
        return f"-{magnitude}"
    return magnitude


def _format_polynomial(expr, var_name):
    poly = sp.Poly(sp.expand(expr), sp.Symbol(var_name))
    terms = []
    for degree, coef in enumerate(poly.all_coeffs()):
        power = poly.degree() - degree
        term = _format_monomial(int(coef), power, var_name)
        if term is not None:
            terms.append(term)

    if not terms:
        return "0"

    formatted = terms[0]
    for term in terms[1:]:
        if term.startswith("-"):
            formatted += f" - {term[1:]}"
        else:
            formatted += f" + {term}"
    return formatted


def _iter_polynomial_terms(expr, var_name, sign=1):
    poly = sp.Poly(sp.expand(expr), sp.Symbol(var_name))
    terms = []
    for degree, coef in enumerate(poly.all_coeffs()):
        power = poly.degree() - degree
        signed_coef = sign * int(coef)
        if signed_coef != 0:
            terms.append((power, signed_coef))
    return terms


def _format_polynomial_terms(terms, var_name):
    if not terms:
        return "0"

    parts = []
    for power, coef in terms:
        monomial = _format_monomial(abs(coef), power, var_name)
        if not parts:
            parts.append(f"-{monomial}" if coef < 0 else monomial)
        elif coef > 0:
            parts.append(f"+ {monomial}")
        else:
            parts.append(f"- {monomial}")
    return " ".join(parts)


def _format_polynomial_sum_distributed(poly_a, poly_b, var_name, op):
    terms = _iter_polynomial_terms(poly_a, var_name, sign=1)
    second_sign = 1 if op == "+" else -1
    terms.extend(_iter_polynomial_terms(poly_b, var_name, sign=second_sign))
    return _format_polynomial_terms(terms, var_name)


def _format_distributive_step(scalar, inner, var_name):
    terms = _iter_polynomial_terms(inner, var_name)
    parts = []
    for power, coef in terms:
        signed_product = scalar * coef
        if power == 0:
            magnitude = f"{abs(scalar)}*{abs(coef)}"
        else:
            monomial = _format_monomial(abs(coef), power, var_name)
            magnitude = f"{abs(scalar)}*{monomial}"

        if not parts:
            parts.append(f"-{magnitude}" if signed_product < 0 else magnitude)
        elif signed_product > 0:
            parts.append(f"+ {magnitude}")
        else:
            parts.append(f"- {magnitude}")
    return " ".join(parts)


def _format_binomial_product(base_text, degree):
    if degree <= 1:
        return base_text
    return "*".join([base_text] * degree)


def _format_binomial(a, b, var_name):
    """Format ax + b as a parenthesized linear binomial."""
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


def _rand_polynomial(var, max_degree):
    degree = random.randint(1, max_degree)
    expr = 0
    for power in range(degree, -1, -1):
        if power == degree:
            coef = _rand_int(exclude_zero=True)
        else:
            coef = _rand_int()
        if coef != 0:
            expr += coef * var**power

    if expr == 0:
        expr = _rand_int(exclude_zero=True) * var**degree
    return sp.expand(expr)


def _verify_expression(value, steps, answer):
    try:
        target = sp.expand(_sympify(value))
        for expression in list(steps) + [answer]:
            if sp.expand(_sympify(expression) - target) != 0:
                return False
    except (sp.SympifyError, TypeError, ValueError):
        return False
    return True


def _verify_equation(equation_text, var, steps, answer):
    try:
        lhs_text, rhs_text = equation_text.split("=", maxsplit=1)
        equation = sp.Eq(_sympify(lhs_text), _sympify(rhs_text))
        answer_lhs, answer_rhs = answer.split("=", maxsplit=1)
        if answer_lhs.strip() != var.name:
            return False
        solution = sp.solve(equation, var)
        if len(solution) != 1:
            return False
        if sp.simplify(solution[0] - _sympify(answer_rhs)) != 0:
            return False

        for step in steps:
            step_lhs, step_rhs = step.split("=", maxsplit=1)
            if step_lhs.strip() != var.name:
                return False
            if sp.simplify(_sympify(step_rhs) - solution[0]) != 0:
                return False
    except (sp.SympifyError, TypeError, ValueError):
        return False
    return True


def _generate_polynomial_expand():
    var = random.choice(VARS)
    var_name = var.name
    min_degree = 2 if POLY_MAX_DEGREE >= 2 else 1
    degree = random.randint(min_degree, POLY_MAX_DEGREE)

    a = _rand_int(exclude_zero=True)
    b = _rand_int()
    base = a * var + b
    expr = sp.expand(base**degree)

    base_text = _format_binomial(a, b, var_name)
    power_text = str(degree) if degree > 1 else ""
    if power_text:
        problem_core = f"{base_text}^{power_text}"
    else:
        problem_core = base_text

    prob = random.choice(EXPAND_KEYWORDS) + problem_core
    answer = _format_polynomial(expr, var_name)
    steps = []
    expanded_form = _format_binomial_product(base_text, degree)
    if expanded_form != problem_core:
        steps.append(expanded_form)
    return prob, steps, answer, expr


def _generate_polynomial_sum():
    var = random.choice(VARS)
    var_name = var.name

    poly_a = _rand_polynomial(var, POLY_MAX_DEGREE)
    poly_b = _rand_polynomial(var, POLY_MAX_DEGREE)
    op = random.choice(["+", "-"])
    expr = sp.expand(poly_a + poly_b if op == "+" else poly_a - poly_b)

    poly_a_text = _format_polynomial(poly_a, var_name)
    poly_b_text = _format_polynomial(poly_b, var_name)
    if op == "+":
        prob = random.choice(SUM_KEYWORDS) + f"({poly_a_text}) + ({poly_b_text})"
    else:
        prob = random.choice(SUM_KEYWORDS) + f"({poly_a_text}) - ({poly_b_text})"
    answer = _format_polynomial(expr, var_name)

    distributed = _format_polynomial_sum_distributed(poly_a, poly_b, var_name, op)
    steps = []
    if distributed != answer:
        steps.append(distributed)

    return prob, steps, answer, expr


def _generate_distributive():
    var = random.choice(VARS)
    var_name = var.name
    poly_degree = random.randint(1, POLY_MAX_DEGREE)

    coeffs = {}
    for power in range(poly_degree, -1, -1):
        if power == poly_degree:
            coeffs[power] = _rand_int(exclude_zero=True)
        else:
            coeffs[power] = _rand_int()

    inner = sum(coef * var**power for power, coef in coeffs.items())
    inner_text = _format_polynomial(inner, var_name)

    scalar = _rand_int(exclude_zero=True)
    if random.random() < 0.5:
        prob_core = f"{scalar}*({inner_text})"
    else:
        prob_core = f"({inner_text})*{scalar}"

    expr = sp.expand(scalar * inner)
    answer = _format_polynomial(expr, var_name)

    prob = random.choice(DISTRIBUTE_KEYWORDS) + prob_core
    distributed = _format_distributive_step(scalar, inner, var_name)
    steps = []
    if distributed != answer:
        steps.append(distributed)
    return prob, steps, answer, expr


def _rand_mixed_term():
    if random.random() < 0.5:
        return _rand_int(), "int"

    num = _rand_int(exclude_zero=True)
    den = _rand_int(low=1, high=FRACTION_DEN_MAX, exclude_zero=True)
    return sp.Rational(num, den), "frac"


def _format_mixed_value(value):
    if isinstance(value, sp.Rational) and value.q != 1:
        return _format_fraction(int(value.p), int(value.q))
    return str(int(value))


def _is_fraction(value):
    return isinstance(value, sp.Rational) and value.q != 1


def _format_mixed_expr(values, ops):
    parts = [_format_mixed_value(values[0])]
    for op, value in zip(ops, values[1:]):
        rendered = _format_mixed_value(value)
        if op == "+":
            if value >= 0:
                parts.append(f"+ {rendered}")
            else:
                parts.append(f"- {_format_mixed_value(-value)}")
        elif op == "-":
            if value >= 0:
                parts.append(f"- {rendered}")
            else:
                parts.append(f"+ {_format_mixed_value(-value)}")
        elif value >= 0:
            parts.append(f"* {rendered}")
        else:
            parts.append(f"* ({rendered})")
    return " ".join(parts)


def _evaluate_mixed_chain(values, ops):
    work_values, work_ops = list(values), list(ops)
    while len(work_values) > 1:
        index = work_ops.index("*") if "*" in work_ops else 0
        left = work_values[index]
        right = work_values[index + 1]
        op = work_ops[index]
        if op == "+":
            result = left + right
        elif op == "-":
            result = left - right
        else:
            result = left * right
        work_values = work_values[:index] + [result] + work_values[index + 2:]
        work_ops = work_ops[:index] + work_ops[index + 1:]
    return sp.simplify(work_values[0])


def _format_unreduced_product(left, right):
    left_r = sp.Rational(left)
    right_r = sp.Rational(right)
    return _format_fraction(int(left_r.p * right_r.p), int(left_r.q * right_r.q))


def _add_fraction_pair_steps(accumulator, next_term, rest):
    steps = []
    acc_rat = sp.Rational(accumulator).limit_denominator()
    next_rat = sp.Rational(next_term).limit_denominator()
    lcd = sp.ilcm(acc_rat.q, next_rat.q)
    acc_scaled = int(acc_rat.p * (lcd // acc_rat.q))
    next_scaled = int(next_rat.p * (lcd // next_rat.q))
    needs_lcd_step = acc_rat.q != lcd or next_rat.q != lcd

    if needs_lcd_step:
        converted_parts = [
            _format_fraction(acc_scaled, lcd),
            (
                f"+ {_format_fraction(next_scaled, lcd)}"
                if next_term >= 0
                else f"- {_format_fraction(abs(next_scaled), lcd)}"
            ),
        ]
        for term in rest:
            _append_signed_term(converted_parts, term)
        steps.append(" ".join(converted_parts))

    combined_num = acc_scaled + next_scaled
    simplified = sp.Rational(combined_num, lcd).limit_denominator()

    if not needs_lcd_step:
        combined_text = _format_fraction(combined_num, lcd)
        sum_parts = [combined_text]
        for term in rest:
            _append_signed_term(sum_parts, term)
        steps.append(" ".join(sum_parts))

    return simplified, steps


def _build_mixed_chain_steps(values, ops):
    steps = []
    work_values, work_ops = list(values), list(ops)

    while "*" in work_ops:
        index = work_ops.index("*")
        left = work_values[index]
        right = work_values[index + 1]
        raw_product = _format_unreduced_product(left, right)
        result = sp.Rational(left * right).limit_denominator()
        simplified_product = _format_sympy(result)
        work_values = work_values[:index] + [result] + work_values[index + 2:]
        work_ops = work_ops[:index] + work_ops[index + 1:]
        if raw_product != simplified_product:
            if len(work_values) > 1:
                steps.append(_format_mixed_expr(work_values, work_ops))
            else:
                steps.append(raw_product)
        elif len(work_values) > 1:
            steps.append(_format_mixed_expr(work_values, work_ops))

    signed_terms = _to_signed_terms(work_values, work_ops)
    steps.extend(_build_mixed_addition_steps(signed_terms))
    return steps


def _to_signed_terms(values, ops):
    signed = [values[0]]
    for op, value in zip(ops, values[1:]):
        signed.append(value if op == "+" else -value)
    return signed


def _format_signed_additive_expr(signed_terms):
    parts = [_format_mixed_value(signed_terms[0])]
    for term in signed_terms[1:]:
        if term >= 0:
            parts.append(f"+ {_format_mixed_value(term)}")
        else:
            parts.append(f"- {_format_mixed_value(-term)}")
    return " ".join(parts)


def _format_remaining_expr(accumulator, remaining_terms):
    parts = [_format_mixed_value(accumulator)]
    for term in remaining_terms:
        if term >= 0:
            parts.append(f"+ {_format_mixed_value(term)}")
        else:
            parts.append(f"- {_format_mixed_value(-term)}")
    return " ".join(parts)


def _format_integer_as_fraction(integer, denominator):
    numerator = abs(int(integer)) * int(denominator)
    return _format_fraction(numerator, int(denominator))


def _append_signed_term(parts, term):
    if term >= 0:
        parts.append(f"+ {_format_mixed_value(term)}")
    else:
        parts.append(f"- {_format_mixed_value(-term)}")


def _build_mixed_addition_steps(signed_terms):
    if not any(_is_fraction(term) for term in signed_terms):
        steps = []
        accumulator = signed_terms[0]
        for index, next_term in enumerate(signed_terms[1:], start=1):
            rest = signed_terms[index + 1 :]
            accumulator = accumulator + next_term
            if rest:
                steps.append(_format_remaining_expr(accumulator, rest))
        return steps

    steps = []
    accumulator = signed_terms[0]

    for index, next_term in enumerate(signed_terms[1:], start=1):
        rest = signed_terms[index + 1 :]

        if _is_fraction(next_term):
            accumulator, pair_steps = _add_fraction_pair_steps(
                accumulator, next_term, rest
            )
            steps.extend(pair_steps)
            continue

        acc_rat = sp.Rational(accumulator).limit_denominator()
        if acc_rat.q > 1:
            converted = _format_integer_as_fraction(next_term, acc_rat.q)
            converted_parts = [
                _format_mixed_value(accumulator),
                f"+ {converted}" if next_term >= 0 else f"- {converted}",
            ]
            for term in rest:
                _append_signed_term(converted_parts, term)
            steps.append(" ".join(converted_parts))

        accumulator = sp.Rational(accumulator + next_term).limit_denominator()
        combined_text = _format_mixed_value(accumulator)
        if rest:
            steps.append(_format_remaining_expr(accumulator, rest))
        else:
            steps.append(combined_text)

    return steps


def _generate_mixed_chain():
    n_terms = random.randint(2, 3)
    values = []
    kinds = []
    for _ in range(n_terms):
        value, kind = _rand_mixed_term()
        values.append(value)
        kinds.append(kind)

    if "frac" not in kinds:
        index = random.randrange(n_terms)
        num = _rand_int(exclude_zero=True)
        den = _rand_int(low=1, high=FRACTION_DEN_MAX, exclude_zero=True)
        values[index] = sp.Rational(num, den)
        kinds[index] = "frac"

    ops = [random.choice(["+", "-", "*"]) for _ in range(n_terms - 1)]
    if "int" in kinds and "frac" in kinds and "*" not in ops:
        ops[random.randrange(len(ops))] = "*"

    prob_expr = _format_mixed_expr(values, ops)
    value = _evaluate_mixed_chain(values, ops)
    steps = _build_mixed_chain_steps(values, ops)
    prob = random.choice(ARITHMETIC_KEYWORDS) + prob_expr
    return prob, steps, _format_sympy(value), value


def _pick_linear_template():
    return random.choice(["add", "subtract", "multiply", "divide"])


def _generate_linear_solve():
    var = random.choice(VARS)
    var_name = var.name
    template = _pick_linear_template()

    if template == "add":
        b = _rand_int()
        x_value = _rand_int()
        c = x_value + b
        equation = f"{var_name} + {b}" if b >= 0 else f"{var_name} - {abs(b)}"
        step_rhs = f"{c} - {b}" if b >= 0 else f"{c} + {abs(b)}"
    elif template == "subtract":
        b = _rand_int(exclude_zero=True)
        x_value = _rand_int()
        c = x_value - b
        equation = f"{var_name} - {b}" if b >= 0 else f"{var_name} + {abs(b)}"
        step_rhs = f"{c} + {b}" if b >= 0 else f"{c} - {abs(b)}"
    elif template == "multiply":
        a = _rand_int(exclude_zero=True)
        x_value = _rand_int()
        c = a * x_value
        equation = f"{a}*{var_name}"
        step_rhs = f"{c}/{a}"
    else:
        a = _rand_int(low=1, high=FRACTION_DEN_MAX, exclude_zero=True)
        x_value = _rand_int(exclude_zero=True)
        c = sp.Rational(x_value, a)
        equation = f"{var_name}/{a}"
        step_rhs = f"{c}*{a}"

    equation_text = f"{equation} = {c}"
    step = f"{var_name} = {step_rhs}"
    answer = f"{var_name} = {_format_sympy(x_value)}"

    keyword = random.choice(SOLVE_KEYWORDS).format(var=var_name)
    prob = keyword + equation_text

    steps = []
    if step != answer:
        steps.append(step)

    return prob, steps, answer, equation_text, var


def generate_level_2():
    """
    Level 2: Polynomials, Distribution, Mixed Arithmetic, and Linear Solving.

    Returns (problem, steps, answer).
    """
    generators = [
        _generate_polynomial_expand,
        _generate_polynomial_sum,
        _generate_distributive,
        _generate_mixed_chain,
        _generate_linear_solve,
    ]

    for _ in range(50):
        generator = random.choice(generators)
        result = generator()

        if generator is _generate_linear_solve:
            prob, steps, answer, equation_text, var = result
            if _verify_equation(equation_text, var, steps, answer):
                return prob, steps, answer
            continue

        prob, steps, answer, value = result
        if _verify_expression(value, steps, answer):
            return prob, steps, answer

    return prob, steps, answer


def _format_sample(prob, steps, answer):
    segments = [f"Problem: {prob}"]
    segments += [f"step: {step}" for step in steps]
    segments.append(f"step: {answer}")
    return "<bos>" + "; ".join(segments) + "<eos>"


def build_dataset(num_samples, filename="level_2_data.jsonl"):
    """Generates the dataset and saves it as a JSON Lines file."""
    print(f"Generating {num_samples} Level 2 samples...")
    print(
        f"Range: [{MIN_INT}, {MAX_INT}] | "
        f"Polynomial max degree: {POLY_MAX_DEGREE}"
    )

    with open(filename, "w", encoding="utf-8") as handle:
        for _ in range(num_samples):
            prob, steps, answer = generate_level_2()
            formatted_text = _format_sample(prob, steps, answer)
            json_record = {"text": formatted_text}
            handle.write(json.dumps(json_record) + "\n")

    print(f"Done! Dataset saved to {filename}")


if __name__ == "__main__":
    build_dataset(100000, filename="level_2_data.jsonl")
