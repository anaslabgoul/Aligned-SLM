"""
Level 3 data generation: Mixed polynomial simplification and fractional distribution.

generate_level_3() randomly produces one of two problem types (forward generation
via SymPy; powers are written with ^, not **):

Each sample is emitted as:

    <bos>Problem: ...; step: ...; step: ...<eos>

1. Mixed polynomial   — a sum/difference of at most MAX_TERMS operands of degree up
                        to POLY_MAX_DEGREE, where each operand is a plain polynomial,
                        a scalar times a polynomial, a product of two linear factors,
                        or a squared binomial. Combines Level 2's polynomial_expand,
                        polynomial_sum and distributive into one problem.
2. Fraction distribute — a sum/difference of first-degree operands whose coefficients
                        may be fractions, at least one of them distributed. Follows
                        Level 1's fraction walkthrough one stage per step: distribute,
                        multiply straight across, reduce, put like terms over a common
                        denominator, combine, reduce.

Both types follow the same chain-of-thought pattern as Levels 1 and 2: open up the
constructs first, then drop parentheses, then combine like terms.

Every generated sample is verified with SymPy before being returned.
"""

import json
import random

import sympy as sp

# --- Configuration -----------------------------------------------------------
MIN_INT = -10
MAX_INT = 10
POLY_MAX_DEGREE = 2
# Upper bound on how many operands are summed/subtracted in types 1 and 2.
MAX_TERMS = 3
# Fractional coefficients (type 2) are built from these ranges. They are kept
# smaller than [MIN_INT, MAX_INT] because distributing multiplies them together.
FRACTION_NUM_MAX = 9
FRACTION_DEN_MAX = 10
# How often the distributed scalar / the coefficients inside the parentheses are
# drawn as fractions rather than integers (type 2 only).
SCALAR_FRACTION_PROB = 0.75
INNER_FRACTION_PROB = 0.25
# How many times a generator retries before accepting a degenerate sample.
MAX_ATTEMPTS = 20
# -----------------------------------------------------------------------------

x, y = sp.symbols("x y")
VARS = (x, y)

SIMPLIFY_KEYWORDS = [
    "Expand and simplify ",
    "Simplify ",
    "Expand ",
    "Multiply out and simplify ",
    "Write in simplest form ",
    "Combine like terms in ",
    "",
]

DISTRIBUTE_KEYWORDS = [
    "Expand ",
    "Use the distributive property on ",
    "Distribute ",
    "Multiply out ",
    "Expand and simplify ",
    "Simplify ",
    "",
]


def _rand_int(low=MIN_INT, high=MAX_INT, exclude_zero=False):
    while True:
        value = random.randint(low, high)
        if not exclude_zero or value != 0:
            return value


def _rand_rational(exclude_zero=False, fraction_prob=0.5):
    """Draw a coefficient that is an integer or a proper fraction."""
    if random.random() >= fraction_prob:
        return sp.Rational(_rand_int(exclude_zero=exclude_zero))

    num = _rand_int(low=-FRACTION_NUM_MAX, high=FRACTION_NUM_MAX, exclude_zero=True)
    den = _rand_int(low=2, high=FRACTION_DEN_MAX)
    return sp.Rational(num, den)


def _format_scalar_factor(scalar):
    """Parenthesize negative scalars so operands never chain into '- -9*(...)'."""
    rendered = _format_rational(scalar)
    if scalar < 0:
        return f"({rendered})"
    return rendered


def _sympify(text):
    return sp.sympify(str(text).replace("^", "**"))


def _format_rational(value):
    """Render a rational as '3', '2/3' or '-2/3' (never as a decimal)."""
    value = sp.Rational(value)
    if value.q == 1:
        return str(int(value.p))
    if value.p < 0:
        return f"-{abs(int(value.p))}/{int(value.q)}"
    return f"{int(value.p)}/{int(value.q)}"


def _is_fraction(value):
    return sp.Rational(value).q != 1


def _format_monomial(coef, degree, var_name):
    """Render a single signed term, e.g. '-2/3*x^2'. Returns None for a zero coef."""
    coef = sp.Rational(coef)
    if coef == 0:
        return None

    magnitude_coef = abs(coef)
    if degree == 0:
        magnitude = _format_rational(magnitude_coef)
    else:
        base = var_name if degree == 1 else f"{var_name}^{degree}"
        if magnitude_coef == 1:
            magnitude = base
        else:
            magnitude = f"{_format_rational(magnitude_coef)}*{base}"

    if coef < 0:
        return f"-{magnitude}"
    return magnitude


def _poly_terms(expr, var, sign=1):
    """[(power, coef)] in descending power with zero coefficients dropped."""
    expr = sp.expand(expr)
    if expr == 0:
        return []

    poly = sp.Poly(expr, var)
    terms = []
    for index, coef in enumerate(poly.all_coeffs()):
        power = poly.degree() - index
        signed_coef = sign * sp.Rational(coef)
        if signed_coef != 0:
            terms.append((power, signed_coef))
    return terms


def _format_terms(terms, var_name):
    """Render a sequence of (power, coef) pairs in the order given, not sorted."""
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


def _format_polynomial(expr, var_name):
    return _format_terms(_poly_terms(expr, sp.Symbol(var_name)), var_name)


def _format_distributive_products(scalar, inner, var_name, sign=1):
    """Render 'k*(a*x + b)' as the unevaluated products 'k*a*x + k*b'.

    `sign` folds an outer minus into the scalar so the products can be written
    without the surrounding parentheses.
    """
    scalar = sign * sp.Rational(scalar)
    parts = []
    for power, coef in _poly_terms(inner, sp.Symbol(var_name)):
        signed_product = scalar * coef
        if power == 0:
            magnitude = f"{_format_rational(abs(scalar))}*{_format_rational(abs(coef))}"
        else:
            monomial = _format_monomial(abs(coef), power, var_name)
            magnitude = f"{_format_rational(abs(scalar))}*{monomial}"

        if not parts:
            parts.append(f"-{magnitude}" if signed_product < 0 else magnitude)
        elif signed_product > 0:
            parts.append(f"+ {magnitude}")
        else:
            parts.append(f"- {magnitude}")
    return " ".join(parts)


def _format_binomial(a, b, var_name):
    """Format a*x + b as a parenthesized linear binomial."""
    parts = []
    if a != 0:
        if a == 1:
            parts.append(var_name)
        elif a == -1:
            parts.append(f"-{var_name}")
        else:
            parts.append(f"{_format_rational(a)}*{var_name}")

    if b != 0:
        if b > 0:
            parts.append(f"+ {_format_rational(b)}" if parts else _format_rational(b))
        else:
            parts.append(
                f"- {_format_rational(abs(b))}" if parts else _format_rational(b)
            )

    if not parts:
        return "(0)"
    return f"({' '.join(parts)})"


def _rand_polynomial(var, max_degree, min_degree=1):
    degree = random.randint(min_degree, max_degree)
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


def _join_operands(texts, ops):
    """Chain operand renderings with the '+'/'-' operators between them."""
    parts = [texts[0]]
    for op, text in zip(ops, texts[1:]):
        parts.append(f"{op} {text}")
    return " ".join(parts)


def _verify_expression(value, steps, answer):
    try:
        target = sp.expand(_sympify(value))
        for expression in list(steps) + [answer]:
            if sp.expand(_sympify(expression) - target) != 0:
                return False
    except (sp.SympifyError, TypeError, ValueError):
        return False
    return True


def _format_polynomial_product(left, right, var_name):
    """Render (a*x + b)*(c*x + d) as its term-by-term products, unevaluated.

    The polynomial counterpart of _format_distributive_products: every term of the
    left factor is written against every term of the right one, so the model sees
    the distribution before any arithmetic happens.
    """
    symbol = sp.Symbol(var_name)
    entries = []
    for left_power, left_coef in _poly_terms(left, symbol):
        for right_power, right_coef in _poly_terms(right, symbol):
            left_text = _format_monomial(abs(left_coef), left_power, var_name)
            right_text = _format_monomial(abs(right_coef), right_power, var_name)
            entries.append(
                (left_coef * right_coef < 0, f"{left_text}*{right_text}")
            )
    return _join_signed(entries)


def _rand_mixed_operand(var, var_name):
    """One operand of the mixed-polynomial problem.

    Returns (problem_text, opened_text, expr) where `opened_text` is the operand
    with its construct written out but not yet evaluated.
    """
    kind = random.choice(["poly", "scaled", "square", "product"])

    if kind == "poly":
        expr = _rand_polynomial(var, POLY_MAX_DEGREE)
        text = f"({_format_polynomial(expr, var_name)})"
        return text, text, expr

    if kind == "scaled":
        inner = _rand_polynomial(var, POLY_MAX_DEGREE)
        scalar = _rand_int(exclude_zero=True)
        inner_text = _format_polynomial(inner, var_name)
        scalar_text = _format_scalar_factor(scalar)
        if random.random() < 0.5:
            text = f"{scalar_text}*({inner_text})"
        else:
            text = f"({inner_text})*{scalar_text}"
        opened = f"({_format_distributive_products(scalar, inner, var_name)})"
        return text, opened, sp.expand(scalar * inner)

    if kind == "product":
        # Two linear factors, so the operand stays within POLY_MAX_DEGREE.
        a = _rand_int(exclude_zero=True)
        b = _rand_int()
        c = _rand_int(exclude_zero=True)
        d = _rand_int()
        left, right = a * var + b, c * var + d
        text = f"{_format_binomial(a, b, var_name)}*{_format_binomial(c, d, var_name)}"
        opened = f"({_format_polynomial_product(left, right, var_name)})"
        return text, opened, sp.expand(left * right)

    a = _rand_int(exclude_zero=True)
    b = _rand_int()
    base_text = _format_binomial(a, b, var_name)
    text = f"{base_text}^2"
    opened = f"{base_text}*{base_text}"
    return text, opened, sp.expand((a * var + b) ** 2)


def _generate_mixed_polynomial():
    var = random.choice(VARS)
    var_name = var.name

    for attempt in range(MAX_ATTEMPTS):
        # On the final attempt nothing is rejected, so a sample is always returned.
        is_last = attempt == MAX_ATTEMPTS - 1
        n_terms = random.randint(2, MAX_TERMS)
        operands = [_rand_mixed_operand(var, var_name) for _ in range(n_terms)]

        # A plain sum of plain polynomials is just Level 2's polynomial_sum, so
        # require at least one construct that has to be opened up first.
        if all(problem == opened for problem, opened, _ in operands):
            index = random.randrange(n_terms)
            while True:
                replacement = _rand_mixed_operand(var, var_name)
                if replacement[0] != replacement[1]:
                    operands[index] = replacement
                    break

        ops = [random.choice(["+", "-"]) for _ in range(n_terms - 1)]
        signs = [1] + [1 if op == "+" else -1 for op in ops]

        expr = sp.expand(
            sum(sign * operand[2] for sign, operand in zip(signs, operands))
        )
        # Reject samples where every variable term cancels — the answer would be a
        # bare constant and the problem far easier than it looks.
        if not is_last and (expr == 0 or sp.Poly(expr, var).degree() < 1):
            continue

        problem_core = _join_operands([operand[0] for operand in operands], ops)
        opened_core = _join_operands([operand[1] for operand in operands], ops)

        expanded_terms = []
        for sign, operand in zip(signs, operands):
            expanded_terms.extend(_poly_terms(operand[2], var, sign=sign))
        expanded = _format_terms(expanded_terms, var_name)

        answer = _format_polynomial(expr, var_name)
        steps = []
        if opened_core != problem_core:
            steps.append(opened_core)
        if expanded != answer:
            steps.append(expanded)

        prob = random.choice(SIMPLIFY_KEYWORDS) + problem_core
        return prob, steps, answer, expr

    return prob, steps, answer, expr


def _rand_fraction_operand(var, var_name, force_scaled=False):
    """One first-degree operand with (possibly) fractional coefficients.

    Returns (problem_text, scalar, inner_expr, expr); `scalar` is None when the
    operand is a bare linear polynomial with nothing to distribute.
    """
    # The fraction usually rides on the scalar (the textbook '1/2*(4*x - 6)' shape).
    # Making the inner coefficients fractional too makes denominators explode once
    # three operands are combined, so they stay integers most of the time.
    a = _rand_rational(exclude_zero=True, fraction_prob=INNER_FRACTION_PROB)
    b = _rand_rational(fraction_prob=INNER_FRACTION_PROB)
    inner = a * var + b

    if not force_scaled and random.random() < 0.3:
        text = _format_binomial(a, b, var_name)
        return text, None, inner, sp.expand(inner)

    scalar = _rand_rational(exclude_zero=True, fraction_prob=SCALAR_FRACTION_PROB)
    inner_text = _format_binomial(a, b, var_name)
    text = f"{_format_scalar_factor(scalar)}*{inner_text}"
    return text, scalar, inner, sp.expand(scalar * inner)


def _monomial_from_text(magnitude_text, degree, var_name):
    """Build a monomial from an already-rendered unsigned coefficient string."""
    if degree == 0:
        return magnitude_text
    base = var_name if degree == 1 else f"{var_name}^{degree}"
    if magnitude_text == "1":
        return base
    return f"{magnitude_text}*{base}"


def _join_signed(entries):
    """Chain (is_negative, magnitude) pairs into 'a - b + c'."""
    parts = []
    for negative, magnitude in entries:
        if not parts:
            parts.append(f"-{magnitude}" if negative else magnitude)
        else:
            parts.append(f"- {magnitude}" if negative else f"+ {magnitude}")
    return " ".join(parts) if parts else "0"


def _raw_product_text(scalar, coef):
    """The product of two rationals multiplied straight across, not reduced."""
    numerator = abs(scalar.p * coef.p)
    denominator = scalar.q * coef.q
    if denominator == 1:
        return str(numerator)
    return f"{numerator}/{denominator}"


def _fraction_contributions(operands, signs, var):
    """Flatten the operands into (power, scalar, coef, product) contributions.

    `scalar` is None for a bare operand that has nothing to distribute; the outer
    '+'/'-' is folded into the scalar (or into the coefficient) so every step can
    be written without parentheses.
    """
    contributions = []
    for sign, (_text, scalar, inner, _expr) in zip(signs, operands):
        for power, coef in _poly_terms(inner, var):
            if scalar is None:
                signed_coef = sign * coef
                contributions.append((power, None, signed_coef, signed_coef))
            else:
                signed_scalar = sign * sp.Rational(scalar)
                contributions.append((power, signed_scalar, coef, signed_scalar * coef))
    return contributions


def _render_factors(contributions, var_name):
    """'k*a*x + k*b' — products written out but not yet evaluated."""
    entries = []
    for power, scalar, coef, product in contributions:
        monomial = _format_monomial(abs(coef), power, var_name)
        if scalar is None:
            magnitude = monomial
        else:
            magnitude = f"{_format_rational(abs(scalar))}*{monomial}"
        entries.append((product < 0, magnitude))
    return _join_signed(entries)


def _render_raw_products(contributions, var_name):
    """The products multiplied straight across but not reduced: '10/6*x'."""
    entries = []
    for power, scalar, coef, product in contributions:
        if scalar is None:
            magnitude = _format_monomial(abs(coef), power, var_name)
        else:
            magnitude = _monomial_from_text(
                _raw_product_text(scalar, coef), power, var_name
            )
        entries.append((product < 0, magnitude))
    return _join_signed(entries)


def _render_values(contributions, var_name):
    """The products reduced to lowest terms, still in problem order."""
    terms = [(power, product) for power, _scalar, _coef, product in contributions]
    return _format_terms(terms, var_name)


def _group_by_power(contributions):
    """power -> list of product coefficients, in first-appearance order."""
    groups = {}
    for power, _scalar, _coef, product in contributions:
        groups.setdefault(power, []).append(product)
    return groups


def _render_common_denominator(contributions, var_name):
    """Put each group of like terms over its common denominator.

    Returns (rendering, lcds); the rendering is None when no group needs one, so
    the step is skipped exactly like Level 1 skips its own LCD step.
    """
    lcds = {}
    for power, products in _group_by_power(contributions).items():
        lcd = 1
        for product in products:
            lcd = int(sp.ilcm(lcd, product.q))
        # A lone term has nothing to combine with, and a group that already shares
        # a denominator needs no rewrite.
        if len(products) > 1 and any(product.q != lcd for product in products):
            lcds[power] = lcd

    if not lcds:
        return None, {}

    entries = []
    for power, _scalar, _coef, product in contributions:
        lcd = lcds.get(power)
        if lcd is None:
            magnitude = _format_monomial(abs(product), power, var_name)
        else:
            scaled = abs(product.p) * (lcd // product.q)
            magnitude = _monomial_from_text(f"{scaled}/{lcd}", power, var_name)
        entries.append((product < 0, magnitude))
    return _join_signed(entries), lcds


def _render_combined(contributions, lcds, var_name):
    """Combine like terms, leaving each coefficient over the common denominator."""
    groups = _group_by_power(contributions)
    entries = []
    for power in sorted(groups, reverse=True):
        total = sum(groups[power])
        if total == 0:
            continue
        lcd = lcds.get(power)
        if lcd is None:
            magnitude = _format_monomial(abs(total), power, var_name)
        else:
            numerator = abs(total.p) * (lcd // total.q)
            magnitude = _monomial_from_text(f"{numerator}/{lcd}", power, var_name)
        entries.append((total < 0, magnitude))
    return _join_signed(entries)


def _build_fraction_steps(contributions, var_name, problem_core, answer):
    """One step per stage, mirroring Level 1's fraction walkthrough.

    distribute -> multiply straight across -> reduce -> common denominator ->
    combine numerators, with the reduced answer appended by the caller. A stage
    that would repeat the previous line or the answer is dropped.
    """
    common_denominator, lcds = _render_common_denominator(contributions, var_name)
    candidates = [
        _render_factors(contributions, var_name),
        _render_raw_products(contributions, var_name),
        _render_values(contributions, var_name),
        common_denominator,
        _render_combined(contributions, lcds, var_name),
    ]

    steps = []
    previous = problem_core
    for candidate in candidates:
        if candidate is None or candidate == previous or candidate == answer:
            continue
        steps.append(candidate)
        previous = candidate
    return steps


def _generate_fraction_distribute():
    var = random.choice(VARS)
    var_name = var.name

    for attempt in range(MAX_ATTEMPTS):
        # On the final attempt nothing is rejected, so a sample is always returned.
        is_last = attempt == MAX_ATTEMPTS - 1
        n_terms = random.randint(2, MAX_TERMS)
        scaled_index = random.randrange(n_terms)
        operands = [
            _rand_fraction_operand(var, var_name, force_scaled=(index == scaled_index))
            for index in range(n_terms)
        ]

        # The point of this operation is fractional coefficients, so make sure at
        # least one shows up somewhere in the problem.
        coefficients = []
        for _, scalar, inner, _expr in operands:
            if scalar is not None:
                coefficients.append(scalar)
            coefficients.extend(coef for _power, coef in _poly_terms(inner, var))
        if not is_last and not any(_is_fraction(coef) for coef in coefficients):
            continue

        ops = [random.choice(["+", "-"]) for _ in range(n_terms - 1)]
        signs = [1] + [1 if op == "+" else -1 for op in ops]

        expr = sp.expand(
            sum(sign * operand[3] for sign, operand in zip(signs, operands))
        )
        if not is_last and (expr == 0 or sp.Poly(expr, var).degree() < 1):
            continue

        problem_core = _join_operands([operand[0] for operand in operands], ops)
        contributions = _fraction_contributions(operands, signs, var)
        answer = _format_polynomial(expr, var_name)

        steps = _build_fraction_steps(contributions, var_name, problem_core, answer)
        prob = random.choice(DISTRIBUTE_KEYWORDS) + problem_core
        return prob, steps, answer, expr

    return prob, steps, answer, expr


def generate_level_3():
    """
    Level 3: Mixed polynomial simplification and fractional distribution.

    Returns (problem, steps, answer).
    """
    generators = [
        _generate_mixed_polynomial,
        _generate_fraction_distribute,
    ]

    for _ in range(50):
        generator = random.choice(generators)
        prob, steps, answer, value = generator()
        if _verify_expression(value, steps, answer):
            return prob, steps, answer

    return prob, steps, answer


def _format_sample(prob, steps, answer):
    segments = [f"Problem: {prob}"]
    segments += [f"step: {step}" for step in steps]
    segments.append(f"step: {answer}")
    return "<bos>" + "; ".join(segments) + "<eos>"


def build_dataset(num_samples, filename="level_3_data.jsonl"):
    """Generates the dataset and saves it as a JSON Lines file."""
    print(f"Generating {num_samples} Level 3 samples...")
    print(
        f"Range: [{MIN_INT}, {MAX_INT}] | "
        f"Polynomial max degree: {POLY_MAX_DEGREE} | "
        f"Max terms: {MAX_TERMS}"
    )

    with open(filename, "w", encoding="utf-8") as handle:
        for _ in range(num_samples):
            prob, steps, answer = generate_level_3()
            formatted_text = _format_sample(prob, steps, answer)
            json_record = {"text": formatted_text}
            handle.write(json.dumps(json_record) + "\n")

    print(f"Done! Dataset saved to {filename}")


if __name__ == "__main__":
    build_dataset(1000000, filename="level_3_data.jsonl")
