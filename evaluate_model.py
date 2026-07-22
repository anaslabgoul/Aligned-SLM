"""
Per-operation accuracy evaluation for the character-level math model.

For a chosen curriculum level, this script generates fresh problems *separately
for each operation type* (e.g. Level 1 has five: integer arithmetic, integer
chains, fraction arithmetic, fraction simplification, linear simplification),
feeds each problem to a trained checkpoint, reads the model's final answer (the
text after the LAST "step:"), and compares it to the ground-truth answer with
SymPy (so 7 + 2*x and 2*x + 7 count as equal).

The output is the percentage of correct answers *per operation*, so you can see
exactly which operations the model gets wrong.

Examples
--------
Evaluate the default checkpoint on Level 1, 50 problems per operation:

    python evaluate_model.py --level 1 --num-samples 50

Evaluate a specific checkpoint on Level 2 and show 5 example failures each:

    python evaluate_model.py --level 2 --num-samples 100 \
        --checkpoint checkpoints/best_model.pt --show-errors 5

Only evaluate two operations and save a JSON report:

    python evaluate_model.py --level 1 --num-samples 200 \
        --operations integer_chain fraction_simplify --save-report report.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import sympy as sp
import torch

PROJECT_ROOT = Path(__file__).resolve().parent

# The data generators live in Generating_data/ and import cleanly on their own,
# so add that directory to the path and import them as top-level modules.
sys.path.insert(0, str(PROJECT_ROOT / "Generating_data"))

import level_1_generating_data as level_1  # noqa: E402
import level_2_generating_data as level_2  # noqa: E402

# Reuse the model-loading and generation helpers from prompt.py so evaluation
# decodes exactly the way single-prompt inference does.
from prompt import (  # noqa: E402
    build_model,
    format_token_ids,
    generate_until_eos,
    load_model_module,
    resolve_path,
    select_device,
)

import wandb_logging  # noqa: E402

DEFAULT_MODEL = PROJECT_ROOT / "models" / "model.py"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pt"


# Operation registry per level. Each entry maps a human-readable operation name
# to (generator, kind). "expression" generators return (prob, steps, answer,
# value) where `value` is the SymPy ground truth; "equation" generators (Level 2
# linear solving) return a longer tuple and an answer of the form "x = 5".
OPERATIONS = {
    1: {
        "integer_arithmetic": (level_1._generate_integer_arithmetic, "expression"),
        "integer_chain": (level_1._generate_integer_chain, "expression"),
        "fraction_arithmetic": (level_1._generate_fraction_arithmetic, "expression"),
        "fraction_simplify": (level_1._generate_fraction_simplify, "expression"),
        "linear_simplify": (level_1._generate_linear_simplify, "expression"),
    },
    2: {
        "polynomial_expand": (level_2._generate_polynomial_expand, "expression"),
        "polynomial_sum": (level_2._generate_polynomial_sum, "expression"),
        "distributive": (level_2._generate_distributive, "expression"),
        "mixed_chain": (level_2._generate_mixed_chain, "expression"),
        "linear_solve": (level_2._generate_linear_solve, "equation"),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint per-operation on generated math data."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to model module, e.g. models/model.py or models.model",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to checkpoint file (.pt).",
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=sorted(OPERATIONS),
        default=1,
        help="Curriculum level to evaluate.",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=50,
        help="Number of problems to generate and evaluate PER operation.",
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        default=None,
        help="Subset of operation names to evaluate (default: all in the level).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate per problem.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy/deterministic; recommended for evaluation).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top-k sampling cutoff (only relevant when temperature > 0).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible problem generation.",
    )
    parser.add_argument(
        "--show-errors",
        type=int,
        default=0,
        help="Print up to this many example wrong answers per operation.",
    )
    parser.add_argument(
        "--save-report",
        default=None,
        help="Optional path to write a JSON report of the results.",
    )
    wandb_logging.add_wandb_args(parser)
    parser.add_argument(
        "--wandb-run-id",
        default=None,
        help=(
            "Attach these results to an existing W&B run (the run id of the "
            "training run that produced this checkpoint) instead of creating a new one."
        ),
    )
    return parser.parse_args()


def build_prompt(problem: str) -> str:
    """Prime the model with the problem, up to the first 'step:' marker.

    Matches the training surface form '<bos>Problem: ...; step: ...' so the model
    continues by producing its reasoning steps and final answer.
    """
    return f"<bos>Problem: {problem}; step:"


# Markers that introduce the model's final answer. The current data format ends
# every sample with 'step: <answer>', so 'step:' is primary; 'answer:' is also
# recognized so legacy checkpoints trained on the older 'Answer:' format still
# evaluate correctly. The LAST occurrence of any marker wins.
ANSWER_MARKERS = ("step:", "answer:")


def extract_final_answer(generated_text: str) -> str:
    """Return the text after the last answer marker — the model's final answer."""
    index = max(generated_text.rfind(marker) for marker in ANSWER_MARKERS)
    if index == -1:
        return ""
    tail = generated_text[index:].split(":", 1)[1].strip()
    # Answers never contain ';'; guard against any trailing generation.
    if ";" in tail:
        tail = tail.split(";", 1)[0].strip()
    return tail


def _sympify(text: str):
    """Parse project-format math text into SymPy, converting '^' powers to '**'."""
    return sp.sympify(text.replace("^", "**"))


def is_expression_correct(predicted: str, value) -> bool:
    """True if `predicted` is mathematically equal to the SymPy ground truth."""
    if not predicted:
        return False
    try:
        return sp.expand(_sympify(predicted) - value) == 0
    except Exception:
        # Model output can be arbitrary/unparseable; any failure = wrong answer.
        return False


def is_equation_correct(predicted: str, reference_answer: str) -> bool:
    """True if `predicted` solves the equation the same as the reference 'x = ...'.

    Compares the right-hand side of the model's answer against the reference
    right-hand side (both of the form 'var = value').
    """
    reference_rhs = reference_answer.split("=", 1)[-1]
    predicted_rhs = predicted.split("=", 1)[-1] if "=" in predicted else predicted
    if not predicted_rhs.strip():
        return False
    try:
        return sp.simplify(_sympify(predicted_rhs) - _sympify(reference_rhs)) == 0
    except Exception:
        return False


def evaluate_operation(name, generator, kind, model, num_samples, gen_kwargs, show_errors):
    """Run one operation's problems through the model and tally correctness."""
    correct = 0
    errors: list[dict] = []

    for _ in range(num_samples):
        result = generator()
        if kind == "equation":
            # linear_solve returns (prob, steps, answer, equation_text, var).
            problem, _steps, answer = result[0], result[1], result[2]
            value = None
        else:
            problem, _steps, answer, value = result

        with torch.no_grad():
            token_ids = generate_until_eos(
                model=model, prompt=build_prompt(problem), **gen_kwargs
            )
        generated = format_token_ids(model.tokenizer, token_ids)
        predicted = extract_final_answer(generated)

        if kind == "equation":
            ok = is_equation_correct(predicted, answer)
        else:
            ok = is_expression_correct(predicted, value)

        if ok:
            correct += 1
        elif len(errors) < show_errors:
            errors.append(
                {"problem": problem, "expected": answer, "predicted": predicted}
            )

    return correct, errors


def load_model(model_arg: str, checkpoint_arg: str, device_arg: str):
    """Load the model architecture and weights, matching prompt.py's loading."""
    checkpoint_path = resolve_path(checkpoint_arg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(load_model_module(model_arg))
    device = select_device(device_arg)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"'model_state_dict' key not found in checkpoint: {checkpoint_path}")

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise SystemExit(
            f"Could not load '{checkpoint_path}' into the architecture defined by "
            f"'{model_arg}'.\nThe checkpoint was trained with different hyperparameters "
            f"(d_model / n_heads / n_layers) than models/model.py currently declares.\n"
            f"Fix: evaluate a checkpoint trained with the current model.py, or restore "
            f"model.py to the architecture this checkpoint was trained with.\n\n"
            f"Original error:\n{exc}"
        ) from exc

    model.to(device)
    model.eval()
    return model, device, checkpoint_path


def select_operations(level: int, requested: list[str] | None):
    """Return the ordered {name: (generator, kind)} mapping to evaluate."""
    available = OPERATIONS[level]
    if not requested:
        return available

    selected = {}
    for name in requested:
        if name not in available:
            raise SystemExit(
                f"Unknown operation '{name}' for level {level}. "
                f"Available: {', '.join(available)}"
            )
        selected[name] = available[name]
    return selected


def print_report(level, num_samples, decoding, results):
    """Print a per-operation accuracy table plus the overall total."""
    print()
    print(f"=== Evaluation: level {level} | {num_samples} samples/operation | {decoding} ===")
    print(f"{'operation':<22}{'correct':>9}{'total':>8}{'accuracy':>11}")
    print("-" * 50)

    total_correct = 0
    total_count = 0
    for name, data in results.items():
        correct = data["correct"]
        total = data["total"]
        total_correct += correct
        total_count += total
        accuracy = (correct / total * 100) if total else 0.0
        print(f"{name:<22}{correct:>9}{total:>8}{accuracy:>10.1f}%")

    print("-" * 50)
    overall = (total_correct / total_count * 100) if total_count else 0.0
    print(f"{'overall':<22}{total_correct:>9}{total_count:>8}{overall:>10.1f}%")

    # Example failures, if requested.
    for name, data in results.items():
        if data["errors"]:
            print(f"\n--- Example errors: {name} ---")
            for err in data["errors"]:
                print(f"  Problem  : {err['problem']}")
                print(f"  Expected : {err['expected']}")
                print(f"  Predicted: {err['predicted']}")
                print()


def log_results_to_wandb(run, level, results):
    """Send per-operation accuracy to W&B as metrics, a table, and summary values."""
    if run is None:
        return

    metrics = {}
    total_correct = 0
    total_count = 0
    rows = []
    for name, data in results.items():
        metrics[f"eval/{name}/accuracy"] = data["accuracy"]
        total_correct += data["correct"]
        total_count += data["total"]
        rows.append([name, data["correct"], data["total"], data["accuracy"]])

    overall = (total_correct / total_count * 100) if total_count else 0.0
    metrics["eval/overall/accuracy"] = overall

    wandb_logging.log(run, metrics)
    wandb_logging.log_table(
        run,
        f"eval/level_{level}/per_operation",
        ["operation", "correct", "total", "accuracy"],
        rows,
    )

    # A bar chart is easier to read than five separate scalar panels.
    import wandb

    wandb_logging.log(
        run,
        {
            f"eval/level_{level}/accuracy_chart": wandb.plot.bar(
                wandb.Table(
                    columns=["operation", "accuracy"],
                    data=[[row[0], row[3]] for row in rows],
                ),
                "operation",
                "accuracy",
                title=f"Level {level} accuracy per operation",
            )
        },
    )

    summary = {f"eval_{name}_accuracy": data["accuracy"] for name, data in results.items()}
    summary["eval_overall_accuracy"] = overall
    wandb_logging.summary(run, summary)

    # Example failures, when --show-errors collected any.
    error_rows = [
        [name, err["problem"], err["expected"], err["predicted"]]
        for name, data in results.items()
        for err in data["errors"]
    ]
    if error_rows:
        wandb_logging.log_table(
            run,
            f"eval/level_{level}/errors",
            ["operation", "problem", "expected", "predicted"],
            error_rows,
        )


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be at least 1.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, device, checkpoint_path = load_model(args.model, args.checkpoint, args.device)
    operations = select_operations(args.level, args.operations)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    decoding = "greedy" if args.temperature == 0 else f"T={args.temperature}"

    run = wandb_logging.init_run(
        args,
        config={
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "decoding": decoding,
            "evaluated_operations": list(operations),
        },
        job_type="eval",
        run_id=args.wandb_run_id,
        # Attaching to a training run means eval metrics land on that run rather
        # than a separate one, so accuracy sits next to the loss curves.
        resume="allow" if args.wandb_run_id else None,
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Level {args.level} | operations: {', '.join(operations)}")
    if run is not None:
        print(f"Tracking evaluation on W&B: {wandb_logging.run_location(run)}")

    try:
        results = {}
        for name, (generator, kind) in operations.items():
            print(f"Evaluating {name} ({args.num_samples} samples)...")
            correct, errors = evaluate_operation(
                name, generator, kind, model, args.num_samples, gen_kwargs, args.show_errors
            )
            results[name] = {
                "correct": correct,
                "total": args.num_samples,
                "accuracy": (correct / args.num_samples * 100) if args.num_samples else 0.0,
                "errors": errors,
            }

        print_report(args.level, args.num_samples, decoding, results)
        log_results_to_wandb(run, args.level, results)
    finally:
        wandb_logging.finish(run)

    if args.save_report:
        report = {
            "checkpoint": str(checkpoint_path),
            "level": args.level,
            "num_samples": args.num_samples,
            "decoding": decoding,
            "seed": args.seed,
            "operations": results,
        }
        report_path = Path(args.save_report)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
