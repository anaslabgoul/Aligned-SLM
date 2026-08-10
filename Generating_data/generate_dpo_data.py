"""
Generate DPO preference data by sampling from a trained (SFT) checkpoint.

Direct Preference Optimization (Rafailov et al., 2023) trains on *pairs* of
responses to the same prompt: a preferred ("chosen") one and a dispreferred
("rejected") one. This project has a rare luxury — every math answer is
*verifiable* — so we can build those pairs with **no human labels**:

    1. For each problem, prime the model with "Problem: ...; step:" and sample
       several chains at temperature > 0 (so they differ).
    2. Check each chain's final answer with SymPy (reusing evaluate_model.py).
    3. Correct chains go in the "chosen" bucket, wrong ones in "rejected".
    4. A problem that produced *both* a correct and a wrong chain yields a
       preference pair. Problems the model always gets right (or always wrong)
       contribute nothing — that is expected.

The reward (correct / incorrect) is used *only* to sort chains into the two
buckets; it never enters the DPO loss itself. That single bit per chain is the
whole supervision signal.

Each output line is one preference pair:

    {"prompt": "Problem: ...; step:",
     "chosen": " ...; step: <right answer>",
     "rejected": " ...; step: <wrong answer>",
     "level": 1, "operation": "integer_chain",
     "reference_answer": "<ground truth>"}

The prompt has no "<bos>" and the completions have no "<eos>"; the trainer adds
both (add_bos on the prompt, add_eos on the completion), exactly matching the
SFT surface form "<bos>Problem: ...; step: ...<eos>".

Examples
--------
Default: sample every operation of all three levels from the default checkpoint:

    python Generating_data/generate_dpo_data.py --checkpoint checkpoints/model_2.pt \
        --output dpo_data.jsonl

Only levels 1 and 2, 8 samples per problem, up to 2 pairs each:

    python Generating_data/generate_dpo_data.py --checkpoint checkpoints/model_2.pt \
        --levels 1 2 --samples-per-problem 8 --max-pairs-per-problem 2 \
        --output dpo_l12.jsonl

Use the canonical reference chain as "chosen" (guarantees a pair whenever the
model produces any wrong chain), rather than an on-policy correct sample:

    python Generating_data/generate_dpo_data.py --checkpoint checkpoints/model_2.pt \
        --chosen-source reference --output dpo_data.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

# This script lives in Generating_data/ but reuses modules from the repo root
# (evaluate_model.py, prompt.py) and its sibling level_3 generator, so PROJECT_ROOT
# is the repo root (one level up) and both directories go on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Generating_data"))

# evaluate_model gives us the level 1/2 operation registry plus the exact
# verification helpers used for scoring, so DPO data is labelled the same way
# accuracy is measured.
import evaluate_model as em  # noqa: E402
import level_3_generating_data as level_3  # noqa: E402
from prompt import format_token_ids, generate_until_eos  # noqa: E402

DEFAULT_MODEL = PROJECT_ROOT / "models" / "model.py"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pt"


# Per-operation generators for every level. Levels 1 and 2 come straight from
# evaluate_model; level 3's two operations are added here. Each entry is
# (generator, kind); "expression" generators return (prob, steps, answer, value)
# and "equation" generators return (prob, steps, answer, equation_text, var).
LEVEL_OPERATIONS = {
    1: em.OPERATIONS[1],
    2: em.OPERATIONS[2],
    3: {
        "mixed_polynomial": (level_3._generate_mixed_polynomial, "expression"),
        "fraction_distribute": (level_3._generate_fraction_distribute, "expression"),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate DPO preference pairs by sampling a trained checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to model module, e.g. models/model.py or models.model",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="SFT checkpoint to sample from (also the DPO reference model later).",
    )
    parser.add_argument(
        "--output",
        default="dpo_data.jsonl",
        help="Output JSONL path for the preference pairs.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        choices=sorted(LEVEL_OPERATIONS),
        default=sorted(LEVEL_OPERATIONS),
        help="Curriculum levels to draw problems from.",
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        default=None,
        help="Restrict to these operation names (must belong to the chosen levels).",
    )
    parser.add_argument(
        "-p",
        "--problems-per-operation",
        type=int,
        default=200,
        help="Distinct problems to attempt PER operation.",
    )
    parser.add_argument(
        "-n",
        "--samples-per-problem",
        type=int,
        default=6,
        help="Chains sampled per problem (needs temperature > 0 for diversity).",
    )
    parser.add_argument(
        "--max-pairs-per-problem",
        type=int,
        default=1,
        help="Cap on (chosen, rejected) pairs emitted per contested problem.",
    )
    parser.add_argument(
        "--chosen-source",
        choices=("model", "reference"),
        default="model",
        help=(
            "'model': chosen is an on-policy correct sample (needs one to exist). "
            "'reference': chosen is the canonical generated chain (always available; "
            "only a wrong sample is needed to form a pair)."
        ),
    )
    parser.add_argument(
        "--dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop duplicate chains within each bucket before pairing.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Must be > 0 to produce varied chains.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top-k sampling cutoff.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate per chain.",
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
        help="Random seed for reproducible problem generation and sampling.",
    )
    return parser.parse_args()


def select_operations(levels, requested):
    """Return an ordered list of (level, name, generator, kind) to sample from."""
    selected = []
    seen = set()
    for level in levels:
        for name, (generator, kind) in LEVEL_OPERATIONS[level].items():
            if requested and name not in requested:
                continue
            selected.append((level, name, generator, kind))
            seen.add(name)

    if requested:
        unknown = [name for name in requested if name not in seen]
        if unknown:
            available = sorted(
                n for lvl in levels for n in LEVEL_OPERATIONS[lvl]
            )
            raise SystemExit(
                f"Unknown operation(s) for levels {levels}: {', '.join(unknown)}.\n"
                f"Available: {', '.join(available)}"
            )
    return selected


def problem_from(generator, kind):
    """Draw one problem and its ground truth.

    Returns (problem, steps, answer, kind, ground_truth) where ground_truth is a
    SymPy value for "expression" problems and the reference answer string for
    "equation" problems.
    """
    result = generator()
    if kind == "equation":
        # linear_solve returns (prob, steps, answer, equation_text, var).
        problem, steps, answer = result[0], result[1], result[2]
        return problem, steps, answer, kind, answer
    problem, steps, answer, value = result
    return problem, steps, answer, kind, value


def is_correct(predicted, kind, ground_truth):
    if kind == "equation":
        return em.is_equation_correct(predicted, ground_truth)
    return em.is_expression_correct(predicted, ground_truth)


def sample_completion(model, prompt, gen_kwargs):
    """Sample one chain and return (completion_text, final_answer).

    completion_text is exactly what the model produced after the prompt, with the
    trailing <eos> removed (the trainer re-adds it). final_answer is the text
    after the last "step:" marker, used for verification.
    """
    prompt_len = model.tokenizer.encode(
        prompt, add_bos=True, add_eos=False
    ).numel()

    token_ids = generate_until_eos(model=model, prompt=prompt, **gen_kwargs)
    completion_ids = token_ids[prompt_len:]
    if completion_ids and completion_ids[-1] == model.tokenizer.eos_token_id:
        completion_ids = completion_ids[:-1]
    completion_text = model.tokenizer.decode(completion_ids)

    generated = format_token_ids(model.tokenizer, token_ids)
    final_answer = em.extract_final_answer(generated)
    return completion_text, final_answer


def reference_completion(steps, answer):
    """Rebuild the canonical chain as the completion that follows '...; step:'.

    Mirrors the SFT surface form: after the prompt's trailing 'step:' the sample
    continues ' <step1>; step: <step2>; step: <answer>' (a leading space, then
    'step: ' between segments). No steps -> just ' <answer>'.
    """
    return " " + "; step: ".join(list(steps) + [answer])


def build_pairs_for_problem(problem, steps, answer, kind, ground_truth, model, args):
    """Sample chains for one problem and return a list of preference-pair dicts."""
    prompt = em.build_prompt(problem)
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }

    correct, wrong = [], []
    for _ in range(args.samples_per_problem):
        completion, final_answer = sample_completion(model, prompt, gen_kwargs)
        if not completion.strip():
            continue
        (correct if is_correct(final_answer, kind, ground_truth) else wrong).append(
            completion
        )

    if args.dedup:
        correct = list(dict.fromkeys(correct))
        wrong = list(dict.fromkeys(wrong))

    if args.chosen_source == "reference":
        chosen_pool = [reference_completion(steps, answer)]
    else:
        chosen_pool = correct

    if not chosen_pool or not wrong:
        return [], len(correct), len(wrong)

    # Pair by zipping shuffled buckets, so a problem does not contribute many
    # near-identical pairs that would dominate the dataset.
    random.shuffle(chosen_pool)
    random.shuffle(wrong)
    prompt_core = f"Problem: {problem}; step:"
    pairs = []
    for chosen, rejected in zip(chosen_pool, wrong):
        pairs.append(
            {
                "prompt": prompt_core,
                "chosen": chosen,
                "rejected": rejected,
                "reference_answer": answer,
            }
        )
        if len(pairs) >= args.max_pairs_per_problem:
            break
    return pairs, len(correct), len(wrong)


def main():
    args = parse_args()
    if args.temperature <= 0:
        raise SystemExit(
            "--temperature must be > 0; DPO pairs need varied chains, and greedy "
            "decoding produces the same chain every time."
        )
    if args.samples_per_problem < 2 and args.chosen_source == "model":
        raise SystemExit(
            "--samples-per-problem must be >= 2 to find both a correct and a wrong "
            "chain (or use --chosen-source reference with >= 1 sample)."
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, device, checkpoint_path = em.load_model(
        args.model, args.checkpoint, args.device
    )
    operations = select_operations(args.levels, args.operations)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Levels: {args.levels} | operations: {', '.join(n for _, n, _, _ in operations)}")
    print(
        f"Sampling {args.samples_per_problem} chains x "
        f"{args.problems_per_operation} problems/operation "
        f"at T={args.temperature} (chosen from: {args.chosen_source})"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    totals = {"pairs": 0, "problems": 0, "contested": 0, "correct": 0, "sampled": 0}
    per_operation = {}

    with output_path.open("w", encoding="utf-8") as handle:
        for level, name, generator, kind in operations:
            op_pairs = 0
            op_contested = 0
            op_correct = 0
            op_sampled = 0
            for _ in range(args.problems_per_operation):
                problem, steps, answer, kind_, gt = problem_from(generator, kind)
                pairs, n_correct, n_wrong = build_pairs_for_problem(
                    problem, steps, answer, kind_, gt, model, args
                )
                op_sampled += n_correct + n_wrong
                op_correct += n_correct
                totals["problems"] += 1
                if pairs:
                    op_contested += 1
                for pair in pairs:
                    pair["level"] = level
                    pair["operation"] = name
                    handle.write(json.dumps(pair) + "\n")
                    op_pairs += 1

            totals["pairs"] += op_pairs
            totals["contested"] += op_contested
            totals["correct"] += op_correct
            totals["sampled"] += op_sampled
            per_operation[name] = {
                "level": level,
                "pairs": op_pairs,
                "contested_problems": op_contested,
                "pass_rate": (op_correct / op_sampled * 100) if op_sampled else 0.0,
            }
            print(
                f"  L{level} {name:<22} pairs: {op_pairs:>6} | "
                f"contested: {op_contested:>5}/{args.problems_per_operation} | "
                f"model pass-rate: {per_operation[name]['pass_rate']:>5.1f}%"
            )

    print("\n=== Summary ===")
    print(f"Problems attempted : {totals['problems']}")
    print(f"Contested problems : {totals['contested']} (produced >= 1 pair)")
    overall_pass = (totals["correct"] / totals["sampled"] * 100) if totals["sampled"] else 0.0
    print(f"Overall pass-rate  : {overall_pass:.1f}%")
    print(f"Preference pairs   : {totals['pairs']}")
    print(f"Saved to           : {output_path}")

    if totals["pairs"] == 0:
        print(
            "\nNo pairs were produced. The model may be at ~0% or ~100% on these "
            "operations. Try a harder level, a higher --temperature, more "
            "--samples-per-problem, or --chosen-source reference."
        )


if __name__ == "__main__":
    main()
