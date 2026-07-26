"""Load and mix one or more JSON/JSONL datasets with configurable proportions."""

from __future__ import annotations

import json
import random
from pathlib import Path


def extract_text(record, line_number: int) -> str:
    if isinstance(record, str):
        return record
    if isinstance(record, dict) and "text" in record:
        return record["text"]
    raise ValueError(
        f"Record at line {line_number} must contain a 'text' field or be a string."
    )


def load_dataset_records(data_path: Path) -> list[str]:
    suffix = data_path.suffix.lower()
    records = []

    if suffix == ".jsonl":
        with data_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                records.append(extract_text(record, line_number))
    elif suffix == ".json":
        with data_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            for index, record in enumerate(payload, start=1):
                records.append(extract_text(record, index))
        elif isinstance(payload, dict) and "text" in payload:
            records.append(extract_text(payload, 1))
        else:
            raise ValueError(
                "JSON file must be a list of records or a single object with a 'text' field."
            )
    else:
        raise ValueError("Data file must be .json or .jsonl")

    if not records:
        raise ValueError(f"No training samples found in {data_path}")

    return records


def normalize_weights(weights: list[float]) -> list[float]:
    if any(weight <= 0 for weight in weights):
        raise ValueError("All --weight values must be > 0.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Sum of --weight values must be > 0.")
    return [weight / total for weight in weights]


def resolve_source_weights(
    data_args: list[str], weight_args: list[float] | None
) -> list[tuple[str, float]]:
    """Pair each --data path with a normalized weight.

    - One dataset and no weights: weight 1.0 (load all unless --mix-total is set).
    - Multiple datasets and no weights: equal weights.
    - Otherwise: one weight per dataset, then normalize.
    """
    if not data_args:
        raise ValueError("At least one --data path is required.")

    if not weight_args:
        weights = [1.0] * len(data_args)
    elif len(weight_args) != len(data_args):
        raise ValueError(
            f"Got {len(data_args)} --data path(s) but {len(weight_args)} --weight "
            "value(s). Provide one --weight per --data, or omit --weight."
        )
    else:
        weights = list(weight_args)

    normalized = normalize_weights(weights)
    return list(zip(data_args, normalized))


def _allocate_counts(
    sizes: list[int], weights: list[float], mix_total: int | None, replace: bool
) -> list[int]:
    """Decide how many samples to draw from each source."""
    if mix_total is not None:
        if mix_total < 1:
            raise ValueError("--mix-total must be >= 1.")
        raw = [weight * mix_total for weight in weights]
        counts = [int(value) for value in raw]
        # Distribute leftover samples to the largest fractional remainders.
        leftover = mix_total - sum(counts)
        order = sorted(
            range(len(raw)),
            key=lambda index: (raw[index] - counts[index], -index),
            reverse=True,
        )
        for index in order[:leftover]:
            counts[index] += 1
    else:
        # Largest total that fits without exceeding any source (no replacement).
        feasible_total = int(
            min(size / weight for size, weight in zip(sizes, weights))
        )
        if feasible_total < 1:
            raise ValueError(
                "Cannot build a mix with the given weights: at least one source "
                "is too small. Lower its weight, add more data, or set --mix-total "
                "with --mix-replace."
            )
        raw = [weight * feasible_total for weight in weights]
        counts = [int(value) for value in raw]
        leftover = feasible_total - sum(counts)
        order = sorted(
            range(len(raw)),
            key=lambda index: (raw[index] - counts[index], -index),
            reverse=True,
        )
        for index in order[:leftover]:
            counts[index] += 1

    if not replace:
        for index, (count, size) in enumerate(zip(counts, sizes)):
            if count > size:
                raise ValueError(
                    f"Source {index + 1} needs {count} samples but only has {size}. "
                    "Lower --mix-total / its --weight, or pass --mix-replace."
                )

    if sum(counts) < 1:
        raise ValueError("Mix produced zero samples.")

    return counts


def sample_texts(
    records: list[str], count: int, rng: random.Random, replace: bool
) -> list[str]:
    if count <= 0:
        return []
    if replace:
        return [rng.choice(records) for _ in range(count)]
    if count > len(records):
        raise ValueError(
            f"Cannot sample {count} texts without replacement from {len(records)}."
        )
    return rng.sample(records, count)


def load_mixed_texts(
    sources: list[tuple[Path, float]],
    *,
    mix_total: int | None = None,
    seed: int = 42,
    replace: bool = False,
) -> tuple[list[str], list[dict]]:
    """Load datasets and subsample them according to normalized weights.

    Returns (mixed_texts, mix_report) where mix_report describes each source.
    """
    if not sources:
        raise ValueError("At least one data source is required.")

    loaded: list[list[str]] = []
    sizes: list[int] = []
    weights: list[float] = []
    paths: list[Path] = []

    for path, weight in sources:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        records = load_dataset_records(path)
        loaded.append(records)
        sizes.append(len(records))
        weights.append(weight)
        paths.append(path)

    # A single source taken whole is not a mix: return it in file order and
    # touch no randomness, so these runs stay bit-identical to a plain
    # single-dataset load. Reordering here would change which samples the
    # seeded train/test split puts on each side.
    single_full_source = len(sources) == 1 and mix_total is None
    if single_full_source:
        return list(loaded[0]), [
            {
                "path": str(paths[0]),
                "available": sizes[0],
                "weight": weights[0],
                "sampled": sizes[0],
            }
        ]

    counts = _allocate_counts(sizes, weights, mix_total, replace=replace)

    rng = random.Random(seed)
    mixed: list[str] = []
    report: list[dict] = []

    for path, weight, records, count in zip(paths, weights, loaded, counts):
        chosen = sample_texts(records, count, rng, replace=replace)
        mixed.extend(chosen)
        report.append(
            {
                "path": str(path),
                "available": len(records),
                "weight": weight,
                "sampled": len(chosen),
            }
        )

    rng.shuffle(mixed)
    return mixed, report


def format_mix_report(report: list[dict]) -> str:
    total = sum(item["sampled"] for item in report)
    lines = ["Dataset mix:"]
    for item in report:
        share = (item["sampled"] / total) if total else 0.0
        lines.append(
            f"  {item['path']}: weight={item['weight']:.4f} | "
            f"sampled={item['sampled']}/{item['available']} "
            f"({share:.1%} of mix)"
        )
    lines.append(f"  total mixed samples: {total}")
    return "\n".join(lines)
