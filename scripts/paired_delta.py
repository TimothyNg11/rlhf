#!/usr/bin/env python
"""CLI entry point for the paired, bootstrapped accuracy delta between two eval runs.

Compares a base model's and a fine-tuned candidate's `run_eval.py` output
directories on the SAME benchmark and problem set, reporting the paired
bootstrap CI of the improvement.

Example:
    python scripts/paired_delta.py --base results/base_model/20260101-000000 \\
        --candidate results/candidate_model/20260102-000000 --benchmark gsm8k
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from grpo_math.stats import paired_delta_ci


def load_per_problem_means(root: str | Path, benchmark: str) -> tuple[list[str], np.ndarray, float]:
    """Read `<root>/<benchmark>/samples.jsonl.gz` and aggregate by problem_id.

    Returns (sorted problem_ids, per-problem mean-verdict array aligned to
    those ids, parse_rate over all rows).
    """
    samples_path = Path(root) / benchmark / "samples.jsonl.gz"
    verdicts_by_problem: dict[str, list[float]] = defaultdict(list)
    n_rows = 0
    n_parseable = 0

    with gzip.open(samples_path, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            verdicts_by_problem[record["problem_id"]].append(record["verdict"])
            n_rows += 1
            if record["parseable"]:
                n_parseable += 1

    problem_ids = sorted(verdicts_by_problem)
    means = np.array([np.mean(verdicts_by_problem[pid]) for pid in problem_ids])
    parse_rate = n_parseable / n_rows if n_rows else 0.0
    return problem_ids, means, parse_rate


def compute_paired(base_root: str | Path, candidate_root: str | Path, benchmark: str) -> dict:
    """Load both runs' per-problem means and return the paired-delta stats dict."""
    base_ids, base_means, base_parse_rate = load_per_problem_means(base_root, benchmark)
    candidate_ids, candidate_means, candidate_parse_rate = load_per_problem_means(candidate_root, benchmark)

    if set(base_ids) != set(candidate_ids):
        n_diff = len(set(base_ids) ^ set(candidate_ids))
        raise ValueError(
            f"base and candidate problem sets differ ({n_diff} problem_ids not shared): "
            f"base has {len(base_ids)} problems, candidate has {len(candidate_ids)} problems"
        )

    delta, ci_lo, ci_hi = paired_delta_ci(candidate_means, base_means)

    return {
        "n_problems": len(base_ids),
        "base_pass_at_1": float(np.mean(base_means)),
        "candidate_pass_at_1": float(np.mean(candidate_means)),
        "delta": delta,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "base_parse_rate": base_parse_rate,
        "candidate_parse_rate": candidate_parse_rate,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap CI of the accuracy delta between two eval runs."
    )
    parser.add_argument("--base", required=True, help="base model's run_eval out-dir root")
    parser.add_argument("--candidate", required=True, help="candidate model's run_eval out-dir root")
    parser.add_argument("--benchmark", required=True, help="benchmark name, e.g. gsm8k")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    stats = compute_paired(args.base, args.candidate, args.benchmark)

    print(f"benchmark:          {args.benchmark}")
    print(f"n_problems:         {stats['n_problems']}")
    print(f"base pass@1:        {stats['base_pass_at_1']:.4f}")
    print(f"candidate pass@1:   {stats['candidate_pass_at_1']:.4f}")
    print(
        f"delta (pp):         {stats['delta'] * 100:+.2f} "
        f"[{stats['ci_lo'] * 100:+.2f}, {stats['ci_hi'] * 100:+.2f}]"
    )
    print(f"base parse_rate:    {stats['base_parse_rate']:.4f}")
    print(f"candidate parse_rate: {stats['candidate_parse_rate']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
