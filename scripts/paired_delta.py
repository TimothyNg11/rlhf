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


_VALID_METRICS = ("strict", "lenient")
_VALID_AGGS = ("mean", "any")


def load_per_problem_means(
    root: str | Path, benchmark: str, *, metric: str = "strict", agg: str = "mean"
) -> tuple[list[str], np.ndarray, float]:
    """Read `<root>/<benchmark>/samples.jsonl.gz` and aggregate by problem_id.

    ``metric`` selects the verdict field: ``"strict"`` (default) reads
    ``verdict`` (present in every run); ``"lenient"`` reads ``verdict_lenient``
    -- if any record lacks it (an old run graded before lenient scoring
    existed), raises a clear error naming the run dir and pointing at
    ``scripts/regrade_lenient.py``.

    ``agg`` selects the per-problem statistic: ``"mean"`` (pass@1, the
    existing behavior) or ``"any"`` (pass@k: 1.0 if any of the problem's k
    samples was correct, else 0.0).

    Returns (sorted problem_ids, per-problem aggregate array aligned to those
    ids, parse_rate over all rows).
    """
    if metric not in _VALID_METRICS:
        raise ValueError(f"--metric must be one of {_VALID_METRICS}; got {metric!r}")
    if agg not in _VALID_AGGS:
        raise ValueError(f"--agg must be one of {_VALID_AGGS}; got {agg!r}")

    verdict_key = "verdict" if metric == "strict" else "verdict_lenient"
    samples_path = Path(root) / benchmark / "samples.jsonl.gz"
    verdicts_by_problem: dict[str, list[float]] = defaultdict(list)
    n_rows = 0
    n_parseable = 0

    with gzip.open(samples_path, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if verdict_key not in record:
                raise ValueError(
                    f"--metric lenient requires '{verdict_key}' in every sample record, "
                    f"but it is missing from run dir {root} (benchmark {benchmark!r}) -- "
                    "this looks like an old run graded before lenient scoring existed. "
                    f"Re-grade it first with scripts/regrade_lenient.py --run {root} "
                    "--out <out-dir>, or use --metric strict."
                )
            verdicts_by_problem[record["problem_id"]].append(record[verdict_key])
            n_rows += 1
            if record["parseable"]:
                n_parseable += 1

    problem_ids = sorted(verdicts_by_problem)
    if agg == "mean":
        means = np.array([np.mean(verdicts_by_problem[pid]) for pid in problem_ids])
    else:
        means = np.array(
            [float(any(v > 0 for v in verdicts_by_problem[pid])) for pid in problem_ids]
        )
    parse_rate = n_parseable / n_rows if n_rows else 0.0
    return problem_ids, means, parse_rate


def compute_paired(
    base_root: str | Path,
    candidate_root: str | Path,
    benchmark: str,
    *,
    metric: str = "strict",
    agg: str = "mean",
    intersect: bool = False,
) -> dict:
    """Load both runs' per-problem aggregates and return the paired-delta stats dict.

    By default (``intersect=False``) the base and candidate problem sets must
    be identical (existing behavior: raises otherwise). With ``intersect=True``
    the two runs are paired on the INTERSECTION of their problem_ids instead
    (a numeric sort of the ids is deliberately NOT used as a shortcut --
    lexicographic order misorders ids like ``gsm8k_10`` vs ``gsm8k_2``).
    """
    base_ids, base_means, base_parse_rate = load_per_problem_means(
        base_root, benchmark, metric=metric, agg=agg
    )
    candidate_ids, candidate_means, candidate_parse_rate = load_per_problem_means(
        candidate_root, benchmark, metric=metric, agg=agg
    )

    if intersect:
        common = sorted(set(base_ids) & set(candidate_ids))
        if not common:
            raise ValueError(
                f"base ({base_root}) and candidate ({candidate_root}) share no "
                "problem_ids to intersect on"
            )
        base_pos = {pid: i for i, pid in enumerate(base_ids)}
        candidate_pos = {pid: i for i, pid in enumerate(candidate_ids)}
        base_means = np.array([base_means[base_pos[pid]] for pid in common])
        candidate_means = np.array([candidate_means[candidate_pos[pid]] for pid in common])
        base_ids = common
    elif set(base_ids) != set(candidate_ids):
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
    parser.add_argument(
        "--metric", choices=["strict", "lenient"], default="strict",
        help="verdict field to compare: 'strict' (default, `verdict`) or "
        "'lenient' (`verdict_lenient`; requires the run was graded with lenient "
        "scoring, or re-graded via scripts/regrade_lenient.py)",
    )
    parser.add_argument(
        "--agg", choices=["mean", "any"], default="mean",
        help="per-problem aggregation: 'mean' (default, pass@1) or 'any' (pass@k)",
    )
    parser.add_argument(
        "--intersect", action="store_true",
        help="pair on the intersection of problem_ids between the two runs "
        "instead of requiring identical problem sets",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    stats = compute_paired(
        args.base, args.candidate, args.benchmark,
        metric=args.metric, agg=args.agg, intersect=args.intersect,
    )

    print(f"benchmark:          {args.benchmark}")
    print(f"metric:             {args.metric}")
    print(f"agg:                {args.agg}")
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
