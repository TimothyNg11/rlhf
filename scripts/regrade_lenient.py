#!/usr/bin/env python
"""CLI entry point for an offline, CPU-only strict-vs-lenient re-grade of an
existing run_eval.py output directory.

Reads `<run>/<benchmark>/samples.jsonl.gz` (written by scripts/run_eval.py --
see grpo_math.eval.runner.run_eval for the record schema) and re-grades every
completion twice: strict (`extraction_mode="boxed"`, matching the original
grading exactly) and lenient (`extraction_mode="lenient"`). This measures how
much of any accuracy gain is real math vs. `\\boxed{}` format compliance,
without re-running the model.

A built-in sanity gate recomputes strict pass@1 and checks it against the
stored `summary.json`'s `pass_at_1` for that benchmark -- if they don't match
(tolerance 1e-9), something is wrong with this re-grader or its assumptions
about the record schema, so it refuses to write output for that benchmark and
exits non-zero rather than let anything ride on unvalidated numbers.

Example:
    python scripts/regrade_lenient.py --run results/sweep/base \\
        --out results/regrade/base --benchmark gsm8k
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from grpo_math.rewards import compute_reward
from grpo_math.rewards.extraction import extract_lenient
from grpo_math.stats import bootstrap_ci, pass_at_1, pass_at_k_any, per_problem_means

_SANITY_TOL = 1e-9


def discover_benchmarks(run_root: str | Path) -> list[str]:
    """Return the sorted names of benchmark subdirs of ``run_root`` that
    contain a ``samples.jsonl.gz`` file."""
    return sorted(p.parent.name for p in Path(run_root).glob("*/samples.jsonl.gz"))


def load_records(samples_path: str | Path) -> list[dict]:
    """Read a `run_eval.py`-written `samples.jsonl.gz` file into a list of dicts."""
    with gzip.open(samples_path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def regrade_record(record: dict) -> dict:
    """Re-grade one `samples.jsonl.gz` record strict + lenient, `format_bonus=0.0`.

    Returns the ORIGINAL record with four fields appended: `verdict_strict`,
    `verdict_lenient`, `extracted_lenient`, `extraction_method` (the method
    used by the lenient extraction chain, regardless of extraction_mode).
    """
    completion = record["completion"]
    gold = record["gold"]
    truncated = record["finish_reason"] == "length"

    strict = compute_reward(
        completion, gold, truncated=truncated, format_bonus=0.0, extraction_mode="boxed"
    )
    lenient = compute_reward(
        completion, gold, truncated=truncated, format_bonus=0.0, extraction_mode="lenient"
    )
    lenient_extraction = extract_lenient(completion)

    return {
        **record,
        "verdict_strict": strict.reward,
        "verdict_lenient": lenient.reward,
        "extracted_lenient": lenient_extraction.value,
        "extraction_method": lenient_extraction.method,
    }


def build_verdict_matrix(records: list[dict], verdict_key: str) -> tuple[list[str], np.ndarray]:
    """Group ``records`` by ``problem_id`` and stack ``verdict_key`` into a
    [n_problems, k] array, indexed by ``sample_idx`` (0..k-1) per problem."""
    by_problem: dict[str, dict[int, float]] = defaultdict(dict)
    for r in records:
        by_problem[r["problem_id"]][r["sample_idx"]] = r[verdict_key]

    problem_ids = sorted(by_problem)
    k = max(len(v) for v in by_problem.values())
    matrix = np.zeros((len(problem_ids), k))
    for i, pid in enumerate(problem_ids):
        for sample_idx, verdict in by_problem[pid].items():
            matrix[i, sample_idx] = verdict
    return problem_ids, matrix


def regrade_benchmark(run_root: str | Path, out_root: str | Path, benchmark: str) -> bool:
    """Re-grade one benchmark. Returns True on success (output written), False
    if the sanity gate failed (nothing written, caller should treat as an
    error)."""
    bench_dir = Path(run_root) / benchmark
    records = load_records(bench_dir / "samples.jsonl.gz")
    stored_summary = json.loads((bench_dir / "summary.json").read_text(encoding="utf-8"))

    regraded = [regrade_record(r) for r in records]

    _, strict_matrix = build_verdict_matrix(regraded, "verdict_strict")
    problem_ids, lenient_matrix = build_verdict_matrix(regraded, "verdict_lenient")

    recomputed_strict_pass_at_1 = pass_at_1(strict_matrix)
    stored_pass_at_1 = stored_summary["pass_at_1"]

    if abs(recomputed_strict_pass_at_1 - stored_pass_at_1) > _SANITY_TOL:
        print(
            f"SANITY GATE FAILED for benchmark {benchmark!r}: recomputed strict pass@1 "
            f"{recomputed_strict_pass_at_1} != stored summary.json pass_at_1 {stored_pass_at_1}",
            file=sys.stderr,
        )
        return False

    seed = stored_summary.get("seed", 0)
    strict_per_problem = per_problem_means(strict_matrix)
    lenient_per_problem = per_problem_means(lenient_matrix)
    strict_ci_lo, strict_ci_hi = bootstrap_ci(strict_per_problem, seed=seed)
    lenient_ci_lo, lenient_ci_hi = bootstrap_ci(lenient_per_problem, seed=seed)

    method_histogram = dict(Counter(r["extraction_method"] for r in regraded))

    summary = {
        "benchmark": benchmark,
        "n_problems": len(problem_ids),
        "k": strict_matrix.shape[1],
        "strict": {
            "pass_at_1": recomputed_strict_pass_at_1,
            "ci_lo": strict_ci_lo,
            "ci_hi": strict_ci_hi,
            "pass_at_k_any": pass_at_k_any(strict_matrix),
        },
        "lenient": {
            "pass_at_1": pass_at_1(lenient_matrix),
            "ci_lo": lenient_ci_lo,
            "ci_hi": lenient_ci_hi,
            "pass_at_k_any": pass_at_k_any(lenient_matrix),
        },
        "extraction_method_histogram": method_histogram,
    }

    out_bench_dir = Path(out_root) / benchmark
    out_bench_dir.mkdir(parents=True, exist_ok=True)

    with gzip.open(out_bench_dir / "samples_regrade.jsonl.gz", "wt", encoding="utf-8") as f:
        for record in regraded:
            f.write(json.dumps(record) + "\n")

    (out_bench_dir / "summary_regrade.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"{benchmark}: strict pass@1={summary['strict']['pass_at_1']:.4f}  "
        f"lenient pass@1={summary['lenient']['pass_at_1']:.4f}  "
        f"lenient pass@k={summary['lenient']['pass_at_k_any']:.4f}  "
        f"methods={method_histogram}"
    )
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline strict-vs-lenient re-grade of an existing run_eval.py output directory."
    )
    parser.add_argument("--run", required=True, help="run_eval.py output root, e.g. results/sweep/base")
    parser.add_argument("--out", required=True, help="output root for regraded samples/summaries")
    parser.add_argument(
        "--benchmark", default=None, help="only re-grade this benchmark subdir (default: all found under --run)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    benchmarks = [args.benchmark] if args.benchmark else discover_benchmarks(args.run)

    ok = True
    for benchmark in benchmarks:
        if not regrade_benchmark(args.run, args.out, benchmark):
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
