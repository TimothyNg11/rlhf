#!/usr/bin/env python
"""CLI entry point for building a per-problem difficulty map over ALL GSM8K
train (+ dev-holdout) problems: k samples/problem, graded strict AND lenient
(format_bonus=0, truncated -> 0, same semantics as the eval harness), used to
filter G2's training data to the band the base model sometimes-but-not-always
solves (see configs/g2_main.yaml's `data.difficulty_map`/`data.difficulty_band`).

Dev-holdout rows (the same tail-slice grpo_math.data.gsm8k.load_gsm8k_train
holds out for the in-loop eval) ARE sampled here too -- a free contamination
measurement (dev-vs-train mean solve rate) -- but the trainer never trains on
dev ids regardless of the difficulty band.

Fake-backend smoke test (no GPU, no network):
    python scripts/build_difficulty_map.py --model dummy --k 2 --temperature 0.9 \\
        --top-p 1.0 --max-tokens 64 --seed 0 --out results/difficulty/smoke \\
        --backend fake --fake-script script.json --limit 4

Real run on the GPU box:
    python scripts/build_difficulty_map.py --model Qwen/Qwen2.5-0.5B-Instruct \\
        --k 8 --temperature 0.9 --top-p 1.0 --max-tokens 3072 --seed 0 \\
        --out results/difficulty/qwen2.5-0.5b_k8
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from grpo_math.data.gsm8k import load_gsm8k_train
from grpo_math.eval.backends import FakeBackend, GenerationBackend, VLLMBackend
from grpo_math.eval.benchmarks import EvalProblem
from grpo_math.eval.runner import grade_dual
from grpo_math.rewards import extract_boxed

_METHODS = ("boxed", "hash", "answer_is", "last_number", "none")
# [1/8, 7/8], [0, 7/8], [1/8, 1] -- the candidate bands under consideration for
# configs/g2_main.yaml's data.difficulty_band (k=8 assumed).
_CANDIDATE_BANDS = [
    ("[1/8, 7/8]", (0.125, 0.875)),
    ("[0, 7/8]", (0.0, 0.875)),
    ("[1/8, 1]", (0.125, 1.0)),
]


def load_all_problems(*, dev_holdout: int = 500) -> tuple[list[EvalProblem], list[str]]:
    """Load ALL GSM8K train problems (train + dev-holdout), tagging each with
    its split. Returns (problems, splits) with ``splits[i]`` = ``"train"`` or
    ``"dev"`` for ``problems[i]``."""
    train, dev = load_gsm8k_train(dev_holdout=dev_holdout)
    return train + dev, ["train"] * len(train) + ["dev"] * len(dev)


def build_difficulty_map(
    problems: list[EvalProblem],
    splits: list[str],
    backend: GenerationBackend,
    *,
    k: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    out_dir: str | Path,
    model: str = "unknown",
) -> dict:
    """Generate ``k`` samples/problem via ``backend``, grade each strict AND
    lenient, and write ``out_dir/{map.jsonl,samples.jsonl.gz,summary.json,summary.md}``.
    Returns the summary dict."""
    assert len(problems) == len(splits), "problems and splits must be aligned 1:1"

    prompts = [p.prompt for p in problems]
    completions = backend.generate(
        prompts, k=k, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    map_rows = []
    sample_records = []
    n_truncated_total = 0
    total_samples = 0

    for problem, split, sample_completions in zip(problems, splits, completions):
        n_correct_strict = 0
        n_correct_lenient = 0
        n_truncated = 0
        method_counts = {m: 0 for m in _METHODS}

        for si, comp in enumerate(sample_completions):
            truncated = comp.finish_reason == "length"
            strict_result, lenient_result, lenient_extraction = grade_dual(
                comp.text, problem.gold, truncated
            )
            extracted = extract_boxed(comp.text)

            if strict_result.reward == 1.0:
                n_correct_strict += 1
            if lenient_result.reward == 1.0:
                n_correct_lenient += 1
            if truncated:
                n_truncated += 1
            method_counts[lenient_extraction.method] += 1
            total_samples += 1

            sample_records.append(
                {
                    "problem_id": problem.problem_id,
                    "sample_idx": si,
                    "completion": comp.text,
                    "finish_reason": comp.finish_reason,
                    "n_tokens": comp.n_tokens,
                    "extracted": extracted,
                    "gold": problem.gold,
                    "verdict": strict_result.reward,
                    "parseable": strict_result.parseable,
                    "verdict_lenient": lenient_result.reward,
                    "extracted_lenient": lenient_extraction.value,
                    "extraction_method": lenient_extraction.method,
                }
            )

        n_truncated_total += n_truncated
        map_rows.append(
            {
                "problem_id": problem.problem_id,
                "split": split,
                "k": len(sample_completions),
                "n_correct_lenient": n_correct_lenient,
                "n_correct_strict": n_correct_strict,
                "n_truncated": n_truncated,
                "methods": method_counts,
            }
        )

    with open(out_dir / "map.jsonl", "w", encoding="utf-8") as f:
        for row in map_rows:
            f.write(json.dumps(row) + "\n")

    with gzip.open(out_dir / "samples.jsonl.gz", "wt", encoding="utf-8") as f:
        for record in sample_records:
            f.write(json.dumps(record) + "\n")

    summary = _build_summary(map_rows, k, model, n_truncated_total, total_samples)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(_render_markdown(summary), encoding="utf-8")

    return summary


def _build_summary(map_rows: list[dict], k: int, model: str, n_truncated_total: int, total_samples: int) -> dict:
    rates = {row["problem_id"]: row["n_correct_lenient"] / row["k"] for row in map_rows}
    train_rates = [rates[row["problem_id"]] for row in map_rows if row["split"] == "train"]
    dev_rates = [rates[row["problem_id"]] for row in map_rows if row["split"] == "dev"]

    histogram = {
        str(n): sum(1 for row in map_rows if row["n_correct_lenient"] == n) for n in range(k + 1)
    }
    band_kept_counts = {
        label: sum(1 for r in rates.values() if lo <= r <= hi) for label, (lo, hi) in _CANDIDATE_BANDS
    }

    return {
        "model": model,
        "k": k,
        "n_problems": len(map_rows),
        "n_train": len(train_rates),
        "n_dev": len(dev_rates),
        "mean_solve_rate_lenient_train": float(np.mean(train_rates)) if train_rates else None,
        "mean_solve_rate_lenient_dev": float(np.mean(dev_rates)) if dev_rates else None,
        "truncation_rate": n_truncated_total / total_samples if total_samples else 0.0,
        "solve_rate_histogram_lenient": histogram,
        "candidate_band_kept_counts": band_kept_counts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _render_markdown(summary: dict) -> str:
    lines = [
        f"# Difficulty map: {summary['model']}",
        "",
        f"- n_problems: {summary['n_problems']} (train={summary['n_train']}, dev={summary['n_dev']})",
        f"- k: {summary['k']}",
        f"- mean solve rate (lenient), train: {summary['mean_solve_rate_lenient_train']}",
        f"- mean solve rate (lenient), dev: {summary['mean_solve_rate_lenient_dev']}",
        f"- truncation_rate: {summary['truncation_rate']:.4f}",
        "",
        "## Solve-rate histogram (lenient)",
        "",
        "| n_correct/k | count |",
        "|---|---|",
    ]
    for key, count in summary["solve_rate_histogram_lenient"].items():
        lines.append(f"| {key}/{summary['k']} | {count} |")
    lines += [
        "",
        "## Candidate band kept-counts",
        "",
        "| band | kept |",
        "|---|---|",
    ]
    for band, count in summary["candidate_band_kept_counts"].items():
        lines.append(f"| {band} | {count} |")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a per-problem difficulty map over GSM8K train (+ dev-holdout)."
    )
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--k", type=int, required=True, help="samples per problem")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True, help="output directory for map/samples/summary")
    parser.add_argument("--limit", type=int, default=None, help="truncate the problem set (smoke usage)")
    parser.add_argument("--backend", choices=["vllm", "fake"], default="vllm")
    parser.add_argument("--fake-script", default=None, help="JSON script path, required for --backend fake")
    return parser


def _build_backend(args: argparse.Namespace) -> GenerationBackend:
    if args.backend == "fake":
        if not args.fake_script:
            raise SystemExit("--backend fake requires --fake-script <json path>")
        script = json.loads(Path(args.fake_script).read_text(encoding="utf-8"))
        return FakeBackend(script)
    return VLLMBackend(model=args.model)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Build (and validate) the backend before loading problems: --backend fake
    # without --fake-script should fail fast, not after the (datasets-requiring)
    # GSM8K load.
    backend = _build_backend(args)

    problems, splits = load_all_problems()
    if args.limit is not None:
        problems = problems[: args.limit]
        splits = splits[: args.limit]

    summary = build_difficulty_map(
        problems,
        splits,
        backend,
        k=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        out_dir=args.out,
        model=args.model,
    )
    print(
        f"wrote {summary['n_problems']} problems (train={summary['n_train']}, "
        f"dev={summary['n_dev']}) to {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
