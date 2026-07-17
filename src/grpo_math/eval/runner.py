"""Eval runner: orchestrates benchmark loading, generation, grading, stats, and output.

Grading runs sequentially in the main thread (see the concurrency constraint in
docs/PLAN.md / the eval-harness brief): math-verify's POSIX timeout uses
``signal.alarm``, which only works in the main thread, so no thread pool is
used around ``compute_reward``.
"""

from __future__ import annotations

import gzip
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from grpo_math.eval.backends import GenerationBackend
from grpo_math.eval.benchmarks import load_benchmark
from grpo_math.rewards import compute_reward, extract_boxed
from grpo_math.stats import bootstrap_ci, pass_at_1, per_problem_means


@dataclass(frozen=True)
class EvalSummary:
    benchmark: str
    model_name: str
    k: int
    temperature: float
    top_p: float
    max_tokens: int
    seed: int
    n_problems: int
    pass_at_1: float
    ci_lo: float
    ci_hi: float
    parse_rate: float
    truncation_rate: float
    mean_completion_tokens: float
    timestamp: str  # UTC ISO 8601
    git_describe: str | None


def _git_describe() -> str | None:
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def run_eval(
    benchmark: str,
    backend: GenerationBackend,
    eval_cfg: dict,
    *,
    k: int | None = None,
    limit: int | None = None,
    out_dir: str | Path,
    model_name: str,
) -> EvalSummary:
    """Run one benchmark end-to-end: load problems, generate, grade, aggregate
    stats, and write ``out_dir/<benchmark>/{samples.jsonl.gz,summary.json,summary.md}``."""
    k = eval_cfg["k_default"] if k is None else k
    temperature = eval_cfg["temperature"]
    top_p = eval_cfg["top_p"]
    max_tokens = eval_cfg["max_tokens"]
    seed = eval_cfg["seed"]

    problems = load_benchmark(benchmark, limit=limit)
    prompts = [p.prompt for p in problems]

    completions = backend.generate(
        prompts, k=k, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed
    )

    n_problems = len(problems)
    verdicts = np.zeros((n_problems, k))
    records = []
    n_parseable = 0
    n_truncated = 0
    total_tokens = 0

    for pi, (problem, sample_completions) in enumerate(zip(problems, completions)):
        for si, comp in enumerate(sample_completions):
            truncated = comp.finish_reason == "length"
            result = compute_reward(comp.text, problem.gold, truncated=truncated)
            extracted = extract_boxed(comp.text)

            verdicts[pi, si] = result.reward
            if result.parseable:
                n_parseable += 1
            if truncated:
                n_truncated += 1
            total_tokens += comp.n_tokens

            records.append(
                {
                    "problem_id": problem.problem_id,
                    "sample_idx": si,
                    "completion": comp.text,
                    "finish_reason": comp.finish_reason,
                    "n_tokens": comp.n_tokens,
                    "extracted": extracted,
                    "gold": problem.gold,
                    "verdict": result.reward,
                    "parseable": result.parseable,
                }
            )

    total_samples = n_problems * k
    per_problem = per_problem_means(verdicts)

    bench_dir = Path(out_dir) / benchmark
    bench_dir.mkdir(parents=True, exist_ok=True)

    with gzip.open(bench_dir / "samples.jsonl.gz", "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    ci_lo, ci_hi = bootstrap_ci(per_problem, seed=seed)
    summary = EvalSummary(
        benchmark=benchmark,
        model_name=model_name,
        k=k,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        n_problems=n_problems,
        pass_at_1=pass_at_1(verdicts),
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        parse_rate=n_parseable / total_samples if total_samples else 0.0,
        truncation_rate=n_truncated / total_samples if total_samples else 0.0,
        mean_completion_tokens=total_tokens / total_samples if total_samples else 0.0,
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_describe=_git_describe(),
    )

    (bench_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    (bench_dir / "summary.md").write_text(_render_markdown(summary), encoding="utf-8")

    return summary


def _render_markdown(summary: EvalSummary) -> str:
    header = (
        "| benchmark | model | k | pass@1 | 95% CI | parse_rate | truncation_rate "
        "| mean_tokens | n_problems | timestamp |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    row = (
        f"| {summary.benchmark} | {summary.model_name} | {summary.k} "
        f"| {summary.pass_at_1:.4f} | [{summary.ci_lo:.4f}, {summary.ci_hi:.4f}] "
        f"| {summary.parse_rate:.4f} | {summary.truncation_rate:.4f} "
        f"| {summary.mean_completion_tokens:.1f} | {summary.n_problems} "
        f"| {summary.timestamp} |\n"
    )
    return header + row


def load_summary(path: str | Path) -> EvalSummary:
    """Round-trip a ``summary.json`` back into an :class:`EvalSummary`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalSummary(**data)
