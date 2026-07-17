#!/usr/bin/env python
"""CLI entry point for running the eval harness against one or more benchmarks.

Fake-backend smoke test (no GPU, no network):
    python scripts/run_eval.py --config configs/eval.yaml --benchmark aime24 \\
        --model dummy --backend fake --fake-script script.json --limit 3

Real run on the GPU box (see report/task-2-report.md "G0 runbook"):
    python scripts/run_eval.py --config configs/eval.yaml --benchmark all \\
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --backend vllm --final
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from grpo_math.config import load_config
from grpo_math.eval.backends import FakeBackend, GenerationBackend, VLLMBackend
from grpo_math.eval.runner import EvalSummary, run_eval

_AIME_BENCHMARKS = {"aime24", "aime25"}


def _slugify_model(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model)


def _default_out_dir(model: str) -> Path:
    slug = _slugify_model(model)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("results") / slug / timestamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the grpo_math eval harness.")
    parser.add_argument("--config", required=True, help="path to eval.yaml")
    parser.add_argument("--benchmark", required=True, help='benchmark name, or "all"')
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--backend", choices=["vllm", "fake"], required=True)
    parser.add_argument("--k", type=int, default=None, help="override k for all benchmarks")
    parser.add_argument("--limit", type=int, default=None, help="truncate each benchmark's problem set")
    parser.add_argument("--out-dir", default=None, help="default: results/<model-slug>/<UTC timestamp>")
    parser.add_argument("--seed", type=int, default=None, help="override eval_cfg['seed']")
    parser.add_argument(
        "--final",
        action="store_true",
        help="use k_final_aime for aime24/aime25 when --k is not given",
    )
    parser.add_argument("--fake-script", default=None, help="JSON script path, required for --backend fake")
    return parser


def _build_backend(args: argparse.Namespace, eval_cfg: dict) -> GenerationBackend:
    if args.backend == "fake":
        if not args.fake_script:
            raise SystemExit("--backend fake requires --fake-script <json path>")
        script = json.loads(Path(args.fake_script).read_text(encoding="utf-8"))
        return FakeBackend(script)
    return VLLMBackend(model=args.model, kv_cache_dtype=eval_cfg.get("kv_cache_dtype", "auto"))


def _print_table(summaries: list[EvalSummary]) -> None:
    print(
        f"{'benchmark':<12} {'k':>4} {'pass@1':>8} {'ci_lo':>8} {'ci_hi':>8} "
        f"{'parse':>7} {'trunc':>7} {'n':>5}"
    )
    for s in summaries:
        print(
            f"{s.benchmark:<12} {s.k:>4} {s.pass_at_1:>8.4f} {s.ci_lo:>8.4f} {s.ci_hi:>8.4f} "
            f"{s.parse_rate:>7.4f} {s.truncation_rate:>7.4f} {s.n_problems:>5}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    eval_cfg = load_config(args.config)
    if args.seed is not None:
        eval_cfg = {**eval_cfg, "seed": args.seed}

    benchmark_names = eval_cfg["benchmarks"] if args.benchmark == "all" else [args.benchmark]
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(args.model)
    backend = _build_backend(args, eval_cfg)

    summaries = []
    for benchmark in benchmark_names:
        k = args.k
        if k is None and args.final and benchmark in _AIME_BENCHMARKS:
            k = eval_cfg["k_final_aime"]
        summary = run_eval(
            benchmark,
            backend,
            eval_cfg,
            k=k,
            limit=args.limit,
            out_dir=out_dir,
            model_name=args.model,
        )
        summaries.append(summary)

    _print_table(summaries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
