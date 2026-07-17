#!/usr/bin/env python
"""CLI entry point for GRPO training (see docs/PLAN.md).

Dry-run smoke test (no GPU, no vllm/transformers):
    python scripts/run_train.py --config configs/g1_100.yaml --dry-run

Real run on the GPU box:
    python scripts/run_train.py --config configs/g1_100.yaml --seed 0
"""

from __future__ import annotations

import os

# Must precede any grpo_math / vllm import: vLLM's V1 multiprocessing executor
# does not play well with the sleep/wake + collective_rpc weight-sync path.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import argparse
import sys
from pathlib import Path

import yaml

from grpo_math.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GRPO training.")
    parser.add_argument("--config", required=True, help="path to a run config yaml")
    parser.add_argument(
        "--seed", type=int, default=None, help="override cfg train.seed (also used in the run-dir name)"
    )
    parser.add_argument("--max-steps", type=int, default=None, help="override cfg max_steps")
    parser.add_argument("--out-dir", default="results/train", help="parent dir for the run dir")
    parser.add_argument("--resume", action="store_true", help="resume from the latest checkpoint")
    parser.add_argument(
        "--limit-prompts", type=int, default=None, help="truncate the loaded train problems"
    )
    parser.add_argument("--sync-check-every", type=int, default=25)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve config + create run dir + write config_resolved.yaml, then exit "
        "without constructing policy/rollout/trainer",
    )
    return parser


def _resolve_cfg(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = _resolve_cfg(args)
    seed = cfg["train"]["seed"]
    run_dir = Path(args.out_dir) / f"{cfg['run_name']}-seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        (run_dir / "config_resolved.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )
        print(f"resolved run dir: {run_dir}")
        return 0

    # Heavy imports (torch, and via GRPOTrainer's None seams vllm/transformers)
    # are deferred past the dry-run exit so the smoke path stays light.
    from grpo_math.trainer.checkpoint import find_latest_checkpoint
    from grpo_math.trainer.loop import GRPOTrainer

    model_path_override = None
    if args.resume:
        ckpt = find_latest_checkpoint(run_dir / "checkpoints")
        if ckpt is not None and (ckpt / "model").exists():
            # Resume reloads the policy from the checkpoint's model dir; the
            # reference model stays the base cfg model.
            model_path_override = str(ckpt / "model")

    trainer = GRPOTrainer(
        cfg,
        run_dir,
        resume=args.resume,
        limit_prompts=args.limit_prompts,
        sync_check_every=args.sync_check_every,
        model_path_override=model_path_override,
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
