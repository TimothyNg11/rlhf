#!/usr/bin/env python
"""G1 reference run: reproduce our custom GRPO trainer's run on the same
config using TRL's ``GRPOTrainer``, so the G1 acceptance gate can compare our
implementation's curves against a well-known reference implementation.

Runs ONLY on the GPU box, in a venv that has ``trl`` (see the ``trl`` extra in
pyproject.toml) plus ``transformers``/``datasets``/``vllm``. Locally (no trl),
this script still imports cleanly and supports ``--help`` / argparse; running
it for real without trl fails loudly with an install hint (see ``main``).

Dry-run smoke test (no GPU, no trl -- argparse only):
    python scripts/run_trl_reference.py --help

Real run on the GPU box:
    python scripts/run_trl_reference.py --config configs/g1_100.yaml --seed 0

Config mapping (``grpo_math`` config key -> ``trl.GRPOConfig`` field):
    output_dir                     = <out_dir>/<run_name>_trl-seed<seed>
    learning_rate                  = train.lr
    lr_scheduler_type               = "constant"                  (matches our fixed-lr schedule)
    weight_decay                   = 0.0
    max_grad_norm                  = train.max_grad_norm
    num_generations                = rollout.group_size            (GRPO group size)
    per_device_train_batch_size    = _PER_DEVICE_BATCH (16; single-GPU micro-batch knob)
    gradient_accumulation_steps    = (train.ppo_mini_batch_size * rollout.group_size)
                                       // _PER_DEVICE_BATCH
    steps_per_generation           = train.prompts_per_step // train.ppo_mini_batch_size
    num_iterations                 = 1
    epsilon                        = train.clip_ratio
    beta                           = train.kl_coef
    loss_type                      = "bnpo"                       (see accepted mismatch 1)
    scale_rewards                  = True
    temperature                    = rollout.temperature
    top_p                          = rollout.top_p
    max_completion_length          = rollout.max_response_tokens
    vllm_max_model_length          = rollout.max_prompt_tokens + rollout.max_response_tokens
                                     (TRL 1.8 dropped max_prompt_length; GSM8K prompts
                                      are all far below our 512-token cap anyway)
    bf16                           = True                         (fp32 master + autocast; deliberately
                                                                     NOT passing model_init_kwargs torch_dtype)
    gradient_checkpointing         = train.grad_checkpointing
    use_vllm                       = True
    vllm_mode                      = "colocate"
    vllm_gpu_memory_utilization    = 0.25
    max_steps                      = cfg max_steps (or --max-steps)
    seed                           = --seed
    logging_steps                  = 1
    save_strategy                  = "no"
    report_to                      = []

NOTE: verify these argument names against the installed TRL version at
runbook time -- if an argument no longer exists on ``GRPOConfig``, TRL raises
a ``TypeError`` at construction time (fail loudly), never silently drops it.

Documented accepted mismatches between this reference run and our trainer
(``src/grpo_math/trainer/loop.py``) -- known, deliberate, and NOT bugs:
  1. TRL's ``loss_type="bnpo"`` normalizes per-device micro-batch; ours
     normalizes by a global per-update token count.
  2. TRL keeps zero-variance groups (advantage 0, tokens still in denominator
     + KL); we drop them.
  3. Truncation->reward-0 is approximated through the reward fn (truncated
     completions are almost always unparseable -> 0.0 + no format bonus for
     parseable-truncated is NOT enforceable through TRL's reward-fn
     interface, which never sees finish_reason; at G0 the truncation rate was
     0.6%, negligible).
  4. Different sampling RNG streams -> G1 compares EMA-smoothed distributional
     agreement, not pointwise equality.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from grpo_math.config import load_config
from grpo_math.data.gsm8k import load_gsm8k_train
from grpo_math.rewards import compute_reward
from grpo_math.trainer.metrics import MetricsLogger

_INSTALL_HINT = (
    "run_trl_reference.py requires the `trl` extra (trl, transformers, datasets; "
    "vllm for the colocated generation backend). Install with: pip install -e .[trl]"
)

# Single-GPU micro-batch knob. 2, not 16: TRL/accelerate materializes fp32
# logits over the 152k vocab per micro-batch (~28 GiB at B=16, ~3.5k padded
# tokens) next to the colocated vLLM engine — B=16 OOMs an 80 GB card. Same
# constraint as our trainer's train.micro_batch_size (see configs/base.yaml).
_PER_DEVICE_BATCH = 2


def derive_trl_batch_shape(cfg: dict) -> tuple[int, int, int]:
    """Derive ``(per_device_train_batch_size, gradient_accumulation_steps,
    steps_per_generation)`` for ``trl.GRPOConfig`` from a ``grpo_math`` cfg.

    ``completions_per_update = train.ppo_mini_batch_size * rollout.group_size``
    must divide evenly by ``_PER_DEVICE_BATCH`` (the single-GPU micro-batch
    size); the quotient is TRL's ``gradient_accumulation_steps``.

    ``steps_per_generation = train.prompts_per_step // train.ppo_mini_batch_size``,
    requiring ``prompts_per_step`` to divide evenly by ``ppo_mini_batch_size``.

    Both divisibility requirements fail loudly (assert) rather than silently
    rounding, since a silent rounding would desync TRL's batch shape from
    ours without any visible symptom.
    """
    completions_per_update = cfg["train"]["ppo_mini_batch_size"] * cfg["rollout"]["group_size"]
    per_device_train_batch_size = _PER_DEVICE_BATCH
    assert completions_per_update % per_device_train_batch_size == 0, (
        f"completions_per_update (train.ppo_mini_batch_size * rollout.group_size = "
        f"{completions_per_update}) must be divisible by per_device_train_batch_size "
        f"({per_device_train_batch_size})"
    )
    gradient_accumulation_steps = completions_per_update // per_device_train_batch_size

    assert cfg["train"]["prompts_per_step"] % cfg["train"]["ppo_mini_batch_size"] == 0, (
        f"train.prompts_per_step ({cfg['train']['prompts_per_step']}) must be divisible "
        f"by train.ppo_mini_batch_size ({cfg['train']['ppo_mini_batch_size']})"
    )
    steps_per_generation = cfg["train"]["prompts_per_step"] // cfg["train"]["ppo_mini_batch_size"]

    return per_device_train_batch_size, gradient_accumulation_steps, steps_per_generation


def reward_fn_factory(format_bonus: float):
    def reward_boxed(completions, gold, **kwargs):
        # completions: conversational -> [{"role":"assistant","content": text}]
        texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        return [
            compute_reward(t, g, truncated=False, format_bonus=format_bonus).reward
            for t, g in zip(texts, gold)
        ]

    return reward_boxed


def _first(logs: dict, *keys: str):
    """Return the first non-None value among ``keys`` in ``logs``, else None."""
    for key in keys:
        value = logs.get(key)
        if value is not None:
            return value
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TRL GRPOTrainer G1 reference run.")
    parser.add_argument("--config", required=True, help="path to a run config yaml (e.g. configs/g1_100.yaml)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results/train", help="parent dir for the run dir")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="override cfg max_steps (g1 config already has 100)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    run_dir = Path(args.out_dir) / f"{cfg['run_name']}_trl-seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # All trl/datasets/transformers imports are lazy and confined to this
    # try/except so the script imports cleanly (argparse, --help, this far
    # into main()) in the trl-free dev venv; this MUST run before any dataset
    # loading below so a missing-trl failure never triggers a gsm8k download.
    try:
        import torch
        import transformers
        import trl
        from datasets import Dataset
        from transformers import TrainerCallback
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        print(f"{_INSTALL_HINT}\n({exc})")
        return 1

    try:
        import vllm

        vllm_version = vllm.__version__
    except ImportError:
        vllm_version = None

    class JsonlLogCallback(TrainerCallback):
        """Mirrors our trainer's train-row schema (see loop.py) so
        plot_curves.py can overlay TRL's curves directly. Missing TRL log
        keys map to None rather than raising."""

        def __init__(self, path: str | Path):
            self._logger = MetricsLogger(path)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "reward" not in logs:
                return
            self._logger.log(
                {
                    "kind": "train",
                    "step": state.global_step,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reward_mean": logs.get("reward"),
                    "kl_mean": logs.get("kl"),
                    "entropy_mean": logs.get("entropy"),
                    "clip_frac": _first(logs, "clip_ratio/region_mean", "clip_ratio"),
                    "response_len_mean": logs.get("completions/mean_length"),
                    "loss": logs.get("loss"),
                    "grad_norm": logs.get("grad_norm"),
                    "lr": logs.get("learning_rate"),
                }
            )

    train_problems, _ = load_gsm8k_train(dev_holdout=cfg["data"]["dev_holdout"])
    # p.prompt already contains the boxed-answer instruction -- do not add it again.
    rows = [{"prompt": [{"role": "user", "content": p.prompt}], "gold": p.gold} for p in train_problems]
    dataset = Dataset.from_list(rows)

    reward_fn = reward_fn_factory(cfg["train"]["format_bonus"])
    max_steps = args.max_steps if args.max_steps is not None else cfg["max_steps"]
    per_device_train_batch_size, gradient_accumulation_steps, steps_per_generation = (
        derive_trl_batch_shape(cfg)
    )

    grpo_config = GRPOConfig(
        output_dir=str(run_dir),
        learning_rate=cfg["train"]["lr"],
        lr_scheduler_type="constant",  # matches our fixed-lr schedule (no warmup/decay)
        weight_decay=0.0,
        max_grad_norm=cfg["train"]["max_grad_norm"],
        num_generations=cfg["rollout"]["group_size"],  # GRPO group size
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        steps_per_generation=steps_per_generation,
        num_iterations=1,
        epsilon=cfg["train"]["clip_ratio"],
        beta=cfg["train"]["kl_coef"],
        loss_type="bnpo",  # accepted mismatch 1: per-device micro-batch norm, not global token count
        scale_rewards=True,
        temperature=cfg["rollout"]["temperature"],
        top_p=cfg["rollout"]["top_p"],
        # TRL 1.8 has no max_prompt_length (prompt capping moved out of the
        # config); GSM8K prompts are all well under our 512-token cap, and
        # vllm_max_model_length mirrors our engine's prompt+response budget.
        max_completion_length=cfg["rollout"]["max_response_tokens"],
        vllm_max_model_length=(
            cfg["rollout"]["max_prompt_tokens"] + cfg["rollout"]["max_response_tokens"]
        ),
        bf16=True,  # fp32 master + autocast; deliberately NOT passing model_init_kwargs torch_dtype
        gradient_checkpointing=cfg["train"]["grad_checkpointing"],
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.25,
        max_steps=max_steps,
        seed=args.seed,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )

    versions = {
        "trl": trl.__version__,
        "transformers": transformers.__version__,
        "vllm": vllm_version,
        "torch": torch.__version__,
    }
    (run_dir / "versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")

    trainer = GRPOTrainer(
        model=cfg["model"]["name"],
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        callbacks=[JsonlLogCallback(run_dir / "metrics.jsonl")],
    )
    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
