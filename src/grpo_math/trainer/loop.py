"""GRPO training orchestration: the single-GPU rollout -> reward -> advantage ->
policy-update loop, its correctness self-checks, and JSONL metrics.

Non-negotiables enforced here (docs/PLAN.md):
  * Old log-probs are recomputed with the policy forward pass (the no-grad
    sweep in step 6); the sampler's log-probs are used ONLY for the
    ``sampler_recompute_logprob_diff`` monitoring metric, never for the loss.
  * The step-0 ratio ~= 1 sanity check and the loss-denominator self-check
    actually execute inside the loop.
  * The weight-sync checksum handshake is a real end-to-end comparison
    (``policy.checksums()`` vs the checksums the rollout computes from the
    tensors it received).

Only ``torch`` + stdlib + already-CPU-safe grpo_math modules are imported at
module level; the real ``VLLMRollout`` / ``HFPolicy`` (which pull in
vllm/transformers) are imported lazily, only when the corresponding injection
seam is ``None``. So the whole module -- and every test that drives the loop
with injected fakes -- imports fine in the vllm/transformers-free dev venv.
"""

from __future__ import annotations

import contextlib
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from grpo_math.data.gsm8k import PromptSampler, load_gsm8k_train
from grpo_math.eval.benchmarks import BENCHMARKS
from grpo_math.eval.runner import run_eval
from grpo_math.rewards import compute_reward
from grpo_math.trainer.algo import (
    TokenBatch,
    collate_token_batch,
    gather_logprobs_and_entropy,
    grpo_microbatch_loss,
    group_normalized_advantages,
    truncated_is_weights,
)
from grpo_math.trainer.checkpoint import (
    find_latest_checkpoint,
    load_rng_state,
    load_trainer_state,
    save_checkpoint,
)
from grpo_math.trainer.metrics import MetricsLogger
from grpo_math.trainer.policy import response_logprobs

# Step-0 sanity checks (docs/PLAN.md): before the first optimizer update the
# recomputed old-logprobs equal the fresh policy's logprobs.
#
# The bf16 production forwards are batch-shape sensitive (observed on H100:
# deterministic per-token |ratio - 1| up to ~0.27, p99 ~0.12, while the mean
# stayed within 1e-2 — kernel-path divergence between the sweep's collation
# and the training collation, not a misalignment; a real off-by-one gives
# ratios of e^2..e^10). So the ratio-mean check runs on the production bf16
# tensors, while the ALIGNMENT check — the one that catches packing/masking/
# shift bugs, which are dtype-independent — reruns both packing paths for the
# first micro-batch's samples in pure fp32 (autocast off), where kernel noise
# is ~1e-4 and the tolerance can stay tight. p99/max of the bf16 ratios are
# logged as diagnostics, not asserted.
_RATIO_MEAN_TOL = 1e-2
_ALIGN_LP_TOL = 1e-2


def pack_response_values(values: list[torch.Tensor], batch: TokenBatch) -> torch.Tensor:
    """Place per-sample 1-D response vectors (e.g. from
    :func:`grpo_math.trainer.policy.response_logprobs`) into a ``[B, L-1]`` fp32
    tensor at the SHIFTED response positions (``batch.response_mask[:, 1:]``),
    zeros elsewhere.

    ``len(values[b])`` must equal ``batch.response_mask[b, 1:].sum()``; this is
    asserted per sample.
    """
    shifted_mask = batch.response_mask[:, 1:]
    batch_size, width = shifted_mask.shape
    out = torch.zeros((batch_size, width), dtype=torch.float32)
    for b in range(batch_size):
        row_mask = shifted_mask[b]
        n = int(row_mask.sum().item())
        assert len(values[b]) == n, (
            f"pack_response_values: sample {b} has {len(values[b])} values but "
            f"{n} shifted response positions"
        )
        out[b, row_mask] = values[b].to(torch.float32)
    return out


def _first_checksum_mismatch(sent: dict, received: dict) -> str:
    if sent.keys() != received.keys():
        return (
            f"key sets differ (sent-only={sorted(set(sent) - set(received))}, "
            f"received-only={sorted(set(received) - set(sent))})"
        )
    for name in sent:
        if sent[name] != received[name]:
            return f"first differing tensor: {name!r}"
    return "no mismatch"


class GRPOTrainer:
    """Drives the GRPO loop. Every external dependency is injectable so the
    loop can be exercised end-to-end on CPU with fakes; when an injection seam
    is ``None`` the real (vllm/transformers-backed) object is built lazily."""

    def __init__(
        self,
        cfg: dict,
        run_dir: str | Path,
        *,
        resume: bool = False,
        rollout=None,
        policy=None,
        ref_policy=None,
        problems: list | None = None,
        sync_check_every: int = 25,
        limit_prompts: int | None = None,
        model_path_override: str | None = None,
    ):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.sync_check_every = sync_check_every
        self.max_steps = cfg["max_steps"]
        self.step0_checks_ran = False

        train_cfg = cfg["train"]
        rollout_cfg = cfg["rollout"]
        model_cfg = cfg["model"]

        # --- init-time correctness checks -----------------------------------
        assert train_cfg["entropy_coef"] == 0.0, (
            "this trainer requires train.entropy_coef == 0.0 (entropy is logged "
            f"as a metric but never added to the loss); got {train_cfg['entropy_coef']!r}"
        )
        # zero_reward: truncated completions get reward 0 and ARE trained on
        #   (pushed down like any wrong answer). mask: truncated completions are
        #   EXCLUDED from the training loss ("overlong filtering", DAPO,
        #   arXiv:2503.14476) -- their reward still informs the group baseline,
        #   but their (cut-off, possibly-valid) reasoning is not punished. This
        #   breaks the length -> truncation -> punishment -> destabilization
        #   feedback that amplifies the entropy runaway (docs/g1_diagnosis.md).
        self.truncation_mode = train_cfg["truncation_mode"]
        assert self.truncation_mode in ("zero_reward", "mask"), (
            "train.truncation_mode must be 'zero_reward' or 'mask'; "
            f"got {self.truncation_mode!r}"
        )
        # True  -> zero-variance groups leave the batch entirely (their tokens
        #          contribute no PG term, no KL term, and no denominator mass).
        # False -> they stay at advantage 0, TRL/DeepSeek-style: no PG signal,
        #          but their tokens still carry KL gradient and still count in
        #          the token denominator, anchoring the policy on the prompts it
        #          answers most consistently. G1 (docs/PLAN.md, report/g1_overlay.png)
        #          traced our entropy blowup to exactly this difference.
        self.skip_zero_variance_groups = bool(train_cfg["skip_zero_variance_groups"])
        # Truncated importance sampling for the rollout-vs-recompute mismatch.
        self.use_tis = bool(train_cfg.get("use_tis", False))
        self.tis_cap = float(train_cfg.get("tis_cap", 2.0))
        expected_holdout = BENCHMARKS["gsm8k_dev"].take_last
        assert cfg["data"]["dev_holdout"] == expected_holdout, (
            f"cfg data.dev_holdout ({cfg['data']['dev_holdout']}) must equal "
            f"BENCHMARKS['gsm8k_dev'].take_last ({expected_holdout}) so the "
            "training train/dev split matches the in-loop eval dev set"
        )

        # --- construct injectable dependencies -------------------------------
        if rollout is None:
            from grpo_math.trainer.rollout import VLLMRollout  # lazy: pulls in vllm

            max_model_len = rollout_cfg["max_prompt_tokens"] + rollout_cfg["max_response_tokens"]
            rollout = VLLMRollout(
                model_cfg["name"],
                max_model_len=max_model_len,
                gpu_memory_utilization=rollout_cfg["gpu_memory_utilization"],
                seed=train_cfg["seed"],
            )
        if policy is None:
            from grpo_math.trainer.policy import HFPolicy  # lazy: pulls in transformers

            policy = HFPolicy(
                model_path_override or model_cfg["name"],
                trainable=True,
                grad_checkpointing=train_cfg["grad_checkpointing"],
            )
        if ref_policy is None:
            from grpo_math.trainer.policy import HFPolicy  # lazy: pulls in transformers

            # Reference model is the BASE cfg model, never the resume override.
            ref_policy = HFPolicy(model_cfg["name"], trainable=False)

        self.rollout = rollout
        self.policy = policy
        self.ref_policy = ref_policy
        self.device = getattr(policy, "device", torch.device("cpu"))

        tokenizer = getattr(policy, "tokenizer", None)
        if tokenizer is not None:
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id
            self.pad_token_id = pad_id
        else:
            self.pad_token_id = 0

        # --- problem set: load, length-filter, limit ------------------------
        if problems is None:
            problems = load_gsm8k_train(dev_holdout=cfg["data"]["dev_holdout"])[0]

        self.n_prompts_filtered = 0
        if tokenizer is not None:
            max_prompt_tokens = rollout_cfg["max_prompt_tokens"]
            kept = []
            for problem in problems:
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": problem.prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if len(ids) <= max_prompt_tokens:
                    kept.append(problem)
            self.n_prompts_filtered = len(problems) - len(kept)
            problems = kept

        if limit_prompts is not None:
            problems = problems[:limit_prompts]
        self.problems = problems

        # --- sampler / optimizer / metrics ----------------------------------
        self.sampler = PromptSampler(
            len(problems), train_cfg["prompts_per_step"], seed=train_cfg["seed"]
        )
        self.optimizer = torch.optim.AdamW(
            policy.model.parameters(), lr=train_cfg["lr"], weight_decay=0.0
        )

        # A `resume=True` request with no checkpoint on disk yet (e.g. the very
        # first run of a `--resume`-flagged job) must behave exactly like a
        # fresh run: fresh metrics.jsonl, fresh config_resolved.yaml. Locate the
        # checkpoint once, up front, so both the MetricsLogger mode and
        # config_resolved.yaml gating (and _restore below) agree on it.
        ckpt_dir = find_latest_checkpoint(self.run_dir / "checkpoints") if resume else None
        resuming = ckpt_dir is not None

        self.metrics = MetricsLogger(self.run_dir / "metrics.jsonl", resume=resuming)
        if not resuming:
            (self.run_dir / "config_resolved.yaml").write_text(
                yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
            )

        # --- resume ----------------------------------------------------------
        self.start_step = 0
        if resuming:
            self._restore(ckpt_dir, model_path_override is not None)

    def _restore(self, ckpt_dir: Path, model_reloaded_externally: bool) -> None:
        """Restore optimizer / sampler / RNG (and, for injected policies, model
        weights) from ``ckpt_dir``. ``model_reloaded_externally`` is True on the
        real path where run_train.py already handed the checkpoint's model dir
        to HFPolicy as ``model_path_override``."""
        model_dir = ckpt_dir / "model"
        if model_dir.exists() and not model_reloaded_externally:
            raise RuntimeError(
                f"checkpoint {ckpt_dir} was saved in save_pretrained format (it "
                "has a model/ dir) but the policy was not reloaded from it -- "
                "resuming now would silently keep the policy's original "
                "(non-resumed) weights while restoring the optimizer/sampler/RNG "
                "state. Pass the checkpoint's model dir as model_path_override "
                "(as scripts/run_train.py does) so the policy is reloaded from "
                "the checkpoint before GRPOTrainer is constructed."
            )
        trainer_state = load_trainer_state(ckpt_dir)
        self.optimizer.load_state_dict(
            torch.load(ckpt_dir / "optimizer.pt", weights_only=False, map_location="cpu")
        )
        self.sampler.load_state_dict(trainer_state["sampler"])
        rng_state = load_rng_state(ckpt_dir)
        if rng_state is not None:
            torch.set_rng_state(rng_state["torch"])
        self.start_step = trainer_state["step"]

        # Model weights: the real path already reloaded HFPolicy from the
        # checkpoint's model dir. With an injected policy, restore the saved
        # state_dict if present (else leave the model as-is).
        model_pt = ckpt_dir / "model.pt"
        if not model_reloaded_externally and model_pt.exists():
            self.policy.model.load_state_dict(
                torch.load(model_pt, weights_only=False, map_location="cpu")
            )

    def _training_autocast(self):
        """bf16 autocast context on non-CPU devices, no-op on CPU -- mirrors
        the rule in :meth:`HFPolicy.forward_logprobs`."""
        device = getattr(self.policy, "device", torch.device("cpu"))
        if device.type == "cpu":
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    def train(self) -> None:
        try:
            self._run_loop()
        finally:
            self.metrics.close()

    # ---------------------------------------------------------------------

    def _run_eval(self, step: int) -> None:
        cfg = self.cfg
        eval_section = cfg["eval_during_training"]
        backend = self.rollout.as_generation_backend()
        for bench in eval_section["benchmarks"]:
            k = eval_section["k"][bench]
            eval_cfg = {
                "k_default": k,
                "temperature": eval_section["temperature"],
                "top_p": eval_section["top_p"],
                "max_tokens": eval_section["max_tokens"],
                "seed": eval_section["seed"],
            }
            summary = run_eval(
                bench,
                backend,
                eval_cfg,
                k=k,
                out_dir=self.run_dir / "eval" / f"step_{step:04d}",
                model_name=cfg["run_name"],
            )
            self.metrics.log(
                {
                    "kind": "eval",
                    "step": step,
                    "benchmark": bench,
                    "pass_at_1": summary.pass_at_1,
                    "ci_lo": summary.ci_lo,
                    "ci_hi": summary.ci_hi,
                    "parse_rate": summary.parse_rate,
                    "truncation_rate": summary.truncation_rate,
                    "mean_completion_tokens": summary.mean_completion_tokens,
                }
            )

    def _run_loop(self) -> None:
        cfg = self.cfg
        train_cfg = cfg["train"]
        rollout_cfg = cfg["rollout"]
        eval_section = cfg["eval_during_training"]

        for step in range(self.start_step, self.max_steps):
            step_start = time.perf_counter()

            # 1. wake + weight-sync handshake -------------------------------
            # Release torch's cached blocks before the engine re-maps its GPU
            # memory: after the first optimizer step the HF side holds AdamW
            # states plus cached (freed) logits buffers, and without this the
            # cumem allocator OOMs on wake_up a few iterations in.
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.rollout.wake()
            do_sync_check = step % self.sync_check_every == 0
            sent = self.policy.checksums() if do_sync_check else None
            received = self.rollout.sync_weights(
                self.policy.sync_iterator(), want_checksums=do_sync_check
            )
            if do_sync_check:
                assert sent == received, (
                    f"weight-sync checksum handshake failed at step {step}: "
                    f"{_first_checksum_mismatch(sent, received)}"
                )

            # 2. eval at cadence --------------------------------------------
            if step % eval_section["every_steps"] == 0:
                self._run_eval(step)

            # 3. sample prompts + generate ----------------------------------
            gen_start = time.perf_counter()
            idx = self.sampler.next_batch()
            prompts = [self.problems[i].prompt for i in idx]
            golds = [self.problems[i].gold for i in idx]
            groups = self.rollout.generate(
                prompts,
                group_size=rollout_cfg["group_size"],
                temperature=rollout_cfg["temperature"],
                top_p=rollout_cfg["top_p"],
                max_tokens=rollout_cfg["max_response_tokens"],
                seed=train_cfg["seed"] * 100003 + step,
            )
            gen_time_s = time.perf_counter() - gen_start
            train_start = time.perf_counter()

            # 4. rewards on the main thread (BEFORE any dropping) -----------
            reward_rows: list[list[float]] = []
            all_rewards: list[float] = []
            all_parseable: list[bool] = []
            all_truncated: list[bool] = []
            all_resp_len: list[int] = []
            masked_rows: list[list[bool]] = []  # per completion: excluded from training loss?
            for group_idx, group in enumerate(groups):
                gold = golds[group_idx]
                row = []
                masked_row = []
                for sample in group:
                    result = compute_reward(
                        sample.text,
                        gold,
                        truncated=(sample.finish_reason == "length"),
                        truncation_mode=self.truncation_mode,
                        format_bonus=train_cfg["format_bonus"],
                    )
                    row.append(result.reward)
                    masked_row.append(result.masked)
                    all_rewards.append(result.reward)
                    all_parseable.append(result.parseable)
                    all_truncated.append(sample.finish_reason == "length")
                    all_resp_len.append(len(sample.response_token_ids))
                reward_rows.append(row)
                masked_rows.append(masked_row)

            reward_mean = float(np.mean(all_rewards))
            frac_correct = float(np.mean([r == 1.0 for r in all_rewards]))
            parse_rate = float(np.mean(all_parseable))
            frac_truncated = float(np.mean(all_truncated))
            response_len_mean = float(np.mean(all_resp_len))
            response_len_p90 = float(np.percentile(all_resp_len, 90))

            # 5. advantages (+ zero-variance groups per skip_zero_variance_groups)
            rewards_t = torch.tensor(reward_rows, dtype=torch.float32)
            advantages, keep_mask = group_normalized_advantages(rewards_t)
            n_zero_var_groups = int((~keep_mask).sum().item())

            # Advantages were computed over the FULL group (masked completions'
            # rewards still inform the group baseline); masked completions are
            # then dropped from the training batch below.
            n_masked_truncated = int(sum(sum(mr) for mr in masked_rows))

            kept_groups: list[list[int]] = []  # each: flat indices into flat_kept
            flat_kept = []  # RolloutSamples of groups in the training batch
            flat_adv: list[float] = []
            for group_idx, group in enumerate(groups):
                # When keeping them, a zero-variance group's advantages are
                # already exactly 0 (the numerator r - mean is identically 0),
                # so no special-casing of the values is needed here.
                if self.skip_zero_variance_groups and not bool(keep_mask[group_idx]):
                    continue
                group_indices = []
                for sample_idx, sample in enumerate(group):
                    if masked_rows[group_idx][sample_idx]:
                        continue  # truncation_mode 'mask': not trained on
                    flat_kept.append(sample)
                    flat_adv.append(float(advantages[group_idx, sample_idx].item()))
                    group_indices.append(len(flat_kept) - 1)
                if group_indices:  # a fully-masked group contributes nothing
                    kept_groups.append(group_indices)

            # training-derived metrics default to the all-dropped case
            entropy_mean = None
            kl_mean = 0.0
            ratio_mean = 0.0
            ratio_max = 0.0
            clip_frac = 0.0
            pg_loss = 0.0
            loss_total = 0.0
            grad_norm = 0.0
            n_updates = 0
            sampler_recompute_logprob_diff = None
            tis_weight_mean = None
            tis_capped_frac = None
            n_completions = len(flat_kept)

            if kept_groups:
                (
                    entropy_mean,
                    kl_mean,
                    ratio_mean,
                    ratio_max,
                    clip_frac,
                    pg_loss,
                    loss_total,
                    grad_norm,
                    n_updates,
                    sampler_recompute_logprob_diff,
                    tis_weight_mean,
                    tis_capped_frac,
                ) = self._train_on_kept(step, kept_groups, flat_kept, flat_adv)

            # 9. log train row ----------------------------------------------
            train_time_s = time.perf_counter() - train_start
            step_time_s = time.perf_counter() - step_start
            self.metrics.log(
                {
                    "kind": "train",
                    "step": step + 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "epoch": self.sampler.epoch,
                    "reward_mean": reward_mean,
                    "frac_correct": frac_correct,
                    "parse_rate": parse_rate,
                    "frac_truncated": frac_truncated,
                    "response_len_mean": response_len_mean,
                    "response_len_p90": response_len_p90,
                    "entropy_mean": entropy_mean,
                    "kl_mean": kl_mean,
                    "ratio_mean": ratio_mean,
                    "ratio_max": ratio_max,
                    "clip_frac": clip_frac,
                    "pg_loss": pg_loss,
                    "loss": loss_total,
                    "grad_norm": grad_norm,
                    "lr": train_cfg["lr"],
                    "n_completions": n_completions,
                    "n_zero_var_groups": n_zero_var_groups,
                    "n_masked_truncated": n_masked_truncated,
                    "n_updates": n_updates,
                    "sampler_recompute_logprob_diff": sampler_recompute_logprob_diff,
                    "tis_weight_mean": tis_weight_mean,
                    "tis_capped_frac": tis_capped_frac,
                    "gen_time_s": gen_time_s,
                    "train_time_s": train_time_s,
                    "step_time_s": step_time_s,
                }
            )

            # 10. checkpoint -------------------------------------------------
            if (step + 1) % train_cfg["save_every"] == 0 or (step + 1) == self.max_steps:
                save_checkpoint(
                    self.run_dir / "checkpoints",
                    step=step + 1,
                    model=self.policy.model,
                    optimizer=self.optimizer,
                    tokenizer=getattr(self.policy, "tokenizer", None),
                    trainer_state={
                        "step": step + 1,
                        "sampler": self.sampler.state_dict(),
                        "run_name": cfg["run_name"],
                    },
                    # Only torch's global RNG needs saving: every numpy RNG in
                    # the loop is freshly seeded per step (rollout + permutation
                    # seeds), so no global numpy state is carried across steps.
                    rng_state={"torch": torch.get_rng_state()},
                )

    def _step0_alignment_check(self, micro_indices, flat_kept) -> None:
        """Dtype-independent packing/shift verification (the real point of the
        step-0 gate): rerun BOTH logprob paths for one micro-batch's samples in
        pure fp32 with autocast off — the per-sample sweep collation
        (``response_logprobs``'s own batching) vs the training collation — and
        require near-equality per token. Kernel noise in fp32 is ~1e-4; any
        packing, masking, or causal-shift bug shows up as |dev| >= ~1."""
        samples = [
            (flat_kept[i].prompt_token_ids, flat_kept[i].response_token_ids)
            for i in micro_indices
        ]
        sweep_lp, _ = response_logprobs(
            self.policy.model,
            samples,
            micro_batch_size=len(samples),
            pad_token_id=self.pad_token_id,
        )
        batch = collate_token_batch(samples, pad_token_id=self.pad_token_id)
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        with torch.no_grad():
            out = self.policy.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = getattr(out, "logits", out)
        train_lp, _ = gather_logprobs_and_entropy(logits[:, :-1], input_ids[:, 1:])
        sweep_packed = pack_response_values(sweep_lp, batch).to(train_lp.device)
        resp_mask = batch.response_mask[:, 1:].to(train_lp.device)
        max_dev = (
            float((train_lp - sweep_packed).abs()[resp_mask].max().item())
            if bool(resp_mask.any())
            else 0.0
        )
        assert max_dev < _ALIGN_LP_TOL, (
            f"step-0 fp32 alignment check failed: max |logprob dev| = {max_dev} "
            f"(expected < {_ALIGN_LP_TOL}). The sweep collation and the training "
            "collation disagree in pure fp32 — this indicates a packing/masking/"
            "causal-shift bug, not kernel noise."
        )

    def _train_on_kept(self, step, kept_groups, flat_kept, flat_adv):
        """Steps 6-8: no-grad old/ref log-prob sweep, then the permuted
        mini-batch / micro-batch policy update with the step-0 self-checks.
        Returns the step's training-derived metrics."""
        train_cfg = self.cfg["train"]
        micro = train_cfg["micro_batch_size"]
        ppo_mini = train_cfg["ppo_mini_batch_size"]

        # 6. no-grad old/ref sweep (old-logprobs recomputed here, NOT taken
        #    from the sampler).
        self.rollout.sleep()
        forward_samples = [(s.prompt_token_ids, s.response_token_ids) for s in flat_kept]
        old_lp, entropy_list = self.policy.forward_logprobs(
            forward_samples, micro_batch_size=micro, compute_entropy=True
        )
        ref_lp, _ = self.ref_policy.forward_logprobs(forward_samples, micro_batch_size=micro)

        entropy_cat = torch.cat(entropy_list) if entropy_list else torch.zeros(0)
        entropy_mean = float(entropy_cat.mean().item()) if entropy_cat.numel() else 0.0

        # Per-token truncated-IS weights (min(exp(old_lp - sampling_lp), cap)),
        # one tensor per kept sample aligned to its response tokens; also the
        # monitoring probe (mean |sampler - recompute| drift). Samples whose
        # rollout gave no sampler logprobs (or TIS disabled) fall back to weight
        # 1 -> no correction, which is what the CPU fakes exercise.
        tis_list: list[torch.Tensor] = []
        diff_sum = 0.0
        diff_count = 0
        tis_sum = 0.0
        tis_count = 0
        tis_capped = 0
        for i, sample in enumerate(flat_kept):
            if self.use_tis and sample.sampler_logprobs is not None:
                sampling_lp = torch.tensor(sample.sampler_logprobs, dtype=torch.float32)
                w = truncated_is_weights(old_lp[i], sampling_lp, cap=self.tis_cap)
                tis_list.append(w)
                tis_sum += float(w.sum().item())
                tis_count += w.numel()
                tis_capped += int((w >= self.tis_cap).sum().item())
            else:
                tis_list.append(torch.ones_like(old_lp[i]))
            if sample.sampler_logprobs is not None:
                d = (torch.tensor(sample.sampler_logprobs, dtype=torch.float32) - old_lp[i]).abs()
                diff_sum += float(d.sum().item())
                diff_count += d.numel()
        sampler_recompute_logprob_diff = diff_sum / diff_count if diff_count else None
        tis_weight_mean = tis_sum / tis_count if tis_count else None
        tis_capped_frac = tis_capped / tis_count if tis_count else None

        # 7. permute kept groups, split into mini-batches of prompt-groups.
        perm = np.random.default_rng(train_cfg["seed"] * 999983 + step).permutation(len(kept_groups))
        permuted_groups = [kept_groups[j] for j in perm]

        n_updates = 0
        grad_norm = 0.0
        step_loss_total = 0.0
        step_ratio_sum = 0.0
        step_ratio_max = 0.0
        step_n_tokens = 0
        step_pg_loss_sum = 0.0
        step_kl_sum = 0.0
        step_clip_count = 0

        for mb_start in range(0, len(permuted_groups), ppo_mini):
            mini_groups = permuted_groups[mb_start : mb_start + ppo_mini]
            mini_flat_indices = [i for grp in mini_groups for i in grp]
            global_token_count = sum(
                len(flat_kept[i].response_token_ids) for i in mini_flat_indices
            )

            mini_loss_item_sum = 0.0
            mini_ratio_sum = 0.0
            mini_ratio_max = 0.0
            mini_ratio_p99 = 0.0
            mini_n_tokens = 0
            mini_pg_loss_sum = 0.0
            mini_kl_sum = 0.0
            mini_clip_count = 0

            for micro_start in range(0, len(mini_flat_indices), micro):
                micro_indices = mini_flat_indices[micro_start : micro_start + micro]
                micro_samples = [
                    (flat_kept[i].prompt_token_ids, flat_kept[i].response_token_ids)
                    for i in micro_indices
                ]
                batch = collate_token_batch(micro_samples, pad_token_id=self.pad_token_id)
                input_ids = batch.input_ids.to(self.device)
                attention_mask = batch.attention_mask.to(self.device)

                with self._training_autocast():
                    out = self.policy.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = getattr(out, "logits", out)
                lp, _ = gather_logprobs_and_entropy(logits[:, :-1], input_ids[:, 1:])

                old_packed = pack_response_values([old_lp[i] for i in micro_indices], batch).to(
                    self.device
                )
                ref_packed = pack_response_values([ref_lp[i] for i in micro_indices], batch).to(
                    self.device
                )
                adv_slice = torch.tensor(
                    [flat_adv[i] for i in micro_indices], dtype=torch.float32, device=self.device
                )
                resp_mask = batch.response_mask[:, 1:].to(self.device)
                tis_packed = pack_response_values(
                    [tis_list[i] for i in micro_indices], batch
                ).to(self.device)

                loss_out = grpo_microbatch_loss(
                    lp,
                    old_packed,
                    ref_packed,
                    adv_slice,
                    resp_mask,
                    clip_ratio=train_cfg["clip_ratio"],
                    kl_coef=train_cfg["kl_coef"],
                    global_token_count=global_token_count,
                    tis_weight=tis_packed,
                )
                loss_out.loss.backward()

                mini_loss_item_sum += float(loss_out.loss.item())
                mini_ratio_sum += loss_out.ratio_sum
                mini_ratio_max = max(mini_ratio_max, loss_out.ratio_max)
                mini_ratio_p99 = max(mini_ratio_p99, loss_out.ratio_abs_dev_p99)
                mini_n_tokens += loss_out.n_tokens
                mini_pg_loss_sum += loss_out.pg_loss_sum
                mini_kl_sum += loss_out.kl_sum
                mini_clip_count += loss_out.clip_count

            # 8. step-0 self-checks (before this mini-batch's optimizer.step)
            if step == 0:
                expected_loss = (
                    mini_pg_loss_sum + train_cfg["kl_coef"] * mini_kl_sum
                ) / global_token_count
                assert abs(mini_loss_item_sum - expected_loss) < 1e-5, (
                    f"step-0 loss-denominator self-check failed at mini-batch "
                    f"{mb_start // ppo_mini}: sum(micro losses)={mini_loss_item_sum} != "
                    f"(sum pg + kl_coef*sum kl)/global_token_count={expected_loss}"
                )
                if mb_start == 0:
                    ratio_mean_mb = mini_ratio_sum / mini_n_tokens if mini_n_tokens else 0.0
                    assert abs(ratio_mean_mb - 1.0) < _RATIO_MEAN_TOL, (
                        f"step-0 ratio_mean sanity check failed: ratio_mean={ratio_mean_mb} "
                        "(expected ~1.0; recomputed old-logprobs must equal the fresh "
                        "policy's logprobs before the first optimizer step; bf16 "
                        f"diagnostics: ratio_max={mini_ratio_max}, p99 |ratio-1|={mini_ratio_p99})"
                    )
                    self._step0_alignment_check(mini_flat_indices[:micro], flat_kept)
                    self.step0_checks_ran = True

            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.policy.model.parameters(), train_cfg["max_grad_norm"]
                )
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            n_updates += 1

            step_loss_total += mini_loss_item_sum
            step_ratio_sum += mini_ratio_sum
            step_ratio_max = max(step_ratio_max, mini_ratio_max)
            step_n_tokens += mini_n_tokens
            step_pg_loss_sum += mini_pg_loss_sum
            step_kl_sum += mini_kl_sum
            step_clip_count += mini_clip_count

        ratio_mean = step_ratio_sum / step_n_tokens if step_n_tokens else 0.0
        kl_mean = step_kl_sum / step_n_tokens if step_n_tokens else 0.0
        clip_frac = step_clip_count / step_n_tokens if step_n_tokens else 0.0
        pg_loss = step_pg_loss_sum / step_n_tokens if step_n_tokens else 0.0

        return (
            entropy_mean,
            kl_mean,
            ratio_mean,
            step_ratio_max,
            clip_frac,
            pg_loss,
            step_loss_total,
            grad_norm,
            n_updates,
            sampler_recompute_logprob_diff,
            tis_weight_mean,
            tis_capped_frac,
        )
