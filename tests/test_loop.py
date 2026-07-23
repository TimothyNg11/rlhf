"""End-to-end tests for grpo_math.trainer.loop: drive the REAL GRPOTrainer loop
on CPU with injected fakes (TinyPolicy + FakeRollout), plus a golden unit test
for pack_response_values.
"""

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from grpo_math.config import load_config
from grpo_math.eval.benchmarks import BENCHMARKS, EvalProblem, load_benchmark, register_local_benchmark
from grpo_math.trainer.algo import collate_token_batch
from grpo_math.trainer.loop import GRPOTrainer, pack_response_values
from grpo_math.trainer.metrics import read_metrics
from grpo_math.trainer.policy import (
    bf16_sync_tensors,
    model_checksums,
    response_logprobs,
    tensor_checksum,
)
from grpo_math.trainer.rollout import RolloutGenerationBackend, RolloutSample

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parent.parent

PAD_ID = 0


# --- module-level fakes -------------------------------------------------------


class TinyLM(nn.Module):
    """Embedding(32, 8) + Linear(8, 32); forward returns raw logits (vocab 32,
    pad id 0). No attention, so per-position logits are padding-invariant --
    which makes the step-0 ratio exactly 1."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 8)
        self.linear = nn.Linear(8, 32)

    def forward(self, input_ids=None, attention_mask=None):
        return self.linear(self.embed(input_ids))


class TinyPolicy:
    def __init__(self):
        self.model = TinyLM()
        self.tokenizer = None
        self.device = torch.device("cpu")

    def forward_logprobs(self, samples, *, micro_batch_size, compute_entropy=False):
        return response_logprobs(
            self.model,
            samples,
            micro_batch_size=micro_batch_size,
            pad_token_id=PAD_ID,
            compute_entropy=compute_entropy,
        )

    def sync_iterator(self):
        return bf16_sync_tensors(self.model)

    def checksums(self):
        return model_checksums(self.model)


class FakeRollout:
    """Fabricates deterministic RolloutSamples from a prompt->texts script and
    records wake/sleep/sync calls. ``sync_weights`` hashes the tensors it
    receives with the same tensor_checksum helper as the policy, so the
    weight-sync handshake really passes end-to-end (unless ``corrupt_checksums``
    forces a mismatch)."""

    def __init__(self, answers, *, corrupt_checksums=False, sampler_logprob=None,
                 truncated_prompts=()):
        self.answers = answers
        self.corrupt_checksums = corrupt_checksums
        # When set, every response token reports this constant sampler logprob,
        # so the trainer's TIS path has non-None sampling_logprobs to work with.
        self.sampler_logprob = sampler_logprob
        # Prompts whose completions all report finish_reason "length" (truncated).
        self.truncated_prompts = set(truncated_prompts)
        self.wake_calls = 0
        self.sleep_calls = 0
        self.sync_calls = 0
        self.want_checksums_calls = []  # want_checksums as seen on each sync_weights call

    def generate(self, prompts, *, group_size, temperature, top_p, max_tokens, seed):
        groups = []
        for prompt_idx, prompt in enumerate(prompts):
            texts = self.answers[prompt]
            group = []
            for i in range(group_size):
                text = texts[i % len(texts)]
                prompt_token_ids = [1 + (hash(prompt) % 20)] * 3
                response_token_ids = [2 + (i % 5)] * (4 + len(text) % 4)
                sampler_logprobs = (
                    None
                    if self.sampler_logprob is None
                    else [self.sampler_logprob] * len(response_token_ids)
                )
                group.append(
                    RolloutSample(
                        prompt_idx=prompt_idx,
                        text=text,
                        finish_reason="length" if prompt in self.truncated_prompts else "stop",
                        prompt_token_ids=prompt_token_ids,
                        response_token_ids=response_token_ids,
                        sampler_logprobs=sampler_logprobs,
                    )
                )
            groups.append(group)
        return groups

    def wake(self):
        self.wake_calls += 1

    def sleep(self):
        self.sleep_calls += 1

    def sync_weights(self, weights, *, want_checksums=True):
        self.sync_calls += 1
        self.want_checksums_calls.append(want_checksums)
        if not want_checksums:
            return {}
        received = {name: tensor_checksum(t) for name, t in weights}
        if self.corrupt_checksums:
            received = {name: "0" * 64 for name in received}
        return received

    def as_generation_backend(self):
        return RolloutGenerationBackend(self)


# --- fixtures / helpers -------------------------------------------------------


@pytest.fixture(autouse=True)
def _register_tiny_math():
    register_local_benchmark("tiny_math", FIXTURES_DIR / "tiny_math.jsonl")
    yield
    BENCHMARKS.pop("tiny_math", None)


def _make_cfg():
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    cfg["max_steps"] = 3
    cfg["train"]["prompts_per_step"] = 4
    cfg["train"]["ppo_mini_batch_size"] = 2
    cfg["train"]["micro_batch_size"] = 2
    cfg["train"]["save_every"] = 2
    # The tiny fake policy has a 32-token vocab (max entropy ln 32 ~ 3.5), a
    # totally different scale from the real 152k-vocab model the 2.0 stop-loss
    # threshold targets; disable it here so tests exercise the full loop. The
    # stop-loss itself is covered by its own dedicated tests.
    cfg["train"]["entropy_abort_threshold"] = None
    cfg["rollout"]["group_size"] = 2
    ev = cfg["eval_during_training"]
    cfg["eval_during_training"] = {
        "every_steps": 2,
        "benchmarks": ["tiny_math"],
        "k": {"tiny_math": 1},
        "temperature": ev["temperature"],
        "top_p": ev["top_p"],
        "max_tokens": ev["max_tokens"],
        "seed": ev["seed"],
    }
    return cfg


def _training_problems():
    return [
        EvalProblem(problem_id="q1", prompt="Q1", gold="1", metadata={}),
        EvalProblem(problem_id="q2", prompt="Q2", gold="2", metadata={}),
        EvalProblem(problem_id="q3", prompt="Q3", gold="3", metadata={}),
        EvalProblem(problem_id="q4", prompt="Q4", gold="4", metadata={}),
    ]


def _variance_answers():
    """Per prompt: one correct (reward 1.0), one parseable-wrong (format_bonus
    0.2) -> group has variance and survives dropping."""
    return {
        "Q1": ["\\boxed{1}", "\\boxed{8}"],
        "Q2": ["\\boxed{2}", "\\boxed{8}"],
        "Q3": ["\\boxed{3}", "\\boxed{8}"],
        "Q4": ["\\boxed{4}", "\\boxed{8}"],
    }


def _tiny_math_eval_answers():
    """One answer per tiny_math prompt (eval runs with k=1). First three
    correct, last three wrong -> deterministic pass@1 = 0.5."""
    problems = load_benchmark("tiny_math")
    answers = {}
    for i, p in enumerate(problems):
        answers[p.prompt] = [f"\\boxed{{{p.gold}}}"] if i < 3 else ["\\boxed{9999}"]
    return answers


def _answers_with_eval(training_answers):
    merged = dict(training_answers)
    merged.update(_tiny_math_eval_answers())
    return merged


def _make_trainer(
    run_dir, problems, answers, cfg=None, *, corrupt_checksums=False, sampler_logprob=None,
    truncated_prompts=(), **kwargs
):
    if cfg is None:
        cfg = _make_cfg()
    torch.manual_seed(0)
    policy = TinyPolicy()
    ref_policy = TinyPolicy()
    rollout = FakeRollout(
        answers, corrupt_checksums=corrupt_checksums, sampler_logprob=sampler_logprob,
        truncated_prompts=truncated_prompts,
    )
    trainer = GRPOTrainer(
        cfg,
        run_dir,
        rollout=rollout,
        policy=policy,
        ref_policy=ref_policy,
        problems=problems,
        **kwargs,
    )
    return trainer, rollout


def _train_rows(run_dir):
    return [r for r in read_metrics(run_dir / "metrics.jsonl") if r["kind"] == "train"]


def _eval_rows(run_dir):
    return [r for r in read_metrics(run_dir / "metrics.jsonl") if r["kind"] == "eval"]


# --- tests --------------------------------------------------------------------


def test_pack_response_values_golden():
    # sample 0: prompt [1,2] + response [3,4,5] -> seq 5
    # sample 1: prompt [6]  + response [7,8]    -> seq 3 (padded to 5)
    samples = [([1, 2], [3, 4, 5]), ([6], [7, 8])]
    batch = collate_token_batch(samples, pad_token_id=PAD_ID)
    values = [torch.tensor([10.0, 11.0, 12.0]), torch.tensor([20.0, 21.0])]

    packed = pack_response_values(values, batch)

    assert packed.shape == (2, 4)  # [B, L-1] with L = 5
    assert torch.equal(packed[0], torch.tensor([0.0, 10.0, 11.0, 12.0]))
    assert torch.equal(packed[1], torch.tensor([20.0, 21.0, 0.0, 0.0]))


def test_three_steps_write_train_metrics_rows(tmp_path):
    trainer, _ = _make_trainer(tmp_path, _training_problems(), _answers_with_eval(_variance_answers()))
    trainer.train()

    rows = _train_rows(tmp_path)
    assert [r["step"] for r in rows] == [1, 2, 3]

    expected_keys = {
        "kind", "step", "timestamp", "epoch", "reward_mean", "frac_correct",
        "parse_rate", "frac_truncated", "response_len_mean", "response_len_p90",
        "entropy_mean", "kl_mean", "ratio_mean", "ratio_max", "clip_frac",
        "pg_loss", "loss", "grad_norm", "lr", "n_completions", "n_zero_var_groups",
        "n_masked_truncated", "n_updates", "sampler_recompute_logprob_diff",
        "tis_weight_mean", "tis_capped_frac", "gen_time_s", "train_time_s", "step_time_s",
    }
    for row in rows:
        assert expected_keys <= row.keys()


def test_two_optimizer_updates_per_iteration(tmp_path):
    cfg = _make_cfg()
    cfg["max_steps"] = 1  # exactly one iteration -> exactly two updates
    trainer, _ = _make_trainer(
        tmp_path, _training_problems(), _answers_with_eval(_variance_answers()), cfg=cfg
    )

    calls = []
    original_step = trainer.optimizer.step

    def counting_step(*args, **kwargs):
        calls.append(1)
        return original_step(*args, **kwargs)

    trainer.optimizer.step = counting_step
    trainer.train()

    assert len(calls) == 2  # 4 prompts / ppo_mini_batch_size 2
    rows = _train_rows(tmp_path)
    assert rows[0]["n_updates"] == 2


def test_zero_variance_groups_dropped(tmp_path):
    cfg = _make_cfg()
    cfg["max_steps"] = 1
    answers = _variance_answers()
    answers["Q1"] = ["\\boxed{1}"]  # both completions correct -> zero variance -> dropped
    trainer, _ = _make_trainer(
        tmp_path, _training_problems(), _answers_with_eval(answers), cfg=cfg
    )
    trainer.train()

    row = _train_rows(tmp_path)[0]
    assert row["n_zero_var_groups"] == 1
    assert row["n_completions"] == (4 - 1) * 2  # (n_prompts - 1) * G


def test_all_groups_zero_variance_step_is_skipped(tmp_path):
    cfg = _make_cfg()
    cfg["max_steps"] = 1
    # Every prompt's completions are identical -> every group zero-variance.
    answers = {"Q1": ["\\boxed{1}"], "Q2": ["\\boxed{2}"], "Q3": ["\\boxed{3}"], "Q4": ["\\boxed{4}"]}
    trainer, _ = _make_trainer(
        tmp_path, _training_problems(), _answers_with_eval(answers), cfg=cfg
    )
    trainer.train()  # must not crash

    row = _train_rows(tmp_path)[0]
    assert row["n_updates"] == 0
    assert row["n_zero_var_groups"] == 4


def test_zero_variance_groups_kept_when_flag_false(tmp_path):
    # TRL/DeepSeek parity path: the group stays in the batch at advantage 0.
    cfg = _make_cfg()
    cfg["max_steps"] = 1
    cfg["train"]["skip_zero_variance_groups"] = False
    answers = _variance_answers()
    answers["Q1"] = ["\\boxed{1}"]  # both completions correct -> zero variance
    trainer, _ = _make_trainer(
        tmp_path, _training_problems(), _answers_with_eval(answers), cfg=cfg
    )
    trainer.train()

    row = _train_rows(tmp_path)[0]
    assert row["n_zero_var_groups"] == 1  # still counted/reported
    assert row["n_completions"] == 4 * 2  # but nothing dropped: all prompts x G


def test_all_zero_variance_still_updates_when_keeping(tmp_path):
    # With every group at advantage 0 the PG term vanishes exactly, but the KL
    # term does not -- so the optimizer still steps. This is the anchoring
    # mechanism that keeps TRL stable where our dropping path drifts.
    cfg = _make_cfg()
    cfg["max_steps"] = 1
    cfg["train"]["skip_zero_variance_groups"] = False
    answers = {"Q1": ["\\boxed{1}"], "Q2": ["\\boxed{2}"], "Q3": ["\\boxed{3}"], "Q4": ["\\boxed{4}"]}
    trainer, _ = _make_trainer(
        tmp_path, _training_problems(), _answers_with_eval(answers), cfg=cfg
    )
    trainer.train()

    row = _train_rows(tmp_path)[0]
    assert row["n_zero_var_groups"] == 4
    assert row["n_updates"] == 2  # vs 0 on the dropping path
    assert row["pg_loss"] == 0.0  # advantages all exactly 0 -> no PG signal
    assert row["kl_mean"] > 0.0  # KL is what remains, and it is what trains


def test_tis_engages_when_sampler_logprobs_present(tmp_path):
    # With sampler logprobs supplied and use_tis on, the trainer reports a
    # non-None mean TIS weight; with the flag off it stays None even though the
    # sampler logprobs are still there.
    cfg = _make_cfg()
    cfg["max_steps"] = 1
    cfg["train"]["use_tis"] = True
    cfg["train"]["tis_cap"] = 2.0
    trainer, _ = _make_trainer(
        tmp_path, _training_problems(), _answers_with_eval(_variance_answers()),
        cfg=cfg, sampler_logprob=-1.0,
    )
    trainer.train()
    row = _train_rows(tmp_path)[0]
    assert row["tis_weight_mean"] is not None
    assert row["tis_capped_frac"] is not None
    assert 0.0 < row["tis_weight_mean"] <= cfg["train"]["tis_cap"]


def test_tis_none_when_disabled(tmp_path):
    cfg = _make_cfg()
    cfg["max_steps"] = 1
    cfg["train"]["use_tis"] = False
    trainer, _ = _make_trainer(
        tmp_path, _training_problems(), _answers_with_eval(_variance_answers()),
        cfg=cfg, sampler_logprob=-1.0,
    )
    trainer.train()
    assert _train_rows(tmp_path)[0]["tis_weight_mean"] is None


def test_truncation_mask_excludes_from_training(tmp_path):
    # In 'mask' mode a truncated completion still gets reward 0 (informing the
    # group baseline) but is dropped from the training batch. Q1's completions
    # are truncated -> that whole group is fully masked -> excluded.
    cfg = _make_cfg()
    cfg["max_steps"] = 1
    cfg["train"]["truncation_mode"] = "mask"
    trainer, _ = _make_trainer(
        tmp_path, _training_problems(), _answers_with_eval(_variance_answers()),
        cfg=cfg, truncated_prompts=["Q1"],
    )
    trainer.train()
    row = _train_rows(tmp_path)[0]
    assert row["n_masked_truncated"] == 2  # Q1's G=2 completions both truncated
    assert row["n_completions"] == (4 - 1) * 2  # Q1's group fully masked out


def test_truncation_zero_reward_still_trains_on_truncated(tmp_path):
    # Default zero_reward mode: truncated completions are NOT masked (masked=False),
    # so they stay in the training batch and n_masked_truncated is 0.
    cfg = _make_cfg()
    cfg["max_steps"] = 1
    cfg["train"]["truncation_mode"] = "zero_reward"
    trainer, _ = _make_trainer(
        tmp_path, _training_problems(), _answers_with_eval(_variance_answers()),
        cfg=cfg, truncated_prompts=["Q1"],
    )
    trainer.train()
    row = _train_rows(tmp_path)[0]
    assert row["n_masked_truncated"] == 0
    assert row["frac_truncated"] > 0.0  # they ARE counted as truncated in stats


def test_invalid_truncation_mode_rejected(tmp_path):
    cfg = _make_cfg()
    cfg["train"]["truncation_mode"] = "bogus"
    with pytest.raises(AssertionError, match="truncation_mode"):
        _make_trainer(tmp_path, _training_problems(),
                      _answers_with_eval(_variance_answers()), cfg=cfg)


def test_lr_schedule_constant(tmp_path):
    cfg = _make_cfg()
    cfg["train"]["lr_schedule"] = "constant"
    cfg["train"]["lr"] = 2e-6
    trainer, _ = _make_trainer(
        tmp_path, _training_problems(), _answers_with_eval(_variance_answers()), cfg=cfg
    )
    assert trainer._current_lr(0) == 2e-6
    assert trainer._current_lr(cfg["max_steps"] - 1) == 2e-6


def test_lr_schedule_cosine_decays_to_floor(tmp_path):
    cfg = _make_cfg()
    cfg["max_steps"] = 100
    cfg["train"]["lr_schedule"] = "cosine"
    cfg["train"]["lr"] = 1e-6
    cfg["train"]["min_lr_ratio"] = 0.1
    trainer, _ = _make_trainer(
        tmp_path, _training_problems(), _answers_with_eval(_variance_answers()), cfg=cfg
    )
    assert trainer._current_lr(0) == pytest.approx(1e-6)  # starts at peak
    assert trainer._current_lr(99) == pytest.approx(0.1e-6)  # ends at floor
    mid = trainer._current_lr(50)
    assert 0.1e-6 < mid < 1e-6  # monotonically between
    # cosine midpoint (frac ~0.5) is ~ floor + 0.9*peak*0.5 = 0.55*peak
    assert mid == pytest.approx(1e-6 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 50 / 99))))


def test_lr_schedule_logged_in_metrics(tmp_path):
    cfg = _make_cfg()
    cfg["max_steps"] = 3
    cfg["train"]["lr_schedule"] = "cosine"
    cfg["train"]["lr"] = 1e-6
    trainer, _ = _make_trainer(
        tmp_path, _training_problems(), _answers_with_eval(_variance_answers()), cfg=cfg
    )
    trainer.train()
    lrs = [r["lr"] for r in _train_rows(tmp_path)]
    assert lrs[0] > lrs[-1]  # decaying across the run


def test_entropy_stop_loss_aborts_early(tmp_path):
    # A tiny threshold guarantees the first step's entropy exceeds it, so the
    # run must abort after step 1 (not run all 3), still logging + checkpointing.
    cfg = _make_cfg()
    cfg["max_steps"] = 3
    cfg["train"]["entropy_abort_threshold"] = 1e-6
    trainer, _ = _make_trainer(
        tmp_path, _training_problems(), _answers_with_eval(_variance_answers()), cfg=cfg
    )
    trainer.train()
    rows = _train_rows(tmp_path)
    assert [r["step"] for r in rows] == [1]  # aborted after the first step
    assert (tmp_path / "checkpoints" / "step_0001").exists()  # checkpointed on abort


def test_entropy_stop_loss_disabled_runs_full(tmp_path):
    cfg = _make_cfg()
    cfg["max_steps"] = 3
    cfg["train"]["entropy_abort_threshold"] = None
    trainer, _ = _make_trainer(
        tmp_path, _training_problems(), _answers_with_eval(_variance_answers()), cfg=cfg
    )
    trainer.train()
    assert [r["step"] for r in _train_rows(tmp_path)] == [1, 2, 3]


def test_step0_assertions_run_and_pass(tmp_path):
    cfg = _make_cfg()
    cfg["max_steps"] = 1
    trainer, _ = _make_trainer(
        tmp_path, _training_problems(), _answers_with_eval(_variance_answers()), cfg=cfg
    )
    trainer.train()  # step-0 ratio + denominator asserts must not raise
    assert trainer.step0_checks_ran is True


def test_checkpoint_written_at_save_every_and_final(tmp_path):
    # max_steps 3, save_every 2 -> save at step 2 (cadence) and step 3 (final).
    trainer, _ = _make_trainer(tmp_path, _training_problems(), _answers_with_eval(_variance_answers()))
    trainer.train()

    ckpt_root = tmp_path / "checkpoints"
    assert (ckpt_root / "step_0002").exists()
    assert (ckpt_root / "step_0003").exists()


def test_resume_continues_at_next_step(tmp_path):
    cfg = _make_cfg()
    cfg["max_steps"] = 2
    trainer, _ = _make_trainer(
        tmp_path, _training_problems(), _answers_with_eval(_variance_answers()), cfg=cfg
    )
    trainer.train()
    assert [r["step"] for r in _train_rows(tmp_path)] == [1, 2]

    cfg2 = _make_cfg()
    cfg2["max_steps"] = 4
    trainer2, _ = _make_trainer(
        tmp_path, _training_problems(), _answers_with_eval(_variance_answers()), cfg=cfg2, resume=True
    )
    trainer2.train()

    rows = _train_rows(tmp_path)
    assert [r["step"] for r in rows] == [1, 2, 3, 4]  # append, no dupes
    assert [r["epoch"] for r in rows] == [0, 1, 2, 3]  # resumed sampler continues


def test_eval_rows_written_at_cadence(tmp_path):
    # every_steps 2, max_steps 3 -> eval at steps 0 and 2.
    trainer, _ = _make_trainer(tmp_path, _training_problems(), _answers_with_eval(_variance_answers()))
    trainer.train()

    eval_rows = _eval_rows(tmp_path)
    assert [r["step"] for r in eval_rows] == [0, 2]
    for row in eval_rows:
        assert row["benchmark"] == "tiny_math"
        assert row["pass_at_1"] == pytest.approx(0.5)  # 3 correct of 6, k=1

    assert (tmp_path / "eval" / "step_0000" / "tiny_math" / "summary.json").exists()
    assert (tmp_path / "eval" / "step_0002" / "tiny_math" / "summary.json").exists()


def test_handshake_mismatch_raises(tmp_path):
    trainer, _ = _make_trainer(
        tmp_path,
        _training_problems(),
        _answers_with_eval(_variance_answers()),
        corrupt_checksums=True,
    )
    with pytest.raises(AssertionError):
        trainer.train()


def test_off_cadence_steps_skip_checksum_computation(tmp_path):
    # sync_check_every defaults to 25, well above max_steps (3), so only step
    # 0 is a checksum-check step; steps 1 and 2 must ask for (and get) an
    # empty checksum dict rather than hashing the full weight set.
    trainer, rollout = _make_trainer(
        tmp_path, _training_problems(), _answers_with_eval(_variance_answers())
    )
    trainer.train()

    assert rollout.want_checksums_calls == [True, False, False]


def test_resume_without_checkpoint_starts_fresh(tmp_path):
    # resume=True on an empty run_dir (no checkpoints written yet) must behave
    # exactly like a fresh run: no double-counted/duplicated metrics rows, and
    # config_resolved.yaml still gets written.
    cfg = _make_cfg()
    trainer, _ = _make_trainer(
        tmp_path, _training_problems(), _answers_with_eval(_variance_answers()), cfg=cfg, resume=True
    )
    trainer.train()

    rows = _train_rows(tmp_path)
    assert [r["step"] for r in rows] == [1, 2, 3]
    assert (tmp_path / "config_resolved.yaml").exists()


def test_resume_hf_format_without_override_raises(tmp_path):
    # A save_pretrained-format checkpoint (a model/ dir) plus resume=True but
    # no model_path_override must raise loudly rather than silently resuming
    # the optimizer/sampler/RNG state against stale (non-resumed) weights.
    ckpt_dir = tmp_path / "checkpoints" / "step_0001"
    (ckpt_dir / "model").mkdir(parents=True)
    (ckpt_dir / "trainer_state.json").write_text(
        json.dumps({"step": 1, "sampler": {"epoch": 0, "batch_idx": 1}, "run_name": "x"})
    )

    tiny_model = nn.Linear(2, 2)
    tiny_optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=0.1)
    tiny_optimizer.zero_grad()
    tiny_model(torch.randn(1, 2)).sum().backward()
    tiny_optimizer.step()
    torch.save(tiny_optimizer.state_dict(), ckpt_dir / "optimizer.pt")

    with pytest.raises(RuntimeError, match="model_path_override"):
        _make_trainer(
            tmp_path, _training_problems(), _answers_with_eval(_variance_answers()), resume=True
        )
