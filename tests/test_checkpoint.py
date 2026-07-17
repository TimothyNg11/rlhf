"""Tests for grpo_math.trainer.checkpoint -- save/restore of model,
optimizer, trainer state, and RNG state, plus the numeric step-dir lookup.
"""

import torch
import torch.nn as nn

from grpo_math.trainer.checkpoint import (
    find_latest_checkpoint,
    load_rng_state,
    load_trainer_state,
    save_checkpoint,
)


def _make_model_and_optimizer():
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    # A couple of real optimizer steps so exp_avg/exp_avg_sq state is
    # non-trivial (not just zero-initialized).
    for _ in range(2):
        optimizer.zero_grad()
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        optimizer.step()
    return model, optimizer


def test_save_and_restore_roundtrip(tmp_path):
    model, optimizer = _make_model_and_optimizer()
    saved_weight = model.weight.detach().clone()
    saved_bias = model.bias.detach().clone()
    saved_exp_avg = [
        state["exp_avg"].clone() for state in optimizer.state_dict()["state"].values()
    ]

    ckpt_dir = save_checkpoint(
        tmp_path / "checkpoints",
        step=5,
        model=model,
        optimizer=optimizer,
        trainer_state={"step": 5},
    )

    # Mutate the live model params so a naive test would fail if reload
    # didn't actually restore anything.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    fresh_model = nn.Linear(4, 3)
    fresh_model.load_state_dict(torch.load(ckpt_dir / "model.pt", weights_only=False))
    assert torch.equal(fresh_model.weight, saved_weight)
    assert torch.equal(fresh_model.bias, saved_bias)

    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=0.1)
    fresh_optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", weights_only=False))
    loaded_exp_avg = [
        state["exp_avg"] for state in fresh_optimizer.state_dict()["state"].values()
    ]
    assert len(loaded_exp_avg) == len(saved_exp_avg)
    for loaded, saved in zip(loaded_exp_avg, saved_exp_avg):
        assert torch.equal(loaded, saved)


def test_find_latest_numeric_order(tmp_path):
    ckpt_root = tmp_path / "checkpoints"
    for name in ["step_0080", "step_0160", "step_1000", "step_200"]:
        (ckpt_root / name).mkdir(parents=True)

    # The trap this test guards against: lexicographically "step_1000" <
    # "step_200" (the first differing character is '1' vs '2'), even though
    # 1000 > 200 numerically. A lexicographic max() would wrongly pick
    # step_200 as "latest". Numerically, step_200 (200) also correctly
    # beats step_0160 (160) despite being less zero-padded.
    assert "step_1000" < "step_200"

    latest = find_latest_checkpoint(ckpt_root)
    assert latest is not None
    assert latest.name == "step_1000"


def test_find_latest_none_when_empty(tmp_path):
    ckpt_root = tmp_path / "checkpoints"
    assert find_latest_checkpoint(ckpt_root) is None

    ckpt_root.mkdir()
    assert find_latest_checkpoint(ckpt_root) is None


def test_trainer_state_roundtrip(tmp_path):
    model, optimizer = _make_model_and_optimizer()
    trainer_state = {
        "step": 42,
        "epoch": 3,
        "sampler_state": {"seed": 7, "indices_consumed": [1, 2, 3], "exhausted": False},
    }

    ckpt_dir = save_checkpoint(
        tmp_path / "checkpoints",
        step=42,
        model=model,
        optimizer=optimizer,
        trainer_state=trainer_state,
    )

    assert load_trainer_state(ckpt_dir) == trainer_state


def test_rng_state_roundtrip(tmp_path):
    model, optimizer = _make_model_and_optimizer()

    torch.manual_seed(123)
    rng_state = {"cpu": torch.get_rng_state()}
    expected_draws = torch.rand(5)

    ckpt_dir = save_checkpoint(
        tmp_path / "checkpoints",
        step=1,
        model=model,
        optimizer=optimizer,
        trainer_state={},
        rng_state=rng_state,
    )

    # Advance the global RNG so a failure to restore would be visible.
    torch.rand(100)

    loaded_rng_state = load_rng_state(ckpt_dir)
    assert loaded_rng_state is not None
    torch.set_rng_state(loaded_rng_state["cpu"])
    actual_draws = torch.rand(5)
    assert torch.equal(actual_draws, expected_draws)


def test_interrupted_save_not_picked_up(tmp_path):
    ckpt_root = tmp_path / "checkpoints"
    ckpt_root.mkdir()

    # Simulate a crash mid-save: a leftover temp dir using save_checkpoint's
    # own temp-name convention (final step dir name + ".tmp").
    (ckpt_root / "step_0100.tmp").mkdir()
    assert find_latest_checkpoint(ckpt_root) is None

    (ckpt_root / "step_0050").mkdir()
    latest = find_latest_checkpoint(ckpt_root)
    assert latest is not None
    assert latest.name == "step_0050"
