"""Tests for grpo_math.data.gsm8k: split_holdout contract + PromptSampler.

``load_gsm8k_train`` itself is NOT exercised here -- it lazy-imports
``datasets``, which is genuinely absent from this venv (see
tests/test_backends.py's analogous no-vllm guarantee). Only the pure
list/numpy pieces and the module's import-without-datasets guarantee are
tested.
"""

from grpo_math.data.gsm8k import PromptSampler, split_holdout


# --- split_holdout ----------------------------------------------------------


def test_split_holdout_sizes_and_disjoint():
    items = list(range(20))
    train, dev = split_holdout(items, 5)
    assert len(train) == 15
    assert len(dev) == 5
    assert set(train).isdisjoint(dev)
    assert train + dev == items


def test_dev_is_last_n():
    items = list(range(20))
    train, dev = split_holdout(items, 5)
    assert dev == items[-5:]
    assert train == items[:-5]


def test_split_matches_benchmark_take_last_convention():
    # Pins the contract: split_holdout's dev slice is exactly the same
    # tail-slice convention load_benchmark applies for BenchmarkSpec.take_last
    # (see grpo_math.eval.benchmarks.load_benchmark).
    items = [f"item_{i}" for i in range(37)]
    n = 9
    assert split_holdout(items, n)[1] == items[-n:]


# --- PromptSampler ------------------------------------------------------------


def test_sampler_deterministic_given_seed():
    s1 = PromptSampler(20, 4, seed=42)
    s2 = PromptSampler(20, 4, seed=42)
    batches1 = [s1.next_batch() for _ in range(5)]
    batches2 = [s2.next_batch() for _ in range(5)]
    assert batches1 == batches2


def test_sampler_epoch_permutation_no_replacement():
    sampler = PromptSampler(20, 4, seed=0)
    seen = []
    for _ in range(5):  # 20 / 4 = exactly one epoch's worth of batches
        seen.extend(sampler.next_batch())
    assert sorted(seen) == list(range(20))


def test_sampler_resume_mid_epoch_continues_identically():
    sampler = PromptSampler(23, 4, seed=7)
    for _ in range(3):
        sampler.next_batch()
    state = sampler.state_dict()
    continued = [sampler.next_batch() for _ in range(4)]

    resumed = PromptSampler(23, 4, seed=7)
    resumed.load_state_dict(state)
    resumed_batches = [resumed.next_batch() for _ in range(4)]

    assert continued == resumed_batches


def test_sampler_drops_partial_tail_batch():
    sampler = PromptSampler(10, 4, seed=1)  # 10 is not divisible by 4
    batch_sizes = [len(sampler.next_batch()) for _ in range(6)]
    assert all(size == 4 for size in batch_sizes)
    assert sampler.epoch >= 1  # the 2-item tail was dropped, rolling the epoch


def test_import_data_gsm8k_without_datasets():
    # module-level import must never require `datasets`; this test is
    # meaningful because `datasets` is genuinely absent from this venv.
    import grpo_math.data.gsm8k  # noqa: F401
