"""Tests for grpo_math.trainer.policy -- the CPU-testable, torch-only
helpers (response_logprobs, bf16_sync_tensors, model_checksums) plus an
import guard for the lazy-transformers HFPolicy wrapper.
"""

import torch
import torch.nn as nn

from grpo_math.trainer.algo import gather_logprobs_and_entropy
from grpo_math.trainer.policy import bf16_sync_tensors, model_checksums, response_logprobs

PAD_TOKEN_ID = 12


class TinyLM(nn.Module):
    """nn.Embedding(13, 8) + nn.Linear(8, 13); forward returns raw logits
    (not wrapped in an object with a .logits attribute), exercising the
    ``getattr(out, "logits", out)`` fallback in response_logprobs."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(13, 8)
        self.linear = nn.Linear(8, 13)

    def forward(self, input_ids, attention_mask=None):
        return self.linear(self.embed(input_ids))


def _naive_response_logprobs(model, samples, compute_entropy=False):
    """Unbatched, unpadded reference: score each sample's own
    (prompt+response) tensor individually at batch size 1, with no padding
    at all -- an independent re-derivation of the shift convention, not a
    reuse of collate_token_batch/response_logprobs."""
    logprobs = []
    entropies = [] if compute_entropy else None
    for prompt_ids, response_ids in samples:
        ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long)
        attention_mask = torch.ones_like(ids, dtype=torch.bool)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=attention_mask)
        targets = ids[:, 1:]
        lp, ent = gather_logprobs_and_entropy(logits[:, :-1], targets)
        p_len, r_len = len(prompt_ids), len(response_ids)
        logprobs.append(lp[0, p_len - 1 : p_len - 1 + r_len])
        if compute_entropy:
            entropies.append(ent[0, p_len - 1 : p_len - 1 + r_len])
    return logprobs, entropies


# Deliberately ragged, non-monotonic total lengths (4, 6, 4, 3, 8) so that
# sorting by length (desc) actually reorders samples and restoring to
# original order actually un-reorders them.
_RAGGED_SAMPLES = [
    ([1, 2], [3, 4]),  # total 4
    ([5], [6, 7, 8, 9, 10]),  # total 6
    ([1, 2, 3], [4]),  # total 4
    ([2], [3, 4]),  # total 3
    ([5, 6, 7, 8], [9, 10, 11, 3]),  # total 8
]


def test_response_logprobs_matches_naive_per_sample():
    torch.manual_seed(42)
    model = TinyLM()

    naive_lp, _ = _naive_response_logprobs(model, _RAGGED_SAMPLES)
    batched_lp, batched_ent = response_logprobs(
        model, _RAGGED_SAMPLES, micro_batch_size=2, pad_token_id=PAD_TOKEN_ID
    )

    assert batched_ent is None
    assert len(batched_lp) == len(_RAGGED_SAMPLES)
    for i, (naive, batched) in enumerate(zip(naive_lp, batched_lp)):
        assert batched.shape == naive.shape == (len(_RAGGED_SAMPLES[i][1]),)
        assert torch.allclose(batched, naive, atol=1e-5)


def test_response_logprobs_order_restored():
    torch.manual_seed(99)
    model = TinyLM()

    # Distinctive per-sample content (no shared token subsequences) and
    # shuffled lengths, so a swapped-order bug would produce a shape or
    # value mismatch against the naive per-sample reference at some index.
    samples = [
        ([1, 2, 3, 4], [5]),  # total 5
        ([1], [2]),  # total 2
        ([6, 7], [8, 9, 10, 11]),  # total 6
        ([3, 4, 5], [6, 7]),  # total 5
        ([2, 3], [4]),  # total 3
        ([1, 2], [3, 4, 5, 6, 7]),  # total 7
    ]
    naive_lp, _ = _naive_response_logprobs(model, samples)
    batched_lp, _ = response_logprobs(model, samples, micro_batch_size=2, pad_token_id=PAD_TOKEN_ID)

    assert len(batched_lp) == len(samples)
    for i, (naive, batched) in enumerate(zip(naive_lp, batched_lp)):
        assert batched.shape == (len(samples[i][1]),)
        assert torch.allclose(batched, naive, atol=1e-5)


def test_response_logprobs_micro_batch_size_invariant():
    torch.manual_seed(7)
    model = TinyLM()

    results = {}
    for mb in (1, 3, len(_RAGGED_SAMPLES)):
        lp, ent = response_logprobs(
            model,
            _RAGGED_SAMPLES,
            micro_batch_size=mb,
            pad_token_id=PAD_TOKEN_ID,
            compute_entropy=True,
        )
        results[mb] = (lp, ent)

    base_lp, base_ent = results[1]
    for mb in (3, len(_RAGGED_SAMPLES)):
        lp, ent = results[mb]
        for i in range(len(_RAGGED_SAMPLES)):
            assert torch.allclose(lp[i], base_lp[i], atol=1e-6)
            assert torch.allclose(ent[i], base_ent[i], atol=1e-6)


def test_entropy_returned_when_requested():
    torch.manual_seed(3)
    model = TinyLM()
    samples = [([1, 2], [3, 4, 5]), ([2], [3])]

    lp, ent = response_logprobs(
        model, samples, micro_batch_size=2, pad_token_id=PAD_TOKEN_ID, compute_entropy=True
    )
    assert ent is not None
    assert len(ent) == len(samples)
    for e, (_, response_ids) in zip(ent, samples):
        assert e.shape == (len(response_ids),)

    lp2, ent2 = response_logprobs(
        model, samples, micro_batch_size=2, pad_token_id=PAD_TOKEN_ID, compute_entropy=False
    )
    assert ent2 is None
    assert len(lp2) == len(samples)


def test_bf16_sync_tensors_dtype_and_contiguity():
    model = TinyLM()
    seen_names = set()
    for name, tensor in bf16_sync_tensors(model):
        assert tensor.dtype == torch.bfloat16
        assert tensor.is_contiguous()
        seen_names.add(name)

    expected_names = {name for name, _ in model.named_parameters()}
    assert seen_names == expected_names


def test_model_checksums_stable_and_sensitive():
    model = TinyLM()
    checksums1 = model_checksums(model)
    checksums2 = model_checksums(model)
    assert checksums1 == checksums2

    name0 = next(iter(checksums1))
    param = dict(model.named_parameters())[name0]
    with torch.no_grad():
        param.add_(1.0)

    checksums3 = model_checksums(model)
    assert checksums3[name0] != checksums1[name0]
    for name in checksums1:
        if name != name0:
            assert checksums3[name] == checksums1[name]


def test_import_policy_without_transformers():
    # module-level import must never require transformers; this test is
    # meaningful because transformers is genuinely absent from this venv.
    import grpo_math.trainer.policy  # noqa: F401
