"""Golden, hand-computed tests for grpo_math.trainer.algo -- the pure-torch
GRPO math core (group-normalized advantages, k3 KL, logprob/entropy
gathering, batch collation, and the clipped PG+KL micro-batch loss).
"""

import math

import pytest
import torch
import torch.nn as nn

from grpo_math.trainer.algo import (
    MicrobatchLossOut,
    TokenBatch,
    collate_token_batch,
    gather_logprobs_and_entropy,
    grpo_microbatch_loss,
    group_normalized_advantages,
    low_var_kl,
)


# --- group_normalized_advantages ---


def test_group_norm_advantages_golden():
    # group0 = [1, 0, 0, 0]: mean = 0.25
    #   diffs: 0.75, -0.25, -0.25, -0.25; squared: 0.5625, 0.0625, 0.0625, 0.0625
    #   sum = 0.75; sample var (n-1=3) = 0.75 / 3 = 0.25 -> std = 0.5
    # group1 = [1, 1, 0, 0]: mean = 0.5
    #   diffs: 0.5, 0.5, -0.5, -0.5; squared: 0.25 each; sum = 1.0
    #   sample var (n-1=3) = 1.0 / 3 -> std = sqrt(1/3) ~= 0.5773502691896258
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    advantages, keep_mask = group_normalized_advantages(rewards, std_eps=0.0)

    std0 = 0.5
    std1 = math.sqrt(1.0 / 3.0)
    expected = torch.tensor(
        [
            [(1.0 - 0.25) / std0, (0.0 - 0.25) / std0, (0.0 - 0.25) / std0, (0.0 - 0.25) / std0],
            [(1.0 - 0.5) / std1, (1.0 - 0.5) / std1, (0.0 - 0.5) / std1, (0.0 - 0.5) / std1],
        ]
    )
    assert torch.allclose(advantages, expected)
    assert torch.equal(keep_mask, torch.tensor([True, True]))


def test_advantage_std_eps_exact():
    # single group [1., 0.]: mean = 0.5
    #   diffs: 0.5, -0.5; squared 0.25 each; sum = 0.5
    #   sample var (n-1=1) = 0.5 / 1 = 0.5 -> std = sqrt(0.5) ~= 0.7071067811865476
    # std_eps = 0.5 -> denom = std + 0.5
    rewards = torch.tensor([[1.0, 0.0]])
    advantages, keep_mask = group_normalized_advantages(rewards, std_eps=0.5)

    std = math.sqrt(0.5)
    denom = std + 0.5
    expected = torch.tensor([[0.5 / denom, -0.5 / denom]])
    # Default tensor dtype here is float32 (~7 decimal digits); use torch's
    # default allclose tolerance rather than an fp64-grade one.
    assert torch.allclose(advantages, expected)
    assert torch.equal(keep_mask, torch.tensor([True]))


def test_zero_variance_group_flagged():
    # group0 = [1, 1, 1, 1]: zero variance -> std = 0 -> keep_mask False.
    # group1 = [1, 0, 1, 0]: nonzero variance -> keep_mask True.
    rewards = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
    advantages, keep_mask = group_normalized_advantages(rewards)
    assert torch.equal(keep_mask, torch.tensor([False, True]))
    # Zero-variance group's numerator is exactly zero, so advantages are all 0
    # regardless of std_eps.
    assert torch.allclose(advantages[0], torch.zeros(4))


# --- low_var_kl ---


def test_low_var_kl_golden():
    # lp = log(0.5), ref = log(0.25) -> delta = ref - lp = log(0.5)
    # k3 = exp(delta) - delta - 1 = 0.5 - log(0.5) - 1 = 0.5 + log(2) - 1
    lp = torch.log(torch.tensor(0.5))
    ref = torch.log(torch.tensor(0.25))
    result = low_var_kl(lp, ref)
    expected = 0.5 + math.log(2.0) - 1.0
    assert result.item() == pytest.approx(expected)


def test_low_var_kl_zero_when_equal():
    lp = torch.tensor([0.1, -2.0, 3.5, 0.0])
    result = low_var_kl(lp, lp.clone())
    assert torch.allclose(result, torch.zeros_like(result))


def test_low_var_kl_nonnegative():
    # k3(x) = exp(x) - x - 1 >= 0 for all real x (equality only at x=0), so
    # this must hold for arbitrary logprob/ref_logprob pairs.
    torch.manual_seed(0)
    lp = torch.randn(200) * 3
    ref = torch.randn(200) * 3
    result = low_var_kl(lp, ref)
    assert torch.all(result >= -1e-6)


# --- gather_logprobs_and_entropy ---


def test_gather_logprobs_and_entropy_manual_tiny_vocab():
    # logits = log([0.2, 0.3, 0.5]); since these probabilities already sum to
    # 1, softmax(log(p)) == p exactly: exp(log p_i) / sum(exp(log p)) = p_i / 1.
    # target = 2 -> chosen logprob = log(0.5).
    # entropy = -(0.2*log0.2 + 0.3*log0.3 + 0.5*log0.5)
    #         = 0.2*1.6094379124341003 + 0.3*1.2039728043259361 + 0.5*0.6931471805599453
    #         ~= 0.32188758 + 0.36119184 + 0.34657359 = 1.02965301...
    logits = torch.tensor([[[math.log(0.2), math.log(0.3), math.log(0.5)]]])  # [1, 1, 3]
    targets = torch.tensor([[2]])  # [1, 1]

    logprobs, entropy = gather_logprobs_and_entropy(logits, targets)

    expected_logprob = math.log(0.5)
    expected_entropy = -(0.2 * math.log(0.2) + 0.3 * math.log(0.3) + 0.5 * math.log(0.5))
    assert logprobs.item() == pytest.approx(expected_logprob)
    assert entropy.item() == pytest.approx(expected_entropy)


def test_entropy_uniform_logits_is_log_v():
    # All-zero logits -> uniform softmax over V=7 -> entropy = ln(7).
    logits = torch.zeros(2, 3, 7)
    targets = torch.zeros(2, 3, dtype=torch.long)
    _, entropy = gather_logprobs_and_entropy(logits, targets)
    assert torch.allclose(entropy, torch.full((2, 3), math.log(7.0)), atol=1e-6)


# --- collate_token_batch ---


def test_collate_token_batch_masks_golden():
    # sample0: prompt [1, 2] + response [3, 4, 5] -> length 5 (no padding needed)
    # sample1: prompt [6] + response [7] -> length 2, padded to 5 with pad_id 9
    samples = [([1, 2], [3, 4, 5]), ([6], [7])]
    batch = collate_token_batch(samples, pad_token_id=9)

    expected_input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 9, 9, 9],
        ]
    )
    expected_attention_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, False, False, False],
        ]
    )
    expected_response_mask = torch.tensor(
        [
            [False, False, True, True, True],
            [False, True, False, False, False],
        ]
    )
    assert isinstance(batch, TokenBatch)
    assert torch.equal(batch.input_ids, expected_input_ids)
    assert torch.equal(batch.attention_mask, expected_attention_mask)
    assert torch.equal(batch.response_mask, expected_response_mask)


# --- grpo_microbatch_loss ---


def test_clip_loss_golden():
    # Two 1-token sequences (B=2, T=1):
    #   seq0: old_lp=0, lp=log(1.5) -> ratio=1.5, A=+1, clip_ratio=0.2
    #     clip band [0.8, 1.2]; ratio > 1.2 and A > 0 -> clip binds
    #     pg = -min(1.5*1, 1.2*1) = -1.2
    #   seq1: old_lp=0, lp=log(0.5) -> ratio=0.5, A=-1
    #     ratio < 0.8 and A < 0 -> clip binds
    #     pg = -min(0.5*-1, 0.8*-1) = -min(-0.5, -0.8) = 0.8
    # kl_coef=0, global_token_count=2 -> loss = (-1.2 + 0.8) / 2 = -0.2
    logprobs = torch.tensor([[math.log(1.5)], [math.log(0.5)]])
    old_logprobs = torch.zeros(2, 1)
    ref_logprobs = torch.zeros(2, 1)
    advantages = torch.tensor([1.0, -1.0])
    response_mask = torch.ones(2, 1, dtype=torch.bool)

    out = grpo_microbatch_loss(
        logprobs,
        old_logprobs,
        ref_logprobs,
        advantages,
        response_mask,
        clip_ratio=0.2,
        kl_coef=0.0,
        global_token_count=2,
    )
    assert isinstance(out, MicrobatchLossOut)
    assert out.loss.item() == pytest.approx(-0.2)
    assert out.clip_count == 2
    assert out.n_tokens == 2


def test_clip_not_binding_inside_band():
    # ratio=1.1 is inside [0.8, 1.2], A=+1 -> pg = -min(1.1, 1.1) = -1.1, no clip.
    logprobs = torch.tensor([[math.log(1.1)]])
    old_logprobs = torch.zeros(1, 1)
    ref_logprobs = torch.zeros(1, 1)
    advantages = torch.tensor([1.0])
    response_mask = torch.ones(1, 1, dtype=torch.bool)

    out = grpo_microbatch_loss(
        logprobs,
        old_logprobs,
        ref_logprobs,
        advantages,
        response_mask,
        clip_ratio=0.2,
        kl_coef=0.0,
        global_token_count=1,
    )
    assert out.loss.item() == pytest.approx(-1.1)
    assert out.clip_count == 0


def test_loss_at_ratio_one_reduces_to_minus_advantage():
    # logprobs == old_logprobs -> ratio == 1 exactly, which is always inside
    # the clip band, so pg = -min(A, A) = -A for every masked token. With
    # kl_coef=0, loss == -(sum of A over masked tokens) / global_token_count.
    torch.manual_seed(1)
    logprobs = torch.randn(3, 4)
    old_logprobs = logprobs.clone()
    ref_logprobs = torch.randn(3, 4)
    advantages = torch.tensor([0.5, -1.5, 2.0])
    response_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, False, False, False],
            [True, True, True, False],
        ]
    )

    out = grpo_microbatch_loss(
        logprobs,
        old_logprobs,
        ref_logprobs,
        advantages,
        response_mask,
        clip_ratio=0.2,
        kl_coef=0.0,
        global_token_count=6,
    )
    # sum of A over masked tokens: row0 has 2 masked tokens (0.5 each occurrence),
    # row1 has 1 (-1.5), row2 has 3 (2.0 each) -> 0.5*2 + (-1.5)*1 + 2.0*3 = 1.0 - 1.5 + 6.0 = 5.5
    expected_sum_a = 0.5 * 2 + (-1.5) * 1 + 2.0 * 3
    assert out.loss.item() == pytest.approx(-expected_sum_a / 6)


def test_masking_excludes_padded_tokens():
    torch.manual_seed(2)
    logprobs = torch.randn(2, 4)
    old_logprobs = torch.randn(2, 4)
    ref_logprobs = torch.randn(2, 4)
    advantages = torch.tensor([0.7, -0.3])
    response_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, False, False, True],
        ]
    )

    kwargs = dict(clip_ratio=0.2, kl_coef=0.1, global_token_count=4)
    out_clean = grpo_microbatch_loss(
        logprobs, old_logprobs, ref_logprobs, advantages, response_mask, **kwargs
    )

    # Same values at masked-in positions, garbage at masked-out positions.
    # Garbage is large but finite (diff ~20 -> exp(20) ~ 4.85e8, well inside
    # float32 range) -- multiplying by mask=0 must still zero it out exactly;
    # an overflowing (inf) garbage value would turn 0 * inf into nan, which
    # would be a bug in the *test*, not evidence about masking correctness.
    logprobs_garbage = logprobs.clone()
    old_logprobs_garbage = old_logprobs.clone()
    ref_logprobs_garbage = ref_logprobs.clone()
    garbage_slots = ~response_mask
    logprobs_garbage[garbage_slots] = 10.0
    old_logprobs_garbage[garbage_slots] = -10.0
    ref_logprobs_garbage[garbage_slots] = 5.0

    out_garbage = grpo_microbatch_loss(
        logprobs_garbage,
        old_logprobs_garbage,
        ref_logprobs_garbage,
        advantages,
        response_mask,
        **kwargs,
    )

    assert out_garbage.loss.item() == pytest.approx(out_clean.loss.item())
    assert out_garbage.n_tokens == out_clean.n_tokens
    assert out_garbage.pg_loss_sum == pytest.approx(out_clean.pg_loss_sum)
    assert out_garbage.kl_sum == pytest.approx(out_clean.kl_sum)
    assert out_garbage.clip_count == out_clean.clip_count
    assert out_garbage.ratio_sum == pytest.approx(out_clean.ratio_sum)
    assert out_garbage.ratio_max == pytest.approx(out_clean.ratio_max)


def test_grad_accum_matches_full_batch_gradients():
    """The single most important test in this project: gradient accumulation
    over micro-batches (same global_token_count passed to every call) must be
    numerically identical to a single full-batch backward call.

    Runs a tiny linear LM (Embedding + Linear, float64) over 6 variable-length
    sequences two ways:
      1. One grpo_microbatch_loss(..., global_token_count=N_total).backward()
         over all 6 sequences at once.
      2. Three separate 2-sequence micro-batches, grads zeroed once up front,
         each calling grpo_microbatch_loss(..., global_token_count=N_total)
         .backward() in turn so gradients accumulate.
    Both passes route forward through gather_logprobs_and_entropy (the real
    path), so gradients flow through the model exactly as they would in
    training. float64 throughout so any discrepancy beyond ordinary
    floating-point summation-order noise (~1e-13ish) is visible immediately.
    """
    torch.manual_seed(0)
    vocab_size = 13
    pad_token_id = 12

    samples = [
        ([1, 2], [3, 4]),
        ([5], [6, 7, 8]),
        ([2, 3, 4], [5]),
        ([1], [2]),
        ([6, 7], [8, 9, 10]),
        ([3], [4, 5, 6, 7]),
    ]

    embed = nn.Embedding(vocab_size, 8, dtype=torch.float64)
    linear = nn.Linear(8, vocab_size, dtype=torch.float64)
    params = list(embed.parameters()) + list(linear.parameters())

    def forward_logprobs(batch: TokenBatch):
        hidden = embed(batch.input_ids)
        logits = linear(hidden)
        logprobs, _ = gather_logprobs_and_entropy(logits[:, :-1], batch.input_ids[:, 1:])
        return logprobs, batch.response_mask[:, 1:]

    full_batch = collate_token_batch(samples, pad_token_id=pad_token_id)
    with torch.no_grad():
        _, full_resp_mask = forward_logprobs(full_batch)
    global_token_count = int(full_resp_mask.sum().item())
    t_full = full_resp_mask.shape[1]

    # Fixed "fake" old/ref logprobs and advantages, independent of the model.
    # Small scale keeps ratios well-behaved; padded-tail columns beyond a given
    # micro-batch's own T are never read (see slicing below), so their values
    # don't matter for the comparison.
    old_logprobs_full = torch.randn(6, t_full, dtype=torch.float64) * 0.1
    ref_logprobs_full = torch.randn(6, t_full, dtype=torch.float64) * 0.1
    advantages_full = torch.randn(6, dtype=torch.float64)

    def run_loss(logprobs, old_lp, ref_lp, adv, resp_mask):
        return grpo_microbatch_loss(
            logprobs,
            old_lp,
            ref_lp,
            adv,
            resp_mask,
            clip_ratio=0.2,
            kl_coef=0.1,
            global_token_count=global_token_count,
        )

    # Pass 1: single full-batch backward.
    for p in params:
        p.grad = None
    logprobs_full, resp_mask_full = forward_logprobs(full_batch)
    out_full = run_loss(
        logprobs_full, old_logprobs_full, ref_logprobs_full, advantages_full, resp_mask_full
    )
    out_full.loss.backward()
    grads_full = [p.grad.clone() for p in params]

    # Pass 2: three 2-sequence micro-batches, grads accumulate via repeated backward().
    for p in params:
        p.grad = None
    for start in range(0, 6, 2):
        mb_samples = samples[start : start + 2]
        mb_batch = collate_token_batch(mb_samples, pad_token_id=pad_token_id)
        logprobs_mb, resp_mask_mb = forward_logprobs(mb_batch)
        t_mb = resp_mask_mb.shape[1]
        out_mb = run_loss(
            logprobs_mb,
            old_logprobs_full[start : start + 2, :t_mb],
            ref_logprobs_full[start : start + 2, :t_mb],
            advantages_full[start : start + 2],
            resp_mask_mb,
        )
        out_mb.loss.backward()
    grads_accum = [p.grad.clone() for p in params]

    assert len(grads_full) == len(grads_accum) > 0
    for g_full, g_accum in zip(grads_full, grads_accum):
        assert torch.allclose(g_full, g_accum, rtol=1e-12, atol=1e-12)
