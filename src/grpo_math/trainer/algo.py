"""Pure-torch GRPO math core: group-normalized advantages, the k3 KL estimator,
token log-prob/entropy gathering, prompt+response collation, and the clipped
PG+KL micro-batch loss.

Imports only ``torch`` + stdlib so this module can be unit-tested (and later
run inside the training loop) without the ``gpu``-extra dependencies
(vllm/transformers/datasets) that the rest of the trainer needs. See
docs/PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def group_normalized_advantages(
    rewards: torch.Tensor, *, std_eps: float = 1e-4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group reward normalization: ``advantages[g, i] = (rewards[g, i] -
    mean_g) / (std_g + std_eps)``, with ``mean_g``/``std_g`` computed per-group
    (``rewards`` is ``[n_groups, G]``, reduced over ``dim=-1``).

    Std uses ``correction=1`` (Bessel-corrected sample std) -- this matches
    TRL's GRPOTrainer (``rewards.view(-1, G).std(dim=1)``, torch's default),
    which we validate against at gate G1 (docs/PLAN.md).

    Returns ``(advantages [n_groups, G], keep_mask [n_groups] bool)`` where
    ``keep_mask[g] = std_g > 0``. A zero-variance group (all rewards in the
    group identical) carries no learning signal; its ``advantages`` row is all
    zeros by construction (the numerator is exactly zero), but callers MUST
    drop it via ``keep_mask`` rather than relying on that.
    """
    mean = rewards.mean(dim=-1)
    std = rewards.std(dim=-1, correction=1)
    advantages = (rewards - mean[:, None]) / (std[:, None] + std_eps)
    keep_mask = std > 0
    return advantages, keep_mask


def low_var_kl(logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
    """Elementwise k3 KL estimator (Schulman's low-variance, unbiased,
    non-negative KL approximation): ``exp(ref - lp) - (ref - lp) - 1``.

    Shape-preserving: works on any matching-shape tensors.
    """
    delta = ref_logprobs - logprobs
    return torch.exp(delta) - delta - 1


def gather_logprobs_and_entropy(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-level log-prob of ``targets`` under ``logits``, plus the
    full-distribution entropy at each position.

    ``logits`` is ``[B, T, V]`` and is cast to fp32 FIRST (regardless of input
    dtype), then ``log_softmax(dim=-1)`` is applied. ``targets`` is ``[B, T]``
    long.

    NOTE: caller is responsible for the causal shift -- logits at position t
    predict the token at position t+1. This function does no shifting; see
    :func:`collate_token_batch` for the exact shift convention consumers use.

    Returns ``(chosen_logprobs [B, T], entropy [B, T])`` where
    ``chosen_logprobs`` is gathered at ``targets`` and
    ``entropy = -(softmax * log_softmax).sum(-1)``.
    """
    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    chosen_logprobs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return chosen_logprobs, entropy


@dataclass(frozen=True)
class TokenBatch:
    input_ids: torch.Tensor  # [B, L] long, right-padded
    attention_mask: torch.Tensor  # [B, L] bool, True on real (non-pad) tokens
    response_mask: torch.Tensor  # [B, L] bool, True ONLY on response-token positions


def collate_token_batch(
    samples: list[tuple[list[int], list[int]]], pad_token_id: int
) -> TokenBatch:
    """Right-pad ``(prompt_ids, response_ids)`` pairs into a :class:`TokenBatch`.

    ``input_ids[b] = prompt_ids + response_ids + padding`` out to the batch's
    max total (prompt + response) length.

    Shift convention consumers must use to score response tokens: feed
    ``input_ids`` through the model, take ``logits[:, :-1]`` against
    ``targets = input_ids[:, 1:]``, and mask with ``response_mask[:, 1:]``
    (the logit at position t predicts the token at position t+1).
    """
    lengths = [len(prompt_ids) + len(response_ids) for prompt_ids, response_ids in samples]
    max_len = max(lengths, default=0)
    batch_size = len(samples)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    response_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for b, (prompt_ids, response_ids) in enumerate(samples):
        p_len, r_len = len(prompt_ids), len(response_ids)
        seq_len = p_len + r_len
        input_ids[b, :seq_len] = torch.tensor(prompt_ids + response_ids, dtype=torch.long)
        attention_mask[b, :seq_len] = True
        response_mask[b, p_len:seq_len] = True

    return TokenBatch(
        input_ids=input_ids, attention_mask=attention_mask, response_mask=response_mask
    )


@dataclass(frozen=True)
class MicrobatchLossOut:
    loss: torch.Tensor  # scalar, ALREADY divided by global_token_count -- caller calls .backward() directly
    n_tokens: int  # masked (response) tokens in this micro-batch
    pg_loss_sum: float  # sum over masked tokens of the pg term (detached)
    kl_sum: float  # sum over masked tokens of the k3 term (detached)
    clip_count: int  # masked tokens where the clip binds (definition below)
    ratio_sum: float  # sum of ratio over masked tokens (detached)
    ratio_max: float  # max ratio over masked tokens (0.0 if n_tokens == 0)


def grpo_microbatch_loss(
    logprobs: torch.Tensor,  # [B, T] (requires grad)
    old_logprobs: torch.Tensor,  # [B, T] (no grad)
    ref_logprobs: torch.Tensor,  # [B, T] (no grad)
    advantages: torch.Tensor,  # [B] -- broadcast over tokens
    response_mask: torch.Tensor,  # [B, T] bool
    *,
    clip_ratio: float,
    kl_coef: float,
    global_token_count: int,
) -> MicrobatchLossOut:
    """Clipped GRPO policy-gradient loss + k3 KL penalty for one micro-batch.

    ``ratio = exp(logprobs - old_logprobs)``; ``A = advantages[:, None]``
    (broadcast over the token dimension). The per-token PG term is
    ``pg = -min(ratio * A, clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * A)``
    (elementwise min), and ``per_token = pg + kl_coef * low_var_kl(logprobs,
    ref_logprobs)``.

    ``loss = (per_token * response_mask).sum() / global_token_count``.
    ``global_token_count`` is the TOTAL masked-token count across the WHOLE
    mini-batch (all micro-batches), NOT just this one -- passing the same
    ``global_token_count`` to every micro-batch in a mini-batch and calling
    ``.backward()`` on each ``loss`` in turn accumulates gradients that are
    exactly equal to a single full-batch token-mean loss's gradient (see
    tests/test_algo.py::test_grad_accum_matches_full_batch_gradients).

    ``clip_count`` counts masked tokens where clipping binds:
    ``((ratio > 1 + clip_ratio) & (A > 0)) | ((ratio < 1 - clip_ratio) & (A < 0))``.

    All fields besides ``loss`` are computed under ``torch.no_grad()`` /
    detached to plain Python numbers; only ``loss`` carries a gradient.
    """
    ratio = torch.exp(logprobs - old_logprobs)
    advantages = advantages[:, None]
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    pg = -torch.min(ratio * advantages, clipped_ratio * advantages)
    kl = low_var_kl(logprobs, ref_logprobs)
    per_token = pg + kl_coef * kl

    mask = response_mask.to(per_token.dtype)
    loss = (per_token * mask).sum() / global_token_count

    with torch.no_grad():
        n_tokens = int(response_mask.sum().item())
        clip_binds = ((ratio > 1.0 + clip_ratio) & (advantages > 0)) | (
            (ratio < 1.0 - clip_ratio) & (advantages < 0)
        )
        clip_count = int((clip_binds & response_mask).sum().item())
        ratio_max = float(ratio[response_mask].max().item()) if n_tokens > 0 else 0.0

        pg_loss_sum = float((pg * mask).sum().item())
        kl_sum = float((kl * mask).sum().item())
        ratio_sum = float((ratio * mask).sum().item())

    return MicrobatchLossOut(
        loss=loss,
        n_tokens=n_tokens,
        pg_loss_sum=pg_loss_sum,
        kl_sum=kl_sum,
        clip_count=clip_count,
        ratio_sum=ratio_sum,
        ratio_max=ratio_max,
    )
