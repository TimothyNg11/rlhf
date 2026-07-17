"""Policy forward passes and weight-sync helpers for the training loop.

The module-level helpers (``response_logprobs``, ``bf16_sync_tensors``,
``model_checksums``) are pure ``torch`` + stdlib so they're CPU-testable
without ``transformers`` installed. :class:`HFPolicy` is the real
transformers-backed wrapper used on the GPU box; it lazy-imports
``transformers`` inside ``__init__`` (pattern: ``src/grpo_math/eval/backends.py``)
so importing this module never requires transformers to be installed.
"""

from __future__ import annotations

import hashlib
from typing import Iterator

import torch

from grpo_math.trainer.algo import collate_token_batch, gather_logprobs_and_entropy


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def response_logprobs(
    model,
    samples: list[tuple[list[int], list[int]]],
    *,
    micro_batch_size: int,
    pad_token_id: int,
    compute_entropy: bool = False,
    autocast_dtype: torch.dtype | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
    """No-grad sweep over ``samples``, returning per-sample response
    log-probs (and optionally entropy).

    ``model`` is any callable ``model(input_ids=..., attention_mask=...)``
    returning either an object with a ``.logits`` attribute or a raw logits
    tensor.

    Returns per-sample fp32 1-D tensors aligned to each sample's response
    tokens (``len(tensor) == len(response_ids)``), in the ORIGINAL input
    order -- plus per-sample entropy tensors (same shapes) when
    ``compute_entropy`` else ``None``.

    Samples are sorted by total (prompt + response) length, descending,
    before being walked ``micro_batch_size`` at a time and collated with
    :func:`~grpo_math.trainer.algo.collate_token_batch`, so that each
    micro-batch pads as little as possible; the result is restored to the
    caller's original order before returning.

    Per-sample (rather than per-micro-batch) return values mean a later
    training step can re-collate its own micro-batches from these samples
    without any alignment coupling to this sweep's batching.
    """
    n = len(samples)
    lengths = [len(prompt_ids) + len(response_ids) for prompt_ids, response_ids in samples]
    order = sorted(range(n), key=lambda i: lengths[i], reverse=True)

    device = _model_device(model)

    logprobs_out: list[torch.Tensor] = [None] * n  # type: ignore[list-item]
    entropy_out: list[torch.Tensor] | None = [None] * n if compute_entropy else None  # type: ignore[list-item]

    for start in range(0, n, micro_batch_size):
        chunk = order[start : start + micro_batch_size]
        mb_samples = [samples[i] for i in chunk]
        batch = collate_token_batch(mb_samples, pad_token_id=pad_token_id)

        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        response_mask = batch.response_mask.to(device)

        with torch.no_grad():
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = getattr(out, "logits", out)

        targets = input_ids[:, 1:]
        chosen_logprobs, entropy = gather_logprobs_and_entropy(logits[:, :-1], targets)
        shifted_response_mask = response_mask[:, 1:]

        for local_pos, sample_idx in enumerate(chunk):
            row_mask = shifted_response_mask[local_pos]
            logprobs_out[sample_idx] = chosen_logprobs[local_pos][row_mask].float().cpu()
            if compute_entropy:
                entropy_out[sample_idx] = entropy[local_pos][row_mask].float().cpu()

    return logprobs_out, entropy_out


def bf16_sync_tensors(model) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, bf16_tensor)`` for every named parameter, converting
    fp32 master weights to contiguous bf16 copies for the vLLM engine."""
    for name, param in model.named_parameters():
        yield name, param.detach().to(torch.bfloat16).contiguous()


def tensor_checksum(t: torch.Tensor) -> str:
    """SHA-256 hexdigest of ``t``'s raw bytes (viewed as ``uint8``).

    Shared by :func:`model_checksums` (trainer side) and the vLLM
    ``WeightSyncWorkerExtension`` (engine side) so the weight-sync handshake
    hashes both ends with byte-identical logic. ``t`` must be contiguous
    (:func:`bf16_sync_tensors` guarantees this)."""
    raw_bytes = t.cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()


def model_checksums(model) -> dict[str, str]:
    """SHA-256 hexdigest of the exact bytes :func:`bf16_sync_tensors` would
    send for each parameter, used for the trainer<->vLLM weight-sync
    handshake. Computed from the bf16-converted, contiguous tensor (not the
    fp32 master)."""
    return {name: tensor_checksum(tensor) for name, tensor in bf16_sync_tensors(model)}


class HFPolicy:
    """Transformers-backed policy: fp32 master weights, trained/generated
    from via bf16 autocast + :func:`bf16_sync_tensors`/:func:`model_checksums`
    for the vLLM weight sync."""

    def __init__(
        self,
        model_name: str,
        *,
        grad_checkpointing: bool = True,
        device: str = "cuda",
        trainable: bool = True,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy: not installed in dev

        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device(device)
        self.model.to(self.device)

        if trainable:
            self.model.train()
            if grad_checkpointing:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            self.model.eval()
            self.model.requires_grad_(False)

    def forward_logprobs(self, samples, *, micro_batch_size, compute_entropy=False):
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        autocast_dtype = torch.bfloat16 if self.device.type != "cpu" else None
        return response_logprobs(
            self.model,
            samples,
            micro_batch_size=micro_batch_size,
            pad_token_id=pad_token_id,
            compute_entropy=compute_entropy,
            autocast_dtype=autocast_dtype,
        )

    def sync_iterator(self):
        return bf16_sync_tensors(self.model)

    def checksums(self):
        return model_checksums(self.model)
