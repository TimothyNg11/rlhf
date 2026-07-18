"""vLLM rollout wrapper: sleep/wake, the trainer<->engine weight-sync
handshake, and an adapter that exposes the rollout as an eval
:class:`~grpo_math.eval.backends.GenerationBackend`.

``vllm`` is imported lazily (inside :class:`VLLMRollout` methods) exactly like
``grpo_math.eval.backends.VLLMBackend`` / ``grpo_math.trainer.policy.HFPolicy``,
so importing this module never requires vllm to be installed (the dev venv has
no vllm). :class:`WeightSyncWorkerExtension` is mixed into the vLLM worker via
``worker_extension_cls`` and only touches attributes the worker provides at
runtime, so it too needs no module-level vllm import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch

from grpo_math.eval.backends import Completion
from grpo_math.trainer.policy import tensor_checksum


@dataclass
class RolloutSample:
    prompt_idx: int  # index into the prompts list passed to generate()
    text: str
    finish_reason: str  # "stop" | "length"
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    sampler_logprobs: list[float] | None  # chosen-token logprobs from the sampler; MONITORING ONLY


class RolloutGenerationBackend:
    """Adapter satisfying :class:`grpo_math.eval.backends.GenerationBackend`,
    wrapping any object with a ``.generate(prompts, *, group_size, temperature,
    top_p, max_tokens, seed) -> list[list[RolloutSample]]`` method (the real
    :class:`VLLMRollout` or a test fake).

    ``generate(prompts, *, k, ...)`` maps ``k`` -> ``group_size`` and each
    :class:`RolloutSample` -> ``Completion(text=..., finish_reason=...,
    n_tokens=len(response_token_ids))``. Pure mapping -- no vllm import.
    """

    def __init__(self, rollout):
        self._rollout = rollout

    def generate(
        self,
        prompts: list[str],
        *,
        k: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
    ) -> list[list[Completion]]:
        groups = self._rollout.generate(
            prompts,
            group_size=k,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        return [
            [
                Completion(
                    text=sample.text,
                    finish_reason=sample.finish_reason,
                    n_tokens=len(sample.response_token_ids),
                )
                for sample in group
            ]
            for group in groups
        ]


class WeightSyncWorkerExtension:
    """Mixed into the vLLM worker via ``worker_extension_cls``. Hashes every
    received tensor with the shared :func:`~grpo_math.trainer.policy.tensor_checksum`
    helper (the same one :func:`~grpo_math.trainer.policy.model_checksums` uses on
    the trainer side) so the weight-sync handshake is a real end-to-end byte
    comparison, then loads the weights into the running model."""

    def grpo_update_weights(
        self, weights: list[tuple[str, torch.Tensor]], want_checksums: bool = True
    ) -> dict[str, str]:
        # Named grpo_update_weights (not update_weights): vLLM >= 0.25 gives the
        # worker a built-in `update_weights(update_info: dict) -> None` and
        # worker_extension_cls refuses to shadow existing worker attributes.
        weights = list(weights)
        self.model_runner.model.load_weights(weights)
        return {name: tensor_checksum(t) for name, t in weights} if want_checksums else {}


class VLLMRollout:
    """Real vLLM offline-engine rollout used on the GPU box."""

    def __init__(
        self,
        model_name: str,
        *,
        max_model_len: int,
        gpu_memory_utilization: float,
        dtype: str = "bfloat16",
        enable_sleep_mode: bool = True,
        seed: int = 0,
    ):
        from vllm import LLM  # lazy: vllm is a gpu-extra dep, not installed in dev

        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_sleep_mode=enable_sleep_mode,
            seed=seed,
            worker_extension_cls="grpo_math.trainer.rollout.WeightSyncWorkerExtension",
        )
        # Fallback if collective_rpc / worker_extension_cls breaks on a future
        # vllm: load weights via the direct attribute path
        #   self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(...)
        # (exact chain has drifted across vllm versions; check before relying on it).

    def generate(
        self,
        prompts: list[str],
        *,
        group_size: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
    ) -> list[list[RolloutSample]]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            n=group_size,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            logprobs=0,
        )
        conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
        outputs = self.llm.chat(conversations, sampling_params)

        groups: list[list[RolloutSample]] = []
        for prompt_idx, out in enumerate(outputs):
            prompt_token_ids = list(out.prompt_token_ids)
            group = []
            for c in out.outputs:
                response_token_ids = list(c.token_ids)
                sampler_logprobs = None
                if c.logprobs is not None:
                    # c.logprobs[t] is a {token_id: Logprob} dict for position t;
                    # take the chosen token's logprob at each position.
                    sampler_logprobs = [
                        float(pos[token_id].logprob)
                        for token_id, pos in zip(response_token_ids, c.logprobs)
                    ]
                group.append(
                    RolloutSample(
                        prompt_idx=prompt_idx,
                        text=c.text,
                        finish_reason=c.finish_reason,
                        prompt_token_ids=prompt_token_ids,
                        response_token_ids=response_token_ids,
                        sampler_logprobs=sampler_logprobs,
                    )
                )
            groups.append(group)
        return groups

    def sleep(self) -> None:
        self.llm.sleep(level=1)

    def wake(self) -> None:
        self.llm.wake_up()

    def sync_weights(
        self, weights: Iterator[tuple[str, torch.Tensor]], *, want_checksums: bool = True
    ) -> dict[str, str]:
        results = self.llm.collective_rpc(
            "grpo_update_weights", args=(list(weights), want_checksums)
        )
        return results[0]

    def as_generation_backend(self) -> RolloutGenerationBackend:
        return RolloutGenerationBackend(self)
