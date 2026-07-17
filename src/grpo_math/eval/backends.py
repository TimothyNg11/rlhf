"""Generation backend protocol + implementations.

``FakeBackend`` is deterministic and used by every test in this repo.
``VLLMBackend`` is exercised only on the GPU box; it lazy-imports vllm inside
``__init__`` so that importing this module never requires vllm to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

_TRUNCATION_SENTINEL = "<TRUNCATED>"


@dataclass
class Completion:
    text: str
    finish_reason: str  # "stop" | "length"
    n_tokens: int


class GenerationBackend(Protocol):
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
        """Generate ``k`` completions per prompt. Outer list aligned with
        ``prompts``; each inner list has length ``k``."""
        ...


class VLLMBackend:
    """Real generation backend using vLLM's offline engine.

    Applies the model's chat template via ``llm.chat`` (a single user-role
    message per prompt) rather than templating manually. NOTE: the R1-distill
    chat template auto-opens ``<think>`` in the generation prompt -- do NOT
    prepend it to ``prompt`` yourself.
    """

    def __init__(self, model: str, **vllm_kwargs):
        from vllm import LLM  # lazy: vllm is a gpu-extra dep, not installed in dev

        self.model = model
        self.llm = LLM(model=model, **vllm_kwargs)

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
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            n=k, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed
        )
        conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
        outputs = self.llm.chat(conversations, sampling_params)

        results = []
        for output in outputs:
            completions = [
                Completion(
                    text=sample.text,
                    finish_reason=sample.finish_reason,
                    n_tokens=len(sample.token_ids),
                )
                for sample in output.outputs
            ]
            results.append(completions)
        return results


class FakeBackend:
    """Deterministic backend for tests: returns canned completions instead of
    calling a model.

    ``script`` is either:
      - a ``dict[str, list[str]]`` mapping prompt -> canned completion texts,
        indexed (with wraparound) by the sample index ``i`` in ``0..k-1``, or
      - a callable ``(prompt, i) -> str`` producing the i-th completion text.

    A canned text ending with the sentinel ``"<TRUNCATED>"`` is reported with
    ``finish_reason="length"`` (the sentinel is stripped from the returned
    text); otherwise ``finish_reason="stop"``. ``n_tokens = len(text.split())``
    (whitespace token count, computed after stripping the sentinel).
    """

    def __init__(self, script: dict[str, list[str]] | Callable[[str, int], str]):
        self.script = script

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
        results = []
        for prompt in prompts:
            completions = []
            for i in range(k):
                text = self._get_text(prompt, i)
                finish_reason = "stop"
                if text.endswith(_TRUNCATION_SENTINEL):
                    text = text[: -len(_TRUNCATION_SENTINEL)]
                    finish_reason = "length"
                completions.append(
                    Completion(text=text, finish_reason=finish_reason, n_tokens=len(text.split()))
                )
            results.append(completions)
        return results

    def _get_text(self, prompt: str, i: int) -> str:
        if callable(self.script):
            return self.script(prompt, i)
        texts = self.script[prompt]
        return texts[i % len(texts)]
