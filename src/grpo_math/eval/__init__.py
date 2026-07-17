"""Evaluation harness: benchmark registry, generation backends, and the eval runner."""

from grpo_math.eval.backends import Completion, FakeBackend, GenerationBackend, VLLMBackend
from grpo_math.eval.benchmarks import (
    BENCHMARKS,
    BenchmarkSpec,
    EvalProblem,
    load_benchmark,
    register_local_benchmark,
)
from grpo_math.eval.runner import EvalSummary, load_summary, run_eval

__all__ = [
    "Completion",
    "FakeBackend",
    "GenerationBackend",
    "VLLMBackend",
    "BENCHMARKS",
    "BenchmarkSpec",
    "EvalProblem",
    "load_benchmark",
    "register_local_benchmark",
    "EvalSummary",
    "load_summary",
    "run_eval",
]
