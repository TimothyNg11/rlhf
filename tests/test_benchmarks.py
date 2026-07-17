"""Tests for grpo_math.eval.benchmarks: registry + problem loading."""

from pathlib import Path

import pytest

from grpo_math.eval.benchmarks import (
    ANSWER_INSTRUCTION,
    BENCHMARKS,
    BenchmarkSpec,
    EvalProblem,
    load_benchmark,
    register_local_benchmark,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def _register_tiny_math():
    register_local_benchmark("tiny_math", FIXTURES_DIR / "tiny_math.jsonl")
    yield
    BENCHMARKS.pop("tiny_math", None)


def test_registry_has_all_expected_benchmarks():
    expected = {"aime24", "aime25", "math500", "gsm8k", "amc23", "mmlu_stem"}
    assert expected <= BENCHMARKS.keys()


@pytest.mark.parametrize(
    "name,dataset_id,split",
    [
        ("aime24", "HuggingFaceH4/aime_2024", "train"),
        ("aime25", "yentinglin/aime_2025", "train"),
        ("math500", "HuggingFaceH4/MATH-500", "test"),
        ("gsm8k", "openai/gsm8k", "test"),
        ("amc23", "math-ai/amc23", "test"),
        ("mmlu_stem", "cais/mmlu", "test"),
    ],
)
def test_hf_registry_fields_match_brief(name, dataset_id, split):
    spec = BENCHMARKS[name]
    assert spec.source == "hf"
    assert spec.dataset_id == dataset_id
    assert spec.split == split


def test_gsm8k_uses_main_config():
    assert BENCHMARKS["gsm8k"].hf_config == "main"


def test_mmlu_stem_uses_all_config():
    assert BENCHMARKS["mmlu_stem"].hf_config == "all"


def test_load_local_benchmark_returns_eval_problems():
    problems = load_benchmark("tiny_math")
    assert len(problems) == 6
    assert all(isinstance(p, EvalProblem) for p in problems)


def test_local_benchmark_appends_answer_instruction():
    problems = load_benchmark("tiny_math")
    for p in problems:
        assert p.prompt.endswith(ANSWER_INSTRUCTION)
        assert p.prompt.startswith("What is") or p.prompt.startswith("Simplify")


def test_local_benchmark_deterministic_order():
    a = load_benchmark("tiny_math")
    b = load_benchmark("tiny_math")
    assert [p.problem_id for p in a] == [p.problem_id for p in b]


def test_local_benchmark_gold_and_metadata_roundtrip():
    problems = load_benchmark("tiny_math")
    first = problems[0]
    assert first.problem_id == "tm_1"
    assert first.gold == "4"
    assert first.metadata == {"topic": "arithmetic"}


def test_limit_truncates_after_ordering():
    full = load_benchmark("tiny_math")
    limited = load_benchmark("tiny_math", limit=3)
    assert len(limited) == 3
    assert [p.problem_id for p in limited] == [p.problem_id for p in full[:3]]


def test_unknown_benchmark_raises():
    with pytest.raises(KeyError):
        load_benchmark("not_a_real_benchmark")


def test_import_does_not_require_datasets_or_vllm():
    # module-level import must never require `datasets` (gpu extra); this test
    # is meaningful because `datasets` is genuinely absent from this venv.
    import grpo_math.eval.benchmarks  # noqa: F401


# --- take_last + gsm8k_dev ------------------------------------------------


def test_gsm8k_dev_registered_with_take_last():
    # No dataset download here: just checking the registry entry's declared spec.
    spec = BENCHMARKS["gsm8k_dev"]
    assert spec.source == "hf"
    assert spec.kind == "gsm8k"
    assert spec.split == "train"
    assert spec.take_last == 500


@pytest.fixture
def _register_tiny_math_take_last():
    BENCHMARKS["tiny_math_take_last"] = BenchmarkSpec(
        source="local", path=FIXTURES_DIR / "tiny_math.jsonl", take_last=2
    )
    yield
    BENCHMARKS.pop("tiny_math_take_last", None)


def test_take_last_slices_local_benchmark(_register_tiny_math_take_last):
    full = load_benchmark("tiny_math")
    limited = load_benchmark("tiny_math_take_last")
    assert len(limited) == 2
    assert [p.problem_id for p in limited] == [p.problem_id for p in full[-2:]]
