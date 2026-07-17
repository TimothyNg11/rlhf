"""Tests for grpo_math.eval.runner: run_eval end-to-end via FakeBackend, plus a
CLI smoke test for scripts/run_eval.py."""

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

from grpo_math.eval.backends import FakeBackend
from grpo_math.eval.benchmarks import BENCHMARKS, load_benchmark, register_local_benchmark
from grpo_math.eval.runner import load_summary, run_eval

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

EVAL_CFG = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 32768,
    "k_default": 2,
    "k_final_aime": 4,
    "seed": 0,
    "benchmarks": ["tiny_math"],
}


@pytest.fixture(autouse=True)
def _register_tiny_math():
    register_local_benchmark("tiny_math", FIXTURES_DIR / "tiny_math.jsonl")
    yield
    BENCHMARKS.pop("tiny_math", None)


def _build_script(problems):
    """Known correct/incorrect/truncated/unparseable mix, keyed by full prompt text.

    tm_1 (gold 4):    correct, incorrect
    tm_2 (gold 7):    correct, correct
    tm_3 (gold 30):   incorrect, incorrect
    tm_4 (gold 3/4):  correct, truncated-but-correct-looking (must score 0)
    tm_5 (gold 49):   unparseable, correct
    tm_6 (gold 4):    unparseable, unparseable
    """
    by_id = {p.problem_id: p for p in problems}
    return {
        by_id["tm_1"].prompt: ["\\boxed{4}", "\\boxed{5}"],
        by_id["tm_2"].prompt: ["\\boxed{7}", "\\boxed{7}"],
        by_id["tm_3"].prompt: ["\\boxed{31}", "\\boxed{29}"],
        by_id["tm_4"].prompt: ["\\boxed{\\frac{3}{4}}", "\\boxed{\\frac{3}{4}}<TRUNCATED>"],
        by_id["tm_5"].prompt: ["no boxed answer here", "\\boxed{49}"],
        by_id["tm_6"].prompt: ["still no box", "nor here either"],
    }


def _read_samples(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _run(out_dir, seed=0):
    problems = load_benchmark("tiny_math")
    backend = FakeBackend(_build_script(problems))
    cfg = {**EVAL_CFG, "seed": seed}
    return run_eval("tiny_math", backend, cfg, out_dir=out_dir, model_name="test-model")


def test_run_eval_exact_pass_at_1(tmp_path):
    summary = _run(tmp_path)
    # verdict sum = 1+2+0+1+1+0 = 5 across 6 problems x k=2 = 12 samples
    assert summary.pass_at_1 == pytest.approx(5 / 12)


def test_run_eval_exact_parse_rate(tmp_path):
    summary = _run(tmp_path)
    # parseable: tm1(2) + tm2(2) + tm3(2) + tm4(2, truncated one still has a boxed answer)
    # + tm5(1) + tm6(0) = 9 / 12
    assert summary.parse_rate == pytest.approx(9 / 12)


def test_run_eval_exact_truncation_rate(tmp_path):
    summary = _run(tmp_path)
    assert summary.truncation_rate == pytest.approx(1 / 12)


def test_truncated_correct_looking_completion_scores_zero(tmp_path):
    _run(tmp_path)
    records = _read_samples(tmp_path / "tiny_math" / "samples.jsonl.gz")
    tm4_truncated = next(
        r for r in records if r["problem_id"] == "tm_4" and r["finish_reason"] == "length"
    )
    assert tm4_truncated["verdict"] == 0.0
    assert tm4_truncated["parseable"] is True  # boxed answer was present, just truncated


def test_ci_deterministic_across_two_runs(tmp_path_factory):
    d1 = tmp_path_factory.mktemp("run1")
    d2 = tmp_path_factory.mktemp("run2")
    s1 = _run(d1)
    s2 = _run(d2)
    assert s1.ci_lo == s2.ci_lo
    assert s1.ci_hi == s2.ci_hi
    assert s1.pass_at_1 == s2.pass_at_1


def test_output_files_written(tmp_path):
    _run(tmp_path)
    bench_dir = tmp_path / "tiny_math"
    assert (bench_dir / "samples.jsonl.gz").exists()
    assert (bench_dir / "summary.json").exists()
    assert (bench_dir / "summary.md").exists()


def test_samples_jsonl_row_count(tmp_path):
    _run(tmp_path)
    records = _read_samples(tmp_path / "tiny_math" / "samples.jsonl.gz")
    assert len(records) == 12  # 6 problems x k=2


def test_load_summary_roundtrip(tmp_path):
    summary = _run(tmp_path)
    loaded = load_summary(tmp_path / "tiny_math" / "summary.json")
    assert loaded == summary


def test_summary_json_has_expected_keys(tmp_path):
    _run(tmp_path)
    data = json.loads((tmp_path / "tiny_math" / "summary.json").read_text(encoding="utf-8"))
    expected = {
        "benchmark",
        "model_name",
        "k",
        "temperature",
        "top_p",
        "max_tokens",
        "seed",
        "n_problems",
        "pass_at_1",
        "ci_lo",
        "ci_hi",
        "parse_rate",
        "truncation_rate",
        "mean_completion_tokens",
        "timestamp",
        "git_describe",
    }
    assert expected <= data.keys()


def test_git_describe_is_none_or_str(tmp_path):
    summary = _run(tmp_path)
    assert summary.git_describe is None or isinstance(summary.git_describe, str)


def test_k_defaults_from_eval_cfg(tmp_path):
    summary = _run(tmp_path)
    assert summary.k == EVAL_CFG["k_default"]


def test_k_override(tmp_path):
    problems = load_benchmark("tiny_math")
    backend = FakeBackend({p.prompt: ["\\boxed{0}"] for p in problems})
    summary = run_eval("tiny_math", backend, EVAL_CFG, k=1, out_dir=tmp_path, model_name="m")
    assert summary.k == 1


def test_limit_reduces_n_problems(tmp_path):
    problems = load_benchmark("tiny_math", limit=3)
    backend = FakeBackend({p.prompt: ["\\boxed{0}"] * 2 for p in problems})
    summary = run_eval("tiny_math", backend, EVAL_CFG, limit=3, out_dir=tmp_path, model_name="m")
    assert summary.n_problems == 3


# --- CLI smoke test -----------------------------------------------------------


def _load_script_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_smoke_run_eval_main(tmp_path):
    problems = load_benchmark("tiny_math", limit=3)
    fake_script = {p.prompt: ["\\boxed{0}", "\\boxed{0}"] for p in problems}
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(fake_script), encoding="utf-8")

    run_eval_cli = _load_script_module("run_eval_cli", "run_eval.py")
    out_dir = tmp_path / "out"

    exit_code = run_eval_cli.main(
        [
            "--config",
            str(REPO_ROOT / "configs" / "eval.yaml"),
            "--benchmark",
            "tiny_math",
            "--model",
            "dummy-model",
            "--backend",
            "fake",
            "--fake-script",
            str(script_path),
            "--k",
            "2",
            "--limit",
            "3",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert (out_dir / "tiny_math" / "summary.json").exists()
    assert (out_dir / "tiny_math" / "samples.jsonl.gz").exists()
    assert (out_dir / "tiny_math" / "summary.md").exists()


def test_cli_fake_backend_requires_fake_script(tmp_path):
    run_eval_cli = _load_script_module("run_eval_cli2", "run_eval.py")
    with pytest.raises(SystemExit):
        run_eval_cli.main(
            [
                "--config",
                str(REPO_ROOT / "configs" / "eval.yaml"),
                "--benchmark",
                "tiny_math",
                "--model",
                "dummy-model",
                "--backend",
                "fake",
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
