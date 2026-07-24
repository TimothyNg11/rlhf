"""Tests for scripts/regrade_lenient.py (importlib pattern, mirroring
tests/test_paired_delta.py): offline strict-vs-lenient re-grade of an existing
run_eval.py output directory (samples.jsonl.gz + summary.json), plus its
built-in sanity gate against the stored strict pass@1.

Fixtures are built via the real run_eval() + FakeBackend (see
tests/test_runner.py's pattern) rather than hand-rolled gzip writing, so the
record schema is guaranteed to match what scripts/run_eval.py actually
produces.
"""

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

from grpo_math.eval.backends import FakeBackend
from grpo_math.eval.benchmarks import BENCHMARKS, load_benchmark, register_local_benchmark
from grpo_math.eval.runner import run_eval

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_script_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


regrade_lenient = _load_script_module("regrade_lenient", "regrade_lenient.py")


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
    """Mix of boxed, unboxed-but-lenient-parseable, wrong, unparseable, and
    truncated completions -- chosen so strict and lenient grading disagree on
    tm_1's second sample (the whole point of this script).

    tm_1 (gold 4):             boxed-correct, unboxed-but-lenient-correct
    tm_2 (gold 7):             boxed-correct, boxed-correct
    tm_3 (gold 30):            boxed-wrong, boxed-wrong
    tm_4 (gold 3/4):           boxed-correct, truncated-but-correct-looking
    tm_5 (gold 49):            unparseable, boxed-correct
    tm_6 (gold 4):             unparseable, unparseable
    """
    by_id = {p.problem_id: p for p in problems}
    return {
        by_id["tm_1"].prompt: ["\\boxed{4}", "The final answer is 4."],
        by_id["tm_2"].prompt: ["\\boxed{7}", "\\boxed{7}"],
        by_id["tm_3"].prompt: ["\\boxed{31}", "\\boxed{29}"],
        by_id["tm_4"].prompt: ["\\boxed{\\frac{3}{4}}", "\\boxed{\\frac{3}{4}}<TRUNCATED>"],
        by_id["tm_5"].prompt: ["no boxed answer here", "\\boxed{49}"],
        by_id["tm_6"].prompt: ["still no box", "nor here either"],
    }


def _write_run(out_dir, seed=0):
    problems = load_benchmark("tiny_math")
    backend = FakeBackend(_build_script(problems))
    cfg = {**EVAL_CFG, "seed": seed}
    return run_eval("tiny_math", backend, cfg, out_dir=out_dir, model_name="test-model")


def _read_regrade_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_regrade_sanity_gate_passes_on_untouched_run(tmp_path):
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    _write_run(run_dir)

    rc = regrade_lenient.main(
        ["--run", str(run_dir), "--out", str(out_dir), "--benchmark", "tiny_math"]
    )

    assert rc == 0
    assert (out_dir / "tiny_math" / "samples_regrade.jsonl.gz").exists()
    assert (out_dir / "tiny_math" / "summary_regrade.json").exists()


def test_regrade_summary_stats(tmp_path):
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    _write_run(run_dir)

    rc = regrade_lenient.main(
        ["--run", str(run_dir), "--out", str(out_dir), "--benchmark", "tiny_math"]
    )
    assert rc == 0

    summary = json.loads((out_dir / "tiny_math" / "summary_regrade.json").read_text(encoding="utf-8"))

    assert summary["benchmark"] == "tiny_math"
    assert summary["n_problems"] == 6
    assert summary["k"] == 2

    # verdict sum: strict = 1+2+0+1+1+0 = 5 (tm_1's 2nd sample is unboxed -> unparseable)
    assert summary["strict"]["pass_at_1"] == pytest.approx(5 / 12)
    # lenient rescues tm_1's 2nd sample ("The final answer is 4.") -> sum = 6
    assert summary["lenient"]["pass_at_1"] == pytest.approx(6 / 12)

    # any-correct: tm_1, tm_2, tm_4, tm_5 have >=1 correct sample in both modes; tm_3, tm_6 never do.
    assert summary["strict"]["pass_at_k_any"] == pytest.approx(4 / 6)
    assert summary["lenient"]["pass_at_k_any"] == pytest.approx(4 / 6)

    assert 0.0 <= summary["strict"]["ci_lo"] <= summary["strict"]["pass_at_1"] <= summary["strict"]["ci_hi"] <= 1.0
    assert 0.0 <= summary["lenient"]["ci_lo"] <= summary["lenient"]["pass_at_1"] <= summary["lenient"]["ci_hi"] <= 1.0

    # extraction_method histogram over all 12 completions' *lenient* extraction:
    # boxed: tm1s0, tm2s0, tm2s1, tm3s0, tm3s1, tm4s0, tm4s1, tm5s1 = 8
    # answer_is: tm1s1 = 1
    # none: tm5s0, tm6s0, tm6s1 = 3
    assert summary["extraction_method_histogram"] == {"boxed": 8, "answer_is": 1, "none": 3}


def test_regrade_records_have_lenient_fields(tmp_path):
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    _write_run(run_dir)

    rc = regrade_lenient.main(
        ["--run", str(run_dir), "--out", str(out_dir), "--benchmark", "tiny_math"]
    )
    assert rc == 0

    records = _read_regrade_records(out_dir / "tiny_math" / "samples_regrade.jsonl.gz")
    assert len(records) == 12

    by_key = {(r["problem_id"], r["sample_idx"]): r for r in records}

    expected_fields = {
        "problem_id", "sample_idx", "completion", "finish_reason", "n_tokens",
        "extracted", "gold", "verdict", "parseable",
        "verdict_strict", "verdict_lenient", "extracted_lenient", "extraction_method",
    }
    assert set(by_key[("tm_1", 0)]) == expected_fields

    # tm_1 sample 1: "The final answer is 4." -- unboxed, strict fails, lenient rescues it.
    tm1_s1 = by_key[("tm_1", 1)]
    assert tm1_s1["verdict_strict"] == 0.0
    assert tm1_s1["verdict_lenient"] == 1.0
    assert tm1_s1["extracted_lenient"] == "4"
    assert tm1_s1["extraction_method"] == "answer_is"

    # tm_4 sample 1: truncated-but-correct-looking -- 0 in both modes despite a boxed answer.
    tm4_s1 = by_key[("tm_4", 1)]
    assert tm4_s1["finish_reason"] == "length"
    assert tm4_s1["verdict_strict"] == 0.0
    assert tm4_s1["verdict_lenient"] == 0.0
    assert tm4_s1["extraction_method"] == "boxed"

    # tm_5 sample 0: "no boxed answer here" -- unparseable in both modes.
    tm5_s0 = by_key[("tm_5", 0)]
    assert tm5_s0["verdict_strict"] == 0.0
    assert tm5_s0["verdict_lenient"] == 0.0
    assert tm5_s0["extracted_lenient"] is None
    assert tm5_s0["extraction_method"] == "none"


def test_regrade_omitting_benchmark_discovers_it(tmp_path):
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    _write_run(run_dir)

    rc = regrade_lenient.main(["--run", str(run_dir), "--out", str(out_dir)])

    assert rc == 0
    assert (out_dir / "tiny_math" / "summary_regrade.json").exists()


def test_regrade_prints_compact_table(tmp_path, capsys):
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    _write_run(run_dir)

    regrade_lenient.main(["--run", str(run_dir), "--out", str(out_dir), "--benchmark", "tiny_math"])

    captured = capsys.readouterr()
    assert "tiny_math" in captured.out
    assert "pass@1" in captured.out


def test_regrade_sanity_gate_fails_on_corrupted_summary(tmp_path):
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    _write_run(run_dir)

    summary_path = run_dir / "tiny_math" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["pass_at_1"] = summary["pass_at_1"] + 0.5  # corrupt the stored value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    rc = regrade_lenient.main(
        ["--run", str(run_dir), "--out", str(out_dir), "--benchmark", "tiny_math"]
    )

    assert rc != 0
    # the gate failed -- nothing should ride on the (unvalidated) regrade output.
    assert not (out_dir / "tiny_math" / "summary_regrade.json").exists()


def test_regrade_sanity_gate_failure_message_names_benchmark_and_values(tmp_path, capsys):
    import re

    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    _write_run(run_dir)

    summary_path = run_dir / "tiny_math" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recomputed = summary["pass_at_1"]  # strict regrade must match this (untouched) value
    stored_wrong = recomputed + 0.5
    summary["pass_at_1"] = stored_wrong
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    regrade_lenient.main(["--run", str(run_dir), "--out", str(out_dir), "--benchmark", "tiny_math"])

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "tiny_math" in output

    numbers = [float(m) for m in re.findall(r"-?\d+\.\d+", output)]
    assert any(abs(n - recomputed) < 1e-3 for n in numbers), (recomputed, numbers)
    assert any(abs(n - stored_wrong) < 1e-3 for n in numbers), (stored_wrong, numbers)
