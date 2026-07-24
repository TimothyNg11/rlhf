"""Tests for scripts/build_difficulty_map.py (importlib pattern, mirroring
tests/test_paired_delta.py): fake-backend end-to-end difficulty-map
construction (raw counts, dev/train split tagging, band kept-counts), plus the
CLI's import/--help cleanliness (no torch/vllm/datasets import at module top,
so this loads fine in the vllm/transformers/datasets-free dev venv -- see
tests/test_data_gsm8k.py's analogous note about load_gsm8k_train)."""

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

from grpo_math.eval.backends import FakeBackend
from grpo_math.eval.benchmarks import EvalProblem

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_script_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_difficulty_map = _load_script_module("build_difficulty_map", "build_difficulty_map.py")


def _problems():
    return [
        EvalProblem(problem_id="p1", prompt="P1", gold="1", metadata={}),
        EvalProblem(problem_id="p2", prompt="P2", gold="2", metadata={}),
        EvalProblem(problem_id="p3", prompt="P3", gold="3", metadata={}),
    ]


def _splits():
    return ["train", "train", "dev"]


# --- import / CLI cleanliness --------------------------------------------------


def test_module_has_expected_public_functions():
    assert hasattr(build_difficulty_map, "build_difficulty_map")
    assert hasattr(build_difficulty_map, "load_all_problems")
    assert hasattr(build_difficulty_map, "build_parser")
    assert hasattr(build_difficulty_map, "main")


def test_help_works():
    with pytest.raises(SystemExit) as exc_info:
        build_difficulty_map.main(["--help"])
    assert exc_info.value.code == 0


def test_cli_fake_backend_requires_fake_script():
    with pytest.raises(SystemExit):
        build_difficulty_map.main(
            [
                "--model", "dummy", "--k", "2", "--temperature", "0.9", "--top-p", "1.0",
                "--max-tokens", "64", "--seed", "0", "--out", "results/x",
                "--backend", "fake",
            ]
        )


# --- build_difficulty_map: fake-backend end-to-end -----------------------------


def test_map_jsonl_raw_counts_and_split_tags(tmp_path):
    problems = _problems()
    splits = _splits()
    # k=4 samples/problem: p1 correct 3/4, p2 correct 0/4, p3 correct 4/4 (lenient).
    script = {
        "P1": ["\\boxed{1}", "\\boxed{1}", "\\boxed{1}", "\\boxed{9}"],
        "P2": ["\\boxed{9}", "\\boxed{9}", "\\boxed{9}", "\\boxed{9}"],
        "P3": ["\\boxed{3}", "\\boxed{3}", "\\boxed{3}", "\\boxed{3}"],
    }
    backend = FakeBackend(script)

    summary = build_difficulty_map.build_difficulty_map(
        problems, splits, backend,
        k=4, temperature=0.9, top_p=1.0, max_tokens=64, seed=0,
        out_dir=tmp_path, model="dummy-model",
    )

    rows = [json.loads(line) for line in (tmp_path / "map.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {r["problem_id"]: r for r in rows}

    assert by_id["p1"]["n_correct_lenient"] == 3
    assert by_id["p1"]["n_correct_strict"] == 3
    assert by_id["p1"]["k"] == 4
    assert by_id["p1"]["split"] == "train"
    assert by_id["p2"]["n_correct_lenient"] == 0
    assert by_id["p2"]["split"] == "train"
    assert by_id["p3"]["n_correct_lenient"] == 4
    assert by_id["p3"]["split"] == "dev"
    assert "methods" in by_id["p1"]
    assert by_id["p1"]["methods"]["boxed"] == 4

    assert summary["n_problems"] == 3
    assert summary["n_train"] == 2
    assert summary["n_dev"] == 1


def test_samples_jsonl_written_with_eval_runner_style_records(tmp_path):
    problems = _problems()
    splits = _splits()
    script = {p.prompt: [f"\\boxed{{{p.gold}}}"] for p in problems}
    backend = FakeBackend(script)

    build_difficulty_map.build_difficulty_map(
        problems, splits, backend,
        k=1, temperature=0.9, top_p=1.0, max_tokens=64, seed=0,
        out_dir=tmp_path, model="dummy-model",
    )

    with gzip.open(tmp_path / "samples.jsonl.gz", "rt", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 3
    expected_fields = {
        "problem_id", "sample_idx", "completion", "finish_reason", "n_tokens",
        "extracted", "gold", "verdict", "parseable",
        "verdict_lenient", "extracted_lenient", "extraction_method",
    }
    assert set(records[0]) == expected_fields
    for r in records:
        assert r["verdict"] == 1.0
        assert r["verdict_lenient"] == 1.0


def test_truncated_samples_score_zero_and_count_in_map(tmp_path):
    problems = _problems()
    splits = _splits()
    script = {
        "P1": ["\\boxed{1}<TRUNCATED>", "\\boxed{1}"],
        "P2": ["\\boxed{9}", "\\boxed{9}"],
        "P3": ["\\boxed{3}", "\\boxed{3}"],
    }
    backend = FakeBackend(script)

    build_difficulty_map.build_difficulty_map(
        problems, splits, backend,
        k=2, temperature=0.9, top_p=1.0, max_tokens=64, seed=0,
        out_dir=tmp_path, model="dummy-model",
    )
    rows = [json.loads(line) for line in (tmp_path / "map.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {r["problem_id"]: r for r in rows}
    assert by_id["p1"]["n_truncated"] == 1
    assert by_id["p1"]["n_correct_lenient"] == 1  # only the non-truncated sample counts


def test_band_kept_counts_and_histogram_in_summary(tmp_path):
    problems = _problems()
    splits = _splits()
    # k=8: p1 rate 1/8 (boundary), p2 rate 0/8, p3 rate 8/8.
    script = {
        "P1": ["\\boxed{1}"] + ["\\boxed{9}"] * 7,
        "P2": ["\\boxed{9}"] * 8,
        "P3": ["\\boxed{3}"] * 8,
    }
    backend = FakeBackend(script)

    summary = build_difficulty_map.build_difficulty_map(
        problems, splits, backend,
        k=8, temperature=0.9, top_p=1.0, max_tokens=64, seed=0,
        out_dir=tmp_path, model="dummy-model",
    )

    band_counts = summary["candidate_band_kept_counts"]
    counts = list(band_counts.values())
    # [1/8, 7/8]: only p1 (rate 0.125) qualifies.
    assert counts[0] == 1
    # [0, 7/8]: p1, p2 qualify (0.125, 0.0); p3 (1.0) excluded.
    assert counts[1] == 2
    # [1/8, 1]: p1, p3 qualify (0.125, 1.0); p2 (0.0) excluded.
    assert counts[2] == 2

    hist = summary["solve_rate_histogram_lenient"]
    assert hist["0"] == 1  # p2
    assert hist["1"] == 1  # p1
    assert hist["8"] == 1  # p3


def test_dev_vs_train_mean_solve_rate(tmp_path):
    problems = _problems()  # p1, p2 train; p3 dev
    splits = _splits()
    script = {
        "P1": ["\\boxed{1}", "\\boxed{9}"],  # 1/2 correct
        "P2": ["\\boxed{9}", "\\boxed{9}"],  # 0/2 correct
        "P3": ["\\boxed{3}", "\\boxed{3}"],  # 2/2 correct
    }
    backend = FakeBackend(script)

    summary = build_difficulty_map.build_difficulty_map(
        problems, splits, backend,
        k=2, temperature=0.9, top_p=1.0, max_tokens=64, seed=0,
        out_dir=tmp_path, model="dummy-model",
    )
    assert summary["mean_solve_rate_lenient_train"] == pytest.approx((0.5 + 0.0) / 2)
    assert summary["mean_solve_rate_lenient_dev"] == pytest.approx(1.0)


def test_summary_md_written(tmp_path):
    problems = _problems()
    splits = _splits()
    script = {p.prompt: [f"\\boxed{{{p.gold}}}"] for p in problems}
    backend = FakeBackend(script)

    build_difficulty_map.build_difficulty_map(
        problems, splits, backend,
        k=1, temperature=0.9, top_p=1.0, max_tokens=64, seed=0,
        out_dir=tmp_path, model="dummy-model",
    )
    assert (tmp_path / "summary.md").exists()
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "dummy-model" in text
