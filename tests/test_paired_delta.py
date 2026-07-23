"""Tests for scripts/paired_delta.py (importlib pattern, mirroring
tests/test_plot_curves.py): per-problem aggregation and the paired bootstrap
delta CI between a base and a candidate eval run."""

import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_script_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paired_delta = _load_script_module("paired_delta", "paired_delta.py")
compute_paired = paired_delta.compute_paired
load_per_problem_means = paired_delta.load_per_problem_means


def _write_samples(root: Path, benchmark: str, verdicts_by_problem: dict, parseable: bool = True):
    """Write `<root>/<benchmark>/samples.jsonl.gz` from {problem_id: [verdicts]},
    mirroring runner.py's writer."""
    bench_dir = root / benchmark
    bench_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(bench_dir / "samples.jsonl.gz", "wt", encoding="utf-8") as f:
        for problem_id, verdicts in verdicts_by_problem.items():
            for sample_idx, verdict in enumerate(verdicts):
                record = {
                    "problem_id": problem_id,
                    "sample_idx": sample_idx,
                    "verdict": verdict,
                    "parseable": parseable,
                }
                f.write(json.dumps(record) + "\n")


def test_load_per_problem_means(tmp_path):
    _write_samples(
        tmp_path,
        "gsm8k",
        {"p1": [1.0, 0.0], "p2": [1.0, 1.0], "p3": [0.0, 0.0]},
    )
    problem_ids, means, parse_rate = load_per_problem_means(tmp_path, "gsm8k")
    assert problem_ids == ["p1", "p2", "p3"]
    np.testing.assert_allclose(means, [0.5, 1.0, 0.0])
    assert parse_rate == pytest.approx(1.0)


def test_compute_paired_happy_path(tmp_path):
    base_root = tmp_path / "base"
    candidate_root = tmp_path / "candidate"

    _write_samples(
        base_root,
        "gsm8k",
        {"p1": [1.0, 0.0], "p2": [0.0, 0.0], "p3": [1.0, 1.0]},
    )
    _write_samples(
        candidate_root,
        "gsm8k",
        {"p1": [1.0, 1.0], "p2": [1.0, 0.0], "p3": [1.0, 1.0]},
    )

    result = compute_paired(base_root, candidate_root, "gsm8k")

    base_means = np.array([0.5, 0.0, 1.0])
    candidate_means = np.array([1.0, 0.5, 1.0])
    expected_delta = candidate_means.mean() - base_means.mean()

    assert result["n_problems"] == 3
    assert result["delta"] == pytest.approx(expected_delta, abs=1e-9)
    assert result["ci_lo"] <= result["delta"] <= result["ci_hi"]
    assert result["base_pass_at_1"] == pytest.approx(base_means.mean())
    assert result["candidate_pass_at_1"] == pytest.approx(candidate_means.mean())
    assert result["base_parse_rate"] == pytest.approx(1.0)
    assert result["candidate_parse_rate"] == pytest.approx(1.0)


def test_compute_paired_mismatched_problem_sets_raises(tmp_path):
    base_root = tmp_path / "base"
    candidate_root = tmp_path / "candidate"

    _write_samples(base_root, "gsm8k", {"a": [1.0], "b": [0.0], "c": [1.0]})
    _write_samples(candidate_root, "gsm8k", {"a": [1.0], "b": [0.0], "d": [1.0]})

    with pytest.raises(ValueError):
        compute_paired(base_root, candidate_root, "gsm8k")
