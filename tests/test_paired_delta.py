"""Tests for scripts/paired_delta.py (importlib pattern, mirroring
tests/test_plot_curves.py): per-problem aggregation and the paired bootstrap
delta CI between a base and a candidate eval run."""

import gzip
import importlib.util
import json
import re
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


def _write_samples(
    root: Path,
    benchmark: str,
    verdicts_by_problem: dict,
    parseable: bool = True,
    lenient_verdicts_by_problem: dict | None = None,
):
    """Write `<root>/<benchmark>/samples.jsonl.gz` from {problem_id: [verdicts]},
    mirroring runner.py's writer. ``lenient_verdicts_by_problem``, if given, adds
    a `verdict_lenient` field per record (omitted entirely otherwise, so tests
    can exercise the "old run, no lenient field" error path)."""
    bench_dir = root / benchmark
    bench_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(bench_dir / "samples.jsonl.gz", "wt", encoding="utf-8") as f:
        for problem_id, verdicts in verdicts_by_problem.items():
            lenient_verdicts = (lenient_verdicts_by_problem or {}).get(problem_id)
            for sample_idx, verdict in enumerate(verdicts):
                record = {
                    "problem_id": problem_id,
                    "sample_idx": sample_idx,
                    "verdict": verdict,
                    "parseable": parseable,
                }
                if lenient_verdicts is not None:
                    record["verdict_lenient"] = lenient_verdicts[sample_idx]
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


# --- --metric lenient -----------------------------------------------------------


def test_metric_lenient_reads_verdict_lenient(tmp_path):
    base_root = tmp_path / "base"
    candidate_root = tmp_path / "candidate"
    _write_samples(
        base_root, "gsm8k", {"p1": [1.0, 0.0], "p2": [0.0, 0.0]},
        lenient_verdicts_by_problem={"p1": [1.0, 1.0], "p2": [1.0, 0.0]},
    )
    _write_samples(
        candidate_root, "gsm8k", {"p1": [1.0, 1.0], "p2": [1.0, 0.0]},
        lenient_verdicts_by_problem={"p1": [1.0, 1.0], "p2": [1.0, 1.0]},
    )

    result = compute_paired(base_root, candidate_root, "gsm8k", metric="lenient")

    assert result["base_pass_at_1"] == pytest.approx(np.mean([1.0, 0.5]))
    assert result["candidate_pass_at_1"] == pytest.approx(np.mean([1.0, 1.0]))


def test_metric_lenient_missing_field_errors_clearly(tmp_path):
    base_root = tmp_path / "base"
    candidate_root = tmp_path / "candidate"
    _write_samples(base_root, "gsm8k", {"p1": [1.0, 0.0]})  # no lenient verdicts written
    _write_samples(candidate_root, "gsm8k", {"p1": [1.0, 1.0]})

    with pytest.raises(ValueError, match="regrade_lenient"):
        compute_paired(base_root, candidate_root, "gsm8k", metric="lenient")


def test_metric_lenient_error_names_run_dir(tmp_path):
    base_root = tmp_path / "base"
    candidate_root = tmp_path / "candidate"
    _write_samples(base_root, "gsm8k", {"p1": [1.0, 0.0]})
    _write_samples(candidate_root, "gsm8k", {"p1": [1.0, 1.0]})

    with pytest.raises(ValueError, match=re.escape(str(base_root))):
        compute_paired(base_root, candidate_root, "gsm8k", metric="lenient")


def test_metric_strict_is_unaffected_by_missing_lenient_field(tmp_path):
    base_root = tmp_path / "base"
    candidate_root = tmp_path / "candidate"
    _write_samples(base_root, "gsm8k", {"p1": [1.0, 0.0]})
    _write_samples(candidate_root, "gsm8k", {"p1": [1.0, 1.0]})

    result = compute_paired(base_root, candidate_root, "gsm8k", metric="strict")
    assert result["base_pass_at_1"] == pytest.approx(0.5)


# --- --agg any --------------------------------------------------------------------


def test_agg_any_computes_pass_at_k(tmp_path):
    base_root = tmp_path / "base"
    candidate_root = tmp_path / "candidate"
    _write_samples(base_root, "gsm8k", {"p1": [0.0, 0.0], "p2": [1.0, 0.0]})
    _write_samples(candidate_root, "gsm8k", {"p1": [1.0, 0.0], "p2": [1.0, 1.0]})

    result = compute_paired(base_root, candidate_root, "gsm8k", agg="any")

    assert result["base_pass_at_1"] == pytest.approx(0.5)  # p1 fails, p2 passes any-of-k
    assert result["candidate_pass_at_1"] == pytest.approx(1.0)  # both pass


def test_agg_mean_is_default_and_matches_existing_behavior(tmp_path):
    base_root = tmp_path / "base"
    candidate_root = tmp_path / "candidate"
    _write_samples(base_root, "gsm8k", {"p1": [1.0, 0.0]})
    _write_samples(candidate_root, "gsm8k", {"p1": [1.0, 1.0]})

    default_result = compute_paired(base_root, candidate_root, "gsm8k")
    explicit_result = compute_paired(base_root, candidate_root, "gsm8k", agg="mean")
    assert default_result["base_pass_at_1"] == explicit_result["base_pass_at_1"]
    assert default_result["base_pass_at_1"] == pytest.approx(0.5)


# --- --intersect --------------------------------------------------------------------


def test_intersect_pairs_on_common_ids(tmp_path):
    base_root = tmp_path / "base"
    candidate_root = tmp_path / "candidate"
    _write_samples(base_root, "gsm8k", {"p1": [1.0], "p2": [0.0], "p3": [1.0]})
    _write_samples(candidate_root, "gsm8k", {"p1": [1.0], "p2": [1.0], "p4": [0.0]})

    result = compute_paired(base_root, candidate_root, "gsm8k", intersect=True)

    assert result["n_problems"] == 2  # p1, p2 shared; p3, p4 dropped


def test_intersect_no_common_ids_raises(tmp_path):
    base_root = tmp_path / "base"
    candidate_root = tmp_path / "candidate"
    _write_samples(base_root, "gsm8k", {"p1": [1.0]})
    _write_samples(candidate_root, "gsm8k", {"p2": [1.0]})

    with pytest.raises(ValueError):
        compute_paired(base_root, candidate_root, "gsm8k", intersect=True)


def test_without_intersect_mismatched_sets_still_raises(tmp_path):
    # --intersect is opt-in: default behavior (error on mismatched sets) is unchanged.
    base_root = tmp_path / "base"
    candidate_root = tmp_path / "candidate"
    _write_samples(base_root, "gsm8k", {"a": [1.0], "b": [0.0], "c": [1.0]})
    _write_samples(candidate_root, "gsm8k", {"a": [1.0], "b": [0.0], "d": [1.0]})

    with pytest.raises(ValueError):
        compute_paired(base_root, candidate_root, "gsm8k", intersect=False)


def test_main_intersect_prints_n(tmp_path, capsys):
    base_root = tmp_path / "base"
    candidate_root = tmp_path / "candidate"
    _write_samples(base_root, "gsm8k", {"p1": [1.0], "p2": [0.0], "p3": [1.0]})
    _write_samples(candidate_root, "gsm8k", {"p1": [1.0], "p2": [1.0], "p4": [0.0]})

    exit_code = paired_delta.main(
        [
            "--base", str(base_root), "--candidate", str(candidate_root),
            "--benchmark", "gsm8k", "--intersect",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert re.search(r"n_problems:\s+2", captured.out)


def test_main_accepts_metric_and_agg_flags(tmp_path, capsys):
    base_root = tmp_path / "base"
    candidate_root = tmp_path / "candidate"
    _write_samples(base_root, "gsm8k", {"p1": [1.0, 0.0]})
    _write_samples(candidate_root, "gsm8k", {"p1": [1.0, 1.0]})

    exit_code = paired_delta.main(
        [
            "--base", str(base_root), "--candidate", str(candidate_root),
            "--benchmark", "gsm8k", "--metric", "strict", "--agg", "any",
        ]
    )
    assert exit_code == 0
