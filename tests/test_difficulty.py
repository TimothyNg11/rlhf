"""Tests for grpo_math.data.difficulty: raw-count difficulty-map loading +
inclusive-band filtering used to restrict G2 training data (configs/g2_main.yaml).
"""

import json

import pytest

from grpo_math.data.difficulty import filter_by_band, load_difficulty_map
from grpo_math.eval.benchmarks import EvalProblem


def _write_map(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _row(problem_id, n_correct_lenient, k=8, **overrides):
    row = {
        "problem_id": problem_id,
        "split": "train",
        "k": k,
        "n_correct_lenient": n_correct_lenient,
        "n_correct_strict": n_correct_lenient,
        "n_truncated": 0,
        "methods": {},
    }
    row.update(overrides)
    return row


def _problems(ids):
    return [EvalProblem(problem_id=pid, prompt="", gold="", metadata={}) for pid in ids]


# --- load_difficulty_map ------------------------------------------------------


def test_load_difficulty_map_computes_rate(tmp_path):
    path = tmp_path / "map.jsonl"
    _write_map(path, [_row("p1", 4, k=8), _row("p2", 0, k=8)])
    rates = load_difficulty_map(path)
    assert rates == {"p1": 0.5, "p2": 0.0}


def test_load_difficulty_map_handles_varying_k(tmp_path):
    path = tmp_path / "map.jsonl"
    _write_map(path, [_row("p1", 2, k=4), _row("p2", 8, k=8)])
    rates = load_difficulty_map(path)
    assert rates["p1"] == pytest.approx(0.5)
    assert rates["p2"] == pytest.approx(1.0)


def test_load_difficulty_map_skips_blank_lines(tmp_path):
    path = tmp_path / "map.jsonl"
    path.write_text(json.dumps(_row("p1", 4, k=8)) + "\n\n", encoding="utf-8")
    rates = load_difficulty_map(path)
    assert rates == {"p1": 0.5}


# --- filter_by_band -----------------------------------------------------------


def test_filter_by_band_inclusive_edges():
    ids = [f"p{i}" for i in range(9)]
    problems = _problems(ids)
    rates = {pid: i / 8 for i, pid in enumerate(ids)}  # 0, 0.125, ..., 1.0
    kept = filter_by_band(problems, rates, (0.125, 0.875))
    assert [p.problem_id for p in kept] == ids[1:8]  # 0.125..0.875, both ends included


def test_filter_by_band_excludes_outside_band():
    problems = _problems(["p1", "p2", "p3"])
    rates = {"p1": 0.0, "p2": 0.5, "p3": 1.0}
    kept = filter_by_band(problems, rates, (0.25, 0.75))
    assert [p.problem_id for p in kept] == ["p2"]


def test_filter_by_band_missing_id_lists_ids_in_error():
    problems = _problems(["p1", "p2", "p3"])
    rates = {"p1": 0.5}
    with pytest.raises(ValueError, match="p2"):
        filter_by_band(problems, rates, (0.0, 1.0))


def test_filter_by_band_missing_id_lists_all_missing_ids():
    problems = _problems(["p1", "p2", "p3"])
    rates = {"p1": 0.5}
    with pytest.raises(ValueError, match="p3"):
        filter_by_band(problems, rates, (0.0, 1.0))


def test_filter_by_band_no_missing_ids_does_not_raise():
    problems = _problems(["p1", "p2"])
    rates = {"p1": 0.5, "p2": 0.5}
    kept = filter_by_band(problems, rates, (0.0, 1.0))
    assert len(kept) == 2


def test_import_difficulty_module_is_cpu_pure():
    # no torch/vllm/datasets import at module level -- must import cleanly.
    import grpo_math.data.difficulty  # noqa: F401
