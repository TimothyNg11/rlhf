"""Tests for scripts/sample_audit.py: stratified sampling + markdown audit rendering."""

import gzip
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sample_audit = _load_module("sample_audit", "sample_audit.py")


def _make_records(n_correct: int, n_incorrect: int) -> list[dict]:
    records = []
    for i in range(n_correct):
        records.append(
            {
                "problem_id": f"c{i}",
                "sample_idx": 0,
                "completion": "x" * 500,
                "finish_reason": "stop",
                "n_tokens": 10,
                "extracted": "4",
                "gold": "4",
                "verdict": 1.0,
                "parseable": True,
            }
        )
    for i in range(n_incorrect):
        records.append(
            {
                "problem_id": f"w{i}",
                "sample_idx": 0,
                "completion": "y" * 500,
                "finish_reason": "stop",
                "n_tokens": 10,
                "extracted": "5",
                "gold": "4",
                "verdict": 0.0,
                "parseable": True,
            }
        )
    return records


def _write_gz(path: Path, records: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_records_round_trip(tmp_path):
    records = _make_records(3, 3)
    path = tmp_path / "samples.jsonl.gz"
    _write_gz(path, records)
    loaded = sample_audit.load_records(path)
    assert len(loaded) == 6
    assert {r["problem_id"] for r in loaded} == {r["problem_id"] for r in records}


def test_stratified_sample_balanced_when_enough_of_both():
    records = _make_records(20, 20)
    sampled = sample_audit.stratified_sample(records, n=10, seed=0)
    assert len(sampled) == 10
    n_correct = sum(1 for r in sampled if r["verdict"] == 1.0)
    n_incorrect = sum(1 for r in sampled if r["verdict"] != 1.0)
    assert n_correct == 5
    assert n_incorrect == 5


def test_stratified_sample_tops_up_when_one_class_short():
    records = _make_records(2, 20)
    sampled = sample_audit.stratified_sample(records, n=10, seed=0)
    assert len(sampled) == 10
    n_correct = sum(1 for r in sampled if r["verdict"] == 1.0)
    n_incorrect = sum(1 for r in sampled if r["verdict"] != 1.0)
    assert n_correct == 2  # all available correct samples used
    assert n_incorrect == 8  # topped up from the incorrect pool


def test_stratified_sample_clips_to_available_pool():
    records = _make_records(2, 3)
    sampled = sample_audit.stratified_sample(records, n=50, seed=0)
    assert len(sampled) == 5


def test_stratified_sample_deterministic():
    records = _make_records(20, 20)
    ids1 = [r["problem_id"] for r in sample_audit.stratified_sample(records, n=10, seed=0)]
    ids2 = [r["problem_id"] for r in sample_audit.stratified_sample(records, n=10, seed=0)]
    assert ids1 == ids2


def test_stratified_sample_different_seed_can_differ():
    records = _make_records(20, 20)
    ids1 = [r["problem_id"] for r in sample_audit.stratified_sample(records, n=10, seed=0)]
    ids2 = [r["problem_id"] for r in sample_audit.stratified_sample(records, n=10, seed=1)]
    assert ids1 != ids2


def test_stratified_sample_without_replacement():
    records = _make_records(5, 5)
    sampled = sample_audit.stratified_sample(records, n=10, seed=0)
    ids = [r["problem_id"] for r in sampled]
    assert len(ids) == len(set(ids))


def test_render_markdown_contains_expected_fields():
    records = _make_records(1, 1)
    md = sample_audit.render_markdown(records)
    assert md.count("auditor_note:") == len(records)
    for r in records:
        assert r["problem_id"] in md
        assert r["gold"] in md


def test_render_markdown_last_400_chars_only():
    records = _make_records(1, 0)
    records[0]["completion"] = "a" * 500 + "TAIL_MARKER"
    md = sample_audit.render_markdown(records)
    tail_slice = ("a" * 500 + "TAIL_MARKER")[-400:]
    assert tail_slice in md
    assert "a" * 500 not in md  # the full untruncated run must not appear


def test_cli_main_writes_output(tmp_path):
    records = _make_records(4, 4)
    samples_path = tmp_path / "samples.jsonl.gz"
    _write_gz(samples_path, records)
    out_path = tmp_path / "audit.md"

    exit_code = sample_audit.main(
        ["--results", str(samples_path), "--n", "6", "--seed", "0", "--out", str(out_path)]
    )

    assert exit_code == 0
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "auditor_note:" in content


def test_cli_main_default_n_and_seed(tmp_path):
    records = _make_records(30, 30)
    samples_path = tmp_path / "samples.jsonl.gz"
    _write_gz(samples_path, records)
    out_path = tmp_path / "audit.md"

    exit_code = sample_audit.main(["--results", str(samples_path), "--out", str(out_path)])

    assert exit_code == 0
    content = out_path.read_text(encoding="utf-8")
    assert content.count("auditor_note:") == 50  # default --n
