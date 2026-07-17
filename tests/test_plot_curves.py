"""Tests for scripts/plot_curves.py (importlib pattern, mirroring
tests/test_train_cli.py). Builds synthetic metrics.jsonl files with the real
train/eval row schema (see grpo_math.trainer.loop) via MetricsLogger."""

import importlib.util
import math
from pathlib import Path

import pytest

from grpo_math.trainer.metrics import MetricsLogger

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_script_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(path: Path, *, reward_start: float, pass_at_1_start: float) -> None:
    with MetricsLogger(path) as logger:
        for i, step in enumerate([1, 2, 3]):
            logger.log(
                {
                    "kind": "train",
                    "step": step,
                    "reward_mean": reward_start + 0.1 * i,
                    "kl_mean": 0.01 * step,
                    # First row's entropy_mean is None to prove null-tolerance.
                    "entropy_mean": None if i == 0 else 1.0 + 0.1 * i,
                    "response_len_mean": 100 + 10 * i,
                    "parse_rate": 0.9 + 0.01 * i,
                }
            )
        for step, pass_at_1 in [(0, pass_at_1_start), (2, pass_at_1_start + 0.1)]:
            logger.log(
                {
                    "kind": "eval",
                    "step": step,
                    "benchmark": "gsm8k_dev",
                    "pass_at_1": pass_at_1,
                    "ci_lo": pass_at_1 - 0.05,
                    "ci_hi": pass_at_1 + 0.05,
                    "parse_rate": 0.95,
                }
            )


@pytest.fixture
def two_runs(tmp_path):
    run_a = tmp_path / "runA" / "metrics.jsonl"
    run_b = tmp_path / "runB" / "metrics.jsonl"
    _write_run(run_a, reward_start=0.1, pass_at_1_start=0.2)
    _write_run(run_b, reward_start=0.2, pass_at_1_start=0.3)
    return run_a, run_b


def test_plot_writes_png(two_runs, tmp_path):
    plot_curves = _load_script_module("plot_curves_cli", "plot_curves.py")
    run_a, run_b = two_runs
    out_png = tmp_path / "out" / "curves.png"

    exit_code = plot_curves.main(
        [
            "--runs",
            str(run_a),
            str(run_b),
            "--labels",
            "runA,runB",
            "--out",
            str(out_png),
        ]
    )

    assert exit_code == 0
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_label_count_mismatch_errors(two_runs, tmp_path):
    plot_curves = _load_script_module("plot_curves_cli2", "plot_curves.py")
    run_a, run_b = two_runs
    out_png = tmp_path / "curves.png"

    with pytest.raises(SystemExit):
        plot_curves.main(
            [
                "--runs",
                str(run_a),
                str(run_b),
                "--labels",
                "onlyone",
                "--out",
                str(out_png),
            ]
        )


def test_ema_helper_golden():
    plot_curves = _load_script_module("plot_curves_cli3", "plot_curves.py")
    result = plot_curves._ema([1.0, 2.0, 3.0], 0.5)
    expected = [1.0, 1.5, 2.25]
    assert len(result) == len(expected)
    for got, want in zip(result, expected):
        assert math.isclose(got, want, rel_tol=1e-9)
