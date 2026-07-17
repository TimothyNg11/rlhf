"""CLI smoke tests for scripts/run_train.py (importlib pattern, mirroring
tests/test_runner.py). The dry-run path never constructs policy/rollout/trainer,
so these run in the vllm/transformers-free dev venv."""

import importlib.util
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_script_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_creates_run_dir_and_config(tmp_path):
    run_train = _load_script_module("run_train_cli", "run_train.py")
    exit_code = run_train.main(
        ["--config", str(REPO_ROOT / "configs" / "g1_100.yaml"), "--dry-run", "--out-dir", str(tmp_path)]
    )
    assert exit_code == 0

    config_path = tmp_path / "g1-seed0" / "config_resolved.yaml"
    assert config_path.exists()
    resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert resolved["run_name"] == "g1"
    assert resolved["max_steps"] == 100


def test_dry_run_seed_and_max_steps_overrides(tmp_path):
    run_train = _load_script_module("run_train_cli2", "run_train.py")
    exit_code = run_train.main(
        [
            "--config",
            str(REPO_ROOT / "configs" / "g1_100.yaml"),
            "--dry-run",
            "--out-dir",
            str(tmp_path),
            "--seed",
            "1",
            "--max-steps",
            "7",
        ]
    )
    assert exit_code == 0

    config_path = tmp_path / "g1-seed1" / "config_resolved.yaml"
    assert config_path.exists()
    resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert resolved["max_steps"] == 7
    assert resolved["train"]["seed"] == 1


def test_env_var_set():
    _load_script_module("run_train_cli3", "run_train.py")
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
