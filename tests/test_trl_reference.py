"""Tests for scripts/run_trl_reference.py (importlib pattern, mirroring
tests/test_train_cli.py). trl/transformers/datasets are absent in this dev
venv, so every test here must stay on the argparse/reward-fn surface that
imports cleanly without them."""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_script_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_and_parser():
    run_trl_reference = _load_script_module("run_trl_reference_cli", "run_trl_reference.py")
    parser = run_trl_reference.build_parser()
    args = parser.parse_args(["--config", "x"])
    assert args.config == "x"
    assert args.seed == 0
    assert args.out_dir == "results/train"
    assert args.max_steps is None


def test_main_without_trl_errors_cleanly(tmp_path, capsys):
    # trl is not installed in this dev venv, so main() must fail with a clean
    # install hint (and returns 1) BEFORE it ever tries to load gsm8k (no
    # network / no `datasets` package here either).
    run_trl_reference = _load_script_module("run_trl_reference_cli2", "run_trl_reference.py")
    exit_code = run_trl_reference.main(
        [
            "--config",
            str(REPO_ROOT / "configs" / "g1_100.yaml"),
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "pip install -e .[trl]" in out


def test_reward_fn_conversational_and_plain():
    run_trl_reference = _load_script_module("run_trl_reference_cli3", "run_trl_reference.py")
    reward_fn = run_trl_reference.reward_fn_factory(0.2)

    completions = [
        [{"role": "assistant", "content": r"blah \boxed{42}"}],  # conversational, correct
        r"blah \boxed{7}",  # plain string, wrong answer -> format bonus
        "no boxed answer here",  # unparseable
    ]
    gold = ["42", "42", "42"]

    rewards = reward_fn(completions, gold)
    assert rewards == [1.0, 0.2, 0.0]


def test_derive_trl_batch_shape_base_shape():
    # base.yaml shape: ppo_mini_batch_size 32, group_size 4, prompts_per_step 64.
    run_trl_reference = _load_script_module("run_trl_reference_cli4", "run_trl_reference.py")
    cfg = {"train": {"ppo_mini_batch_size": 32, "prompts_per_step": 64}, "rollout": {"group_size": 4}}
    assert run_trl_reference.derive_trl_batch_shape(cfg) == (16, 8, 2)


def test_derive_trl_batch_shape_ablation_g8_shape():
    # ablation_g8.yaml shape: ppo_mini_batch_size 32, group_size 8, prompts_per_step 32.
    run_trl_reference = _load_script_module("run_trl_reference_cli5", "run_trl_reference.py")
    cfg = {"train": {"ppo_mini_batch_size": 32, "prompts_per_step": 32}, "rollout": {"group_size": 8}}
    assert run_trl_reference.derive_trl_batch_shape(cfg) == (16, 16, 1)
