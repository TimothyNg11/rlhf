"""Tests for grpo_math.config: YAML loading + inherit-based deep merge."""

from pathlib import Path

import pytest
import yaml

from grpo_math.config import load_config

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

REQUIRED_KEYS = {"run_name", "model", "train", "rollout"}


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_base_config_no_inherit():
    cfg = load_config(CONFIGS_DIR / "base.yaml")
    assert cfg["run_name"] == "base"
    assert cfg["model"]["name"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg["train"]["lr"] == 2.0e-6
    assert cfg["train"]["micro_batch_size"] == 2
    assert cfg["train"]["use_tis"] is True
    assert cfg["train"]["tis_cap"] == 2.0
    assert cfg["train"]["entropy_abort_threshold"] == 2.0
    assert cfg["train"]["format_bonus"] == 0.2
    assert cfg["train"]["max_grad_norm"] == 1.0
    assert cfg["train"]["save_every"] == 80
    assert cfg["rollout"]["gpu_memory_utilization"] == 0.6
    assert cfg["eval_during_training"]["temperature"] == 0.6
    assert cfg["eval_during_training"]["top_p"] == 0.95
    assert cfg["eval_during_training"]["max_tokens"] == 3072
    assert cfg["eval_during_training"]["seed"] == 0


def test_inherit_merges_scalars_and_overrides(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.dump(
            {
                "run_name": "base",
                "train": {"lr": 1e-6, "seed": 0},
                "rollout": {"group_size": 8},
            }
        )
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        yaml.dump(
            {
                "inherit": "base.yaml",
                "run_name": "child",
                "train": {"seed": 42},
            }
        )
    )
    cfg = load_config(child)
    assert cfg["run_name"] == "child"  # overridden
    assert cfg["train"]["lr"] == 1e-6  # inherited, untouched
    assert cfg["train"]["seed"] == 42  # overridden
    assert cfg["rollout"]["group_size"] == 8  # inherited, untouched
    assert "inherit" not in cfg  # inherit key itself is not leaked into the result


def test_inherit_replaces_lists_not_merges(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.dump({"benchmarks": ["math500", "amc23"]}))
    child = tmp_path / "child.yaml"
    child.write_text(yaml.dump({"inherit": "base.yaml", "benchmarks": ["aime24"]}))
    cfg = load_config(child)
    assert cfg["benchmarks"] == ["aime24"]


def test_inherit_chain_two_levels(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.dump({"train": {"lr": 1e-6, "kl_coef": 0.001}}))
    mid = tmp_path / "mid.yaml"
    mid.write_text(yaml.dump({"inherit": "base.yaml", "train": {"kl_coef": 0.0}}))
    leaf = tmp_path / "leaf.yaml"
    leaf.write_text(yaml.dump({"inherit": "mid.yaml", "train": {"seed": 5}}))
    cfg = load_config(leaf)
    assert cfg["train"]["lr"] == 1e-6
    assert cfg["train"]["kl_coef"] == 0.0
    assert cfg["train"]["seed"] == 5


@pytest.mark.parametrize(
    "path",
    sorted(CONFIGS_DIR.rglob("*.yaml")),
    ids=lambda p: str(p.relative_to(CONFIGS_DIR)),
)
def test_every_config_loads_without_error(path):
    load_config(path)  # must not raise (includes configs/stretch/)


# eval.yaml and eval_milestone.yaml (inherit: eval.yaml) are eval-sweep
# parameter files, not training run configs -- they have no
# run_name/model/train/rollout (see their own required-keys checks below), so
# the run-config required-keys check excludes them.
_EVAL_CONFIG_NAMES = {"eval.yaml", "eval_milestone.yaml"}
_RUN_CONFIG_PATHS = sorted(p for p in CONFIGS_DIR.glob("*.yaml") if p.name not in _EVAL_CONFIG_NAMES)


@pytest.mark.parametrize("path", _RUN_CONFIG_PATHS, ids=lambda p: p.name)
def test_every_run_config_has_required_keys(path):
    cfg = load_config(path)
    missing = REQUIRED_KEYS - cfg.keys()
    assert not missing, f"{path.name} missing required keys: {missing}"


def test_eval_config_has_its_own_required_keys():
    cfg = load_config(CONFIGS_DIR / "eval.yaml")
    expected = {
        "temperature",
        "top_p",
        "max_tokens",
        "kv_cache_dtype",
        "k_default",
        "k_final_aime",
        "seed",
        "benchmarks",
    }
    missing = expected - cfg.keys()
    assert not missing, f"eval.yaml missing keys: {missing}"


def test_eval_milestone_config_has_its_own_required_keys():
    cfg = load_config(CONFIGS_DIR / "eval_milestone.yaml")
    expected = {
        "temperature",
        "top_p",
        "max_tokens",
        "kv_cache_dtype",
        "k_default",
        "k_final_aime",
        "seed",
        "benchmarks",
    }
    missing = expected - cfg.keys()
    assert not missing, f"eval_milestone.yaml missing keys: {missing}"
    assert cfg["benchmarks"] == ["gsm8k"]
    assert cfg["k_default"] == 4


def test_g2_main_resolves_lenient_extraction_and_difficulty_filter():
    cfg = load_config(CONFIGS_DIR / "g2_main.yaml")
    assert cfg["train"]["extraction_mode"] == "lenient"
    assert cfg["train"]["format_bonus"] == 0.0
    assert cfg["train"]["lr_schedule"] == "cosine"
    assert cfg["train"]["kl_coef"] == 0.006
    assert cfg["rollout"]["group_size"] == 8
    assert cfg["data"]["difficulty_band"] == [0.125, 0.875]
    assert cfg["data"]["difficulty_map"] == "results/difficulty/qwen2.5-0.5b_k8/map.jsonl"
    # inherited from g1_robust -> g1_100 -> base
    assert cfg["train"]["truncation_mode"] == "mask"
    assert cfg["train"]["lr"] == 1.0e-6


def test_g2_15b_reparents_onto_g2_main():
    cfg = load_config(CONFIGS_DIR / "g2_15b.yaml")
    assert cfg["model"]["name"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg["train"]["prompts_per_step"] == 32
    assert cfg["train"]["ppo_mini_batch_size"] == 16
    assert cfg["rollout"]["gpu_memory_utilization"] == 0.5
    assert cfg["data"]["difficulty_map"] == "results/difficulty/qwen2.5-1.5b_k8/map.jsonl"
    # inherited from g2_main, untouched
    assert cfg["train"]["extraction_mode"] == "lenient"
    assert cfg["data"]["difficulty_band"] == [0.125, 0.875]
    assert cfg["rollout"]["group_size"] == 8


def test_configs_dir_is_nonempty():
    assert list(CONFIGS_DIR.glob("*.yaml"))


def test_ablation_kl_off_resolves_reparented_onto_g1_robust():
    cfg = load_config(CONFIGS_DIR / "ablation_kl_off.yaml")
    assert cfg["train"]["kl_coef"] == 0.0
    assert cfg["train"]["lr"] == 1.0e-6
    assert cfg["train"]["truncation_mode"] == "mask"
    assert cfg["max_steps"] == 100
    assert cfg["rollout"]["group_size"] == 4


def test_ablation_g8_resolves_reparented_onto_g1_robust():
    cfg = load_config(CONFIGS_DIR / "ablation_g8.yaml")
    assert cfg["rollout"]["group_size"] == 8
    assert cfg["train"]["prompts_per_step"] == 32
    assert cfg["train"]["ppo_mini_batch_size"] == 16
    assert cfg["train"]["lr"] == 1.0e-6
    assert cfg["train"]["kl_coef"] == 0.003
    assert cfg["train"]["truncation_mode"] == "mask"
