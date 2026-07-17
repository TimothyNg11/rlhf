"""YAML run-config loader with `inherit:`-based deep merge."""

from __future__ import annotations

from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` over `base`. Dicts merge key-by-key;
    scalars and lists are replaced wholesale by the override's value."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict:
    """Load a YAML run config, resolving a top-level `inherit: <relative path>` key.

    `inherit` is resolved relative to the directory of the file that declares it,
    and may chain across multiple files. The current file's keys are deep-merged
    over the inherited config (dicts merge recursively; scalars/lists are
    replaced). Returns a plain dict; the `inherit` key itself is never present
    in the result.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    inherit = cfg.pop("inherit", None)
    if inherit is not None:
        parent_path = path.parent / inherit
        parent_cfg = load_config(parent_path)
        cfg = _deep_merge(parent_cfg, cfg)

    return cfg
