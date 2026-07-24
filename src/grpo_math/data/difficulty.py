"""Difficulty-map loading + band filtering for G2's training-data curriculum
(configs/g2_main.yaml): restrict training to problems the base model
sometimes-but-not-always solves, per ``scripts/build_difficulty_map.py``'s
raw-count ``map.jsonl``.

CPU-pure: no torch/vllm/datasets imports, so this module is safe to import
anywhere (including the trainer's module-level imports).
"""

from __future__ import annotations

import json
from pathlib import Path


def load_difficulty_map(path: str | Path) -> dict[str, float]:
    """Load a ``map.jsonl`` file (see ``scripts/build_difficulty_map.py``) into
    a ``{problem_id: rate}`` dict, where ``rate = n_correct_lenient / k``.

    ``map.jsonl`` stores raw counts, not the rate, so any band can be applied
    later (via :func:`filter_by_band`) without re-running generation.
    """
    rates: dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rates[row["problem_id"]] = row["n_correct_lenient"] / row["k"]
    return rates


def filter_by_band(problems: list, rates: dict[str, float], band) -> list:
    """Keep only ``problems`` whose difficulty ``rates[problem.problem_id]``
    falls in ``band``, inclusive on both edges (``band[0] <= rate <= band[1]``).

    Every ``problem.problem_id`` must be present in ``rates``; a stale or
    mismatched difficulty map must not silently train on everything, so any
    missing ids raise ``ValueError`` listing them.
    """
    missing = [p.problem_id for p in problems if p.problem_id not in rates]
    if missing:
        raise ValueError(
            f"{len(missing)} problem_id(s) missing from the difficulty map, "
            f"refusing to silently train on them: {missing}"
        )
    lo, hi = band
    return [p for p in problems if lo <= rates[p.problem_id] <= hi]
