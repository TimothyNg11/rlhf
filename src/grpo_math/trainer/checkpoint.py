"""Checkpoint save/restore for the training loop.

Layout per run: ``<run_dir>/checkpoints/step_XXXX/`` (``XXXX`` = the step
number, zero-padded to 4 digits). ``save_checkpoint`` writes to a temp
directory first and ``os.replace``s it into place so a crash mid-save never
leaves a partial directory that :func:`find_latest_checkpoint` would pick up.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import torch

_TEMP_SUFFIX = ".tmp"
_STEP_DIR_RE = re.compile(r"^step_(\d+)$")


def _step_dir_name(step: int) -> str:
    return f"step_{step:04d}"


def save_checkpoint(
    ckpt_root: str | Path,
    *,
    step: int,
    model,
    optimizer,
    trainer_state: dict,
    tokenizer=None,
    rng_state: dict | None = None,
) -> Path:
    """Write a checkpoint for ``step`` under ``ckpt_root``, returning the
    final ``step_XXXX`` directory.

    Writes to a sibling temp directory (final name + ``.tmp``) and
    ``os.replace``s it to the final name only once every file has been
    written, so a crash mid-save leaves only an ignorable ``.tmp`` dir.
    """
    ckpt_root = Path(ckpt_root)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    final_dir = ckpt_root / _step_dir_name(step)
    tmp_dir = ckpt_root / (_step_dir_name(step) + _TEMP_SUFFIX)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(tmp_dir / "model")
    else:
        torch.save(model.state_dict(), tmp_dir / "model.pt")

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(tmp_dir / "model")

    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    (tmp_dir / "trainer_state.json").write_text(json.dumps(trainer_state, indent=2))

    if rng_state is not None:
        torch.save(rng_state, tmp_dir / "rng.pt")

    os.replace(tmp_dir, final_dir)
    return final_dir


def find_latest_checkpoint(ckpt_root: str | Path) -> Path | None:
    """Return the ``step_*`` dir under ``ckpt_root`` with the highest step
    number, parsed NUMERICALLY (not lexicographically) from the suffix.
    ``None`` if ``ckpt_root`` doesn't exist or has no valid step dirs.

    Directories that don't fully match ``step_<digits>`` (e.g. in-progress
    ``.tmp`` dirs left by an interrupted :func:`save_checkpoint`) are
    ignored.
    """
    ckpt_root = Path(ckpt_root)
    if not ckpt_root.exists():
        return None

    best_dir = None
    best_step = -1
    for entry in ckpt_root.iterdir():
        if not entry.is_dir():
            continue
        match = _STEP_DIR_RE.match(entry.name)
        if not match:
            continue
        step_num = int(match.group(1))
        if step_num > best_step:
            best_step = step_num
            best_dir = entry
    return best_dir


def load_trainer_state(step_dir: str | Path) -> dict:
    step_dir = Path(step_dir)
    return json.loads((step_dir / "trainer_state.json").read_text())


def load_rng_state(step_dir: str | Path) -> dict | None:
    """Return the saved RNG state dict, or ``None`` if this checkpoint has
    no ``rng.pt`` (``rng_state`` was ``None`` at save time)."""
    step_dir = Path(step_dir)
    rng_path = step_dir / "rng.pt"
    if not rng_path.exists():
        return None
    return torch.load(rng_path, weights_only=False)
