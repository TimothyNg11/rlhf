"""math-verify wrapper + binary reward computation for GRPO rollouts."""

from __future__ import annotations

import os
from dataclasses import dataclass

from math_verify import parse, verify

from grpo_math.rewards.extraction import extract_boxed

# math_verify's built-in per-call timeout uses multiprocessing.Process on Windows
# (there is no signal.alarm there). In this environment, process spawn / handle
# duplication is restricted, so every parse()/verify() call raises OSError /
# PermissionError from the child bootstrap and would otherwise be swallowed as a
# false verification failure. Disable the built-in timeout on Windows; POSIX
# uses a cheap in-process signal.alarm timeout and is unaffected, so we keep
# math-verify's own default (5s) there -- this matters once this module runs on
# the Linux training box (see docs/PLAN.md).
_PARSE_TIMEOUT = None if os.name == "nt" else 5
_VERIFY_TIMEOUT = None if os.name == "nt" else 5

_VALID_TRUNCATION_MODES = {"zero_reward", "mask"}


def verify_answer(pred: str, gold: str) -> bool:
    """Check whether ``pred`` is mathematically equivalent to ``gold``.

    Both ``pred`` and ``gold`` are plain answer strings (e.g. already extracted
    via :func:`extract_boxed`), not full completions. Wraps both in ``\\boxed{}``
    before handing them to ``math_verify.parse`` -- required for math-verify's
    LaTeX-aware parsing to correctly disambiguate intervals/sets/tuples/percent/
    thousands-separator notation. Gold is parsed as the target, pred as the
    candidate, per math-verify's documented (non-symmetric) ``verify(gold, answer)``
    argument order. Never raises: any parsing/verification error returns False.
    """
    try:
        gold_parsed = parse(f"\\boxed{{{gold}}}", parsing_timeout=_PARSE_TIMEOUT)
        pred_parsed = parse(f"\\boxed{{{pred}}}", parsing_timeout=_PARSE_TIMEOUT)
        return bool(verify(gold_parsed, pred_parsed, timeout_seconds=_VERIFY_TIMEOUT))
    except Exception:
        return False


@dataclass(frozen=True)
class RewardResult:
    reward: float
    parseable: bool
    masked: bool


def compute_reward(
    completion: str,
    gold: str,
    *,
    truncated: bool,
    truncation_mode: str = "zero_reward",
) -> RewardResult:
    """Binary reward: 1.0 iff a boxed answer is extracted from ``completion`` AND
    verifies against ``gold``, else 0.0.

    If ``truncated``: mode ``"zero_reward"`` forces reward 0.0 (masked False);
    mode ``"mask"`` forces reward 0.0 and masked True (the trainer excludes
    masked samples from the loss). Unknown modes raise ``ValueError``.
    """
    if truncation_mode not in _VALID_TRUNCATION_MODES:
        raise ValueError(
            f"Unknown truncation_mode {truncation_mode!r}; expected one of "
            f"{sorted(_VALID_TRUNCATION_MODES)}"
        )

    extracted = extract_boxed(completion)
    parseable = extracted is not None
    correct = parseable and verify_answer(extracted, gold)
    reward = 1.0 if correct else 0.0

    if truncated:
        masked = truncation_mode == "mask"
        return RewardResult(reward=0.0, parseable=parseable, masked=masked)

    return RewardResult(reward=reward, parseable=parseable, masked=False)
