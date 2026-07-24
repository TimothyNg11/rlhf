"""math-verify wrapper + binary reward computation for GRPO rollouts."""

from __future__ import annotations

import os
from dataclasses import dataclass

from math_verify import parse, verify

from grpo_math.rewards.extraction import extract_boxed, extract_lenient

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
_VALID_EXTRACTION_MODES = {"boxed", "lenient"}


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
    method: str = "none"


def compute_reward(
    completion: str,
    gold: str,
    *,
    truncated: bool,
    truncation_mode: str = "zero_reward",
    format_bonus: float = 0.0,
    extraction_mode: str = "boxed",
) -> RewardResult:
    """Reward: 1.0 iff an answer is extracted from ``completion`` AND verifies
    against ``gold``. Otherwise, if an answer was extracted but is wrong,
    reward is ``format_bonus`` (does NOT stack on top of the 1.0 for correct
    answers). If nothing was extracted, reward is 0.0.

    ``extraction_mode`` selects how the answer is extracted: ``"boxed"``
    (default) uses :func:`extract_boxed` only, matching the original behavior
    byte-for-byte; ``"lenient"`` uses :func:`extract_lenient`'s fallback chain
    (boxed -> ``#### `` -> "answer is" phrase -> last bare number). Unknown
    modes raise ``ValueError``. ``RewardResult.method`` records which stage
    produced the extraction (``"boxed"``, ``"hash"``, ``"answer_is"``,
    ``"last_number"``, or ``"none"``); in ``"boxed"`` mode this is always
    ``"boxed"`` or ``"none"``.

    If ``truncated``: mode ``"zero_reward"`` forces reward 0.0 (masked False);
    mode ``"mask"`` forces reward 0.0 and masked True (the trainer excludes
    masked samples from the loss). A truncated completion never gets the format
    bonus, even if it looks parseable/correct. Unknown truncation modes raise
    ``ValueError``. ``method`` is still recorded on truncated results.

    ``format_bonus`` defaults to 0.0, which preserves the exact binary
    1.0/0.0 behavior of earlier versions of this function -- the eval runner
    never passes it, so its pass@1 semantics are unaffected.
    """
    if truncation_mode not in _VALID_TRUNCATION_MODES:
        raise ValueError(
            f"Unknown truncation_mode {truncation_mode!r}; expected one of "
            f"{sorted(_VALID_TRUNCATION_MODES)}"
        )
    if extraction_mode not in _VALID_EXTRACTION_MODES:
        raise ValueError(
            f"Unknown extraction_mode {extraction_mode!r}; expected one of "
            f"{sorted(_VALID_EXTRACTION_MODES)}"
        )

    if extraction_mode == "lenient":
        lenient = extract_lenient(completion)
        extracted = lenient.value
        method = lenient.method
    else:
        extracted = extract_boxed(completion)
        method = "boxed" if extracted is not None else "none"

    parseable = extracted is not None
    correct = parseable and verify_answer(extracted, gold)

    if truncated:
        masked = truncation_mode == "mask"
        return RewardResult(reward=0.0, parseable=parseable, masked=masked, method=method)

    if correct:
        reward = 1.0
    elif parseable:
        reward = format_bonus
    else:
        reward = 0.0

    return RewardResult(reward=reward, parseable=parseable, masked=False, method=method)
