"""Golden-file tests for extract_boxed + verify_answer, exercised together.

Cases live in tests/golden/answers.jsonl. Each case's "equivalent" field records
math-verify's actual OBSERVED behavior (verified by direct probing of the
installed math-verify package), not a guess -- if a case is marked "xfail":
true, it's a genuine known checker limitation, kept in the suite honestly
rather than deleted.
"""

import json
from pathlib import Path

import pytest

from grpo_math.rewards.extraction import extract_boxed
from grpo_math.rewards.verify import compute_reward, verify_answer

GOLDEN_PATH = Path(__file__).parent / "golden" / "answers.jsonl"


def _load_cases() -> list[dict]:
    cases = []
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def _case_id(case: dict) -> str:
    return f"{case.get('category', 'case')}-{case['note'][:30]}"


def _build_params():
    params = []
    for case in _load_cases():
        marks = []
        if case.get("xfail"):
            marks.append(pytest.mark.xfail(reason=case["note"], strict=True))
        params.append(pytest.param(case, marks=marks, id=_case_id(case)))
    return params


def test_golden_file_has_expected_case_count():
    cases = _load_cases()
    assert len(cases) >= 50


@pytest.mark.parametrize("case", _build_params())
def test_golden_case(case: dict):
    extracted = extract_boxed(case["pred"])
    result = extracted is not None and verify_answer(extracted, case["gold"])
    assert result == case["equivalent"]


# --- compute_reward: truncation-mode semantics, not exercised by the golden file ---


def test_compute_reward_correct_untruncated():
    result = compute_reward(r"\boxed{42}", "42", truncated=False)
    assert result.reward == 1.0
    assert result.parseable is True
    assert result.masked is False


def test_compute_reward_incorrect_untruncated():
    result = compute_reward(r"\boxed{41}", "42", truncated=False)
    assert result.reward == 0.0
    assert result.parseable is True
    assert result.masked is False


def test_compute_reward_no_boxed_untruncated():
    result = compute_reward("I have no boxed answer", "42", truncated=False)
    assert result.reward == 0.0
    assert result.parseable is False
    assert result.masked is False


def test_compute_reward_truncated_zero_reward_mode():
    # Even a correct answer gets zeroed out and is not masked in this mode.
    result = compute_reward(r"\boxed{42}", "42", truncated=True, truncation_mode="zero_reward")
    assert result.reward == 0.0
    assert result.parseable is True
    assert result.masked is False


def test_compute_reward_truncated_mask_mode():
    result = compute_reward(r"\boxed{42}", "42", truncated=True, truncation_mode="mask")
    assert result.reward == 0.0
    assert result.parseable is True
    assert result.masked is True


def test_compute_reward_unknown_truncation_mode_raises():
    with pytest.raises(ValueError):
        compute_reward(r"\boxed{42}", "42", truncated=True, truncation_mode="bogus")


def test_reward_result_is_frozen():
    result = compute_reward(r"\boxed{42}", "42", truncated=False)
    with pytest.raises(Exception):
        result.reward = 5.0  # type: ignore[misc]


# --- compute_reward: format_bonus semantics ---


def test_format_bonus_parseable_wrong_gets_bonus():
    result = compute_reward(r"\boxed{41}", "42", truncated=False, format_bonus=0.2)
    assert result.reward == pytest.approx(0.2)
    assert result.parseable is True
    assert result.masked is False


def test_format_bonus_correct_still_one():
    # The bonus does not stack on top of a correct answer's reward.
    result = compute_reward(r"\boxed{42}", "42", truncated=False, format_bonus=0.2)
    assert result.reward == 1.0
    assert result.parseable is True
    assert result.masked is False


def test_format_bonus_unparseable_zero():
    result = compute_reward("I have no boxed answer", "42", truncated=False, format_bonus=0.2)
    assert result.reward == 0.0
    assert result.parseable is False
    assert result.masked is False


def test_format_bonus_truncated_zero():
    # A truncated completion never gets the bonus, in either truncation mode.
    zero_reward = compute_reward(
        r"\boxed{41}", "42", truncated=True, truncation_mode="zero_reward", format_bonus=0.2
    )
    assert zero_reward.reward == 0.0
    assert zero_reward.masked is False

    mask = compute_reward(
        r"\boxed{41}", "42", truncated=True, truncation_mode="mask", format_bonus=0.2
    )
    assert mask.reward == 0.0
    assert mask.masked is True


def test_format_bonus_default_zero_backward_compat():
    # Omitting format_bonus preserves the exact pre-bonus binary behavior.
    result = compute_reward(r"\boxed{41}", "42", truncated=False)
    assert result.reward == 0.0
    assert result.parseable is True
    assert result.masked is False
