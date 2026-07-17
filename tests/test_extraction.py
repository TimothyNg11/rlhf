"""Unit tests for extract_boxed (no math-verify involved)."""

from grpo_math.rewards.extraction import extract_boxed


def test_simple():
    assert extract_boxed(r"The answer is \boxed{42}.") == "42"


def test_nested_braces():
    assert extract_boxed(r"So \boxed{\frac{1}{2}} is the answer.") == r"\frac{1}{2}"


def test_deeply_nested_braces():
    assert extract_boxed(r"\boxed{\frac{1}{\frac{2}{3}}}") == r"\frac{1}{\frac{2}{3}}"


def test_brace_less_form():
    assert extract_boxed(r"\boxed 5") == "5"


def test_last_of_many():
    text = r"First \boxed{1} then \boxed{2} and finally \boxed{3}."
    assert extract_boxed(text) == "3"


def test_last_of_many_mixed_forms():
    text = r"\boxed 1 then \boxed{2}"
    assert extract_boxed(text) == "2"


def test_unbalanced_returns_none():
    assert extract_boxed(r"\boxed{unbalanced") is None


def test_absent_returns_none():
    assert extract_boxed("no boxed answer here") is None


def test_empty_string_returns_none():
    assert extract_boxed("") is None


def test_empty_boxed_content():
    assert extract_boxed(r"\boxed{}") == ""


def test_boxed_with_surrounding_latex():
    text = r"\text{The final answer is } \boxed{\dfrac{3}{4}} \text{.}"
    assert extract_boxed(text) == r"\dfrac{3}{4}"
