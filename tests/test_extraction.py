"""Unit tests for extract_boxed and extract_lenient (no math-verify involved)."""

from grpo_math.rewards.extraction import LenientExtraction, extract_boxed, extract_lenient


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


# --- extract_lenient: fallback chain precedence ---


def test_extract_lenient_boxed_beats_all():
    text = "#### 1\nanswer: 2\nFinal answer is 4\n\\boxed{3}"
    assert extract_lenient(text) == LenientExtraction(value="3", method="boxed")


def test_extract_lenient_hash_beats_answer_is_and_last_number():
    text = "The final answer is 4\n#### 3\n5"
    assert extract_lenient(text) == LenientExtraction(value="3", method="hash")


def test_extract_lenient_answer_is_beats_last_number():
    text = "The final answer is 4. (measured across 99 trials)"
    assert extract_lenient(text) == LenientExtraction(value="4", method="answer_is")


def test_extract_lenient_empty_boxed_falls_through_to_hash():
    text = r"\boxed{}" + "\n#### 72"
    assert extract_lenient(text) == LenientExtraction(value="72", method="hash")


def test_extract_lenient_hash_line():
    assert extract_lenient("Steps...\n#### 72") == LenientExtraction(value="72", method="hash")


def test_extract_lenient_final_answer_is_phrase():
    result = extract_lenient("The final answer is 42.")
    assert result == LenientExtraction(value="42", method="answer_is")


def test_extract_lenient_answer_is_markdown_bold():
    result = extract_lenient("answer: **18**")
    assert result == LenientExtraction(value="18", method="answer_is")


def test_extract_lenient_dollar_comma_trailing_period():
    result = extract_lenient("The total cost is $1,234.")
    assert result == LenientExtraction(value="1234", method="last_number")


def test_extract_lenient_percent():
    result = extract_lenient("Success rate improved to 85%")
    assert result == LenientExtraction(value="85", method="last_number")


def test_extract_lenient_mmlu_letter():
    result = extract_lenient("Answer: C")
    assert result == LenientExtraction(value="C", method="answer_is")


def test_extract_lenient_no_number_returns_none():
    result = extract_lenient("There is no numeric answer in this text at all.")
    assert result == LenientExtraction(value=None, method="none")


def test_extract_lenient_last_of_many_numbers_wins():
    result = extract_lenient("We have 1, then 2, then 3, and finally 4 apples.")
    assert result == LenientExtraction(value="4", method="last_number")


def test_extract_lenient_negative_number():
    result = extract_lenient("The change in temperature was -5 degrees.")
    assert result == LenientExtraction(value="-5", method="last_number")
