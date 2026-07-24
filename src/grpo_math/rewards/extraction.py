"""Boxed-answer extraction from raw model completions."""

from __future__ import annotations

import re
from dataclasses import dataclass


def extract_boxed(text: str) -> str | None:
    """Return the content of the LAST ``\\boxed{...}`` (or brace-less ``\\boxed x``) in ``text``.

    Handles correctly nested braces inside the boxed content (e.g. ``\\boxed{\\frac{1}{2}}``
    -> ``\\frac{1}{2}``). Occurrences whose braces never close are skipped (they do not count
    as a match); if no valid ``\\boxed`` is found anywhere in ``text``, returns ``None``.
    """
    marker = r"\boxed"
    last_match: str | None = None
    search_from = 0

    while True:
        idx = text.find(marker, search_from)
        if idx == -1:
            break

        after_marker = idx + len(marker)
        j = after_marker
        while j < len(text) and text[j] == " ":
            j += 1

        if j < len(text) and text[j] == "{":
            depth = 0
            k = j
            content_start = j + 1
            end = None
            while k < len(text):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        end = k
                        break
                k += 1
            if end is not None:
                last_match = text[content_start:end]
                search_from = end + 1
            else:
                # Unbalanced: no matching close brace for this occurrence, skip it.
                search_from = after_marker
        elif j < len(text):
            # Brace-less form: \boxed 5 -> take the following run of non-whitespace.
            k = j
            while k < len(text) and not text[k].isspace():
                k += 1
            last_match = text[j:k]
            search_from = k
        else:
            # "\boxed" at the very end of text with nothing following.
            search_from = after_marker

    return last_match


# Bare numeric token: optional sign/currency prefix, digit run (commas allowed as
# thousands separators), optional decimal part, optional trailing percent sign.
_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?%?"

_HASH_RE = re.compile(r"####\s*([^\n]+)")
_ANSWER_IS_RE = re.compile(
    rf"(?:final\s+answer|answer)\s*(?:is|:|=)\s*\**\s*({_NUM}|[A-D]\b)",
    re.IGNORECASE,
)
_LAST_NUMBER_RE = re.compile(_NUM)


def _clean_numeric(raw: str) -> str:
    """Strip whitespace, ``$``, ``%``, thousands-separator commas, and any
    trailing ``.`` from a matched numeric token (e.g. ``"$1,234."`` -> ``"1234"``,
    ``"85%"`` -> ``"85"``)."""
    cleaned = raw.strip().replace(",", "").replace("$", "").replace("%", "")
    return cleaned.rstrip(".")


def _clean_hash(raw: str) -> str:
    """Clean a ``#### ...`` capture GSM8K-gold-style: strip whitespace, drop commas."""
    return raw.strip().replace(",", "")


@dataclass(frozen=True)
class LenientExtraction:
    value: str | None
    method: str  # "boxed" | "hash" | "answer_is" | "last_number" | "none"


def extract_lenient(text: str) -> LenientExtraction:
    """Extract a final answer from ``text`` via a lenient fallback chain.

    Tries, in order: a boxed LaTeX answer (:func:`extract_boxed`), a GSM8K-style
    ``#### `` line, an "answer is/:/=" phrase (a number, or an MMLU A-D letter),
    and finally the last bare number anywhere in ``text``. Each stage takes the
    LAST match in ``text``; a stage whose cleaned value is empty/whitespace
    falls through to the next stage (e.g. ``extract_boxed("\\boxed{}")`` is
    ``""`` and is skipped in favor of a later stage). Returns
    ``LenientExtraction(None, "none")`` if nothing matches anywhere.
    """
    boxed = extract_boxed(text)
    if boxed is not None and boxed.strip():
        return LenientExtraction(value=boxed, method="boxed")

    hash_matches = _HASH_RE.findall(text)
    if hash_matches:
        cleaned = _clean_hash(hash_matches[-1])
        if cleaned:
            return LenientExtraction(value=cleaned, method="hash")

    answer_is_matches = _ANSWER_IS_RE.findall(text)
    if answer_is_matches:
        raw = answer_is_matches[-1]
        value = raw if raw.isalpha() else _clean_numeric(raw)
        if value:
            return LenientExtraction(value=value, method="answer_is")

    last_number_matches = _LAST_NUMBER_RE.findall(text)
    if last_number_matches:
        cleaned = _clean_numeric(last_number_matches[-1])
        if cleaned:
            return LenientExtraction(value=cleaned, method="last_number")

    return LenientExtraction(value=None, method="none")
