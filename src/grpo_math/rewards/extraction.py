"""Boxed-answer extraction from raw model completions."""


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
