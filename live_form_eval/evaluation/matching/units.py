"""The units a free text is read by: the items of a list (one per line), or the sentences of a text.

The comparison (matching/free_text.py) counts them, and the NLI judge reads a text that is too long for it unit
by unit, in windows (matching/judge/nli.py). Both cut a text here, so they always mean the same units.
"""

from __future__ import annotations

import re
from typing import Any

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9\"'(])")
_BULLET = re.compile(r"^\s*(?:[-*•–]|\d+[.)])\s+")


def split_items(text: str) -> list[str]:
    """A list field's items: one per line, without bullets or numbering."""
    return [item for item in (_BULLET.sub("", line).strip() for line in text.splitlines()) if item]


def split_sentences(text: str) -> list[str]:
    """A text's sentences: split at line breaks (bullets dropped), and after . ! ? when a new sentence starts."""
    return [part.strip() for line in split_items(text) for part in _SENTENCE_END.split(line) if part.strip()]


def units_of(text: Any, is_list: bool) -> list[str]:
    """What a shown text is judged by: its items (a list field) or its sentences."""
    if text is None:
        return []
    return split_items(str(text)) if is_list else split_sentences(str(text))
