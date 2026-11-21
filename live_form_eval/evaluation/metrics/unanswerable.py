"""Unanswerable accuracy: the share of fields that should stay empty and did.

Counted over free-text fields whose gold is empty: questions the meeting doesn't answer, and lists with no item.
Anything written in such a field is an invented answer.
"""

from __future__ import annotations

from collections.abc import Iterable

from live_form_eval.evaluation.timeline.free_text import FieldSupport


def unanswerable_accuracy(fields: Iterable[FieldSupport]) -> tuple[int, int]:
    """(fields left empty, fields that should be empty)."""
    empty = [support for support in fields if support.should_be_empty]
    return sum(support.shown_units == 0 for support in empty), len(empty)
