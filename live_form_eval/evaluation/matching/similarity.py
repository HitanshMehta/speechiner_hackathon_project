"""How close is a shown value to a gold value? A number from 0 (nothing in common) to 1 (the same value).

Used wherever a display is not the gold value itself: to grade it (precision) and to find the closest display of a
gold value the form never showed (completeness, speed). One rule per kind of field:

- written values (text, phone, email, url, time, date): the share of their pairs of consecutive characters that
  the two values have in common, the start of the value marked::

      2 × shared pairs ÷ (pairs of the shown value + pairs of the gold value)

      "0172" → ^0 01 17 72      "0171" → ^0 01 17 71      3 shared      2×3 ÷ (4+4) = 0.75

  The start mark makes two values that begin differently share less; a value that is a part of the other
  ("jacques" for "jacques ung") keeps the pairs of that part.
- choices and amounts (select, checkbox, integer, float, slider): the same value or not.
- multiselect: the same formula on the ticked options.
- An empty field against a value: 0.

Free text is not compared here: a judge decides what a text supports (matching/judge).
"""

from __future__ import annotations

from typing import Any

from live_form_eval.config import MatchingConfig
from live_form_eval.evaluation.matching import normalise
from live_form_eval.evaluation.matching.compare import options_of, same
from live_form_eval.interface.schema import FieldSpec, FieldType

WRITTEN_TYPES = frozenset({FieldType.text, FieldType.phone, FieldType.email, FieldType.url, FieldType.time,
                           FieldType.date, FieldType.datetime})


def pairs(text: str) -> list[str]:
    """The pairs of consecutive characters of ``text``, the first one starting with the start mark "^"."""
    marked = "^" + text
    return [marked[k: k + 2] for k in range(len(marked) - 1)]


def pair_similarity(a: str, b: str) -> float:
    """2 × shared pairs ÷ (pairs of a + pairs of b); a pair that occurs twice counts twice."""
    left, pool = pairs(a), pairs(b)
    total = len(left) + len(pool)
    if not total:
        return 0.0
    shared = 0
    for pair in left:
        if pair in pool:
            pool.remove(pair)
            shared += 1
    return 2 * shared / total


def written(spec: FieldSpec, value: Any, gold: Any) -> str:
    """``value`` as the characters that are compared: the field type's own writing, without formatting.

    A text whose gold holds a digit is an identifier (an ID, a plate): only its letters and digits count, as in
    :func:`compare.same`.
    """
    kind = spec.type
    if kind == FieldType.phone:
        return normalise.digits(value)
    if kind == FieldType.email:
        return normalise.email(value)
    if kind == FieldType.url:
        return normalise.url(value)
    if kind == FieldType.time:
        return normalise.time(value)
    if kind == FieldType.text:
        identifier = any(character.isdigit() for character in str(gold))
        return normalise.squash(value) if identifier else normalise.text(value)
    return str(value).strip()


def similarity(spec: FieldSpec, shown: Any, gold: Any, config: MatchingConfig) -> float:
    """0–1: how close ``shown`` is to the gold value of this field; 1 exactly when :func:`compare.same` holds."""
    if normalise.is_empty(shown) or normalise.is_empty(gold):
        return float(normalise.is_empty(shown) and normalise.is_empty(gold))
    if same(spec, shown, gold, config):
        return 1.0
    if spec.type == FieldType.multiselect:
        ticked, wanted = options_of(shown), options_of(gold)
        return 2 * len(ticked & wanted) / (len(ticked) + len(wanted))
    if spec.type in WRITTEN_TYPES:
        return pair_similarity(written(spec, shown, gold), written(spec, gold, gold))
    return 0.0
