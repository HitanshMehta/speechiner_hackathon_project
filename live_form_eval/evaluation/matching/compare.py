"""Is a shown value the same as the gold value? One rule per field type.

Free-text fields (textarea) aren't compared here: a judge decides whether a text supports a statement
(matching/judge). Multiselect fields are compared option by option (see :func:`options_of`).
"""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz

from live_form_eval.config import MatchingConfig
from live_form_eval.evaluation.matching import normalise
from live_form_eval.interface.schema import FieldSpec, FieldType

# Types whose value is one thing that is right or wrong at any moment: these go into live F1 and JGA.
SCALAR_TYPES = frozenset({FieldType.text, FieldType.email, FieldType.phone, FieldType.url, FieldType.integer,
                          FieldType.float, FieldType.slider, FieldType.checkbox, FieldType.select, FieldType.date,
                          FieldType.time, FieldType.datetime})


def same(spec: FieldSpec, shown: Any, gold: Any, config: MatchingConfig) -> bool:
    """True when ``shown`` counts as the gold value of this field. Two empty values are the same."""
    if normalise.is_empty(shown) or normalise.is_empty(gold):
        return normalise.is_empty(shown) and normalise.is_empty(gold)
    kind = spec.type
    if kind == FieldType.text:
        return _same_text(shown, gold, config)
    if kind == FieldType.textarea:
        return normalise.text(shown) == normalise.text(gold)
    if kind == FieldType.email:
        return normalise.email(shown) == normalise.email(gold)
    if kind == FieldType.phone:
        return normalise.digits(shown) == normalise.digits(gold)
    if kind == FieldType.url:
        return normalise.url(shown) == normalise.url(gold)
    if kind in (FieldType.integer, FieldType.float, FieldType.slider):
        return _same_number(shown, gold)
    if kind == FieldType.checkbox:
        return bool(shown) == bool(gold)
    if kind == FieldType.select:
        return str(shown) == str(gold)
    if kind == FieldType.multiselect:
        return options_of(shown) == options_of(gold)
    if kind == FieldType.time:
        return normalise.time(shown) == normalise.time(gold)
    if kind in (FieldType.date, FieldType.datetime):
        return str(shown).strip() == str(gold).strip()
    if kind == FieldType.template:
        return _same_template(shown, gold)
    raise ValueError(f"no comparison for field type {kind}")


def options_of(value: Any) -> frozenset[str]:
    """The options a multiselect value holds (an empty set for an empty field)."""
    if normalise.is_empty(value):
        return frozenset()
    return frozenset(str(option) for option in value) if isinstance(value, (list, tuple, set, frozenset)) \
        else frozenset({str(value)})


def _same_text(shown: Any, gold: Any, config: MatchingConfig) -> bool:
    """Identifiers (the gold has a digit: an ID, a plate) match exactly once squashed; names match fuzzily.

    Names follow the dataset's spelling while the transcript may not ("live house" for "limehouse"), so they
    match from ``fuzzy_name_threshold``, on the better of a word-order-free and a character comparison. Both
    compare the whole name: a part of it ("john" for "john henley", "guest house" for "acorn guest house") is
    not the name.
    """
    a, b = normalise.text(shown), normalise.text(gold)
    if a == b or normalise.squash(a) == normalise.squash(b):
        return True
    if any(character.isdigit() for character in b):
        return False
    score = max(fuzz.token_sort_ratio(a, b), fuzz.ratio(normalise.squash(a), normalise.squash(b)))
    return score >= config.fuzzy_name_threshold


def _same_number(shown: Any, gold: Any) -> bool:
    try:
        return abs(float(shown) - float(gold)) < 1e-9
    except (TypeError, ValueError):
        return False


def _same_template(shown: Any, gold: Any) -> bool:
    if not isinstance(shown, dict) or not isinstance(gold, dict):
        return False
    keys = {key for key, value in gold.items() if not normalise.is_empty(value)}
    return keys == {key for key, value in shown.items() if not normalise.is_empty(value)} and all(
        normalise.text(shown[key]) == normalise.text(gold[key]) for key in keys)
