"""What each field should show over time: its gold values as the walk reads them (timeline/walk.py).

The processed gold has one entry per transcript word that changes a value: a digit of a dictated number, a word of
a name, a whole value when it is one piece of information. An extractor only ever sees whole stream messages, so
once the gold's times are moved to the extractor's clock (timeline/arrival.py), the entries of one field that
arrive with the same message are one step: the last of them. In between two steps there is nothing an extractor
could have shown.

    words         "0 1 5"  "6 2 3"                          spoken one after the other
    messages      (0 1) (5 6 2) (3)                         as the stream cut them
    gold steps    "01"   "01562"   "015623"                 one per message that changes the field

A multiselect field's steps are the options named so far. A free-text field's steps are the facts that hold: a
fact from the message that completes it, and an earlier version of a point until the message that changes it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from live_form_eval.data.common.model import Gold, GoldFact
from live_form_eval.evaluation.matching.compare import SCALAR_TYPES
from live_form_eval.interface.schema import FieldType, Form

WALKED_TYPES = SCALAR_TYPES | {FieldType.multiselect, FieldType.textarea}
StepList = list[tuple[float, Any]]  # (said, value) in time order; value None = withdrawn


def _spoken(entry: Any) -> float:
    return entry.detail.get("spoken", entry.said)


def _one_per_message(entries: list[tuple[float, float, Any]]) -> StepList:
    """(said, spoken, value) → steps: the last entry of each message, without steps that change nothing."""
    steps: StepList = []
    for said, _, value in sorted(entries, key=lambda entry: (entry[0], entry[1])):
        if steps and steps[-1][0] == said:
            steps.pop()
        if not steps or steps[-1][1] != value:
            steps.append((said, value))
    return steps


def _fact_steps(facts: Sequence[GoldFact]) -> StepList:
    """A free-text field's steps: the facts that hold after each message that starts or ends one.

    The value of a step is the tuple of those facts' texts, in the order they were said; when none holds any
    more, it is None (withdrawn).
    """
    facts = sorted(facts, key=lambda fact: fact.said)
    moments = sorted({fact.said for fact in facts} | {fact.until for fact in facts if fact.until is not None})
    steps: StepList = []
    for moment in moments:
        holding = tuple(fact.text for fact in facts
                        if fact.said <= moment and (fact.until is None or moment < fact.until))
        if (holding or None) != (steps[-1][1] if steps else None):
            steps.append((moment, holding or None))
    return steps


def gold_steps(form: Form, gold: Gold) -> dict[str, StepList]:
    """Field → its steps, for the fields that have gold. ``gold`` is in extractor time."""
    steps: dict[str, StepList] = {}
    for spec in form.fields:
        if spec.type in SCALAR_TYPES:
            field_steps = _one_per_message([(value.said, _spoken(value), value.value) for value in gold.values
                                            if value.field_id == spec.field_id])
        elif spec.type == FieldType.multiselect:
            items = sorted((item for item in gold.items if item.field_id == spec.field_id),
                           key=lambda item: (item.said, _spoken(item)))
            field_steps = _one_per_message([(item.said, _spoken(item), tuple(other.text for other in items[: k + 1]))
                                            for k, item in enumerate(items)])
        elif spec.type == FieldType.textarea:
            field_steps = _fact_steps([fact for fact in gold.facts if fact.field_id == spec.field_id])
        else:
            continue
        if field_steps:
            steps[spec.field_id] = field_steps
    return steps


def scoring_end(steps: dict[str, StepList], input_end: float, grace_s: float) -> float:
    """Until when a conversation is scored: the end of the input, or ``grace_s`` after the last value was said.

    ``grace_s`` is the speed limit: later than that, the last value could no longer earn any speed.
    """
    last_said = max((said for field_steps in steps.values() for said, _ in field_steps), default=0.0)
    return max(input_end, last_said + grace_s)
