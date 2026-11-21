"""The form at the end of a conversation, against the gold: one outcome per thing a user would have to check.

A unit is a typed field's value, one option of a multiselect field, one gold item of a free-text field, or an
"extra" (something shown that the gold doesn't support). Each unit has one outcome:

    correct          the field shows the gold value
    wrong            it shows another value
    missing          it shows nothing, or the gold item isn't in the text
    invented         it shows something where nothing belongs (an extra)
    correctly_empty  nothing belongs there and nothing is shown

Every metric about the final form (final F1, fixes, cost per form, latency, invented rate) reads these outcomes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from live_form_eval.data.common.model import Gold, GoldValue
from live_form_eval.evaluation.matching.compare import SCALAR_TYPES, options_of
from live_form_eval.evaluation.timeline.shown import Shown
from live_form_eval.interface.schema import FieldSpec, FieldType, Form

Same = Callable[[FieldSpec, Any, Any], bool]  # (field, shown value, gold value) → equal?


@dataclass(frozen=True)
class Outcome:
    conversation: str
    field_id: str
    field_type: str
    unit: str  # "value", "option", "item" or "extra"
    label: str  # the gold value, option or item; for an extra, what was shown
    outcome: str  # "correct", "wrong", "missing", "invented" or "correctly_empty"
    gold: Any = None
    shown: Any = None
    said: float | None = None  # when the gold's words reached the extractor (None when nothing belongs there)
    spoken: float | None = None  # when those words were spoken in the audio
    source: str | None = None  # where that time comes from
    settled_at: float | None = None  # since when the right thing stayed on screen until the end; None = it didn't


def spoken_of(entry: Any) -> float:
    """The audio time of a gold entry (its ``said`` once moved to extractor time keeps it in ``detail``)."""
    return entry.detail.get("spoken", entry.said)


def settled_since(shown: Shown, is_right: Callable[[Any], bool], final_time: float) -> float | None:
    """Since when the field kept showing a right value until ``final_time``; None if it doesn't show one then."""
    pieces = shown.intervals(0.0, final_time)
    if not pieces or not is_right(pieces[-1][2]):
        return None
    start = pieces[-1][0]
    for a, _, value in reversed(pieces[:-1]):
        if not is_right(value):
            break
        start = a
    return start


def typed_outcomes(conversation: str, form: Form, gold: Gold, shown: dict[str, Shown], same: Same,
                   final_time: float) -> list[Outcome]:
    """Outcomes of every typed field (one value each) and multiselect field (one per option).

    The form is read at ``final_time``, against the gold said by then. At the end of a conversation that is all
    of its gold; an earlier ``final_time`` gives the outcomes as they stood at that moment.
    """
    final_value: dict[str, GoldValue] = {}
    for value in sorted(gold.values, key=lambda v: v.said):  # stable: the later entry wins a tie
        if value.said <= final_time:
            final_value[value.field_id] = value
    out: list[Outcome] = []
    for spec in form.fields:
        timeline = shown.get(spec.field_id, Shown())
        on_screen = timeline.at(final_time)
        if spec.type in SCALAR_TYPES or spec.type == FieldType.template:
            last = final_value.get(spec.field_id)
            if last is None or last.value is None:
                result = "correctly_empty" if on_screen is None else "invented"
                out.append(Outcome(conversation, spec.field_id, spec.type.value, "value", "", result,
                                   shown=on_screen))
                continue
            wanted = last.value
            result = ("missing" if on_screen is None else "correct" if same(spec, on_screen, wanted) else "wrong")
            out.append(Outcome(
                conversation, spec.field_id, spec.type.value, "value", str(wanted), result, gold=wanted,
                shown=on_screen, said=last.said, spoken=spoken_of(last), source=last.source,
                settled_at=settled_since(
                    timeline, lambda v, s=spec, w=wanted: v is not None and same(s, v, w), final_time)))
        elif spec.type == FieldType.multiselect:
            wanted_options = {item.text: item for item in gold.items
                              if item.field_id == spec.field_id and item.said <= final_time}
            ticked = options_of(on_screen)
            for option in spec.options:
                item = wanted_options.get(option)
                if item is None:
                    result = "invented" if option in ticked else "correctly_empty"
                    out.append(Outcome(conversation, spec.field_id, spec.type.value, "option", option, result,
                                       shown=option in ticked))
                    continue
                out.append(Outcome(
                    conversation, spec.field_id, spec.type.value, "option", option,
                    "correct" if option in ticked else "missing", gold=True, shown=option in ticked,
                    said=item.said, spoken=spoken_of(item), source=item.source,
                    settled_at=settled_since(timeline, lambda v, o=option: o in options_of(v), final_time)))
    return out
