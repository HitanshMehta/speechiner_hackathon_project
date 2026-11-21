"""Precision: of the time a field shows something the form put there, how much of it is right?

Every display has a grade from the walk (timeline/walk.py): 1 when it shows a gold value the form has reached
(a landing, or coming back to it), else its similarity to the closest gold value already said. A display counts
for the seconds it stays on screen, so what a user sees for long weighs more than a flash::

    precision of a field = Σ grade × seconds ÷ Σ seconds        over its displays (a clear is a display)
    precision of a form  = mean over its fields that displayed something

A value shown late keeps grade 1: lateness is charged once, in speed. A field nobody talked about has no gold
value, so whatever it shows has grade 0.
"""

from __future__ import annotations

from collections.abc import Iterable

from live_form_eval.evaluation.timeline.walk import FieldWalk


def field_precision(walk: FieldWalk) -> float | None:
    """None for a field that displayed nothing."""
    seconds = sum(display.seconds for display in walk.displays)
    if seconds <= 0:
        return None
    return sum(display.grade * display.seconds for display in walk.displays) / seconds


def precision(walks: Iterable[FieldWalk]) -> float | None:
    values = [value for value in map(field_precision, walks) if value is not None]
    return sum(values) / len(values) if values else None
