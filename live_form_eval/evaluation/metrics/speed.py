"""Speed: how soon after a value was said did the form show it?

For every gold value, the delay runs from the moment its words reached the extractor to the display the walk
gave it (its landing, else its closest display; timeline/walk.py). The delay becomes a number between 0 and 1::

    speed of a gold value = (1 − delay ÷ limit) ^ power      0 from the limit on, and without a display

With power 0.5 this is a square-root curve read from right to left: it stays close to 1 for short delays and
falls steeply near the limit (limit 8 s: 0.5 s → 0.97, 2 s → 0.87, 5 s → 0.61, 7 s → 0.35).

    speed of a field = mean over its gold values
    speed of a form  = mean over its fields that have gold values
"""

from __future__ import annotations

from collections.abc import Iterable

from live_form_eval.evaluation.timeline.walk import FieldWalk


def curve(delay: float | None, limit_s: float, power: float) -> float:
    if delay is None or delay >= limit_s:
        return 0.0
    return (1 - max(delay, 0.0) / limit_s) ** power


def field_speed(walk: FieldWalk, limit_s: float, power: float) -> float | None:
    """None for a field nobody talked about."""
    if not walk.steps:
        return None
    return sum(curve(step.delay, limit_s, power) for step in walk.steps) / len(walk.steps)


def speed(walks: Iterable[FieldWalk], limit_s: float, power: float) -> float | None:
    values = [value for value in (field_speed(walk, limit_s, power) for walk in walks) if value is not None]
    return sum(values) / len(values) if values else None
