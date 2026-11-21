"""Completeness: of what was said, how much did the form show?

Every gold value of a field gets a score from the walk (timeline/walk.py):

- 1 when the form showed it (a landing);
- else the similarity of the closest display made after it was said: a phone number shown in two pieces still
  earns most of the score of the digits in between;
- 0 when nothing similar was displayed after it was said.

    completeness of a field = mean score of its gold values
    completeness of a form  = mean over its fields that have gold values
"""

from __future__ import annotations

from collections.abc import Iterable

from live_form_eval.evaluation.timeline.walk import FieldWalk


def field_completeness(walk: FieldWalk) -> float | None:
    """None for a field nobody talked about."""
    if not walk.steps:
        return None
    return sum(step.score for step in walk.steps) / len(walk.steps)


def completeness(walks: Iterable[FieldWalk]) -> float | None:
    values = [value for value in map(field_completeness, walks) if value is not None]
    return sum(values) / len(values) if values else None
