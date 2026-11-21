"""One field of one conversation: its gold values against what the form displayed.

Two lists go in. ``gold`` is what the field should show over time: one value per stream message that changes it,
"said" at the moment that message reached the extractor (timeline/steps.py). ``shown`` is every visible change of
the field (timeline/shown.py); a clear is a display whose value is empty.

0. A gold value that the form already shows at the moment it is said counts as displayed at that moment::

       for g in gold:
           d = the display on screen at g.said
           if d equals g:   insert a display (g.said, d.value)

   From then on it is a display like any other, and steps 1 and 2 are the same with or without it. This is what
   rewards a good guess, a name completed from its first words, or a label that is a little late: the display
   made before the value was said keeps its own grade for the seconds until then (against what had been said),
   and the value counts from the moment it is said, with no delay.

1. The walk goes through the displays once, in time order, and follows where the form is in the gold::

       at = None                                  # the gold value the form has reached
       for d in displays:
           ahead = gold values after `at`, already said, that d equals
           if ahead:                              # LANDING: the form moves on to the most recent of them
               at = the most recent of ahead;  d.grade = 1
           elif d equals at:                      # REPAIR: back on the gold value the form had reached
               d.grade = 1
           else:                                  # anything else: how close to what it should have been
               d.grade = the highest similarity between d and a gold value from `at` on, already said
           d.seconds = time until the next display, or until the end

2. The match gives every gold value one display. A gold value with a landing takes that display and scores 1.
   The others are taken run by run (a run = neighbouring gold values without a landing)::

       window = the displays from the landing before the run (else the first display)
                to the landing after it (else the last display), both included
       for g in run, from the last to the first:
           allowed = displays of the window shown after g was said, and not after the display of the gold value
                     that follows g in the run
           g takes the allowed display most similar to it (the earliest if equal; none if the best is 0)
           g.score = that similarity

   Going from the last to the first keeps the order: the display of an earlier gold value is never after the
   display of a later one.

The four terms read the result: completeness the scores, speed the delays, precision the grades and seconds
(metrics/completeness.py, speed.py, precision.py); outcome only needs the final form (metrics/outcome.py).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

Same = Callable[[Any, Any], bool]  # (shown value, gold value) → the same value?
Similarity = Callable[[Any, Any], float]  # (shown value, gold value) → 0–1


@dataclass
class Display:
    shown: float  # when the field changed to this value
    value: Any  # None = the field was cleared
    seconds: float = 0.0  # how long it stayed on screen
    grade: float = 0.0  # 1 for a landing or a repair, else its similarity to what it should have been
    kind: str = "other"  # "landing", "repair" or "other"
    step: int | None = None  # the gold value it lands on or repairs
    kept: bool = False  # not a change of the field: the value was already there when it was said (step 0)


@dataclass
class Step:
    said: float  # when the message that brought this value reached the extractor
    value: Any  # None = the value was withdrawn
    landed: bool = False  # a display equal to it arrived after it was said, while the form was not past it
    display: int | None = None  # the display it takes (its landing, else the closest one); None = none
    score: float = 0.0  # 1 when landed, else the similarity of that display
    delay: float | None = None  # from said to that display; None without a display


@dataclass
class FieldWalk:
    steps: list[Step] = field(default_factory=list)
    displays: list[Display] = field(default_factory=list)


def walk_field(gold: Sequence[tuple[float, Any]], shown: Sequence[tuple[float, Any]], end: float, same: Same,
               similarity: Similarity) -> FieldWalk:
    """Walk and match one field. ``gold`` and ``shown`` are (time, value) pairs in time order; displays made at
    ``end`` or later are not seen."""
    steps = [Step(said=said, value=value) for said, value in gold]
    displays = [Display(shown=t, value=value) for t, value in shown if t < end]

    kept = []
    for step in steps:  # step 0: the value is already on screen when it is said
        on_screen = next((d for d in reversed(displays) if d.shown <= step.said), None)
        if on_screen and on_screen.shown < step.said < end and same(on_screen.value, step.value):
            kept.append(Display(shown=step.said, value=on_screen.value, kept=True))
    displays = sorted(displays + kept, key=lambda d: d.shown)

    at = -1
    for k, d in enumerate(displays):
        ahead = [j for j in range(at + 1, len(steps)) if steps[j].said <= d.shown and same(d.value, steps[j].value)]
        if ahead:
            at = ahead[-1]
            steps[at].landed, steps[at].display, steps[at].score = True, k, 1.0
            d.kind, d.step, d.grade = "landing", at, 1.0
        elif at >= 0 and same(d.value, steps[at].value):
            d.kind, d.step, d.grade = "repair", at, 1.0
        else:
            owed = [step for step in steps[max(at, 0):] if step.said <= d.shown]
            d.grade = max((similarity(d.value, step.value) for step in owed), default=0.0)
        d.seconds = (displays[k + 1].shown if k + 1 < len(displays) else end) - d.shown

    j = 0
    while j < len(steps):
        if steps[j].landed:
            j += 1
            continue
        stop = j
        while stop < len(steps) and not steps[stop].landed:
            stop += 1
        first = steps[j - 1].display if j else 0
        last = steps[stop].display if stop < len(steps) else len(displays) - 1
        limit = last
        for step in reversed(steps[j:stop]):
            best, best_score = None, 0.0
            for k in range(first, limit + 1):
                if displays[k].shown >= step.said:
                    score = similarity(displays[k].value, step.value)
                    if score > best_score:
                        best, best_score = k, score
            if best is not None:
                step.display, step.score, limit = best, best_score, best
        j = stop

    for step in steps:
        if step.display is not None:
            step.delay = displays[step.display].shown - step.said
    return FieldWalk(steps=steps, displays=displays)
