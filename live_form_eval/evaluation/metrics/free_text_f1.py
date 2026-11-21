"""Free-text F1: are the notes and answers right and complete at the end?

From the judge's verdicts on each free-text field (timeline/free_text.py), added over the fields::

    correct  (precision) = shown items or sentences that the gold supports ÷ shown items or sentences
    complete (recall)    = gold items or sentences that the shown text supports ÷ gold items or sentences
    F1                   = their harmonic mean

Fields whose gold is empty (unanswerable questions, empty lists) are left out here; metrics/unanswerable.py
covers them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from live_form_eval.evaluation.timeline.free_text import FieldSupport


@dataclass(frozen=True)
class FreeTextF1:
    shown_units: int
    shown_supported: int
    gold_units: int
    gold_supported: int

    @property
    def correct(self) -> float | None:
        return self.shown_supported / self.shown_units if self.shown_units else None

    @property
    def complete(self) -> float | None:
        return self.gold_supported / self.gold_units if self.gold_units else None

    @property
    def f1(self) -> float | None:
        if self.gold_units == 0:
            return None
        p, r = self.correct or 0.0, self.complete or 0.0  # nothing shown: nothing right
        return 2 * p * r / (p + r) if p + r else 0.0


def free_text_f1(fields: Iterable[FieldSupport]) -> FreeTextF1:
    fields = [support for support in fields if not support.should_be_empty]
    return FreeTextF1(shown_units=sum(s.shown_units for s in fields),
                      shown_supported=sum(s.shown_supported for s in fields),
                      gold_units=sum(s.gold_units for s in fields),
                      gold_supported=sum(s.gold_supported for s in fields))
