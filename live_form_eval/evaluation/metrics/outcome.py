"""Outcome: is the form right at the end? The final F1 of one form.

Counted over the outcomes of the final form (timeline/final.py), like final F1 (metrics/final_f1.py)::

    outcome of a form = 2 × correct ÷ (2 × correct + wrong + missing + invented)

which is the harmonic mean of "correct ÷ everything filled" and "correct ÷ everything that should be filled".
A form that fills nothing right scores 0; a form with nothing to fill and nothing filled has no outcome.
"""

from __future__ import annotations

from collections.abc import Iterable

from live_form_eval.evaluation.metrics.final_f1 import final_f1
from live_form_eval.evaluation.timeline.final import Outcome


def outcome(outcomes: Iterable[Outcome]) -> float | None:
    final = final_f1(outcomes)
    counted = 2 * final.correct + final.fixes
    return 2 * final.correct / counted if counted else None
