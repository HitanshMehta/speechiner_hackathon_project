"""Invented-value rate: of everything that should be empty at the end, how much shows a value?

Counted over the final form's units where no gold belongs (typed fields, multiselect options, and free-text
fields that should stay empty)::

    invented rate = invented ÷ (invented + correctly empty)

A check to report next to the ranking, not part of it: it shows whether an extractor fills fields nobody
talked about.
"""

from __future__ import annotations

from collections.abc import Iterable

from live_form_eval.evaluation.timeline.final import Outcome


def invented_rate(outcomes: Iterable[Outcome]) -> float | None:
    outcomes = [outcome for outcome in outcomes if outcome.unit != "extra"
                and outcome.outcome in ("invented", "correctly_empty")]
    return sum(outcome.outcome == "invented" for outcome in outcomes) / len(outcomes) if outcomes else None
