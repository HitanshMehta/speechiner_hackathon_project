"""Final F1: how right is the form at the end?

Counted over the outcomes of the final form (timeline/final.py)::

    precision = correct ÷ everything filled               (correct + wrong + invented)
    recall    = correct ÷ everything that should be filled (correct + wrong + missing)

Correctly empty fields are left out on purpose: most fields of a form stay empty, and counting them would hide
the errors. ``fixes`` is what a user would have to correct by hand: missing + wrong + invented.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from live_form_eval.evaluation.timeline.final import Outcome


@dataclass(frozen=True)
class FinalF1:
    correct: int
    wrong: int
    missing: int
    invented: int
    correctly_empty: int

    @property
    def precision(self) -> float | None:
        filled = self.correct + self.wrong + self.invented
        return self.correct / filled if filled else None

    @property
    def recall(self) -> float | None:
        expected = self.correct + self.wrong + self.missing
        return self.correct / expected if expected else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def fixes(self) -> int:
        return self.missing + self.wrong + self.invented


def final_f1(outcomes: Iterable[Outcome]) -> FinalF1:
    counts = Counter(outcome.outcome for outcome in outcomes)
    return FinalF1(correct=counts["correct"], wrong=counts["wrong"], missing=counts["missing"],
                   invented=counts["invented"], correctly_empty=counts["correctly_empty"])
