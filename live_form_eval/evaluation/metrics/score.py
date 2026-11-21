"""The score: completeness, precision, speed and outcome in one number, the one the ranking sorts by.

    score = harmonic mean of the four terms = 4 ÷ (1/completeness + 1/precision + 1/speed + 1/outcome)

Each term is first averaged over the forms of a run (a form without the term, e.g. one where nothing was said,
is left out of that term's average). The harmonic mean stays close to the weakest term, so a submission cannot
buy a high score with one strong term: a form that is fast but wrong, or right but slow, scores low.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

TERMS = ("completeness", "precision", "speed", "outcome")


@dataclass(frozen=True)
class Terms:
    """The four terms of one form, or of one run (the means over its forms). None = nothing to measure."""

    completeness: float | None
    precision: float | None
    speed: float | None
    outcome: float | None

    @property
    def score(self) -> float | None:
        return harmonic_mean([self.completeness, self.precision, self.speed, self.outcome])


def harmonic_mean(values: Sequence[float | None]) -> float | None:
    """0 as soon as one value is 0; None when a value is missing and none is 0."""
    if any(value == 0 for value in values):
        return 0.0
    if any(value is None for value in values):
        return None
    return len(values) / sum(1 / value for value in values)


def mean_terms(forms: Iterable[Terms]) -> Terms:
    """Each term averaged over the forms that have it."""
    forms = list(forms)

    def mean(name: str) -> float | None:
        values = [getattr(form, name) for form in forms if getattr(form, name) is not None]
        return sum(values) / len(values) if values else None

    return Terms(*(mean(name) for name in TERMS))
