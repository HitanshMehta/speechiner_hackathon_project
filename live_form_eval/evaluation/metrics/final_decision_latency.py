"""Final Decision latency: how long after a value was said did the form show it for good?

For every unit of the final form that has a gold (a value, a multiselect option, a free-text item)::

    latency = the moment from which the right thing stayed on screen until the end − the moment it was said

- A unit that isn't right at the end never settled: its latency is infinite.
- A unit shown before it was said has latency 0 (and is counted as ``early``). This is also what a gold said
  "at the end of the conversation" (no supporting words) gets when it is there by then.

Reported: the median, the config's percentile, and the share that never settled. Infinite latencies take part
in the percentile, so a run that leaves more than (100 − percentile)% of its values unsettled has an infinite one.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from live_form_eval.evaluation.timeline.final import Outcome


def unit_latency(outcome: Outcome) -> float | None:
    """One unit's latency in seconds; None when no gold belongs there (nothing to wait for)."""
    if outcome.said is None:
        return None
    if outcome.outcome != "correct" or outcome.settled_at is None:
        return math.inf
    return max(0.0, outcome.settled_at - outcome.said)


def percentile(values: list[float], p: float) -> float | None:
    """The smallest value that at least p% of the values don't exceed (nearest rank); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


@dataclass(frozen=True)
class Latency:
    n: int  # units with a gold
    median_s: float | None
    percentile_s: float | None  # at the config's percentile
    never_settled: float | None  # share of units that weren't right at the end
    early: int  # units shown before they were said (latency 0)


def final_decision_latency(outcomes: Iterable[Outcome], p: float) -> Latency:
    outcomes = [outcome for outcome in outcomes if outcome.said is not None]
    values = [unit_latency(outcome) for outcome in outcomes]
    early = sum(1 for outcome in outcomes if outcome.outcome == "correct" and outcome.settled_at is not None
                and outcome.settled_at < outcome.said)
    return Latency(n=len(values), median_s=percentile(values, 50), percentile_s=percentile(values, p),
                   never_settled=sum(math.isinf(v) for v in values) / len(values) if values else None, early=early)
