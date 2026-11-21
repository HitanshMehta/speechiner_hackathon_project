"""Turn slot F1: SpokenWOZ's per-turn measure next to joint goal accuracy, for a rough comparison.

After each caller turn (read LAG seconds later), every labelled slot is a hit if the form shows its value,
a miss if not; every filled field that the labels don't have, or have differently, is a false alarm.
The counts are added over all turns and give one precision, recall and F1.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from live_form_eval.data.common.model import GoldSnapshot
from live_form_eval.evaluation.matching.normalise import is_empty
from live_form_eval.evaluation.timeline.shown import Shown
from live_form_eval.interface.schema import FieldSpec

Same = Callable[[FieldSpec, Any, Any], bool]


@dataclass
class SlotCounts:
    hits: int = 0  # labelled slots shown with the right value
    misses: int = 0  # labelled slots not shown, or shown with another value
    false_alarms: int = 0  # filled fields without that label

    def add(self, other: SlotCounts) -> None:
        self.hits += other.hits
        self.misses += other.misses
        self.false_alarms += other.false_alarms

    @property
    def f1(self) -> float | None:
        if self.hits + self.misses + self.false_alarms == 0:
            return None
        return 2 * self.hits / (2 * self.hits + self.misses + self.false_alarms)


def turn_slot_counts(snapshots: Sequence[GoldSnapshot], shown: dict[str, Shown], specs: Sequence[FieldSpec],
                     same: Same, lag_s: float) -> SlotCounts:
    counts = SlotCounts()
    for snapshot in snapshots:
        moment = snapshot.t + lag_s
        for spec in specs:
            value = shown[spec.field_id].at(moment) if spec.field_id in shown else None
            label = snapshot.values.get(spec.field_id)
            if is_empty(label):
                counts.false_alarms += not is_empty(value)
            elif same(spec, value, label):
                counts.hits += 1
            else:
                counts.misses += 1
                counts.false_alarms += not is_empty(value)
    return counts
