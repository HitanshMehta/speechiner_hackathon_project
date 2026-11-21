"""Joint goal accuracy (JGA): SpokenWOZ's own measure, for a rough comparison with published trackers.

After each caller turn, the dataset labels every slot. A turn counts when the whole form is right: every field
shows its labelled value and nothing else is filled. The form is read LAG seconds after the caller's turn ends.

Published systems read finished turns without any time limit and most are trained on SpokenWOZ, so this number
is only an orientation (the paper's best tracker: 25.65%; ChatGPT: 13.75%).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from live_form_eval.data.common.model import GoldSnapshot
from live_form_eval.evaluation.timeline.shown import Shown
from live_form_eval.interface.schema import FieldSpec

Same = Callable[[FieldSpec, Any, Any], bool]


def turn_is_right(snapshot: GoldSnapshot, shown: dict[str, Shown], specs: Sequence[FieldSpec], same: Same,
                  lag_s: float) -> bool:
    """Every field in ``specs`` shows its labelled value (or nothing, when it has none) lag_s after the turn."""
    moment = snapshot.t + lag_s
    return all(same(spec, shown[spec.field_id].at(moment) if spec.field_id in shown else None,
                    snapshot.values.get(spec.field_id)) for spec in specs)


def joint_goal_accuracy(snapshots: Sequence[GoldSnapshot], shown: dict[str, Shown], specs: Sequence[FieldSpec],
                        same: Same, lag_s: float) -> tuple[int, int]:
    """(turns where the whole form is right, turns)."""
    return sum(turn_is_right(snapshot, shown, specs, same, lag_s) for snapshot in snapshots), len(snapshots)
