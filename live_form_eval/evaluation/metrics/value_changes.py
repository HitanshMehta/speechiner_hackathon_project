"""Value changes: how often a field's value changed on screen while people were talking.

Every visible change of a field is one of three kinds::

    filled    the field was empty before
    revised   the field held another value before
    cleared   the field is empty after

A form that finds each value once shows one "filled" per field and nothing else. A revision is a value that
grows while it is dictated, a note that gets longer, or an extractor that changes its mind::

    revisions per filled field  = revised ÷ fields that were ever filled
    filled once                 = fields filled once and never changed again ÷ fields that were ever filled

Reported next to the score, not ranked: the score already counts what stood on screen (precision) and when it
got there (speed). These numbers say how restless the form was to look at.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from live_form_eval.evaluation.timeline.shown import Shown

KINDS = ("filled", "revised", "cleared")


@dataclass(frozen=True)
class Change:
    """One visible change of a field."""

    t: float
    kind: str  # one of KINDS
    before: Any  # None = the field was empty
    after: Any  # None = the field is empty


@dataclass(frozen=True)
class ValueChanges:
    fields: int  # fields that were ever filled
    filled: int
    revised: int
    cleared: int
    once: int  # fields filled once and never changed again

    @property
    def changes(self) -> int:
        return self.filled + self.revised + self.cleared

    @property
    def revisions_per_field(self) -> float | None:
        return self.revised / self.fields if self.fields else None

    @property
    def once_share(self) -> float | None:
        return self.once / self.fields if self.fields else None


def field_changes(shown: Shown) -> list[Change]:
    """The changes of one field, in time order."""
    changes: list[Change] = []
    before: Any = None
    for t, value in shown.updates:
        kind = "filled" if before is None else "cleared" if value is None else "revised"
        changes.append(Change(t, kind, before, value))
        before = value
    return changes


def value_changes(fields: Iterable[Sequence[Change]]) -> ValueChanges:
    """Adds up the changes of several fields (of a form, a field type or a run)."""
    touched = [changes for changes in fields if changes]
    count = {kind: sum(change.kind == kind for changes in touched for change in changes) for kind in KINDS}
    return ValueChanges(fields=len(touched), filled=count["filled"], revised=count["revised"],
                        cleared=count["cleared"], once=sum(len(changes) == 1 for changes in touched))
