"""What a field showed over time, rebuilt from a run's predictions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from live_form_eval.evaluation.matching.normalise import is_empty


@dataclass(frozen=True)
class Shown:
    """One field's values over a conversation: (time, value) pairs in time order.

    An empty value ("" or []) is stored as None, and an update that repeats the value already shown is dropped,
    so every pair is a visible change of the field.
    """

    updates: tuple[tuple[float, Any], ...] = ()

    @classmethod
    def from_updates(cls, updates: Iterable[tuple[float, Any]]) -> Shown:
        kept: list[tuple[float, Any]] = []
        current: Any = None
        for t, value in sorted(updates, key=lambda update: update[0]):
            value = None if is_empty(value) else value
            if value != current:
                kept.append((t, value))
                current = value
        return cls(tuple(kept))

    def at(self, t: float) -> Any:
        """The value on screen at time ``t`` (None before the first update)."""
        value = None
        for time, update in self.updates:
            if time > t:
                break
            value = update
        return value

    def intervals(self, start: float, end: float) -> list[tuple[float, float, Any]]:
        """The field over [start, end) as pieces of constant value (empty pieces included)."""
        out: list[tuple[float, float, Any]] = []
        times = [start] + [t for t, _ in self.updates if start < t < end] + [end]
        for a, b in pairwise(times):
            if b > a:
                out.append((a, b, self.at(a)))
        return out

    def changes(self, until: float) -> int:
        """How many times the field changed up to ``until``."""
        return sum(1 for t, _ in self.updates if t <= until)

    def published(self, t: float) -> float | None:
        """When the value on screen at ``t`` was published (None if the field is empty then)."""
        moment = None
        for time, _ in self.updates:
            if time > t:
                break
            moment = time
        return moment if self.at(t) is not None else None


def shown_fields(predictions: Iterable[dict[str, Any]], conversation: str) -> dict[str, Shown]:
    """Every field's timeline for one conversation of a run."""
    updates: dict[str, list[tuple[float, Any]]] = {}
    for line in predictions:
        if line["conversation"] == conversation:
            updates.setdefault(line["field_id"], []).append((line["t"], line["value"]))
    return {field_id: Shown.from_updates(items) for field_id, items in updates.items()}
