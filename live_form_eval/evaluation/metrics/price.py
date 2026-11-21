"""Price: what the extractor's uses cost, in euros at the vendor's price.

Added up from every use a run reported (``session.report_usage``), its extraction attempts and its other uses::

    € per form          = total € ÷ conversations
    € per audio minute  = total € ÷ minutes of conversation audio

The euros are whatever the extractor reported: a submission that doesn't report its uses shows a price of 0
and can't be ranked fairly (the contract check warns about it). metrics/calls.py splits the price by unit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Price:
    eur: float
    calls: int  # extraction attempts
    other_uses: int
    conversations: int
    audio_minutes: float

    @property
    def eur_per_form(self) -> float | None:
        return self.eur / self.conversations if self.conversations else None

    @property
    def eur_per_audio_minute(self) -> float | None:
        return self.eur / self.audio_minutes if self.audio_minutes else None


def price(uses: Sequence[dict[str, Any]], conversations: int, audio_seconds: float) -> Price:
    """``uses`` are a run's lines of calls.jsonl."""
    attempts = sum(1 for use in uses if use.get("extraction", True) is not False)
    return Price(eur=sum(use.get("eur") or 0.0 for use in uses), calls=attempts, other_uses=len(uses) - attempts,
                 conversations=conversations, audio_minutes=audio_seconds / 60)
