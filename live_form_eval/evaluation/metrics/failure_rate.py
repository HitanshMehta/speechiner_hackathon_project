"""Failure rate: the share of conversations whose run did not end normally.

A conversation fails when the extractor raised an error, or had not returned ``final_grace_s`` after the last
partial. Whatever it published before failing is still scored; this rate says how often that happened.
A check to report next to the ranking, not part of it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def failure_rate(statuses: Sequence[dict[str, Any]]) -> float | None:
    """``statuses`` are the per-conversation records of a run.json."""
    return sum(status["status"] != "ok" for status in statuses) / len(statuses) if statuses else None
