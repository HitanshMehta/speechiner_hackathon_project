"""The conversation clock: seconds since the conversation started, at ``speed`` × real time."""

from __future__ import annotations

import asyncio
import time


class Clock:
    """Starts once; ``now()`` is conversation time. ``speed`` > 1 replays faster (development only)."""

    def __init__(self, speed: float = 1.0) -> None:
        if speed <= 0:
            raise ValueError("speed must be > 0")
        self.speed = speed
        self._t0: float | None = None

    @property
    def started(self) -> bool:
        return self._t0 is not None

    def start(self) -> None:
        if self._t0 is not None:
            raise RuntimeError("the clock is already running")
        self._t0 = time.monotonic()

    def now(self) -> float:
        if self._t0 is None:
            raise RuntimeError("the clock hasn't started")
        return (time.monotonic() - self._t0) * self.speed

    async def sleep_until(self, t: float) -> None:
        """Wait until conversation time ``t`` (returns at once if it has passed)."""
        delay = (t - self.now()) / self.speed
        if delay > 0:
            await asyncio.sleep(delay)
