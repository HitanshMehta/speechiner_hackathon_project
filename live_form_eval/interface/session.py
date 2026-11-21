"""One conversation, as an extractor sees it: the form, the live transcript, and where values are published."""

from __future__ import annotations

import math
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol

from live_form_eval.interface.schema import Form, value_error
from live_form_eval.interface.transcript import Partial

# The units of a model call that the report knows by name. ``input_tokens`` is the input no cache held: the input
# a vendor read from its cache, or wrote to it, is counted in its own unit, never twice. Any other name is a unit
# too (``audio_seconds``, ``requests``, …).
TOKEN_UNITS = ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens")
_UNIT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_RESERVED = frozenset({"conversation", "t"})  # the run folder stamps every use with these


class InvalidValue(ValueError):
    """A published value doesn't fit its field (wrong type, not an option, outside the range, unknown field)."""


class SessionClosed(RuntimeError):
    """A value was published before the conversation started or after the session was closed."""


class Clock(Protocol):
    started: bool

    def now(self) -> float: ...

    async def sleep_until(self, t: float) -> None: ...


class Recorder(Protocol):
    def prediction(self, conversation: str, t: float, field_id: str, value: Any) -> None: ...

    def call(self, conversation: str, t: float, record: dict[str, Any]) -> None: ...


def _number(what: str, value: Any) -> float:
    """A reported amount or price: a finite number, 0 or more."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{what} must be a number ≥ 0, got {value!r}")
    return value


class Session:
    """Created by the harness for one conversation. Every method stamps its call with the conversation clock."""

    def __init__(self, conversation_id: str, form: Form, stream: Sequence[Partial], clock: Clock,
                 recorder: Recorder) -> None:
        self.conversation_id = conversation_id
        self.form = form
        self._stream = tuple(stream)
        self._clock = clock
        self._recorder = recorder
        self._fields = {spec.field_id: spec for spec in form.fields}
        self._iterated = False
        self.closed = False
        self.late_updates = 0  # values published after the session was closed (ignored)
        self.calls = 0  # uses reported as extraction attempts
        self.other_uses = 0  # … and as anything else

    def now(self) -> float:
        """Seconds since the conversation started (0 before it starts)."""
        return self._clock.now() if self._clock.started else 0.0

    async def partials(self) -> AsyncIterator[Partial]:
        """The live transcript: each partial or finished turn at its arrival time; ends after the last one.

        Call it once per conversation (a second call raises). Iterating doesn't wait for your processing: if you
        are slow, the next partials are already due and arrive at once, as from a real vendor's socket.
        """
        if self._iterated:
            raise RuntimeError("partials() can be iterated only once per session")
        self._iterated = True
        for partial in self._stream:
            await self._clock.sleep_until(partial.t)
            yield partial

    @property
    def stream_end(self) -> float:
        """Arrival time of the last partial."""
        return self._stream[-1].t if self._stream else 0.0

    async def sleep(self, seconds: float) -> None:
        """Wait ``seconds`` of conversation time. Use it instead of ``asyncio.sleep`` for pauses and timers, so
        that a development run at ``--speed 5`` behaves like a real-time run."""
        await self._clock.sleep_until(self.now() + seconds)

    async def sleep_until(self, t: float) -> None:
        """Wait until conversation time ``t`` (returns at once if it has passed)."""
        await self._clock.sleep_until(t)

    def update(self, field_id: str, value: Any) -> None:
        """Show ``value`` in one field from now on. ``None`` clears the field."""
        self._check_open()
        self._check(field_id, value)
        self._recorder.prediction(self.conversation_id, self.now(), field_id, value)

    def update_snapshot(self, values: dict[str, Any]) -> None:
        """Show several values at once, all at the same moment. Fields not in ``values`` keep their value."""
        self._check_open()
        for field_id, value in values.items():
            self._check(field_id, value)
        t = self.now()
        for field_id, value in values.items():
            self._recorder.prediction(self.conversation_id, t, field_id, value)

    def report_usage(self, *, model: str, extraction: bool, usage: Mapping[str, float] | None = None,
                     prices: Mapping[str, float] | None = None, latency_s: float | None = None, ok: bool = True,
                     error: str | None = None, **meta: Any) -> None:
        """Report one use of a model or a paid service, what it used and what it cost. Counts into the run's price.

        ``extraction``: True for an attempt to fill the form (a call whose answer may change its values); False
        for any other use (a model that only decides whether to extract now, a summary kept for later, …). Only
        extraction attempts count as model calls: calls per minute, tokens per call, calls that changed a value.

        ``usage``: how much of each unit it used, by the unit's name. The report knows ``input_tokens`` (input no
        cache held), ``cache_read_tokens``, ``cache_write_tokens`` and ``output_tokens``; any other name is a unit
        too (``audio_seconds``, ``requests``, …). ``prices``: your vendor's price of one unit, in euros, for every
        unit of ``usage`` (0 for a free one). Its price is the sum of usage × price over the units.

        ``latency_s``: how long the use took. ``ok=False`` or an ``error`` marks a use that failed. Anything else
        you pass is kept with it in the run folder.
        """
        if not isinstance(extraction, bool):
            raise TypeError(f"extraction must be True or False, got {extraction!r}")
        usage, prices = dict(usage or {}), dict(prices or {})
        for unit, amount in usage.items():
            if not isinstance(unit, str) or not _UNIT_NAME.match(unit):
                raise ValueError(f"a unit is named with letters, digits and _, starting with a letter: {unit!r}")
            _number(f"usage[{unit!r}]", amount)
            if unit not in prices:
                raise ValueError(f"no price for {unit!r}: give prices[{unit!r}], in euros per unit (0 if it's free)")
            _number(f"prices[{unit!r}]", prices[unit])
        if latency_s is not None:
            _number("latency_s", latency_s)
        clash = _RESERVED & meta.keys()
        if clash:
            raise ValueError(f"{', '.join(sorted(clash))}: the run folder sets these itself")
        if extraction:
            self.calls += 1
        else:
            self.other_uses += 1
        record = {"model": model, "extraction": extraction, "usage": usage,
                  "prices": {unit: prices[unit] for unit in usage},
                  "eur": sum(amount * prices[unit] for unit, amount in usage.items()),
                  "latency_s": latency_s, "ok": bool(ok) and error is None, **meta}
        if error is not None:
            record["error"] = str(error)
        self._recorder.call(self.conversation_id, self.now(), record)

    def _check(self, field_id: str, value: Any) -> None:
        spec = self._fields.get(field_id)
        if spec is None:
            raise InvalidValue(f"unknown field {field_id!r}")
        error = value_error(spec, value)
        if error:
            raise InvalidValue(f"{field_id}: {error}")

    def _check_open(self) -> None:
        if self.closed:
            self.late_updates += 1
            raise SessionClosed("the session is closed: values published after run() ends are ignored")
        if not self._clock.started:
            raise SessionClosed("the conversation hasn't started: publish values in run(), not in prepare()")
