"""What a run used: the model calls that tried to fill the form, its other uses, what each carried and cost.

From the uses a run reported (``session.report_usage``) and the values it published::

    extraction attempt      a use reported with ``extraction=True``: a model call whose answer may change the form.
                            These are the "model calls" of the report: per audio minute, tokens per call, changed a
                            value, how long a call takes
    other use               a use reported with ``extraction=False`` (a model that decides whether to extract now,
                            …): it counts into the price, and in nothing else
    calls per audio minute  = extraction attempts ÷ minutes of conversation audio
    tokens, by kind         read from cache · fresh input · written to cache · output, of the extraction attempts
    cached share            = input tokens read from cache ÷ all input tokens
    changed a value         an extraction attempt counts as changing the form when a change of a field was
                            published nearest to it: every change goes to the attempt reported closest to it in time
                            (to the earlier one when two are equally close)
    price by unit           = Σ amount × price of the unit, over every use

A run made before units had prices reported each call's total in euros and its tokens, with ``input_tokens``
counting all input (cached or not). It is read as extraction attempts whose price is not split by unit
(``UNSPLIT``), and whose fresh input is what is left of ``input_tokens`` once the cached parts are taken out.

Reported next to the price, not ranked.
"""

from __future__ import annotations

import bisect
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from live_form_eval.evaluation.metrics.final_decision_latency import percentile
from live_form_eval.interface.session import TOKEN_UNITS

TOKEN_KINDS = ("cache_read", "fresh_input", "cache_write", "output")  # from the cheapest token to the dearest
KIND_UNITS = {"cache_read": "cache_read_tokens", "fresh_input": "input_tokens", "cache_write": "cache_write_tokens",
              "output": "output_tokens"}  # the unit an extractor reports each kind in
UNSPLIT = "(not split)"  # the price of a use reported only as a total; no unit can be named like this


@dataclass(frozen=True)
class Call:
    """One use a run reported: an extraction attempt (a model call whose answer may change the form), or another
    use."""

    conversation: str
    t: float  # when the run reported it, on the conversation's clock
    model: str
    extraction: bool
    usage: dict[str, float]  # unit → amount
    eur_by_unit: dict[str, float]  # unit → euros; a price reported only as a total is under UNSPLIT
    eur: float
    seconds: float | None  # how long the use took, when the run says so
    ok: bool
    changes: int  # changes of a field published nearest to this use: extraction attempts only (0 for the others)

    def tokens(self, kind: str) -> int:
        return int(self.usage.get(KIND_UNITS[kind], 0))

    @property
    def cache_read(self) -> int:
        """Input tokens the vendor read from its cache."""
        return self.tokens("cache_read")

    @property
    def fresh_input(self) -> int:
        """Input tokens no cache held."""
        return self.tokens("fresh_input")

    @property
    def cache_write(self) -> int:
        """Input tokens the vendor wrote to its cache."""
        return self.tokens("cache_write")

    @property
    def output(self) -> int:
        return self.tokens("output")

    @property
    def input(self) -> int:
        """Every input token, cached or not."""
        return self.cache_read + self.fresh_input + self.cache_write


@dataclass(frozen=True)
class CallStats:
    calls: int  # extraction attempts
    changing: int  # … that changed a value
    failed: int  # … that failed
    other_uses: int
    eur: float  # every use
    eur_other: float  # … of it, the other uses
    eur_by_unit: dict[str, float]  # every use, by unit
    tokens: dict[str, int]  # extraction attempts, by kind (TOKEN_KINDS)
    seconds_median: float | None  # how long an extraction attempt takes
    seconds_percentile: float | None  # … at the config's latency percentile
    conversations: int
    audio_minutes: float

    @property
    def per_form(self) -> float | None:
        return self.calls / self.conversations if self.conversations else None

    @property
    def per_audio_minute(self) -> float | None:
        return self.calls / self.audio_minutes if self.audio_minutes else None

    @property
    def changing_share(self) -> float | None:
        return self.changing / self.calls if self.calls else None

    @property
    def tokens_per_call(self) -> float | None:
        return sum(self.tokens.values()) / self.calls if self.calls else None

    @property
    def cached_share(self) -> float | None:
        sent = sum(self.tokens[kind] for kind in TOKEN_KINDS if kind != "output")
        return self.tokens["cache_read"] / sent if sent else None

    def tokens_per_form(self, kind: str) -> float | None:
        return self.tokens[kind] / self.conversations if self.conversations else None

    @property
    def other_uses_per_form(self) -> float | None:
        return self.other_uses / self.conversations if self.conversations else None

    @property
    def eur_other_per_form(self) -> float | None:
        return self.eur_other / self.conversations if self.conversations else None


def unit_order(units: Iterable[str]) -> list[str]:
    """Units in the order the report lists them: the tokens from the cheapest kind to the dearest, then the other
    units by name, then a price that is not split by unit."""
    known = [KIND_UNITS[kind] for kind in TOKEN_KINDS]
    present = set(units)
    return ([unit for unit in known if unit in present]
            + sorted(present - set(known) - {UNSPLIT}) + ([UNSPLIT] if UNSPLIT in present else []))


def _amount(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else 0.0


def _time(record: dict[str, Any]) -> float:
    return float(record.get("t") or 0.0)


def _extraction(record: dict[str, Any]) -> bool:
    """Was the use an attempt to fill the form? Runs made before the question existed reported only those."""
    return record.get("extraction", True) is not False


def usage_of(record: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    """What one reported use carried, by unit, and what each unit cost in euros (a line of calls.jsonl)."""
    if isinstance(record.get("usage"), dict):
        usage = {str(unit): _amount(amount) for unit, amount in record["usage"].items()}
        prices = record.get("prices") if isinstance(record.get("prices"), dict) else {}
        return usage, {unit: amount * _amount(prices.get(unit)) for unit, amount in usage.items()}
    # a run made before units had prices: input_tokens counts all input, the price is a total
    cache_read, cache_write = _amount(record.get("cache_read_tokens")), _amount(record.get("cache_write_tokens"))
    usage = {TOKEN_UNITS[0]: max(0.0, _amount(record.get("input_tokens")) - cache_read - cache_write),
             "cache_read_tokens": cache_read, "cache_write_tokens": cache_write,
             "output_tokens": _amount(record.get("output_tokens"))}
    eur = _amount(record.get("eur"))
    return {unit: amount for unit, amount in usage.items() if amount}, ({UNSPLIT: eur} if eur else {})


def conversation_calls(records: Sequence[dict[str, Any]], change_times: Sequence[float]) -> list[Call]:
    """The uses of one conversation in time order; each extraction attempt with the number of field changes
    published nearest to it. ``records`` are that conversation's lines of calls.jsonl; ``change_times`` holds one
    time per change."""
    ordered = sorted(records, key=_time)
    attempts = [index for index, record in enumerate(ordered) if _extraction(record)]
    times = [_time(ordered[index]) for index in attempts]
    nearest: Counter[int] = Counter()
    for moment in change_times if times else ():
        after = bisect.bisect_left(times, moment)  # the first attempt reported at or after the change
        before = after - 1
        earlier = after == len(times) or (before >= 0 and moment - times[before] <= times[after] - moment)
        nearest[attempts[before if earlier else after]] += 1
    calls = []
    for index, record in enumerate(ordered):
        usage, eur_by_unit = usage_of(record)
        seconds, eur = record.get("latency_s"), record.get("eur")
        calls.append(Call(
            conversation=str(record.get("conversation", "")), t=_time(record), model=str(record.get("model", "")),
            extraction=_extraction(record), usage=usage, eur_by_unit=eur_by_unit,
            eur=_amount(eur) if eur is not None else sum(eur_by_unit.values()),
            seconds=_amount(seconds) if isinstance(seconds, (int, float)) and math.isfinite(seconds) else None,
            ok=record.get("ok") is not False and not record.get("error"), changes=nearest[index]))
    return calls


def call_stats(calls: Sequence[Call], conversations: int, audio_seconds: float, p: float) -> CallStats:
    """Adds up the uses of a form or of a run; ``p`` is the percentile reported next to the median call time."""
    attempts = [call for call in calls if call.extraction]
    seconds = [call.seconds for call in attempts if call.seconds is not None]
    by_unit: dict[str, float] = {}
    for call in calls:
        for unit, eur in call.eur_by_unit.items():
            by_unit[unit] = by_unit.get(unit, 0.0) + eur
    return CallStats(
        calls=len(attempts), changing=sum(call.changes > 0 for call in attempts),
        failed=sum(not call.ok for call in attempts), other_uses=len(calls) - len(attempts),
        eur=sum(call.eur for call in calls), eur_other=sum(call.eur for call in calls if not call.extraction),
        eur_by_unit={unit: by_unit[unit] for unit in unit_order(by_unit)},
        tokens={kind: sum(call.tokens(kind) for call in attempts) for kind in TOKEN_KINDS},
        seconds_median=percentile(seconds, 50), seconds_percentile=percentile(seconds, p),
        conversations=conversations, audio_minutes=audio_seconds / 60)
