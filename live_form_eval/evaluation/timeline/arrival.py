"""When did the words that say a value reach the extractor?

The datasets time every word at the moment it was spoken. An extractor can only react once the live stream has
delivered that word, so every clock of the evaluation (latency, live F1, the per-turn snapshots) starts there:
at the arrival of the first partial or finished turn that covers the word. The time the speech-to-text service
takes to deliver a word is therefore not counted against the extractor.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from typing import Any

from live_form_eval.config import Config, StreamConfig
from live_form_eval.data.common.model import Conversation, Gold, GoldFact
from live_form_eval.harness.stream import build_stream
from live_form_eval.interface.transcript import Partial

_TOLERANCE = 1e-3  # gold times are rounded to the millisecond


class Arrival:
    """Maps a moment of the audio to the moment the stream delivered the words spoken until then."""

    def __init__(self, stream: Sequence[Partial]) -> None:
        self._covered: list[float] = []  # audio time covered after each message that brought new words
        self._at: list[float] = []  # … and when that message arrived
        covered = float("-inf")
        for message in stream:
            reach = max((word.end for word in message.words), default=covered)
            if reach > covered:
                covered = reach
                self._covered.append(covered)
                self._at.append(message.t)
        self.end = stream[-1].t if stream else 0.0  # when the last message arrived: the end of the input

    def __call__(self, spoken: float) -> float:
        """When the stream first covered audio time ``spoken`` (the last message's arrival for later times)."""
        index = bisect.bisect_left(self._covered, spoken - _TOLERANCE)
        return self._at[index] if index < len(self._at) else self.end


def stream_config_of(run: dict[str, Any], config: Config) -> StreamConfig:
    """The stream settings a run was replayed with (recorded in its run.json), else the current ones."""
    recorded = (run.get("config") or {}).get("stream")
    return StreamConfig.model_validate(recorded) if recorded else config.stream


def arrival_of(conversation: Conversation, stream_config: StreamConfig) -> Arrival:
    """The arrival map of one conversation: the same stream the extractor received, rebuilt."""
    return Arrival(build_stream(conversation.words, stream_config, key=conversation.id))


def in_extractor_time(gold: Gold, arrival: Arrival) -> Gold:
    """The gold with every said time moved to when its words reached the extractor.

    The spoken (audio) time is kept in each entry's ``detail["spoken"]``.
    """

    def move(entry: Any) -> Any:
        return entry.model_copy(update={"said": arrival(entry.said), "detail": {**entry.detail, "spoken": entry.said}})

    def move_fact(fact: GoldFact) -> GoldFact:
        """A fact holds from the arrival of the words that complete it to the arrival of those that end it."""
        moved = move(fact)
        return moved if fact.until is None else moved.model_copy(update={"until": arrival(fact.until)})

    return gold.model_copy(update={
        "values": tuple(move(value) for value in gold.values),
        "items": tuple(move(item) for item in gold.items),
        "texts": tuple(move(text) for text in gold.texts),
        "facts": tuple(move_fact(fact) for fact in gold.facts),
        "snapshots": tuple(snapshot.model_copy(update={"t": arrival(snapshot.t)}) for snapshot in gold.snapshots),
    })
