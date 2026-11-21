"""An oracle: it reads the gold and publishes every value a fixed delay after its words arrive.

It is not a submission (it cheats). It exists to check the environment and to show what the scores mean:
with no noise it must reach final F1 1.0 and a latency equal to its delay, and each kind of noise moves
exactly the metric it should.

    uv run lfe run examples.oracle_extractor:Oracle --dataset spokenwoz --n 5 --speed 20
    uv run lfe run examples.oracle_extractor:SloppyOracle --dataset spokenwoz --n 5 --speed 20
"""

from __future__ import annotations

import asyncio
import random
from collections import deque
from typing import Any

from live_form_eval import Extractor, FieldType, Session, load_config
from live_form_eval.data.common.model import Gold
from live_form_eval.data.common.store import load_dataset


class Oracle(Extractor):
    delay_s = 1.0  # every value appears this long after the stream delivered the words that say it
    missing = 0.0  # share of values never shown
    wrong = 0.0  # share of typed values shown wrong
    invented = 0.0  # share of empty text fields filled with something anyway
    seed = 0

    def __init__(self) -> None:
        self._config = load_config()
        self._gold: dict[str, Gold] = {}
        self._plans: dict[str, list[tuple[float, str, Any]]] = {}

    async def prepare(self, session: Session) -> None:
        """Read the gold before the conversation's clock starts, and off the event loop: reading files while
        other conversations are replayed would hold up their clocks (30 times as much at ``--speed 30``)."""
        self._plans[session.conversation_id] = await asyncio.to_thread(self._plan, session)

    def _gold_of(self, conversation_id: str) -> Gold:
        if conversation_id not in self._gold:
            for dataset, split in (("spokenwoz", "dev"), ("spokenwoz", "test"), ("ami", None)):
                try:
                    data = load_dataset(self._config, dataset, split=split, ids=[conversation_id])
                except (KeyError, FileNotFoundError):
                    continue
                self._gold.update(data.gold)
                break
        return self._gold[conversation_id]

    def _plan(self, session: Session) -> list[tuple[float, str, Any]]:
        """(when its words were spoken, field, value) for everything the oracle will publish."""
        rng = random.Random(f"{self.seed}:{session.conversation_id}")
        gold = self._gold_of(session.conversation_id)
        fields = {spec.field_id: spec for spec in session.form.fields}
        plan: list[tuple[float, str, Any]] = []
        for value in gold.values:
            if rng.random() < self.missing:
                continue
            shown = value.value
            if shown is not None and rng.random() < self.wrong:
                shown = _another(fields[value.field_id], shown)
            plan.append((value.said, value.field_id, shown))
        options: dict[str, list[tuple[float, str]]] = {}
        for item in gold.items:  # multiselect fields: the options named so far
            if fields[item.field_id].type == FieldType.multiselect and rng.random() >= self.missing:
                options.setdefault(item.field_id, []).append((item.said, item.text))
        for field_id, items in options.items():
            items.sort()
            plan += [(spoken, field_id, [text for _, text in items[: k + 1]]) for k, (spoken, _) in enumerate(items)]
        facts: dict[str, list[Any]] = {}
        for fact in gold.facts:  # free-text fields: the facts that hold, one per line
            if rng.random() >= self.missing:
                facts.setdefault(fact.field_id, []).append(fact)
        for field_id, known in facts.items():
            changes = {fact.said for fact in known} | {fact.until for fact in known if fact.until is not None}
            for moment in sorted(changes):
                holding = [fact.text for fact in known
                           if fact.said <= moment and (fact.until is None or moment < fact.until)]
                plan.append((moment, field_id, "\n".join(holding)))
        filled = {field_id for _, field_id, _ in plan}
        for spec in session.form.fields:
            if spec.field_id not in filled and spec.type == FieldType.text and rng.random() < self.invented:
                plan.append((rng.uniform(5, 30), spec.field_id, "something nobody said"))
        return sorted(plan, key=lambda step: step[0])

    async def run(self, session: Session) -> None:
        waiting = deque(self._plans.pop(session.conversation_id))
        publishing: list[asyncio.Task] = []
        heard = float("-inf")  # the audio time the stream has covered so far

        async def publish(when: float, field_id: str, value: Any) -> None:
            await session.sleep_until(when)
            session.update(field_id, value)

        async for partial in session.partials():
            heard = max([heard] + [word.end for word in partial.words])
            due: dict[str, Any] = {}
            while waiting and waiting[0][0] <= heard + 1e-3:  # its words have just arrived
                _, field_id, value = waiting.popleft()
                due[field_id] = value  # several pieces of one value in one message: the last one stands
            for field_id, value in due.items():
                publishing.append(asyncio.create_task(publish(partial.t + self.delay_s, field_id, value)))
        for _, field_id, value in waiting:  # nothing in practice: the last turn covers the whole audio
            session.update(field_id, value)
        await asyncio.gather(*publishing)
        session.report_usage(model="oracle", extraction=True)  # it calls no model: nothing used, nothing to pay


class SloppyOracle(Oracle):
    """The oracle with typical errors: slower, 10% missing, 5% wrong, a few invented."""

    delay_s, missing, wrong, invented = 2.0, 0.10, 0.05, 0.02


def _another(spec, value: Any) -> Any:
    """A wrong value of the right type."""
    if spec.type == FieldType.select:
        return next(option for option in spec.options if option != value)
    if spec.type == FieldType.checkbox:
        return not value
    if spec.type == FieldType.integer:
        return value + 1
    if spec.type in (FieldType.float, FieldType.slider):
        low, high = spec.range or (0.0, value + 1)
        return high if value != high else low
    if spec.type == FieldType.time:
        return "00:01" if value != "00:01" else "00:02"
    return f"{value} x"
