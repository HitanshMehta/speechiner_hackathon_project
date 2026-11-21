"""Replay the gated extractor with in-memory responses and no provider access."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from live_form_eval import Form, Partial, Session, load_config
from live_form_eval.data.common.store import load_dataset
from live_form_eval.harness.clock import Clock
from live_form_eval.harness.runner import run_conversation
from my_extractor import MyExtractor
from strategies.delta import parse_delta
from strategies.openrouter import Completion

ROOT = Path(__file__).resolve().parents[1]
EMPTY_DELTA = '{"set":[],"facts":[]}'


@dataclass
class Records:
    predictions: list[tuple[float, str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def prediction(self, conversation: str, t: float, field_id: str, value: Any) -> None:
        self.predictions.append((t, field_id, value))

    def call(self, conversation: str, t: float, record: dict[str, Any]) -> None:
        self.calls.append({**record, "t": t})


class MockBackend:
    def __init__(self, *replies: str) -> None:
        self.replies = iter(replies)
        self.requests: list[dict[str, Any]] = []

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> Completion:
        self.requests.append(json.loads(messages[-1]["content"]))
        return Completion(next(self.replies, EMPTY_DELTA), "mock/gate", {"requests": 1}, {"requests": 0}, 0)


def assignment(field_id: str, value: Any, quote: str) -> str:
    return json.dumps({"set": [{"field": field_id, "value": value, "quote": quote}], "facts": []})


async def synthetic_run(form: Form, events: list[Partial], backend: MockBackend) -> Records:
    clock, records = Clock(speed=10), Records()
    session = Session("smoke", form, events, clock, records)
    clock.start()
    await asyncio.wait_for(MyExtractor(backend=backend).run(session), timeout=5)
    assert session.calls == len(backend.requests)
    return records


async def check_scheduling() -> None:
    form = Form.from_import_file({"fields": [
        {"field_id": "day", "type": "select", "options": ["monday", "tuesday"]},
        {"field_id": "name", "type": "text"},
    ]}, "smoke")
    stream = [
        Partial(t=1, kind="partial", turn_id="1", text="Book the room for Monday", speaker="A"),
        Partial(t=3, kind="turn", turn_id="1", text="Book the room for Monday", speaker="A"),
        Partial(t=8, kind="turn", turn_id="2", text="Actually make it Tuesday", speaker="A"),
    ]
    mock = MockBackend(EMPTY_DELTA, assignment("day", "monday", "room for Monday"),
                       assignment("day", "tuesday", "make it Tuesday"))
    records = await synthetic_run(form, stream, mock)
    assert [(field, value) for _, field, value in records.predictions] == [("day", "monday"), ("day", "tuesday")]
    assert len(mock.requests) == 3 and "new_finished_turns" in mock.requests[1]

    growing = [
        Partial(t=t, kind="partial", turn_id="1", speaker="A", text=text)
        for t, text in [(1, "My name"), (3, "My name is"), (5, "My name is Jane"),
                        (8.5, "My name is Jane and"), (11, "My name is Jane and yours")]
    ]
    mock = MockBackend(assignment("name", "Jane", "My name is Jane"))
    records = await synthetic_run(form, growing, mock)
    assert 6.8 <= records.predictions[0][0] < 8.5, records.predictions

    # All the turns arrive together, so the final buffer needs more than one request.
    backlog = [Partial(t=1, kind="turn", turn_id=str(i), text=f"guest {i} " + "booking " * 100, speaker="A")
               for i in range(8)]
    mock = MockBackend()
    await synthetic_run(form, backlog, mock)
    assert len(mock.requests) == 3
    assert "guest 7" in mock.requests[-1]["new_finished_turns"]
    assert [m.get("final_pass", False) for m in mock.requests] == [False, False, True]
    print("Passed: correction, empty partial review, stable deadline and final buffer drain.", flush=True)


async def main() -> None:
    config = load_config(ROOT / "config.yaml")
    await check_scheduling()
    for name in ("spokenwoz", "ami"):
        data = load_dataset(config, name, official_sample=True)
        mock, records = MockBackend(), Records()
        candidate = MyExtractor(backend=mock)
        if name == "ami":
            form = next(iter(data.forms.values()))
            fact = "The Project Manager scheduled a meeting."
            more = "The Marketing Expert volunteered to write the brief."
            message = json.dumps({"set": [], "facts": [
                {"field": "actions", "add": [more, more], "remove": [], "quote": "I can write the brief"},
            ]})
            assert parse_delta(form, {"actions": fact}, message, evidence_sources=("C: I can write the brief",)) == {
                "actions": f"{fact}\n{more}",
            }
        else:
            form = next(iter(data.forms.values()))
            greeting = "A: I'm looking for a train"
            guess = assignment("train_day", "wednesday", "looking for a train")
            assert parse_delta(form, {}, guess, evidence_sources=(greeting,)) == {}
            time = "quarter past eight in the morning"
            arrival = assignment("train_arrive_by", "08:15", time)
            assert parse_delta(form, {}, arrival, evidence_sources=(f"A: Can I arrive by {time}?",)) == {
                "train_arrive_by": "08:15",
            }
            reported_time = assignment("train_leave_at", "09:20", "09:20")
            assert parse_delta(form, {}, reported_time, evidence_sources=(
                "A: What time does the train leave? B: The departure time is 09:20. A: Thank you.",
            )) == {}
            assert parse_delta(form, {}, reported_time, evidence_sources=(
                "A: I want to leave at 09:20.",
            )) == {"train_leave_at": "09:20"}
            assert parse_delta(form, {}, reported_time, evidence_sources=(
                "B: The train leaves at 09:20. A: That works for me.",
            )) == {"train_leave_at": "09:20"}
            station = assignment("train_departure", "ely", "leaving eli")
            assert parse_delta(form, {}, station, evidence_sources=("A: We're leaving Eli today.",)) == {
                "train_departure": "ely",
            }
            mock_service = MockBackend()
            local = await synthetic_run(form, [
                Partial(t=1, kind="partial", turn_id="1", speaker="B",
                        text="How can I help? I'm looking for a taxi."),
                Partial(t=3, kind="turn", turn_id="1", speaker="B",
                        text="How can I help? I'm looking for a taxi."),
            ], mock_service)
            assert [(field, value) for _, field, value in local.predictions] == [("services", ["taxi"])]
            assert local.predictions[0][0] < 2
        outcomes = await asyncio.gather(*(
            run_conversation(candidate, convo, data.form_of(convo.id), config, records, speed=75)
            for convo in data.conversations.values()
        ))
        assert all(row["status"] == "ok" and row["late_updates"] == 0 for row in outcomes), outcomes
        assert len(mock.requests) == len(records.calls) == sum(row["calls"] for row in outcomes) > 0
        path, = sorted((ROOT / "speechineer").glob(f"*-{name}/run.json"))
        baseline = json.loads(path.read_text(encoding="utf-8"))
        baseline_calls = sum(row["calls"] for row in baseline["conversations"])
        print(f"{name} · {len(outcomes)} conversations · {len(mock.requests)} mocked requests · "
              f"{baseline_calls} saved baseline calls", flush=True)
    print("Mock request counts measure scheduling only. No API calls or scoring were performed.")


if __name__ == "__main__":
    asyncio.run(main())
