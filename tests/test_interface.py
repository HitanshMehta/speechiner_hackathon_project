"""The interface a submission codes against: field values, form files, the session, the contract check."""

import asyncio

import pytest

from live_form_eval import Extractor, InvalidValue, Session
from live_form_eval.data.common.model import Conversation
from live_form_eval.harness.clock import Clock
from live_form_eval.harness.runner import run_conversation
from live_form_eval.interface.contract import TOY_FORM, check_extractor, toy_words
from live_form_eval.interface.schema import FieldType, Form, value_error
from live_form_eval.interface.session import SessionClosed
from live_form_eval.interface.transcript import Partial


def test_toy_form_has_every_field_type():
    assert {spec.type for spec in TOY_FORM.fields} == set(FieldType)


@pytest.mark.parametrize("field_id, good, bad", [
    ("guest_name", "Anna", 3), ("people", 2, "2"), ("people", 2, True), ("budget", 89.5, "89.5"),
    ("stars", 4.0, 6.0), ("breakfast", True, "yes"), ("area", "north", "North"),
    ("extras", ["parking"], ["a pool"]), ("extras", [], "parking"), ("arrival_date", "2027-06-05", "05.06.2027"),
    ("arrival_date", "2027-06-05", "2027-13-45"), ("arrival_time", "15:30", "3:30 pm"),
    ("check_in", "2027-06-05T15:30", "2027-06-05 15:30"), ("contact", {"name": "Anna"}, "Anna"),
])
def test_value_types(field_id, good, bad):
    spec = TOY_FORM.field(field_id)
    assert value_error(spec, good) is None
    assert value_error(spec, bad) is not None
    assert value_error(spec, None) is None  # clearing a field is always allowed


def test_form_file_round_trip():
    assert Form.from_import_file(TOY_FORM.to_import_file(), form_id="check") == TOY_FORM


class _Memory:
    def __init__(self):
        self.predictions, self.calls = [], []

    def prediction(self, conversation, t, field_id, value):
        self.predictions.append((field_id, value))

    def call(self, conversation, t, record):
        self.calls.append(record)


def test_session_stamps_and_refuses():
    async def scenario():
        memory, clock = _Memory(), Clock(speed=100)
        stream = [Partial(t=0.5, kind="turn", turn_id="0", text="hello")]
        session = Session("c", TOY_FORM, stream, clock, memory)
        with pytest.raises(SessionClosed):
            session.update("people", 2)  # before the conversation started
        clock.start()
        assert [p.text async for p in session.partials()] == ["hello"]
        session.update_snapshot({"people": 2, "area": "north"})
        with pytest.raises(InvalidValue):
            session.update_snapshot({"guest_name": "Anna", "people": "two"})  # nothing of a bad snapshot is kept
        with pytest.raises(InvalidValue):
            session.update("no_such_field", 1)
        session.report_usage(model="m", extraction=True, usage={"input_tokens": 5}, prices={"input_tokens": 0.002})
        session.closed = True
        with pytest.raises(SessionClosed):
            session.update("people", 3)
        return memory, session

    memory, session = asyncio.run(scenario())
    assert memory.predictions == [("people", 2), ("area", "north")]
    assert memory.calls[0]["eur"] == pytest.approx(0.01) and session.late_updates == 1


def test_a_use_is_reported_unit_by_unit():
    memory, clock = _Memory(), Clock(speed=100)
    session = Session("c", TOY_FORM, [], clock, memory)
    prices = {"input_tokens": 2e-6, "cache_read_tokens": 5e-7, "output_tokens": 8e-6, "audio_seconds": 1e-4}
    # a model call that tries to fill the form, with the usual units and one of its own; a price list may hold more
    session.report_usage(model="m", extraction=True, prices=prices, latency_s=0.8, trigger="partial",
                         usage={"input_tokens": 300, "cache_read_tokens": 1000, "output_tokens": 20, "audio_seconds": 4.5})
    call = memory.calls[0]
    assert (call["extraction"], call["usage"]["audio_seconds"], call["latency_s"], call["trigger"]) == (True, 4.5, 0.8, "partial")
    assert call["prices"] == prices  # one price per unit used
    assert call["eur"] == pytest.approx(300 * 2e-6 + 1000 * 5e-7 + 20 * 8e-6 + 4.5 * 1e-4)
    # another use (a model that only decides whether to extract now), and one that failed
    session.report_usage(model="gate", extraction=False, usage={"output_tokens": 1}, prices={"output_tokens": 8e-6})
    session.report_usage(model="m", extraction=True, error="timeout")
    assert [(use["extraction"], use["ok"]) for use in memory.calls[1:]] == [(False, True), (True, False)]
    assert memory.calls[2]["eur"] == 0 and memory.calls[2]["error"] == "timeout"
    assert (session.calls, session.other_uses) == (2, 1)

    with pytest.raises(ValueError, match="no price for 'requests'"):
        session.report_usage(model="m", extraction=True, usage={"requests": 1}, prices=prices)
    with pytest.raises(ValueError, match="number ≥ 0"):
        session.report_usage(model="m", extraction=True, usage={"input_tokens": -3}, prices=prices)
    with pytest.raises(ValueError, match="letters, digits and _"):
        session.report_usage(model="m", extraction=True, usage={"input tokens": 3}, prices={"input tokens": 1e-6})
    with pytest.raises(TypeError, match="True or False"):
        session.report_usage(model="m", extraction="yes")
    with pytest.raises(ValueError, match="sets these itself"):
        session.report_usage(model="m", extraction=True, t=3.0)
    assert len(memory.calls) == 3  # nothing of a refused use is kept


class Careful(Extractor):
    """Publishes one valid value per partial and reports a call."""

    async def run(self, session: Session) -> None:
        async for partial in session.partials():
            session.update("notes", partial.labelled_text())
            session.report_usage(model="none", extraction=True)


class Careless(Extractor):
    async def run(self, session: Session) -> None:
        async for _ in session.partials():
            session.update("people", "two")  # a string in an integer field


class NeverReturns(Extractor):
    async def run(self, session: Session) -> None:
        async for _ in session.partials():
            pass
        await asyncio.sleep(3600)


class HoldsSomethingOpen(Extractor):
    """Opens something for the conversation (think of a billed session) and ends it in close()."""

    def __init__(self) -> None:
        self.open: set[str] = set()
        self.closed: list[str] = []

    async def run(self, session: Session) -> None:
        self.open.add(session.conversation_id)
        async for _ in session.partials():
            pass

    async def close(self, session: Session) -> None:
        await asyncio.sleep(0)  # ending a session takes a request: close() must be let to finish
        self.open.discard(session.conversation_id)
        self.closed.append(session.conversation_id)


def test_contract_check_passes_a_careful_extractor(config):
    report = check_extractor(Careful(), config)
    assert report.ok and not report.warnings, report.lines()


def test_contract_check_names_the_problem(config):
    report = check_extractor(Careless(), config)
    assert not report.ok and "can't hold" in " ".join(report.problems)


def test_contract_check_catches_a_run_that_never_returns(config):
    fast = config.model_copy(update={"stream": config.stream.model_copy(update={"final_grace_s": 2})})
    report = check_extractor(NeverReturns(), fast)
    assert not report.ok and "had not returned" in " ".join(report.problems)


def test_a_run_that_is_stopped_still_closes_its_extractor(config):
    # the page's "Stop run" and Ctrl+C on `lfe run` cancel the conversation: close() is owed all the same, or an
    # extractor that holds a session open would leave it open
    extractor = HoldsSomethingOpen()
    call = Conversation(id="c", dataset="spokenwoz", split="dev", end=toy_words()[-1].end, roles={}, meta={},
                        words=tuple(toy_words()))

    async def scenario():
        run = asyncio.ensure_future(run_conversation(extractor, call, TOY_FORM, config, _Memory(), speed=1.0))
        while not extractor.open:  # until the conversation is going on
            await asyncio.sleep(0.01)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):  # the stop itself goes on once close() is done
            await run

    asyncio.run(scenario())
    assert (extractor.open, extractor.closed) == (set(), ["c"])


def test_made_up_call_is_two_speakers():
    assert {word.speaker for word in toy_words()} == {"caller", "agent"}
