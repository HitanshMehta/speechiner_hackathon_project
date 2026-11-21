"""`lfe check`: run an extractor on a short made-up call and check that it keeps the contract.

The call has two speakers and a form with one field of every type. Two copies run at the same time, as
conversations do in a real run. The check looks at how the extractor behaves, not at whether its values are right:
it must return in time, raise nothing, and publish only values its fields can hold.
"""

from __future__ import annotations

import asyncio
import inspect
import traceback
from dataclasses import dataclass, field
from typing import Any

from live_form_eval.config import Config
from live_form_eval.data.common.model import SpokenWord
from live_form_eval.harness.clock import Clock
from live_form_eval.harness.runner import wait_for
from live_form_eval.harness.stream import build_stream
from live_form_eval.interface.extractor import Extractor
from live_form_eval.interface.schema import FieldSpec, FieldType, Form, OptionsConfig, RangeConfig
from live_form_eval.interface.session import InvalidValue, Session

CHECK_SPEED = 8.0  # the made-up call lasts about 45 s; the check replays it in about 6
_CALL = (
    ("caller", "Hello, I'd like to book a room for two people in the north of town."),
    ("agent", "Sure. For which date, and for how many nights?"),
    ("caller", "From the fifth of June twenty twenty seven, arriving around half past three in the afternoon."),
    ("agent", "Would you like breakfast or parking?"),
    ("caller", "Breakfast and parking, please. My budget is eighty nine euros fifty a night, four stars if possible."),
    ("agent", "Can I have your name, your email and a phone number?"),
    ("caller", "Anna Keller, anna dot keller at example dot com, and zero one seven six five five five zero one four two."),
    ("agent", "Thank you. Anything else?"),
    ("caller", "Our company site is example dot com slash travel. Please note that we arrive by train."),
)
_T = FieldType
TOY_FORM = Form(
    form_id="check", name="Contract check",
    system_prompts={"form_data_extraction": "You fill a hotel booking form during a phone call. The caller is the "
                                            "first speaker. Fill a field only once the caller has stated its value."},
    fields=(
        FieldSpec(field_id="guest_name", label="Guest name", type=_T.text, prompt="The caller's name."),
        FieldSpec(field_id="notes", label="Notes", type=_T.textarea, prompt="Anything else the hotel should know."),
        FieldSpec(field_id="email", label="Email", type=_T.email, prompt="The caller's email address."),
        FieldSpec(field_id="phone", label="Phone", type=_T.phone, prompt="The caller's phone number, digits only."),
        FieldSpec(field_id="website", label="Website", type=_T.url, prompt="The company website the caller mentions."),
        FieldSpec(field_id="people", label="People", type=_T.integer, prompt="Number of guests."),
        FieldSpec(field_id="budget", label="Budget per night (€)", type=_T.float, prompt="Budget per night in euros."),
        FieldSpec(field_id="stars", label="Stars", type=_T.slider, prompt="Star rating the caller wants.",
                  field_configuration=RangeConfig(range=(0, 5))),
        FieldSpec(field_id="breakfast", label="Breakfast", type=_T.checkbox, prompt="Whether the caller wants breakfast."),
        FieldSpec(field_id="area", label="Area", type=_T.select, prompt="Part of town.",
                  field_configuration=OptionsConfig(options=("centre", "north", "south", "east", "west"))),
        FieldSpec(field_id="extras", label="Extras", type=_T.multiselect, prompt="Extras the caller asks for.",
                  field_configuration=OptionsConfig(options=("breakfast", "parking", "late checkout", "airport transfer"))),
        FieldSpec(field_id="arrival_date", label="Arrival date", type=_T.date, prompt="Date of arrival."),
        FieldSpec(field_id="arrival_time", label="Arrival time", type=_T.time, prompt="Time of arrival."),
        FieldSpec(field_id="check_in", label="Check-in", type=_T.datetime, prompt="Date and time of arrival."),
        FieldSpec(field_id="contact", label="Contact", type=_T.template,
                  prompt="The caller's contact details as {\"name\": …, \"email\": …, \"phone\": …}."),
    ))


def toy_words() -> list[SpokenWord]:
    """The made-up call as timed words: 0.3 s per word, 0.9 s between turns."""
    words, t = [], 0.5
    for speaker, text in _CALL:
        for token in text.split():
            words.append(SpokenWord(text=token, start=round(t, 2), end=round(t + 0.25, 2), speaker=speaker))
            t += 0.3
        t += 0.9
    return words


class _Memory:
    """Keeps what a session publishes, in memory."""

    def __init__(self) -> None:
        self.predictions: list[tuple[str, float, str, Any]] = []
        self.calls: list[tuple[str, float, dict[str, Any]]] = []

    def prediction(self, conversation: str, t: float, field_id: str, value: Any) -> None:
        self.predictions.append((conversation, t, field_id, value))

    def call(self, conversation: str, t: float, record: dict[str, Any]) -> None:
        self.calls.append((conversation, t, record))


class _CheckedSession(Session):
    """A session that remembers every value it had to refuse."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.refused: list[str] = []

    def update(self, field_id: str, value: Any) -> None:
        try:
            super().update(field_id, value)
        except InvalidValue as error:
            self.refused.append(str(error))
            raise

    def update_snapshot(self, values: dict[str, Any]) -> None:
        try:
            super().update_snapshot(values)
        except InvalidValue as error:
            self.refused.append(str(error))
            raise


@dataclass
class ContractReport:
    problems: list[str] = field(default_factory=list)  # the extractor breaks the contract
    warnings: list[str] = field(default_factory=list)  # allowed, but probably not what you want
    facts: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def lines(self) -> list[str]:
        out = [f"  {fact}" for fact in self.facts]
        out += [f"WARNING  {warning}" for warning in self.warnings]
        out += [f"PROBLEM  {problem}" for problem in self.problems]
        out.append("contract check passed" if self.ok else "contract check FAILED")
        return out


async def _one(extractor: Extractor, config: Config, memory: _Memory, key: str, report: ContractReport) -> _CheckedSession:
    stream = build_stream(toy_words(), config.stream, key=key)
    clock = Clock(CHECK_SPEED)
    session = _CheckedSession(key, TOY_FORM, stream, clock, memory)
    try:
        await extractor.prepare(session)
    except Exception:  # noqa: BLE001 - any failure of a submission is reported
        report.problems.append(f"prepare() raised:\n{traceback.format_exc(limit=4)}")
        return session
    clock.start()
    deadline = session.stream_end + config.stream.final_grace_s
    try:
        await asyncio.wait_for(extractor.run(session), timeout=max(0.0, deadline - clock.now()) / CHECK_SPEED)
    except TimeoutError:
        report.problems.append(f"run() had not returned {config.stream.final_grace_s:g} s after the last partial: "
                               "return once the stream ends and your last value is published")
    except InvalidValue as error:
        report.problems.append(f"run() stopped on a value its field can't hold ({error}): check the value before "
                               "publishing it, or catch live_form_eval.InvalidValue")
    except Exception:  # noqa: BLE001
        report.problems.append(f"run() raised:\n{traceback.format_exc(limit=4)}")
    session.closed = True
    try:
        await extractor.close(session)
    except Exception:  # noqa: BLE001
        report.problems.append(f"close() raised:\n{traceback.format_exc(limit=4)}")
    return session


def check_extractor(extractor: Extractor, config: Config) -> ContractReport:
    """Run the made-up call twice at the same time and report how the extractor behaved."""
    report = ContractReport()
    if not isinstance(extractor, Extractor):
        report.problems.append("the class must subclass live_form_eval.Extractor")
        return report
    if not inspect.iscoroutinefunction(extractor.run):
        report.problems.append("run() must be `async def`")
        return report
    memory = _Memory()

    async def both() -> list[_CheckedSession]:
        return list(await asyncio.gather(*(_one(extractor, config, memory, key, report) for key in ("check-1", "check-2"))))

    sessions = wait_for(both())
    per_call = {key: sum(1 for p in memory.predictions if p[0] == key) for key in ("check-1", "check-2")}
    fields = sorted({p[2] for p in memory.predictions})
    report.facts.append(f"values published: {per_call['check-1']} and {per_call['check-2']} (two calls at the same "
                        f"time), in {len(fields)} of {len(TOY_FORM.fields)} fields")
    attempts = sum(1 for c in memory.calls if c[2]["extraction"])
    report.facts.append(f"uses reported: {attempts} model calls to fill the form, {len(memory.calls) - attempts} "
                        f"other uses, €{sum(c[2]['eur'] for c in memory.calls):.4f} in total")
    refused = [text for session in sessions for text in session.refused]
    if refused and not any("can't hold" in problem for problem in report.problems):
        report.warnings.append(f"{len(refused)} values were refused because their field can't hold them, "
                               f"e.g. {refused[0]}")
    if not memory.predictions:
        report.warnings.append("no value was published: call session.update() or session.update_snapshot()")
    if not memory.calls:
        report.warnings.append("no use was reported: call session.report_usage(model=…, extraction=True, usage=…, "
                               "prices=…) after each model call, or your price counts as 0 and the run can't be "
                               "ranked fairly")
    elif not attempts:
        report.warnings.append("no use was reported as a model call to fill the form (extraction=True): calls per "
                               "minute, tokens per call and the calls that changed a value stay empty")
    if any(session.late_updates for session in sessions):
        report.warnings.append("values were published after run() had ended; they are ignored")
    if min(per_call.values()) == 0 < max(per_call.values()):
        report.warnings.append("only one of the two simultaneous calls got values: keep per-conversation state "
                               "inside run(), not on self")
    return report
