"""A complete extractor to start from: an LLM re-reads the transcript whenever it has changed.

It talks to any OpenAI-compatible API. Set the constants below (or the environment variables), then:

    uv sync --extra openai
    export OPENAI_API_KEY=…            # and OPENAI_BASE_URL=… for another provider
    uv run lfe check examples.llm_extractor:LLMExtractor
    uv run lfe run examples.llm_extractor:LLMExtractor --dataset spokenwoz --n 2 --speed 5
    uv run lfe evaluate runs/<run id>

How it works:
- ``listen`` keeps the transcript up to date: finished turns, plus the turn being spoken.
- ``extract`` calls the model whenever the transcript has changed since the last call, one call at a time,
  and once more after the conversation has ended.
- Each call returns the whole form; valid values are published at once, and the call's cost is reported.

Everything here is yours to change: the prompt, the model, when to call it, what to send.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from openai import AsyncOpenAI

from live_form_eval import Extractor, FieldSpec, FieldType, Form, InvalidValue, Session

MODEL = os.environ.get("LFE_EXAMPLE_MODEL", "gpt-4.1-mini")
PRICES = {  # your vendor's price of one unit for this model, in euros
    "input_tokens": 0.37e-6,  # input it had not seen before
    "cache_read_tokens": 0.09e-6,  # input it read from its cache (the part of a prompt it has seen before)
    "output_tokens": 1.47e-6,
}

SYSTEM = """You fill a form from the live transcript of a conversation. {rules}

Speakers are labelled A, B, …; "PENDING" marks a few words whose speaker isn't known yet. The last line may be an
unfinished turn.

Return one JSON object with a key for every field you can fill now, and no other keys. Use null to clear a field
you filled before and that no longer holds. Leave out every field the conversation gives no value for.
Value formats: {formats}"""
FORMATS = ("text, textarea, email, phone, url: a string · integer: a whole number · float, slider: a number · "
           "checkbox: true or false · select: exactly one of the field's options · multiselect: a list of the "
           "field's options · date: \"YYYY-MM-DD\" · time: \"HH:MM\" (24 h) · datetime: \"YYYY-MM-DDTHH:MM\".")


def describe(spec: FieldSpec) -> dict[str, Any]:
    """One field as the model sees it."""
    out: dict[str, Any] = {"id": spec.field_id, "type": spec.type.value, "meaning": spec.prompt}
    if spec.options:
        out["options"] = list(spec.options)
    if spec.range:
        out["range"] = list(spec.range)
    return out


def fits(spec: FieldSpec, value: Any) -> Any:
    """Small repairs of what models often get almost right; the session refuses anything that still doesn't fit."""
    if value is None:
        return None
    if spec.type in (FieldType.float, FieldType.slider) and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if spec.type == FieldType.integer and isinstance(value, float) and value.is_integer():
        return int(value)
    if spec.type == FieldType.multiselect and isinstance(value, str):
        return [value]
    return value


class LLMExtractor(Extractor):
    def __init__(self) -> None:
        self.client = AsyncOpenAI()  # shared by every conversation of the run

    async def run(self, session: Session) -> None:
        turns: list[str] = []  # state of this conversation only: it lives inside run()
        partial = ""
        changed = asyncio.Event()
        ended = False

        async def listen() -> None:
            nonlocal partial, ended
            async for p in session.partials():
                if p.kind == "turn":
                    turns.append(p.labelled_text())
                    partial = ""
                else:
                    partial = p.labelled_text()
                changed.set()
            ended = True
            changed.set()

        async def extract() -> None:
            published: dict[str, Any] = {}
            while True:
                await changed.wait()
                changed.clear()
                last_round = ended  # read before the call: what arrives during it triggers another round
                transcript = "\n".join(turns + ([partial] if partial else []))
                if transcript:
                    values = await self.ask(session, transcript, published)
                    for field_id, value in values.items():
                        try:
                            session.update(field_id, fits(session.form.field(field_id), value))
                            published[field_id] = value
                        except (InvalidValue, KeyError):
                            pass  # a value the field can't hold, or an unknown field: skip it, keep the rest
                if last_round:
                    return

        await asyncio.gather(listen(), extract())

    async def ask(self, session: Session, transcript: str, published: dict[str, Any]) -> dict[str, Any]:
        """One model call: the whole form from the transcript so far. Reports its cost."""
        form: Form = session.form
        started = time.monotonic()
        try:
            response = await self.client.chat.completions.create(
                model=MODEL, temperature=0, response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM.format(rules=form.description, formats=FORMATS)},
                    {"role": "user", "content": json.dumps({
                        "fields": [describe(spec) for spec in form.fields],
                        "filled_so_far": published, "transcript": transcript}, ensure_ascii=False)},
                ])
        except Exception as error:  # noqa: BLE001 - a failed call must not end the conversation
            session.report_usage(model=MODEL, extraction=True, latency_s=time.monotonic() - started,
                                 error=repr(error))
            return {}
        usage = response.usage
        # How much of the input the vendor read from its cache, when it says so: it is cheaper, so it is a unit of
        # its own. OpenAI counts it inside prompt_tokens; input_tokens is only the rest.
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None) or 0
        session.report_usage(model=MODEL, extraction=True,  # a call that tries to fill the form
                             usage={"input_tokens": usage.prompt_tokens - cached, "cache_read_tokens": cached,
                                    "output_tokens": usage.completion_tokens},
                             prices=PRICES, latency_s=time.monotonic() - started)
        try:
            values = json.loads(response.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            return {}
        return values if isinstance(values, dict) else {}
