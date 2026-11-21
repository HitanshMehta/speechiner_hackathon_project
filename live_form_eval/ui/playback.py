"""One conversation as the page needs it: its live stream, its form, what each field should show from when, and
(for a run) what the form showed and how the evaluation read it.

Every time here is on the extractor's clock, as in the evaluation: a gold value is "said" when the stream delivered
its last word (timeline/arrival.py). The moment those words are heard in the recording is given next to it
(``spoken``), about a second earlier.
"""

from __future__ import annotations

import bisect
from collections import Counter
from typing import Any

from live_form_eval.config import Config, StreamConfig
from live_form_eval.data.common.model import Conversation, Gold
from live_form_eval.data.common.speakers import speaker_letters
from live_form_eval.evaluation.evaluate import FORM_READ_WHEN_RUN_ENDS, ConversationScore
from live_form_eval.evaluation.matching.compare import SCALAR_TYPES
from live_form_eval.evaluation.matching.free_text import TextMatch, units_of
from live_form_eval.evaluation.metrics.calls import TOKEN_KINDS, Call
from live_form_eval.evaluation.timeline.arrival import Arrival, in_extractor_time
from live_form_eval.evaluation.timeline.free_text import gold_facts
from live_form_eval.evaluation.timeline.shown import Shown
from live_form_eval.evaluation.timeline.steps import gold_steps, scoring_end
from live_form_eval.harness.stream import build_stream
from live_form_eval.interface.schema import FieldType, Form
from live_form_eval.interface.transcript import Partial

_HEARD_WORDS = 8  # how many words of the conversation are quoted with a gold value: the ones that end it


def stream_payload(stream: list[Partial]) -> list[dict[str, Any]]:
    """The live stream, message by message: when it arrives, whether the turn is finished, and its text speaker
    by speaker ([speaker, text] pairs; a partial has one)."""
    return [{"t": message.t, "kind": message.kind, "turn": message.turn_id,
             "parts": [[speaker or "", text] for speaker, text in message.pieces()]} for message in stream]


def _heard(conversation: Conversation, ends: list[float], spoken: float | None) -> str | None:
    """The last words of the conversation up to the audio time ``spoken``: what was being said when a gold value
    became complete."""
    if spoken is None:
        return None
    index = bisect.bisect_right(ends, spoken + 1e-3)
    return " ".join(word.text for word in conversation.words[max(0, index - _HEARD_WORDS): index]) or None


def _plain(value: Any) -> Any:
    """A gold or shown value as JSON holds it: the options or facts of a step as a list."""
    return list(value) if isinstance(value, tuple) else value


def conversation_payload(config: Config, conversation: Conversation, form: Form, gold: Gold,
                         stream_config: StreamConfig) -> dict[str, Any]:
    """Everything about one conversation that doesn't depend on a run: its stream as ``stream_config`` cuts it,
    its form, and per field the gold steps (what it should show from when)."""
    stream = build_stream(conversation.words, stream_config, key=conversation.id)
    arrival = Arrival(stream)
    clocked = in_extractor_time(gold, arrival)
    steps = gold_steps(form, clocked)
    lists = {field_id: facts.is_list for field_id, facts in gold_facts(form, clocked).items()}
    ends = sorted(word.end for word in conversation.words)
    by_id = {spec.field_id: spec for spec in form.fields}

    entries: dict[str, list[Any]] = {}  # field → the gold entries behind its steps, to say where each step comes from
    for value in clocked.values:
        if value.field_id in by_id and by_id[value.field_id].type in SCALAR_TYPES:
            entries.setdefault(value.field_id, []).append(value)
    for item in clocked.items:
        if item.field_id in by_id and by_id[item.field_id].type == FieldType.multiselect:
            entries.setdefault(item.field_id, []).append(item)
    for fact in clocked.facts:
        if fact.field_id in by_id and by_id[fact.field_id].type == FieldType.textarea:
            entries.setdefault(fact.field_id, []).append(fact)

    fields = []
    for spec in form.fields:
        timeline = []
        for said, value in steps.get(spec.field_id, []):
            # the entry that completes this step: the one of this message that was spoken last
            behind = max((entry for entry in entries.get(spec.field_id, []) if entry.said == said),
                         key=lambda entry: entry.detail.get("spoken", 0.0), default=None)
            spoken = behind.detail.get("spoken") if behind is not None else None
            source = behind.source if behind is not None else None
            timeline.append({
                "said": said, "spoken": spoken, "value": _plain(value), "source": source,
                "heard": None if source == "end" else (behind.detail.get("quote") if behind is not None else None)
                or _heard(conversation, ends, spoken)})
        fields.append({
            "id": spec.field_id, "label": spec.label, "type": spec.type.value, "prompt": spec.prompt,
            "options": list(spec.options), "range": list(spec.range) if spec.range else None,
            "list": lists.get(spec.field_id), "steps": timeline})

    letters = speaker_letters(conversation.words)
    limit_s = config.score.speed_limit_s[conversation.dataset]
    return {
        "dataset": conversation.dataset, "split": conversation.split, "id": conversation.id,
        "seconds": conversation.end,  # the recording's last word
        "input_end": arrival.end,  # when the last message arrives
        # until when a run can go on: the extractor may take ``final_grace_s`` after the last message
        "end": arrival.end + stream_config.final_grace_s,
        # until when the form is scored, where that doesn't depend on the run (docs/metrics.md)
        "scored_until": None if conversation.dataset in FORM_READ_WHEN_RUN_ENDS
        else scoring_end(steps, arrival.end, limit_s),
        "speed_limit_s": limit_s,
        "form": {"id": form.form_id, "name": form.name, "rules": form.description, "fields": fields},
        "speakers": {letter: conversation.roles.get(speaker, speaker) for speaker, letter in letters.items()},
        "stream": stream_payload(stream),
    }


def updates_payload(shown: dict[str, Shown]) -> dict[str, list[list[Any]]]:
    """Field → its visible changes as [time, value] pairs (None = cleared), as the evaluation reads them."""
    return {field_id: [[t, value] for t, value in timeline.updates] for field_id, timeline in shown.items()}


def calls_payload(calls: list[Call]) -> list[dict[str, Any]]:
    """Every use a run reported, as the page draws it: when, whether it was an attempt to fill the form, what it
    cost in all and unit by unit ({unit: euros}), and the tokens of each kind it carried."""
    return [{"t": call.t, "model": call.model, "extraction": call.extraction, "eur": call.eur,
             "units": call.eur_by_unit, "seconds": call.seconds, "ok": call.ok, "changes": call.changes,
             **{kind: call.tokens(kind) for kind in TOKEN_KINDS}} for call in calls]


def terms_payload(score: ConversationScore, t: float) -> dict[str, Any]:
    """The four terms and the score of one scoring, as a point of the page's graphs."""
    terms = score.terms
    return {"t": t, "score": terms.score, "completeness": terms.completeness, "precision": terms.precision,
            "speed": terms.speed, "outcome": terms.outcome}


def walks_payload(form: Form, score: ConversationScore, match: TextMatch | None,
                  only_latest: bool = False) -> dict[str, dict[str, Any]]:
    """Field → how the evaluation read it: per gold step its score, its delay and the moment the form showed it;
    per display its grade and what it counts as.

    A free-text field also gets, for each text that was judged, its sentences or items and the judge's verdicts
    against each gold step it met (``judged``): which facts the text says, and which of its units the facts
    support. ``only_latest`` keeps those for the last judged text only (a live run only ever shows the present).
    """
    by_id = {spec.field_id: spec for spec in form.fields}
    lists = {support.field_id: support.is_list for support in score.supports}
    accepted = {support.field_id: support.accepted for support in score.supports}
    out: dict[str, dict[str, Any]] = {}
    for field_id, walk in score.walks.items():
        steps = [{"said": step.said, "score": step.score, "delay": step.delay, "landed": step.landed,
                  "shown": walk.displays[step.display].shown if step.display is not None else None}
                 for step in walk.steps]
        displays = [{"t": display.shown, "grade": display.grade, "kind": display.kind, "step": display.step,
                     "kept": display.kept, "seconds": display.seconds} for display in walk.displays]
        entry: dict[str, Any] = {"steps": steps, "displays": displays}
        if by_id[field_id].type == FieldType.textarea and match is not None:
            is_list = lists.get(field_id, True)
            judged = []
            texts = [display for display in walk.displays if not display.kept and display.value is not None]
            for display in texts[-1:] if only_latest else texts:
                text, until = str(display.value), display.shown + display.seconds
                units = units_of(text, is_list)
                against = {}
                for index, step in enumerate(walk.steps):
                    following = walk.steps[index + 1].said if index + 1 < len(walk.steps) else float("inf")
                    if step.value is None or step.said >= until or following <= display.shown:
                        continue  # the step did not hold while this text was on screen
                    facts = list(step.value)
                    reference = "\n".join([*facts, *accepted.get(field_id, ())])
                    against[str(index)] = {"facts": [match.says(text, fact) for fact in facts],
                                           "units": [match.says(reference, unit) for unit in units]}
                judged.append({"t": display.shown, "units": units, "against": against})
            entry["judged"] = judged
        out[field_id] = entry
    return out


def final_payload(score: ConversationScore) -> dict[str, Any]:
    """The conversation at its end: the official numbers of this form."""
    counts = Counter(outcome.outcome for outcome in score.outcomes)
    return {**terms_payload(score, score.walk_end), "status": score.status, "eur": score.eur, "calls": score.calls,
            "other_uses": score.other_uses,
            **{name: counts.get(name, 0) for name in ("correct", "wrong", "missing", "invented")}}
