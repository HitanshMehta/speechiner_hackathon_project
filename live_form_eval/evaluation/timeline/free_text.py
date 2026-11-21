"""Free-text fields over a conversation: which of their texts are judged, and the final text against the gold.

A free-text field is a list (one item per line: decisions, actions, …) or a text (an abstract, an answer). Its
gold is a timeline of facts (``Gold.facts``). The walk compares what the field showed with the facts that held at
each moment (timeline/walk.py, with matching/free_text.py as its comparison). This module compares the final text
with the facts that hold at the end, in both directions:

- complete: is each final fact supported by the whole shown text?   → "correct" or "missing"
- correct:  is each shown unit supported by the final facts?        → unsupported ones are "invented" extras

Every comparison costs judge calls, so the number of texts judged per field is bounded: one per ``interval_s``
seconds of the conversation (``looks``). A field that changes no more often than that has every text judged.
One that is rewritten more often is only looked at every ``interval_s`` seconds, and a text that was replaced
before the next look is not seen. A field without facts (an unanswerable question, an empty list) must stay
empty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from live_form_eval.data.common.model import Gold, GoldFact
from live_form_eval.evaluation.matching.free_text import TextMatch, units_of
from live_form_eval.evaluation.timeline.final import Outcome, spoken_of
from live_form_eval.evaluation.timeline.shown import Shown
from live_form_eval.interface.schema import FieldType, Form


@dataclass(frozen=True)
class FieldFacts:
    """The gold of one free-text field."""

    is_list: bool  # its text is read as items (one per line); else as sentences
    facts: tuple[GoldFact, ...]
    accepted: tuple[str, ...] = ()  # statements the field may also make: not owed, and not counted as invented

    def holding(self, at: float) -> list[GoldFact]:
        """The facts that hold at the moment ``at``: said by then, and not changed by the conversation by then."""
        return [fact for fact in self.facts if fact.said <= at and (fact.until is None or at < fact.until)]

    @property
    def final(self) -> list[GoldFact]:
        """The facts that hold at the end."""
        return self.holding(math.inf)


@dataclass
class FieldSupport:
    """One free-text field of one conversation at the end: the counts behind its precision and recall."""

    field_id: str
    is_list: bool
    gold_units: int = 0  # final facts
    gold_supported: int = 0  # complete: final facts the shown text supports
    shown_units: int = 0
    shown_supported: int = 0  # correct: shown units the final facts support
    should_be_empty: bool = False
    accepted: tuple[str, ...] = ()  # the statements the field may also make
    outcomes: list[Outcome] = field(default_factory=list)


def gold_facts(form: Form, gold: Gold) -> dict[str, FieldFacts]:
    """Free-text field → its facts. A field is a text when the gold has a reference text for it, else a list."""
    texts = {text.field_id for text in gold.texts}
    with_reference = {text.field_id for text in gold.texts if text.text} | {item.field_id for item in gold.items}
    out: dict[str, FieldFacts] = {}
    for spec in form.fields:
        if spec.type != FieldType.textarea:
            continue
        facts = tuple(fact for fact in gold.facts if fact.field_id == spec.field_id)
        if not facts and spec.field_id in with_reference:
            raise RuntimeError(f"the gold of {gold.conversation_id} has a reference for {spec.field_id!r} but no "
                               "facts: it was built before free text had a timeline. Run `lfe prepare` again.")
        out[spec.field_id] = FieldFacts(spec.field_id not in texts, facts, tuple(
            fact.text for fact in gold.accepted if fact.field_id == spec.field_id))
    return out


def looks(shown: Shown, interval_s: float, end: float) -> list[tuple[float, Any]]:
    """The displays of a free-text field that are judged: (when it was published, the text; None = cleared).

    A field may have one text judged per ``interval_s`` seconds of the conversation (``end`` ÷ ``interval_s``
    texts). A field that changed no more often than that has every change judged. One that changed more often
    is looked at every ``interval_s`` seconds and at ``end``, and only the texts on screen at those moments are
    kept, each with the time it was published.
    """
    updates = [(t, value) for t, value in shown.updates if t <= end]
    if len(updates) > math.ceil(end / interval_s):
        last = len(updates) - 1
        updates = [(t, value) for k, (t, value) in enumerate(updates)
                   if k == last or max(1, math.ceil(t / interval_s)) * interval_s < updates[k + 1][0]]
    kept: list[tuple[float, Any]] = []
    for t, value in updates:
        if value != (kept[-1][1] if kept else None):  # what stood between two kept texts was not seen
            kept.append((t, value))
    return kept


def free_text_support(conversation: str, form: Form, gold: Gold, shown: dict[str, Shown], match: TextMatch,
                      interval_s: float, final_time: float, end: float) -> list[FieldSupport]:
    """Every free-text field of one conversation at the end, judged. All questions are asked in one batch.

    The form is read at ``final_time``, against the facts that hold then (at the end of a conversation: its
    final facts); ``end`` is when the conversation's scoring ends (it bounds how many texts are judged, see
    ``looks``).
    """
    plans = []
    questions: list[tuple[str, str]] = []
    for field_id, field_facts in gold_facts(form, gold).items():
        timeline = shown.get(field_id, Shown())
        final_text = timeline.at(final_time)
        final_text = "" if final_text is None else str(final_text)
        candidates = units_of(final_text, field_facts.is_list)
        final = field_facts.holding(final_time)
        gold_text = "\n".join([*(fact.text for fact in final), *field_facts.accepted])
        seen = [(published, "" if text is None else str(text)) for published, text in looks(timeline, interval_s, end)]
        questions += [(text, fact.text) for text in [final_text] + [text for _, text in seen] for fact in final]
        questions += [(gold_text, candidate) for candidate in candidates]
        plans.append((field_id, field_facts, final, final_text, candidates, seen, gold_text))
    match.ask(questions)

    results = []
    kind = FieldType.textarea.value
    for field_id, field_facts, final, final_text, candidates, seen, gold_text in plans:
        support = FieldSupport(field_id=field_id, is_list=field_facts.is_list, gold_units=len(final),
                               shown_units=len(candidates), should_be_empty=not final,
                               accepted=field_facts.accepted)
        if not final:
            result = "correctly_empty" if not candidates else "invented"
            support.outcomes.append(Outcome(conversation, field_id, kind, "value", "", result,
                                            shown=final_text or None))
            results.append(support)
            continue
        for fact in final:
            supported = match.says(final_text, fact.text)
            support.gold_supported += supported
            settled = None
            if supported:  # since which of the texts that were seen has it been there without a break?
                for published, text in reversed(seen):
                    if not match.says(text, fact.text):
                        break
                    settled = published
            support.outcomes.append(Outcome(
                conversation, field_id, kind, "item", fact.text, "correct" if supported else "missing",
                gold=fact.text, shown=final_text or None, said=fact.said, spoken=spoken_of(fact), source=fact.source,
                settled_at=settled))
        for candidate in candidates:
            supported = match.says(gold_text, candidate)
            support.shown_supported += supported
            if not supported:
                support.outcomes.append(Outcome(conversation, field_id, kind, "extra", candidate, "invented",
                                                shown=candidate))
        results.append(support)
    return results
