"""Free text against what a field should say: a shown text and the facts of a gold value, through a judge.

A free-text field holds a list (one item per line) or a text. Its gold value at a moment is a set of facts: the
statements the field should make then (timeline/steps.py). The comparison goes both ways, with one judge question
per unit::

    complete = facts that the shown text supports ÷ facts
    correct  = shown units (items or sentences) that the facts support ÷ shown units

    similarity = the harmonic mean of the two (their F1)
    same       = both are 1: the text says every fact, and nothing that the facts don't say

An empty field against no facts is the same. An empty field against facts, or a text against no facts, is 0.
The judge's verdicts are kept, and ``ask`` takes every question of a conversation in one batch.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from live_form_eval.evaluation.matching.judge.nli import Judge
from live_form_eval.evaluation.matching.units import units_of

Question = tuple[str, str]  # (text, statement): does the text say the statement?


@dataclass(frozen=True)
class Support:
    """The counts behind one comparison."""

    units: int  # items or sentences of the shown text
    units_supported: int  # … that the facts support
    facts: int
    facts_supported: int  # facts that the shown text supports

    @property
    def correct(self) -> float:
        return self.units_supported / self.units if self.units else 0.0

    @property
    def complete(self) -> float:
        return self.facts_supported / self.facts if self.facts else 0.0

    @property
    def f1(self) -> float:
        total = self.correct + self.complete
        return 2 * self.correct * self.complete / total if total else 0.0


class TextMatch:
    """``same`` and ``similarity`` for free text, as the walk needs them (timeline/walk.py)."""

    def __init__(self, judge: Judge) -> None:
        self.judge = judge
        self._verdicts: dict[Question, bool] = {}

    @staticmethod
    def questions(text: Any, facts: Sequence[str] | None, is_list: bool,
                  accepted: Sequence[str] = ()) -> list[Question]:
        """The judge questions behind one comparison (none when a side is empty). ``accepted`` are statements
        the text may also make: a unit they support is not counted against the text, and none of them is owed."""
        units = units_of(text, is_list)
        if not units or not facts:
            return []
        return [(str(text), fact) for fact in facts] + [("\n".join([*facts, *accepted]), unit) for unit in units]

    def ask(self, questions: Iterable[Question]) -> None:
        """Put the questions that have no verdict yet to the judge, in one batch."""
        new = [question for question in dict.fromkeys(questions)
               if question not in self._verdicts and question[0].strip() and question[1].strip()]
        if new:
            self._verdicts.update(zip(new, self.judge.supports(new)))

    def says(self, text: str, statement: str) -> bool:
        """Does the text say the statement? An empty side never does."""
        if not text.strip() or not statement.strip():
            return False
        if (text, statement) not in self._verdicts:
            self.ask([(text, statement)])
        return self._verdicts[(text, statement)]

    def support(self, text: Any, facts: Sequence[str] | None, is_list: bool, accepted: Sequence[str] = ()) -> Support:
        units = units_of(text, is_list)
        facts = list(facts or ())
        if not units or not facts:
            return Support(len(units), 0, len(facts), 0)
        self.ask(self.questions(text, facts, is_list, accepted))
        gold_text = "\n".join([*facts, *accepted])
        return Support(len(units), sum(self.says(gold_text, unit) for unit in units),
                       len(facts), sum(self.says(str(text), fact) for fact in facts))

    def similarity(self, text: Any, facts: Sequence[str] | None, is_list: bool, accepted: Sequence[str] = ()) -> float:
        support = self.support(text, facts, is_list, accepted)
        return 1.0 if not support.units and not support.facts else support.f1

    def same(self, text: Any, facts: Sequence[str] | None, is_list: bool, accepted: Sequence[str] = ()) -> bool:
        support = self.support(text, facts, is_list, accepted)
        return support.units_supported == support.units and support.facts_supported == support.facts
