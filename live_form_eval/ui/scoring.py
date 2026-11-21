"""The score so far: the conversation scored as if it had ended at a given moment.

It is the evaluation's own scoring (``score_conversation`` with ``until``), on what was said and shown by that
moment, with the form read at that moment. So a value that was just said and is not on the form yet counts as
missing until the form shows it: the lines of the page dip at every new value for as long as the extractor takes.
From the end of the scoring on, the score so far is the official score of that form.

Free text needs the judge. Its verdicts are kept (results/.cache), so a run that was evaluated before is scored
again from them; only a text pair nobody judged yet loads the model.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from typing import Any

from live_form_eval.config import Config, StreamConfig
from live_form_eval.data.common.model import Conversation, Gold
from live_form_eval.evaluation.evaluate import ConversationScore, LazyMatch, score_conversation
from live_form_eval.evaluation.matching.free_text import TextMatch
from live_form_eval.evaluation.timeline.arrival import arrival_of, in_extractor_time
from live_form_eval.evaluation.timeline.shown import Shown, shown_fields
from live_form_eval.evaluation.timeline.steps import gold_steps
from live_form_eval.interface.schema import FieldType, Form
from live_form_eval.ui.playback import terms_payload

SCORING = threading.Lock()  # one scoring at a time: the judge and its verdicts are shared by everything the page shows
_JUST_AFTER = 1e-3  # a moment is scored just after what happens at it, so that the point shows its effect


class Scorer:
    """Scores one conversation of one run, at its end or as far as it has gone."""

    def __init__(self, config: Config, conversation: Conversation, form: Form, gold: Gold,
                 stream_config: StreamConfig, match: LazyMatch) -> None:
        self.config, self.conversation, self.form, self.gold = config, conversation, form, gold
        self.stream_config, self.match = stream_config, match
        self.arrival = arrival_of(conversation, stream_config)
        steps = gold_steps(form, in_extractor_time(gold, self.arrival))
        self.said = sorted({said for field_steps in steps.values() for said, _ in field_steps})
        self.free_text = any(spec.type == FieldType.textarea for spec in form.fields)

    def text_match(self) -> TextMatch | None:
        """The judge's verdicts, for a form that has free text."""
        return self.match.get() if self.free_text else None

    def score(self, shown: dict[str, Shown], calls: Sequence[dict[str, Any]], status: dict[str, Any],
              until: float | None = None) -> ConversationScore:
        """The conversation scored at its end, or as if it had ended at ``until``."""
        with SCORING:
            return score_conversation(self.conversation, self.form, self.gold, shown, calls, status,
                                      self.conversation.dataset, self.config, self.stream_config, self.match,
                                      until=until, arrival=self.arrival)

    def moments(self, shown: dict[str, Shown], after: float, until: float, every_s: float) -> list[float]:
        """The moments to score between ``after`` and ``until``: just after each gold value is said and each
        change of the form (where the score jumps), and every ``every_s`` seconds in between (precision moves
        with every second a value stays on screen)."""
        events = self.said + [t for timeline in shown.values() for t, _ in timeline.updates]
        grid = [k * every_s for k in range(int(after / every_s) + 1, int(until / every_s) + 1)]
        return sorted({t + _JUST_AFTER for t in events if after <= t < until} | {t for t in grid if after < t < until})

    def series(self, predictions: Sequence[dict[str, Any]], calls: Sequence[dict[str, Any]], status: dict[str, Any],
               every_s: float) -> Iterator[dict[str, Any]]:
        """The score so far over a whole run: one point per moment of ``moments``, and the official score at the
        end of the scoring."""
        shown = shown_fields(predictions, self.conversation.id)
        official = self.score(shown, calls, status)
        for t in self.moments(shown, 0.0, official.walk_end, every_s):
            yield terms_payload(self.score(shown, calls, status, until=t), t)
        yield terms_payload(official, official.walk_end)
