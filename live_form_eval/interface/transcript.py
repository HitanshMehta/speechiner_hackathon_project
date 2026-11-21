"""The live transcript an extractor receives: partials and finished turns, as a speech-to-text vendor sends them.

A :class:`Partial` is Speechineer's ``TranscriptDelta`` (``kind``, ``text``, ``turn_id``) plus what AssemblyAI's
streaming API also gives with speaker labels on: the arrival time, the turn's dominant speaker and the words
with their times and speakers.

- ``kind="partial"``: the turn so far. It **replaces** the previous partial of that turn (never append it).
- ``kind="turn"``: the finished turn. It replaces the turn's last partial; the turn's text no longer changes.

A turn ends at a pause, not at a speaker change, so one turn can hold several speakers: read ``words`` for who
said what. Speakers are "A", "B", … in order of first speech; "PENDING" marks audio too short to attribute.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Word(BaseModel):
    """One word: its text as written, its audio time and, in finished turns, its speaker."""

    model_config = ConfigDict(frozen=True)

    text: str
    start: float  # seconds since the conversation started
    end: float
    speaker: str | None = None  # "A", "B", … or "PENDING"; None in partials, or when speaker labels are off


class Partial(BaseModel):
    """One message of the live transcript."""

    model_config = ConfigDict(frozen=True)

    t: float  # arrival time, seconds since the conversation started
    kind: Literal["partial", "turn"]
    turn_id: str
    text: str  # the whole turn so far (partial) or the finished turn (turn)
    speaker: str | None = None  # the turn's dominant speaker so far; "PENDING" under ~1 s of audio; None when off
    words: tuple[Word, ...] = ()

    def pieces(self) -> list[tuple[str | None, str]]:
        """The text speaker by speaker: [(speaker, what they say), …], in order.

        Finished turns use the word speakers, so a turn that holds a question and its quick answer has two
        pieces; partials have only the dominant speaker, so they are one piece (speaker None when unlabelled).
        """
        if self.kind == "turn" and any(w.speaker for w in self.words):
            pieces: list[tuple[str | None, str]] = []
            for word in self.words:
                if pieces and pieces[-1][0] == word.speaker:
                    pieces[-1] = (word.speaker, f"{pieces[-1][1]} {word.text}")
                else:
                    pieces.append((word.speaker, word.text))
            return pieces
        return [(self.speaker, self.text)]

    def labelled_text(self) -> str:
        """The text with a speaker prefix at each change, e.g. "A: hello there B: hi" (plain text when unlabelled)."""
        return " ".join(f"{speaker}: {text}" if speaker else text for speaker, text in self.pieces())
