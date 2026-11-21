"""What `lfe prepare` builds for each conversation: its timed words, its form and its gold.

Times are seconds of audio since the conversation started. A gold value is "said" at the end of the words that
support it, or at the end of the conversation when the data points at no words. ``source`` says where a said
time comes from:

- ``span``  the dataset's own annotation (a caller span, a summary's linked dialogue acts, a topic segment, …)
- ``rule``  SpokenWOZ, no span: the caller's turn ends with the value's last piece, so it is that turn's end;
            or a service at the word of the turn that names it ("a train")
- ``llm``   an LLM pointed at the words (annotations/, shipped): a SpokenWOZ value without span or rule, or a
            fact of an AMI free-text field
- ``turn``  the end of the caller's turn (a withdrawn value; a service the caller asks about without naming it)
- ``end``   no supporting words: the end of the conversation
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# where a gold value's said time comes from; "label": the words two readers quoted for a label they corrected
Source = Literal["span", "label", "rule", "llm", "turn", "end"]
PRECISE_SOURCES = ("span", "label", "rule", "llm")  # said times that point at words (usable for live scoring)


class SpokenWord(BaseModel):
    """One word of the dataset transcript, with punctuation attached (as a speech-to-text vendor writes it)."""

    model_config = ConfigDict(frozen=True)

    text: str
    start: float
    end: float
    speaker: str  # the dataset's speaker id ("caller", "agent"; AMI "A"–"D"); its role is in Conversation.roles


class Conversation(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    dataset: Literal["spokenwoz", "ami"]
    split: str
    # Every speaker's words in transcript order: a dataset turn (SpokenWOZ) or dialogue act (AMI) stays together
    # and the units are ordered by start time, so overlapping speech isn't shuffled word by word.
    words: tuple[SpokenWord, ...]
    end: float  # end of the last word
    roles: dict[str, str]  # speaker id → role ("caller", "agent"; "Project Manager", …)
    meta: dict[str, Any] = Field(default_factory=dict)  # form_id; unit_sizes (words per turn / dialogue act)


class GoldValue(BaseModel):
    """From ``said`` on, the field should show ``value`` (None = the value was withdrawn)."""

    model_config = ConfigDict(frozen=True)

    field_id: str
    value: Any
    said: float
    source: Source
    detail: dict[str, Any] = Field(default_factory=dict)  # e.g. the caller turn and the words of the span


class GoldItem(BaseModel):
    """One list item (a decision, an action, a multiselect option) that should be shown from ``said`` on."""

    model_config = ConfigDict(frozen=True)

    field_id: str
    text: str
    said: float
    source: Source
    detail: dict[str, Any] = Field(default_factory=dict)


class GoldText(BaseModel):
    """A text field's reference (an abstract, an answer) as the dataset gives it: what the field should hold
    at the end.

    ``said`` is the end of the words that support it (the end of the conversation when there are none).
    ``text`` "" means the field should stay empty (an unanswerable question). What the field should hold while
    the conversation runs is in the gold's ``facts``.
    """

    model_config = ConfigDict(frozen=True)

    field_id: str
    text: str
    said: float
    source: Source
    detail: dict[str, Any] = Field(default_factory=dict)


class GoldFact(BaseModel):
    """One statement a free-text field should make from ``said`` on, until ``until`` (None = to the end).

    The facts of a field are its timeline: at any moment the field should say the facts that hold then and
    nothing else. A fact with an ``until`` is something the conversation settled and changed later; the facts
    without one are what the field should hold at the end.
    """

    model_config = ConfigDict(frozen=True)

    field_id: str
    text: str
    said: float
    until: float | None = None
    source: Source
    detail: dict[str, Any] = Field(default_factory=dict)  # the line and the words an LLM pointed at, …


class GoldSnapshot(BaseModel):
    """SpokenWOZ's own label: every field's value after one caller turn (for joint goal accuracy)."""

    model_config = ConfigDict(frozen=True)

    t: float  # end of that caller turn
    values: dict[str, Any]


class Gold(BaseModel):
    model_config = ConfigDict(frozen=True)

    conversation_id: str
    values: tuple[GoldValue, ...] = ()  # typed fields; several per field when the value changes over time
    items: tuple[GoldItem, ...] = ()  # list fields (textarea with one item per line; multiselect options)
    texts: tuple[GoldText, ...] = ()
    facts: tuple[GoldFact, ...] = ()  # free-text fields over time: the statements an answer has to make
    accepted: tuple[GoldFact, ...] = ()  # … and those it may also make: never owed, never counted as invented
    snapshots: tuple[GoldSnapshot, ...] = ()
