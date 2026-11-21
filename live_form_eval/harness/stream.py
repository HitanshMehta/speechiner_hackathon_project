"""Timed dataset words → the live stream a speech-to-text vendor would send (partials and finished turns).

No dataset has partial annotations, only final words with times, so the stream is simulated. Every parameter is in
config.yaml (``stream``); the values were measured on real AssemblyAI u3-rt-pro sessions through Speechineer's
engine (docs/stream.md). Speaker labels follow AssemblyAI's streaming diarization:

1. The words, in transcript order, are cut into turns the way the vendor did it on real dialogues: after a
   sentence end followed by a silence ≥ ``turn_pause_s`` (it never cut mid-sentence), after any silence
   ≥ ``turn_max_silence_s``, or once a turn lasts ``max_turn_s``. So a turn can hold several speakers (a quick
   reply), and a speaker's long turn is split at its sentence ends.
2. While a turn has new words, a partial arrives every ``partial_every_s`` (± ``partial_jitter_s``) with the
   turn's words that ended at least ``word_delay_s`` earlier.
3. The finished turn arrives ``turn_close_s`` after its last word ends, before the next turn's first partial.
4. With ``speaker_labels``: speakers are "A", "B", … in order of first speech. A turn's ``speaker`` is its dominant
   speaker (most audio), "PENDING" under ``pending_under_s`` of audio. Finished turns label each word, and a
   same-speaker run shorter than ``pending_under_s`` inside a turn is "PENDING" (a backchannel). Partials carry
   the dominant label but unlabelled words, as AssemblyAI labels only final words.

Not simulated: recognition errors beyond the dataset's own, the vendor rewriting earlier words of a partial
(7–27% of real partials), number formatting, and speaker-label mistakes.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from live_form_eval.config import StreamConfig
from live_form_eval.data.common.model import SpokenWord
from live_form_eval.data.common.speakers import speaker_letters
from live_form_eval.interface.transcript import Partial, Word

PENDING = "PENDING"
_SENTENCE_END = (".", "?", "!")
_EPSILON = 0.001


def cut_turns(words: Sequence[SpokenWord], config: StreamConfig) -> list[list[SpokenWord]]:
    """Words (in transcript order) → turns: see the module's rule 1."""
    turns: list[list[SpokenWord]] = []
    current: list[SpokenWord] = []
    latest_end = 0.0
    for word in words:
        silence = word.start - latest_end
        if current and ((silence >= config.turn_pause_s and current[-1].text.endswith(_SENTENCE_END))
                        or silence >= config.turn_max_silence_s
                        or word.end - current[0].start > config.max_turn_s):
            turns.append(current)
            current = []
        latest_end = max(latest_end, word.end) if current else word.end
        current.append(word)
    if current:
        turns.append(current)
    return turns


def _dominant(words: Sequence[SpokenWord], letters: dict[str, str], config: StreamConfig) -> str:
    """The speaker with the most audio; PENDING when the words span less than pending_under_s."""
    if max(w.end for w in words) - min(w.start for w in words) < config.pending_under_s:
        return PENDING
    audio: dict[str, float] = {}
    for word in words:
        audio[word.speaker] = audio.get(word.speaker, 0.0) + (word.end - word.start)
    return letters[max(audio, key=lambda speaker: (audio[speaker], letters[speaker]))]


def _labelled_words(turn: Sequence[SpokenWord], letters: dict[str, str], config: StreamConfig) -> tuple[Word, ...]:
    """Words of a finished turn with their speakers; a short same-speaker run is PENDING."""
    out: list[Word] = []
    k = 0
    while k < len(turn):
        j = k
        while j + 1 < len(turn) and turn[j + 1].speaker == turn[k].speaker:
            j += 1
        run = turn[k: j + 1]
        label = PENDING if run[-1].end - run[0].start < config.pending_under_s else letters[run[0].speaker]
        out += [Word(text=w.text, start=w.start, end=w.end, speaker=label) for w in run]
        k = j + 1
    return tuple(out)


def build_stream(words: Sequence[SpokenWord], config: StreamConfig, key: str) -> list[Partial]:
    """The partials and finished turns of one conversation, in arrival order. ``key`` seeds the jitter."""
    rng = random.Random(f"{config.seed}:{key}")
    letters = speaker_letters(words)
    labels = config.speaker_labels
    stream: list[Partial] = []
    previous_close = float("-inf")
    for order, turn in enumerate(cut_turns(words, config)):
        turn_id = str(order)
        close = max(w.end for w in turn) + config.turn_close_s
        # the first partial: once a first word is visible, at a random phase of the vendor's cadence
        t = max(min(w.end for w in turn) + config.word_delay_s + rng.uniform(0, config.partial_every_s),
                previous_close + _EPSILON)
        shown = 0
        last_partial = None
        while t < close:
            visible = [w for w in turn if w.end <= t - config.word_delay_s]
            if len(visible) > shown:
                shown = len(visible)
                last_partial = round(t, 3)
                stream.append(Partial(
                    t=last_partial, kind="partial", turn_id=turn_id, text=" ".join(w.text for w in visible),
                    speaker=_dominant(visible, letters, config) if labels else None,
                    words=tuple(Word(text=w.text, start=w.start, end=w.end) for w in visible)))
            t += config.partial_every_s + rng.uniform(-config.partial_jitter_s, config.partial_jitter_s)
        close = max(close, previous_close + _EPSILON, (last_partial or 0.0) + _EPSILON)
        stream.append(Partial(
            t=round(close, 3), kind="turn", turn_id=turn_id, text=" ".join(w.text for w in turn),
            speaker=_dominant(turn, letters, config) if labels else None,
            words=_labelled_words(turn, letters, config) if labels
            else tuple(Word(text=w.text, start=w.start, end=w.end) for w in turn)))
        previous_close = close
    return stream
