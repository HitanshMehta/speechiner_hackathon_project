"""Speaker labels as a streaming diarizer gives them: "A", "B", … in order of first speech."""

from __future__ import annotations

import string
from collections.abc import Sequence

from live_form_eval.data.common.model import SpokenWord


def speaker_letters(words: Sequence[SpokenWord]) -> dict[str, str]:
    """Dataset speaker id → "A", "B", …: the first speaker to talk is "A", the next new one "B", and so on."""
    first: dict[str, float] = {}
    for word in words:
        first[word.speaker] = min(first.get(word.speaker, word.start), word.start)
    ordered = sorted(first, key=lambda speaker: (first[speaker], speaker))
    return {speaker: string.ascii_uppercase[k] if k < 26 else f"S{k}" for k, speaker in enumerate(ordered)}
