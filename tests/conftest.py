"""Shared test helpers: a stream config with round numbers, and words built from a compact script."""

import pytest

from live_form_eval.config import StreamConfig, load_config
from live_form_eval.data.common.model import SpokenWord


@pytest.fixture(scope="session")
def config():
    """The repo's own config.yaml (it must always load)."""
    return load_config()


@pytest.fixture
def stream_config() -> StreamConfig:
    return StreamConfig(partial_every_s=1.0, partial_jitter_s=0.0, word_delay_s=0.4, turn_pause_s=0.2,
                        turn_max_silence_s=1.5, turn_close_s=0.3, max_turn_s=60, speaker_labels=True,
                        pending_under_s=1.0, final_grace_s=15, seed=0)


def words(script: list[tuple[str, float, str]], word_s: float = 0.3) -> list[SpokenWord]:
    """[(speaker, start of the first word, "some text."), …] → timed words, one every ``word_s``."""
    out = []
    for speaker, start, text in script:
        for k, token in enumerate(text.split()):
            out.append(SpokenWord(text=token, start=round(start + k * word_s, 3),
                                  end=round(start + k * word_s + word_s - 0.05, 3), speaker=speaker))
    return out
