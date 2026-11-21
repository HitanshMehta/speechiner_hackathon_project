"""AMI's recordings, one meeting at a time (data/common/audio.py keeps them).

The corpus has one file per meeting: the mix of the participants' headset microphones, 16 kHz, 16-bit, one
channel, about 2 MB a minute (27 MB for a meeting of 14 minutes). Licence: CC BY 4.0, as the rest of the corpus.
"""

from __future__ import annotations

from pathlib import Path

from live_form_eval.data.common.audio import Progress, download

RECORDING = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{meeting}/audio/{meeting}.Mix-Headset.wav"


def fetch(split: str, conversation_id: str, target: Path, by_hand: Path, progress: Progress | None = None) -> None:
    """Write the recording of one meeting to ``target`` as the corpus has it (``split`` and ``by_hand`` are not
    needed here: every meeting has a file of its own)."""
    download(RECORDING.format(meeting=conversation_id), target, progress)
