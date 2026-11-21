"""SpokenWOZ's recordings, one call at a time (data/common/audio.py keeps them).

The test split's recordings are one zip archive of 2.2 GB on Hugging Face. A zip can be read in parts, so one
call's recording is taken out of it with a few byte-range requests (about 3 MB a call), without downloading the
archive. The train and dev recordings come as one compressed tar archive of 10 GB, which can't be read in
parts: a dev call is only heard when its WAV file is put into the audio folder by hand.

A recording is 8 kHz, 16-bit, with the caller and the agent on a channel each. Licence: CC BY-NC 4.0, as the
rest of SpokenWOZ. The source is pinned to the revision this environment was built on.
"""

from __future__ import annotations

import threading
import urllib.error
import zipfile
from pathlib import Path

from live_form_eval.data.common.audio import AudioUnavailable, Progress, RemoteFile, copy_with_progress

TEST_ARCHIVE = ("https://huggingface.co/datasets/ssz1111/SpokenWOZ-Test-Audio-Fixed/resolve/"
                "937f06db7b47e265c56a9927d868a85c94d701b5/audio_5700_test.zip")
_lock = threading.Lock()  # one reader of the archive at a time: its file position is shared
_archive: zipfile.ZipFile | None = None  # opened once per process: its directory is one request of 1 MB


def fetch(split: str, conversation_id: str, target: Path, by_hand: Path, progress: Progress | None = None) -> None:
    """Write the recording of one call to ``target`` as the dataset has it. ``by_hand`` is where a file can be
    put for a call whose recording can't be fetched."""
    global _archive
    if split != "test":
        raise AudioUnavailable(
            f"SpokenWOZ's {split} recordings come as one archive of 10 GB that can't be read call by call, so "
            f"they aren't fetched. To hear this call, save its recording as {by_hand}")
    with _lock:
        try:
            if _archive is None:
                _archive = zipfile.ZipFile(RemoteFile(TEST_ARCHIVE))
            name = next((member for member in _archive.namelist()
                         if member.endswith(f"/{conversation_id}.wav") and not member.startswith("__MACOSX")), None)
            if name is None:
                raise AudioUnavailable(f"the archive of SpokenWOZ's test recordings holds no {conversation_id}.wav")
            target.parent.mkdir(parents=True, exist_ok=True)
            with _archive.open(name) as member, target.open("wb") as out:
                copy_with_progress(member, out, _archive.getinfo(name).file_size, progress)
        except (urllib.error.URLError, zipfile.BadZipFile, OSError) as error:
            target.unlink(missing_ok=True)
            raise AudioUnavailable(f"could not fetch the recording of {conversation_id}: {error}") from error
