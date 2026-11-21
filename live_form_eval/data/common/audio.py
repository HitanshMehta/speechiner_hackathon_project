"""Recordings: the sound of one conversation at a time, for the page of `lfe ui`.

The processed data holds no audio: runs and evaluations only need the words and their times. The page can play a
conversation with its sound, so a recording is fetched when somebody first asks for it and kept under
``paths.audio`` as ``<dataset>/<conversation id>.wav``: 16-bit PCM, one channel (a recording with one speaker per
channel is mixed down, so that both are heard in both ears), at the dataset's sample rate. Its time zero is the
conversation's: second 12.3 of the file is second 12.3 of the word times.

Each dataset says where its recordings are (data/<dataset>/audio.py). A file put at that path by hand is used as
it is, which is the way to listen to a conversation whose recording can't be fetched one by one.
"""

from __future__ import annotations

import io
import shutil
import sys
import urllib.error
import urllib.request
import wave
from array import array
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from live_form_eval.config import Config
from live_form_eval.data.common.download import AGENT

Progress = Callable[[int, int], None]  # (bytes so far, bytes in all; 0 when the size is unknown)
_CHUNK = 1 << 16


class AudioUnavailable(RuntimeError):
    """A recording can't be fetched; the message says why and what to do about it."""


def recording_path(config: Config, dataset: str, conversation_id: str) -> Path:
    """Where the recording of a conversation is kept once it is fetched."""
    return config.paths.audio / dataset / f"{conversation_id}.wav"


class RemoteFile:
    """A file on a server, read through byte ranges, with the interface ``zipfile`` needs of a file.

    Short reads are served from blocks of 1 MiB that are kept in memory (the most recently used ones), so that
    reading an archive's directory or one of its members takes a handful of requests instead of the download of
    the whole archive.
    """

    BLOCK = 1 << 20
    KEPT = 16

    def __init__(self, url: str) -> None:
        self.url = url
        self._position = 0
        self._blocks: OrderedDict[int, bytes] = OrderedDict()
        try:
            with self._open(0, 0) as response:
                total = response.headers.get("Content-Range", "").rpartition("/")[2]
        except urllib.error.URLError as error:
            raise AudioUnavailable(f"could not reach {url}: {error}") from error
        if not total.isdigit():
            raise AudioUnavailable(f"{url} can't be read in parts (its server gives no byte ranges)")
        self.size = int(total)

    def _open(self, first: int, last: int):
        request = urllib.request.Request(self.url, headers={**AGENT, "Range": f"bytes={first}-{last}"})
        return urllib.request.urlopen(request)  # the URL is one of the datasets' own, set in code

    def _fetch(self, start: int, stop: int) -> bytes:
        with self._open(start, stop - 1) as response:
            return response.read()

    def _block(self, index: int) -> bytes:
        if index in self._blocks:
            self._blocks.move_to_end(index)
        else:
            self._blocks[index] = self._fetch(index * self.BLOCK, min(self.size, (index + 1) * self.BLOCK))
            if len(self._blocks) > self.KEPT:
                self._blocks.popitem(last=False)
        return self._blocks[index]

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self.size}[whence]
        self._position = base + offset
        return self._position

    def read(self, size: int = -1) -> bytes:
        stop = self.size if size is None or size < 0 else min(self.size, self._position + size)
        if stop <= self._position:
            return b""
        first, last = self._position // self.BLOCK, (stop - 1) // self.BLOCK
        data = b"".join(self._block(index) for index in range(first, last + 1))
        offset = self._position - first * self.BLOCK
        self._position = stop
        return data[offset: offset + stop - (first * self.BLOCK + offset)]


def copy_with_progress(source: BinaryIO, target: BinaryIO, total: int, progress: Progress | None) -> None:
    """Copy ``source`` to ``target`` chunk by chunk, telling ``progress`` how far it is."""
    done = 0
    while chunk := source.read(_CHUNK):
        target.write(chunk)
        done += len(chunk)
        if progress is not None:
            progress(done, total)


def download(url: str, target: Path, progress: Progress | None = None) -> None:
    """Download ``url`` to ``target`` (through a ``.part`` file, so that a broken download leaves nothing)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT)) as response, \
                partial.open("wb") as out:  # the URL is one of the datasets' own, set in code
            copy_with_progress(response, out, int(response.headers.get("Content-Length") or 0), progress)
    except (urllib.error.URLError, OSError) as error:
        partial.unlink(missing_ok=True)
        raise AudioUnavailable(f"could not download {url}: {error}") from error
    partial.replace(target)


def to_mono(source: Path, target: Path) -> None:
    """Write the WAV file ``source`` to ``target`` with one channel: the mean of its channels, sample by sample.

    A file that already has one channel, or whose samples aren't 16 bits wide, is copied as it is.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with wave.open(str(source), "rb") as recording:
        channels, width, rate = recording.getnchannels(), recording.getsampwidth(), recording.getframerate()
        frames = recording.readframes(recording.getnframes()) if channels > 1 and width == 2 else None
    if frames is None:
        shutil.copyfile(source, partial)
    else:
        samples = array("h")
        samples.frombytes(frames)
        if sys.byteorder == "big":  # WAV samples are little-endian
            samples.byteswap()
        mixed = array("h", (sum(frame) // channels for frame in zip(*(samples[k::channels] for k in range(channels)))))
        if sys.byteorder == "big":
            mixed.byteswap()
        with wave.open(str(partial), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)
            out.writeframes(mixed.tobytes())
    partial.replace(target)


def fetch_recording(config: Config, dataset: str, split: str, conversation_id: str,
                    progress: Progress | None = None) -> Path:
    """The recording of one conversation as a mono WAV file, fetched now if it isn't there yet.

    Raises ``AudioUnavailable`` when it can't be fetched (no network, or a dataset split whose recordings only
    come as one large archive).
    """
    target = recording_path(config, dataset, conversation_id)
    if target.exists() and target.stat().st_size > 0:
        return target
    if dataset == "spokenwoz":
        from live_form_eval.data.spokenwoz import audio as source
    elif dataset == "ami":
        from live_form_eval.data.ami import audio as source
    else:
        raise AudioUnavailable(f"no recordings are known for the dataset {dataset!r}")
    original = target.with_name(target.stem + ".original.wav")
    source.fetch(split, conversation_id, original, target, progress)
    try:
        to_mono(original, target)
    except (wave.Error, EOFError) as error:
        raise AudioUnavailable(f"the recording of {conversation_id} is not a WAV file that can be read: "
                               f"{error}") from error
    finally:
        original.unlink(missing_ok=True)
    return target
