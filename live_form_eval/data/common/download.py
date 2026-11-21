"""Fetch files and archives into the raw data folder (outside the repo). Nothing is fetched twice."""

from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

AGENT = {"User-Agent": "live-form-eval"}


def fetch(url: str, target: Path) -> bool:
    """Download ``url`` to ``target`` unless it is already there. True when it was downloaded."""
    if target.exists() and target.stat().st_size > 0:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT)) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)
    partial.replace(target)
    return True


def fetch_zip(url: str, folder: Path, *, strip_top: bool, keep: tuple[str, ...] = ()) -> bool:
    """Download a zip and extract it into ``folder`` unless ``folder`` already has files.

    ``strip_top`` drops the archive's single top folder (GitHub archives have one). ``keep`` limits the
    extraction to paths starting with one of these prefixes (after stripping). Returns True when it downloaded.
    """
    if folder.exists() and any(folder.iterdir()):
        return False
    folder.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT)) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    root = folder.resolve()
    for member in archive.infolist():
        name = member.filename.split("/", 1)[1] if strip_top and "/" in member.filename else member.filename
        if not name or member.is_dir() or (keep and not name.startswith(keep)):
            continue
        target = (folder / name).resolve()
        if root not in target.parents:  # a path that would leave the folder
            raise ValueError(f"unsafe path in archive: {member.filename}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, target.open("wb") as out:
            shutil.copyfileobj(source, out)
    return True
