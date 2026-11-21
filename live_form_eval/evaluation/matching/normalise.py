"""Bring a value into a comparable shape: what counts as "the same" before two values are compared."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_SPACES = re.compile(r"\s+")
_NOT_ALNUM = re.compile(r"[^a-z0-9]+")
_TIME = re.compile(r"^\s*(\d{1,2})\s*[:.h]\s*(\d{2})\s*$")


def text(value: Any) -> str:
    """Lower case, accents kept, punctuation at the edges and repeated spaces removed."""
    out = unicodedata.normalize("NFKC", str(value)).lower()
    out = _SPACES.sub(" ", out).strip()
    return out.strip(" .,;:!?\"'")


def squash(value: Any) -> str:
    """Letters and digits only, lower case: "FI06 UBJ" → "fi06ubj", "017 655" → "017655"."""
    return _NOT_ALNUM.sub("", unicodedata.normalize("NFKC", str(value)).lower())


def digits(value: Any) -> str:
    """The digits of a phone number, without spaces, dashes, brackets or a leading +."""
    return re.sub(r"\D+", "", str(value))


def email(value: Any) -> str:
    return re.sub(r"\s+", "", str(value)).lower()


def url(value: Any) -> str:
    out = re.sub(r"\s+", "", str(value)).lower()
    out = re.sub(r"^https?://", "", out)
    out = re.sub(r"^www\.", "", out)
    return out.rstrip("/")


def time(value: Any) -> str:
    """"9:30", "09.30" → "09:30"; anything else is returned stripped."""
    match = _TIME.match(str(value))
    return f"{int(match.group(1)):02d}:{match.group(2)}" if match else str(value).strip()


def is_empty(value: Any) -> bool:
    """A field shows nothing: None, an empty or blank string, an empty list or dict."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False
