"""Which conversations a run uses: a reproducible random sample."""

from __future__ import annotations

import random
from collections.abc import Iterable


def sample_ids(ids: Iterable[str], n: int | None, seed: int) -> list[str]:
    """``n`` ids drawn with ``seed`` (the same n and seed always give the same ids); every id when ``n`` is None."""
    ordered = sorted(ids)
    if n is None or n >= len(ordered):
        return ordered
    return sorted(random.Random(seed).sample(ordered, n))
