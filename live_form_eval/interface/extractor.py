"""What a submission implements: an extractor that fills a form while a conversation is going on."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from live_form_eval.interface.session import Session


class Extractor(ABC):
    """One object serves every conversation of a run; several conversations run at the same time.

    Keep per-conversation state inside :meth:`run` (or keyed by ``session.conversation_id``), never on ``self``.
    Shared, reusable things (an HTTP client, a loaded model) belong on ``self``.
    """

    async def prepare(self, session: Session) -> None:
        """Before the conversation starts; not timed. Optional: warm up, build prompts from ``session.form``."""

    @abstractmethod
    async def run(self, session: Session) -> None:
        """While the conversation is going on; timed.

        Read ``session.partials()``, publish values with ``session.update`` / ``session.update_snapshot`` and
        report every model call with ``session.report_usage``. Return once your last value is published: at most
        ``final_grace_s`` after the last partial, or the run is stopped and counted as a failure.
        """

    async def close(self, session: Session) -> None:
        """After :meth:`run` returned (or was stopped); not timed. Optional: free per-conversation resources."""
