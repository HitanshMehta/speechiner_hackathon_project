"""A starting point for your extractor: it follows the live transcript and asks your gating rule when to extract.

It calls no model, so it runs without any API key: `extract` is a stand-in that fills nothing. Replace it with
your model call (examples/llm_extractor.py has a complete one), and put your gating strategy in `should_extract`.

    uv run lfe check my_extractor:MyExtractor
    uv run lfe run my_extractor:MyExtractor --dataset spokenwoz --n 2 --speed 5
    uv run lfe evaluate runs/<run id>
"""

from __future__ import annotations

from typing import Any

from live_form_eval import Extractor, Session


class MyExtractor(Extractor):
    async def run(self, session: Session) -> None:
        turns: list[str] = []
        partial = ""
        async for p in session.partials():  # arrives in real time, as from a speech-to-text service
            if p.kind == "turn":  # a finished turn
                turns.append(p.labelled_text())
                partial = ""
            else:  # the turn so far: replaces the previous partial
                partial = p.labelled_text()
            if self.should_extract(turns, partial, finished_turn=p.kind == "turn"):
                values = await self.extract(session, turns, partial)
                if values:
                    session.update_snapshot(values)  # shown in the form from this moment

    def should_extract(self, turns: list[str], partial: str, *, finished_turn: bool) -> bool:
        """The gating strategy: is now a good moment to call the model? This one extracts after every finished
        turn."""
        return finished_turn

    async def extract(self, session: Session, turns: list[str], partial: str) -> dict[str, Any]:
        """Fill the form from the transcript so far. A stand-in: call your model here, report what it cost with
        session.report_usage(...), and return {field_id: value}."""
        return {}
