"""The run folder: what was published when, what each call cost, and how each conversation went.

runs/<run id>/
  run.json            the run: extractor, dataset, conversations and their status, speed, config hash, official?
  predictions.jsonl   {"conversation", "t", "field_id", "value"}: one line per published value, in order
  calls.jsonl         one line per reported use (session.report_usage): {"conversation", "t", "model", "extraction",
                      "usage": {unit: amount}, "prices": {unit: € per unit}, "eur", "latency_s", "ok", …}. Runs made
                      before units had prices hold {"model", "eur", "input_tokens", "output_tokens", …} instead
                      (metrics/calls.py reads both)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Recorder:
    """Appends to the run folder's files as values and calls come in (one process, one event loop)."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._predictions = (self.run_dir / "predictions.jsonl").open("w", encoding="utf-8")
        self._calls = (self.run_dir / "calls.jsonl").open("w", encoding="utf-8")

    def prediction(self, conversation: str, t: float, field_id: str, value: Any) -> None:
        line = {"conversation": conversation, "t": round(t, 4), "field_id": field_id, "value": value}
        self._predictions.write(json.dumps(line, ensure_ascii=False) + "\n")

    def call(self, conversation: str, t: float, record: dict[str, Any]) -> None:
        line = {"conversation": conversation, "t": round(t, 4), **record}
        self._calls.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")

    def flush(self) -> None:
        self._predictions.flush()
        self._calls.flush()

    def finish(self, run: dict[str, Any]) -> Path:
        """Close the line files and write run.json."""
        self._predictions.close()
        self._calls.close()
        path = self.run_dir / "run.json"
        path.write_text(json.dumps(run, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        return path


def read_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """(run.json, predictions, calls) of a run folder."""
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    predictions = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    calls = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    return run, predictions, calls
