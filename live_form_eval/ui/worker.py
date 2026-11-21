"""The process of one live run: an extractor on one conversation, started by the page's server (ui/live.py).

    python -m live_form_eval.ui.worker --dataset spokenwoz --split test --id MUL0229 --speed 2 \
        --extractor examples.oracle_extractor:Oracle

It replays the conversation with the harness's own ``run_conversation`` and writes a run folder like `lfe run`
does (unofficial: one conversation), so the run can be reviewed and evaluated afterwards. While it runs, it writes
one line of JSON per event to its standard output:

    {"event": "preparing", "run": "<run id>"}                    the extractor is loaded, its prepare() is running
    {"event": "started", "speed": 2.0}                           the conversation's clock starts
    {"event": "update", "t": 15.53, "field": "…", "value": …}    a value was published
    {"event": "call", "t": 15.40, "model": "…", "eur": …, …}     a model call was reported
    {"event": "tick", "t": 16.0}                                 the conversation's clock, every now and then
    {"event": "ended", "status": {…}, "run": "<run id>"}         how the conversation went; the run folder is written
    {"event": "failed", "error": "…"}                            the run could not start

Whatever the extractor prints goes to the standard error, so that it can't break those lines. The process stops
its extractor when its standard input closes: that is how the server stops a run, and it is also what happens
when the server itself goes away, so an extractor never keeps calling a model with nobody watching.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from live_form_eval.config import load_config
from live_form_eval.data.common.store import load_dataset
from live_form_eval.harness.clock import Clock
from live_form_eval.harness.recorder import Recorder
from live_form_eval.harness.runner import describe_run, finished_run, load_extractor, run_conversation
from live_form_eval.interface.extractor import Extractor


def load(spec: str) -> Extractor:
    """An extractor from "module:Class", or from "path/to/file.py:Class" for a file whose path is no module name."""
    target, _, class_name = spec.rpartition(":")
    if not target.endswith(".py"):
        return load_extractor(spec)
    path = Path(target).resolve()
    sys.path.insert(0, str(path.parent))  # what the file imports from its own folder
    module_spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    extractor = getattr(module, class_name)()
    if not isinstance(extractor, Extractor):
        raise TypeError(f"{spec} must subclass live_form_eval.Extractor")
    return extractor


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m live_form_eval.ui.worker")
    parser.add_argument("--config")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--extractor", required=True, help="module:Class or path/to/file.py:Class")
    args = parser.parse_args(argv)

    events = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")  # the lines of JSON, one per event
    os.dup2(2, 1)  # from here on, anything written to the standard output is the extractor's: it goes to the log
    sys.stdout = sys.stderr

    def emit(event: dict[str, Any]) -> None:
        events.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    class Feed(Recorder):
        """The run folder's recorder, which also tells the page."""

        def prediction(self, conversation: str, t: float, field_id: str, value: Any) -> None:
            super().prediction(conversation, t, field_id, value)
            emit({"event": "update", "t": round(t, 4), "field": field_id, "value": value})

        def call(self, conversation: str, t: float, record: dict[str, Any]) -> None:
            super().call(conversation, t, record)
            emit({"event": "call", "t": round(t, 4), **record})

    class Announced(Clock):
        def start(self) -> None:
            super().start()
            emit({"event": "started", "speed": self.speed})

    try:
        config = load_config(args.config)
        if "" not in sys.path:
            sys.path.insert(0, "")  # the folder the page was started in, as for `lfe run`
        extractor = load(args.extractor)
        data = load_dataset(config, args.dataset, split=args.split, ids=[args.id])
        conversation, form = data.conversations[args.id], data.form_of(args.id)
        info = describe_run(type(extractor).__name__, args.extractor, data, config, speed=args.speed, concurrency=1,
                            official=False)
        while (config.paths.runs / info["run_id"]).exists():  # a run of the same name started in this very second
            time.sleep(0.2)
            info = describe_run(type(extractor).__name__, args.extractor, data, config, speed=args.speed,
                                concurrency=1, official=False)
        info["started_from"] = "lfe ui"
        recorder = Feed(config.paths.runs / info["run_id"])
    except Exception:  # noqa: BLE001 - whatever keeps the run from starting is told to the page
        emit({"event": "failed", "error": traceback.format_exc(limit=6)})
        sys.exit(1)

    clock = Announced(args.speed)
    started = time.time()
    emit({"event": "preparing", "run": info["run_id"]})

    async def play() -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        run = asyncio.ensure_future(run_conversation(extractor, conversation, form, config, recorder, args.speed,
                                                     clock))

        def stop_when_told() -> None:
            sys.stdin.read()  # returns when the server closes it, or is gone
            loop.call_soon_threadsafe(run.cancel)

        threading.Thread(target=stop_when_told, daemon=True).start()

        async def tick() -> None:
            while True:
                await asyncio.sleep(config.ui.score_every_s)
                if clock.started:
                    emit({"event": "tick", "t": round(clock.now(), 3)})

        ticking = asyncio.ensure_future(tick())
        try:
            return await run
        except asyncio.CancelledError:
            return {"conversation": conversation.id, "status": "stopped", "error": "stopped before its end",
                    "audio_end": conversation.end, "run_end": round(clock.now(), 3) if clock.started else 0.0}
        finally:
            ticking.cancel()

    status = asyncio.run(play())
    if clock.started:
        emit({"event": "tick", "t": status.get("run_end") or round(clock.now(), 3)})
    recorder.finish(finished_run(info, [status], started))
    emit({"event": "ended", "status": status, "run": info["run_id"]})


if __name__ == "__main__":
    main()
