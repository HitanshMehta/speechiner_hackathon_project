"""Run an extractor on a dataset: every conversation is replayed as a live stream, and the run folder records
what the extractor published and what its calls cost."""

from __future__ import annotations

import asyncio
import datetime as dt
import importlib
import re
import sys
import time
import traceback
from collections.abc import Coroutine, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from live_form_eval.config import Config, load_config
from live_form_eval.data.common.model import Conversation
from live_form_eval.data.common.store import Dataset, load_dataset
from live_form_eval.harness.clock import Clock
from live_form_eval.harness.recorder import Recorder
from live_form_eval.harness.stream import build_stream
from live_form_eval.interface.extractor import Extractor
from live_form_eval.interface.schema import Form
from live_form_eval.interface.session import Session


def load_extractor(spec: str) -> Extractor:
    """An extractor from "module:Class" (the module is looked up in the current folder too)."""
    if ":" not in spec:
        raise ValueError(f"expected module:Class, got {spec!r}")
    module_name, class_name = spec.split(":", 1)
    if "" not in sys.path and "." not in sys.path:
        sys.path.insert(0, "")
    extractor = getattr(importlib.import_module(module_name), class_name)()
    if not isinstance(extractor, Extractor):
        raise TypeError(f"{spec} must subclass live_form_eval.Extractor")
    return extractor


async def run_conversation(extractor: Extractor, conversation: Conversation, form: Form, config: Config,
                           recorder: Recorder, speed: float, clock: Clock | None = None) -> dict[str, Any]:
    """Replay one conversation for the extractor; returns how it went. ``clock`` is the conversation's clock,
    for a caller that wants to follow it (default: a new one at ``speed``); it must not have started."""
    stream = build_stream(conversation.words, config.stream, key=conversation.id)
    clock = clock or Clock(speed)
    session = Session(conversation.id, form, stream, clock, recorder)
    status: dict[str, Any] = {"conversation": conversation.id, "status": "ok", "error": None,
                              "audio_end": conversation.end, "stream_end": session.stream_end}
    try:
        await extractor.prepare(session)
    except Exception:  # noqa: BLE001 - a submission may fail in any way; the run goes on
        status.update(status="error", error="prepare(): " + traceback.format_exc(limit=3))
        return status
    clock.start()
    deadline = session.stream_end + config.stream.final_grace_s
    try:
        await asyncio.wait_for(extractor.run(session), timeout=max(0.0, deadline - clock.now()) / clock.speed)
    except TimeoutError:
        status.update(status="timeout", error=f"run() had not returned {config.stream.final_grace_s} s after "
                                              "the last partial")
    except asyncio.CancelledError:
        # Stopped from outside: the page's "Stop run", Ctrl+C on `lfe run`. close() is owed all the same, so that
        # an extractor can end what it opened for this conversation (a session, a connection); then the stop goes on.
        failure = await _close(extractor, session)
        if failure:
            print(f"{conversation.id}: stopped, and{failure}", file=sys.stderr, flush=True)
        raise
    except Exception:  # noqa: BLE001
        status.update(status="error", error="run(): " + traceback.format_exc(limit=3))
    status["run_end"] = round(clock.now(), 3)
    failure = await _close(extractor, session)
    if failure:
        status["error"] = (status["error"] or "") + failure
    status.update(calls=session.calls, other_uses=session.other_uses, late_updates=session.late_updates)
    return status


async def _close(extractor: Extractor, session: Session) -> str:
    """The extractor's close() for a conversation whose run() was started, however that run ended: it returned,
    failed, ran out of time or was stopped. Returns what close() raised, as text ("" when it went well)."""
    session.closed = True
    try:
        await extractor.close(session)
    except Exception:  # noqa: BLE001 - a submission may fail in any way
        return " close(): " + traceback.format_exc(limit=3)
    return ""


def describe_run(name: str, spec: str, data: Dataset, config: Config, *, speed: float, concurrency: int,
                 official: bool, n: int | None = None, seed: int | None = None) -> dict[str, Any]:
    """What run.json says about a run from its start: its id (start time, name and dataset), what it runs on,
    and the settings it was made with."""
    run_id = f"{dt.datetime.now():%Y%m%d-%H%M%S}-{re.sub(r'[^A-Za-z0-9_]+', '-', name).strip('-')}-{data.name}"
    return {"run_id": run_id, "name": name, "extractor": spec, "dataset": data.name, "split": data.split,
            "speed": speed, "concurrency": concurrency, "official": official,
            "sample": {"n": n, "seed": seed, "ids": sorted(data.conversations)},
            "config_fingerprint": config.fingerprint(), "config": config.scoring_settings(),
            "started_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}


def finished_run(info: dict[str, Any], statuses: list[dict[str, Any]], started: float) -> dict[str, Any]:
    """run.json once the run is over: how each conversation went, and how long the run took (``started`` is
    ``time.time()`` at its start)."""
    return {**info, "conversations": statuses, "wall_seconds": round(time.time() - started, 1),
            "finished_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}


async def run_async(extractor: Extractor, dataset: Dataset, config: Config, run_dir: Path, *, speed: float,
                    concurrency: int, info: dict[str, Any]) -> Path:
    recorder = Recorder(run_dir)
    gate = asyncio.Semaphore(concurrency)
    started = time.time()
    done = 0

    async def one(conversation: Conversation) -> dict[str, Any]:
        nonlocal done
        async with gate:
            status = await run_conversation(extractor, conversation, dataset.form_of(conversation.id), config,
                                            recorder, speed)
        recorder.flush()
        done += 1
        print(f"  [{done}/{len(dataset.conversations)}] {conversation.id}: {status['status']}"
              f"{' — ' + status['error'].strip().splitlines()[-1] if status['error'] else ''}", flush=True)
        return status

    statuses = await asyncio.gather(*(one(c) for c in dataset.conversations.values()))
    recorder.finish(finished_run(info, list(statuses), started))
    return run_dir


def wait_for[T](coroutine: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to its end from ordinary code, also inside a notebook (where a loop is already running)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


def run(extractor: Extractor | str, dataset: str, *, config: Config | None = None, split: str | None = None,
        n: int | None = None, seed: int | None = None, ids: Iterable[str] | None = None, speed: float = 1.0,
        concurrency: int = 4, name: str | None = None) -> Path:
    """Run ``extractor`` on ``dataset`` ("spokenwoz" or "ami"); returns the run folder.

    Without ``split``, ``n``, ``seed`` and ``ids``, and at ``speed`` 1, the run is official: the config's sample
    of that dataset, replayed in real time. Anything else is a development run and is marked unofficial.
    """
    config = config or load_config()
    spec = extractor if isinstance(extractor, str) else f"{type(extractor).__module__}:{type(extractor).__name__}"
    if isinstance(extractor, str):
        extractor = load_extractor(extractor)
    official = split is None and n is None and seed is None and ids is None and speed == 1.0
    if official:
        data = load_dataset(config, dataset, official_sample=True)
    else:
        data = load_dataset(config, dataset, split=split, ids=ids, n=n, seed=seed)
    info = describe_run(name or type(extractor).__name__, spec, data, config, speed=speed, concurrency=concurrency,
                        official=official, n=n, seed=seed)
    print(f"{info['run_id']}: {len(data.conversations)} conversations of {dataset}/{data.split}, speed ×{speed:g}, "
          f"{concurrency} at a time{'' if official else '  (unofficial)'}", flush=True)
    return wait_for(run_async(extractor, data, config, config.paths.runs / info["run_id"], speed=speed,
                              concurrency=concurrency, info=info))
