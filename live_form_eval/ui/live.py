"""A live run from the page: one extractor on one conversation, in a process of its own (ui/worker.py).

The process sends what happens as lines of JSON: the conversation's clock starting, every value the extractor
publishes, every model call it reports, a tick of the clock, the end. This module passes them on to the page
through a feed, scores the conversation as far as it has gone at every tick (ui/scoring.py), and stops the
process when the page asks for it or goes away. Whatever the extractor prints goes to the page as its log.

A process of its own means: the extractor's file is read afresh at every start (edit it, start again), a
blocking or crashing extractor can't take the page's server down, and stopping is always possible.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from live_form_eval.config import Config
from live_form_eval.data.common.model import Conversation, Gold
from live_form_eval.evaluation.evaluate import LazyMatch
from live_form_eval.evaluation.metrics.calls import conversation_calls
from live_form_eval.evaluation.timeline.shown import shown_fields
from live_form_eval.interface.schema import Form
from live_form_eval.ui.playback import calls_payload, final_payload, terms_payload, walks_payload
from live_form_eval.ui.scoring import Scorer

_LAST_SCORE_WAIT_S = 4.0  # how long the end of a run waits for the scoring that is under way
_LOG_LINES = 2000  # the page gets this many lines of an extractor's output; a run that prints more is cut there
_LOG_WIDTH = 2000


class Feed:
    """What a page follows: events in order, kept, so that a page can read them from any point on."""

    def __init__(self) -> None:
        self.id = uuid.uuid4().hex
        self._events: list[dict[str, Any]] = []
        self._changed = threading.Condition()
        self.closed = False
        self.watchers = 0
        self.unwatched_since = time.monotonic()

    def add(self, event: dict[str, Any]) -> None:
        with self._changed:
            if not self.closed:
                self._events.append(event)
                self._changed.notify_all()

    def close(self) -> None:
        with self._changed:
            self.closed = True
            self._changed.notify_all()

    def wait(self, start: int, timeout: float) -> tuple[list[dict[str, Any]], bool]:
        """The events from ``start`` on, waiting up to ``timeout`` seconds for one; and whether the feed is over."""
        with self._changed:
            if len(self._events) <= start and not self.closed:
                self._changed.wait(timeout)
            return self._events[start:], self.closed

    def watch(self) -> None:
        with self._changed:
            self.watchers += 1

    def unwatch(self) -> None:
        with self._changed:
            self.watchers -= 1
            self.unwatched_since = time.monotonic()


class LiveRun:
    """One extractor running on one conversation for the page."""

    def __init__(self, config: Config, conversation: Conversation, form: Form, gold: Gold, extractor: str,
                 speed: float, root: Path, match: LazyMatch, config_path: Path | None = None) -> None:
        self.config, self.conversation, self.form = config, conversation, form
        self.extractor, self.speed = extractor, speed
        self.feed = Feed()
        self.scorer = Scorer(config, conversation, form, gold, config.stream, match)
        self._lock = threading.Lock()
        self._predictions: list[dict[str, Any]] = []
        self._calls: list[dict[str, Any]] = []
        self._now: float | None = None  # the conversation's clock at its last tick
        self._ticked = threading.Event()
        self._status: dict[str, Any] | None = None
        self._stopping: str | None = None  # why the run is being stopped
        self._over = False
        command = [sys.executable, "-m", "live_form_eval.ui.worker", "--dataset", conversation.dataset,
                   "--split", conversation.split, "--id", conversation.id, "--speed", str(speed),
                   "--extractor", extractor]
        if config_path is not None:
            command += ["--config", str(config_path)]
        self.process = subprocess.Popen(  # the extractor is one of the classes found in the folder
            command, cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", bufsize=1)
        self._scoring = threading.Thread(target=self._score_as_it_goes, daemon=True)
        for thread in (self._scoring, threading.Thread(target=self._read_events, daemon=True),
                       threading.Thread(target=self._read_log, daemon=True)):
            thread.start()

    @property
    def over(self) -> bool:
        return self._over

    def about(self) -> dict[str, Any]:
        return {"feed": self.feed.id, "dataset": self.conversation.dataset, "split": self.conversation.split,
                "id": self.conversation.id, "extractor": self.extractor, "speed": self.speed, "over": self._over}

    def stop(self, reason: str = "stopped from the page") -> None:
        """Ask the process to stop its extractor and write its run folder; end it if it doesn't.

        The process stops the extractor's run() and then lets its close() end what it opened for the conversation
        (harness/runner.py). It gets ``ui.stop_wait_s`` for that: ended sooner, an extractor that holds a billed
        session open would leave it open.
        """
        if self._stopping is not None or self.process.poll() is not None:
            return
        self._stopping = reason
        try:
            self.process.stdin.close()  # the process reads this as "stop"
        except OSError:
            pass
        threading.Thread(target=self._make_sure_it_ends, daemon=True).start()

    def _make_sure_it_ends(self) -> None:
        wait = self.config.ui.stop_wait_s
        try:
            self.process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def _read_events(self) -> None:
        """What the process sends, passed on to the page; at its end, the final score of the conversation."""
        conversation = self.conversation.id
        for line in self.process.stdout:
            try:
                event = json.loads(line)
                kind = event["event"]
            except (ValueError, KeyError, TypeError):
                self.feed.add({"event": "log", "line": line.rstrip("\n")[:_LOG_WIDTH]})
                continue
            if kind == "update":
                with self._lock:
                    self._predictions.append({"conversation": conversation, "t": event["t"],
                                              "field_id": event["field"], "value": event["value"]})
            elif kind == "call":  # the page gets it as a review gets it: tokens by kind, price by unit
                record = {"conversation": conversation, **{key: value for key, value in event.items() if key != "event"}}
                with self._lock:
                    self._calls.append(record)
                event = {"event": "call", **calls_payload(conversation_calls([record], []))[0]}
            elif kind == "tick":
                self._now = event["t"]
                self._ticked.set()
            elif kind == "ended":
                self._status = event["status"]
                if self._stopping is not None:
                    event["reason"] = self._stopping
            self.feed.add(event)
        self.process.wait()
        if self._status is None:  # the process ended without saying how: it failed before the run, or was ended
            self._status = {"status": "stopped" if self._stopping else "error", "run_end": self._now,
                            "error": self._stopping or "the extractor's process ended unexpectedly (see its output)"}
            self.feed.add({"event": "ended", "status": self._status, "run": None, "reason": self._stopping})
        self._over = True
        self._ticked.set()
        self._scoring.join(timeout=_LAST_SCORE_WAIT_S)  # so that the last score is the last thing the page gets
        if self._now is not None:  # the conversation had started: its last score
            try:
                self._score(final=True)
            except Exception as error:  # noqa: BLE001 - a failed scoring must still end the feed
                self.feed.add({"event": "problem", "message": f"the run could not be scored: {error}"})
        self.feed.add({"event": "done"})
        self.feed.close()

    def _read_log(self) -> None:
        """Whatever the extractor prints, line by line."""
        for count, line in enumerate(self.process.stderr):
            if count < _LOG_LINES:
                self.feed.add({"event": "log", "line": line.rstrip("\n")[:_LOG_WIDTH]})
            elif count == _LOG_LINES:
                self.feed.add({"event": "log", "line": f"… more output is not shown (after {_LOG_LINES} lines)"})

    def _score(self, final: bool = False, after: float | None = None) -> None:
        """Send the score so far to the page; with ``final``, the score of the run as it ended."""
        with self._lock:
            predictions, calls = list(self._predictions), list(self._calls)
        shown = shown_fields(predictions, self.conversation.id)
        status = self._status or {}
        ended_by_itself = final and status.get("status") != "stopped"
        now = None if ended_by_itself else self._now  # a run that ended by itself is scored as the evaluation does
        points = []
        if after is not None and now is not None:  # the moments since the last tick at which the score jumped
            points = [terms_payload(self.scorer.score(shown, calls, status, until=t), t)
                      for t in self.scorer.moments(shown, after, now, float("inf"))]
        score = self.scorer.score(shown, calls, status, until=now)
        event = {"event": "score", "points": points + [terms_payload(score, score.walk_end if now is None else now)],
                 "fields": walks_payload(self.form, score, self.scorer.text_match(), only_latest=not final)}
        if ended_by_itself:
            event["final"] = final_payload(score)
        self.feed.add(event)

    def _score_as_it_goes(self) -> None:
        """At every tick of the conversation's clock: the score so far. Also stops a run nobody watches."""
        last = 0.0
        keeping_up = True
        failed = False
        while not self._over:
            self._ticked.wait(timeout=1.0)
            self._ticked.clear()
            if self._over:
                break
            unwatched = time.monotonic() - self.feed.unwatched_since
            if self.feed.watchers == 0 and unwatched > self.config.ui.unwatched_stop_s:
                self.stop("its page was closed")
            now = self._now
            if now is None or now <= last or failed:
                continue
            started = time.monotonic()
            try:
                self._score(after=last if keeping_up else None)
            except Exception as error:  # noqa: BLE001 - the run goes on without its score so far
                failed = True
                self.feed.add({"event": "problem", "message": f"the score so far could not be computed: {error}"})
            # a scoring that takes longer than a tick (a judge reading long notes) leaves out the moments in between
            keeping_up = time.monotonic() - started <= self.config.ui.score_every_s
            last = now
