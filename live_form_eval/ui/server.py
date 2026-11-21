"""`lfe ui`: the server behind the page (docs/ui.md).

Plain HTTP from the standard library, one thread per request:

    GET  /                      the page (static/)
    GET  /api/state             what can be chosen: forms, extractors, runs; asked again every few seconds
    GET  /api/form              one conversation: its stream, its form, what each field should show from when
    GET  /api/review            one conversation of one run: what the form showed, the calls, how it was scored
    POST /api/live              start an extractor on one conversation      POST /api/live/stop   stop it
    GET  /api/feed              what happens, as server-sent events: a live run, or the score over a reviewed run
    GET  /api/audio             a conversation's recording (byte ranges, so that a player can jump in it)
    GET  /api/audio/state       is the recording there?                     POST /api/audio/fetch  fetch it

The page is only for the person at this machine: the server listens on ``ui.host`` (127.0.0.1), answers only
requests addressed to that host, and starts or stops a run only for a request the page itself made (a header
that another site's page can't send).
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
import traceback
import webbrowser
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from live_form_eval.config import Config, StreamConfig
from live_form_eval.data.common.audio import AudioUnavailable, fetch_recording, recording_path
from live_form_eval.evaluation.evaluate import LazyMatch
from live_form_eval.evaluation.report.style import CHROME, DATASET_NAMES, SERIES, STATUS
from live_form_eval.evaluation.timeline.arrival import stream_config_of
from live_form_eval.evaluation.timeline.shown import shown_fields
from live_form_eval.harness.recorder import read_run
from live_form_eval.ui.catalog import ExtractorFinder, Library, entries, find_runs
from live_form_eval.ui.live import Feed, LiveRun
from live_form_eval.ui.playback import (
    calls_payload,
    conversation_payload,
    final_payload,
    updates_payload,
    walks_payload,
)
from live_form_eval.ui.scoring import Scorer

STATIC = Path(__file__).parent / "static"
_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
_FROM_THE_PAGE = "X-Lfe-Page"  # the header the page's own requests carry
_KEPT_REVIEWS = 8  # the scores over time of this many reviewed conversations are kept ready
_BATCH_S = 0.2  # points of a score over time are sent to the page this often while they are computed
_CHUNK = 1 << 16


class Refused(Exception):
    """A request that can't be done; the message tells the page's user why."""


class Recordings:
    """Which recordings are there, and the ones being fetched."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._fetching: dict[tuple[str, str], list[int]] = {}  # (dataset, id) → [bytes so far, bytes in all]
        self._failed: dict[tuple[str, str], str] = {}

    def state(self, dataset: str, conversation_id: str) -> dict[str, Any]:
        key = (dataset, conversation_id)
        if recording_path(self.config, dataset, conversation_id).exists():
            return {"state": "ready"}
        with self._lock:
            if key in self._fetching:
                done, total = self._fetching[key]
                return {"state": "fetching", "done": done, "total": total}
            if key in self._failed:
                return {"state": "failed", "message": self._failed[key]}
        return {"state": "missing"}

    def fetch(self, dataset: str, split: str, conversation_id: str) -> dict[str, Any]:
        """Start fetching a recording unless it is there or on its way; returns its state."""
        key = (dataset, conversation_id)
        with self._lock:
            starting = key not in self._fetching and not recording_path(self.config, dataset, conversation_id).exists()
            if starting:
                self._fetching[key] = [0, 0]
                self._failed.pop(key, None)
        if starting:
            threading.Thread(target=self._work, args=(dataset, split, conversation_id), daemon=True).start()
        return self.state(dataset, conversation_id)

    def _work(self, dataset: str, split: str, conversation_id: str) -> None:
        key = (dataset, conversation_id)

        def progress(done: int, total: int) -> None:
            self._fetching[key][:] = [done, total]

        try:
            fetch_recording(self.config, dataset, split, conversation_id, progress)
        except AudioUnavailable as error:
            with self._lock:
                self._failed[key] = str(error)
        except Exception as error:  # noqa: BLE001 - whatever went wrong is told to the page
            with self._lock:
                self._failed[key] = f"the recording could not be fetched: {error}"
        finally:
            with self._lock:
                self._fetching.pop(key, None)


class App:
    """Everything the page's requests share."""

    def __init__(self, config: Config, root: Path, config_path: Path | None = None) -> None:
        self.config, self.root, self.config_path = config, Path(root).resolve(), config_path
        self.library = Library(config)
        self.finder = ExtractorFinder(self.root, config.ui.extractor_folders, config.ui.skip_folders)
        self.match = LazyMatch(config, None)  # the judge of free text, shared: its verdicts are kept
        self.recordings = Recordings(config)
        self.live: LiveRun | None = None
        self._lock = threading.RLock()
        self._feeds: dict[str, Feed] = {}
        self._forms: list[dict[str, Any]] | None = None
        self._conversations: dict[tuple[str, str, str, StreamConfig], dict[str, Any]] = {}
        self._reviews: OrderedDict[tuple[str, str, float], Feed] = OrderedDict()

    def close(self) -> None:
        if self.live is not None:
            self.live.stop("the page's server was stopped")

    def feed(self, feed_id: str) -> Feed:
        with self._lock:
            return self._feeds[feed_id]

    def state(self) -> dict[str, Any]:
        mode, ui = self.config.plots.mode, self.config.ui
        with self._lock:
            if self._forms is None:
                self._forms = self.library.forms()
            live = self.live.about() if self.live is not None and not self.live.over else None
        return {"datasets": self._forms, "problems": self.library.problems,
                "extractors": entries(self.finder.find()), "runs": entries(find_runs(self.config)), "live": live,
                "settings": {"speeds": list(ui.speeds), "audio_max_speed": ui.audio_max_speed,
                             "refresh_s": ui.refresh_s, "mode": mode, "datasets": DATASET_NAMES,
                             "folder": str(self.root)},
                "colors": {"chrome": CHROME[mode], "status": STATUS, "series": SERIES[mode]}}

    def conversation(self, dataset: str, split: str, conversation_id: str,
                     stream_config: StreamConfig | None = None) -> dict[str, Any]:
        """One conversation as the page shows it, with the stream as ``stream_config`` cuts it (default: the
        config's)."""
        stream_config = stream_config or self.config.stream
        key = (dataset, split, conversation_id, stream_config)
        with self._lock:
            if key not in self._conversations:
                conversation, form, gold = self.library.conversation(dataset, split, conversation_id)
                self._conversations[key] = conversation_payload(self.config, conversation, form, gold, stream_config)
            payload = self._conversations[key]
        return {**payload, "audio": self.recordings.state(dataset, conversation_id)}

    def review(self, run_id: str, conversation_id: str) -> dict[str, Any]:
        """One conversation of one run: what the form showed, the calls, the evaluation's reading of it, and
        the feed that brings its score over time."""
        run_dir = self.config.paths.runs / Path(run_id).name  # a run folder's name, never a path
        if not (run_dir / "run.json").is_file():
            raise Refused(f"there is no run {run_id!r} in {self.config.paths.runs}")
        run, predictions, calls = read_run(run_dir)
        status = next((item for item in run["conversations"] if item["conversation"] == conversation_id), None)
        if status is None:
            raise Refused(f"the run {run_id} was not made on {conversation_id}")
        stream_config = stream_config_of(run, self.config)
        conversation, form, gold = self.library.conversation(run["dataset"], run["split"], conversation_id)
        scorer = Scorer(self.config, conversation, form, gold, stream_config, self.match)
        predictions = [line for line in predictions if line["conversation"] == conversation_id]
        calls = [call for call in calls if call["conversation"] == conversation_id]
        shown = shown_fields(predictions, conversation_id)
        score = scorer.score(shown, calls, status)
        latest = max([score.walk_end] + [line["t"] for line in predictions] + [call.t for call in score.model_calls])

        key = (run_id, conversation_id, (run_dir / "predictions.jsonl").stat().st_mtime)
        with self._lock:
            feed = self._reviews.get(key)
            if feed is None:
                feed = self._reviews[key] = Feed()
                self._feeds[feed.id] = feed
                while len(self._reviews) > _KEPT_REVIEWS:
                    _, old = self._reviews.popitem(last=False)
                    self._feeds.pop(old.id, None)
                moments = len(scorer.moments(shown, 0.0, score.walk_end, self.config.ui.score_every_s)) + 1
                threading.Thread(target=self._score_over_time, daemon=True,
                                 args=(feed, scorer, predictions, calls, status, moments)).start()
            else:
                self._reviews.move_to_end(key)
        return {"conversation": self.conversation(run["dataset"], run["split"], conversation_id, stream_config),
                "run": {"id": run_id, "name": run["name"], "extractor": run.get("extractor"),
                        "speed": run.get("speed"), "started_at": run.get("started_at")},
                "status": status, "updates": updates_payload(shown), "calls": calls_payload(score.model_calls),
                "end": latest, "scored_until": score.walk_end, "final": final_payload(score),
                "fields": walks_payload(form, score, scorer.text_match()), "feed": feed.id}

    def _score_over_time(self, feed: Feed, scorer: Scorer, predictions: list[dict[str, Any]],
                         calls: list[dict[str, Any]], status: dict[str, Any], moments: int) -> None:
        feed.add({"event": "plan", "moments": moments})
        batch: list[dict[str, Any]] = []
        sent = time.monotonic()
        try:
            for point in scorer.series(predictions, calls, status, self.config.ui.score_every_s):
                batch.append(point)
                if time.monotonic() - sent > _BATCH_S:
                    feed.add({"event": "points", "points": batch})
                    batch, sent = [], time.monotonic()
        except Exception as error:  # noqa: BLE001 - told to the page; the rest of the review still works
            traceback.print_exc()
            feed.add({"event": "problem", "message": f"the score over time could not be computed: {error}"})
        if batch:
            feed.add({"event": "points", "points": batch})
        feed.add({"event": "done"})
        feed.close()

    def start_live(self, dataset: str, split: str, conversation_id: str, extractor: str, speed: float) -> dict[str, Any]:
        if extractor not in {entry.id for entry in self.finder.find() if entry.error is None}:
            raise Refused(f"no extractor {extractor!r} is found under {self.root} any more")
        if speed not in self.config.ui.speeds:
            raise Refused(f"the speed must be one of {list(self.config.ui.speeds)} (ui.speeds in config.yaml)")
        sample = self.library.samples().get(dataset)
        if sample is None or sample.split != split or conversation_id not in sample.conversations:
            raise Refused(f"{conversation_id} is not in the config's sample of {dataset}: a live run is made on one "
                          "of its forms")
        conversation, form, gold = self.library.conversation(dataset, split, conversation_id)
        with self._lock:
            if self.live is not None:
                self.live.stop("another run was started")
                self._feeds.pop(self.live.feed.id, None)
            self.live = LiveRun(self.config, conversation, form, gold, extractor, speed, self.root, self.match,
                                self.config_path)
            self._feeds[self.live.feed.id] = self.live.feed
            return self.live.about()

    def stop_live(self) -> None:
        with self._lock:
            if self.live is not None:
                self.live.stop()


def _finite(value: Any) -> Any:
    """A payload without the numbers JSON can't hold (infinity, not-a-number): they become null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def dumps(payload: Any) -> bytes:
    try:
        return json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str).encode()
    except ValueError:
        return json.dumps(_finite(payload), ensure_ascii=False, default=str).encode()


class Handler(BaseHTTPRequestHandler):
    app: App  # set by make_server
    protocol_version = "HTTP/1.1"
    server_version = "lfe-ui"

    def log_message(self, format: str, *args: Any) -> None:  # the base class's signature
        pass  # one line per request would bury what matters in the terminal

    # ---- answers

    def _send(self, body: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        self._send(dumps(payload), "application/json; charset=utf-8", status)

    def _static(self, name: str) -> None:
        path = STATIC / Path(name).name  # a file of the page's folder, never a path
        if not path.is_file() or path.suffix not in _TYPES:
            self._json({"error": f"no such file: {name}"}, HTTPStatus.NOT_FOUND)
            return
        self._send(path.read_bytes(), _TYPES[path.suffix])

    def _file(self, path: Path, content_type: str) -> None:
        """A file, whole or the byte range the request asks for (a player jumps in a recording that way)."""
        size = path.stat().st_size
        first, last, status = 0, size - 1, HTTPStatus.OK
        asked = self.headers.get("Range", "")
        if asked.startswith("bytes=") and "," not in asked:
            start, _, stop = asked[6:].partition("-")
            try:
                if start:
                    first, last = int(start), min(int(stop), size - 1) if stop else size - 1
                else:  # the last so many bytes
                    first = max(0, size - int(stop))
                status = HTTPStatus.PARTIAL_CONTENT
            except ValueError:
                pass
            if first > last or first >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(last - first + 1))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as source:
            source.seek(first)
            left = last - first + 1
            while left > 0:
                chunk = source.read(min(_CHUNK, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def _events(self, feed: Feed, start: int) -> None:
        """A feed as server-sent events, from event ``start`` on, until the feed is over or the page goes away."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        feed.watch()
        try:
            index = start
            while True:
                events, over = feed.wait(index, timeout=10.0)
                for event in events:
                    self.wfile.write(b"id: %d\ndata: " % index + dumps(event) + b"\n\n")
                    index += 1
                if not events:
                    if over:
                        break
                    self.wfile.write(b": still here\n\n")  # keeps the connection open, and notices a page that left
                self.wfile.flush()
        finally:
            feed.unwatch()

    # ---- requests

    def _for_this_page(self) -> bool:
        """Only requests addressed to the host the page is served on are answered: a name that another site
        points at this machine is not."""
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in {"localhost", "127.0.0.1", "::1", self.app.config.ui.host.lower()}

    def _answer(self, method: str) -> None:
        url = urlsplit(self.path)
        query = dict(parse_qsl(url.query))
        app = self.app
        try:
            if not self._for_this_page():
                self._json({"error": "this page is only served to its own address"}, HTTPStatus.FORBIDDEN)
            elif method == "GET" and url.path == "/":
                self._static("index.html")
            elif method == "GET" and url.path.startswith("/static/"):
                self._static(url.path)
            elif method == "GET" and url.path == "/api/state":
                self._json(app.state())
            elif method == "GET" and url.path == "/api/form":
                self._json(app.conversation(query["dataset"], query["split"], query["id"]))
            elif method == "GET" and url.path == "/api/review":
                self._json(app.review(query["run"], query["id"]))
            elif method == "GET" and url.path == "/api/feed":
                start = query.get("from") or self.headers.get("Last-Event-ID")
                self._events(app.feed(query["id"]), int(start) + (0 if "from" in query else 1) if start else 0)
            elif method == "GET" and url.path == "/api/audio/state":
                self._json(app.recordings.state(query["dataset"], query["id"]))
            elif method == "GET" and url.path == "/api/audio":
                path = recording_path(app.config, Path(query["dataset"]).name, Path(query["id"]).name)
                if path.is_file():
                    self._file(path, "audio/wav")
                else:
                    self._json({"error": "this recording has not been fetched"}, HTTPStatus.NOT_FOUND)
            elif method == "POST" and self.headers.get(_FROM_THE_PAGE) != "1":
                self._json({"error": "only the page itself can ask for this"}, HTTPStatus.FORBIDDEN)
            elif method == "POST" and url.path == "/api/live":
                body = self._body()
                self._json(app.start_live(str(body["dataset"]), str(body["split"]), str(body["id"]),
                                          str(body["extractor"]), float(body["speed"])))
            elif method == "POST" and url.path == "/api/live/stop":
                self._body()
                app.stop_live()
                self._json({})
            elif method == "POST" and url.path == "/api/audio/fetch":
                body = self._body()
                self._json(app.recordings.fetch(str(body["dataset"]), str(body["split"]), str(body["id"])))
            else:
                self._json({"error": f"nothing is served at {url.path}"}, HTTPStatus.NOT_FOUND)
        except Refused as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except (KeyError, FileNotFoundError) as error:
            self._json({"error": f"not found: {error}"}, HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True  # the page went away in the middle of an answer
        except Exception as error:  # noqa: BLE001 - told to the page, and in full to the terminal
            traceback.print_exc()
            try:
                self._json({"error": f"{type(error).__name__}: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            except OSError:
                self.close_connection = True

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(body, dict):
            raise Refused("the request must be a JSON object")
        return body

    def do_GET(self) -> None:  # the names the base class calls
        self._answer("GET")

    def do_POST(self) -> None:
        self._answer("POST")


def make_server(config: Config, root: Path, port: int | None = None,
                config_path: Path | None = None) -> tuple[ThreadingHTTPServer, App]:
    """The page's server, not yet serving. ``port`` 0 takes any free port (``server.server_address`` has it)."""
    app = App(config, root, config_path)
    port = config.ui.port if port is None else port
    try:
        server = ThreadingHTTPServer((config.ui.host, port), type("PageHandler", (Handler,), {"app": app}))
    except OSError as error:
        raise RuntimeError(f"the page can't be served on {config.ui.host}:{port} ({error.strerror or error}). If "
                           f"another program uses that port, choose another one: lfe ui --port {port + 1}") from error
    server.daemon_threads = True
    return server, app


def serve(config: Config, *, root: Path | None = None, port: int | None = None, open_browser: bool = True,
          config_path: Path | None = None) -> None:
    """Serve the page until Ctrl+C. ``root`` is the folder extractors are looked for in (default: the current one)."""
    server, app = make_server(config, root or Path.cwd(), port, config_path)
    url = f"http://{config.ui.host}:{server.server_address[1]}/"
    print(f"lfe ui: the page is at {url}\n  extractors are looked for in {app.root}\n  Ctrl+C stops it", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
    finally:
        app.close()
        server.server_close()
