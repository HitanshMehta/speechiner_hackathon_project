"""`lfe ui`: what the page lets you choose, a conversation scored as far as it has gone, the page's server on a
made-up call (a review, a live run), and the recordings."""

import io
import json
import threading
import time
import urllib.error
import urllib.request
import wave
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest
import yaml
from conftest import words
from test_report import CALL, CALLS, FORM, GOLD

from live_form_eval.data.common import audio
from live_form_eval.data.common.model import Conversation, Gold, GoldFact
from live_form_eval.data.common.store import dataset_dir, load_dataset, write_dataset
from live_form_eval.data.spokenwoz import audio as spokenwoz_audio
from live_form_eval.evaluation.evaluate import LazyMatch
from live_form_eval.evaluation.timeline.shown import Shown, shown_fields
from live_form_eval.harness.recorder import Recorder
from live_form_eval.harness.runner import describe_run, finished_run
from live_form_eval.interface.schema import FieldSpec, FieldType, Form
from live_form_eval.ui.catalog import ExtractorFinder, find_runs
from live_form_eval.ui.playback import conversation_payload, walks_payload
from live_form_eval.ui.scoring import Scorer
from live_form_eval.ui.server import make_server

EXTRACTOR = '''"""A made-up extractor for the tests of the page."""
from pathlib import Path

from live_form_eval import Extractor


class Mine(Extractor):
    async def run(self, session):
        print("listening")
        async for partial in session.partials():
            if "ely" in partial.text:
                session.update("destination", "ely")
        session.report_usage(model="none", extraction=True, usage={"input_tokens": 10, "output_tokens": 2},
                             prices={"input_tokens": 0.00005, "output_tokens": 0.00025})

    async def close(self, session):  # where an extractor ends what it opened for the conversation
        with Path("closed.txt").open("a", encoding="utf-8") as closed:
            closed.write(session.conversation_id + "\\n")
'''
PREDICTIONS = [{"conversation": "call1", "t": 4.0, "field_id": "destination", "value": "ely"},
               {"conversation": "call1", "t": 5.0, "field_id": "day", "value": "tuesday"},
               {"conversation": "call1", "t": 7.0, "field_id": "day", "value": "monday"}]


class WordJudge:
    """Supports a statement when the text holds all of its words (enough for a test)."""

    name = "test"

    def supports(self, pairs):
        return [set(statement.lower().split()) <= set(text.lower().split()) for text, statement in pairs]


# ---- what can be chosen

def test_extractors_are_found_by_reading_their_files(tmp_path):
    (tmp_path / "team").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / ".venv").mkdir()
    (tmp_path / "my-team").mkdir()
    (tmp_path / "mine.py").write_text(
        '"""What this file is about."""\nfrom live_form_eval import Extractor\n\n'
        'class Base(Extractor):\n    """Shared, with no run of its own."""\n\n'
        'class Mine(Base):\n    """Fills the form.\n\n    More about it."""\n    async def run(self, session): ...\n\n'
        'class Helper:\n    async def run(self, session): ...\n')
    (tmp_path / "team" / "theirs.py").write_text(  # a subclass of a class of another file, by its name
        "import live_form_eval\nfrom mine import Mine\n\nclass Faster(Mine):\n    pass\n\n"
        "class Direct(live_form_eval.Extractor):\n    async def run(self, session): ...\n")
    (tmp_path / "my-team" / "one.py").write_text("from live_form_eval import Extractor\n\nclass One(Extractor):\n"
                                                 "    async def run(self, session): ...\n")
    (tmp_path / "broken.py").write_text("from live_form_eval import Extractor\n\nclass Oops(Extractor)\n    pass\n")
    for folder in ("tests", ".venv"):  # folders that are not searched
        (tmp_path / folder / "hidden.py").write_text("from live_form_eval import Extractor\n\nclass No(Extractor):\n"
                                                     "    async def run(self, session): ...\n")
    finder = ExtractorFinder(tmp_path, (".",), ("tests",))
    found = {entry.id: entry for entry in finder.find()}
    assert set(found) == {"mine:Mine", "team.theirs:Faster", "team.theirs:Direct", "my-team/one.py:One", "broken.py"}
    assert found["mine:Mine"].about == "Fills the form."  # the first line of what the class says about itself
    assert found["team.theirs:Direct"].about == ""
    assert (found["mine:Mine"].file, found["team.theirs:Faster"].file) == ("mine.py", "team/theirs.py")
    assert found["broken.py"].error.startswith("line 3:")  # a file being edited doesn't vanish without a word
    # a class that is added, and a file that is repaired, show up the next time the page asks
    (tmp_path / "broken.py").write_text('"""Mended."""\nfrom live_form_eval import Extractor\n\n'
                                        "class Oops(Extractor):\n    async def run(self, session): ...\n")
    (tmp_path / "team" / "theirs.py").unlink()
    again = {entry.id: entry for entry in finder.find()}
    assert set(again) == {"mine:Mine", "my-team/one.py:One", "broken:Oops"}
    assert (again["broken:Oops"].error, again["broken:Oops"].about) == (None, "Mended.")  # … what its file says


# ---- the score so far

def test_the_score_so_far_is_the_score_as_if_the_conversation_ended_then(config, stream_config):
    scorer = Scorer(config, CALL, FORM, GOLD, stream_config, LazyMatch(config, None))
    shown = shown_fields(PREDICTIONS, "call1")
    official = scorer.score(shown, CALLS, {"status": "ok"})
    # the destination is said with the message of 1.891 s and the day with the finished turn at 2.35 s; the form is
    # read 8 s after the last value: at 10.35 s
    assert (scorer.said, official.walk_end) == ([1.891, 2.35], 10.35)

    def at(until):
        return scorer.score(shown, CALLS, {"status": "ok"}, until=until).terms

    assert at(1.0).score is None and at(1.0).completeness is None  # nothing is said yet: nothing to measure
    said = at(3.0)  # both are said, nothing is shown: every value counts as missing until the form shows it
    assert (said.completeness, said.precision, said.speed, said.outcome, said.score) == (0.0, None, 0.0, 0.0, 0.0)
    one = at(4.5)  # the destination came 2.109 s after it was said; the day is still missing
    assert (one.completeness, one.precision, one.outcome) == (0.5, 1.0, pytest.approx(2 / 3))
    assert one.speed == pytest.approx((1 - 2.109 / 8) ** 0.5 / 2)
    wrong = at(6.0)  # "tuesday" stands where "monday" was said
    assert (wrong.completeness, wrong.precision, wrong.outcome) == (0.5, 0.5, pytest.approx(2 / 3))
    assert at(8.0).outcome == 1.0 and at(8.0).precision == pytest.approx((1 + 1 / 3) / 2)  # repaired at 7 s
    assert at(10.35) == at(60.0) == official.terms  # from the end of the scoring on: the official score
    assert official.terms.completeness == 1.0 and official.terms.score == pytest.approx(0.8775, abs=1e-4)

    moments = scorer.moments(shown, 0.0, official.walk_end, 0.5)
    assert {1.892, 2.351, 4.001, 5.001, 7.001} <= set(moments)  # just after each thing said or shown: the jumps
    assert {0.5, 5.5, 10.0} <= set(moments) and max(moments) < official.walk_end
    series = list(scorer.series(PREDICTIONS, CALLS, {"status": "ok"}, 0.5))
    assert [point["t"] for point in series[:-1]] == moments
    assert (series[-1]["t"], series[-1]["score"]) == (10.35, official.terms.score)
    assert next(point for point in series if point["t"] == 4.001)["completeness"] == 0.5


def test_a_note_is_shown_with_what_the_judge_found(config, stream_config):
    form = Form(form_id="m", name="Meeting", fields=(
        FieldSpec(field_id="decisions", label="Decisions", prompt="One per line.", type=FieldType.textarea),))
    meeting = Conversation(id="m1", dataset="ami", split="all", end=31.0, roles={"A": "Project Manager"}, meta={},
                           words=tuple(words([("A", 0.0, "the remote is yellow."), ("A", 30.0, "it has a wheel.")])))
    gold = Gold(conversation_id="m1", facts=(
        GoldFact(field_id="decisions", text="The remote is yellow.", said=1.15, source="llm",
                 detail={"quote": "remote is yellow."}),
        GoldFact(field_id="decisions", text="It has a wheel.", said=31.15, source="llm")))
    page = conversation_payload(config, meeting, form, gold, stream_config)
    steps = page["form"]["fields"][0]["steps"]
    assert [step["value"] for step in steps] == [["The remote is yellow."], ["The remote is yellow.", "It has a wheel."]]
    assert steps[0]["heard"] == "remote is yellow." and steps[0]["spoken"] == 1.15 < steps[0]["said"]
    assert steps[1]["heard"] == "the remote is yellow. it has a wheel."  # no quote: the words up to that moment
    assert (page["form"]["fields"][0]["list"], page["speakers"], page["scored_until"]) == (True, {"A": "Project Manager"}, None)

    scorer = Scorer(config, meeting, form, gold, stream_config, LazyMatch(config, WordJudge()))
    shown = {"decisions": Shown.from_updates([(20.0, "The remote is yellow.\nIt is cheap.")])}
    score = scorer.score(shown, [], {"status": "ok", "run_end": 40.0})
    walk = walks_payload(form, score, scorer.text_match())["decisions"]
    assert [display["grade"] for display in walk["displays"]] == [pytest.approx(2 / 3)]  # one fact of one, one line of two
    judged = walk["judged"][0]
    assert (judged["t"], judged["units"]) == (20.0, ["The remote is yellow.", "It is cheap."])
    # against what was said when it was written, and against what is said once the second fact has arrived
    assert judged["against"] == {"0": {"facts": [True], "units": [True, False]},
                                 "1": {"facts": [True, False], "units": [True, False]}}
    assert score.terms.completeness == pytest.approx(1 / 3)  # the second step never got a text of its own


# ---- the page's server

@pytest.fixture
def home(config, stream_config, tmp_path):
    """A folder to start the page in: the made-up call as the only prepared data, a config file that says so,
    and an extractor's file."""
    paths = config.paths.model_copy(update={name: tmp_path / name for name in ("processed", "audio", "runs", "results")})
    calls = config.datasets.spokenwoz.model_copy(update={"split": "dev", "n": None})
    here = config.model_copy(update={"paths": paths, "stream": stream_config,
                                     "datasets": config.datasets.model_copy(update={"spokenwoz": calls})})
    write_dataset(dataset_dir(here, "spokenwoz"), [CALL], [GOLD], {"f": FORM})
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(here.model_dump(mode="json")), encoding="utf-8")
    (tmp_path / "mine.py").write_text(EXTRACTOR, encoding="utf-8")
    return here, tmp_path


@pytest.fixture
def page(home):
    """The page's server, serving; its address."""
    here, folder = home
    server, app = make_server(here, folder, port=0, config_path=folder / "config.yaml")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    app.close()
    server.shutdown()
    server.server_close()


def get(url, **headers):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
        return json.loads(response.read())


def post(url, body, **headers):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def refused(call, *args, **headers) -> tuple[int, str]:
    with pytest.raises(urllib.error.HTTPError) as error:
        call(*args, **headers)
    return error.value.code, json.loads(error.value.read())["error"]


def events(url):
    """A feed's events until it is over."""
    with urllib.request.urlopen(url, timeout=60) as stream:
        for raw in stream:
            if raw.startswith(b"data: "):
                event = json.loads(raw[6:])
                yield event
                if event["event"] == "done":
                    return


def a_run(here) -> str:
    """A run folder of the made-up call, as `lfe run` leaves it."""
    data = load_dataset(here, "spokenwoz", ids=["call1"])
    info = describe_run("Careful", "mine:Mine", data, here, speed=1.0, concurrency=1, official=False)
    recorder = Recorder(here.paths.runs / info["run_id"])
    for line in PREDICTIONS:
        recorder.prediction("call1", line["t"], line["field_id"], line["value"])
    recorder.call("call1", 4.0, {key: value for key, value in CALLS[0].items() if key not in ("conversation", "t")})
    recorder.finish(finished_run(info, [{"conversation": "call1", "status": "ok", "error": None, "run_end": 7.5}],
                                 time.time()))
    return info["run_id"]


def test_the_page_lists_what_can_be_chosen_and_replays_a_run(home, page):
    here, _ = home
    run = a_run(here)
    state = get(f"{page}/api/state")
    assert state["datasets"] == [{"name": "spokenwoz", "split": "dev", "forms": [
        {"id": "call1", "seconds": 4.25, "fields": 4, "said": 2}]}]
    assert "is missing" in state["problems"][0]  # the other dataset is not prepared in this folder
    assert [(entry["id"], entry["about"]) for entry in state["extractors"]] == [
        ("mine:Mine", "A made-up extractor for the tests of the page.")]
    assert [(item["id"], item["name"], item["official"], item["conversations"]) for item in state["runs"]] == [
        (run, "Careful", False, [["call1", "ok"]])]
    assert [item.id for item in find_runs(here)] == [run]
    assert state["settings"]["speeds"] == list(here.ui.speeds) and state["live"] is None

    form = get(f"{page}/api/form?dataset=spokenwoz&split=dev&id=call1")
    assert [(message["t"], message["kind"], message["parts"]) for message in form["stream"]] == [
        (0.891, "partial", [["PENDING", "a"]]), (1.891, "partial", [["A", "a train to ely on"]]),
        (2.35, "turn", [["A", "a train to ely on monday please."]]), (4.55, "turn", [["PENDING", "okay."]])]
    destination, day = form["form"]["fields"][:2]
    assert destination["steps"] == [{"said": 1.891, "spoken": 1.15, "value": "ely", "source": "span",
                                     "heard": "a train to ely"}]
    assert (day["options"], day["steps"][0]["said"]) == (["monday", "tuesday"], 2.35)
    # what an extractor is told, which the page shows next to the values: the form's rules and each field's prompt
    assert (form["form"]["rules"], destination["prompt"], day["prompt"]) == (
        "Fill it as the caller speaks.", "Where to?", "Which day?")
    assert (form["speakers"], form["end"], form["scored_until"], form["audio"]) == (
        {"A": "caller", "B": "agent"}, 19.55, 10.35, {"state": "missing"})

    review = get(f"{page}/api/review?run={run}&id=call1")
    assert review["updates"] == {"destination": [[4.0, "ely"]], "day": [[5.0, "tuesday"], [7.0, "monday"]]}
    assert (review["final"]["score"], review["final"]["correct"], review["end"]) == (pytest.approx(0.8775, abs=1e-4), 2, 10.35)
    assert [(display["t"], display["grade"], display["kind"]) for display in review["fields"]["day"]["displays"]] == [
        (5.0, 0.0, "other"), (7.0, 1.0, "landing")]
    assert review["calls"][0]["cache_read"] == 800 and review["calls"][0]["fresh_input"] == 400
    feed = list(events(f"{page}/api/feed?id={review['feed']}"))
    points = [point for event in feed if event["event"] == "points" for point in event["points"]]
    assert feed[0] == {"event": "plan", "moments": len(points)} and feed[-1] == {"event": "done"}
    assert (points[-1]["t"], points[-1]["score"]) == (10.35, review["final"]["score"])
    assert get(f"{page}/api/review?run={run}&id=call1")["feed"] == review["feed"]  # computed once

    # only the page itself, at its own address, is served: no other site can read it or start a run through it
    assert refused(get, f"{page}/api/state", Host="elsewhere.example")[0] == 403
    assert refused(post, f"{page}/api/live", {}) == (403, "only the page itself can ask for this")
    assert refused(get, f"{page}/api/review?run=..&id=call1")[0] == 400
    assert refused(get, f"{page}/api/nothing")[0] == 404


def test_a_recording_is_served_in_parts(home, page):
    here, _ = home
    path = audio.recording_path(here, "spokenwoz", "call1")
    path.parent.mkdir(parents=True)
    path.write_bytes(bytes(range(200)))
    assert get(f"{page}/api/audio/state?dataset=spokenwoz&id=call1") == {"state": "ready"}
    request = urllib.request.Request(f"{page}/api/audio?dataset=spokenwoz&id=call1", headers={"Range": "bytes=10-19"})
    with urllib.request.urlopen(request, timeout=30) as response:  # a player jumps into a recording this way
        assert (response.status, response.headers["Content-Range"], response.read()) == (
            206, "bytes 10-19/200", bytes(range(10, 20)))
    assert refused(get, f"{page}/api/audio?dataset=spokenwoz&id=another")[0] == 404


def test_a_live_run_from_the_page(home, page):
    here, folder = home
    ask = {"dataset": "spokenwoz", "split": "dev", "id": "call1", "extractor": "mine:Mine", "speed": 30}
    mine = {"X-Lfe-Page": "1"}
    assert "no extractor" in refused(post, f"{page}/api/live", {**ask, "extractor": "os:system"}, **mine)[1]
    assert "speed" in refused(post, f"{page}/api/live", {**ask, "speed": 7}, **mine)[1]
    assert "not in the config's sample" in refused(post, f"{page}/api/live", {**ask, "id": "call2"}, **mine)[1]

    live = post(f"{page}/api/live", ask, **mine)
    assert get(f"{page}/api/state")["live"]["feed"] == live["feed"]
    feed = list(events(f"{page}/api/feed?id={live['feed']}"))
    kinds = [event["event"] for event in feed]
    assert kinds[0] == "preparing" and kinds[-1] == "done"
    assert kinds.index("started") < kinds.index("update") < kinds.index("ended")
    assert {"event": "log", "line": "listening"} in feed  # what the extractor prints
    filled = next(event for event in feed if event["event"] == "update")
    assert (filled["field"], filled["value"]) == ("destination", "ely")
    used = [event for event in feed if event["event"] == "call"]  # as a review has it: tokens by kind, € by unit
    assert [(event["eur"], event["extraction"], event["fresh_input"]) for event in used] == [(pytest.approx(0.001), True, 10)]
    assert used[0]["units"] == {"input_tokens": pytest.approx(0.0005), "output_tokens": pytest.approx(0.0005)}
    ended = next(event for event in feed if event["event"] == "ended")
    assert ended["status"]["status"] == "ok"
    last = [event for event in feed if event["event"] == "score"][-1]  # the run as the evaluation scores it
    assert (last["final"]["correct"], last["final"]["missing"], last["final"]["eur"]) == (1, 1, pytest.approx(0.001))
    assert last["fields"]["destination"]["displays"][0]["grade"] == 1.0 and last["points"][-1]["t"] == 10.35
    # it left a run folder like any run: it can be reviewed and evaluated
    run = json.loads((here.paths.runs / ended["run"] / "run.json").read_text())
    assert (run["name"], run["official"], run["speed"], run["started_from"]) == ("Mine", False, 30.0, "lfe ui")
    assert run["sample"]["ids"] == ["call1"] and run["config"]["stream"] == here.stream.model_dump(mode="json")
    assert [item.id for item in find_runs(here)] == [ended["run"]]
    closed = folder / "closed.txt"  # the made-up extractor notes every close() there
    assert closed.read_text().split() == ["call1"]

    # a run that is stopped keeps what it published until then, and its extractor still gets its close(): what
    # it opened for the conversation (a billed session, say) must not stay open
    slow = post(f"{page}/api/live", {**ask, "speed": 0.5}, **mine)
    stopped = None
    for event in events(f"{page}/api/feed?id={slow['feed']}"):
        if event["event"] == "started":
            post(f"{page}/api/live/stop", {}, **mine)
        if event["event"] == "ended":
            stopped = event
    assert (stopped["status"]["status"], stopped["reason"]) == ("stopped", "stopped from the page")
    assert json.loads((here.paths.runs / stopped["run"] / "run.json").read_text())["conversations"][0]["status"] == "stopped"
    assert closed.read_text().split() == ["call1", "call1"]


# ---- the recordings

class _Parts(BaseHTTPRequestHandler):
    """A server that gives a file in the byte ranges it is asked for, and counts them."""

    payload = b""
    asked: ClassVar[list[int]] = []

    def do_GET(self):  # the name the base class calls
        first, _, last = self.headers["Range"].removeprefix("bytes=").partition("-")
        part = self.payload[int(first): int(last) + 1]
        type(self).asked.append(len(part))
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {first}-{int(first) + len(part) - 1}/{len(self.payload)}")
        self.send_header("Content-Length", str(len(part)))
        self.end_headers()
        self.wfile.write(part)

    def log_message(self, *args):
        pass


def test_one_call_is_read_out_of_the_archive_and_mixed_down(home, monkeypatch):
    here, _ = home
    recording = io.BytesIO()
    with wave.open(recording, "wb") as out:  # two speakers, one per channel
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(8000)
        out.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in (1000, 3000, -1000, 1000)))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
        out.writestr("audio_5700_test/other.wav", b"x" * 5000)
        out.writestr("audio_5700_test/call1.wav", recording.getvalue())
        out.writestr("__MACOSX/audio_5700_test/._call1.wav", b"not a recording")
    _Parts.payload, _Parts.asked = archive.getvalue(), []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Parts)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(spokenwoz_audio, "TEST_ARCHIVE", f"http://127.0.0.1:{server.server_address[1]}/archive.zip")
    monkeypatch.setattr(spokenwoz_audio, "_archive", None)
    try:
        seen = []
        path = audio.fetch_recording(here, "spokenwoz", "test", "call1", lambda done, total: seen.append((done, total)))
        assert path == here.paths.audio / "spokenwoz" / "call1.wav" and seen[-1] == (len(recording.getvalue()),) * 2
        with wave.open(str(path)) as mixed:  # one channel: the mean of the two speakers, sample by sample
            assert (mixed.getnchannels(), mixed.getframerate(), mixed.getnframes()) == (1, 8000, 2)
            assert mixed.readframes(2) == (2000).to_bytes(2, "little", signed=True) + (0).to_bytes(2, "little", signed=True)
        assert [item.name for item in path.parent.iterdir()] == ["call1.wav"]  # nothing else is left behind
        asked = len(_Parts.asked)
        assert asked <= 4  # the archive's size, its directory and the call: not the whole archive
        assert audio.fetch_recording(here, "spokenwoz", "test", "call1") == path and len(_Parts.asked) == asked
        with pytest.raises(audio.AudioUnavailable, match="holds no nobody.wav"):
            audio.fetch_recording(here, "spokenwoz", "test", "nobody")
        with pytest.raises(audio.AudioUnavailable, match="10 GB"):  # the dev recordings can't be read call by call
            audio.fetch_recording(here, "spokenwoz", "dev", "call2")
    finally:
        server.shutdown()
        server.server_close()
