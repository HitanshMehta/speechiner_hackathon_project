"""The report's pages on a made-up call scored for two runs, the run folders behind a report, and the judge's
verdicts kept on disk."""

from pathlib import Path

import pandas as pd
import pytest
from conftest import words

from live_form_eval.cli import _run_folders
from live_form_eval.data.common.model import Conversation, Gold, GoldItem, GoldValue
from live_form_eval.data.common.store import Dataset
from live_form_eval.evaluation.evaluate import LazyMatch, RunScore, score_conversation
from live_form_eval.evaluation.matching.judge.nli import VerdictStore
from live_form_eval.evaluation.report.pages import (
    _change,
    _talk,
    form_entries,
    form_page,
    gold_entries,
    run_names,
    write_report,
)
from live_form_eval.evaluation.report.plots import (
    SHOULD_SHOW,
    by_field_type,
    latency_curves,
    outcomes_by_field_type,
    run_key,
    score_against_price,
    terms_by_run,
    timeline_chart,
)
from live_form_eval.evaluation.report.style import SERIES
from live_form_eval.evaluation.report.tables import build_result
from live_form_eval.evaluation.timeline.shown import Shown
from live_form_eval.interface.schema import FieldSpec, FieldType, Form, OptionsConfig

FORM = Form(form_id="f", name="Train booking", system_prompts={"form_data_extraction": "Fill it as the caller speaks."},
            fields=(FieldSpec(field_id="destination", label="Destination", prompt="Where to?", type=FieldType.text),
                    FieldSpec(field_id="day", label="Day", prompt="Which day?", type=FieldType.select,
                              field_configuration=OptionsConfig(options=("monday", "tuesday"))),
                    FieldSpec(field_id="people", label="People", prompt="How many?", type=FieldType.integer),
                    FieldSpec(field_id="extras", label="Extras", prompt="What else?", type=FieldType.multiselect,
                              field_configuration=OptionsConfig(options=("bike", "seat")))))
# With the tests' stream settings, the call reaches an extractor in four messages: "a" at 0.891 s, "a train to ely
# on" at 1.891 s, the finished turn at 2.35 s, and the agent's "okay." at 4.55 s.
CALL = Conversation(id="call1", dataset="spokenwoz", split="dev", end=4.25, roles={"caller": "caller", "agent": "agent"},
                    words=tuple(words([("caller", 0.0, "a train to ely on monday please."), ("agent", 4.0, "okay.")])),
                    meta={"form_id": "f", "unit_sizes": [7, 1]})
GOLD = Gold(conversation_id="call1", values=(
    GoldValue(field_id="destination", value="ely", said=1.15, source="span"),  # the end of "ely"
    GoldValue(field_id="day", value="monday", said=1.75, source="span")))
# Two model calls: both values come with the first one (800 of its 1,200 input tokens read from a cache), the
# second changes nothing.
CALLS = [{"conversation": "call1", "t": 4.0, "model": "a-model", "eur": 0.01, "input_tokens": 1200,
          "output_tokens": 30, "latency_s": 0.8, "cache_read_tokens": 800},
         {"conversation": "call1", "t": 9.0, "model": "a-model", "eur": 0.01, "input_tokens": 1200,
          "output_tokens": 30, "latency_s": 0.8}]


def scored(config, stream_config, name: str, day: str, calls=CALLS, gold: Gold = GOLD, shown=None) -> RunScore:
    """A run of one call: the destination right and the day as given, or what ``shown`` says."""
    shown = shown or {"destination": Shown.from_updates([(4.0, "ely")]), "day": Shown.from_updates([(5.0, day)])}
    item = score_conversation(CALL, FORM, gold, shown, calls, {"status": "ok"}, "spokenwoz", config, stream_config,
                              LazyMatch(config, None))
    run = {"run_id": f"20260101-000000-{name}-spokenwoz", "name": name, "dataset": "spokenwoz", "split": "dev",
           "official": False, "conversations": [{"conversation": "call1", "status": "ok"}],
           "config": {"stream": stream_config.model_dump(mode="json")}}  # the stream the run was replayed with
    data = Dataset(name="spokenwoz", split="dev", conversations={"call1": CALL}, gold={"call1": gold},
                   forms={"f": FORM})
    return RunScore(run_dir=Path("runs") / run["run_id"], run=run, dataset="spokenwoz", data=data,
                    field_types={spec.field_id: spec.type.value for spec in FORM.fields}, conversations=[item],
                    calls=list(calls))


# ---- the page of a form

def test_a_form_page_shows_the_form_and_every_run_on_it(config, stream_config):
    scores = [scored(config, stream_config, "Careful", "monday"), scored(config, stream_config, "Hasty", "tuesday")]
    entries = form_entries(scores)
    assert [(entry.dataset, entry.conversation, len(entry.runs)) for entry in entries] == [("spokenwoz", "call1", 2)]
    assert entries[0].lowest == min(score.conversations[0].terms.score for score in scores)
    page = form_page(entries[0], entries, config)
    careful, hasty = SERIES[config.plots.mode][:2]
    for expected in ("<h1>call1</h1>", "2 of its 4 fields have something to show in this conversation",
                     "Train booking", "<b>Rules:</b> Fill it as the caller speaks.",  # the form itself
                     "<td>Which day?</td><td>monday, tuesday</td><td><span class='chip'>said here</span></td>",
                     "<td>How many?</td><td></td><td></td>",  # a field nobody talks about in this call
                     f"<p class='key'><span><span class='dot' style='color:{careful}'>■</span>&nbsp;Careful</span>",
                     "Hasty", "<span class='chip correct'>correct</span>",
                     "<span class='chip wrong'>wrong</span>",  # Hasty's day
                     "<a href='#field-day'>Day</a>", "<details id='field-day'>",  # from the form at the end to a field
                     "2 more fields stayed empty", "People, Extras",
                     "<a href='#calls'>Model calls</a>", "<h2 id='calls'>Model calls and price on this form</h2>",
                     # its calls: 2, of which 1 changed a value, none failed; tokens by kind; no other use; price;
                     # seconds
                     ("<td>a-model</td><td>2</td><td>1</td><td>0</td><td>800</td><td>1,600</td><td>0</td><td>60</td>"
                      "<td>0</td><td>0.0000</td><td>0.0200</td><td>0.80</td><td>0.80</td>"),
                     # the price unit by unit: a run made before units had prices reported only its total
                     "<th>Not split by unit (€)</th>", "<td>0.02000</td>",
                     "<a href='#transcript'>Transcript</a>", "<details id='transcript'>",  # the parts of the page
                     "<option value='spokenwoz-call1.html' selected>call1"):  # the selector at the top
        assert expected in page, expected
    assert page.count("<h3>What it should show</h3>") == 2  # destination and day have gold values over time
    # the conversation as the extractor received it: per turn, the gold it completes and what each run filled
    assert "The conversation as the extractor received it: 2 turns, 4 messages" in page
    assert (f"<th>Gold said here</th><th>Filled here by <span class='dot' style='color:{careful}'>■</span>&nbsp;"
            f"Careful</th><th>Filled here by <span class='dot' style='color:{hasty}'>■</span>&nbsp;Hasty</th>") in page
    ely = "<div class='said'><span class='chip correct' title='right'>✓</span><b>Destination</b> ely</div>"
    assert ("<td><span class='who' title='caller'>A</span>a train to ely on monday please.</td>"  # the caller's turn:
            "<td><div class='said'><b>Destination</b> ely</div><div class='said'><b>Day</b> monday</div></td>"
            f"<td>{ely}</td><td>{ely}</td></tr>") in page  # both runs filled the destination before the agent spoke
    assert ("<td><div class='said'><span class='chip correct' title='right'>✓</span><b>Day</b> monday</div></td>"
            "<td><div class='said'><span class='chip wrong' title='wrong'>✕</span><b>Day</b> tuesday</div></td>") in page
    assert "<div data-view-in='talk' data-view='messages' hidden>" in page  # the other view: message by message
    assert "tooltip: chartTip" in page  # the plots' values open in the page's window, in full
    # a run that reports no call, or only counts its calls without tokens or a price: no part about calls
    for calls in ([], [{"conversation": "call1", "t": 4.0, "model": "an-oracle", "eur": 0.0}]):
        quiet = form_entries([scored(config, stream_config, "Quiet", "monday", calls)])
        assert "id='calls'" not in form_page(quiet[0], quiet, config)


def test_the_conversation_shows_message_by_message_what_was_said_and_filled(config, stream_config):
    gold = Gold(conversation_id="call1", values=(
        GoldValue(field_id="destination", value="e", said=0.25, source="span"),  # dictated: a piece per message
        GoldValue(field_id="destination", value="el", said=1.15, source="span"),
        GoldValue(field_id="destination", value="ely", said=2.05, source="span"),
        GoldValue(field_id="day", value=None, said=4.25, source="turn")),  # withdrawn with the agent's turn
        items=(GoldItem(field_id="extras", text="bike", said=0.55, source="rule"),
               GoldItem(field_id="extras", text="seat", said=1.75, source="rule")))
    long = "ely " * 50  # a value longer than a cell
    shown = {"people": Shown.from_updates([(0.5, 3)]),  # before anything arrived, and nothing is said for it
             "destination": Shown.from_updates([(1.0, "e"), (2.0, "el"), (3.0, "ely"), (5.0, long)]),
             "extras": Shown.from_updates([(1.5, ["bike"]), (2.5, ["bike", "seat"])])}  # "bike" before it was said
    entry = form_entries([scored(config, stream_config, "Mine", "monday", gold=gold, shown=shown)])[0]
    texts: list[str] = []
    table = _talk(CALL, FORM, entry, stream_config, {}, {}, texts)
    turns, messages = table.split("<div data-view-in='talk' data-view='messages' hidden>")

    def right(label, value, note=""):
        return f"<div class='said'><span class='chip correct' title='right'>✓</span><b>{label}</b> {value}{note}</div>"

    def said(label, value):
        return f"<div class='said'><b>{label}</b> {value}</div>"

    early = " <span class='sub'>(0.4 s before it was said)</span>"  # shown at 1.5, said with the message of 1.891
    rows = messages.split("<tr>")[2:]  # after the head: a line per message
    assert len(rows) == 4
    # the first partial brings "a": the gold's first piece. What was filled before it counts with it.
    assert rows[0] == ("<td>0.9</td><td><span class='who' title='too short to tell who speaks'>?</span>"
                       "<span class='old'></span>a</td>" + f"<td>{said('Destination', 'e')}</td>"
                       "<td><div class='said'><span class='chip wrong' title='wrong'>✕</span><b>People</b> 3</div>"
                       + right("Destination", "e") + right("Extras", "bike", early) + "</td></tr>")
    # the next partial: the turn's earlier words in grey, then the new ones; the next piece, and a first option
    assert rows[1] == ("<td>1.9</td><td><span class='who' title='caller'>A</span><span class='old'>a</span> train to "
                       f"ely on</td><td>{said('Destination', 'el')}{said('Extras', '+ bike')}</td>"
                       f"<td>{right('Destination', 'el')}</td></tr>")
    assert rows[2].startswith("<td>2.4</td><td><span class='who' title='caller'>A</span>a train to ely on monday "
                              "please. <span class='sub'>(the turn is finished)</span></td>"
                              f"<td>{said('Destination', 'ely')}{said('Extras', '+ seat')}</td>")
    assert right("Extras", "bike, seat") + right("Destination", "ely") in rows[2]
    assert said("Day", "withdrawn") in rows[3]
    # a value longer than a cell is cut there, and names its whole text for the page's window
    assert texts == [long]
    assert f"<b>Destination</b> <span data-ref='0' data-name='Destination'>{long[:160].rstrip()} …</span>" in rows[3]
    # turn by turn, the same things stand on the line of their turn
    lines = turns.split("<tr>")[2:]
    assert len(lines) == 2 and all(said("Destination", piece) in lines[0] for piece in ("e", "el", "ely"))
    assert right("Destination", "ely") in lines[0] and said("Day", "withdrawn") in lines[1]
    assert [(item.spec.field_id, item.text) for item in gold_entries(FORM, gold)] == [
        ("destination", "e"), ("extras", "bike"), ("destination", "el"), ("extras", "seat"), ("destination", "ely"),
        ("day", None)]


def test_a_field_of_pieces_shows_what_changes():
    assert _change([], ["a", "b"]) == "<ul class='diff'><li class='add'>a</li><li class='add'>b</li></ul>"
    assert _change(["a", "b"], ["a", "c"]) == ("<ul class='diff'><li class='add'>c</li><li class='gone'>b</li></ul>"
                                               "<div class='sub'>1 more, unchanged</div>")
    assert _change(["a"], []) == "<ul class='diff'><li class='gone'>a</li></ul><div class='sub'>nothing is left</div>"
    assert "the same as before" in _change(["a"], ["a"]) and "empty" in _change([], [])


def test_the_timeline_names_every_line_and_marks_every_change():
    rows = [{"field": "Day", "lane": SHOULD_SHOW, "state": SHOULD_SHOW, "value": "monday", "start": 1, "end": 9,
             "grade": None, "full": "monday"},
            {"field": "Day", "lane": "Hasty", "state": "Shown: wrong", "value": "tuesday", "start": 5, "end": 9,
             "grade": 0.0, "full": "tuesday"},
            {"field": "People", "lane": "Careful", "state": "Shown: wrong", "value": "2", "start": 3, "end": 9,
             "grade": 0.0, "full": "2"}]
    marks = [{"field": "Day", "lane": "Hasty", "start": 5, "change": "First fill", "before": "—", "after": "tuesday",
              "before_full": None, "after_full": "tuesday"},
             {"field": "People", "lane": "Careful", "start": 30, "change": "Revision", "before": "2", "after": "3"},
             {"field": "Extras", "lane": "Careful", "start": 3, "change": "First fill", "before": "—", "after": "bike"}]
    spec = timeline_chart(rows, marks, ["Careful", "Hasty"], "title", "subtitle", 9.0, "light").to_dict()
    bars, points = spec["layer"]
    lines = {row["value"]: row["line"] for row in spec["datasets"][bars["data"]["name"]]}
    assert lines == {"monday": 0, "tuesday": 2, "2": 5}  # per field: its gold, then a line per run, then a gap
    assert bars["encoding"]["y"]["axis"]["labelExpr"] == (
        '["Day", "Careful", "Hasty", "", "People", "Careful", "Hasty"][datum.value]')
    assert bars["encoding"]["y"]["scale"]["domain"] == list(range(7))
    assert bars["encoding"]["x"]["scale"] == {"domain": [0, 9.0], "nice": False}
    # a mark sits on its run's line; one after the end of the plot, or of a field without a line, is left out
    assert [(row["after"], row["line"]) for row in spec["datasets"][points["data"]["name"]]] == [("tuesday", 2)]
    shapes = points["encoding"]["shape"]["scale"]
    assert shapes["domain"] == ["First fill", "Revision", "Cleared"] and shapes["range"][:2] == ["circle", "triangle-up"]
    assert len(timeline_chart(rows, [], ["Careful", "Hasty"], "title", "subtitle", 9.0, "light").to_dict()["layer"]) == 1
    # a plot of its own keeps the values as it shows them; a page takes them in full, each once, by their place
    assert all("full" not in row and "value_ref" not in row for row in spec["datasets"][bars["data"]["name"]])
    texts: list[str] = []
    paged = timeline_chart(rows, marks, ["Careful", "Hasty"], "title", "subtitle", 9.0, "light", texts).to_dict()
    bars, points = paged["layer"]
    assert texts == ["monday", "tuesday", "2"]
    assert [row["value_ref"] for row in paged["datasets"][bars["data"]["name"]]] == [0, 1, 2]
    assert [row["after_ref"] for row in paged["datasets"][points["data"]["name"]]] == [1]  # "tuesday" is there once
    assert {"field": "value_ref", "type": "quantitative", "title": "value_ref", "format": "d"} in bars["encoding"]["tooltip"]


# ---- the report of several runs

def test_runs_that_share_a_name_are_told_apart():
    def run(run_id):
        return RunScore(run_dir=Path(run_id), run={"run_id": run_id, "name": "Mine"}, dataset="ami", data=None,
                        field_types={})

    assert run_names([run("20260101-000000-Mine-ami"), run("20260102-000000-Mine-ami")]) == [
        "Mine (20260101-000000)", "Mine (20260102-000000)"]
    assert run_names([run("20260101-000000-Mine-ami")]) == ["Mine"]


def test_the_tables_hold_the_calls_and_the_value_changes(config, stream_config, tmp_path):
    scores = [scored(config, stream_config, "Careful", "monday"), scored(config, stream_config, "Hasty", "tuesday")]
    result = build_result(scores, config, tmp_path)
    run = result.summary.iloc[0]
    assert (run["calls_per_form"], run["calls_changing_share"], run["failed_calls"]) == (2, 0.5, 0)
    assert (run["cache_read_tokens_per_form"], run["fresh_input_tokens_per_form"], run["cache_write_tokens_per_form"],
            run["output_tokens_per_form"]) == (800, 1600, 0, 60)
    assert run["cached_input_share"] == pytest.approx(800 / 2400)
    assert run["tokens_per_call"] == pytest.approx(2460 / 2)
    assert (run["call_seconds_median"], run["eur_per_form"]) == (0.8, 0.02)
    # two fields were filled, each once: no revision
    assert (run["fields_filled_per_form"], run["value_changes_per_form"], run["revisions_per_filled_field"],
            run["filled_once_share"]) == (2, 2, 0, 1)
    assert list(result.calls["changes"]) == [2, 0, 2, 0]  # per run: both values came with the first call
    assert list(result.calls["cache_read_tokens"]) == [800, 0, 800, 0]
    form = result.per_conversation.iloc[0]
    assert (form["calls"], form["calls_changing_share"], form["value_changes"], form["revisions"]) == (2, 0.5, 2, 0)
    types = result.per_field_type[result.per_field_type["run"] == run["run"]].set_index("field_type")
    assert (types.loc["select", "fields_filled"], types.loc["select", "value_changes"]) == (1, 1)
    result.write()
    assert len(pd.read_csv(tmp_path / "calls.csv")) == 4


def test_the_plots_of_a_dataset_show_all_its_runs(config, stream_config, tmp_path):
    scores = [scored(config, stream_config, "Hasty", "tuesday"), scored(config, stream_config, "Careful", "monday")]
    result = build_result(scores, config, tmp_path)
    key = run_key(result.summary, "light", "spokenwoz")
    assert key.order == ["Careful", "Hasty"]  # the highest score first …
    assert (key.colors["Hasty"], key.colors["Careful"]) == tuple(SERIES["light"][:2])  # … the colours as given
    everyone = run_key(result.summary, "light")
    assert everyone.order == ["Hasty", "Careful"]
    nobody = run_key(result.summary, "light", "ami")
    assert nobody.names == {}
    scatter = score_against_price(result.summary, everyone, "light").to_dict()
    assert [panel["title"]["text"] for panel in scatter["hconcat"]] == ["SpokenWOZ"]
    terms = terms_by_run(result.summary, key, "light").to_dict()
    assert [panel["title"]["text"] for panel in terms["hconcat"]] == ["Score", "Completeness", "Precision", "Speed",
                                                                    "Outcome"]
    for chart in (terms_by_run(result.summary, key, "light"), latency_curves(result.values, key, "light"),
                  by_field_type(result.per_field_type, key, "light"),
                  outcomes_by_field_type(result.per_field_type, key, "light")):
        assert {row["run"] for rows in chart.to_dict()["datasets"].values() for row in rows} == {"Careful", "Hasty"}
    assert terms_by_run(result.summary, nobody, "light") is None
    assert latency_curves(result.values, nobody, "light") is None
    assert by_field_type(result.per_field_type, nobody, "light") is None
    assert outcomes_by_field_type(result.per_field_type, nobody, "light") is None


def test_a_submission_on_two_datasets_keeps_one_colour(config, stream_config, tmp_path):
    result = build_result([scored(config, stream_config, "Careful", "monday"),
                           scored(config, stream_config, "Hasty", "tuesday")], config, tmp_path)
    twice = pd.concat([result.summary, result.summary.assign(  # the same two submissions on a second dataset
        dataset="ami", run=result.summary["run"].str.replace("spokenwoz", "ami"))], ignore_index=True)
    everyone = run_key(twice, "light")
    assert len(everyone.names) == 4 and everyone.order == ["Careful", "Hasty"]
    assert everyone.scale().to_dict() == {"domain": ["Careful", "Hasty"], "range": SERIES["light"][:2]}
    scatter = score_against_price(twice, everyone, "light").to_dict()
    assert [panel["title"]["text"] for panel in scatter["hconcat"]] == ["SpokenWOZ", "AMI"]
    assert run_key(twice, "light", "ami").colors == everyone.colors


def test_the_report_is_an_overview_by_dataset_and_a_page_per_form(config, stream_config, tmp_path):
    scores = [scored(config, stream_config, "Careful", "monday"), scored(config, stream_config, "Hasty", "tuesday")]
    result = build_result(scores, config, tmp_path)
    (tmp_path / "forms").mkdir()
    (tmp_path / "forms" / "spokenwoz-old.html").write_text("a form of an earlier evaluation")
    report = write_report(result, scores, config)
    assert sorted(path.name for path in (tmp_path / "forms").iterdir()) == ["spokenwoz-call1.html"]
    overview = report.read_text()
    assert "<a href='forms/spokenwoz-call1.html'>call1</a>" in overview  # the table of forms links to each page
    assert "<option value='forms/spokenwoz-call1.html'>call1" in overview and "Choose a form" in overview
    assert list(result.ranking["name"]) == ["Careful", "Hasty"]  # both runs in one ranking
    # one section per dataset holds all its runs, in the same tables and plots: there is no section per run
    assert overview.count("<h2 id='spokenwoz'>SpokenWOZ</h2>") == 1 and "<a href='#spokenwoz'>SpokenWOZ</a>" in overview
    for part in ("<h3>Scores</h3>", "<h3>By field type</h3>", "<h3>Model calls, tokens and price</h3>",
                 "<h3>Value changes</h3>"):
        assert overview.count(part) == 1, part
    assert "<div class='tiles'>" not in overview  # two runs: a table and plots, not the tiles of one run
    assert "<th>Field type</th><th>Run</th>" in overview  # by field type: a row per type and run
    careful, hasty = SERIES[config.plots.mode][:2]
    assert f"<span class='dot' style='color:{careful}'>■</span>&nbsp;Careful" in overview
    assert f"<span class='dot' style='color:{hasty}'>■</span>&nbsp;Hasty" in overview
    # a run has the same colour on the page of a form
    assert f"<span class='dot' style='color:{hasty}'>■</span>&nbsp;Hasty" in (
        tmp_path / "forms" / "spokenwoz-call1.html").read_text()
    alone = build_result(scores[:1], config, tmp_path / "alone")
    single = write_report(alone, scores[:1], config).read_text()
    assert "<h2 id='spokenwoz'>SpokenWOZ</h2>" in single and "<div class='tiles'>" in single
    assert "<th>Field type</th><th>Correct</th>" in single  # one run: its name is not repeated on every row
    # runs that only count their calls: the report has no part about calls
    counted = [scored(config, stream_config, "Quiet", "monday", [{"conversation": "call1", "t": 4.0, "eur": 0.0}])]
    quiet = write_report(build_result(counted, config, tmp_path / "quiet"), counted, config).read_text()
    assert "Model calls, tokens and price" not in quiet and "<h3>Value changes</h3>" in quiet


def test_reports_are_combined_through_their_runs(config, tmp_path, monkeypatch):
    runs, results = tmp_path / "runs", tmp_path / "results"
    for run_id in ("r1", "r2"):
        (runs / run_id).mkdir(parents=True)
        (runs / run_id / "run.json").write_text("{}")
    (results / "mine").mkdir(parents=True)
    pd.DataFrame([{"run": "r1"}, {"run": "r2"}]).to_csv(results / "mine" / "summary.csv", index=False)
    there = config.model_copy(update={"paths": config.paths.model_copy(update={"runs": runs})})
    # a results folder stands for the runs it was made from; a run named twice is scored once
    assert _run_folders([results / "mine", runs / "r2"], there) == [runs / "r1", runs / "r2"]
    (results / "empty").mkdir()
    with pytest.raises(SystemExit, match="neither a run folder"):
        _run_folders([results / "empty"], there)
    pd.DataFrame([{"run": "gone"}]).to_csv(results / "empty" / "summary.csv", index=False)
    with pytest.raises(SystemExit, match="the run gone, which is not in"):
        _run_folders([results / "empty"], there)


def test_verdicts_are_kept_from_one_evaluation_to_the_next(tmp_path):
    store = VerdictStore(tmp_path, "nli:some/model")
    assert store.get(("a text", "a statement")) is None
    store.add({("a text", "a statement"): True, ("a text", "another"): False})
    again = VerdictStore(tmp_path, "nli:some/model")  # the next evaluation reads the file
    assert (again.get(("a text", "a statement")), again.get(("a text", "another"))) == (True, False)
    assert again.get(("a", "text a statement")) is None  # the two sides are kept apart
    assert VerdictStore(tmp_path, "nli:other/model").get(("a text", "a statement")) is None  # one file per judge
