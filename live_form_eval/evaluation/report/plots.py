"""The plots of an evaluation's scores (Altair): the score against the price, the runs of a dataset next to each
other, field types, latency, value changes, and the timeline of a form. report/pages.py puts them into the HTML
report next to the tables. The plots of the model calls are in report/call_plots.py; what all plots share
(colours, sizes, how the runs are named and coloured) is in report/style.py.

A plot of several runs takes a ``RunKey``: which runs, under which names, in which colours. ``run_key`` makes the
one the report uses, so a notebook draws the same plot. Only ``conversation_timeline`` returns a themed chart;
the others are themed by the caller (``themed``).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd

from live_form_eval.config import Config
from live_form_eval.data.common.store import load_dataset
from live_form_eval.evaluation.evaluate import FORM_READ_WHEN_RUN_ENDS, walk_fields
from live_form_eval.evaluation.matching.free_text import TextMatch
from live_form_eval.evaluation.matching.judge.nli import Judge
from live_form_eval.evaluation.metrics.value_changes import Change, field_changes
from live_form_eval.evaluation.report.style import (
    BAR,
    CHROME,
    DATASET_NAMES,
    ROW,
    SMALL,
    STATUS,
    WIDTH,
    Measure,
    RunKey,
    missing,
    run_colors,
    small_multiples,
    themed,
)
from live_form_eval.evaluation.timeline.arrival import arrival_of, in_extractor_time, stream_config_of
from live_form_eval.evaluation.timeline.shown import shown_fields
from live_form_eval.evaluation.timeline.steps import gold_steps, scoring_end
from live_form_eval.evaluation.timeline.walk import FieldWalk
from live_form_eval.harness.recorder import read_run
from live_form_eval.interface.schema import FieldType, Form

OUTCOME_NAMES = {"correct": "Correct", "missing": "Missing", "invented": "Invented", "wrong": "Wrong"}
TERM_NAMES = {"completeness": "Completeness", "precision": "Precision", "speed": "Speed"}
MEASURES = {"score": "Score", **TERM_NAMES, "outcome": "Outcome"}  # what the runs of a dataset are compared by
BY_TYPE = {"final_f1": "Final F1", **TERM_NAMES}  # what the field types are compared by
CHANGE_NAMES = {"filled": "First fill", "revised": "Revision", "cleared": "Cleared"}  # the marks of a timeline …
# … and their shapes, in that order: ●, ▲ and ✕ (the last one as a path, since the library's own cross is a +)
CHANGE_SHAPES = ["circle", "triangle-up", "M-1,-0.6L-0.6,-1L0,-0.4L0.6,-1L1,-0.6L0.4,0L1,0.6L0.6,1L0,0.4L-0.6,1L-1,0.6L-0.4,0Z"]
PANEL = 400  # width of one dataset's plot of the score against the price, when there are several datasets
TIMELINE_WIDTH, LINE = 860, 14  # the timeline: its plot width, and the height of one of its lines
TOP_AXIS_FROM = 30  # a timeline of more lines than this has its time axis on top as well
SHOULD_SHOW = "Should show"  # the timeline's first line of every field: its gold steps
_TOOLTIP_CHARACTERS = 300  # a note can be long: the tables of a form's page hold it in full
_MARK_CHARACTERS = 120  # … and a mark's tooltip holds two of them, before and after


# ---- which runs, under which names, in which colours

def run_labels(runs: Iterable[tuple[str, str, str]]) -> dict[str, str]:
    """How each run is called in the report: by its name, with its start time when two runs share name and
    dataset. ``runs``: (run id, name, dataset), in the order the runs were given."""
    runs = list(runs)
    shared = Counter((name, dataset) for _, name, dataset in runs)
    return {run_id: name if shared[(name, dataset)] == 1 else f"{name} ({run_id[:15]})"
            for run_id, name, dataset in runs}


def run_key(summary: pd.DataFrame, mode: str, dataset: str | None = None) -> RunKey:
    """The runs of an evaluation as its report names and colours them: all of them, or those of one dataset with
    the highest score first. ``summary`` is the table with one row per run."""
    labels = run_labels(zip(summary["run"], summary["name"], summary["dataset"]))
    rows = summary if dataset is None else summary[summary["dataset"] == dataset].sort_values(
        "score", ascending=False, na_position="last")
    return RunKey({run: labels[run] for run in rows["run"]}, run_colors(labels.values(), mode))


# ---- the runs next to each other

def score_against_price(summary: pd.DataFrame, key: RunKey, mode: str) -> alt.HConcatChart | None:
    """Every run as a point, in one plot per dataset: what a form costs against its score. The upper left is
    where to be."""
    rows = [{"run": key.names[row["run"]], "dataset": row["dataset"], "score": row["score"],
             "eur": 0.0 if missing(row["eur_per_form"]) else row["eur_per_form"],
             **{term: row[term] for term in (*TERM_NAMES, "outcome")}}
            for row in summary.to_dict("records") if row["run"] in key.names and not missing(row["score"])]
    if not rows:
        return None
    c = CHROME[mode]
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    width = WIDTH if len(datasets) == 1 else PANEL
    panels = []
    for dataset in datasets:
        mine = sorted((row for row in rows if row["dataset"] == dataset), key=lambda row: -row["score"])
        right = max(max(row["eur"] for row in mine) * 1.15, 0.01)  # room on the right; a scale even at price 0
        for k, row in enumerate(mine):  # the labels of points that sit close together are moved apart, downwards
            row["label_y"] = min([row["score"]] + [other["label_y"] - 0.05 for other in mine[:k]
                                                    if abs(other["eur"] - row["eur"]) < 0.3 * right])
            row["side"] = "left" if row["eur"] > 0.5 * right else "right"  # the label goes where there is room
        frame = pd.DataFrame(mine)
        x = alt.X("eur:Q", title="Price per form (€)", scale=alt.Scale(domain=[0, right]),
                  axis=alt.Axis(format=".3f", tickCount=5))
        layers = [alt.Chart(frame).mark_point(size=110, filled=True, opacity=1, stroke=c["surface"],
                                              strokeWidth=2).encode(
            # no row labels on this axis: it does not need the room the bar charts keep for them
            x=x, y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(minExtent=0)),
            color=alt.Color("run:N", scale=key.scale(), title=None, legend=None),
            tooltip=[alt.Tooltip("run:N", title="Run"), alt.Tooltip("score:Q", title="Score", format=".3f"),
                     alt.Tooltip("completeness:Q", title="Completeness", format=".3f"),
                     alt.Tooltip("precision:Q", title="Precision", format=".3f"),
                     alt.Tooltip("speed:Q", title="Speed", format=".3f"),
                     alt.Tooltip("outcome:Q", title="Outcome", format=".3f"),
                     alt.Tooltip("eur:Q", title="Price per form (€)", format=".4f")])]
        for side, align, dx in (("right", "left", 11), ("left", "right", -11)):
            named = frame[frame["side"] == side]
            if not named.empty:
                layers.append(alt.Chart(named).mark_text(
                    align=align, baseline="middle", dx=dx, color=c["secondary"], fontSize=11,
                    limit=0.55 * width).encode(x=x, y=alt.Y("label_y:Q", title="Score"), text="run:N"))
        panels.append(alt.layer(*layers).properties(width=width, height=260, title=alt.Title(
            DATASET_NAMES.get(dataset, dataset), fontSize=12, fontWeight=500, color=c["primary"], anchor="start")))
    return alt.hconcat(*panels, spacing=28).properties(title=alt.Title(
        "Score against price", subtitle="Upper left is better: a higher score for a cheaper form. A run has the same "
                                        "colour in every plot of this report"))


def terms_by_run(summary: pd.DataFrame, key: RunKey, mode: str) -> alt.HConcatChart | None:
    """The runs of ``key`` next to each other: a small plot for the score and for each of its four terms."""
    return small_multiples(summary, [Measure(column, title, ".3f", (0, 1)) for column, title in MEASURES.items()],
                           key, "The score and its four terms", "Each from 0 to 1; the highest score first", mode)


def changes_by_run(summary: pd.DataFrame, key: RunKey, mode: str) -> alt.HConcatChart | None:
    """The runs of ``key`` next to each other: how often the values of a form changed on screen."""
    return small_multiples(summary, [
        Measure("fields_filled_per_form", "Fields filled per form", ".1f"),
        Measure("value_changes_per_form", "Value changes per form", ".1f"),
        Measure("revisions_per_filled_field", "Revisions per filled field", ".2f"),
        Measure("filled_once_share", "Filled once, never changed", ".0%", (0, 1)),
    ], key, "How often values change", [
        "Every visible change of a field is a first fill, a revision or a clearing. A form that finds each value",
        "once has no revisions; a value that grows while it is said, or a model changing its mind, has some"], mode)


def latency_curves(values: pd.DataFrame, key: RunKey, mode: str) -> alt.Chart | None:
    """The share of values shown for good within x seconds of being said: one line per run."""
    rows = key.frame(values[values["latency_s"].notna()])
    if rows.empty:
        return None
    latencies = {name: group["latency_s"].tolist() for name, group in rows.groupby("run")}
    settled = {name: sorted(value for value in items if math.isfinite(value)) for name, items in latencies.items()}
    top = max([5.0] + [items[int(0.95 * (len(items) - 1))] for items in settled.values() if items])
    points = []
    for name in key.order:
        if name in latencies:
            total, done = len(latencies[name]), settled[name]
            points.append({"run": name, "seconds": 0.0, "share": sum(value <= 0 for value in done) / total})
            points += [{"run": name, "seconds": value, "share": (k + 1) / total}
                       for k, value in enumerate(done) if value <= top]
            points.append({"run": name, "seconds": top, "share": sum(value <= top for value in done) / total})
    return alt.Chart(pd.DataFrame(points), title=alt.Title("Final Decision latency", subtitle=[
        "Share of values on screen for good within x seconds of their words reaching the extractor.",
        "A line that ends under 100% has values that came later than the plot is wide, or never settled"])).mark_line(
        strokeWidth=2, interpolate="step-after").encode(
        x=alt.X("seconds:Q", title="Seconds after the words reached the extractor", scale=alt.Scale(domain=[0, top])),
        y=alt.Y("share:Q", title="Share of values", axis=alt.Axis(format="%", minExtent=0),
                scale=alt.Scale(domain=[0, 1])),
        color=key.color(lines=True),
        tooltip=[alt.Tooltip("run:N", title="Run"), alt.Tooltip("seconds:Q", title="Seconds", format=".1f"),
                 alt.Tooltip("share:Q", title="Share", format=".0%")]).properties(width=WIDTH, height=220)


def by_field_type(per_field_type: pd.DataFrame, key: RunKey, mode: str) -> alt.FacetChart | None:
    """Final F1, completeness, precision and speed by field type: a small plot for each, the runs as bars of
    their colour under each other."""
    rows = key.frame(per_field_type)
    long = rows.melt(id_vars=["run", "field_type"], value_vars=list(BY_TYPE), var_name="measure",
                     value_name="value").dropna()
    if long.empty:
        return None
    long = long.assign(measure=long["measure"].map(BY_TYPE))
    order = rows.groupby("field_type")["completeness"].mean().sort_values(
        ascending=False, na_position="last").index.tolist()
    step = max(10, math.ceil(ROW / len(key.names)))  # the height of one run's bar within a field type
    bars = alt.Chart(long).mark_bar(size=min(BAR, step - 3), cornerRadiusEnd=3).encode(
        y=alt.Y("field_type:N", title=None, sort=order, scale=alt.Scale(paddingInner=0.25)),
        yOffset=alt.YOffset("run:N", sort=key.order),
        x=alt.X("value:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(values=[0, 0.5, 1])),
        color=key.color(),
        tooltip=[alt.Tooltip("run:N", title="Run"), alt.Tooltip("field_type:N", title="Field type"),
                 alt.Tooltip("measure:N", title="Measure"), alt.Tooltip("value:Q", title="Value", format=".3f")]).properties(
        width=SMALL + 30, height=alt.Step(step))  # with a y offset, the step is the height of one bar's band
    return bars.facet(column=alt.Column("measure:N", title=None, sort=list(BY_TYPE.values()),
                                        header=alt.Header(labelAnchor="start"))).properties(title=alt.Title(
        "By field type", subtitle=[
            "Final F1: the form at the end. Completeness: was each value shown? Precision: was what stood on screen",
            "right? Speed: how soon? Each from 0 to 1"]))


def outcomes_by_field_type(per_field_type: pd.DataFrame, key: RunKey, mode: str) -> alt.Chart | alt.FacetChart | None:
    """The final form by field type: how many values are correct, missing, invented or wrong, a plot per run."""
    rows = key.frame(per_field_type)
    long = rows.melt(id_vars=["run", "field_type"], value_vars=list(STATUS), var_name="outcome", value_name="values")
    long = long[long["values"] > 0]
    if long.empty:
        return None
    long = long.assign(outcome_name=long["outcome"].map(OUTCOME_NAMES),
                       order=long["outcome"].map({name: k for k, name in enumerate(STATUS)}))
    order = long.groupby("field_type")["values"].sum().sort_values(ascending=False).index.tolist()
    present = [name for name in key.order if name in set(long["run"])]
    title = alt.Title("Final form, by field type", subtitle="Values that should be filled or were filled; empty "
                                                            "fields left empty are not counted")
    bars = alt.Chart(long).mark_bar(size=BAR, stroke=CHROME[mode]["surface"], strokeWidth=2).encode(
        y=alt.Y("field_type:N", title=None, sort=order),
        x=alt.X("values:Q", title="Values", stack="zero"),
        color=alt.Color("outcome_name:N", title=None,
                        scale=alt.Scale(domain=list(OUTCOME_NAMES.values()), range=list(STATUS.values()))),
        order=alt.Order("order:Q"),
        tooltip=[alt.Tooltip("run:N", title="Run"), alt.Tooltip("field_type:N", title="Field type"),
                 alt.Tooltip("outcome_name:N", title="Outcome"), alt.Tooltip("values:Q", title="Values")]).properties(
        width=WIDTH if len(present) == 1 else max(SMALL, min(360, 760 // len(present))), height=alt.Step(ROW))
    if len(present) == 1:
        return bars.properties(title=title)
    return bars.facet(column=alt.Column("run:N", title=None, sort=present,
                                        header=alt.Header(labelAnchor="start"))).properties(title=title)


# ---- one conversation over time

def value_text(value: Any, field_type: FieldType) -> str:
    """A field's value as one line of text: options by commas, the facts or lines of a note one after another."""
    if isinstance(value, (list, tuple)):
        return (" " if field_type == FieldType.textarea else ", ").join(str(part) for part in value)
    return str(value)


def _clipped(value: Any, field_type: FieldType, limit: int) -> str:
    text = value_text(value, field_type)
    return text[:limit] + " …" if len(text) > limit else text


def timeline_rows(form: Form, walks: dict[str, dict[str, FieldWalk]], end: float) -> list[dict[str, Any]]:
    """The bars of a timeline. ``walks`` maps a line's name (a run, or "Shown") to the walks of one form (field →
    walk); every field also gets a "Should show" line, from the gold steps that all its walks share."""
    rows = []
    for spec in form.fields:
        mine = {name: by_field[spec.field_id] for name, by_field in walks.items() if spec.field_id in by_field}
        if not mine:
            continue

        def bar(lane: str, state: str, value: Any, start: float, stop: float, grade: float | None, spec=spec):
            return {"field": spec.label, "lane": lane, "state": state,
                    "value": _clipped(value, spec.type, _TOOLTIP_CHARACTERS), "full": value_text(value, spec.type),
                    "start": start, "end": stop, "grade": grade}

        steps = max((walk.steps for walk in mine.values()), key=len)
        for index, step in enumerate(steps):
            if step.value is not None:
                until = steps[index + 1].said if index + 1 < len(steps) else end
                rows.append(bar(SHOULD_SHOW, SHOULD_SHOW, step.value, step.said, until, None))
        for name, walk in mine.items():
            for display in walk.displays:
                if display.value is not None:
                    state = ("Shown: right" if display.grade == 1 else "Shown: close" if display.grade > 0
                             else "Shown: wrong")
                    rows.append(bar(name, state, display.value, display.shown, display.shown + display.seconds,
                                    display.grade))
    return rows


def timeline_marks(form: Form, changes: dict[str, dict[str, Sequence[Change]]]) -> list[dict[str, Any]]:
    """The marks of a timeline: one per visible change of a field. ``changes`` maps a line's name (a run, or
    "Shown") to the changes of one form (field → changes). A note is graded at most every few seconds (its
    bars), but every rewrite of it has its mark."""
    return [{"field": spec.label, "lane": name, "start": change.t, "change": CHANGE_NAMES[change.kind],
             "before": "—" if change.before is None else _clipped(change.before, spec.type, _MARK_CHARACTERS),
             "after": "—" if change.after is None else _clipped(change.after, spec.type, _MARK_CHARACTERS),
             "before_full": None if change.before is None else value_text(change.before, spec.type),
             "after_full": None if change.after is None else value_text(change.after, spec.type)}
            for spec in form.fields for name, by_field in changes.items() for change in by_field.get(spec.field_id, ())]


def timeline_chart(rows: Sequence[dict[str, Any]], marks: Sequence[dict[str, Any]], lanes: Sequence[str], title: str,
                   subtitle: str | list[str], end: float, mode: str, texts: list[str] | None = None) -> alt.LayerChart:
    """Per field, a line named after the field with what should be on screen (grey) and, under it, one line per
    name of ``lanes`` with what was (green = right, amber = close, red = wrong), and on it a mark at every change
    of the field (● first fill, ▲ revision, ✕ cleared). The time axis ends at ``end``. Not themed yet.

    A bar or a mark holds its value cut to a few hundred characters: enough for a plot's own tooltip. A page that
    shows values in full passes ``texts``: every value is then added to that list once, and each bar and mark
    carries the place of its texts in it (``value_ref``, ``before_ref``, ``after_ref``), for the page's window
    on a value (report/pages.py)."""
    c = CHROME[mode]
    places = {text: k for k, text in enumerate(texts)} if texts is not None else {}

    def held(row: dict[str, Any]) -> dict[str, Any]:
        """A bar or mark as the plot holds it: without its texts in full, with their places in ``texts``."""
        kept = {key: value for key, value in row.items() if not key.endswith("full")}
        for key, value in row.items() if texts is not None else ():
            if key.endswith("full") and value is not None:
                if value not in places:
                    places[value] = len(texts)
                    texts.append(value)
                kept[("value" if key == "full" else key.removesuffix("_full")) + "_ref"] = places[value]
        return kept

    def in_full(*names: str) -> list[alt.Tooltip]:
        return [alt.Tooltip(f"{name}_ref:Q", title=f"{name}_ref", format="d") for name in names] if texts is not None else []

    states = [SHOULD_SHOW, "Shown: right", "Shown: close", "Shown: wrong"]
    order = [SHOULD_SHOW, *lanes]
    period = len(order) + 1  # the lines of one field, and an empty one before the next field
    at = {name: k for k, name in enumerate(dict.fromkeys(row["field"] for row in rows))}
    labels = [label for name in at for label in (name, *lanes, "")][:-1]

    def line(row: dict[str, Any]) -> int:
        return at[row["field"]] * period + order.index(row["lane"])

    of_field = f"datum.value % {period} === 0"  # the line that says what the field should show carries its name
    y = alt.Y("line:O", title=None, scale=alt.Scale(domain=list(range(len(labels)))), axis=alt.Axis(
        labelExpr=f"{json.dumps(labels)}[datum.value]", ticks=False, labelPadding=8,
        labelColor={"condition": {"test": of_field, "value": c["primary"]}, "value": c["secondary"]},
        labelFontWeight={"condition": {"test": of_field, "value": 600}, "value": 400}))
    stretch = alt.Scale(domain=[0, end], nice=False)
    x = alt.X("start:Q", title="Seconds since the conversation started", scale=stretch)
    frame = pd.DataFrame([{**held(row), "line": line(row)} for row in rows])
    layers = [alt.Chart(frame).mark_bar(size=8, cornerRadius=2).encode(
        y=y, x=x, x2="end:Q",
        color=alt.Color("state:N", title=None, scale=alt.Scale(
            domain=states, range=[c["muted"], STATUS["correct"], STATUS["missing"], STATUS["wrong"]])),
        tooltip=[alt.Tooltip("field:N", title="Field"), alt.Tooltip("lane:N", title="Line"),
                 alt.Tooltip("state:N", title="State"), alt.Tooltip("value:N", title="Value"),
                 alt.Tooltip("grade:Q", title="Grade", format=".2f"),
                 alt.Tooltip("start:Q", title="From (s)", format=".1f"),
                 alt.Tooltip("end:Q", title="To (s)", format=".1f"), *in_full("value")])]
    placed = [{**held(mark), "line": line(mark)} for mark in marks
              if mark["field"] in at and mark["lane"] in order and mark["start"] <= end]
    if placed:
        layers.append(alt.Chart(pd.DataFrame(placed)).mark_point(
            size=24, filled=True, opacity=1, color=c["primary"], stroke=c["surface"], strokeWidth=0.8).encode(
            y=y, x=x,
            shape=alt.Shape("change:N", title=None,
                            scale=alt.Scale(domain=list(CHANGE_NAMES.values()), range=CHANGE_SHAPES),
                            legend=alt.Legend(symbolFillColor=c["primary"], symbolStrokeColor=c["surface"])),
            tooltip=[alt.Tooltip("field:N", title="Field"), alt.Tooltip("lane:N", title="Line"),
                     alt.Tooltip("change:N", title="Change"), alt.Tooltip("start:Q", title="At (s)", format=".1f"),
                     alt.Tooltip("before:N", title="Before"), alt.Tooltip("after:N", title="After"),
                     *in_full("before", "after")]))
    chart = alt.layer(*layers)
    if len(labels) > TOP_AXIS_FROM:  # a tall plot: the time axis on top too, from marks nobody sees (drawn first,
        # so that the bars and marks above them keep their tooltips)
        top = alt.Chart(frame).mark_bar(size=8, opacity=0).encode(
            y=y, x=alt.X("start:Q", title=None, scale=stretch, axis=alt.Axis(orient="top", grid=False)), x2="end:Q")
        chart = alt.layer(top, chart).resolve_axis(x="independent")
    return chart.properties(title=alt.Title(title, subtitle=subtitle), width=TIMELINE_WIDTH, height=alt.Step(LINE))


def conversation_timeline(run_dir: Path | str, conversation_id: str, config: Config, mode: str | None = None,
                          judge: Judge | None = None):
    """One conversation of one run, field by field, to look at in a notebook: what should be on screen (grey),
    what was (green = right, amber = close, red = wrong), and a mark at every change of a field.

    Covers the fields that go through the walk, graded as precision grades them. Free-text fields are included
    when a ``judge`` is given (their texts are judged again here). Returns a themed chart, or None when nothing
    was said or shown. The report of an evaluation has this plot on every form's page, for all its runs at once
    (report/pages.py).
    """
    mode = mode or config.plots.mode
    run, predictions, _ = read_run(Path(run_dir))
    dataset = load_dataset(config, run["dataset"], split=run["split"], ids=[conversation_id])
    conversation, form = dataset.conversations[conversation_id], dataset.form_of(conversation_id)
    stream_config = stream_config_of(run, config)
    arrival = arrival_of(conversation, stream_config)
    gold = in_extractor_time(dataset.gold[conversation_id], arrival)
    steps = gold_steps(form, gold)
    shown = shown_fields(predictions, conversation_id)
    end = scoring_end(steps, arrival.end, config.score.speed_limit_s[run["dataset"]])
    if run["dataset"] in FORM_READ_WHEN_RUN_ENDS:
        status = next((s for s in run["conversations"] if s["conversation"] == conversation_id), {})
        end = status.get("run_end") or arrival.end + stream_config.final_grace_s
    walks = walk_fields(form, gold, steps, shown, end, config, (lambda: TextMatch(judge)) if judge else None)
    rows = timeline_rows(form, {"Shown": walks}, end)
    if not rows:
        return None
    marks = timeline_marks(form, {"Shown": {field_id: field_changes(timeline) for field_id, timeline in shown.items()}})
    chart = timeline_chart(rows, marks, ["Shown"], f"{conversation_id}: the form over time",
                           "Per field: what it should show, from the moment its words reached the extractor, and "
                           "under it what was shown, with a mark at every change",
                           max([end] + [mark["start"] for mark in marks]), mode)
    return themed(chart, mode)
