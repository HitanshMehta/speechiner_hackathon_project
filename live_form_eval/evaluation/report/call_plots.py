"""The plots of what a run used: how many model calls, what they carried by kind of token, and what everything
cost, unit by unit (Altair).

Two places use them: a dataset's section of report.html, with all its runs in each plot, and a form's page, with
what every run used on that conversation. A run reports each use as an extraction attempt (a model call that may
change the form) or as another use (metrics/calls.py): the plots of model calls show the extraction attempts, the
plots of the price show every use. Colours and sizes come from report/style.py. No plot here is themed yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import altair as alt
import pandas as pd

from live_form_eval.evaluation.metrics.calls import KIND_UNITS, TOKEN_KINDS, UNSPLIT, Call, unit_order
from live_form_eval.evaluation.report.style import BAR, CHROME, PARTS, ROW, WIDTH, Measure, RunKey, small_multiples

TOKEN_NAMES = {"cache_read": "Read from cache", "fresh_input": "Fresh input", "cache_write": "Written to cache",
               "output": "Output"}
UNIT_NAMES = {**{KIND_UNITS[kind]: name for kind, name in TOKEN_NAMES.items()}, UNSPLIT: "Not split by unit"}
CALLS_SO_FAR, CHANGES_SO_FAR = "Model calls so far", "Value changes so far"
BINS = 16  # tokens over a conversation: about this many points per run …
_BIN_SECONDS = (10, 15, 20, 30, 60, 90, 120, 180, 300, 600)  # … each over one of these stretches
CALL_PANEL = 110  # height of one run's plot of tokens per call
PRICE_PANEL = 150  # … and of its price so far
_DASHES = ([6, 3], [2, 2], [8, 3, 2, 3], [4, 2, 1, 2])  # the lines of units other than tokens


def unit_name(unit: str) -> str:
    """A unit as the report names it: the kinds of tokens in words, any other unit by its name."""
    return UNIT_NAMES.get(unit, unit.replace("_", " "))


def unit_styles(units: Sequence[str], mode: str) -> dict[str, tuple[str, list[int]]]:
    """Each unit's grey and dash pattern. The four kinds of tokens keep the greys of the token plots (the cheapest
    faint, the dearest strong) and a solid line; another unit takes the strongest grey and a dashed line; a price
    that is not split by unit, a dotted one."""
    greys = PARTS[mode]
    places = {KIND_UNITS[kind]: k for k, kind in enumerate(TOKEN_KINDS)}
    styles, others = {}, 0
    for unit in units:
        if unit in places:
            styles[unit] = (greys[places[unit]], [1, 0])
        elif unit == UNSPLIT:
            styles[unit] = (greys[2], [1, 3])
        else:
            styles[unit] = (greys[3], _DASHES[others % len(_DASHES)])
            others += 1
    return styles


def _amount(amount: float) -> str:
    return f"{amount:,.0f}" if float(amount).is_integer() else f"{amount:,.2f}"


def _kinds(mode: str) -> alt.Color:
    """The kind of token as a grey, from the cheapest (faint) to the dearest (strong)."""
    return alt.Color("kind:N", title=None, scale=alt.Scale(domain=list(TOKEN_NAMES.values()), range=PARTS[mode]))


# ---- a dataset's section: all its runs in one plot

def calls_by_run(summary: pd.DataFrame, key: RunKey, mode: str) -> alt.HConcatChart | None:
    """The runs next to each other: how often each calls a model, how long a call takes, what it carries and
    what a minute of audio costs. ``summary`` has one row per run."""
    return small_multiples(summary, [
        Measure("calls_per_audio_minute", "Calls per audio minute", ".1f"),
        Measure("calls_changing_share", "… that changed a value", ".0%", (0, 1)),
        Measure("call_seconds_median", "A call takes, median (s)", ".2f"),
        Measure("tokens_per_call", "Tokens per call", ",.0f"),
        Measure("cached_input_share", "Input read from cache", ".0%", (0, 1)),
        Measure("eur_per_audio_minute", "Price per audio minute (€)", ".4f"),
    ], key, "Model calls and price", [
        "How often a run calls a model to fill the form, how many of those calls changed a value (the others cost",
        "the same and changed nothing), how long a call takes, what it carries, and what a minute of audio costs"],
        mode)


def token_mix(summary: pd.DataFrame, key: RunKey, mode: str) -> alt.Chart | None:
    """Where the tokens of a form go, by kind: one stacked bar per run."""
    rows = key.frame(summary)
    columns = {f"{kind}_tokens_per_form": kind for kind in TOKEN_KINDS}
    present = [column for column in columns if column in rows.columns]
    long = rows.melt(id_vars=["run"], value_vars=present, var_name="kind", value_name="tokens").dropna()
    long = long[long["tokens"] > 0]
    if long.empty:
        return None
    long = long.assign(order=long["kind"].map({column: k for k, column in enumerate(columns)}),
                       kind=long["kind"].map(lambda column: TOKEN_NAMES[columns[column]]))
    return alt.Chart(long, title=alt.Title("Where the tokens go", subtitle=[
        "Tokens per form by kind, from the cheapest (read from the vendor's cache) to the dearest (output), of the",
        "model calls that tried to fill the form. Input grows when the transcript is sent again with every call"])).mark_bar(
        size=BAR, stroke=CHROME[mode]["surface"], strokeWidth=2).encode(
        y=alt.Y("run:N", title=None, scale=alt.Scale(domain=key.order)),
        x=alt.X("tokens:Q", title="Tokens per form", stack="zero", axis=alt.Axis(format="~s")),
        color=_kinds(mode), order=alt.Order("order:Q"),
        tooltip=[alt.Tooltip("run:N", title="Run"), alt.Tooltip("kind:N", title="Kind"),
                 alt.Tooltip("tokens:Q", title="Tokens per form", format=",.0f")]).properties(
        width=WIDTH, height=alt.Step(ROW))


def price_by_unit(calls: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """The price of a form, unit by unit: one row per run and unit (run, unit, eur_per_form). ``calls`` has one row
    per use with a column "eur_<unit>" per unit; ``summary`` one row per run with its number of conversations."""
    columns = [column for column in calls.columns if column.startswith("eur_")]
    if calls.empty or not columns:
        return pd.DataFrame(columns=["run", "unit", "eur_per_form"])
    spent = calls.groupby("run")[columns].sum(min_count=1)
    forms = summary.set_index("run")["conversations"]
    rows = [{"run": run, "unit": column.removeprefix("eur_"), "eur_per_form": eur / forms[run]}
            for run, row in spent.iterrows() for column, eur in row.items()
            if pd.notna(eur) and eur > 0 and run in forms.index and forms[run]]
    order = {unit: k for k, unit in enumerate(unit_order(row["unit"] for row in rows))}
    return pd.DataFrame(rows, columns=["run", "unit", "eur_per_form"]).sort_values(
        "unit", key=lambda units: units.map(order), kind="stable").reset_index(drop=True)


def price_mix(units: pd.DataFrame, key: RunKey, mode: str) -> alt.Chart | None:
    """Where the money goes: the price of a form by unit, one stacked bar per run. ``units`` comes from
    ``price_by_unit``."""
    rows = key.frame(units)
    rows = rows[rows["eur_per_form"] > 0]
    if rows.empty:
        return None
    order = unit_order(rows["unit"])
    styles = unit_styles(order, mode)
    rows = rows.assign(name=rows["unit"].map(unit_name), order=rows["unit"].map({unit: k for k, unit in enumerate(order)}))
    return alt.Chart(rows, title=alt.Title("Where the money goes", subtitle=[
        "The price of a form by unit, from every use a run reported: its model calls that tried to fill the form",
        "and its other uses. The kinds of tokens go from the cheapest (faint) to the dearest (strong)"])).mark_bar(
        size=BAR, stroke=CHROME[mode]["surface"], strokeWidth=2).encode(
        y=alt.Y("run:N", title=None, scale=alt.Scale(domain=key.order)),
        x=alt.X("eur_per_form:Q", title="Price per form (€)", stack="zero", axis=alt.Axis(format=".4f")),
        color=alt.Color("name:N", title=None, scale=alt.Scale(domain=[unit_name(unit) for unit in order],
                                                              range=[styles[unit][0] for unit in order])),
        order=alt.Order("order:Q"),
        tooltip=[alt.Tooltip("run:N", title="Run"), alt.Tooltip("name:N", title="Unit"),
                 alt.Tooltip("eur_per_form:Q", title="Price per form (€)", format=".5f")]).properties(
        width=WIDTH, height=alt.Step(ROW))


def tokens_over_conversation(calls: pd.DataFrame, key: RunKey, mode: str) -> alt.LayerChart | None:
    """Input tokens per model call as a conversation goes on: one line per run, each point the mean over a stretch
    of all its conversations. ``calls`` has one row per use (run, conversation, t, extraction, all_input_tokens)."""
    rows = key.frame(calls)
    rows = rows[rows["extraction"].astype(bool) & (rows["all_input_tokens"] > 0)]
    if rows.empty:  # a call reported without its tokens says nothing here
        return None
    longest = float(rows["t"].max())
    stretch = next((seconds for seconds in _BIN_SECONDS if longest / seconds <= BINS), _BIN_SECONDS[-1])
    rows = rows.assign(minutes=((rows["t"] // stretch) * stretch + stretch / 2) / 60)
    points = rows.groupby(["run", "minutes"], as_index=False).agg(
        tokens=("all_input_tokens", "mean"), calls=("t", "size"), conversations=("conversation", "nunique"))
    x = alt.X("minutes:Q", title="Minutes into the conversation", axis=alt.Axis(tickCount=8))
    y = alt.Y("tokens:Q", title="Input tokens per call", axis=alt.Axis(format="~s", minExtent=0))
    base = alt.Chart(points)
    line = base.mark_line(strokeWidth=2).encode(x=x, y=y, color=key.color(lines=True))
    dots = base.mark_point(size=40, filled=True, opacity=1, stroke=CHROME[mode]["surface"], strokeWidth=1).encode(
        x=x, y=y, color=key.color(lines=True),
        tooltip=[alt.Tooltip("run:N", title="Run"), alt.Tooltip("minutes:Q", title="Minutes in", format=".1f"),
                 alt.Tooltip("tokens:Q", title="Input tokens per call", format=",.0f"),
                 alt.Tooltip("calls:Q", title="Calls in this stretch"),
                 alt.Tooltip("conversations:Q", title="Conversations still running")])
    return (line + dots).properties(width=WIDTH, height=220, title=alt.Title(
        "What a call sends, as the conversation goes on", subtitle=[
            f"Input tokens per model call, the mean over each {stretch:g} s of all conversations. A line that climbs",
            "is a prompt that grows with the transcript: every later call costs more than the one before"]))


def price_against_length(per_conversation: pd.DataFrame, key: RunKey, mode: str) -> alt.Chart | None:
    """Every form as a point: what it cost against how long the conversation is. With a column "page", a point
    leads to the form's page."""
    rows = key.frame(per_conversation)
    rows = rows[rows["eur"] > 0]
    if rows.empty:
        return None
    rows = rows.assign(minutes=rows["audio_s"] / 60)
    linked = "page" in rows.columns
    mark = {"cursor": "pointer"} if linked else {}
    link = {"href": alt.Href("page:N")} if linked else {}
    return alt.Chart(rows, title=alt.Title("Price of a form against its length", subtitle=[
        "Every form is a point" + (": click one to open its page" if linked else "") + ". A price that grows faster",
        "than the length is a run whose every call sends the whole conversation again"])).mark_point(
        size=60, filled=True, opacity=0.85, stroke=CHROME[mode]["surface"], strokeWidth=1, **mark).encode(
        x=alt.X("minutes:Q", title="Minutes of audio", axis=alt.Axis(tickCount=8)),
        y=alt.Y("eur:Q", title="Price of the form (€)", axis=alt.Axis(format=".3f", minExtent=0)),
        color=key.color(),
        tooltip=[alt.Tooltip("run:N", title="Run"), alt.Tooltip("conversation:N", title="Form"),
                 alt.Tooltip("minutes:Q", title="Minutes of audio", format=".1f"),
                 alt.Tooltip("calls:Q", title="Model calls"), alt.Tooltip("other_uses:Q", title="Other uses"),
                 alt.Tooltip("eur:Q", title="Price (€)", format=".4f")], **link).properties(width=WIDTH, height=240)


# ---- a form's page: what every run used on one conversation

def _call_rows(runs: dict[str, Sequence[Call]]) -> pd.DataFrame:
    """One row per use; ``runs`` maps a run's name to its uses on the conversation, in time order."""
    rows: list[dict[str, Any]] = []
    for name, calls in runs.items():
        so_far, attempts = 0.0, 0
        for number, call in enumerate(calls, start=1):
            so_far += call.eur
            attempts += call.extraction
            parts = " · ".join(f"{unit_name(unit)}"
                               + (f" {_amount(call.usage[unit])}" if unit in call.usage else "")
                               + f" (€{call.eur_by_unit.get(unit, 0.0):.6f})"
                               for unit in unit_order({*call.usage, *call.eur_by_unit}))
            rows.append({"run": name, "use": number, "call": attempts if call.extraction else None,
                         "what": "model call to fill the form" if call.extraction else "other use",
                         "extraction": call.extraction, "model": call.model, "t": call.t, "eur": call.eur,
                         "eur_so_far": so_far, "seconds": call.seconds, "changes": call.changes, "units": parts or "—",
                         "input": call.input, **{kind: call.tokens(kind) for kind in TOKEN_KINDS}})
    return pd.DataFrame(rows)


def _seconds(end: float, title: bool = True) -> alt.X:
    """The time axis of a form's page: the same stretch in every plot, so that they line up under each other."""
    return alt.X("t:Q", title="Seconds since the conversation started" if title else None,
                 scale=alt.Scale(domain=[0, end], nice=False))


def _so_far(calls: Sequence[Call], end: float) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Every unit's price so far at each use of one run, from 0 at the start to its last value at ``end``: one row
    per unit and moment (unit, t, eur_so_far). Also the run's price by unit at the end."""
    units = unit_order({unit for call in calls for unit in call.eur_by_unit})
    spent = dict.fromkeys(units, 0.0)
    rows = [{"unit": unit, "t": 0.0, "eur_so_far": 0.0} for unit in units]
    for call in calls:
        for unit, eur in call.eur_by_unit.items():
            spent[unit] += eur
        rows += [{"unit": unit, "t": call.t, "eur_so_far": spent[unit]} for unit in units]
    rows += [{"unit": unit, "t": max(end, calls[-1].t), "eur_so_far": spent[unit]} for unit in units] if calls else []
    return rows, spent


def price_so_far(runs: dict[str, Sequence[Call]], key: RunKey, end: float, width: int, mode: str) -> alt.VConcatChart | None:
    """What each run has spent on the form so far, one plot per run under each other: the whole price (a line in
    the run's colour, a mark per use) and the price of each unit (grey lines). None when nothing had a price."""
    frame = _call_rows(runs)
    if frame.empty or not frame["eur"].any():
        return None
    present = [name for name in key.order if name in runs and runs[name] and any(call.eur for call in runs[name])]
    top = max(float(frame[frame["run"] == name]["eur_so_far"].max()) for name in present)  # one scale: they compare
    every = unit_order({unit for name in present for call in runs[name] for unit in call.eur_by_unit})
    styles = unit_styles(every, mode)
    names = [unit_name(unit) for unit in every]
    y = alt.Y("eur_so_far:Q", title=None, scale=alt.Scale(domain=[0, top]), axis=alt.Axis(format=".4f"))
    unit_color = alt.Color("name:N", title=None, scale=alt.Scale(domain=names, range=[styles[unit][0] for unit in every]),
                           legend=alt.Legend(symbolType="stroke", symbolStrokeWidth=2))
    unit_dash = alt.StrokeDash("name:N", title=None, scale=alt.Scale(domain=names, range=[styles[unit][1] for unit in every]))
    tooltip = [alt.Tooltip("run:N", title="Run"), alt.Tooltip("use:Q", title="Use"), alt.Tooltip("what:N", title="What"),
               alt.Tooltip("model:N", title="Model"), alt.Tooltip("t:Q", title="Reported at (s)", format=".1f"),
               alt.Tooltip("seconds:Q", title="It took (s)", format=".2f"),
               alt.Tooltip("changes:Q", title="Values it changed"), alt.Tooltip("units:N", title="Its units"),
               alt.Tooltip("eur:Q", title="Price of this use (€)", format=".6f"),
               alt.Tooltip("eur_so_far:Q", title="Price so far (€)", format=".5f")]
    panels = []
    for name in present:
        calls, colour, mine = runs[name], key.colors[name], frame[frame["run"] == name]
        rows, spent = _so_far(calls, end)
        lines = pd.DataFrame(rows).assign(run=name)
        lines = lines.assign(name=lines["unit"].map(unit_name))
        whole = pd.concat([pd.DataFrame([{"t": 0.0, "eur_so_far": 0.0}]), mine[["t", "eur_so_far"]],
                           pd.DataFrame([{"t": max(end, float(mine["t"].max())), "eur_so_far": float(mine["eur_so_far"].max())}])],
                          ignore_index=True)
        x = _seconds(end, title=name == present[-1])
        layers = [alt.Chart(lines).mark_line(strokeWidth=1.6, interpolate="step-after").encode(
                      x=x, y=y, color=unit_color, strokeDash=unit_dash,
                      tooltip=[alt.Tooltip("run:N", title="Run"), alt.Tooltip("name:N", title="Unit"),
                               alt.Tooltip("t:Q", title="At (s)", format=".1f"),
                               alt.Tooltip("eur_so_far:Q", title="Its price so far (€)", format=".6f")]),
                  alt.Chart(whole).mark_line(strokeWidth=2.5, interpolate="step-after", color=colour).encode(x=x, y=y)]
        for chosen, mark in ((mine[mine["extraction"] & (mine["changes"] == 0)],
                              {"shape": "circle", "size": 26, "filled": False, "strokeWidth": 1.2,
                               "fill": CHROME[mode]["surface"]}),
                             (mine[mine["extraction"] & (mine["changes"] > 0)],
                              {"shape": "circle", "size": 42, "filled": True, "stroke": CHROME[mode]["surface"],
                               "strokeWidth": 1}),
                             (mine[~mine["extraction"]],
                              {"shape": "square", "size": 30, "filled": False, "strokeWidth": 1.2,
                               "fill": CHROME[mode]["surface"]})):
            if not chosen.empty:
                layers.append(alt.Chart(chosen).mark_point(opacity=1, color=colour, **mark).encode(
                    x=x, y=y, tooltip=tooltip))
        parts = ", ".join(f"{unit_name(unit).lower()} €{eur:.4f}" for unit, eur in spent.items() if eur)
        panels.append(alt.layer(*layers).properties(width=width, height=PRICE_PANEL, title=alt.Title(
            f"{name}: €{sum(spent.values()):.4f}" + (f" ({parts})" if parts else ""), fontSize=12, fontWeight=500,
            color=CHROME[mode]["primary"], anchor="start")))
    return alt.vconcat(*panels, spacing=14).properties(title=alt.Title("Price so far (€), unit by unit", subtitle=[
        "The coloured line is everything a run has spent so far; each grey line, what one unit has cost so far. A mark",
        "per use: ● a model call that changed a value of the form, ○ one that changed nothing, □ another use"]))


def tokens_per_call(runs: dict[str, Sequence[Call]], key: RunKey, end: float, width: int, mode: str) -> alt.VConcatChart | None:
    """The tokens of every model call by kind, one plot per run under each other."""
    frame = _call_rows(runs)
    if frame.empty:
        return None
    frame = frame[frame["extraction"]]
    long = frame.melt(id_vars=["run", "call", "t"], value_vars=list(TOKEN_KINDS), var_name="kind", value_name="tokens")
    long = long[long["tokens"] > 0]
    if long.empty:
        return None
    long = long.assign(order=long["kind"].map({kind: k for k, kind in enumerate(TOKEN_KINDS)}),
                       kind=long["kind"].map(TOKEN_NAMES))
    present = [name for name in key.order if name in set(long["run"])]
    top = float(long.groupby(["run", "call"])["tokens"].sum().max())  # one scale for all runs: they compare
    panels = [alt.Chart(long[long["run"] == name]).mark_bar(size=3).encode(
        x=_seconds(end, title=name == present[-1]),
        y=alt.Y("tokens:Q", title=None, stack="zero", scale=alt.Scale(domain=[0, top]), axis=alt.Axis(format="~s")),
        color=_kinds(mode), order=alt.Order("order:Q"),
        tooltip=[alt.Tooltip("run:N", title="Run"), alt.Tooltip("call:Q", title="Call"),
                 alt.Tooltip("t:Q", title="Reported at (s)", format=".1f"), alt.Tooltip("kind:N", title="Kind"),
                 alt.Tooltip("tokens:Q", title="Tokens", format=",.0f")]).properties(
        width=width, height=CALL_PANEL,
        title=alt.Title(name, fontSize=12, fontWeight=500, color=CHROME[mode]["primary"], anchor="start"))
        for name in present]
    return alt.vconcat(*panels, spacing=14).properties(title=alt.Title("Tokens per call, by kind", subtitle=[
        "A bar per model call that tried to fill the form. Fresh input that grows from call to call is the transcript",
        "sent again; once the vendor caches the repeated part, it moves to “read from cache”, the cheap kind"]))


def calls_and_changes(runs: dict[str, Sequence[Call]], changes: dict[str, Sequence[float]], key: RunKey, end: float,
                      width: int, mode: str) -> alt.Chart | None:
    """Per run: the model calls made so far to fill the form (dashed) and the value changes published so far
    (solid). ``changes`` maps a run's name to the times its fields changed."""
    rows: list[dict[str, Any]] = []
    for line, moments in ((CALLS_SO_FAR, {name: [call.t for call in calls if call.extraction]
                                          for name, calls in runs.items()}),
                          (CHANGES_SO_FAR, changes)):
        for name, times in moments.items():
            times = sorted(times)
            if times:  # from zero at the start, flat from the last one to the end
                rows += [{"run": name, "line": line, "t": t, "count": count}
                         for count, t in enumerate([0.0, *times, max(end, times[-1])])]
                rows[-1]["count"] = len(times)
    if not rows:
        return None
    dashes = alt.Scale(domain=[CHANGES_SO_FAR, CALLS_SO_FAR], range=[[1, 0], [4, 3]])
    return alt.Chart(pd.DataFrame(rows), title=alt.Title("Model calls and value changes so far", subtitle=[
        "Per run: the model calls it made to fill the form (dashed) and the changes of a field it published (solid):",
        "a field filled, revised or cleared. Where the dashed line climbs and the solid one stays flat, calls changed nothing"])).mark_line(
        strokeWidth=2, interpolate="step-after").encode(
        x=_seconds(end), y=alt.Y("count:Q", title=None), color=key.color(lines=True),
        strokeDash=alt.StrokeDash("line:N", title=None, scale=dashes, legend=alt.Legend(symbolType="stroke")),
        tooltip=[alt.Tooltip("run:N", title="Run"), alt.Tooltip("line:N", title="Line"),
                 alt.Tooltip("t:Q", title="At (s)", format=".1f"), alt.Tooltip("count:Q", title="So far")]).properties(
        width=width, height=220)
