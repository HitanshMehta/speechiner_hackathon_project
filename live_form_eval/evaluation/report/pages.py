"""The HTML report of an evaluation: report.html for the runs together, and one page per form.

``report.html`` holds the ranking and the score against the price, then one section per dataset with all its runs
in the same tables and plots (scores, field types, model calls and tokens, value changes), and the table of the
forms. Under ``forms/`` every conversation of the evaluated runs has a page of its own (a hundred forms in one
file would be too heavy): how each run scored on it, a timeline of what every field should have shown and what
each run showed, the model calls each run made on it, the form at the end, every field's values over time, the
conversation as the extractor received it (turn by turn, or message by message) with the gold said and the values
each run filled at each line, and the form itself (its rules and all its fields). A bar at the top of every page
switches from form to form. A run keeps one colour on every page (report/style.py).

A value that a plot or a table cuts short opens in full in a window when the pointer rests on it; the window
stays open while the pointer is inside it, and it scrolls.

All text from outside (values, notes, transcripts) is escaped; chart specs are embedded as JSON.
"""

from __future__ import annotations

import html
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd

from live_form_eval.config import Config, StreamConfig
from live_form_eval.data.common.model import Conversation, Gold
from live_form_eval.data.common.speakers import speaker_letters
from live_form_eval.evaluation.evaluate import ConversationScore, RunScore
from live_form_eval.evaluation.matching.compare import SCALAR_TYPES
from live_form_eval.evaluation.matching.free_text import TextMatch, units_of
from live_form_eval.evaluation.metrics.calls import TOKEN_KINDS, call_stats, unit_order
from live_form_eval.evaluation.metrics.completeness import field_completeness
from live_form_eval.evaluation.metrics.final_decision_latency import final_decision_latency
from live_form_eval.evaluation.metrics.final_f1 import final_f1
from live_form_eval.evaluation.metrics.precision import field_precision
from live_form_eval.evaluation.metrics.speed import curve, field_speed
from live_form_eval.evaluation.metrics.value_changes import value_changes
from live_form_eval.evaluation.report import call_plots
from live_form_eval.evaluation.report.plots import (
    TIMELINE_WIDTH,
    by_field_type,
    changes_by_run,
    latency_curves,
    outcomes_by_field_type,
    run_key,
    run_labels,
    score_against_price,
    terms_by_run,
    timeline_chart,
    timeline_marks,
    timeline_rows,
    value_text,
)
from live_form_eval.evaluation.report.style import (
    CHROME,
    DATASET_NAMES,
    FONT,
    STATUS,
    RunKey,
    missing,
    run_colors,
    themed,
)
from live_form_eval.evaluation.timeline.arrival import stream_config_of
from live_form_eval.evaluation.timeline.final import Outcome
from live_form_eval.evaluation.timeline.free_text import gold_facts
from live_form_eval.harness.stream import build_stream
from live_form_eval.interface.schema import FieldSpec, FieldType, Form

FORMS_DIR = "forms"
_CELL_CHARACTERS = 160  # a value longer than this is cut in a cell of the conversation's table (its window has it all)
_TAIL_CHARACTERS = 90  # how much of a turn's earlier words a partial's line repeats before its new words
Columns = Sequence[tuple[str, str, str | None]]  # of a table: the column, its header, how its numbers are written
RANKING: Columns = (
    ("datasets", "Datasets", None), ("score", "Score", "{:.3f}"), ("score_spokenwoz", "… on SpokenWOZ", "{:.3f}"),
    ("score_ami", "… on AMI", "{:.3f}"), ("completeness", "Completeness", "{:.3f}"),
    ("precision", "Precision", "{:.3f}"), ("speed", "Speed", "{:.3f}"), ("outcome", "Outcome", "{:.3f}"),
    ("eur_per_form_spokenwoz", "Price per form, SpokenWOZ (€)", "{:.4f}"),
    ("eur_per_form_ami", "Price per form, AMI (€)", "{:.4f}"), ("official", "Official", None))
CHANGES: Columns = (
    ("fields_filled_per_form", "Fields filled per form", "{:.1f}"),
    ("value_changes_per_form", "Value changes per form", "{:.1f}"),
    ("revisions_per_filled_field", "Revisions per filled field", "{:.2f}"),
    ("filled_once_share", "Filled once, never changed", "{:.0%}"))
_OUTCOME_WORDS = {"correct": "correct", "wrong": "wrong", "missing": "missing", "invented": "invented",
                  "correctly_empty": "correctly empty"}
PIECE_TYPES = (FieldType.textarea, FieldType.multiselect)  # fields that grow piece by piece: facts, lines, options
# A link that opens or folds every field of the page at once: .format("true", "Open all").
_FOLD = ("<a href='#fields' onclick=\"document.querySelectorAll('details[id^=field-]').forEach(d => d.open = {}); "
         "return false\">{}</a>")
_SCRIPT = """
document.querySelectorAll('table.sortable th').forEach(th => th.addEventListener('click', () => {
  const k = th.cellIndex, body = th.closest('table').tBodies[0];
  const up = th.dataset.up = th.dataset.up === '1' ? '0' : '1';
  const key = row => { const text = row.cells[k].dataset.sort ?? row.cells[k].textContent;
                       const number = parseFloat(text); return isNaN(number) ? text : number; };
  [...body.rows].sort((a, b) => { const x = key(a), y = key(b);
                                  return (x > y ? 1 : x < y ? -1 : 0) * (up === '1' ? 1 : -1); })
                .forEach(row => body.appendChild(row));
}));
const reveal = () => {  // a link to a folded block (a field, the transcript) opens it
  const target = location.hash && document.getElementById(decodeURIComponent(location.hash.slice(1)));
  if (target && target.tagName === 'DETAILS') { target.open = true; target.scrollIntoView(); }
};
window.addEventListener('hashchange', reveal); reveal();
// The window on a value (a bar or mark of a plot, a value cut short in a table): it holds the whole value, it stays
// open while the pointer is on the value or inside the window, and it scrolls. So a long note can be read in full.
const tip = document.createElement('div');
tip.className = 'tip'; tip.hidden = true; document.body.append(tip);
let tipOf = null, tipTimer = null;
const closeTip = () => { clearTimeout(tipTimer); tipTimer = setTimeout(() => { tip.hidden = true; tipOf = null; }, 320); };
const openTip = (owner, rows, x, y) => {
  clearTimeout(tipTimer);
  if (owner === tipOf) return;  // the same value: the window stays where it is, so that the pointer can reach it
  const put = () => {
    tipOf = owner;
    const table = document.createElement('table');
    for (const [name, text] of rows) { const row = table.insertRow(); row.insertCell().textContent = name; row.insertCell().textContent = text; }
    tip.replaceChildren(table); tip.hidden = false; tip.scrollTop = 0;
    tip.style.left = Math.max(8, Math.min(x + 12, innerWidth - tip.offsetWidth - 8)) + 'px';
    tip.style.top = Math.max(8, Math.min(y + 12, innerHeight - tip.offsetHeight - 8)) + 'px';
  };
  if (tip.hidden) put(); else tipTimer = setTimeout(put, 200);  // another value crossed on the way into the window
};
tip.addEventListener('mouseenter', () => clearTimeout(tipTimer));
tip.addEventListener('mouseleave', closeTip);
const inFull = {Value: 'value_ref', Before: 'before_ref', After: 'after_ref'};  // where a plot's tooltip names a text
const chartTip = (handler, event, item, value) => {
  if (value === null || value === undefined || value === '') { closeTip(); return; }
  const empty = ['null', 'NaN', 'undefined', ''];  // a line of the plot's tooltip that has nothing to say
  const rows = Object.entries(value).filter(([name, text]) => !name.endsWith('_ref') && !empty.includes(String(text))).map(([name, text]) => {
    const whole = texts[Number(value[inFull[name]])];
    return [name, whole === undefined ? String(text) : whole];
  });
  openTip(item, rows, event.clientX, event.clientY);
};
document.addEventListener('mouseover', event => {
  const owner = event.target instanceof Element && event.target.closest('[data-ref]');
  if (owner) openTip(owner, [[owner.dataset.name || '', texts[Number(owner.dataset.ref)] || '']], event.clientX, event.clientY);
  else if (tipOf instanceof Element && !tip.contains(event.target)) closeTip();
});
document.querySelectorAll('[data-view-of]').forEach(button => button.addEventListener('click', () => {  // two views of one table
  const group = button.dataset.viewOf;
  document.querySelectorAll(`[data-view-of="${group}"]`).forEach(other => other.setAttribute('aria-pressed', String(other === button)));
  document.querySelectorAll(`[data-view-in="${group}"]`).forEach(view => { view.hidden = view.dataset.view !== button.dataset.view; });
}));"""


@dataclass(frozen=True)
class FormEntry:
    """One conversation of the evaluation, with every run that covers it."""

    dataset: str
    conversation: str
    runs: tuple[tuple[str, RunScore, ConversationScore], ...]  # (the run's name on the pages, the run, its score)

    @property
    def page(self) -> str:
        return f"{self.dataset}-{self.conversation}.html"

    @property
    def lowest(self) -> float | None:
        """The lowest score a run got on this form: what the selector shows next to its id."""
        scores = [item.terms.score for _, _, item in self.runs if item.terms.score is not None]
        return min(scores) if scores else None


@dataclass(frozen=True)
class GoldEntry:
    """One thing the gold says a field should show, as the dataset times it (audio time)."""

    spoken: float
    source: str  # where that time comes from (data/common/model.py)
    spec: FieldSpec
    text: str | None  # None = the value is withdrawn


def run_names(scores: list[RunScore]) -> list[str]:
    """How each run is called on the pages: its name, with its start time when two runs share name and dataset."""
    return list(run_labels((score.run["run_id"], score.run["name"], score.dataset) for score in scores).values())


def form_entries(scores: list[RunScore]) -> list[FormEntry]:
    """Every conversation of the runs, by dataset (in the runs' order) and id."""
    order = list(dict.fromkeys(score.dataset for score in scores))
    found: dict[tuple[str, str], list[tuple[str, RunScore, ConversationScore]]] = {}
    for name, score in zip(run_names(scores), scores):
        for item in score.conversations:
            found.setdefault((score.dataset, item.conversation), []).append((name, score, item))
    return [FormEntry(dataset, conversation, tuple(runs))
            for (dataset, conversation), runs in sorted(found.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1]))]


# ---- small pieces

def _esc(value: Any) -> str:
    return html.escape(str(value))


def _number(value: Any, style: str) -> str:
    if missing(value):
        return "—"
    return "∞" if isinstance(value, float) and math.isinf(value) else style.format(value)


def _plain(value: Any) -> str:
    """A cell of the table of all numbers."""
    if missing(value):
        return "—"
    if isinstance(value, float):
        return "∞" if math.isinf(value) else f"{value:.4f}"
    return str(value)


def _td(value: Any, style: str | None) -> str:
    """A cell of a sortable table: a number as ``style`` writes it, sorted by its value; anything else as text."""
    if missing(value):
        return "<td>—</td>"
    if style is None or pd.api.types.is_bool(value):
        return f"<td class='t'>{_esc('yes' if value is True else 'no' if value is False else value)}</td>"
    if isinstance(value, float) and math.isinf(value):
        return "<td data-sort='1e99'>∞</td>"
    return f"<td data-sort='{value}'>{_esc(style.format(value))}</td>"


def _run(name: str, colors: dict[str, str]) -> str:
    """A run's name behind a square of its colour (none for a name that is not a run of the report). The square
    is a character and the space after it does not break, so it never stands alone at the end of a line."""
    dot = f"<span class='dot' style='color:{colors[name]}'>■</span>&nbsp;" if name in colors else ""
    return dot + _esc(name)


def _key(names: Sequence[str], colors: dict[str, str]) -> str:
    """The runs of a page with their colours: the legend of every plot and table on it."""
    return "<p class='key'>" + "".join(f"<span>{_run(name, colors)}</span>" for name in dict.fromkeys(names)) + "</p>"


def _value(value: Any, field_type: FieldType) -> str:
    """A field's value as HTML: a note or its facts line by line, options by commas, nothing as a dash."""
    if value is None or value == "" or value == ():
        return "<span class='none'>—</span>"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if field_type == FieldType.textarea:
        lines = [str(part) for part in value] if isinstance(value, (list, tuple)) else str(value).splitlines()
        return "<ul>" + "".join(f"<li>{_esc(line)}</li>" for line in lines if line.strip()) + "</ul>"
    if isinstance(value, (list, tuple)):
        return _esc(", ".join(str(part) for part in value))
    return _esc(value)


def _pieces(value: Any, spec: FieldSpec, is_list: bool) -> list[str]:
    """The pieces a note or a multiselect holds: its facts, items or sentences (what the judge reads), or its
    options."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(part) for part in value]
    return units_of(value, is_list) if spec.type == FieldType.textarea else [str(value)]


def _change(before: list[str], now: list[str]) -> str:
    """What a gold step or a display changes in a field of pieces: what it adds (+) and what it drops (−).
    Writing out the whole note at every change would bury the change."""
    added = [piece for piece in now if piece not in before]
    gone = [piece for piece in before if piece not in now]
    if not added and not gone:
        return f"<span class='none'>{'the same as before' if now else 'empty'}</span>"
    items = ("".join(f"<li class='add'>{_esc(piece)}</li>" for piece in added)
             + "".join(f"<li class='gone'>{_esc(piece)}</li>" for piece in gone))
    kept = len(now) - len(added)
    rest = f"{kept} more, unchanged" if kept else "" if now else "nothing is left"
    return f"<ul class='diff'>{items}</ul>" + (f"<div class='sub'>{rest}</div>" if rest else "")


def _chips(outcomes: list[Outcome]) -> str:
    """The outcomes of one field at the end: one chip for a value, counts for options or facts."""
    counts = Counter(outcome.outcome for outcome in outcomes)
    if len(counts) > 1:
        counts.pop("correctly_empty", None)  # options that are rightly not ticked say nothing here
    return "".join(f"<span class='chip {name}'>{f'{counts[name]} ' if len(outcomes) > 1 else ''}"
                   f"{_OUTCOME_WORDS[name]}</span>" for name in _OUTCOME_WORDS if counts.get(name))


def _tiles(row: dict[str, Any]) -> str:
    tiles = [("Score", _number(row["score"], "{:.3f}"), "harmonic mean of the next four"),
             ("Completeness", _number(row["completeness"], "{:.3f}"), "was each value shown?"),
             ("Precision", _number(row["precision"], "{:.3f}"), "was what stood on screen right?"),
             ("Speed", _number(row["speed"], "{:.3f}"), "how soon after it was said?"),
             ("Outcome", _number(row["outcome"], "{:.3f}"), "the form at the end"),
             ("Price per form", _number(row["eur_per_form"], "€{:.4f}"), "model calls")]
    cells = "".join(f"<div class='tile'><div class='label'>{_esc(label)}</div><div class='value'>"
                    f"{_esc(value)}</div><div class='note'>{_esc(note)}</div></div>"
                    for label, value, note in tiles)
    return f"<div class='tiles'>{cells}</div>"


def _rows(head: list[str], rows: list[list[str]], css: str = "text") -> str:
    """A table from cells that are already HTML."""
    return (f"<div class='wide'><table class='{css}'><thead><tr>" + "".join(f"<th>{cell}</th>" for cell in head)
            + "</tr></thead><tbody>" + "".join("<tr>" + "".join(row) + "</tr>" for row in rows)
            + "</tbody></table></div>")


def _table(frame: pd.DataFrame, columns: Columns, names: dict[str, str], colors: dict[str, str],
           first: tuple[str, str] = ("run", "Run"), lead: tuple[str, str] | None = None) -> str:
    """A table with a row per row of ``frame`` that sorts by a click on a header. One column names a run, behind
    a dot of its colour: ``first`` is that column and its header, ``names`` maps its values to the names on the
    pages. ``lead`` puts a column of text before it. Columns that ``frame`` doesn't have are left out."""
    columns = [column for column in columns if column[0] in frame.columns]
    rows = []
    for row in frame.to_dict("records"):
        cells = [f"<td class='t'>{_run(names.get(row[first[0]], row[first[0]]), colors)}</td>"]
        if lead is not None:
            cells.insert(0, f"<td>{_esc(row[lead[0]])}</td>")
        rows.append(cells + [_td(row[column], style) for column, _, style in columns])
    head = [*([lead[1]] if lead is not None else []), first[1], *(_esc(label) for _, label, _ in columns)]
    return _rows(head, rows, css="sortable")


def _bar(entries: list[FormEntry], current: FormEntry | None, to_forms: str, to_report: str,
         jumps: list[tuple[str, str]]) -> str:
    """The bar on top of every page: back to the overview, the form selector, the form before and after, and
    the parts of this page (``jumps``: anchor and label)."""
    hint = f"{len(entries)} forms; the number next to a form is the lowest score a run got on it"
    options = ["<option value=''>Choose a form …</option>"] if current is None else []
    for dataset in dict.fromkeys(entry.dataset for entry in entries):
        options.append(f"<optgroup label='{_esc(DATASET_NAMES.get(dataset, dataset))}'>")
        for entry in entries:
            if entry.dataset == dataset:
                score = "" if entry.lowest is None else f" · {entry.lowest:.2f}"
                options.append(f"<option value='{to_forms}{_esc(entry.page)}'{' selected' if entry is current else ''}>"
                               f"{_esc(entry.conversation)}{score}</option>")
        options.append("</optgroup>")
    steps = ""
    if current is not None:
        at = entries.index(current)
        before = f"<a class='step' href='{_esc(entries[at - 1].page)}'>◀ {_esc(entries[at - 1].conversation)}</a>" \
            if at > 0 else "<span class='step off'>◀</span>"
        after = f"<a class='step' href='{_esc(entries[at + 1].page)}'>{_esc(entries[at + 1].conversation)} ▶</a>" \
            if at + 1 < len(entries) else "<span class='step off'>▶</span>"
        steps = before + after
    links = "".join(f"<a href='#{_esc(anchor)}'>{_esc(label)}</a>" for anchor, label in jumps)
    return (f"<nav class='bar'><a class='home' href='{to_report}'>Overview</a><label>Form "
            f"<select title='{_esc(hint)}' onchange=\"if (this.value) location.href = this.value\">"
            + "".join(options) + f"</select></label>{steps}<span class='jump'>{links}</span></nav>")


def _script_json(value: Any) -> str:
    """A value as JSON inside a script element: a text that holds "</script>" must not end the script."""
    return json.dumps(value).replace("</", "<\\/")


def _page(title: str, bar: str, body: str, specs: list[dict[str, Any]], mode: str, texts: Sequence[str] = ()) -> str:
    """A page of the report. ``texts`` are the values that its plots and tables show cut short: the page's
    window on a value shows them in full (a bar, a mark or a cell names its text by its place in this list)."""
    c = CHROME[mode]
    chips = "".join(f".chip.{name} {{ background: {colour}; color: #0b0b0b; }} " for name, colour in STATUS.items())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{_esc(title)}</title>
<script src="https://cdn.jsdelivr.net/npm/vega@{alt.VEGA_VERSION}"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@{alt.VEGALITE_VERSION}"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@{alt.VEGAEMBED_VERSION}"></script>
<style>
body {{ font: 14px/1.5 {FONT}; color: {c['primary']}; background: {c['page']}; margin: 0; }}
main {{ margin: 24px auto 64px; max-width: 1180px; padding: 0 20px; }}
h1 {{ font-size: 22px; }} h2 {{ font-size: 17px; margin-top: 40px; }} h3 {{ font-size: 14px; margin-top: 24px; }}
h2, details {{ scroll-margin-top: 60px; }}
a {{ color: {c['primary']}; }}
.meta, .lede {{ color: {c['secondary']}; margin-top: -6px; }} .note {{ color: {c['secondary']}; margin-top: 10px; }}
.bar {{ position: sticky; top: 0; z-index: 5; display: flex; gap: 14px; align-items: center; flex-wrap: wrap;
  background: {c['surface']}; border-bottom: 1px solid {c['grid']}; padding: 10px 20px; font-size: 13px; }}
.bar select {{ font: inherit; max-width: 260px; padding: 3px 6px; }} .bar .home {{ font-weight: 600; }}
.bar .step {{ text-decoration: none; padding: 3px 8px; border: 1px solid {c['grid']}; border-radius: 6px; }}
.bar .off, .none, .sub {{ color: {c['muted']}; }}
.bar .jump {{ margin-left: auto; display: flex; gap: 14px; flex-wrap: wrap; }}
.bar .jump a {{ color: {c['secondary']}; text-decoration: none; }} .bar .jump a:hover {{ text-decoration: underline; }}
.key {{ display: flex; flex-wrap: wrap; gap: 4px 18px; margin: 8px 0 0; font-size: 13px; color: {c['secondary']}; }}
.dot {{ font-size: 12px; line-height: 1; }}
ul.runs {{ list-style: none; padding: 0; margin: 6px 0 0; }} ul.runs li {{ margin: 3px 0; }}
.flag {{ font-size: 12px; color: {c['secondary']}; margin-left: 17px; }}
.tiles {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin: 16px 0; }}
.tile {{ background: {c['surface']}; border: 1px solid {c['grid']}; border-radius: 8px; padding: 10px 12px; }}
.tile .label, .tile .note {{ color: {c['secondary']}; font-size: 12px; }} .tile .value {{ font-size: 22px; font-weight: 600; }}
.chart {{ background: {c['surface']}; border: 1px solid {c['grid']}; border-radius: 8px; margin: 16px 0; overflow-x: auto; }}
.wide {{ overflow-x: auto; }}
table {{ border-collapse: collapse; background: {c['surface']}; font-size: 12px; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: right; padding: 4px 10px; border-bottom: 1px solid {c['grid']}; }}
th:first-child, td:first-child, td.t {{ text-align: left; }} th {{ color: {c['secondary']}; font-weight: 500; }}
table.text th, table.text td {{ text-align: left; vertical-align: top; }} table.text td {{ max-width: 420px; }}
table.sortable th {{ cursor: pointer; }} table.sortable td.t {{ max-width: 250px; }}
td ul {{ margin: 0; padding-left: 16px; }}
ul.diff {{ list-style: none; padding-left: 0; }} ul.diff li {{ padding-left: 14px; text-indent: -14px; }}
ul.diff li::before {{ font-weight: 600; margin-right: 5px; }} ul.diff li.gone {{ color: {c['muted']}; }}
ul.diff li.add::before {{ content: "+"; color: {STATUS['correct']}; }}
ul.diff li.gone::before {{ content: "−"; color: {STATUS['wrong']}; }}
.sub {{ font-size: 11px; font-weight: 400; }} .rules {{ max-width: 900px; color: {c['secondary']}; }}
.said {{ font-size: 11px; color: {c['secondary']}; }} .said b {{ color: {c['primary']}; font-weight: 600; }}
.chip {{ display: inline-block; margin: 3px 4px 0 0; padding: 0 7px; border-radius: 9px; font-size: 11px;
  color: {c['primary']}; background: {c['grid']}; }} {chips}
details {{ background: {c['surface']}; border: 1px solid {c['grid']}; border-radius: 8px; margin: 8px 0; padding: 6px 12px; }}
details .lede {{ margin-top: 0; }} summary {{ cursor: pointer; font-weight: 500; }}
details[open] > summary {{ margin-bottom: 8px; }}
.tip {{ position: fixed; z-index: 20; max-width: min(560px, calc(100vw - 16px)); max-height: 60vh; overflow: auto;
  padding: 8px 10px; background: {c['surface']}; border: 1px solid {c['axis']}; border-radius: 8px;
  box-shadow: 0 6px 24px rgba(0, 0, 0, .18); font-size: 12px; }}
.tip table {{ background: none; }} .tip td {{ border: 0; padding: 2px 10px 2px 0; text-align: left; vertical-align: top;
  white-space: pre-wrap; overflow-wrap: anywhere; }} .tip td:first-child {{ color: {c['secondary']}; white-space: nowrap; }}
[data-ref] {{ cursor: help; }}
.views {{ display: inline-flex; margin-left: 6px; vertical-align: 1px; }}
.views button {{ font: inherit; font-size: 12px; padding: 1px 10px; border: 1px solid {c['axis']}; background: {c['surface']};
  color: {c['secondary']}; cursor: pointer; }} .views button + button {{ border-left: 0; }}
.views button[aria-pressed='true'] {{ background: {c['grid']}; color: {c['primary']}; font-weight: 600; }}
[data-view-in] > .wide {{ max-height: 78vh; overflow: auto; }}  /* a long table scrolls under its own headers */
table.talk th {{ position: sticky; top: 0; z-index: 1; background: {c['surface']}; box-shadow: 0 1px 0 {c['grid']}; }}
table.talk td {{ max-width: 320px; }} table.talk .old {{ color: {c['muted']}; }}
table.talk .said {{ margin-bottom: 3px; }} table.talk .chip {{ margin: 0 4px 0 0; }}
.who {{ display: inline-block; margin-right: 4px; padding: 0 5px; border-radius: 4px; font-size: 11px; font-weight: 600;
  background: {c['grid']}; color: {c['secondary']}; }}
@media (max-width: 760px) {{ .tiles {{ grid-template-columns: repeat(2, 1fr); }} }}
</style></head><body>
{bar}
<main>
{body}
</main>
<script>
const specs = {_script_json(specs)};
const texts = {_script_json(list(texts))};  // the values that plots and tables show cut short, in full{_SCRIPT}
specs.forEach((spec, k) => vegaEmbed('#chart' + k, spec,
  {{actions: false, renderer: 'svg', loader: {{target: '_self'}}, tooltip: chartTip}}));
</script></body></html>"""


def _chart(chart: Any, specs: list[dict[str, Any]]) -> str:
    """The place of a themed chart in the page; its spec is drawn there when the page opens."""
    specs.append(chart.to_dict())
    return f"<div class='chart' id='chart{len(specs) - 1}'></div>"


# ---- the overview

def _notes(records: list[dict[str, Any]], names: dict[str, str], colors: dict[str, str]) -> str:
    """The runs of a dataset, one per line: what each was made on, and what a reader has to know about it."""
    items = []
    for row in records:
        notes = [f"{row['dataset']}/{row['split']}", f"{row['conversations']} conversations"]
        if not missing(row.get("judge")):
            notes.append(f"judge: {row['judge']}")
        flags = []
        if not row["official"]:
            flags.append("Unofficial: made on a development sample or at another speed, or the official sample or "
                         "the stream settings have changed since.")
        elif not row["config_matches"]:
            notes.append("scored with today's evaluation settings, which have changed since the run was made")
        if row.get("failure_rate"):
            flags.append(f"{row['failure_rate']:.0%} of the conversations failed.")
        items.append(f"<li>{_run(names[row['run']], colors)} <span class='sub'>{_esc(' · '.join(notes))} · "
                     f"{_esc(row['run'])}</span>{''.join(f'<div class=flag>{_esc(flag)}</div>' for flag in flags)}</li>")
    return f"<ul class='runs'>{''.join(items)}</ul>"


def _unit_table(units: pd.DataFrame, runs: pd.DataFrame, names: dict[str, str], colors: dict[str, str]) -> str:
    """The price of a form unit by unit, one row per run in the order of ``runs`` ("" when no unit has a price).
    ``units`` comes from call_plots.price_by_unit."""
    if units.empty:
        return ""
    order = unit_order(units["unit"])
    wide = units.pivot(index="run", columns="unit", values="eur_per_form").reindex(
        [run for run in runs["run"] if run in set(units["run"])]).reset_index()
    return _table(wide, [(unit, f"{call_plots.unit_name(unit)} (€ per form)", "{:.5f}") for unit in order],
                  names, colors)


def _dataset(result: Any, dataset: str, config: Config, specs: list[dict[str, Any]]) -> str:
    """One dataset's section of report.html: all its runs in the same tables and plots, the highest score first."""
    mode, p = config.plots.mode, f"{config.latency.percentile:g}"
    key = run_key(result.summary, mode, dataset)
    names, colors = key.names, key.colors
    mine = result.summary[result.summary["run"].isin(names)].sort_values("score", ascending=False, na_position="last")
    records = mine.to_dict("records")
    several = len(records) > 1
    parts = [f"<h2 id='{_esc(dataset)}'>{_esc(DATASET_NAMES.get(dataset, dataset))}</h2>", _notes(records, names, colors)]

    def plot(chart: Any) -> None:
        if chart is not None:
            parts.append(_chart(themed(chart, mode), specs))

    parts.append("<h3>Scores</h3>")
    if not several:
        parts.append(_tiles(records[0]))
    parts.append(_table(mine, [
        ("conversations", "Conversations", "{:,.0f}"), ("score", "Score", "{:.3f}"),
        ("completeness", "Completeness", "{:.3f}"), ("precision", "Precision", "{:.3f}"), ("speed", "Speed", "{:.3f}"),
        ("outcome", "Outcome", "{:.3f}"), ("final_f1", "Final F1", "{:.3f}"),
        ("latency_median_s", "Latency, median (s)", "{:.2f}"), (f"latency_p{p}_s", f"Latency, p{p} (s)", "{:.2f}"),
        ("never_settled", "Never settled", "{:.1%}"), ("fixes_per_form", "Fixes per form", "{:.2f}"),
        ("eur_per_form", "Price per form (€)", "{:.4f}"), ("official", "Official", None)], names, colors))
    if several:
        plot(terms_by_run(mine, key, mode))
    plot(latency_curves(result.values, key, mode))

    types = result.per_field_type[result.per_field_type["run"].isin(names)]
    parts.append("<h3>By field type</h3><p class='lede'>Which kinds of field a run gets right, at the end and while "
                 "people were talking. Click a column of the table to sort by it.</p>")
    plot(by_field_type(types, key, mode))
    plot(outcomes_by_field_type(types, key, mode))
    listed = types.assign(place=types["run"].map({run: k for k, run in enumerate(names)})).sort_values(
        ["field_type", "place"])
    by_type: Columns = [
        ("correct", "Correct", "{:,.0f}"), ("wrong", "Wrong", "{:,.0f}"), ("missing", "Missing", "{:,.0f}"),
        ("invented", "Invented", "{:,.0f}"), ("correctly_empty", "Correctly empty", "{:,.0f}"),
        ("final_f1", "Final F1", "{:.3f}"), ("completeness", "Completeness", "{:.3f}"),
        ("precision", "Precision", "{:.3f}"), ("speed", "Speed", "{:.3f}"),
        ("latency_median_s", "Latency, median (s)", "{:.2f}"), (f"latency_p{p}_s", f"Latency, p{p} (s)", "{:.2f}"),
        ("never_settled", "Never settled", "{:.1%}"), ("invented_rate", "Invented rate", "{:.1%}"),
        ("revisions_per_filled_field", "Revisions per filled field", "{:.2f}")]
    if several:  # a row per field type and run
        parts.append(_table(listed, by_type, names, colors, lead=("field_type", "Field type")))
    else:
        parts.append(_table(listed, by_type, {}, colors, first=("field_type", "Field type")))

    # a run that only counts its calls, without tokens or a price, has nothing to show here
    if mine[["tokens_per_call", "eur_per_form"]].fillna(0).gt(0).to_numpy().any():
        parts.append("<h3>Model calls, tokens and price</h3><p class='lede'>Where the price comes from: how often a "
                     "run calls a model to fill the form, what each call carries, how many calls changed nothing on "
                     "screen, and what the run's other uses cost (a model that only decides whether to extract now, "
                     "…). A run that reports what its vendor read from a cache has its input told apart "
                     "(docs/metrics.md).</p>")
        parts.append(_table(mine, [
            ("calls_per_form", "Calls per form", "{:,.1f}"), ("calls_per_audio_minute", "… per audio minute", "{:.1f}"),
            ("calls_changing_share", "… that changed a value", "{:.0%}"), ("failed_calls", "Failed calls", "{:,.0f}"),
            ("call_seconds_median", "A call takes, median (s)", "{:.2f}"), (f"call_seconds_p{p}", f"… p{p} (s)", "{:.2f}"),
            ("tokens_per_call", "Tokens per call", "{:,.0f}"),
            ("cache_read_tokens_per_form", "Read from cache, per form", "{:,.0f}"),
            ("fresh_input_tokens_per_form", "Fresh input, per form", "{:,.0f}"),
            ("cache_write_tokens_per_form", "Written to cache, per form", "{:,.0f}"),
            ("output_tokens_per_form", "Output, per form", "{:,.0f}"),
            ("cached_input_share", "Input read from cache", "{:.0%}"),
            ("other_uses_per_form", "Other uses per form", "{:,.1f}"),
            ("eur_other_per_form", "… their price per form (€)", "{:.4f}"),
            ("eur_per_form", "Price per form (€)", "{:.4f}"),
            ("eur_per_audio_minute", "Price per audio minute (€)", "{:.4f}")], names, colors))
        if several:
            plot(call_plots.calls_by_run(mine, key, mode))
        plot(call_plots.token_mix(mine, key, mode))
        units = call_plots.price_by_unit(result.calls[result.calls["run"].isin(names)], mine)
        plot(call_plots.price_mix(units, key, mode))
        parts.append(_unit_table(units, mine, names, colors))
        plot(call_plots.tokens_over_conversation(result.calls, key, mode))
        forms = result.per_conversation
        plot(call_plots.price_against_length(forms.assign(
            page=FORMS_DIR + "/" + forms["dataset"] + "-" + forms["conversation"] + ".html"), key, mode))

    parts.append("<h3>Value changes</h3><p class='lede'>How restless the form was to look at: every visible change of "
                 "a field is a first fill, a revision or a clearing. A value that grows while it is dictated is "
                 "revised as it should be; a model that changes its mind is revised too. Precision says whether what "
                 "stood on screen was right.</p>")
    parts.append(_table(mine, CHANGES, names, colors))
    if several:
        plot(changes_by_run(mine, key, mode))
    return "".join(parts)


def _overview(result: Any, entries: list[FormEntry], config: Config,
              specs: list[dict[str, Any]]) -> tuple[str, list[tuple[str, str]]]:
    """The body of report.html, and the parts of it the bar on top links to."""
    mode = config.plots.mode
    summary = result.summary
    everyone = run_key(summary, mode)  # every run: its name on the pages and its colour
    names, colors = everyone.names, everyone.colors
    datasets = list(dict.fromkeys(summary["dataset"]))
    about = (f"{len(summary)} run{'s' if len(summary) != 1 else ''} of "
             f"{_esc(', '.join(dict.fromkeys(summary['name'])))} · {len(entries)} forms. Choose a form at the top "
             "to see it in detail: what it asks, what was said, and what each run showed.")
    body = [f"<h1>Evaluation report</h1><p class='meta'>{about}</p>" + _key(everyone.order, colors)]
    body.append("<h2 id='ranking'>Ranking</h2>" + _table(result.ranking, RANKING, {}, colors, ("name", "Submission")))
    scatter = score_against_price(summary, everyone, mode)
    if scatter is not None:
        body.append(_chart(themed(scatter, mode), specs))
    body += [_dataset(result, dataset, config, specs) for dataset in datasets]
    jumps = [("ranking", "Ranking"), *((dataset, DATASET_NAMES.get(dataset, dataset)) for dataset in datasets),
             ("forms", "Forms"), ("numbers", "All numbers")]
    body.append("<h2 id='forms'>Forms</h2><p class='lede'>Every conversation, with each run's score on it. Open one "
                "to see the form, its timeline, every field and the transcript; click a column to sort by it. The "
                "selector at the top of every page lists the same forms.</p>")
    rows = []
    for entry in entries:
        by_name = {name: item for name, _, item in entry.runs}
        cells = [f"<td><a href='{FORMS_DIR}/{_esc(entry.page)}'>{_esc(entry.conversation)}</a></td>",
                 f"<td class='t'>{_esc(entry.dataset)}</td>", f"<td>{entry.runs[0][2].audio_s:.0f}</td>"]
        cells += [f"<td>{_number(by_name[name].terms.score, '{:.3f}') if name in by_name else ''}</td>"
                  for name in everyone.order]
        rows.append(cells)
    body.append(_rows(["Form", "Dataset", "Audio (s)", *(_run(name, colors) for name in everyone.order)], rows,
                      css="sortable"))
    records = summary.drop(columns=["run_dir"]).to_dict("records")
    head = ["Metric", *(f"{_run(names[row['run']], colors)}<div class='sub'>{_esc(row['dataset'])}</div>"
                        for row in records)]
    rows = [[f"<td>{_esc(metric)}</td>", *(f"<td>{_esc(_plain(row[metric]))}</td>" for row in records)]
            for metric in records[0] if metric not in ("name", "dataset")]
    body.append("<h2 id='numbers'>All numbers</h2><p class='lede'>Every number of every run, as in summary.csv."
                "</p>" + _rows(head, rows, css=""))
    return "".join(body), jumps


# ---- one form

def gold_entries(form: Form, gold: Gold) -> list[GoldEntry]:
    """What the gold says the form's fields should show, in the order it is spoken: a value of a typed field, an
    option of a multiselect, a fact of a note or an answer. ``gold`` is the dataset's (audio time)."""
    by_id = {spec.field_id: spec for spec in form.fields}
    entries = [GoldEntry(value.said, value.source, by_id[value.field_id],
                         None if value.value is None else value_text(value.value, by_id[value.field_id].type))
               for value in gold.values if value.field_id in by_id and by_id[value.field_id].type in SCALAR_TYPES]
    entries += [GoldEntry(item.said, item.source, by_id[item.field_id], item.text)
                for item in gold.items if item.field_id in by_id
                and by_id[item.field_id].type == FieldType.multiselect]
    entries += [GoldEntry(fact.said, fact.source, by_id[fact.field_id], fact.text)
                for fact in gold.facts if fact.field_id in by_id and by_id[fact.field_id].type == FieldType.textarea]
    return sorted(entries, key=lambda entry: entry.spoken)


def _who(speaker: str | None, roles: dict[str, str]) -> str:
    """A speaker's letter as the stream gives it ("?" while a few words can't be attributed yet)."""
    if not speaker:
        return ""
    unknown = speaker == "PENDING"
    title = "too short to tell who speaks" if unknown else roles.get(speaker, "speaker")
    return f"<span class='who' title='{_esc(title)}'>{'?' if unknown else _esc(speaker)}</span>"


def _talk(conversation: Conversation, form: Form, entry: FormEntry, stream_config: StreamConfig,
          lists: dict[str, bool], colors: dict[str, str], texts: list[str]) -> str:
    """The conversation as the extractor received it, as a timeline with a column per thing to compare: what
    arrived, the gold values those words complete ("Gold said here"), and what each run filled while that was the
    latest thing it had received.

    Two tables, one shown at a time: one line per turn, and one line per message (every partial and every
    finished turn). The second shows why the gold of a dictated value has several steps: the value arrives in
    several messages, and each step is what the field should show once that message is there. A value longer
    than a cell is cut there; its whole text goes to ``texts``, for the page's window on a value.
    """
    stream = build_stream(conversation.words, stream_config, key=conversation.id)
    letters = speaker_letters(conversation.words)
    roles = {letter: conversation.roles.get(speaker, speaker) for speaker, letter in letters.items()}
    places = {text: k for k, text in enumerate(texts)}

    def cut(text: str, name: str) -> str:
        if len(text) <= _CELL_CHARACTERS:
            return _esc(text)
        if text not in places:
            places[text] = len(texts)
            texts.append(text)
        return (f"<span data-ref='{places[text]}' data-name='{_esc(name)}'>"
                f"{_esc(text[:_CELL_CHARACTERS].rstrip())} …</span>")

    said: list[tuple[float, str]] = []  # (when, the gold step as a line of a cell)
    for spec in form.fields:
        steps = max((item.walks[spec.field_id].steps for _, _, item in entry.runs if spec.field_id in item.walks),
                    key=len, default=[])
        before: list[str] = []
        for step in steps:
            if spec.type in PIECE_TYPES:  # what the step adds to the field, and what it ends
                now = _pieces(step.value, spec, lists.get(spec.field_id, True))
                text = "\n".join([f"+ {piece}" for piece in now if piece not in before]
                                 + [f"− {piece}" for piece in before if piece not in now]) or "unchanged"
                before = now
            else:
                text = "withdrawn" if step.value is None else value_text(step.value, spec.type)
            said.append((step.said, f"<div class='said'><b>{_esc(spec.label)}</b> {cut(text, spec.label)}</div>"))
    said.sort(key=lambda moment: moment[0])

    filled: dict[str, list[tuple[float, str]]] = {}  # run → (when, the value it published as a line of a cell)
    for name, _, item in entry.runs:
        mine = []
        for spec in form.fields:
            walk = item.walks.get(spec.field_id)
            for change in item.changes.get(spec.field_id, ()):
                at = next((k for k, display in enumerate(walk.displays if walk else ())
                           if not display.kept and abs(display.shown - change.t) < 1e-6), None)
                display = walk.displays[at] if at is not None else None
                after = walk.displays[at + 1] if at is not None and at + 1 < len(walk.displays) else None
                # shown before the words that say it had arrived: it counts as right from the moment they do
                early = display is not None and display.grade < 1 and after is not None and after.kept and after.grade >= 1
                kind = ("" if display is None else "correct" if early or display.grade >= 1
                        else "missing" if display.grade > 0 else "wrong")
                sign = {"correct": "✓", "missing": "≈", "wrong": "✕"}.get(kind, "?")
                read = {"correct": "right", "missing": "close", "wrong": "wrong"}.get(kind, "not graded")
                value = "cleared" if change.after is None else value_text(change.after, spec.type)
                note = (f" <span class='sub'>({after.shown - display.shown:.1f} s before it was said)</span>"
                        if early else "")
                line = (f"<div class='said'><span class='chip {kind}' title='{read}'>{sign}</span>"
                        f"<b>{_esc(spec.label)}</b> {cut(value, spec.label)}{note}</div>")
                mine.append((change.t, line))
        filled[name] = sorted(mine, key=lambda moment: moment[0])

    def table(groups: list[tuple[int, int]], view: str) -> str:
        rows = []
        for k, (first, last) in enumerate(groups):
            start = 0.0 if k == 0 else stream[first].t  # what happened before the first message counts with it
            until = stream[groups[k + 1][0]].t if k + 1 < len(groups) else math.inf
            message = stream[last]
            if view == "messages" and message.kind == "partial":  # the turn so far: its last words, then the new ones
                earlier = stream[last - 1].text if last and stream[last - 1].turn_id == message.turn_id else ""
                kept = earlier if message.text.startswith(earlier) else ""
                tail = kept if len(kept) <= _TAIL_CHARACTERS else "… " + kept[-_TAIL_CHARACTERS:].split(" ", 1)[-1]
                text = (f"{_who(message.speaker, roles)}<span class='old'>{_esc(tail)}</span>"
                        f"{_esc(message.text[len(kept):])}")
            else:
                text = " ".join(_who(speaker, roles) + _esc(words) for speaker, words in message.pieces())
                if view == "messages":
                    text += " <span class='sub'>(the turn is finished)</span>"
            rows.append([f"<td>{stream[first].t:.1f}</td>", f"<td>{text}</td>",
                         f"<td>{''.join(line for t, line in said if start <= t < until)}</td>",
                         *(f"<td>{''.join(line for t, line in filled[name] if start <= t < until)}</td>"
                           for name, _, _ in entry.runs)])
        head = ["Arrives at (s)", "What the extractor receives", "Gold said here",
                *(f"Filled here by {_run(name, colors)}" for name, _, _ in entry.runs)]
        return (f"<div data-view-in='talk' data-view='{view}'{'' if view == 'turns' else ' hidden'}>"
                + _rows(head, rows, css="text talk") + "</div>")

    turns: list[tuple[int, int]] = []
    for index, message in enumerate(stream):
        if turns and stream[turns[-1][0]].turn_id == message.turn_id:
            turns[-1] = (turns[-1][0], index)
        else:
            turns.append((index, index))
    views = ("<span class='views'><button type='button' data-view-of='talk' data-view='turns' aria-pressed='true'>"
             "Turns</button><button type='button' data-view-of='talk' data-view='messages' aria-pressed='false'>"
             "Every message</button></span>")
    speakers = ", ".join(f"{letter} = {_esc(role)}" for letter, role in sorted(roles.items()))
    return (f"<details id='transcript'><summary>The conversation as the extractor received it: {len(turns)} turns, "
            f"{len(stream)} messages</summary><p class='lede'>Each line is what reached the extractor at that "
            "moment (docs/stream.md), next to the gold values those words complete and to what each run filled "
            "before the next line arrived. A value that is spoken piece by piece (a phone number, a spelled name) "
            "arrives in several messages, so its gold has a step per message: “Every message” lists each partial "
            "and finished turn as it was sent, with the turn's earlier words in grey; “Turns” shows each turn once, "
            f"as it ended. Speakers: {speakers}. A ✓ is a value graded right, ≈ close, ✕ wrong. A long value is cut: "
            f"rest the pointer on it to read all of it. Show {views}</p>"
            + table(turns, "turns") + table([(index, index) for index in range(len(stream))], "messages")
            + "</details>")


def _setting(spec: FieldSpec) -> str:
    if spec.options:
        return _esc(", ".join(spec.options))
    return _esc(f"{spec.range[0]:g} to {spec.range[1]:g}") if spec.range else ""


def _definition(form: Form, said: set[str]) -> str:
    """The form as an extractor gets it: its rules and every field, with the fields this conversation fills."""
    rows = [[f"<td>{_esc(spec.label)}<div class='sub'>{_esc(spec.field_id)}</div></td>",
             f"<td>{_esc(spec.type.value)}</td>", f"<td>{_esc(spec.prompt)}</td>", f"<td>{_setting(spec)}</td>",
             "<td><span class='chip'>said here</span></td>" if spec.field_id in said else "<td></td>"]
            for spec in form.fields]
    return (f"<p class='lede'>“{_esc(form.name)}”, as every extractor gets it: its rules, and for each field its "
            "id, its type, what it asks for and the options or the range it allows.</p>"
            f"<p class='rules'><b>Rules:</b> {_esc(form.description)}</p>"
            + _rows(["Field", "Type", "What it asks for", "Options or range", "In this conversation"], rows))


def _scores(entry: FormEntry, colors: dict[str, str], config: Config) -> str:
    rows = []
    for name, _, item in entry.runs:
        final = final_f1(item.outcomes)
        latency = final_decision_latency(item.outcomes, config.latency.percentile)
        moved = value_changes(item.changes.values())
        rows.append([f"<td>{_run(name, colors)}</td>"] + [f"<td>{cell}</td>" for cell in (
            _number(item.terms.score, "{:.3f}"), _number(item.terms.completeness, "{:.3f}"),
            _number(item.terms.precision, "{:.3f}"), _number(item.terms.speed, "{:.3f}"),
            _number(item.terms.outcome, "{:.3f}"), final.correct, final.wrong, final.missing, final.invented,
            final.fixes, moved.changes, moved.revised, _number(latency.median_s, "{:.2f}"), _esc(item.status))])
    return _rows(["Run", "Score", "Completeness", "Precision", "Speed", "Outcome", "Correct", "Wrong", "Missing",
                  "Invented", "Fixes", "Value changes", "… of them revisions", "Latency, median (s)", "Status"],
                 rows, css="")


def _calls(entry: FormEntry, colors: dict[str, str], config: Config) -> str:
    """What each run used on this form: its model calls that tried to fill the form (what they carried, how long
    they took), its other uses, and the price of everything, unit by unit ("" when no run reported a use)."""
    rows, spent = [], []
    for name, _, item in entry.runs:
        if not item.model_calls:
            continue
        stats = call_stats(item.model_calls, 1, item.audio_s, config.latency.percentile)
        seconds = [call.seconds for call in item.model_calls if call.extraction and call.seconds is not None]
        models = ", ".join(sorted({call.model for call in item.model_calls if call.model}))
        rows.append([f"<td>{_run(name, colors)}</td>", f"<td>{_esc(models)}</td>"] + [f"<td>{cell}</td>" for cell in (
            stats.calls, stats.changing, stats.failed, *(f"{stats.tokens[kind]:,}" for kind in TOKEN_KINDS),
            stats.other_uses, _number(stats.eur_other, "{:.4f}"), _number(stats.eur, "{:.4f}"),
            _number(stats.seconds_median, "{:.2f}"), _number(max(seconds) if seconds else None, "{:.2f}"))])
        used: dict[str, float] = {}
        for call in item.model_calls:
            for unit, amount in call.usage.items():
                used[unit] = used.get(unit, 0.0) + amount
        spent.append((name, stats.eur_by_unit, used))
    if not rows:
        return ""
    table = _rows(["Run", "Model", "Calls", "… that changed a value", "Failed", "Read from cache", "Fresh input",
                   "Written to cache", "Output", "Other uses", "… their price (€)", "Price (€)",
                   "A call takes, median (s)", "The longest (s)"], rows, css="")
    order = unit_order({unit for _, euros, _ in spent for unit in euros})
    if not order:
        return table
    by_unit = [[f"<td>{_run(name, colors)}</td>"]
               + [f"<td>{_number(euros.get(unit), '{:.5f}')}"
                  + (f"<div class='sub'>{used[unit]:,.0f} used</div>" if unit in used else "") + "</td>"
                  for unit in order] for name, euros, used in spent]
    return table + ("<p class='lede'>The price unit by unit: what each unit cost on this form, over every use, and "
                    "how much of it was used.</p>"
                    + _rows(["Run", *(f"{_esc(call_plots.unit_name(unit))} (€)" for unit in order)], by_unit, css=""))


def _fields(form: Form, entry: FormEntry, colors: dict[str, str], limit_s: float, power: float) -> str:
    """The form at the end: per field, what it should hold and what each run shows, with the field's terms."""
    rows, untouched = [], []
    for spec in form.fields:
        walks = [item.walks.get(spec.field_id) for _, _, item in entry.runs]
        if not any(walks):
            untouched.append(spec.label)
            continue
        steps = max((walk.steps for walk in walks if walk), key=len)
        should = _value(steps[-1].value, spec.type) if steps else "<span class='none'>nothing: nobody said it</span>"
        name = (f"<td><a href='#field-{_esc(spec.field_id)}'>{_esc(spec.label)}</a><div class='sub'>"
                f"{_esc(spec.prompt)}</div><div class='sub'>{_esc(spec.type.value)}</div></td>")
        cells = [name, f"<td>{should}</td>"]
        for (_, _, item), walk in zip(entry.runs, walks):
            outcomes = [outcome for outcome in item.outcomes if outcome.field_id == spec.field_id]
            shown = walk.displays[-1].value if walk and walk.displays else None
            terms = "" if walk is None else (
                f"completeness {_number(field_completeness(walk), '{:.2f}')} · precision "
                f"{_number(field_precision(walk), '{:.2f}')} · speed {_number(field_speed(walk, limit_s, power), '{:.2f}')}")
            counted = _chips(outcomes)
            support = next((one for one in item.supports if one.field_id == spec.field_id), None)
            if support is not None and support.gold_units:  # an answer in facts: both counts, in words
                counted = (f"<span class='sub'>Extractor output {support.shown_units} "
                           f"fact{'' if support.shown_units == 1 else 's'}, gold has {support.gold_units}: </span>"
                           f"<span class='chip correct'>{support.shown_supported} correct</span>"
                           f"<span class='chip invented'>{support.shown_units - support.shown_supported} not</span>"
                           f"<span class='chip missing'>{support.gold_units - support.gold_supported} missing</span>")
            cells.append(f"<td>{_value(shown, spec.type)}<div>{counted}</div><div class='sub'>{terms}</div></td>")
        rows.append(cells)
    rest = ""
    if untouched:
        many = len(untouched) > 1
        rest = (f"<p class='note'>{len(untouched)} more field{'s' if many else ''} stayed empty, as "
                f"{'they' if many else 'it'} should: nobody said a value for {'them' if many else 'it'} and no run "
                f"filled {'them' if many else 'it'} ({_esc(', '.join(untouched))}).</p>")
    return _rows(["Field", "Should hold at the end", *(_run(name, colors) for name, _, _ in entry.runs)], rows) + rest


def _held(value: Any, spec: FieldSpec, name: str, texts: list[str]) -> str:
    """A displayed value in a cell: cut when it is long, with its whole text in the page's window."""
    if value is None:
        return "<span class='none'>cleared</span>"
    text = value_text(value, spec.type)
    if len(text) <= _CELL_CHARACTERS:
        return _esc(text)
    if text not in texts:
        texts.append(text)
    return (f"<span data-ref='{texts.index(text)}' data-name='{_esc(name)}'>"
            f"{_esc(text[:_CELL_CHARACTERS].rstrip())} …</span>")


def _matched(form: Form, entry: FormEntry, colors: dict[str, str], lists: dict[str, bool],
             accepted: dict[str, tuple[str, ...]], limit_s: float, power: float, texts: list[str],
             text_match: TextMatch | None) -> str:
    """Every gold value of every field next to the display each run's walk gave it (timeline/walk.py): the
    pairs that completeness and speed are computed from. A note also says how its text compares with the facts
    of that moment, when the judge's verdicts are at hand (``text_match``)."""
    rows = []
    for spec in form.fields:
        walks = [item.walks.get(spec.field_id) for _, _, item in entry.runs]
        if not any(walk and walk.steps for walk in walks):
            continue
        steps = max((walk.steps for walk in walks if walk), key=len)
        in_pieces, is_list = spec.type in PIECE_TYPES, lists.get(spec.field_id, True)
        before: list[str] = []
        for k, step in enumerate(steps):
            if in_pieces:
                now = _pieces(step.value, spec, is_list)
                what, before = _change(before, now), now
            else:
                what = "<span class='none'>withdrawn</span>" if step.value is None else _value(step.value, spec.type)
            name = f"<a href='#field-{_esc(spec.field_id)}'>{_esc(spec.label)}</a>" if k == 0 else ""
            cells = [f"<td>{name}</td>", f"<td>{step.said:.1f}</td>", f"<td>{what}</td>"]
            for (run_name, _, _), walk in zip(entry.runs, walks):
                mine = walk.steps[k] if walk and k < len(walk.steps) else None
                if mine is None or mine.display is None:
                    cells.append("<td><span class='none'>nothing: no display made after it was said comes close"
                                 "</span><div class='sub'>score 0.00 · speed 0.00</div></td>")
                    continue
                display = walk.displays[mine.display]
                how = "closest display" if not mine.landed else (
                    "already on screen when it was said" if display.kept else "shown")
                counts = ""
                if spec.type == FieldType.textarea and text_match is not None and step.value and display.value:
                    support = text_match.support(display.value, step.value, is_list, accepted.get(spec.field_id, ()))
                    counts = (f"it says {support.facts_supported} of the {support.facts} facts said by then · "
                              f"{support.units_supported} of its {support.units} "
                              f"{'lines' if is_list else 'sentences'} are in the gold<br>")
                cells.append(
                    f"<td>{_held(display.value, spec, f'{spec.label}, {run_name}', texts)}<div class='sub'>{how} at "
                    f"{display.shown:.1f} s, {max(mine.delay, 0.0):.1f} s after<br>{counts}score {mine.score:.2f} · "
                    f"speed {curve(mine.delay, limit_s, power):.2f}</div></td>")
            rows.append(cells)
    return _rows(["Field", "Said at (s)", "Should show from then on", *(_run(name, colors) for name, _, _ in entry.runs)],
                 rows)


def _fact_rows(walks: list[tuple[str, Any, Any]], colors: dict[str, str], text_match: TextMatch) -> str:
    """A note's gold, fact by fact: when each fact was said, and from when each run's answer holds it. Two facts
    said with the same words are two rows."""
    steps = max((walk.steps for _, _, walk in walks), key=len)
    facts: list[tuple[float, str]] = []
    before: tuple[str, ...] = ()
    for step in steps:
        now = tuple(step.value or ())
        facts += [(step.said, fact) for fact in now if fact not in before]
        before = now
    rows = []
    for number, (said, fact) in enumerate(facts, 1):
        cells = [f"<td>{number}</td>", f"<td>{said:.1f}</td>", f"<td>{_esc(fact)}</td>"]
        for _, _, walk in walks:
            # the texts that were on screen once the fact was said: the judge read each of them against it
            later = [display for display in walk.displays
                     if not display.kept and display.value is not None and display.shown + display.seconds > said]
            first = next((display.shown for display in later if text_match.says(str(display.value), fact)), None)
            at_end = bool(later) and text_match.says(str(later[-1].value), fact)
            when = ("never" if first is None else "already there when it was said" if first <= said
                    else f"from {first:.1f} s, {first - said:.1f} s after")
            cells.append(f"<td>{when}<div class='sub'>at the end: {'found' if at_end else 'not found'}</div></td>")
        rows.append(cells)
    return _rows(["#", "Said at (s)", "Fact", *(_run(name, colors) for name, _, _ in walks)], rows)


def _history(form: Form, entry: FormEntry, colors: dict[str, str], lists: dict[str, bool],
             text_match: TextMatch | None = None, accepted: dict[str, tuple[str, ...]] | None = None) -> str:
    """Per field: its gold values over time with the display each run's walk gave them, and every display. A
    field of pieces (a note, a multiselect) shows what each step or display changes, not the whole text again.
    With the judge's verdicts at hand (``text_match``), a note's gold is listed fact by fact, and each of its
    texts says how many facts it holds and how many of its lines are in the gold."""
    accepted = accepted or {}
    blocks = []
    for spec in form.fields:
        walks = [(name, item, item.walks.get(spec.field_id)) for name, _, item in entry.runs]
        walks = [(name, item, walk) for name, item, walk in walks if walk]
        if not walks:
            continue
        in_pieces, is_list = spec.type in PIECE_TYPES, lists.get(spec.field_id, True)
        steps = max((walk.steps for _, _, walk in walks), key=len)
        about = " · ".join(part for part in (_esc(spec.type.value), _setting(spec), _esc(spec.prompt)) if part)
        parts = [f"<p class='lede'>{about}</p>"]
        noted = spec.type == FieldType.textarea and text_match is not None  # a note, with the judge's verdicts
        if steps and noted:
            parts.append("<h3>What it should say, fact by fact</h3>" + _fact_rows(walks, colors, text_match))
        elif steps:
            rows, before = [], []
            for k, step in enumerate(steps):
                if in_pieces:
                    now = _pieces(step.value, spec, is_list)
                    what, before = _change(before, now), now
                else:
                    what = "<span class='none'>withdrawn</span>" if step.value is None else _value(step.value, spec.type)
                cells = [f"<td>{k + 1}</td>", f"<td>{step.said:.1f}</td>", f"<td>{what}</td>"]
                for _, _, walk in walks:
                    mine = walk.steps[k] if k < len(walk.steps) else None
                    if mine is None or mine.display is None:
                        cells.append("<td><span class='none'>no display</span></td>")
                    else:
                        cells.append(f"<td>{'shown' if mine.landed else 'closest display'} at "
                                     f"{walk.displays[mine.display].shown:.1f} s<div class='sub'>score "
                                     f"{mine.score:.2f} · delay {mine.delay:.1f} s</div></td>")
                rows.append(cells)
            parts.append("<h3>What it should show</h3>" + _rows(
                ["#", "Said at (s)", "The step adds (+) or ends (−)" if in_pieces else "Gold value",
                 *(_run(name, colors) for name, _, _ in walks)], rows))
        for name, _, walk in walks:
            rows, before = [], []
            for display in walk.displays:
                if in_pieces:
                    now = _pieces(display.value, spec, is_list)
                    what, before = _change(before, now), now
                else:
                    what = "<span class='none'>cleared</span>" if display.value is None else _value(display.value,
                                                                                                    spec.type)
                counts = {"landing": f"gold value {display.step + 1 if display.step is not None else ''}",
                          "repair": f"back on gold value {display.step + 1 if display.step is not None else ''}"}.get(
                    display.kind, "no gold value: graded by how close it is")
                if display.kept:
                    counts += " (already on screen when it was said)"
                held = []
                if noted:  # against the facts said when the text was shown: the two counts its grade is made of
                    said = [step for step in walk.steps if step.said <= display.shown and step.value]
                    held = ["<td></td>", "<td></td>"]
                    if said and display.value is not None:
                        support = text_match.support(display.value, said[-1].value, is_list,
                                                     accepted.get(spec.field_id, ()))
                        held = [f"<td>{support.facts_supported} of {support.facts}</td>",
                                f"<td>{support.units_supported} of {support.units}</td>"]
                rows.append([f"<td>{display.shown:.1f}</td>", f"<td>{what}</td>", *held,
                             f"<td>{display.grade:.2f}</td>", f"<td>{_esc(counts)}</td>",
                             f"<td>{display.seconds:.1f}</td>"])
            parts.append(f"<h3>What {_run(name, colors)} showed</h3>" + (_rows(
                ["Shown at (s)", "It adds (+) or drops (−)" if in_pieces else "Value",
                 *(["Facts it says", "Its lines in the gold"] if noted else []), "Grade", "Counts as",
                 "Seconds on screen"], rows) if rows else "<p class='lede'>Nothing.</p>"))
        several = len(entry.runs) > 1
        ends = " ".join((f"<span class='sub'>{_esc(name)}</span> " if several else "")
                        + _chips([outcome for outcome in item.outcomes if outcome.field_id == spec.field_id])
                        for name, item, _ in walks)
        blocks.append(f"<details id='field-{_esc(spec.field_id)}'><summary>{_esc(spec.label)} {ends}</summary>"
                      f"{''.join(parts)}</details>")
    return "".join(blocks)


def form_page(entry: FormEntry, entries: list[FormEntry], config: Config, colors: dict[str, str] | None = None,
              text_match: TextMatch | None = None) -> str:
    """The page of one form: the runs' scores on it, its timeline, the model calls made on it, the form at the
    end, every field over time, the transcript and the form itself. ``colors`` maps a run's name to its colour
    in the report (default: the palette, in the order of this form's runs)."""
    mode = config.plots.mode
    first = entry.runs[0][1]
    conversation = first.data.conversations[entry.conversation]
    form, gold = first.data.form_of(entry.conversation), first.data.gold[entry.conversation]
    limit_s, power = config.score.speed_limit_s[entry.dataset], config.score.speed_power
    names = [name for name, _, _ in entry.runs]
    colors = colors or run_colors(names, mode)
    key = RunKey({name: name for name in names}, colors)
    said = gold_entries(form, gold)
    said_fields = {item.spec.field_id for item in said}
    lists = {field_id: facts.is_list for field_id, facts in gold_facts(form, gold).items()}
    accepted = {field_id: facts.accepted for field_id, facts in gold_facts(form, gold).items()}
    calls = {name: item.model_calls for name, _, item in entry.runs}
    changed = {name: [change.t for changes in item.changes.values() for change in changes] for name, _, item in entry.runs}
    walk_end = max(item.walk_end for _, _, item in entry.runs)
    # every plot of the page over the same stretch of time: until the form is read, or the last call or change
    end = max([walk_end] + [call.t for made in calls.values() for call in made] + [t for times in changed.values() for t in times])
    specs: list[dict[str, Any]] = []
    texts: list[str] = []  # the values the page shows cut short, in full: for its window on a value
    about = (f"{_esc(entry.dataset)}/{_esc(first.run['split'])} · {conversation.end:.0f} s of audio · form "
             f"“{_esc(form.name)}”: {len(said_fields)} of its {len(form.fields)} fields have something to show in "
             f"this conversation · {len(names)} run{'s' if len(names) > 1 else ''}")
    jumps = [("scores", "Scores")]
    body = [f"<h1>{_esc(entry.conversation)}</h1><p class='meta'>{about}</p>" + _key(names, colors),
            "<h2 id='scores'>Scores on this form</h2>" + _scores(entry, colors, config)]
    rows = timeline_rows(form, {name: item.walks for name, _, item in entry.runs}, walk_end)
    if rows:
        jumps.append(("timeline", "Timeline"))
        chart = timeline_chart(rows, timeline_marks(form, {name: item.changes for name, _, item in entry.runs}), names,
                               "The form over time",
                               "Per field: what it should show (grey), from the moment its words reached the "
                               "extractor, and under it what each run showed", end, mode, texts)
        body.append("<h2 id='timeline'>Timeline</h2><p class='lede'>Times are on the extractor's clock: a value "
                    "is said when the stream delivered its last word, about a second after it was spoken. A mark on "
                    "a run's line is a change of the field: ● its first fill, ▲ a revision, ✕ a clearing. A note is "
                    "graded at most every few seconds (its bars), but every rewrite of it has its mark. Rest the "
                    "pointer on a bar or a mark to read its value in full: the window stays open while the pointer "
                    "is on the bar or inside the window, where a long value scrolls.</p>"
                    + _chart(themed(chart, mode), specs))
    # a run that only counts its calls, without tokens or a price, has nothing to show here
    if any(call.eur or call.usage for made in calls.values() for call in made):
        jumps.append(("calls", "Model calls"))
        body.append("<h2 id='calls'>Model calls and price on this form</h2><p class='lede'>What each run used on "
                    "this conversation: its model calls that tried to fill the form (what they carried, cost and "
                    "took) and its other uses, which count into the price only. The plots share the timeline's "
                    "time axis.</p>" + _calls(entry, colors, config))
        for chart in (call_plots.price_so_far(calls, key, end, TIMELINE_WIDTH, mode),
                      call_plots.tokens_per_call(calls, key, end, TIMELINE_WIDTH, mode),
                      call_plots.calls_and_changes(calls, changed, key, end, TIMELINE_WIDTH, mode)):
            if chart is not None:
                body.append(_chart(themed(chart, mode), specs))
    jumps += [("end", "Form at the end"), ("matched", "How it was matched"), ("fields", "Field by field"),
              ("transcript", "Transcript"), ("form", "The form itself")]
    body.append("<h2 id='end'>The form at the end</h2><p class='lede'>Every field somebody talked about or a run "
                "filled: what it should hold, what each run shows, and the field's own completeness, precision and "
                "speed. A field's name leads to its values over time.</p>" + _fields(form, entry, colors, limit_s, power))
    body.append("<h2 id='matched'>How each value was matched</h2><p class='lede'>One row for every value the gold "
                "says a field should show, with the display the scoring gave it for each run. A value the form "
                "showed after it was said counts 1. Otherwise the closest display made after it was said counts by "
                "how close it is, and nothing counts 0. <b>Completeness</b> of a field is the mean of the scores in "
                "its rows. <b>Speed</b> is the mean of the speeds: 1 with no delay, 0 from "
                f"{limit_s:g} s on. <b>Precision</b> is not in this table: it grades every display for the seconds "
                "it stays on screen (Field by field, below).</p>"
                + _matched(form, entry, colors, lists, accepted, limit_s, power, texts, text_match))
    body.append("<h2 id='fields'>Field by field, over time</h2><p class='lede'>Open a field to see each of its "
                "gold values with the moment it was said (on the extractor's clock), and everything a run "
                f"displayed in it. {_FOLD.format('true', 'Open all')} · {_FOLD.format('false', 'Close all')}</p>"
                + _history(form, entry, colors, lists, text_match, accepted))
    body.append("<h2>The conversation</h2>"
                + _talk(conversation, form, entry, stream_config_of(first.run, config), lists, colors, texts))
    body.append("<h2 id='form'>The form itself</h2>" + _definition(form, said_fields))
    return _page(f"{entry.conversation} — evaluation report", _bar(entries, entry, "", "../report.html", jumps),
                 "".join(body), specs, mode, texts)


def write_report(result: Any, scores: list[RunScore], config: Config, text_match: TextMatch | None = None) -> Path:
    """results/<name>/report.html and results/<name>/forms/<dataset>-<conversation>.html; returns the former.
    ``text_match`` holds the judge's verdicts of the evaluation: with it, a form's page shows how each note
    compares with its facts."""
    entries = form_entries(scores)
    colors = run_key(result.summary, config.plots.mode).colors
    specs: list[dict[str, Any]] = []
    body, jumps = _overview(result, entries, config, specs)
    result.out.mkdir(parents=True, exist_ok=True)
    path = result.out / "report.html"
    path.write_text(_page("Evaluation report", _bar(entries, None, f"{FORMS_DIR}/", "report.html", jumps), body,
                          specs, config.plots.mode), encoding="utf-8")
    forms = result.out / FORMS_DIR
    forms.mkdir(exist_ok=True)
    for stale in forms.glob("*.html"):  # the forms of an earlier evaluation into the same folder
        stale.unlink()
    for entry in entries:
        (forms / entry.page).write_text(form_page(entry, entries, config, colors, text_match), encoding="utf-8")
    return path
