"""What every plot and page of the report shares: colours, fonts, sizes, and how the runs are named and coloured.

Design rules: a run keeps one colour on every page of a report, and the runs are also named on an axis or in a
legend, so that colour is never the only way to tell them apart; a horizontal bar for every comparison; the
status colours for outcomes (correct, wrong, missing, invented), always with their names; greys for the parts of
a whole (the kinds of tokens), so that the colours stay with the runs; thin marks with a surface-coloured gap
between stacked segments, hairline grid, text in ink colours only, a tooltip on every mark. Every number in a
plot is also in a table of the report.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import altair as alt
import pandas as pd

alt.data_transformers.disable_max_rows()

SERIES = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
          "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]}
CHROME = {"light": {"surface": "#fcfcfb", "page": "#f9f9f7", "primary": "#0b0b0b", "secondary": "#52514e",
                    "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
          "dark": {"surface": "#1a1a19", "page": "#0d0d0d", "primary": "#f0efec", "secondary": "#c3c2b7",
                   "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"}}
STATUS = {"correct": "#0ca30c", "missing": "#fab219", "invented": "#ec835a", "wrong": "#d03b3b"}
# The parts of a whole, from the faintest to the strongest: four greys that stand apart on the mode's surface,
# also as bars a few pixels wide.
PARTS = {"light": ["#b8b6ad", "#8a8881", "#5a5854", "#1c1c1a"], "dark": ["#5c5c58", "#8a8983", "#b9b8b1", "#f0efec"]}
DATASET_NAMES = {"spokenwoz": "SpokenWOZ", "ami": "AMI"}
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
WIDTH, BAR, ROW, GUTTER = 560, 16, 28, 200  # plot width, bar thickness, row height, room for the row labels
SMALL = 118  # width of one of the small plots that stand next to each other


def themed(chart: Any, mode: str) -> Any:
    c = CHROME[mode]
    return (chart.configure(background=c["surface"], font=FONT, padding=16)
            .configure_view(stroke=None)
            .configure_axis(domainColor=c["axis"], gridColor=c["grid"], gridWidth=1, tickColor=c["axis"],
                            labelColor=c["secondary"], titleColor=c["secondary"], labelFontSize=11, titleFontSize=11,
                            titleFontWeight="normal", labelLimit=GUTTER)
            .configure_axisLeft(minExtent=GUTTER)
            .configure_header(labelColor=c["primary"], labelFontSize=12, titleColor=c["secondary"])
            .configure_legend(labelColor=c["secondary"], titleColor=c["secondary"], labelFontSize=11,
                              titleFontSize=11, titleFontWeight="normal", orient="top", symbolType="square",
                              labelLimit=320)
            .configure_title(color=c["primary"], subtitleColor=c["secondary"], fontSize=14, subtitleFontSize=11,
                             anchor="start", fontWeight=600))


def missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def run_colors(names: Iterable[str], mode: str) -> dict[str, str]:
    """A colour for every run name: the palette's hues in their fixed order, by first appearance. A report with
    more names than hues leaves the later ones grey; their labels tell them apart."""
    palette = SERIES[mode]
    return {name: palette[k] if k < len(palette) else CHROME[mode]["muted"]
            for k, name in enumerate(dict.fromkeys(names))}


@dataclass(frozen=True)
class RunKey:
    """The runs of a plot: how each is named on the pages and which colour it keeps.

    ``names`` maps a run id to its name, in the order the runs are listed; ``colors`` maps a name to its colour.
    """

    names: dict[str, str]
    colors: dict[str, str]

    @property
    def order(self) -> list[str]:
        """The names in the order the runs are listed, each once (a submission's runs on two datasets share one)."""
        return list(dict.fromkeys(self.names.values()))

    def scale(self) -> alt.Scale:
        return alt.Scale(domain=self.order, range=[self.colors[name] for name in self.order])

    def color(self, legend: bool = True, lines: bool = False) -> alt.Color:
        """The run as a colour. The legend is left out where the runs are named on an axis, and for one run;
        ``lines`` draws its symbols as strokes, for a plot of lines. The layers of one plot take the same call,
        so that their legends are one."""
        shown = None
        if legend and len(self.names) > 1:
            shown = alt.Legend(columns=4, symbolType="stroke", symbolStrokeWidth=2) if lines else alt.Legend(columns=4)
        return alt.Color("run:N", scale=self.scale(), title=None, legend=shown)

    def frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The rows of these runs, with each run's name in place of its id (column "run")."""
        kept = frame[frame["run"].isin(self.names)]
        return kept.assign(run=kept["run"].map(self.names))


@dataclass(frozen=True)
class Measure:
    """One of the small plots that stand next to each other: a column of a table with one row per run."""

    column: str
    title: str
    style: str = ".3f"  # how its numbers are written (a d3 format)
    domain: tuple[float, float] | None = None  # None = from zero to the largest value


def small_multiples(frame: pd.DataFrame, measures: Sequence[Measure], key: RunKey, title: str,
                    subtitle: str | list[str], mode: str) -> alt.HConcatChart | None:
    """One small bar chart per measure: the runs as rows, in the same order and colours in each. ``frame`` has one
    row per run (column "run" = its id). A measure no run has a value for is left out; None when none has one.
    Not themed yet."""
    rows = key.frame(frame)
    measures = [measure for measure in measures if measure.column in rows.columns and rows[measure.column].notna().any()]
    if rows.empty or not measures:
        return None
    c = CHROME[mode]
    panels = []
    for k, measure in enumerate(measures):
        scale = alt.Scale(domain=list(measure.domain)) if measure.domain else alt.Scale(zero=True, nice=True)
        # on the axis, a plain decimal number is written without its trailing zeros: 0.5 for 0.500
        ticks = measure.style[:-1] + "~f" if measure.style.endswith("f") and "," not in measure.style else measure.style
        panels.append(alt.Chart(rows).mark_bar(size=BAR, cornerRadiusEnd=3).encode(
            # every run keeps its row in every plot, also where it has no value
            y=alt.Y("run:N", title=None, scale=alt.Scale(domain=key.order), axis=alt.Axis() if k == 0 else None),
            x=alt.X(f"{measure.column}:Q", title=None, scale=scale, axis=alt.Axis(format=ticks, tickCount=3)),
            color=key.color(legend=False),
            tooltip=[alt.Tooltip("run:N", title="Run"),
                     alt.Tooltip(f"{measure.column}:Q", title=measure.title, format=measure.style)]).properties(
            width=SMALL, height=alt.Step(ROW),
            title=alt.Title(measure.title, fontSize=12, fontWeight=500, color=c["primary"], anchor="start")))
    return alt.hconcat(*panels, spacing=18).properties(title=alt.Title(title, subtitle=subtitle))
