"""What a run used: model calls to fill the form, other uses, their units and prices, and value changes: the
numbers next to the price, and their plots."""

from __future__ import annotations

import pandas as pd
import pytest

from live_form_eval.evaluation.metrics.calls import (
    TOKEN_KINDS,
    UNSPLIT,
    call_stats,
    conversation_calls,
    unit_order,
    usage_of,
)
from live_form_eval.evaluation.metrics.value_changes import field_changes, value_changes
from live_form_eval.evaluation.report import call_plots
from live_form_eval.evaluation.report.style import SERIES, Measure, RunKey, run_colors, small_multiples, themed
from live_form_eval.evaluation.timeline.shown import Shown

PRICES = {"input_tokens": 1e-5, "cache_read_tokens": 1e-6, "output_tokens": 5e-5}


def record(t: float, **more) -> dict:
    """A line of calls.jsonl as runs made before units had prices wrote it: a total in euros, all input in one."""
    return {"conversation": "c1", "t": t, "model": "m", "eur": 0.001, "input_tokens": 100, "output_tokens": 10,
            "latency_s": 0.5, **more}


def use(t: float, extraction: bool = True, **usage) -> dict:
    """A line of calls.jsonl as session.report_usage writes it."""
    return {"conversation": "c1", "t": t, "model": "m" if extraction else "gate", "extraction": extraction,
            "usage": usage, "prices": {unit: PRICES[unit] for unit in usage},
            "eur": sum(amount * PRICES[unit] for unit, amount in usage.items()), "latency_s": 0.5, "ok": True}


def test_a_change_is_a_fill_a_revision_or_a_clearing():
    shown = Shown.from_updates([(1.0, "lon"), (2.0, "london"), (2.5, "london"), (4.0, ""), (6.0, "paris")])
    assert [(change.t, change.kind) for change in field_changes(shown)] == [
        (1.0, "filled"), (2.0, "revised"), (4.0, "cleared"), (6.0, "filled")]  # the repeat at 2.5 is no change
    assert field_changes(shown)[1].before == "lon"
    assert field_changes(Shown()) == []


def test_value_changes_add_up_over_fields():
    grown = field_changes(Shown.from_updates([(1.0, "9"), (2.0, "93"), (3.0, "935")]))
    once = field_changes(Shown.from_updates([(5.0, "monday")]))
    total = value_changes([grown, once, []])  # a field that never showed anything is not counted
    assert (total.fields, total.filled, total.revised, total.cleared, total.once) == (2, 2, 2, 0, 1)
    assert total.changes == 4
    assert total.revisions_per_field == 1.0
    assert total.once_share == 0.5
    nothing = value_changes([])
    assert nothing.revisions_per_field is None
    assert nothing.once_share is None


def test_a_change_goes_to_the_call_reported_nearest_to_it():
    records = [record(10.0), record(20.0), record(30.0)]
    # 10.0: at the call; 14.0: nearer to the first; 15.0: as near to both, so the earlier; 16.0: nearer to the
    # second; 5.0: before every call; 99.0: after the last
    calls = conversation_calls(records, [10.0, 14.0, 15.0, 16.0, 5.0, 99.0])
    assert [call.changes for call in calls] == [4, 1, 1]
    assert [call.changes for call in conversation_calls(list(reversed(records)), [29.0])] == [0, 0, 1]
    assert conversation_calls([], [1.0]) == []


def test_a_change_goes_to_the_nearest_model_call_never_to_another_use():
    # a model that decides whether to extract runs at 14 s: the change at 14 s still goes to the call of 10 s
    calls = conversation_calls([use(10.0, input_tokens=100), use(14.0, False, output_tokens=1),
                                use(20.0, input_tokens=100)], [14.0, 19.0])
    assert [(call.extraction, call.changes) for call in calls] == [(True, 1), (False, 0), (True, 1)]
    only_others = conversation_calls([use(5.0, False, output_tokens=1)], [5.0])
    assert [call.changes for call in only_others] == [0]  # no model call to fill the form: the change goes nowhere


def test_a_use_is_read_unit_by_unit():
    call = conversation_calls([use(3.0, False, input_tokens=300, cache_read_tokens=1000, output_tokens=20)], [])[0]
    assert (call.extraction, call.model, call.fresh_input, call.cache_read, call.output, call.input) == (
        False, "gate", 300, 1000, 20, 1300)
    assert call.eur_by_unit == pytest.approx({"input_tokens": 0.003, "cache_read_tokens": 0.001, "output_tokens": 0.001})
    assert call.eur == pytest.approx(0.005)
    seconds = {"conversation": "c1", "t": 1.0, "model": "stt", "extraction": False, "usage": {"audio_seconds": 12.5},
               "prices": {"audio_seconds": 0.0002}, "eur": 0.0025}
    assert usage_of(seconds) == ({"audio_seconds": 12.5}, {"audio_seconds": pytest.approx(0.0025)})
    assert unit_order(["audio_seconds", UNSPLIT, "output_tokens", "requests", "cache_read_tokens"]) == [
        "cache_read_tokens", "output_tokens", "audio_seconds", "requests", UNSPLIT]


def test_tokens_are_told_apart_by_kind():
    cached = conversation_calls([record(1.0, input_tokens=1000, cache_read_tokens=700, cache_write_tokens=100)], [])[0]
    assert (cached.cache_read, cached.fresh_input, cached.cache_write, cached.output) == (700, 200, 100, 10)
    assert cached.input == 1000
    # a run made before units had prices: every call tried to fill the form, its price is not split by unit
    assert (cached.extraction, cached.eur_by_unit, cached.eur) == (True, {UNSPLIT: 0.001}, 0.001)
    plain = conversation_calls([record(1.0)], [])[0]  # a run that says nothing about a cache: all input is fresh
    assert (plain.cache_read, plain.fresh_input, plain.cache_write) == (0, 100, 0)
    failed = conversation_calls([record(1.0, ok=False), record(2.0, error="timeout"), record(3.0, latency_s=None)], [])
    assert [call.ok for call in failed] == [False, False, True]
    assert failed[2].seconds is None


def test_call_stats_of_a_run():
    records = [record(10.0, input_tokens=1000, cache_read_tokens=600, latency_s=0.4),
               record(20.0, input_tokens=1000, cache_read_tokens=900, latency_s=0.8),
               record(30.0, input_tokens=2000, latency_s=1.2, ok=False),
               use(25.0, False, input_tokens=40, output_tokens=2)]  # another use: in the price only
    stats = call_stats(conversation_calls(records, [10.0, 10.0]), conversations=2, audio_seconds=120.0, p=90)
    assert (stats.calls, stats.changing, stats.failed, stats.other_uses) == (3, 1, 1, 1)
    assert stats.tokens == {"cache_read": 1500, "fresh_input": 2500, "cache_write": 0, "output": 30}
    assert stats.per_form == 1.5
    assert stats.per_audio_minute == 1.5
    assert stats.changing_share == pytest.approx(1 / 3)
    assert stats.cached_share == pytest.approx(1500 / 4000)
    assert stats.tokens_per_call == pytest.approx(4030 / 3)
    assert stats.tokens_per_form("fresh_input") == 1250
    assert stats.seconds_median == pytest.approx(0.8)
    assert stats.eur == pytest.approx(0.003 + 0.0005)
    assert (stats.eur_other, stats.other_uses_per_form, stats.eur_other_per_form) == (
        pytest.approx(0.0005), 0.5, pytest.approx(0.00025))
    assert list(stats.eur_by_unit) == ["input_tokens", "output_tokens", UNSPLIT]
    assert stats.eur_by_unit[UNSPLIT] == pytest.approx(0.003)
    empty = call_stats([], conversations=2, audio_seconds=120.0, p=90)
    assert (empty.calls, empty.changing_share, empty.cached_share, empty.seconds_median) == (0, None, None, None)
    assert empty.per_audio_minute == 0


def test_every_run_keeps_its_colour():
    colors = run_colors(["b", "a", "b", *(f"run {k}" for k in range(9))], "light")
    assert list(colors)[:2] == ["b", "a"]
    assert (colors["b"], colors["a"]) == (SERIES["light"][0], SERIES["light"][1])
    assert len(set(colors.values())) == len(SERIES["light"]) + 1  # past the last hue, the names share one grey
    key = RunKey({"id-a": "a", "id-b": "b"}, colors)
    scale = key.scale().to_dict()
    assert scale == {"domain": ["a", "b"], "range": [colors["a"], colors["b"]]}
    frame = key.frame(pd.DataFrame({"run": ["id-b", "id-a", "id-other"], "score": [0.5, 0.9, 0.1]}))
    assert frame["run"].tolist() == ["b", "a"]


def test_small_plots_leave_out_a_measure_without_values():
    key = RunKey({"id-a": "a", "id-b": "b"}, run_colors(["a", "b"], "light"))
    frame = pd.DataFrame({"run": ["id-a", "id-b"], "score": [0.9, 0.5], "price": [None, None], "calls": [3.0, None]})
    measures = [Measure("score", "Score", ".3f", (0, 1)), Measure("price", "Price"), Measure("calls", "Calls", ".1f"),
                Measure("absent", "Not a column")]
    chart = small_multiples(frame, measures, key, "Title", "Subtitle", "light")
    spec = themed(chart, "light").to_dict()
    assert [panel["title"]["text"] for panel in spec["hconcat"]] == ["Score", "Calls"]
    assert all(panel["encoding"]["y"]["scale"]["domain"] == ["a", "b"] for panel in spec["hconcat"])
    assert spec["hconcat"][1]["encoding"]["y"]["axis"] is None  # the runs are named once, on the left
    assert small_multiples(frame, [Measure("price", "Price")], key, "Title", "Subtitle", "light") is None


def test_the_plots_of_a_dataset_take_all_its_runs():
    key = RunKey({"id-a": "a", "id-b": "b"}, run_colors(["a", "b"], "light"))
    summary = pd.DataFrame({
        "run": ["id-a", "id-b"], "conversations": [2, 1], "calls_per_audio_minute": [28.0, 6.0],
        "calls_changing_share": [0.35, 0.8], "call_seconds_median": [0.6, 2.4], "tokens_per_call": [2700.0, 900.0],
        "cached_input_share": [0.7, None], "eur_per_audio_minute": [0.01, 0.002],
        "cache_read_tokens_per_form": [50_000.0, 0.0], "fresh_input_tokens_per_form": [20_000.0, 9_000.0],
        "cache_write_tokens_per_form": [0.0, 0.0], "output_tokens_per_form": [900.0, 400.0]})
    assert len(themed(call_plots.calls_by_run(summary, key, "light"), "light").to_dict()["hconcat"]) == 6
    mix = call_plots.token_mix(summary, key, "dark").to_dict()
    kinds = {row["kind"] for row in mix["datasets"][mix["data"]["name"]]}
    assert kinds == {"Read from cache", "Fresh input", "Output"}  # nothing was written to a cache: no such segment
    calls = pd.DataFrame({"run": ["id-a"] * 5 + ["id-b"] * 2, "conversation": ["c1", "c1", "c2", "c2", "c2", "c1", "c1"],
                          "t": [5.0, 65.0, 6.0, 66.0, 7.0, 30.0, 90.0],
                          "extraction": [True, True, True, True, False, True, True],  # another use: left out
                          "all_input_tokens": [1000, 3000, 1200, 3400, 99_999, 800, 800]})
    growth = call_plots.tokens_over_conversation(calls, key, "light").to_dict()  # 90 s: stretches of 10 s
    points = growth["datasets"][growth["data"]["name"]]
    assert {(row["run"], row["conversations"]) for row in points} == {("a", 2), ("b", 1)}
    assert sorted(row["tokens"] for row in points if row["run"] == "a") == [1100.0, 3200.0]
    forms = pd.DataFrame({"run": ["id-a", "id-b", "id-b"], "conversation": ["c1", "c1", "c2"], "audio_s": [120.0] * 3,
                          "eur": [0.02, 0.004, 0.0], "calls": [56, 12, 0], "other_uses": [3, 0, 0],
                          "page": ["forms/x-c1.html"] * 3})
    scatter = call_plots.price_against_length(forms, key, "light").to_dict()
    assert scatter["encoding"]["href"]["field"] == "page"
    assert len(scatter["datasets"][scatter["data"]["name"]]) == 2  # the form that cost nothing is left out
    assert call_plots.price_against_length(forms.assign(eur=0.0), key, "light") is None
    assert call_plots.token_mix(summary.drop(columns=[f"{kind}_tokens_per_form" for kind in TOKEN_KINDS]), key,
                                "light") is None
    # the price of a form unit by unit: run a split it, run b (made before units had prices) did not
    spent = pd.DataFrame({"run": ["id-a", "id-a", "id-b"], "eur_input_tokens": [0.004, 0.002, None],
                          "eur_output_tokens": [0.001, 0.001, None], f"eur_{UNSPLIT}": [None, None, 0.003]})
    units = call_plots.price_by_unit(spent, summary)
    assert [(row["run"], row["unit"], round(row["eur_per_form"], 4)) for row in units.to_dict("records")] == [
        ("id-a", "input_tokens", 0.003), ("id-a", "output_tokens", 0.001), ("id-b", UNSPLIT, 0.003)]
    money = call_plots.price_mix(units, key, "light").to_dict()
    assert money["encoding"]["color"]["scale"]["domain"] == ["Fresh input", "Output", "Not split by unit"]
    assert call_plots.price_mix(units.assign(eur_per_form=0.0), key, "light") is None


def test_the_plots_of_a_form_follow_its_uses():
    key = RunKey({"a": "a", "b": "b"}, run_colors(["a", "b"], "light"))
    # a model call that changed a value, another use (a model deciding whether to extract), a call that changed nothing
    runs = {"a": conversation_calls([use(10.0, input_tokens=100, output_tokens=10), use(15.0, False, output_tokens=4),
                                     use(20.0, input_tokens=200)], [10.0]), "b": []}
    price = call_plots.price_so_far(runs, key, 60.0, 860, "light").to_dict()
    assert len(price["vconcat"]) == 1  # one plot per run that spent something
    panel = price["vconcat"][0]
    assert panel["title"]["text"] == "a: €0.0037 (fresh input €0.0030, output €0.0007)"
    units, whole, hollow, filled, other = (price["datasets"][layer["data"]["name"]] for layer in panel["layer"])
    assert panel["layer"][0]["encoding"]["color"]["scale"]["domain"] == ["Fresh input", "Output"]
    assert [(row["t"], round(row["eur_so_far"], 6)) for row in units if row["unit"] == "input_tokens"] == [
        (0.0, 0.0), (10.0, 0.001), (15.0, 0.001), (20.0, 0.003), (60.0, 0.003)]
    assert [(row["t"], round(row["eur_so_far"], 6)) for row in whole] == [
        (0.0, 0.0), (10.0, 0.0015), (15.0, 0.0017), (20.0, 0.0037), (60.0, 0.0037)]
    assert panel["layer"][1]["mark"]["color"] == key.colors["a"]  # the whole price: in the run's colour
    assert ([row["use"] for row in hollow], [row["use"] for row in filled], [row["use"] for row in other]) == ([3], [1], [2])
    assert panel["layer"][4]["mark"]["shape"] == "square"
    assert panel["layer"][0]["encoding"]["x"]["scale"] == {"domain": [0, 60.0], "nice": False}
    old = call_plots.price_so_far({"a": conversation_calls([record(10.0)], [])}, key, 60.0, 860, "light").to_dict()
    assert old["vconcat"][0]["layer"][0]["encoding"]["color"]["scale"]["domain"] == ["Not split by unit"]

    tokens = call_plots.tokens_per_call(runs, key, 60.0, 860, "light").to_dict()
    assert [panel["title"]["text"] for panel in tokens["vconcat"]] == ["a"]  # the run without a call has no plot
    bars = [row for rows in tokens["datasets"].values() for row in rows]
    assert {row["kind"] for row in bars} == {"Fresh input", "Output"}
    assert {row["t"] for row in bars} == {10.0, 20.0}  # the other use is no model call
    so_far = call_plots.calls_and_changes(runs, {"a": [10.0], "b": [12.0, 30.0]}, key, 60.0, 860, "light").to_dict()
    rows = so_far["datasets"][so_far["data"]["name"]]
    assert [(row["t"], row["count"]) for row in rows if row["run"] == "b"] == [(0.0, 0), (12.0, 1), (30.0, 2), (60.0, 2)]
    assert [(row["t"], row["count"]) for row in rows if row["run"] == "a" and row["line"] == "Model calls so far"] == [
        (0.0, 0), (10.0, 1), (20.0, 2), (60.0, 2)]
    assert call_plots.price_so_far({"b": []}, key, 60.0, 860, "light") is None
    assert call_plots.tokens_per_call({"b": []}, key, 60.0, 860, "light") is None
    assert call_plots.calls_and_changes({"b": []}, {}, key, 60.0, 860, "light") is None
