"""The tables of an evaluation: one row per run, per submission, per field, per field type, per conversation,
per value and per model call. Every number comes from the metric files; this module only adds up and lays out."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from live_form_eval.config import Config
from live_form_eval.evaluation.evaluate import ConversationScore, RunScore
from live_form_eval.evaluation.metrics.calls import TOKEN_KINDS, UNSPLIT, Call, call_stats, unit_order
from live_form_eval.evaluation.metrics.completeness import completeness
from live_form_eval.evaluation.metrics.failure_rate import failure_rate
from live_form_eval.evaluation.metrics.final_decision_latency import final_decision_latency, unit_latency
from live_form_eval.evaluation.metrics.final_f1 import final_f1
from live_form_eval.evaluation.metrics.free_text_f1 import free_text_f1
from live_form_eval.evaluation.metrics.invented_rate import invented_rate
from live_form_eval.evaluation.metrics.precision import precision
from live_form_eval.evaluation.metrics.price import price
from live_form_eval.evaluation.metrics.score import TERMS, mean_terms
from live_form_eval.evaluation.metrics.speed import speed
from live_form_eval.evaluation.metrics.turn_slot_f1 import SlotCounts
from live_form_eval.evaluation.metrics.unanswerable import unanswerable_accuracy
from live_form_eval.evaluation.metrics.value_changes import Change, value_changes
from live_form_eval.evaluation.timeline.final import Outcome
from live_form_eval.evaluation.timeline.walk import FieldWalk

# calls.csv: one row per use. Its units follow, one column each (the amount), with their price in euros ("eur_" and
# the unit); "all_input_tokens" adds up the three kinds of input of a model call.
CALL_COLUMNS = ["run", "dataset", "conversation", "t", "model", "extraction", "eur", "all_input_tokens", "seconds",
                "ok", "changes"]


@dataclass
class EvalResult:
    summary: pd.DataFrame  # one row per run
    ranking: pd.DataFrame  # one row per submission: its runs combined over the datasets
    per_field: pd.DataFrame
    per_field_type: pd.DataFrame
    per_conversation: pd.DataFrame
    values: pd.DataFrame  # one row per value, option or item of every final form
    calls: pd.DataFrame  # one row per model call the runs reported
    out: Path

    def write(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        for name in ("summary", "ranking", "per_field", "per_field_type", "per_conversation", "values", "calls"):
            getattr(self, name).to_csv(self.out / f"{name}.csv", index=False)

    def summary_text(self) -> str:
        """The main numbers of each run, for the terminal."""
        lines = []
        for row in self.summary.to_dict("records"):
            latency_column = next(column for column in row if column.startswith("latency_p"))
            lines.append(f"{row['run']}  ({row['dataset']}/{row['split']}, {row['conversations']} conversations"
                         f"{'' if row['official'] else ', UNOFFICIAL'})")
            for label, key, style in (
                    ("score", "score", "{:.3f}"), ("  completeness", "completeness", "{:.3f}"),
                    ("  precision", "precision", "{:.3f}"), ("  speed", "speed", "{:.3f}"),
                    ("  outcome", "outcome", "{:.3f}"), ("final F1", "final_f1", "{:.3f}"),
                    ("free-text F1", "free_text_f1", "{:.3f}"), ("fixes per form", "fixes_per_form", "{:.2f}"),
                    ("latency median", "latency_median_s", "{:.2f} s"),
                    (f"latency p{latency_column.split('_p')[1].split('_')[0]}", latency_column, "{:.2f} s"),
                    ("never settled", "never_settled", "{:.1%}"), ("joint goal accuracy", "jga", "{:.1%}"),
                    ("revisions per field", "revisions_per_filled_field", "{:.2f}"),
                    ("model calls per form", "calls_per_form", "{:.1f}"),
                    ("  changed a value", "calls_changing_share", "{:.0%}"),
                    ("  input from cache", "cached_input_share", "{:.0%}"),
                    ("price per form", "eur_per_form", "€{:.4f}")):
                value = row.get(key)
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    lines.append(f"  {label:20} {style.format(value)}")
        return "\n".join(lines)


def _metrics(outcomes: Sequence[Outcome], walks: Sequence[FieldWalk], dataset: str, config: Config) -> dict[str, Any]:
    """The numbers every breakdown shares: final F1 counts, the terms of the walked fields, latency."""
    final = final_f1(outcomes)
    latency = final_decision_latency(outcomes, config.latency.percentile)
    limit_s, power = config.score.speed_limit_s[dataset], config.score.speed_power
    return {
        "correct": final.correct, "wrong": final.wrong, "missing": final.missing, "invented": final.invented,
        "correctly_empty": final.correctly_empty, "final_precision": final.precision, "final_recall": final.recall,
        "final_f1": final.f1, "completeness": completeness(walks), "precision": precision(walks),
        "speed": speed(walks, limit_s, power),
        "latency_median_s": latency.median_s, f"latency_p{config.latency.percentile:g}_s": latency.percentile_s,
        "never_settled": latency.never_settled, "shown_before_said": latency.early,
        "invented_rate": invented_rate(outcomes),
    }


def _call_numbers(calls: Sequence[Call], conversations: int, audio_s: float, config: Config) -> dict[str, Any]:
    """What the uses of a run, or of one form, carried, cost and took (metrics/calls): the model calls that tried
    to fill the form, and the other uses."""
    stats = call_stats(calls, conversations, audio_s, config.latency.percentile)
    return {
        "calls_per_audio_minute": stats.per_audio_minute, "calls_changing_share": stats.changing_share,
        "failed_calls": stats.failed, "call_seconds_median": stats.seconds_median,
        f"call_seconds_p{config.latency.percentile:g}": stats.seconds_percentile,
        "tokens_per_call": stats.tokens_per_call, "cached_input_share": stats.cached_share,
        **{f"{kind}_tokens_per_form": stats.tokens_per_form(kind) for kind in TOKEN_KINDS},
        "other_uses_per_form": stats.other_uses_per_form, "eur_other_per_form": stats.eur_other_per_form,
    }


def _change_numbers(fields: Iterable[Sequence[Change]]) -> dict[str, Any]:
    """How often the values of some fields changed on screen (metrics/value_changes)."""
    moved = value_changes(fields)
    return {"fields_filled": moved.fields, "value_changes": moved.changes, "revisions": moved.revised,
            "clears": moved.cleared, "revisions_per_filled_field": moved.revisions_per_field,
            "filled_once_share": moved.once_share}


def _run_row(score: RunScore, config: Config) -> dict[str, Any]:
    conversations = score.conversations
    n = len(conversations)
    outcomes = [outcome for item in conversations for outcome in item.outcomes]
    final = final_f1(outcomes)
    supports = [support for item in conversations for support in item.supports]
    free_text = free_text_f1(supports)
    left_empty, should_be_empty = unanswerable_accuracy(supports)
    slots = SlotCounts()
    for item in conversations:
        slots.add(item.slots)
    turns = sum(item.turns for item in conversations)
    audio_s = sum(item.audio_s for item in conversations)
    cost = price(score.calls, n, audio_s)
    terms = mean_terms(item.terms for item in conversations)  # each term: the mean over the forms
    row = {"run": score.run["run_id"], "name": score.run["name"], "dataset": score.dataset,
           "split": score.run["split"], "official": bool(score.run["official"]) and score.run_settings_match,
           "conversations": n, "score": terms.score}
    row.update(_metrics(outcomes, [walk for item in conversations for walk in item.walks.values()], score.dataset,
                        config))
    row.update({name: getattr(terms, name) for name in TERMS})
    row.update({
        "fixes_per_form": final.fixes / n if n else None,
        "jga": sum(item.turns_right for item in conversations) / turns if turns else None,
        "turn_slot_f1": slots.f1,
        "free_text_correct": free_text.correct, "free_text_complete": free_text.complete,
        "free_text_f1": free_text.f1,
        "unanswerable_accuracy": left_empty / should_be_empty if should_be_empty else None,
        "failure_rate": failure_rate(score.run["conversations"]),
        "eur_per_form": cost.eur_per_form, "eur_per_audio_minute": cost.eur_per_audio_minute,
        "calls_per_form": cost.calls / n if n else None,
    })
    row.update(_call_numbers([call for item in conversations for call in item.model_calls], n, audio_s, config))
    moved = _change_numbers(changes for item in conversations for changes in item.changes.values())
    row.update({
        "fields_filled_per_form": moved["fields_filled"] / n if n else None,
        "value_changes_per_form": moved["value_changes"] / n if n else None,
        "revisions_per_filled_field": moved["revisions_per_filled_field"],
        "filled_once_share": moved["filled_once_share"],
        "config_matches": score.config_matches, "judge": score.judge, "run_dir": str(score.run_dir),
    })
    # For information: how long the stream took to deliver the words (not part of any number above).
    delays = sorted(outcome.said - outcome.spoken for outcome in outcomes
                    if outcome.said is not None and outcome.spoken is not None)
    row["stream_delay_median_s"] = delays[len(delays) // 2] if delays else None
    return row


def _conversation_row(score: RunScore, item: ConversationScore, config: Config) -> dict[str, Any]:
    final = final_f1(item.outcomes)
    row = {"run": score.run["run_id"], "dataset": score.dataset, "conversation": item.conversation,
           "status": item.status, "audio_s": round(item.audio_s, 1), "score": item.terms.score}
    row.update(_metrics(item.outcomes, list(item.walks.values()), score.dataset, config))
    row.update({name: getattr(item.terms, name) for name in TERMS})
    row.update({"fixes": final.fixes, "jga": item.turns_right / item.turns if item.turns else None,
                "eur": item.eur, "calls": item.calls, "other_uses": item.other_uses})
    row.update(_call_numbers(item.model_calls, 1, item.audio_s, config))
    row.update(_change_numbers(item.changes.values()))
    return row


def _breakdown(score: RunScore, key: str, config: Config) -> list[dict[str, Any]]:
    """Rows per field ("field_id") or per field type ("field_type") of one run."""
    outcomes: dict[str, list[Outcome]] = {}
    walks: dict[str, list[FieldWalk]] = {}
    moves: dict[str, list[Sequence[Change]]] = {}
    for item in score.conversations:
        for outcome in item.outcomes:
            outcomes.setdefault(getattr(outcome, key), []).append(outcome)
        for field_id, walk in item.walks.items():
            group = field_id if key == "field_id" else score.field_types[field_id]
            walks.setdefault(group, []).append(walk)
        for field_id, changes in item.changes.items():
            group = field_id if key == "field_id" else score.field_types.get(field_id)
            moves.setdefault(group, []).append(changes)
    rows = []
    for group in sorted(outcomes):
        row = {"run": score.run["run_id"], "dataset": score.dataset, key: group}
        if key == "field_id":
            row["field_type"] = score.field_types.get(group)
        row.update(_metrics(outcomes[group], walks.get(group, []), score.dataset, config))
        row.update(_change_numbers(moves.get(group, [])))
        rows.append(row)
    return rows


def _value_rows(score: RunScore) -> list[dict[str, Any]]:
    rows = []
    for item in score.conversations:
        for outcome in item.outcomes:
            latency = unit_latency(outcome)
            rows.append({"run": score.run["run_id"], "dataset": score.dataset, "conversation": outcome.conversation,
                         "field_id": outcome.field_id, "field_type": outcome.field_type, "unit": outcome.unit,
                         "label": outcome.label, "outcome": outcome.outcome, "gold": outcome.gold,
                         "shown": outcome.shown, "spoken_s": outcome.spoken, "available_s": outcome.said,
                         "said_from": outcome.source, "settled_at_s": outcome.settled_at, "latency_s": latency})
    return rows


def _unit_column(unit: str) -> str:
    """The column of a unit's amount in calls.csv: its name, unless a column of the table is named so."""
    return f"unit_{unit}" if unit in CALL_COLUMNS else unit


def _call_rows(score: RunScore) -> list[dict[str, Any]]:
    """One row per use of a run (a model call that tried to fill the form, or another use): its amount and its
    price in euros of every unit, and the values that changed with it."""
    return [{"run": score.run["run_id"], "dataset": score.dataset, "conversation": item.conversation, "t": call.t,
             "model": call.model, "extraction": call.extraction, "eur": call.eur, "all_input_tokens": call.input,
             "seconds": call.seconds, "ok": call.ok, "changes": call.changes,
             **{_unit_column(unit): amount for unit, amount in call.usage.items()},
             **{f"eur_{unit}": eur for unit, eur in call.eur_by_unit.items()}}
            for item in score.conversations for call in item.model_calls]


def _calls_table(scores: Sequence[RunScore]) -> pd.DataFrame:
    """calls.csv: the fixed columns, then each unit's amount and each unit's price, in the report's unit order."""
    rows = [row for score in scores for row in _call_rows(score)]
    units = unit_order({unit for score in scores for item in score.conversations for call in item.model_calls
                        for unit in (*call.usage, *call.eur_by_unit)})
    amounts = [_unit_column(unit) for unit in units if unit != UNSPLIT] + [f"eur_{unit}" for unit in units]
    table = pd.DataFrame(rows, columns=CALL_COLUMNS + amounts)
    table[amounts] = table[amounts].fillna(0)  # a unit a use doesn't name: it used none of it
    return table


def _over_datasets(by_dataset: dict[str, float], weights: dict[str, float]) -> float | None:
    """The weighted average over the datasets that were run; None when none carries a weight."""
    total = sum(weights.get(name, 0.0) for name in by_dataset)
    return sum(value * weights.get(name, 0.0) for name, value in by_dataset.items()) / total if total else None


def _ranking(summary: pd.DataFrame, config: Config) -> pd.DataFrame:
    """One row per submission, best score first (official submissions before the others)."""
    rows = []
    for name, runs in summary.groupby("name", sort=False):
        records = runs.to_dict("records")

        def known(column: str, records=records) -> dict[str, float]:
            return {row["dataset"]: row[column] for row in records
                    if row[column] is not None and not math.isnan(row[column])}

        scores = known("score")
        row = {"name": name, "datasets": ", ".join(runs["dataset"]),
               "score": _over_datasets(scores, config.score.dataset_weights),
               **{f"score_{dataset}": value for dataset, value in scores.items()}}
        row.update({term: _over_datasets(known(term), config.score.dataset_weights) for term in TERMS})
        row.update({f"eur_per_form_{dataset}": value for dataset, value in known("eur_per_form").items()})
        row["official"] = bool(runs["official"].all())
        rows.append(row)
    ranking = pd.DataFrame(rows)
    return ranking.sort_values(["official", "score"], ascending=[False, False]).reset_index(drop=True)


def build_result(scores: Sequence[RunScore], config: Config, out: Path) -> EvalResult:
    summary = pd.DataFrame([_run_row(score, config) for score in scores])
    return EvalResult(
        summary=summary, ranking=_ranking(summary, config),
        per_field=pd.DataFrame([row for score in scores for row in _breakdown(score, "field_id", config)]),
        per_field_type=pd.DataFrame([row for score in scores for row in _breakdown(score, "field_type", config)]),
        per_conversation=pd.DataFrame([_conversation_row(score, item, config) for score in scores
                                       for item in score.conversations]),
        values=pd.DataFrame([row for score in scores for row in _value_rows(score)]),
        calls=_calls_table(scores), out=Path(out))
