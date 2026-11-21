#!/usr/bin/env python3
"""Print a Markdown audit of the saved Speechineer runs and bundled data.

From the project environment, run `python scripts/audit_baseline.py`.
The audit uses the checked-in scorer and an offline, in-memory NLI judge
when its weights are cached. It prints to stdout and reads local files only.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from itertools import islice
from pathlib import Path

# Set these before importing benchmark modules that may load the judge.
# ruff: noqa: E402
sys.dont_write_bytecode = True
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from live_form_eval.config import Config, load_config
from live_form_eval.data.common.store import load_dataset
from live_form_eval.evaluation.evaluate import LazyMatch, same_run_settings, score_conversation
from live_form_eval.evaluation.matching.judge.nli import NliJudge
from live_form_eval.evaluation.metrics.calls import call_stats
from live_form_eval.evaluation.metrics.final_decision_latency import final_decision_latency
from live_form_eval.evaluation.metrics.final_f1 import final_f1
from live_form_eval.evaluation.metrics.price import price
from live_form_eval.evaluation.metrics.score import mean_terms
from live_form_eval.evaluation.timeline.arrival import stream_config_of
from live_form_eval.evaluation.timeline.shown import shown_fields
from live_form_eval.harness.recorder import read_run
from live_form_eval.interface.schema import FieldType

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"spokenwoz": "test", "ami": "all"}
CATEGORICAL = {"select", "multiselect", "checkbox"}


def cached_judge(config: Config) -> tuple[NliJudge | None, str]:
    """Use the configured NLI model only if its snapshot is available locally."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    if config.judge.kind != "nli":
        return None, f"Configured judge `{config.judge.kind}` is outside this offline audit."
    model = config.judge.nli_model
    try:
        snapshot = Path(snapshot_download(model, local_files_only=True))
    except LocalEntryNotFoundError:
        return None, f"Configured NLI model `{model}` is absent from the local cache."
    weight_names = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin",
                    "pytorch_model.bin.index.json")
    if not any((snapshot / name).is_file() for name in weight_names):
        return None, f"Cached NLI snapshot for `{model}` lacks model weights."
    local_config = config.judge.model_copy(update={"nli_model": str(snapshot)})
    return NliJudge(local_config, device="cpu"), f"Free text scored locally with `{model}`."


def describe_data(dataset: str, split: str, processed: Path) -> None:
    folder = processed / dataset / split
    forms = json.loads((folder / "forms.json").read_text(encoding="utf-8"))
    print(f"\n### {dataset}/{split}")
    for form_id, form in forms.items():
        fields_by_type: dict[str, list[str]] = defaultdict(list)
        for field in form["fields"]:
            field_id = field["field_id"]
            if "options" in field:
                field_id += f" ({len(field['options'])} options)"
            fields_by_type[field["type"]].append(field_id)
        fields = len(form["fields"])
        categorical = sum(len(ids) for kind, ids in fields_by_type.items() if kind in CATEGORICAL)
        free_text = len(fields_by_type.get("textarea", ()))
        print(f"- `{form_id}`: {fields} fields, {categorical} categorical, "
              f"{fields - categorical - free_text} short slots, {free_text} free text.")
        for kind, ids in sorted(fields_by_type.items()):
            print(f"  - `{kind}` ({len(ids)}): " + ", ".join(f"`{field_id}`" for field_id in ids))
    with (folder / "conversations.jsonl").open(encoding="utf-8") as lines:
        sample = [json.loads(line) for line in islice(lines, 10)]
    print(f"- Transcript records read: {len(sample)} (first 10 lines or EOF).")
    for conversation in sample:
        form_id = conversation["meta"]["form_id"]
        print(f"  - `{conversation['id']}`: {len(conversation['words'])} timed words, "
              f"{conversation['end']:.3f} s, {len(conversation['roles'])} speaker roles, form `{form_id}`.")


def formatted(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec).replace("inf", "∞")


def audit_run(folder: Path, config: Config) -> None:
    run, predictions, calls = read_run(folder)
    dataset_name = run["dataset"]
    dataset = load_dataset(config, dataset_name, split=run["split"], ids=run["sample"]["ids"])
    stream = stream_config_of(run, config)
    statuses = {item["conversation"]: item for item in run["conversations"]}
    has_free_text = any(
        field.type == FieldType.textarea for form in dataset.forms.values() for field in form.fields
    )
    judge, judge_note = cached_judge(config) if has_free_text else (None, "")
    typed_only = has_free_text and judge is None
    match = LazyMatch(config, judge)
    scores = []
    for conversation_id, conversation in sorted(dataset.conversations.items()):
        form = dataset.form_of(conversation_id)
        if typed_only:
            fields = tuple(field for field in form.fields if field.type != FieldType.textarea)
            form = form.model_copy(update={"fields": fields})
        scores.append(score_conversation(
            conversation, form, dataset.gold[conversation_id], shown_fields(predictions, conversation_id),
            [call for call in calls if call["conversation"] == conversation_id],
            statuses[conversation_id], dataset_name, config, stream, match,
        ))

    model_calls = [call for score in scores for call in score.model_calls]
    extraction_calls = [call for call in model_calls if call.extraction]
    zero_change = [call for call in extraction_calls if call.changes == 0]
    audio_seconds = sum(score.audio_s for score in scores)
    p = config.latency.percentile
    stats = call_stats(model_calls, len(scores), audio_seconds, p)
    cost = price(calls, len(scores), audio_seconds)
    outcomes = [item for score in scores for item in score.outcomes]
    terms = mean_terms(score.terms for score in scores)
    final = final_f1(outcomes)
    decision = final_decision_latency(outcomes, p)
    avg_input = sum(call.input for call in extraction_calls) / stats.calls if stats.calls else None
    avg_output = stats.tokens["output"] / stats.calls if stats.calls else None
    zero_share = len(zero_change) / stats.calls if stats.calls else None
    official = run["official"] and same_run_settings(run, config)
    print(f"\n### {dataset_name}/{run['split']}")
    print(f"- `{run['run_id']}` · Conversations: {len(scores)}, prediction rows: {len(predictions)}, "
          f"{run['speed']:g}× playback, official settings: {'yes' if official else 'no'}.")
    print(f"- Recorded cost: €{cost.eur:.8f}, €{formatted(cost.eur_per_form, '.8f')}/form. "
          f"Reported uses: {len(calls)} ({stats.calls} extraction calls, {stats.failed} failed, "
          f"{stats.other_uses} other uses).")
    print(f"- Zero visible changes: {len(zero_change)}/{stats.calls} ({formatted(zero_share, '.2%')}), "
          f"€{sum(call.eur for call in zero_change):.8f} in logged spend.")
    print(f"- Mean tokens per extraction call: {formatted(avg_input, ',.2f')} input including cache, "
          f"{formatted(avg_output, ',.2f')} output.")
    print(f"- Model call latency: median {formatted(stats.seconds_median, '.3f')} s, "
          f"p{p:g} {formatted(stats.seconds_percentile, '.3f')} s.")
    if judge_note:
        print(f"- {judge_note}")
    if typed_only:
        print("- Full-form speed, final F1, outcome and final decision latency are unavailable.")
    scope = "Typed fields only" if typed_only else "Full form"
    print(f"- {scope}: speed {formatted(terms.speed, '.6f')}, "
          f"final F1 {formatted(final.f1, '.6f')}, form-mean outcome {formatted(terms.outcome, '.6f')}.")
    print(f"- {scope} final units: {final.correct} correct, {final.wrong} wrong, "
          f"{final.missing} missing, {final.invented} invented.")
    print(f"- {scope} final decision latency: median {formatted(decision.median_s, '.4f')} s, "
          f"p{p:g} {formatted(decision.percentile_s, '.4f')} s, "
          f"{formatted(decision.never_settled, '.2%')} never settled.")


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    print("## Saved baseline metrics")
    print("Each visible field change is attributed to the nearest extraction call in its conversation.")
    for dataset, split in DATASETS.items():
        folders = sorted((ROOT / "speechineer").glob(f"*-{dataset}"))
        if not folders:
            raise FileNotFoundError(f"No saved baseline for {dataset}/{split}")
        for folder in folders:
            audit_run(folder, config)

    print("\n## Bundled forms and transcript sample")
    for dataset, split in DATASETS.items():
        describe_data(dataset, split, config.paths.processed)


if __name__ == "__main__":
    main()
