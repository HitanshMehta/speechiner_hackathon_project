"""evaluate(): run folders + gold → every metric, the ranking, tables and a report.

This file only orchestrates. Per conversation it rebuilds what each field showed over time (timeline/shown),
moves the gold's said times to the moment their words reached the extractor (timeline/arrival), reads the gold as
one step per stream message (timeline/steps), walks each field's displays along its steps (timeline/walk),
compares the final form with the gold (timeline/final, and timeline/free_text with a judge for free text), and
hands those to the metrics (metrics/*, one file each). Per run, report/tables adds the conversations up.

Which clock? The extractor's. A value's time starts when the stream delivered the words that say it, so the
simulated speech-to-text delay is not counted against the extractor.

When is "the end"? On SpokenWOZ the form is read at T_END = max(end of the input, last value + the speed limit):
later than that, the last value could no longer earn any speed. On AMI, where notes may take tens of seconds,
the form is read when the run ends (at most ``final_grace_s`` after the last partial).

Every field with gold goes through the walk, so it carries completeness, precision and speed. A typed field is
compared with the rules of its type (matching/compare, matching/similarity); a free-text field is compared by a
judge with the facts that hold at each moment (matching/free_text), on the texts that timeline/free_text keeps.

Next to the score, every visible change of a field is counted (metrics/value_changes) and the uses a run reported
are read with their units and prices (metrics/calls): its model calls that tried to fill the form, and its other
uses. Neither is ranked.

A conversation can also be scored as if it had ended earlier (``score_conversation(..., until=…)``): the same
steps on what was said and shown by then. The page of `lfe ui` draws the score over time with it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from live_form_eval.config import Config, StreamConfig, load_config
from live_form_eval.data.common.model import Conversation, Gold
from live_form_eval.data.common.store import Dataset, load_dataset
from live_form_eval.evaluation.matching.compare import SCALAR_TYPES, same
from live_form_eval.evaluation.matching.free_text import TextMatch
from live_form_eval.evaluation.matching.judge.nli import Judge, load_judge
from live_form_eval.evaluation.matching.similarity import similarity
from live_form_eval.evaluation.metrics.calls import Call, conversation_calls
from live_form_eval.evaluation.metrics.completeness import completeness
from live_form_eval.evaluation.metrics.joint_goal_accuracy import joint_goal_accuracy
from live_form_eval.evaluation.metrics.outcome import outcome
from live_form_eval.evaluation.metrics.precision import precision
from live_form_eval.evaluation.metrics.score import Terms
from live_form_eval.evaluation.metrics.speed import speed
from live_form_eval.evaluation.metrics.turn_slot_f1 import SlotCounts, turn_slot_counts
from live_form_eval.evaluation.metrics.value_changes import Change, field_changes
from live_form_eval.evaluation.timeline.arrival import Arrival, arrival_of, in_extractor_time, stream_config_of
from live_form_eval.evaluation.timeline.final import Outcome, typed_outcomes
from live_form_eval.evaluation.timeline.free_text import FieldSupport, free_text_support, gold_facts, looks
from live_form_eval.evaluation.timeline.shown import Shown, shown_fields
from live_form_eval.evaluation.timeline.steps import WALKED_TYPES, StepList, gold_steps, scoring_end
from live_form_eval.evaluation.timeline.walk import FieldWalk, walk_field
from live_form_eval.harness.recorder import read_run
from live_form_eval.interface.schema import FieldType, Form

FORM_READ_WHEN_RUN_ENDS = ("ami",)  # the other datasets are read at T_END (see the module docstring)


@dataclass
class ConversationScore:
    conversation: str
    status: str
    audio_s: float
    end_of_scoring: float
    walk_end: float  # until when displays count: T_END, or the end of the run where the form is read then
    outcomes: list[Outcome]
    walks: dict[str, FieldWalk]  # field → its walk (every field with gold values or displays)
    terms: Terms  # completeness, precision, speed and outcome of this form
    turns_right: int
    turns: int
    slots: SlotCounts
    supports: list[FieldSupport]  # free-text fields
    eur: float  # every use the run reported
    calls: int  # the uses that were extraction attempts
    other_uses: int  # … and the others
    changes: dict[str, list[Change]]  # field → every visible change of it (the fields that ever showed a value)
    # every use the run reported, in time order; an extraction attempt with the changes published nearest to it
    model_calls: list[Call]


@dataclass
class RunScore:
    run_dir: Path
    run: dict[str, Any]
    dataset: str
    field_types: dict[str, str]  # field id → type, over the run's forms
    data: Dataset  # the conversations, forms and gold the run was scored on (the report's form pages show them)
    conversations: list[ConversationScore] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    run_settings_match: bool = True  # the run's sample and stream settings are still the config's (official?)
    config_matches: bool = True  # the whole config is the one the run was made with (for information)
    judge: str | None = None


def same_run_settings(run: dict[str, Any], config: Config) -> bool:
    """Are the settings a run cannot change afterwards still the config's?

    Those are the sample of the run's dataset and the stream: they decide which conversations an extractor got
    and how. Every other setting (the score, the judge, how the gold is built) is applied when a run is
    evaluated, so a run stays official when they change: it is simply scored again.
    """
    recorded = run.get("config")
    if not recorded:  # a run folder without its settings: only the fingerprint of the whole config can be compared
        return run.get("config_fingerprint") == config.fingerprint()
    now = config.model_dump(mode="json")
    return (recorded.get("datasets", {}).get(run["dataset"]) == now["datasets"][run["dataset"]]
            and recorded.get("stream") == now["stream"])


def same_scoring_settings(run: dict[str, Any], config: Config) -> bool:
    """Is every setting that decides a score today the one the run was made with? (For information: a run is scored
    with today's settings anyway.)

    Compared section by section on today's scoring sections, so a section that no longer decides a score (one
    that only builds the data) doesn't make an older run look different."""
    recorded = run.get("config")
    if not recorded:  # a run folder without its settings: only the fingerprint can be compared
        return run.get("config_fingerprint") == config.fingerprint()
    return all(recorded.get(name) == value for name, value in config.scoring_settings().items())


def judge_store(config: Config) -> Path:
    """Where the judge keeps its verdicts from one evaluation to the next (a text pair is only judged once)."""
    return config.paths.results / ".cache"


class LazyMatch:
    """Loads the judge the first time a free-text field needs it (SpokenWOZ never does)."""

    def __init__(self, config: Config, judge: Judge | None, device: str | None = None) -> None:
        self._config, self._judge, self._device = config, judge, device
        self._match: TextMatch | None = None

    def get(self) -> TextMatch:
        if self._match is None:
            if self._judge is None:
                self._judge = load_judge(self._config.judge, self._device, judge_store(self._config))
                # the NLI judge says where it runs itself, once it has to load its model (matching/judge/nli.py)
                if not hasattr(self._judge, "device"):
                    print(f"free text is judged by {self._judge.name}", flush=True)
                elif self._judge.device:
                    print(f"free text is judged by {self._judge.name} on {self._judge.device}", flush=True)
            self._match = TextMatch(self._judge)
        return self._match


def walk_fields(form: Form, gold: Gold, steps: dict[str, StepList], shown: dict[str, Shown], end: float,
                config: Config, match: Callable[[], TextMatch] | None) -> dict[str, FieldWalk]:
    """The walk of every field that has gold values or displays, in form order.

    ``gold`` and ``steps`` are in extractor time. Free-text fields need the judge behind ``match``; without one
    they are left out. Their questions are put to the judge in one batch before the walks read the verdicts.
    """
    plans = []
    for spec in form.fields:
        timeline = shown.get(spec.field_id, Shown())
        if spec.type == FieldType.textarea:
            if match is None:
                continue
            displays = looks(timeline, config.judge.interval_s, end)
        elif spec.type in WALKED_TYPES:
            displays = list(timeline.updates)
        else:
            continue
        if steps.get(spec.field_id) or displays:
            plans.append((spec, steps.get(spec.field_id, []), displays))

    is_list: dict[str, bool] = {}
    accepted: dict[str, tuple[str, ...]] = {}  # field → the statements its answer may also make
    text_match: TextMatch | None = None
    if any(spec.type == FieldType.textarea for spec, _, _ in plans):
        text_match = match()
        field_facts = gold_facts(form, gold)
        is_list = {field_id: facts.is_list for field_id, facts in field_facts.items()}
        accepted = {field_id: facts.accepted for field_id, facts in field_facts.items()}
        questions = []
        for spec, history, displays in plans:
            if spec.type != FieldType.textarea:
                continue
            for k, (_, text) in enumerate(displays):  # a text can meet every gold value said before it is replaced
                replaced = displays[k + 1][0] if k + 1 < len(displays) else math.inf
                for said, facts in history:
                    if said < replaced:
                        questions += TextMatch.questions(text, facts, is_list[spec.field_id], accepted[spec.field_id])
        text_match.ask(questions)

    walks: dict[str, FieldWalk] = {}
    for spec, history, displays in plans:
        if spec.type == FieldType.textarea:
            walks[spec.field_id] = walk_field(
                history, displays, end,
                lambda a, b, lines=is_list[spec.field_id], more=accepted[spec.field_id]: text_match.same(a, b, lines, more),
                lambda a, b, lines=is_list[spec.field_id], more=accepted[spec.field_id]: text_match.similarity(
                    a, b, lines, more))
        else:
            walks[spec.field_id] = walk_field(
                history, displays, end, lambda a, b, s=spec: same(s, a, b, config.matching),
                lambda a, b, s=spec: similarity(s, a, b, config.matching))
    return walks


def score_conversation(conversation: Conversation, form: Form, gold: Gold, shown: dict[str, Shown],
                       calls: Sequence[dict[str, Any]], status: dict[str, Any], dataset: str, config: Config,
                       stream_config: StreamConfig, match: LazyMatch, until: float | None = None,
                       arrival: Arrival | None = None) -> ConversationScore:
    """One conversation of one run, scored.

    ``until`` scores it as if it had ended at that moment of the extractor's clock: only what was said and shown
    by then counts, and the form is read then. From the end of the scoring on, it changes nothing. Nothing
    official uses it; the page of `lfe ui` draws the score over time with it. ``arrival`` is the conversation's
    arrival map when the caller already has it (it is rebuilt otherwise).
    """
    arrival = arrival or arrival_of(conversation, stream_config)
    gold = in_extractor_time(gold, arrival)  # from here on, every gold time is when its words arrived
    steps = gold_steps(form, gold)
    limit_s, power = config.score.speed_limit_s[dataset], config.score.speed_power
    end_of_scoring = scoring_end(steps, arrival.end, limit_s)
    if dataset in FORM_READ_WHEN_RUN_ENDS:
        final_time = math.inf
        walk_end = status.get("run_end") or arrival.end + stream_config.final_grace_s
    else:
        final_time = walk_end = end_of_scoring
    if until is not None and until < walk_end:  # as if it had ended then: the gold said by then, the form read then
        final_time = walk_end = until
        steps = {field_id: said for field_id, field_steps in steps.items()
                 if (said := [step for step in field_steps if step[0] <= until])}

    def same_in(spec, a, b):
        return same(spec, a, b, config.matching)

    outcomes = typed_outcomes(conversation.id, form, gold, shown, same_in, final_time)
    walks = walk_fields(form, gold, steps, shown, walk_end, config, match.get)
    scalar = [spec for spec in form.fields if spec.type in SCALAR_TYPES]
    read_after_s = config.turn_metrics.read_after_s
    turns_right, turns = joint_goal_accuracy(gold.snapshots, shown, scalar, same_in, read_after_s)
    slots = turn_slot_counts(gold.snapshots, shown, scalar, same_in, read_after_s)
    supports: list[FieldSupport] = []
    if any(spec.type == FieldType.textarea for spec in form.fields):
        supports = free_text_support(conversation.id, form, gold, shown, match.get(), config.judge.interval_s,
                                     final_time, walk_end)
        outcomes += [outcome for support in supports for outcome in support.outcomes]
    terms = Terms(completeness=completeness(walks.values()), precision=precision(walks.values()),
                  speed=speed(walks.values(), limit_s, power), outcome=outcome(outcomes))
    changes = {field_id: field_changes(timeline) for field_id, timeline in shown.items() if timeline.updates}
    model_calls = conversation_calls(calls, sorted(change.t for items in changes.values() for change in items))
    return ConversationScore(
        conversation=conversation.id, status=status.get("status", "ok"), audio_s=conversation.end,
        end_of_scoring=end_of_scoring, walk_end=walk_end, outcomes=outcomes, walks=walks, terms=terms,
        turns_right=turns_right,
        turns=turns, slots=slots, supports=supports, eur=sum(call.eur for call in model_calls),
        calls=sum(call.extraction for call in model_calls),
        other_uses=sum(not call.extraction for call in model_calls), changes=changes, model_calls=model_calls)


def score_run(run_dir: Path, config: Config, match: LazyMatch) -> RunScore:
    run, predictions, calls = read_run(run_dir)
    dataset = load_dataset(config, run["dataset"], split=run["split"], ids=run["sample"]["ids"])
    stream_config = stream_config_of(run, config)
    by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in predictions:
        by_conversation[line["conversation"]].append(line)
    calls_by_conversation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        calls_by_conversation[call["conversation"]].append(call)
    statuses = {status["conversation"]: status for status in run["conversations"]}
    score = RunScore(run_dir=Path(run_dir), run=run, dataset=run["dataset"], calls=calls, data=dataset,
                     field_types={spec.field_id: spec.type.value for form in dataset.forms.values()
                                  for spec in form.fields},
                     run_settings_match=same_run_settings(run, config),
                     config_matches=same_scoring_settings(run, config))
    for conversation_id in sorted(dataset.conversations):
        conversation = dataset.conversations[conversation_id]
        score.conversations.append(score_conversation(
            conversation, dataset.form_of(conversation_id), dataset.gold[conversation_id],
            shown_fields(by_conversation[conversation_id], conversation_id), calls_by_conversation[conversation_id],
            statuses.get(conversation_id, {}), run["dataset"], config, stream_config, match))
    if any(item.supports for item in score.conversations):
        score.judge = match.get().judge.name
    return score


def evaluate(runs: Sequence[Path | str] | Path | str, *, config: Config | None = None, out: Path | str | None = None,
             judge: Judge | None = None, write: bool = True, device: str | None = None):
    """Score one or more run folders. Returns an ``EvalResult`` (tables as DataFrames, see report/tables.py).

    Pass one run per dataset of the same submission to get its combined ranking line, and the runs of several
    submissions to compare them: every table, plot and form page then shows them together, each run in its own
    colour. With ``write``, the tables, ``report.html`` and one page per form go to ``out`` (default:
    <results>/<the runs' name>, or <results>/comparison when the runs have several names). Free text needs the
    judge: it is loaded when no ``judge`` is given, on ``device`` ("cuda", "mps" or "cpu"; None = the best one
    there is), and it keeps its verdicts in <results>/.cache, so scoring the same texts again costs nothing.
    """
    from live_form_eval.evaluation.report.pages import write_report
    from live_form_eval.evaluation.report.tables import build_result

    config = config or load_config()
    if isinstance(runs, (str, Path)):
        runs = [runs]
    lazy = LazyMatch(config, judge, device)
    scores = [score_run(Path(run), config, lazy) for run in runs]
    names = list(dict.fromkeys(score.run["name"] for score in scores))
    out_dir = Path(out) if out else config.paths.results / (names[0] if len(names) == 1 else "comparison")
    result = build_result(scores, config, out_dir)
    if write:
        result.write()
        notes = any(item.supports for score in scores for item in score.conversations)  # a form with free text
        write_report(result, scores, config, lazy.get() if notes else None)
    return result
