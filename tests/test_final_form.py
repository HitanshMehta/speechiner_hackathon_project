"""The final form against the gold: outcomes, final F1, latency, free text; and the gold read as steps."""

import math

import pytest

from live_form_eval.config import MatchingConfig
from live_form_eval.data.common.model import Gold, GoldFact, GoldItem, GoldText, GoldValue
from live_form_eval.evaluation.matching.compare import same
from live_form_eval.evaluation.matching.free_text import TextMatch
from live_form_eval.evaluation.matching.units import split_items, split_sentences
from live_form_eval.evaluation.metrics.final_decision_latency import final_decision_latency, percentile
from live_form_eval.evaluation.metrics.final_f1 import final_f1
from live_form_eval.evaluation.metrics.free_text_f1 import free_text_f1
from live_form_eval.evaluation.metrics.invented_rate import invented_rate
from live_form_eval.evaluation.metrics.outcome import outcome
from live_form_eval.evaluation.metrics.unanswerable import unanswerable_accuracy
from live_form_eval.evaluation.timeline.final import typed_outcomes
from live_form_eval.evaluation.timeline.free_text import free_text_support
from live_form_eval.evaluation.timeline.shown import Shown
from live_form_eval.evaluation.timeline.steps import gold_steps, scoring_end
from live_form_eval.interface.schema import FieldSpec, FieldType, Form, OptionsConfig

MATCHING = MatchingConfig(fuzzy_name_threshold=85)
FORM = Form(form_id="f", name="f", fields=(
    FieldSpec(field_id="day", label="Day", prompt="", type=FieldType.select,
              field_configuration=OptionsConfig(options=("monday", "tuesday"))),
    FieldSpec(field_id="people", label="People", prompt="", type=FieldType.integer),
    FieldSpec(field_id="name", label="Name", prompt="", type=FieldType.text),
    FieldSpec(field_id="time", label="Time", prompt="", type=FieldType.time),
    FieldSpec(field_id="services", label="Services", prompt="", type=FieldType.multiselect,
              field_configuration=OptionsConfig(options=("train", "taxi", "hotel"))),
    FieldSpec(field_id="actions", label="Actions", prompt="", type=FieldType.textarea),
    FieldSpec(field_id="answer", label="Answer", prompt="", type=FieldType.textarea),
    FieldSpec(field_id="unanswerable", label="Unanswerable", prompt="", type=FieldType.textarea),
))
GOLD = Gold(conversation_id="c", values=(
    GoldValue(field_id="day", value="monday", said=10, source="span"),
    GoldValue(field_id="day", value="tuesday", said=40, source="span"),
    GoldValue(field_id="people", value=2, said=20, source="span"),
    GoldValue(field_id="name", value="anna keller", said=30, source="span"),
), items=(
    GoldItem(field_id="services", text="train", said=12, source="turn"),
    GoldItem(field_id="services", text="taxi", said=50, source="turn"),
    GoldItem(field_id="actions", text="Anna will book the train.", said=60, source="span"),
    GoldItem(field_id="actions", text="Ben will call the hotel.", said=70, source="span"),
), texts=(
    GoldText(field_id="answer", text="They chose the early train. It leaves at nine.", said=80, source="span"),
    GoldText(field_id="unanswerable", text="", said=100, source="end"),
), facts=(  # the lists and the texts above, as what the field should say over time
    GoldFact(field_id="actions", text="Anna will book the train.", said=60, source="span"),
    GoldFact(field_id="actions", text="Ben will call the hotel.", said=70, source="span"),
    GoldFact(field_id="answer", text="They chose the early train.", said=80, source="span"),
    GoldFact(field_id="answer", text="It leaves at nine.", said=80, source="span"),
))
SHOWN = {
    "day": Shown.from_updates([(11, "monday"), (45, "tuesday")]),  # right at the end, settled at 45
    "people": Shown.from_updates([(22, 3)]),  # wrong
    # name: never shown → missing
    "time": Shown.from_updates([(15, "09:00")]),  # nobody said a time → invented
    "services": Shown.from_updates([(13, ["train"]), (48, ["train", "hotel"])]),  # taxi missing, hotel invented
    "actions": Shown.from_updates([(65, "Anna will book the train."),
                                   (95, "Anna will book the train.\nSomeone will bring cake.")]),
    "answer": Shown.from_updates([(85, "They chose the early train.")]),
    "unanswerable": Shown.from_updates([(90, "The budget is ten euros.")]),
}


def outcomes():
    return typed_outcomes("c", FORM, GOLD, SHOWN, lambda spec, a, b: same(spec, a, b, MATCHING), final_time=100.0)


def test_typed_outcomes():
    result = {(o.field_id, o.label): o.outcome for o in outcomes()}
    assert result[("day", "tuesday")] == "correct"
    assert result[("people", "2")] == "wrong"
    assert result[("name", "anna keller")] == "missing"
    assert result[("time", "")] == "invented"
    assert result[("services", "train")] == "correct"
    assert result[("services", "taxi")] == "missing"
    assert result[("services", "hotel")] == "invented"


def test_final_f1_counts_and_fixes():
    final = final_f1(outcomes())
    assert (final.correct, final.wrong, final.missing, final.invented) == (2, 1, 2, 2)
    assert final.precision == pytest.approx(2 / 5) and final.recall == pytest.approx(2 / 5)
    assert final.fixes == 5
    assert invented_rate(outcomes()) == pytest.approx(2 / 2)  # time and hotel should both be empty
    assert outcome(outcomes()) == pytest.approx(2 * 2 / (2 * 2 + 5))  # the same counts, as one number per form


def test_latency_is_counted_from_the_said_time():
    by_field = {(o.field_id, o.label): o for o in outcomes()}
    assert by_field[("day", "tuesday")].settled_at == 45  # said at 40 → 5 s
    assert by_field[("services", "train")].settled_at == 13  # stayed ticked when "hotel" was added
    latency = final_decision_latency(outcomes(), p=90)
    assert latency.n == 5  # day, people, name, train, taxi
    assert latency.median_s == math.inf  # 3 of the 5 never settled: wrong, missing, missing
    assert latency.never_settled == pytest.approx(3 / 5)


def test_percentile_is_nearest_rank():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90) == 9
    assert percentile([1.0, math.inf], 50) == 1.0
    assert percentile([], 90) is None


class WordJudge:
    """Supports a statement when the text holds all of its words (enough for a test)."""

    name = "test"

    def supports(self, pairs):
        return [set(statement.lower().split()) <= set(text.lower().split()) for text, statement in pairs]


def test_free_text_in_both_directions_and_over_time():
    fields = {s.field_id: s for s in free_text_support("c", FORM, GOLD, SHOWN, TextMatch(WordJudge()), interval_s=30,
                                                       final_time=math.inf, end=100)}
    actions = fields["actions"]
    assert (actions.gold_units, actions.gold_supported) == (2, 1)  # Ben's task is missing
    assert (actions.shown_units, actions.shown_supported) == (2, 1)  # the cake is invented
    found = next(o for o in actions.outcomes if o.label.startswith("Anna"))
    assert found.outcome == "correct" and found.settled_at == 65  # supported from the first text on
    assert found.said == 60  # each fact has its own said time
    answer = fields["answer"]
    assert (answer.gold_units, answer.gold_supported, answer.shown_units, answer.shown_supported) == (2, 1, 1, 1)
    assert fields["unanswerable"].outcomes[0].outcome == "invented"
    assert unanswerable_accuracy(fields.values()) == (0, 1)
    score = free_text_f1(fields.values())
    assert score.correct == pytest.approx(2 / 3) and score.complete == pytest.approx(2 / 4)


def test_a_reference_without_facts_is_refused():
    stale = Gold(conversation_id="c", items=GOLD.items, texts=GOLD.texts)  # built before free text had a timeline
    with pytest.raises(RuntimeError, match="lfe prepare"):
        free_text_support("c", FORM, stale, SHOWN, TextMatch(WordJudge()), interval_s=30, final_time=math.inf, end=100)


def test_splitting():
    assert split_sentences("One thing. Another thing! A third?") == ["One thing.", "Another thing!", "A third?"]
    assert split_sentences("* It costs 12.50 euros. It is yellow.\n* Done") == [
        "It costs 12.50 euros.", "It is yellow.", "Done"]
    assert split_items("- first\n2. second\n\n• third") == ["first", "second", "third"]


def test_gold_steps_by_field_type():
    steps = gold_steps(FORM, GOLD)
    assert steps["day"] == [(10, "monday"), (40, "tuesday")]
    assert steps["services"] == [(12, ("train",)), (50, ("train", "taxi"))]  # the options named so far
    assert steps["actions"] == [(60, ("Anna will book the train.",)),  # the facts that hold
                                (70, ("Anna will book the train.", "Ben will call the hotel."))]
    assert steps["answer"] == [(80, ("They chose the early train.", "It leaves at nine."))]
    assert "time" not in steps and "unanswerable" not in steps  # no gold
    assert scoring_end(steps, input_end=45.0, grace_s=8.0) == 88.0  # 8 s after the last value (the answer at 80)
    assert scoring_end(steps, input_end=90.0, grace_s=8.0) == 90.0


def test_gold_steps_of_free_text_follow_what_the_conversation_changed():
    def fact(text, said, until=None):
        return GoldFact(field_id="actions", text=text, said=said, until=until, source="llm")

    revised = Gold(conversation_id="c", facts=(fact("The remote is red.", 10, until=40), fact("It has a wheel.", 20),
                                               fact("The remote is yellow.", 40)))
    assert gold_steps(FORM, revised)["actions"] == [
        (10, ("The remote is red.",)), (20, ("The remote is red.", "It has a wheel.")),
        (40, ("It has a wheel.", "The remote is yellow."))]  # red ends in the message that brings yellow
    dropped = Gold(conversation_id="c", facts=(fact("The remote is red.", 10, until=40),))
    assert gold_steps(FORM, dropped)["actions"] == [(10, ("The remote is red.",)), (40, None)]
    confirmed = Gold(conversation_id="c", facts=(fact("The remote is red.", 5, until=10),
                                                 fact("The remote is blue.", 10, until=20),
                                                 fact("The remote is red.", 20)))
    assert [value for _, value in gold_steps(FORM, confirmed)["actions"]] == [
        ("The remote is red.",), ("The remote is blue.",), ("The remote is red.",)]


def test_gold_entries_that_arrive_with_one_message_are_one_step():
    """Three words of a number: the first arrives alone, the next two with the same message."""

    def digit(value, said, spoken):
        return GoldValue(field_id="name", value=value, said=said, source="span", detail={"spoken": spoken})

    gold = Gold(conversation_id="c", values=(digit("0", 1.0, 0.4), digit("01", 2.3, 0.9), digit("015", 2.3, 1.2)))
    assert gold_steps(FORM, gold)["name"] == [(1.0, "0"), (2.3, "015")]
    back = Gold(conversation_id="c", values=(digit("0", 1.0, 0.4), digit("01", 2.3, 0.9), digit("0", 2.3, 1.2)))
    assert gold_steps(FORM, back)["name"] == [(1.0, "0")]  # said and taken back within one message: no step
