"""The score, on cases worked out by hand: similarity, the walk and the match, then the four terms."""

import math

import pytest

from live_form_eval.config import MatchingConfig
from live_form_eval.evaluation.matching.similarity import pair_similarity, pairs, similarity
from live_form_eval.evaluation.metrics.completeness import completeness, field_completeness
from live_form_eval.evaluation.metrics.outcome import outcome
from live_form_eval.evaluation.metrics.precision import field_precision, precision
from live_form_eval.evaluation.metrics.score import Terms, harmonic_mean, mean_terms
from live_form_eval.evaluation.metrics.speed import curve, field_speed, speed
from live_form_eval.evaluation.timeline.final import Outcome
from live_form_eval.evaluation.timeline.walk import walk_field
from live_form_eval.interface.schema import FieldSpec, FieldType, OptionsConfig

MATCHING = MatchingConfig(fuzzy_name_threshold=85)
BACK = [(0, "Jacques"), (5, "Jacques Ung"), (10, "Jacques"), (15, "Jacques Ung")]
LU = [(0, "Jacques"), (1, "Jacques Lu"), (2, "Jacques Lu Sac"), (3, "Jacques Lu Sac Dupont")]
NUMBER = [(1, "0"), (2, "01"), (3, "015"), (4, "0156"), (5, "01562"), (6, "015623")]  # one digit per message


def equal(shown, gold):
    return shown == gold


def written(shown, gold):
    return 0.0 if shown is None or gold is None else pair_similarity(shown.lower(), gold.lower())


def option(shown, gold):
    return float(shown == gold)


def walk(gold, shown, end=60, close=written):
    return walk_field(gold, shown, end, equal, close)


def spec(kind: FieldType) -> FieldSpec:
    config = OptionsConfig(options=("train", "taxi", "hotel")) if kind in (FieldType.select, FieldType.multiselect) \
        else None
    return FieldSpec(field_id="f", label="F", prompt="", type=kind, field_configuration=config)


# ---- similarity

def test_pairs_of_characters_with_the_start_marked():
    assert pairs("0172") == ["^0", "01", "17", "72"]
    assert pair_similarity("0172", "0171") == pytest.approx(2 * 3 / (4 + 4))
    assert pair_similarity("jacques", "jacques ung") == pytest.approx(2 * 7 / (7 + 11))
    assert pair_similarity("jacques l", "jacques lu") == pytest.approx(2 * 9 / (9 + 10))
    assert pair_similarity("gonzalo", "moritz") == 0


@pytest.mark.parametrize("kind, shown, gold, expected", [
    (FieldType.text, "Acorn Guest House", "acorn guest house", 1.0),  # the same value
    (FieldType.text, "jacques", "jacques ung", 2 * 7 / (7 + 11)),
    (FieldType.text, "FI06 UB", "fi06ubj", 2 * 6 / (6 + 7)),  # an identifier: letters and digits only
    (FieldType.phone, "0171 55", "0171555", 2 * 6 / (6 + 7)),
    (FieldType.time, "01:30", "13:30", 2 * 2 / (5 + 5)),
    (FieldType.select, "train", "taxi", 0.0),  # a choice is the same or not
    (FieldType.select, "taxi", "taxi", 1.0),
    (FieldType.integer, 3, 2, 0.0),
    (FieldType.multiselect, ["train", "hotel"], ["train", "taxi"], 2 * 1 / (2 + 2)),
    (FieldType.text, None, "jacques", 0.0),
    (FieldType.text, None, None, 1.0),
])
def test_similarity_by_field_type(kind, shown, gold, expected):
    assert similarity(spec(kind), shown, gold, MATCHING) == pytest.approx(expected)


# ---- the walk

def test_a_display_lands_on_the_most_recent_gold_value_it_equals():
    """The owner's case B: "Jacques Ung" at 16 s is the fourth gold value, and going back to "Jacques" is wrong."""
    result = walk(BACK, [(2, "Jacques"), (8, "Jacques Ung"), (12, "Gonzalo"), (16, "Jacques Ung"), (17, "Jacques")])
    assert [d.kind for d in result.displays] == ["landing", "landing", "other", "landing", "other"]
    assert [d.step for d in result.displays] == [0, 1, None, 3, None]
    assert [d.grade for d in result.displays] == pytest.approx([1, 1, 0, 1, 2 * 7 / (7 + 11)])
    assert [d.seconds for d in result.displays] == pytest.approx([6, 4, 4, 1, 43])
    third = result.steps[2]  # skipped: the closest display between the two landings, made after it was said
    assert (third.landed, third.display, third.delay) == (False, 3, 6)
    assert third.score == pytest.approx(2 * 7 / (7 + 11))
    assert field_completeness(result) == pytest.approx((3 + 14 / 18) / 4)
    assert field_precision(result) == pytest.approx((6 + 4 + 0 + 1 + 43 * 14 / 18) / 58)


def test_a_value_already_on_screen_counts_from_the_moment_it_is_said():
    """A good guess: wrong for the 8 s before the value is said, then right, with no delay."""
    result = walk([(10, "Moritz")], [(2, "Moritz")])
    assert [(d.shown, d.kind, d.grade, d.seconds, d.kept) for d in result.displays] == [
        (2, "other", 0, 8, False), (10, "landing", 1, 50, True)]
    assert (result.steps[0].landed, result.steps[0].delay) == (True, 0)
    assert field_precision(result) == pytest.approx(50 / 58)
    wrong_guess = walk([(10, "Moritz")], [(2, "Gonzalo"), (11, "Moritz")])
    assert [d.kept for d in wrong_guess.displays] == [False, False]  # nothing to keep: the guess was wrong
    on_time = walk([(10, "Moritz")], [(10, "Moritz")])
    assert [(d.kind, d.kept) for d in on_time.displays] == [("landing", False)]  # shown as it is said: one display


def test_a_value_that_comes_back_while_the_form_never_moved():
    result = walk(BACK[:3], [(1, "Jacques")])  # "Jacques", "Jacques Ung", then "Jacques" again
    assert [(d.shown, d.kind, d.step) for d in result.displays] == [(1, "landing", 0), (10, "landing", 2)]
    middle = result.steps[1]  # never shown: its closest display is the one that counts from 10 s
    assert (middle.landed, middle.display, middle.delay) == (False, 1, 5)
    assert field_completeness(result) == pytest.approx((1 + 14 / 18 + 1) / 3)


def test_a_name_completed_before_its_last_word_is_said():
    gold = [(10, "the"), (11.3, "the lensfield"), (12.6, "the lensfield hotel")]
    result = walk(gold, [(10.5, "the lensfield hotel")])
    assert [(d.shown, d.kind, d.kept) for d in result.displays] == [(10.5, "other", False), (12.6, "landing", True)]
    assert [step.display for step in result.steps] == [0, 1, 1]
    assert [step.score for step in result.steps] == pytest.approx([2 * 3 / (19 + 3), 2 * 13 / (19 + 13), 1])
    assert [step.delay for step in result.steps] == pytest.approx([0.5, 1.3, 0])


def test_a_value_shown_several_gold_values_ahead():
    """The whole name at 11.8 s, three words before it is complete: it counts for itself from 15.2 s on, and the
    values it jumped take it with the similarity of the whole name."""
    name = "cherry hinton hall and grounds"  # 30 characters
    gold = [(10, "cherry"), (11.3, "cherry hinton"), (12.6, "cherry hinton hall"), (13.9, "cherry hinton hall and"),
            (15.2, name)]
    result = walk(gold, [(11.8, name)])
    assert [(d.shown, d.kind, d.kept) for d in result.displays] == [(11.8, "other", False), (15.2, "landing", True)]
    assert result.displays[0].grade == pytest.approx(2 * 13 / (30 + 13))  # against "cherry hinton", said by then
    assert [step.display for step in result.steps] == [0, 0, 1, 1, 1]
    scores = [2 * 6 / (30 + 6), 2 * 13 / (30 + 13), 2 * 18 / (30 + 18), 2 * 22 / (30 + 22), 1]
    assert [step.score for step in result.steps] == pytest.approx(scores)
    assert [step.delay for step in result.steps] == pytest.approx([1.8, 0.5, 2.6, 1.3, 0])
    assert field_completeness(result) == pytest.approx(sum(scores) / 5)
    assert field_precision(result) == pytest.approx((2 * 13 / 43 * 3.4 + 44.8) / 48.2)
    waited = walk(gold, [(15.7, name)])  # the same form, shown once the name is complete
    assert field_completeness(waited) == pytest.approx(sum(scores) / 5)
    assert field_speed(result, 8, 0.5) > field_speed(waited, 8, 0.5)


def test_coming_back_to_the_value_the_form_reached_is_a_repair():
    result = walk([(10, "Moritz")], [(11, "Moritz"), (20, "Gonzalo"), (50, "Moritz")])
    assert [d.kind for d in result.displays] == ["landing", "other", "repair"]
    assert field_precision(result) == pytest.approx((9 * 1 + 30 * 0 + 10 * 1) / 49)
    assert result.steps[0].delay == 1


def test_going_back_after_the_next_value_is_wrong():
    result = walk([(0, "sunday"), (10, "monday")], [(1, "sunday"), (11, "monday"), (12, "sunday")], close=option)
    assert [d.kind for d in result.displays] == ["landing", "landing", "other"]
    assert field_completeness(result) == 1
    assert field_precision(result) == pytest.approx((10 + 1 + 0) / 59)


def test_a_clear_is_a_display():
    result = walk([(3, "Jacques Ung"), (28, "Moritz")], [(4, "Jacques Ung"), (5, None), (29, "Moritz")])
    assert [d.grade for d in result.displays] == [1, 0, 1]
    assert field_precision(result) == pytest.approx((1 + 0 * 24 + 31) / 56)
    withdrawn = walk([(1, "monday"), (5, None)], [(2, "monday"), (6, None)], close=option)
    assert [d.kind for d in withdrawn.displays] == ["landing", "landing"]  # clearing a withdrawn value is right


def test_a_field_nobody_talked_about():
    result = walk([], [(5, "Gonzalo")])
    assert field_completeness(result) is None and field_speed(result, 8, 0.5) is None
    assert field_precision(result) == 0


def test_displays_after_the_end_are_not_seen():
    result = walk([(1, "monday")], [(2, "monday"), (70, "tuesday")], close=option)
    assert len(result.displays) == 1 and result.displays[0].seconds == 58


# ---- the match

def test_a_skipped_value_takes_its_closest_display():
    """The owner's example: "Jacques Lu" is skipped and takes "Jacques L", not "Jacques Lu S"."""
    result = walk(LU[:3], [(0.1, "Jacques"), (1.1, "Jacques Lu S"), (1.5, "Jacques L"), (2.5, "Jacques Lu Sac")])
    skipped = result.steps[1]
    assert (skipped.landed, skipped.display) == (False, 2)
    assert skipped.score == pytest.approx(2 * 9 / (9 + 10)) and skipped.delay == pytest.approx(0.5)


def test_skipped_values_keep_the_order_of_the_displays():
    """Each on its own, "Jacques Lu" would take "Jacques L" (2.6 s), after the display of "Jacques Lu Sac"."""
    result = walk(LU, [(0.1, "Jacques"), (2.2, "Jacques Lu Sa"), (2.6, "Jacques L"), (3.5, "Jacques Lu Sac Dupont")])
    assert [step.display for step in result.steps] == [0, 1, 1, 3]
    assert result.steps[2].score == pytest.approx(2 * 13 / (13 + 14))
    assert result.steps[1].score == pytest.approx(2 * 10 / (13 + 10))


def test_reverse_order_when_the_favourites_cross():
    result = walk(NUMBER, [(6.5, "015624"), (7.5, "02")])
    assert [step.display for step in result.steps] == [0] * 6  # forwards, "0" would drag every value to "02"
    assert [step.score for step in result.steps] == pytest.approx([2 / 7, 4 / 8, 6 / 9, 8 / 10, 10 / 11, 10 / 12])


def test_no_landing_at_all():
    result = walk(NUMBER, [(3.5, "016"), (6.5, "016623")])
    assert not any(step.landed for step in result.steps)
    assert [step.display for step in result.steps] == [0, 0, 0, 1, 1, 1]
    assert [step.score for step in result.steps] == pytest.approx([2 / 4, 4 / 5, 4 / 6, 4 / 10, 6 / 11, 8 / 12])
    assert [step.delay for step in result.steps] == pytest.approx([2.5, 1.5, 0.5, 2.5, 1.5, 0.5])


def test_the_end_follows_the_same_rule():
    slow = walk(NUMBER, [(said + 1.5, value) for said, value in NUMBER[:5]])
    assert slow.steps[5].display == 4 and slow.steps[5].delay == pytest.approx(0.5)  # "01562", shown at 6.5 s
    assert slow.steps[5].score == pytest.approx(2 * 5 / (5 + 6))
    fast = walk(NUMBER, [(said + 0.5, value) for said, value in NUMBER[:5]])
    assert fast.steps[5].display is None and fast.steps[5].score == 0  # nothing was displayed after 6 s


def test_a_gold_value_that_comes_back():
    result = walk(BACK[:3], [(12, "Jacques")])
    assert result.displays[0].step == 2  # the most recent "Jacques"
    assert [step.landed for step in result.steps] == [False, False, True]
    assert [step.score for step in result.steps] == pytest.approx([1, 2 * 7 / (7 + 11), 1])
    assert [step.delay for step in result.steps] == [12, 7, 2]


# ---- the terms

def test_speed_curve():
    assert curve(0, 8, 0.5) == 1
    assert curve(2, 8, 0.5) == pytest.approx(math.sqrt(0.75))
    assert curve(8, 8, 0.5) == 0 and curve(20, 8, 0.5) == 0 and curve(None, 8, 0.5) == 0
    late = walk([(10, "Moritz")], [(15, "Moritz")])
    assert field_speed(late, 8, 0.5) == pytest.approx(math.sqrt(1 - 5 / 8))
    assert field_completeness(late) == 1 and field_precision(late) == 1  # lateness is charged in speed only


def test_terms_of_a_form_are_means_over_its_fields():
    said = walk([(10, "Moritz")], [(11, "Moritz")])  # complete, right, 1 s late
    silent = walk([(10, "monday")], [], close=option)  # said, never shown
    invented = walk([], [(5, "Gonzalo")])  # shown, never said
    fields = [said, silent, invented]
    assert completeness(fields) == pytest.approx((1 + 0) / 2)  # over the fields that have gold values
    assert speed(fields, 8, 0.5) == pytest.approx((math.sqrt(1 - 1 / 8) + 0) / 2)
    assert precision(fields) == pytest.approx((1 + 0) / 2)  # over the fields that displayed something
    assert completeness([invented]) is None and precision([silent]) is None


def test_outcome_is_the_final_f1_of_the_form():
    def unit(result):
        return Outcome("c", "f", "text", "value", "", result)

    units = [unit("correct")] * 2 + [unit("wrong"), unit("missing"), unit("invented")] + [unit("correctly_empty")] * 3
    assert outcome(units) == pytest.approx(2 * 2 / (2 * 2 + 3))
    assert outcome([unit("missing")]) == 0
    assert outcome([unit("correctly_empty")]) is None


def test_score_is_the_harmonic_mean():
    assert harmonic_mean([1, 0.5, 1, 1]) == pytest.approx(4 / 5)
    assert harmonic_mean([1, 0, None, 1]) == 0
    assert harmonic_mean([1, None, 1, 1]) is None
    forms = [Terms(1.0, 1.0, 0.8, 1.0), Terms(0.5, None, 0.4, 0.0)]
    run = mean_terms(forms)
    assert (run.completeness, run.precision, run.speed, run.outcome) == pytest.approx((0.75, 1.0, 0.6, 0.5))
    assert run.score == pytest.approx(4 / (1 / 0.75 + 1 / 1.0 + 1 / 0.6 + 1 / 0.5))
