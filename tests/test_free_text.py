"""Free text: the comparison by a judge, which texts are judged, and the walk of a free-text field."""

from types import SimpleNamespace

import pytest

from live_form_eval.data.common.model import Gold, GoldFact, GoldText
from live_form_eval.evaluation.evaluate import walk_fields
from live_form_eval.evaluation.matching.free_text import Support, TextMatch
from live_form_eval.evaluation.metrics.completeness import field_completeness
from live_form_eval.evaluation.metrics.precision import field_precision
from live_form_eval.evaluation.timeline.arrival import in_extractor_time
from live_form_eval.evaluation.timeline.free_text import looks
from live_form_eval.evaluation.timeline.shown import Shown
from live_form_eval.evaluation.timeline.steps import gold_steps
from live_form_eval.interface.schema import FieldSpec, FieldType, Form

YELLOW, WHEEL = "The remote is yellow.", "It has a wheel."
FORM = Form(form_id="f", name="f", fields=(
    FieldSpec(field_id="decisions", label="Decisions", prompt="", type=FieldType.textarea),
    FieldSpec(field_id="answer", label="Answer", prompt="", type=FieldType.textarea),
))


class WordJudge:
    """Supports a statement when the text holds all of its words; remembers each batch it was asked."""

    name = "test"

    def __init__(self):
        self.batches = []

    def supports(self, pairs):
        self.batches.append(list(pairs))
        return [set(statement.lower().split()) <= set(text.lower().split()) for text, statement in pairs]


def walk(config, updates, facts=((YELLOW, 10), (WHEEL, 40)), judge=None):
    """The walk of the list field "decisions": its facts with their said times, and what it showed until 200 s."""
    gold = Gold(conversation_id="c", texts=(GoldText(field_id="answer", text="", said=200, source="end"),),
                facts=tuple(GoldFact(field_id="decisions", text=text, said=said, source="llm") for text, said in facts))
    judge = judge or WordJudge()
    walks = walk_fields(FORM, gold, gold_steps(FORM, gold), {"decisions": Shown.from_updates(updates)}, 200.0, config,
                        lambda: TextMatch(judge))
    return walks.get("decisions")


# ---- the comparison

def test_a_text_is_compared_in_both_directions():
    match = TextMatch(WordJudge())
    both = f"{YELLOW}\n{WHEEL}"
    assert match.same(both, (YELLOW, WHEEL), is_list=True) and match.similarity(both, (YELLOW, WHEEL), True) == 1
    assert match.support(YELLOW, (YELLOW, WHEEL), True) == Support(1, 1, 2, 1)  # right, but a fact is missing
    assert match.similarity(YELLOW, (YELLOW, WHEEL), True) == pytest.approx(2 * 1 * 0.5 / 1.5)
    assert match.support(f"{YELLOW}\nIt is cheap.", (YELLOW,), True) == Support(2, 1, 1, 1)  # complete, one extra
    assert not match.same(f"{YELLOW}\nIt is cheap.", (YELLOW,), True)
    assert match.similarity("It is cheap.", (YELLOW,), True) == 0


def test_empty_sides():
    match = TextMatch(WordJudge())
    assert match.same(None, None, True) and match.similarity(None, None, True) == 1  # nothing to say, nothing said
    assert not match.same(None, (YELLOW,), True) and match.similarity(None, (YELLOW,), True) == 0
    assert not match.same(YELLOW, None, True) and match.similarity(YELLOW, None, True) == 0
    assert match.judge.batches == []  # none of these needs the judge


def test_a_list_is_read_by_lines_and_a_text_by_sentences():
    match = TextMatch(WordJudge())
    text = "The remote is yellow. It has a wheel."
    assert match.support(text, (YELLOW,), is_list=False) == Support(2, 1, 1, 1)  # the second sentence is extra
    assert match.support(text, (YELLOW,), is_list=True) == Support(1, 0, 1, 1)  # one item, and it says more


# ---- which texts are judged

def test_which_texts_of_a_field_are_judged():
    changes = Shown.from_updates([(5, "a"), (12, "b"), (40, "c"), (45, "d")])
    assert looks(changes, interval_s=30, end=200) == list(changes.updates)  # 4 changes in 200 s: all are judged
    # 4 changes in 60 s are more than one per 30 s: only what is on screen at 30 s and at the end is judged
    assert looks(changes, interval_s=30, end=60) == [(12, "b"), (45, "d")]
    assert looks(changes, interval_s=30, end=20) == [(12, "b")]  # what was shown after the end is not seen at all
    assert looks(Shown.from_updates([(5, "a"), (40, "b"), (45, "a")]), 30, 60) == [(5, "a")]  # the flash isn't seen
    assert looks(Shown.from_updates([(5, "a"), (40, "")]), 30, 200) == [(5, "a"), (40, None)]  # a clear is a display


# ---- the walk

def test_each_fact_shown_after_it_is_said(config):
    result = walk(config, [(15, YELLOW), (100, f"{YELLOW}\n{WHEEL}")])
    assert [(d.shown, d.kind, d.step) for d in result.displays] == [(15, "landing", 0), (100, "landing", 1)]
    assert [step.delay for step in result.steps] == [5, 60]
    assert field_completeness(result) == 1 and field_precision(result) == 1


def test_something_the_meeting_did_not_say(config):
    result = walk(config, [(15, f"{YELLOW}\nIt is cheap.")], facts=((YELLOW, 10),))
    assert result.displays[0].kind == "other" and result.displays[0].grade == pytest.approx(2 / 3)
    assert (result.steps[0].score, result.steps[0].delay) == (pytest.approx(2 / 3), 5)  # its closest display
    assert field_precision(result) == pytest.approx(2 / 3)


def test_a_fact_written_before_it_is_said(config):
    """Both facts at 15 s, the second 25 s early: it counts from 40 s on, and the 25 s are graded as a text that
    says one thing too many."""
    result = walk(config, [(15, f"{YELLOW}\n{WHEEL}")])
    assert [(d.shown, d.kind, d.kept, d.seconds) for d in result.displays] == [
        (15, "other", False, 25), (40, "landing", True, 160)]
    assert result.displays[0].grade == pytest.approx(2 / 3)
    assert [(step.score, step.delay) for step in result.steps] == [(pytest.approx(2 / 3), 5), (1, 0)]
    assert field_completeness(result) == pytest.approx((2 / 3 + 1) / 2)
    assert field_precision(result) == pytest.approx((2 / 3 * 25 + 160) / 185)


def test_a_field_that_should_stay_empty(config):
    result = walk(config, [(20, "Something.")], facts=())
    assert field_completeness(result) is None and field_precision(result) == 0
    assert walk(config, [], facts=()) is None  # nothing to say, nothing shown: no walk


def test_the_judge_gets_every_question_of_a_form_in_one_batch(config):
    judge = WordJudge()
    walk(config, [(15, f"{YELLOW}\n{WHEEL}"), (100, WHEEL), (150, f"{YELLOW}\n{WHEEL}")], judge=judge)
    assert len(judge.batches) == 1  # the walk found every verdict it needed
    assert len(judge.batches[0]) == len(set(judge.batches[0]))  # and no question twice


def test_without_a_judge_free_text_is_left_out(config):
    gold = Gold(conversation_id="c", facts=(GoldFact(field_id="decisions", text=YELLOW, said=10, source="llm"),))
    walks = walk_fields(FORM, gold, gold_steps(FORM, gold), {"decisions": Shown.from_updates([(15, YELLOW)])}, 200.0,
                        config, None)
    assert walks == {}


# ---- the extractor's clock

def test_a_fact_holds_from_the_arrival_of_its_words_to_the_arrival_of_those_that_end_it():
    gold = Gold(conversation_id="c", facts=(
        GoldFact(field_id="decisions", text="The remote is red.", said=10.0, until=40.0, source="llm"),
        GoldFact(field_id="decisions", text=YELLOW, said=40.0, source="llm")))
    red, yellow = in_extractor_time(gold, lambda spoken: spoken + 1.5).facts
    assert (red.said, red.until, red.detail["spoken"]) == (11.5, 41.5, 10.0)
    assert (yellow.said, yellow.until) == (41.5, None)


# ---- is a run still official?

def test_a_run_stays_official_when_only_the_scoring_changes(config):
    from live_form_eval.evaluation.evaluate import same_run_settings

    recorded = config.model_dump(mode="json", exclude={"paths", "plots"})  # what a run stores in run.json
    run = {"dataset": "spokenwoz", "config": recorded, "config_fingerprint": config.fingerprint()}
    assert same_run_settings(run, config)
    rescored = {**run, "config": {**recorded, "score": {**recorded["score"], "speed_power": 1.0}}}
    assert same_run_settings(rescored, config)  # the speed curve is applied when the run is evaluated
    other_sample = {**recorded["datasets"], "spokenwoz": {**recorded["datasets"]["spokenwoz"], "seed": 7}}
    assert not same_run_settings({**run, "config": {**recorded, "datasets": other_sample}}, config)
    assert same_run_settings({**run, "dataset": "ami", "config": {**recorded, "datasets": other_sample}}, config)
    other_stream = {**recorded["stream"], "word_delay_s": 0.9}  # the extractor got the words at other moments
    assert not same_run_settings({**run, "config": {**recorded, "stream": other_stream}}, config)
    assert same_run_settings({"dataset": "ami", "config_fingerprint": config.fingerprint()}, config)
    assert not same_run_settings({"dataset": "ami", "config_fingerprint": "another"}, config)


# ---- a text that is too long for the model

class Reader:
    """A stand-in tokenizer: a token per word, 12 tokens per question, 2 of them the model's own marks."""

    model_max_length = 12

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def num_special_tokens_to_add(self, pair=True):
        return 2


class Encoder:
    """A stand-in cross-encoder: it reads the first words of a question only, as the real one does, and says
    "entailment" when what it read of the text holds every word of the statement."""

    device = "cpu"
    tokenizer = Reader()

    def __init__(self):
        self.asked = []
        self.config = SimpleNamespace(id2label={0: "entailment", 1: "neutral"})  # as a real model has it

    def predict(self, questions, apply_softmax=True, batch_size=1, show_progress_bar=False):
        import numpy

        self.asked.append(list(questions))
        room = [self.tokenizer.model_max_length - 2 - len(statement.split()) for _, statement in questions]
        read = [set(text.split()[:max(space, 0)]) for (text, _), space in zip(questions, room)]
        return numpy.array([[1.0, 0.0] if set(statement.split()) <= seen else [0.0, 1.0]
                            for (_, statement), seen in zip(questions, read)])


def test_windows_take_what_fits_and_start_half_a_window_further():
    from live_form_eval.evaluation.matching.judge.nli import windows

    assert windows([3, 3, 3, 3, 3], room=7, step=0.5) == [(0, 2), (1, 3), (2, 4), (3, 5)]  # two units fit
    assert windows([2] * 6, room=8, step=0.5) == [(0, 4), (2, 6)]  # four fit: the next starts two further
    assert windows([2] * 6, room=8, step=1.0) == [(0, 4), (4, 6)]  # a whole window further: no overlap
    assert windows([9, 2, 2], room=7, step=0.5) == [(0, 1), (1, 3)]  # a unit that is too long stands alone
    assert windows([2, 2], room=8, step=0.5) == [(0, 2)]  # everything fits: one window


def test_a_statement_at_the_end_of_a_long_text_is_found(config, tmp_path, monkeypatch):
    from live_form_eval.evaluation.matching.judge.nli import NliJudge, VerdictStore

    long = "The remote is yellow. It has a wheel. It costs twelve euros. It is small. It is made of rubber."
    short = "It is cheap. It is small."
    encoder = Encoder()
    judge = NliJudge(config.judge, store=tmp_path, encoder=encoder)
    # the stand-in reads 12 − 2 − 3 = 7 words of a text next to a statement of 3 words: the long text has 19
    assert judge.supports([(long, "made of rubber."), (long, "It is green."), (short, "It is small.")]) == [True, False, True]
    asked = [question for batch in encoder.asked for question in batch]
    assert (short, "It is small.") in asked  # a pair that fits is asked as it is
    assert all(len(text.split()) <= 7 for text, statement in asked if statement == "made of rubber.")
    assert sum(statement == "made of rubber." for _, statement in asked) > 1  # … a long one in several windows
    assert encoder.predict([(long, "made of rubber.")])[0][0] == 0.0  # asked whole, the model would miss it

    again = NliJudge(config.judge, store=tmp_path, encoder=Encoder())  # the verdicts are kept, windows and all
    assert again.supports([(long, "made of rubber.")]) == [True] and again._model.asked == []
    # a verdict from before the windows (kept under the pair alone) is never taken for a long text
    VerdictStore(tmp_path, judge.name).add({(long, "It is green."): True})
    assert NliJudge(config.judge, store=tmp_path, encoder=Encoder()).supports([(long, "It is green.")]) == [False]

    def no_libraries(self):
        raise ImportError("transformers")

    monkeypatch.setattr(NliJudge, "_reader", no_libraries)  # an environment without the judge's libraries …
    bare = NliJudge(config.judge, store=tmp_path)
    assert bare.supports([(long, "made of rubber."), (short, "It is small.")]) == [True, True]  # … reads what was kept


# ---- where the judge runs

def test_the_judge_is_loaded_once_on_the_device_asked_for(config, monkeypatch, capsys):
    from live_form_eval.evaluation import evaluate as module

    class OnGpu(WordJudge):
        device = "cuda:0"

    asked = []
    monkeypatch.setattr(module, "load_judge",
                        lambda judge_config, device=None, store=None: asked.append(device) or OnGpu())
    lazy = module.LazyMatch(config, None, "cuda")
    assert lazy.get() is lazy.get() and asked == ["cuda"]
    assert "on cuda:0" in capsys.readouterr().out  # the evaluation says where free text is judged
    module.LazyMatch(config, WordJudge(), "cuda").get()  # a judge that is passed in is used as it is
    assert asked == ["cuda"]
