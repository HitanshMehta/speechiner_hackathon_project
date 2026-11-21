"""The evaluation's clock: a value's time starts when the stream delivered its words, not when they were spoken."""

from conftest import words

from live_form_eval.data.common.model import Gold, GoldItem, GoldSnapshot, GoldValue
from live_form_eval.evaluation.timeline.arrival import Arrival, in_extractor_time
from live_form_eval.harness.stream import build_stream

SCRIPT = words([("caller", 0.0, "i need a train to ely on monday morning please."),
                ("agent", 4.0, "sure. what time would you like to leave?")])


def test_arrival_is_the_first_message_that_shows_the_word(stream_config):
    stream = build_stream(SCRIPT, stream_config, key="x")
    arrival = Arrival(stream)
    for word in SCRIPT:
        first = next(p.t for p in stream if any(w.start == word.start and w.text == word.text for w in p.words))
        assert arrival(word.end) == first
        assert arrival(word.end) >= word.end + stream_config.turn_close_s - 1e-6 or \
            arrival(word.end) >= word.end + stream_config.word_delay_s - 1e-6


def test_times_after_the_last_word_arrive_with_the_last_message(stream_config):
    stream = build_stream(SCRIPT, stream_config, key="x")
    arrival = Arrival(stream)
    assert arrival.end == stream[-1].t
    assert arrival(SCRIPT[-1].end + 30) == arrival.end


def test_gold_moves_to_extractor_time_and_keeps_the_spoken_time(stream_config):
    arrival = Arrival(build_stream(SCRIPT, stream_config, key="x"))
    ely, monday = SCRIPT[5], SCRIPT[7]
    gold = Gold(conversation_id="x",
                values=(GoldValue(field_id="to", value="ely", said=ely.end, source="span"),
                        GoldValue(field_id="day", value="monday", said=monday.end, source="span")),
                items=(GoldItem(field_id="services", text="train", said=SCRIPT[9].end, source="turn"),),
                snapshots=(GoldSnapshot(t=SCRIPT[9].end, values={"to": "ely"}),))
    moved = in_extractor_time(gold, arrival)
    assert [value.said for value in moved.values] == [arrival(ely.end), arrival(monday.end)]
    assert [value.detail["spoken"] for value in moved.values] == [ely.end, monday.end]
    assert all(value.said > value.detail["spoken"] for value in moved.values)  # the stream is never instant
    assert moved.items[0].said == moved.snapshots[0].t == arrival(SCRIPT[9].end)
