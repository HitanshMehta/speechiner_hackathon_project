"""The simulated speech-to-text stream: turns, partials, timing and speaker labels."""

from itertools import pairwise

from conftest import words

from live_form_eval.harness.stream import PENDING, build_stream, cut_turns


def test_turn_ends_after_a_sentence_end_and_a_short_silence(stream_config):
    script = words([("caller", 0.0, "i need a train."), ("agent", 1.6, "where to?")])  # 0.45 s of silence
    assert [len(turn) for turn in cut_turns(script, stream_config)] == [4, 2]


def test_turn_does_not_end_mid_sentence(stream_config):
    script = words([("caller", 0.0, "i need a"), ("caller", 2.0, "train to ely.")])  # 1.15 s, no sentence end
    assert [len(turn) for turn in cut_turns(script, stream_config)] == [6]


def test_long_silence_ends_the_turn_anywhere(stream_config):
    script = words([("caller", 0.0, "i need a"), ("caller", 3.0, "train to ely.")])  # 2.15 s ≥ turn_max_silence_s
    assert [len(turn) for turn in cut_turns(script, stream_config)] == [3, 3]


def test_quick_reply_shares_the_turn(stream_config):
    script = words([("agent", 0.0, "for which day?"), ("caller", 0.95, "monday.")])  # 0.1 s after the question
    assert [len(turn) for turn in cut_turns(script, stream_config)] == [4]


def test_stream_is_ordered_and_complete(stream_config):
    script = words([("caller", 0.0, "i need a train to ely on monday morning please."),
                    ("agent", 4.0, "sure. what time would you like to leave?")])
    stream = build_stream(script, stream_config, key="x")
    assert all(a.t <= b.t for a, b in pairwise(stream))
    turns = [p for p in stream if p.kind == "turn"]
    assert " ".join(p.text for p in turns) == " ".join(w.text for w in script)
    for partial in stream:  # every delta of a turn starts the same way as its finished turn
        final = next(p for p in turns if p.turn_id == partial.turn_id)
        assert final.text.startswith(partial.text)


def test_words_appear_no_sooner_than_the_delay(stream_config):
    script = words([("caller", 0.0, "one two three four five six seven eight nine ten.")])
    first_seen = {}
    for partial in build_stream(script, stream_config, key="x"):
        if partial.kind == "partial":
            for word in partial.words:
                first_seen.setdefault(word.text, partial.t)
    assert first_seen  # the turn is long enough to get partials
    for word in script:
        if word.text in first_seen:
            assert first_seen[word.text] - word.end >= stream_config.word_delay_s - 1e-6


def test_finished_turn_arrives_after_its_last_word(stream_config):
    script = words([("caller", 0.0, "a train to ely.")])
    final = build_stream(script, stream_config, key="x")[-1]
    assert final.kind == "turn" and final.t == round(script[-1].end + stream_config.turn_close_s, 3)


def test_speaker_letters_follow_first_speech(stream_config):
    script = words([("agent", 0.0, "hello, how can i help you today?"), ("caller", 3.0, "i need a taxi to the station.")])
    turns = [p for p in build_stream(script, stream_config, key="x") if p.kind == "turn"]
    assert [p.speaker for p in turns] == ["A", "B"]  # the first to speak is A, whatever the dataset calls them
    assert {w.speaker for w in turns[0].words} == {"A"}


def test_short_audio_is_pending_and_partials_have_no_word_labels(stream_config):
    script = words([("caller", 0.0, "yes."), ("agent", 2.0, "okay, and for how many people is the booking?")])
    stream = build_stream(script, stream_config, key="x")
    assert stream[0].kind == "turn" and stream[0].speaker == PENDING  # 0.25 s of audio
    partial = next(p for p in stream if p.kind == "partial")
    assert all(word.speaker is None for word in partial.words)


def test_two_speakers_in_one_turn(stream_config):
    script = words([("agent", 0.0, "which day would you like to travel on?"),
                    ("caller", 2.45, "on monday morning please.")])
    final = next(p for p in build_stream(script, stream_config, key="x") if p.kind == "turn")
    assert final.speaker == "A"  # the dominant speaker
    assert final.labelled_text() == "A: which day would you like to travel on? B: on monday morning please."


def test_reply_under_a_second_inside_a_turn_is_pending(stream_config):
    script = words([("agent", 0.0, "which day would you like to travel on?"), ("caller", 2.45, "on monday.")])
    final = next(p for p in build_stream(script, stream_config, key="x") if p.kind == "turn")
    assert [word.speaker for word in final.words][-2:] == [PENDING, PENDING]  # too short to attribute


def test_labels_off_gives_the_plain_adapter_output(stream_config):
    script = words([("caller", 0.0, "i need a train to ely.")])
    stream = build_stream(script, stream_config.model_copy(update={"speaker_labels": False}), key="x")
    assert all(p.speaker is None and all(w.speaker is None for w in p.words) for p in stream)
    assert stream[-1].labelled_text() == "i need a train to ely."


def test_same_key_gives_the_same_stream(config):
    script = words([("caller", 0.0, "i need a train to ely on monday morning please.")])
    assert build_stream(script, config.stream, key="a") == build_stream(script, config.stream, key="a")
