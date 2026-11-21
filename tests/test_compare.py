"""What counts as the same value, per field type."""

import pytest

from live_form_eval.config import MatchingConfig
from live_form_eval.evaluation.matching.compare import options_of, same
from live_form_eval.interface.schema import FieldSpec, FieldType, OptionsConfig, RangeConfig

CONFIG = MatchingConfig(fuzzy_name_threshold=85)


def spec(kind: FieldType) -> FieldSpec:
    config = None
    if kind in (FieldType.select, FieldType.multiselect):
        config = OptionsConfig(options=("north", "south", "centre"))
    if kind == FieldType.slider:
        config = RangeConfig(range=(0, 5))
    return FieldSpec(field_id="f", label="F", prompt="", type=kind, field_configuration=config)


@pytest.mark.parametrize("kind, shown, gold, expected", [
    (FieldType.text, "Acorn Guest House", "acorn guest house", True),  # case
    (FieldType.text, "acorn guesthouse", "acorn guest house", True),  # spacing
    (FieldType.text, "the acorn guest house", "acorn guest house", True),  # an article more
    (FieldType.text, "live house", "limehouse", True),  # as the recogniser heard it
    (FieldType.text, "brittany chase", "brittany chason", True),
    (FieldType.text, "john", "john henley", False),  # a part of the name is not the name
    (FieldType.text, "guest house", "acorn guest house", False),
    (FieldType.text, "hilton", "acorn guest house", False),
    (FieldType.text, "FI06 UBJ", "fi06ubj", True),  # an identifier: exact once squashed
    (FieldType.text, "fi06ubx", "fi06ubj", False),  # … and never fuzzy
    (FieldType.text, "12345", "12346", False),
    (FieldType.phone, "0176 555-0142", "01765550142", True),
    (FieldType.phone, "01765550143", "01765550142", False),
    (FieldType.email, "Anna.Keller@Example.com", "anna.keller@example.com", True),
    (FieldType.url, "https://www.example.com/travel/", "example.com/travel", True),
    (FieldType.integer, 2, 2, True),
    (FieldType.integer, 3, 2, False),
    (FieldType.slider, 4.0, 4, True),
    (FieldType.checkbox, True, True, True),
    (FieldType.checkbox, False, True, False),
    (FieldType.select, "north", "north", True),
    (FieldType.select, "North", "north", False),  # an option must match exactly
    (FieldType.time, "9:30", "09:30", True),
    (FieldType.date, "2027-06-05", "2027-06-05", True),
    (FieldType.multiselect, ["south", "north"], ["north", "south"], True),
    (FieldType.multiselect, ["north"], ["north", "south"], False),
    (FieldType.template, {"name": "Anna", "email": ""}, {"name": "anna"}, True),
])
def test_same(kind, shown, gold, expected):
    assert same(spec(kind), shown, gold, CONFIG) is expected


def test_empty_values_only_equal_each_other():
    assert same(spec(FieldType.text), "", None, CONFIG)
    assert same(spec(FieldType.multiselect), [], None, CONFIG)
    assert not same(spec(FieldType.text), "x", None, CONFIG)
    assert not same(spec(FieldType.checkbox), None, False, CONFIG)  # "no" is a value; empty is not "no"


def test_options_of():
    assert options_of(None) == frozenset()
    assert options_of(["a", "b"]) == {"a", "b"}
