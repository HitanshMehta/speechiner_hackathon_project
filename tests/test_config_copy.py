"""A config without the sections that build the data (the hackathon's copy has none) scores runs exactly as the
full config does, and the message for missing data names what works where it is shown."""

import importlib.util

import yaml

from live_form_eval.config import BUILDING, REPO_ROOT, load_config
from live_form_eval.data.common import store


def test_a_config_without_the_building_sections_scores_the_same(tmp_path, monkeypatch):
    monkeypatch.delenv("LFE_CONFIG", raising=False)
    full = load_config()
    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({key: value for key, value in raw.items() if key not in BUILDING}), encoding="utf-8")
    copy = load_config(path)
    assert not copy.can_build_data()
    assert copy.fingerprint() == full.fingerprint()
    assert copy.scoring_settings() == full.scoring_settings()


def test_the_missing_data_message_names_what_works_here(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())  # the code that builds the data is here
    assert "lfe prepare" in store.how_to_get_the_data()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)  # a copy without it
    assert "unzip" in store.how_to_get_the_data() and "lfe prepare" not in store.how_to_get_the_data()
