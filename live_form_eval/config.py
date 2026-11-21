"""config.yaml: how submissions are evaluated. Every key is required (no defaults in code) and documented in
docs/config.md. Nothing a submission tunes belongs here.

The sections that only build the data (``said_time``, ``atoms``, ``quotes``, ``key``, ``labels``: read by `lfe
prepare` and `lfe annotate`) may be left out as a whole: a copy that gets the data ready-made has no use for them.
They decide no score, so they are not part of the fingerprint either way."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = ("spokenwoz", "ami")
BUILDING = frozenset({"said_time", "atoms", "quotes", "key", "labels"})  # sections that only build the data
# sections that decide no score at evaluation time: where things are, how they look, how the data is built
NOT_SCORED = frozenset({"paths", "plots", "ui"}) | BUILDING


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Paths(_Strict):
    raw: Path  # downloaded datasets (outside the repo)
    processed: Path  # conversations, forms and gold built by `lfe prepare`
    audio: Path  # recordings fetched one by one when `lfe ui` plays a conversation with its sound
    runs: Path  # run folders
    results: Path  # evaluation results

    @model_validator(mode="after")
    def _resolve(self) -> Paths:
        for name in ("raw", "processed", "audio", "runs", "results"):
            path = Path(getattr(self, name)).expanduser()
            object.__setattr__(self, name, path if path.is_absolute() else (REPO_ROOT / path).resolve())
        return self


class SpokenWOZSample(_Strict):
    split: Literal["dev", "test"]
    n: int | None = Field(ge=1)  # None = every call of the split
    seed: int


class AMISample(_Strict):
    max_minutes: float = Field(gt=0)
    n: int | None = Field(ge=1)
    seed: int
    layers: tuple[Literal["summary", "topics", "qmsum", "mised"], ...]


class Datasets(_Strict):
    spokenwoz: SpokenWOZSample
    ami: AMISample


class StreamConfig(_Strict):
    partial_every_s: float = Field(gt=0)
    partial_jitter_s: float = Field(ge=0)
    word_delay_s: float = Field(ge=0)
    turn_pause_s: float = Field(gt=0)
    turn_max_silence_s: float = Field(gt=0)
    turn_close_s: float = Field(ge=0)
    max_turn_s: float = Field(gt=0)
    speaker_labels: bool
    pending_under_s: float = Field(ge=0)
    final_grace_s: float = Field(ge=0)
    seed: int

    @model_validator(mode="after")
    def _check(self) -> StreamConfig:
        if self.partial_jitter_s >= self.partial_every_s:
            raise ValueError("stream.partial_jitter_s must be below stream.partial_every_s")
        if self.turn_close_s < self.turn_pause_s:
            raise ValueError("stream.turn_close_s must be ≥ stream.turn_pause_s: a vendor can only close a turn "
                             "after it has heard the pause that ends it")
        if self.turn_max_silence_s < self.turn_pause_s:
            raise ValueError("stream.turn_max_silence_s must be ≥ stream.turn_pause_s")
        return self


class SaidTimeConfig(_Strict):
    matcher_model: str
    matcher_window_turns: int = Field(ge=1)
    matcher_batch: int = Field(ge=1)
    matcher_max_tokens: int = Field(ge=1)
    claude_model: str
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"]
    claude_max_tokens: int = Field(ge=256)


class AtomsConfig(_Strict):
    min_aligned: float = Field(ge=0, le=1)


class QuotesConfig(_Strict):
    """How the words a reader of AMI's answer key quotes are found in the transcript (data/ami/quotes.py)."""

    window: int = Field(ge=0)
    min_share: float = Field(gt=0, le=1)


class KeyReader(_Strict):
    """One of the two readers of an LLM annotation: AMI's answer key, SpokenWOZ's label check."""

    name: str
    model: str
    effort: Literal["low", "medium", "high", "xhigh", "max"]


class KeyConfig(_Strict):
    """How `lfe annotate --dataset ami` makes annotations/ami/key/ (data/ami/key.py)."""

    readers: tuple[KeyReader, KeyReader]
    lines_per_part: int = Field(ge=10)
    near_lines: int = Field(ge=0)
    questions_per_request: int = Field(ge=1)
    max_tokens: int = Field(ge=256)


class LabelsConfig(_Strict):
    """How `lfe annotate --dataset spokenwoz --labels` checks SpokenWOZ's labels, and which of them `lfe prepare`
    corrects (data/spokenwoz/labels.py)."""

    readers: tuple[KeyReader, KeyReader]
    max_tokens: int = Field(ge=256)
    sure_only: bool  # correct a label only when neither reader has a doubt


class MatchingConfig(_Strict):
    fuzzy_name_threshold: float = Field(ge=0, le=100)


class ScoreConfig(_Strict):
    speed_limit_s: dict[str, float]  # per dataset
    speed_power: float = Field(gt=0)
    dataset_weights: dict[str, float]

    @model_validator(mode="after")
    def _all_datasets(self) -> ScoreConfig:
        missing = [d for d in DATASETS if d not in self.speed_limit_s]
        if missing:
            raise ValueError(f"score.speed_limit_s needs a limit for {missing}")
        if any(limit <= 0 for limit in self.speed_limit_s.values()):
            raise ValueError("score.speed_limit_s: every limit must be above 0")
        return self


class TurnMetricsConfig(_Strict):
    read_after_s: float = Field(ge=0)


class LatencyConfig(_Strict):
    percentile: float = Field(gt=0, le=100)


class JudgeConfig(_Strict):
    kind: Literal["nli", "llm_facts"]
    nli_model: str
    llm_model: str | None
    interval_s: float = Field(gt=0)
    batch: int = Field(ge=1)
    window_step: float = Field(gt=0, le=1)


class PlotsConfig(_Strict):
    mode: Literal["light", "dark"]


class UiConfig(_Strict):
    """`lfe ui`: the page that plays one conversation and shows its form filling (docs/ui.md)."""

    host: str
    port: int = Field(ge=1, le=65535)
    extractor_folders: tuple[str, ...]  # searched with their subfolders, relative to where `lfe ui` is started
    skip_folders: tuple[str, ...]  # folder names never searched
    refresh_s: float = Field(gt=0)
    speeds: tuple[float, ...]
    audio_max_speed: float = Field(gt=0)
    score_every_s: float = Field(gt=0)
    unwatched_stop_s: float = Field(ge=0)
    stop_wait_s: float = Field(gt=0)  # how long a stopped run gets for its extractor's close()

    @model_validator(mode="after")
    def _check(self) -> UiConfig:
        if not self.extractor_folders:
            raise ValueError("ui.extractor_folders needs at least one folder")
        if 1 not in self.speeds or any(speed <= 0 for speed in self.speeds):
            raise ValueError("ui.speeds must hold 1 (real time) and only speeds above 0")
        return self


class Config(_Strict):
    paths: Paths
    datasets: Datasets
    stream: StreamConfig
    said_time: SaidTimeConfig | None = None  # the five BUILDING sections: absent in a copy that doesn't build data
    atoms: AtomsConfig | None = None
    quotes: QuotesConfig | None = None
    key: KeyConfig | None = None
    labels: LabelsConfig | None = None
    matching: MatchingConfig
    score: ScoreConfig
    turn_metrics: TurnMetricsConfig
    latency: LatencyConfig
    judge: JudgeConfig
    plots: PlotsConfig
    ui: UiConfig

    def scoring_settings(self) -> dict[str, Any]:
        """Every setting that decides a score, as a run stores them (paths, plots, `lfe ui` and the sections that
        build the data excluded)."""
        return self.model_dump(mode="json", exclude=set(NOT_SCORED))

    def can_build_data(self) -> bool:
        """Does this config hold the sections that build the data (`lfe prepare`, `lfe annotate`)?"""
        return all(getattr(self, name) is not None for name in BUILDING)

    def fingerprint(self) -> str:
        """Short hash of every setting that decides a score."""
        return hashlib.sha256(json.dumps(self.scoring_settings(), sort_keys=True).encode()).hexdigest()[:12]


def load_config(path: str | Path | None = None) -> Config:
    """The config at ``path``, else ``$LFE_CONFIG``, else the repo's config.yaml."""
    path = Path(path or os.environ.get("LFE_CONFIG") or REPO_ROOT / "config.yaml")
    return Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
