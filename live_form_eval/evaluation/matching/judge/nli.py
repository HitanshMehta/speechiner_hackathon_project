"""The default judge for free text: a natural-language-inference model decides whether a text supports a statement.

``supports([(text, statement), …])`` is True where the model's most likely label is "entailment": the text says
the statement. "Neutral" (not mentioned) and "contradiction" both count as not supported. Statements are judged
one sentence or list item at a time, which is what makes long texts comparable (a whole-text comparison is all
or nothing).

A text that is too long for the model is read in windows. The model reads a limited number of tokens per
question (512 for the default model) and silently drops the end of a longer text, so a statement made near the
end of a long text would never be found. Such a text is cut into its units (sentences, or the lines of a list;
matching/units.py), and with ``t`` the number of tokens and ``room = limit − the model's own marks − t(statement)``::

    a window         as many units as fit in ``room`` tokens, at least one
    the next window  starts ``window_step`` of a window's units further (0.5: half a window), so two units that
                     are closer than that always share a window; the last window ends with the last unit
    the verdict      the text says the statement when one of its windows does

A question that fits is asked as before. A lone unit that is longer than the room is cut by the model, as before.

A text pair is only judged once: verdicts are kept in memory and, when a ``store`` folder is given, on disk
(``VerdictStore``). So evaluating a run again, or several runs that wrote the same notes, costs no model call.
The model itself is only loaded when a pair has to be judged that no earlier evaluation judged; telling whether
a pair fits only needs the model's tokenizer, and an environment that has neither can still read the verdicts
that were kept.

The model runs on a CUDA GPU when there is one, else on Apple's GPU, else on the CPU; ``device`` forces one of
them ("cuda", "mps", "cpu"). The device changes the speed, not what is asked: on 240 questions, an Apple GPU and
the CPU of the same machine gave the same verdicts (docs/kaggle.md has the speeds).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from live_form_eval.config import JudgeConfig
from live_form_eval.evaluation.matching.units import split_sentences

Pair = tuple[str, str]  # (text, statement)


class Judge(Protocol):
    name: str

    def supports(self, pairs: Sequence[Pair]) -> list[bool]:
        """For each (text, statement): does the text say the statement?"""


def windows(sizes: Sequence[int], room: int, step: float) -> list[tuple[int, int]]:
    """The windows over units of these token sizes, each as (first unit, one past its last unit).

    A window takes as many units as fit in ``room`` tokens (at least one). The next one starts ``step`` of the
    window's units further (at least one unit), and the last one ends with the last unit.
    """
    out: list[tuple[int, int]] = []
    start = 0
    while start < len(sizes):
        stop, used = start, 0
        while stop < len(sizes) and (stop == start or used + sizes[stop] <= room):
            used += sizes[stop]
            stop += 1
        out.append((start, stop))
        if stop >= len(sizes):
            break
        start += max(1, math.floor((stop - start) * step))
    return out


class VerdictStore:
    """One judge's verdicts on disk: a line per text pair (a digest of the pair, and the verdict).

    A pair that was read in windows is kept under a digest of its own (it names the window step), so a verdict
    from the time when the end of a long text was dropped is never taken for it.
    """

    def __init__(self, folder: Path | str, judge_name: str) -> None:
        self.path = Path(folder) / f"judge-{re.sub(r'[^A-Za-z0-9]+', '-', judge_name).strip('-')}.jsonl"
        self._verdicts: dict[str, bool] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as lines:
                for line in lines:
                    digest, verdict = json.loads(line)
                    self._verdicts[digest] = bool(verdict)

    @staticmethod
    def _digest(pair: Pair, reading: str = "") -> str:
        return hashlib.sha1(f"{pair[0]}\x1f{pair[1]}{reading}".encode()).hexdigest()

    def get(self, pair: Pair, reading: str = "") -> bool | None:
        """The verdict of an earlier evaluation; None when the pair was never judged in this ``reading`` ("" for
        a pair that fits the model, else how it is cut into windows)."""
        return self._verdicts.get(self._digest(pair, reading))

    def add(self, verdicts: dict[Pair, bool], readings: dict[Pair, str] | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as out:
            for pair, verdict in verdicts.items():
                digest = self._digest(pair, (readings or {}).get(pair, ""))
                self._verdicts[digest] = verdict
                out.write(json.dumps([digest, int(verdict)]) + "\n")


class NliJudge:
    def __init__(self, config: JudgeConfig, device: str | None = None, store: Path | str | None = None,
                 encoder: Any = None) -> None:
        """``encoder`` is a cross-encoder that is already loaded (the tests pass a stand-in); without one, the
        config's model is loaded when the first unjudged pair comes."""
        self.name = f"nli:{config.nli_model}"
        self._config = config
        self._wanted_device = device  # None: the best device this machine has
        self._model: Any = None  # loaded with the first pair that no earlier evaluation judged
        self._tokenizer: Any = None  # loaded with the first pair that may be too long for the model
        self._entailment = 0
        self.device: str | None = None  # where the model runs, once it is loaded
        self._cache: dict[Pair, bool] = {}
        self._store = VerdictStore(store, self.name) if store is not None else None
        self._reading = f"\x1fwindows:{config.window_step:g}"  # how a pair that is too long is kept in the store
        if encoder is not None:
            self._use(encoder)

    def _use(self, model: Any) -> None:
        self._model, self._tokenizer = model, model.tokenizer
        self.device = str(getattr(model, "device", None) or getattr(model, "_target_device", self._wanted_device))
        labels = {index: label.lower() for index, label in model.config.id2label.items()}
        self._entailment = next(index for index, label in labels.items() if label == "entailment")

    def _load(self) -> None:
        from sentence_transformers import CrossEncoder

        self._use(CrossEncoder(self._config.nli_model, device=self._wanted_device))
        print(f"free text is judged by {self.name} on {self.device}", flush=True)

    def _reader(self) -> Any:
        """The model's tokenizer. Without the model it is loaded alone: enough to tell whether a pair fits."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._config.nli_model)
        return self._tokenizer

    def _tokens(self, text: str) -> int:
        return len(self._reader()(text, add_special_tokens=False)["input_ids"])

    def _windows(self, pair: Pair) -> list[Pair] | None:
        """The questions a pair is read by when its text is too long for the model; None when the pair fits."""
        text, statement = pair
        reader = self._reader()
        limit = reader.model_max_length - reader.num_special_tokens_to_add(pair=True)
        if len(text) + len(statement) <= limit:
            return None  # a token is at least one character: a pair this short fits without being counted
        room = limit - self._tokens(statement)
        if self._tokens(text) <= room:
            return None
        units = split_sentences(text)
        if len(units) < 2 or room <= 0:
            return None  # nothing to cut by: the model drops the end, as it always did
        joiner = "\n" if "\n" in text.strip() else " "
        spans = windows([self._tokens(unit) for unit in units], room, self._config.window_step)
        return [(joiner.join(units[first:stop]), statement) for first, stop in spans]

    def supports(self, pairs: Sequence[Pair]) -> list[bool]:
        new = {pair for pair in pairs if pair not in self._cache and pair[0].strip() and pair[1].strip()}
        try:
            parts = {pair: self._windows(pair) for pair in new}  # None: the pair fits and is asked as it is
        except ImportError:  # an environment without the judge's libraries can still read the stored verdicts
            parts = {}
        if self._store is not None:  # what an earlier evaluation already judged
            for pair in list(new):
                verdict = self._store.get(pair, self._reading)  # a long text that was read in windows
                if verdict is None and parts.get(pair) is None:
                    verdict = self._store.get(pair)
                if verdict is not None:
                    self._cache[pair] = verdict
                    new.discard(pair)
        if new:
            if self._model is None:
                self._load()
            parts = {pair: parts[pair] if pair in parts else self._windows(pair) for pair in new}
            readings = {pair: self._reading for pair in new if parts[pair] is not None}
            # by length: a batch of questions of similar length needs little padding, so it is judged faster
            questions = sorted({question for pair in new for question in (parts[pair] or [pair])},
                               key=lambda question: (len(question[0]) + len(question[1]), question))
            scores = self._model.predict(questions, apply_softmax=True, batch_size=self._config.batch,
                                         show_progress_bar=False)
            said = {question: int(row.argmax()) == self._entailment for question, row in zip(questions, scores)}
            fresh = {pair: any(said[question] for question in (parts[pair] or [pair])) for pair in new}
            self._cache.update(fresh)
            if self._store is not None:
                self._store.add(fresh, readings)
        return [self._cache.get(pair, False) for pair in pairs]


def load_judge(config: JudgeConfig, device: str | None = None, store: Path | str | None = None) -> Judge:
    """The judge the config asks for. ``device`` and ``store`` only concern the NLI judge: where it runs (None =
    chosen automatically) and the folder it keeps its verdicts in (None = in memory only)."""
    if config.kind == "nli":
        return NliJudge(config, device, store)
    from live_form_eval.evaluation.matching.judge.llm_facts import LlmFactsJudge

    return LlmFactsJudge(config)
