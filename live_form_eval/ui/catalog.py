"""What the page lets you choose: the forms of the config's sample, the extractors in the folder, the runs.

Extractors are found by reading the Python files, not by importing them: a class counts when it subclasses
``Extractor`` (directly, or through another class that does) and it or one of those classes has a ``run``
method. Reading is cheap, runs none of a submission's code and needs none of its dependencies, so the list can be
refreshed every few seconds: a new file, a new class or a renamed one shows up while the page is open.
"""

from __future__ import annotations

import ast
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from live_form_eval.config import Config
from live_form_eval.data.common.model import Conversation, Gold
from live_form_eval.data.common.store import Dataset, load_dataset
from live_form_eval.evaluation.evaluate import same_run_settings
from live_form_eval.evaluation.timeline.steps import gold_steps
from live_form_eval.interface.schema import Form

_BASE = "Extractor"  # the interface's class: what every extractor subclasses


@dataclass(frozen=True)
class ExtractorEntry:
    """One extractor class found in a file, or a file that can't be read as Python (``error``)."""

    id: str  # what a run is started with: "module:Class", or "path/to/file.py:Class" when the path is no module
    name: str  # the class
    file: str  # its file, relative to the folder `lfe ui` was started in
    about: str  # the first line of its docstring
    error: str | None = None  # why the file can't be used (a syntax error); then there is no class


@dataclass(frozen=True)
class RunEntry:
    id: str
    name: str
    dataset: str
    split: str
    speed: float
    official: bool  # made on the official sample in real time, and those settings are still the config's
    started_at: str
    conversations: tuple[tuple[str, str], ...]  # (conversation id, how it ended)


@dataclass(frozen=True)
class _Class:
    file: str
    name: str
    bases: tuple[str, ...]
    has_run: bool
    about: str


def _base_name(node: ast.expr) -> str | None:
    """The name a base class is written with: ``Extractor`` and ``live_form_eval.Extractor`` → "Extractor"."""
    if isinstance(node, ast.Name):
        return node.id
    return node.attr if isinstance(node, ast.Attribute) else None


def _classes(path: Path, file: str) -> list[_Class]:
    """The classes a file defines at its top level. Raises SyntaxError for a file that isn't valid Python."""
    found = []
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            # what the class says about itself, else what its file says
            doc = (ast.get_docstring(node) or ast.get_docstring(tree) or "").strip().splitlines()
            found.append(_Class(
                file=file, name=node.name, bases=tuple(name for name in map(_base_name, node.bases) if name),
                has_run=any(isinstance(item, (ast.AsyncFunctionDef, ast.FunctionDef)) and item.name == "run"
                            for item in node.body),
                about=doc[0] if doc else ""))
    return found


def _module(file: str) -> str | None:
    """The module a file is imported as from the folder it was found under; None when its path is no module name
    (a folder or file name that is not a Python identifier)."""
    parts = Path(file).with_suffix("").parts
    return ".".join(parts) if all(part.isidentifier() for part in parts) else None


class ExtractorFinder:
    """Finds the extractor classes under a folder. A file is only read again when it has changed."""

    def __init__(self, root: Path, folders: tuple[str, ...], skip: tuple[str, ...]) -> None:
        self.root = Path(root).resolve()
        self._folders = folders
        self._skip = set(skip)
        self._read: dict[str, tuple[tuple[float, int], list[_Class] | str]] = {}  # file → (its state, what it holds)

    def _files(self) -> list[Path]:
        files: dict[Path, None] = {}
        for folder in self._folders:
            top = (self.root / folder).resolve()
            for here, folders, names in os.walk(top):
                # hidden folders, caches, virtual environments and the folders the config names are not searched
                folders[:] = sorted(name for name in folders
                                    if name not in self._skip and not name.startswith((".", "_"))
                                    and not (Path(here) / name / "pyvenv.cfg").exists())
                for name in sorted(names):
                    if name.endswith(".py") and not name.startswith("."):
                        files[Path(here) / name] = None
        return list(files)

    def find(self) -> list[ExtractorEntry]:
        """Every extractor class, by file and in the order of the file; a file with a syntax error that mentions
        ``Extractor`` is listed with its error, so that a class doesn't vanish without a word while it is edited."""
        classes: list[_Class] = []
        broken: list[ExtractorEntry] = []
        seen = set()
        for path in self._files():
            try:
                file = path.relative_to(self.root).as_posix()
                state = (path.stat().st_mtime, path.stat().st_size)
            except (OSError, ValueError):
                continue
            seen.add(file)
            if file not in self._read or self._read[file][0] != state:
                try:
                    content: list[_Class] | str = _classes(path, file)
                except SyntaxError as error:
                    mentions = _BASE in path.read_text(encoding="utf-8", errors="replace")
                    content = f"line {error.lineno}: {error.msg}" if mentions else []
                except (OSError, ValueError):
                    content = []
                self._read[file] = (state, content)
            content = self._read[file][1]
            if isinstance(content, str):
                broken.append(ExtractorEntry(id=file, name="", file=file, about="", error=content))
            else:
                classes += content
        for file in set(self._read) - seen:  # a file that is gone
            del self._read[file]

        by_name: dict[str, list[_Class]] = {}
        for item in classes:
            by_name.setdefault(item.name, []).append(item)
        extractors = {_BASE}  # the names of classes that are extractors, found from base to subclass
        grew = True
        while grew:
            grew = False
            for item in classes:
                if item.name not in extractors and extractors.intersection(item.bases):
                    extractors.add(item.name)
                    grew = True

        def runs(item: _Class, visited: frozenset[str] = frozenset()) -> bool:
            """Does the class, or a class it builds on, have a ``run`` method?"""
            if item.has_run:
                return True
            return any(runs(base, visited | {item.name}) for name in item.bases if name not in visited
                       for base in by_name.get(name, ()))

        found = [ExtractorEntry(id=f"{_module(item.file) or item.file}:{item.name}", name=item.name, file=item.file,
                                about=item.about)
                 for item in classes if item.name in extractors and item.name != _BASE and runs(item)]
        return found + broken


def find_runs(config: Config) -> list[RunEntry]:
    """The run folders under ``paths.runs``, the newest first. A folder without run.json (a run that is still
    going on, or was cut off) is not listed."""
    runs = []
    if not config.paths.runs.is_dir():
        return runs
    for folder in config.paths.runs.iterdir():
        try:
            run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
            runs.append(RunEntry(
                id=folder.name, name=str(run["name"]), dataset=str(run["dataset"]), split=str(run["split"]),
                speed=float(run.get("speed") or 1.0),
                official=bool(run.get("official")) and same_run_settings(run, config),
                started_at=str(run.get("started_at") or ""),
                conversations=tuple((str(item["conversation"]), str(item.get("status") or "ok"))
                                    for item in run.get("conversations", []))))
        except (OSError, ValueError, KeyError, TypeError):
            continue  # not a run folder, or one that can't be read
    return sorted(runs, key=lambda run: (run.started_at, run.id), reverse=True)


class Library:
    """The conversations the page can show: the config's sample of each dataset, and any conversation a run was
    made on. Each is read from the processed data once."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._samples: dict[str, Dataset] | None = None
        self.problems: list[str] = []  # what keeps a dataset from being shown (it isn't prepared)
        self._others: dict[tuple[str, str, str], Dataset] = {}

    def samples(self) -> dict[str, Dataset]:
        """Dataset → the official sample of it, for the datasets that are prepared."""
        if self._samples is None:
            self._samples, self.problems = {}, []
            for name in ("spokenwoz", "ami"):
                try:
                    self._samples[name] = load_dataset(self.config, name, official_sample=True)
                except FileNotFoundError as error:
                    self.problems.append(str(error))
        return self._samples

    def conversation(self, dataset: str, split: str, conversation_id: str) -> tuple[Conversation, Form, Gold]:
        """One conversation with its form and gold. Raises KeyError or FileNotFoundError when there is none."""
        sample = self.samples().get(dataset)
        if sample is not None and sample.split == split and conversation_id in sample.conversations:
            data = sample
        else:
            key = (dataset, split, conversation_id)
            if key not in self._others:
                self._others[key] = load_dataset(self.config, dataset, split=split, ids=[conversation_id])
            data = self._others[key]
        return data.conversations[conversation_id], data.form_of(conversation_id), data.gold[conversation_id]

    def forms(self) -> list[dict[str, Any]]:
        """The config's sample as the page lists it: per dataset, its conversations with their length and the
        number of fields that have something to show."""
        out = []
        for name, data in self.samples().items():
            forms = []
            for conversation_id in sorted(data.conversations):
                conversation, form = data.conversations[conversation_id], data.form_of(conversation_id)
                forms.append({"id": conversation_id, "seconds": conversation.end, "fields": len(form.fields),
                              "said": len(gold_steps(form, data.gold[conversation_id]))})
            out.append({"name": name, "split": data.split, "forms": forms})
        return out


def entries(items: list[ExtractorEntry] | list[RunEntry]) -> list[dict[str, Any]]:
    """Entries as the page gets them."""
    return [asdict(item) for item in items]
