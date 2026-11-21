"""The processed datasets on disk: what `lfe prepare` writes and what runs and evaluations read.

<processed>/<dataset>/<split>/
  conversations.jsonl   one Conversation per line (timed words, roles; meta.form_id names its form)
  gold.jsonl            one Gold per line
  forms.json            {form_id: form in Speechineer's import format}
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from live_form_eval.config import Config
from live_form_eval.data.common.model import Conversation, Gold
from live_form_eval.data.common.sampling import sample_ids
from live_form_eval.interface.schema import Form


@dataclass(frozen=True)
class Dataset:
    name: str
    split: str
    conversations: dict[str, Conversation]
    gold: dict[str, Gold]
    forms: dict[str, Form]  # form id → form

    def form_of(self, conversation_id: str) -> Form:
        return self.forms[self.conversations[conversation_id].meta["form_id"]]


def split_of(config: Config, dataset: str) -> str:
    """The split a dataset is evaluated on (AMI has a single one)."""
    return config.datasets.spokenwoz.split if dataset == "spokenwoz" else "all"


def dataset_dir(config: Config, dataset: str, split: str | None = None) -> Path:
    return config.paths.processed / dataset / (split or split_of(config, dataset))


def write_dataset(folder: Path, conversations: Iterable[Conversation], golds: Iterable[Gold],
                  forms: dict[str, Form]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "conversations.jsonl").open("w", encoding="utf-8") as out:
        for conversation in conversations:
            out.write(conversation.model_dump_json() + "\n")
    with (folder / "gold.jsonl").open("w", encoding="utf-8") as out:
        for gold in golds:
            out.write(gold.model_dump_json() + "\n")
    data = {form_id: form.to_import_file() for form_id, form in forms.items()}
    (folder / "forms.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def how_to_get_the_data() -> str:
    """What to do when the processed data is missing: build it where the code that builds it is, else unzip the
    data that comes with a copy that has no such code (the hackathon's)."""
    from importlib.util import find_spec

    if find_spec("live_form_eval.data.spokenwoz.build") is not None:
        return "run `lfe prepare` first"
    return 'unzip the data you were given there (README, "Install, and get the data")'


def load_dataset(config: Config, dataset: str, *, split: str | None = None, ids: Iterable[str] | None = None,
                 n: int | None = None, seed: int | None = None, official_sample: bool = False) -> Dataset:
    """A processed dataset.

    ``ids`` picks conversations by id; otherwise ``n`` and ``seed`` draw a sample. ``official_sample`` uses the
    config's ``n`` and ``seed`` for that dataset: the sample every official run is scored on.
    """
    split = split or split_of(config, dataset)
    folder = dataset_dir(config, dataset, split)
    if not (folder / "conversations.jsonl").exists():
        raise FileNotFoundError(f"{folder} is missing: {how_to_get_the_data()}")
    wanted = set(ids) if ids is not None else None
    conversations: dict[str, Conversation] = {}
    with (folder / "conversations.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            # the id is read first so that only the wanted conversations are fully parsed
            if wanted is None or json.loads(line)["id"] in wanted:
                conversation = Conversation.model_validate_json(line)
                conversations[conversation.id] = conversation
    if wanted is not None and wanted - set(conversations):
        raise KeyError(f"{dataset}/{split} has no conversation {sorted(wanted - set(conversations))}")
    if wanted is None:
        sample = getattr(config.datasets, dataset)
        if official_sample:
            n, seed = sample.n, sample.seed
        keep = set(sample_ids(conversations, n, sample.seed if seed is None else seed))
        conversations = {cid: c for cid, c in conversations.items() if cid in keep}
    gold: dict[str, Gold] = {}
    with (folder / "gold.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            if json.loads(line)["conversation_id"] in conversations:
                item = Gold.model_validate_json(line)
                gold[item.conversation_id] = item
    raw_forms = json.loads((folder / "forms.json").read_text(encoding="utf-8"))
    used = {c.meta["form_id"] for c in conversations.values()}
    forms = {form_id: Form.from_import_file(raw, form_id) for form_id, raw in raw_forms.items() if form_id in used}
    return Dataset(name=dataset, split=split, conversations=conversations, gold=gold, forms=forms)
