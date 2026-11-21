"""Forms: the field types, field specs and form file format, with Speechineer's names.

Written for this environment (no Speechineer code), so that a form file works in both:

- ``FieldType`` has Speechineer's 15 types, with the same names and value types.
- ``FieldSpec`` is Speechineer's spec (``field_id``, ``prompt``, ``type``, ``field_configuration``) plus the
  ``label`` that Speechineer's form files carry.
- ``Form`` reads and writes Speechineer's form import file: ``{"name", "system_prompts", "fields"}``, where a
  field carries ``options`` (select, multiselect) or ``range`` (slider). Its ``system_prompts`` hold the task
  rules (``form_data_extraction``) and, for a later speech-to-text stage, vocabulary hints (``transcription``).
"""

from __future__ import annotations

import datetime as dt
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FieldType(StrEnum):
    """Speechineer's field types. The value an extractor publishes must have :attr:`value_type`."""

    text = "text"  # single-line text
    textarea = "textarea"  # multi-line text; for lists, one item per line
    email = "email"
    phone = "phone"
    url = "url"
    integer = "integer"
    float = "float"
    slider = "slider"  # a number inside the field's range
    checkbox = "checkbox"  # true / false
    select = "select"  # exactly one of the field's options
    multiselect = "multiselect"  # a list of the field's options
    date = "date"  # "YYYY-MM-DD"
    time = "time"  # "HH:MM", 24-hour
    datetime = "datetime"  # "YYYY-MM-DDTHH:MM"
    template = "template"  # a dict of sub-fields

    @property
    def value_type(self) -> type:
        return _VALUE_TYPES[self]


_VALUE_TYPES: dict[FieldType, type] = {
    FieldType.text: str, FieldType.textarea: str, FieldType.email: str, FieldType.phone: str, FieldType.url: str,
    FieldType.integer: int, FieldType.float: float, FieldType.slider: float, FieldType.checkbox: bool,
    FieldType.select: str, FieldType.multiselect: list, FieldType.date: str, FieldType.time: str,
    FieldType.datetime: str, FieldType.template: dict,
}


class OptionsConfig(BaseModel):
    """Options of a select or multiselect field."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["options"] = "options"
    options: tuple[str, ...]
    append_options_to_prompt: bool = False


class RangeConfig(BaseModel):
    """Range of a slider field: (min, max)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["range"] = "range"
    range: tuple[float, float]


FieldTypeConfig = Annotated[OptionsConfig | RangeConfig, Field(discriminator="kind")]


class FieldSpec(BaseModel):
    """One field to fill: what it means (``prompt``), its type and, for some types, its options or range."""

    model_config = ConfigDict(frozen=True)

    field_id: str
    label: str
    prompt: str
    type: FieldType
    field_configuration: FieldTypeConfig | None = None

    @model_validator(mode="after")
    def _check(self) -> FieldSpec:
        if self.type in (FieldType.select, FieldType.multiselect) and not (
                isinstance(self.field_configuration, OptionsConfig) and self.field_configuration.options):
            raise ValueError(f"{self.field_id}: a {self.type} field needs options")
        if self.type == FieldType.slider:
            if not isinstance(self.field_configuration, RangeConfig):
                raise ValueError(f"{self.field_id}: a slider field needs a range")
            low, high = self.field_configuration.range
            if low >= high:
                raise ValueError(f"{self.field_id}: slider range min {low} must be below max {high}")
        return self

    @property
    def options(self) -> tuple[str, ...]:
        """The options of a select or multiselect field; empty for the other types."""
        return self.field_configuration.options if isinstance(self.field_configuration, OptionsConfig) else ()

    @property
    def range(self) -> tuple[float, float] | None:
        """The (min, max) of a slider field; None for the other types."""
        return self.field_configuration.range if isinstance(self.field_configuration, RangeConfig) else None


class FieldValue(BaseModel):
    """A field and its value, as in Speechineer's results."""

    model_config = ConfigDict(frozen=True)

    field_id: str
    value: Any = None


class Form(BaseModel):
    """The form of one conversation: its fields and the task rules."""

    model_config = ConfigDict(frozen=True)

    form_id: str
    name: str
    system_prompts: dict[str, str] = Field(default_factory=dict)
    fields: tuple[FieldSpec, ...]

    @model_validator(mode="after")
    def _unique_ids(self) -> Form:
        ids = [f.field_id for f in self.fields]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"form {self.form_id}: duplicate field ids {duplicates}")
        return self

    @property
    def description(self) -> str:
        """The task rules: what to fill, and when (Speechineer's ``form_data_extraction`` system prompt)."""
        return self.system_prompts.get("form_data_extraction", "")

    def field(self, field_id: str) -> FieldSpec:
        for spec in self.fields:
            if spec.field_id == field_id:
                return spec
        raise KeyError(f"form {self.form_id} has no field {field_id!r}")

    @classmethod
    def from_import_file(cls, data: dict[str, Any] | str | Path, form_id: str | None = None) -> Form:
        """A form from Speechineer's form import file (a path or the parsed JSON)."""
        if not isinstance(data, dict):
            path = Path(data)
            data = json.loads(path.read_text(encoding="utf-8"))
            form_id = form_id or path.stem
        fields = []
        for raw in data["fields"]:
            config: OptionsConfig | RangeConfig | None = None
            if "options" in raw:
                config = OptionsConfig(options=tuple(raw["options"]),
                                       append_options_to_prompt=bool(raw.get("append_options_to_prompt", False)))
            elif "range" in raw:
                config = RangeConfig(range=tuple(raw["range"]))
            fields.append(FieldSpec(field_id=raw["field_id"], label=raw.get("label") or raw["field_id"],
                                    prompt=raw.get("prompt", ""), type=FieldType(raw["type"]),
                                    field_configuration=config))
        name = str(data.get("name") or form_id or "form")
        return cls(form_id=form_id or name, name=name, system_prompts=dict(data.get("system_prompts") or {}),
                   fields=tuple(fields))

    def to_import_file(self) -> dict[str, Any]:
        """This form in Speechineer's form import format."""
        fields = []
        for spec in self.fields:
            raw: dict[str, Any] = {"field_id": spec.field_id, "label": spec.label, "type": spec.type.value,
                                   "prompt": spec.prompt}
            if isinstance(spec.field_configuration, OptionsConfig):
                raw["options"] = list(spec.field_configuration.options)
                if spec.field_configuration.append_options_to_prompt:
                    raw["append_options_to_prompt"] = True
            elif isinstance(spec.field_configuration, RangeConfig):
                raw["range"] = list(spec.field_configuration.range)
            fields.append(raw)
        return {"name": self.name, "system_prompts": dict(self.system_prompts), "fields": fields}


_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}T([01]\d|2[0-3]):[0-5]\d$")


def value_error(spec: FieldSpec, value: Any) -> str | None:
    """Why ``value`` can't be published in ``spec``'s field; None when it can. ``None`` (clear) always can."""
    if value is None:
        return None
    t = spec.type
    if t in (FieldType.text, FieldType.textarea, FieldType.email, FieldType.phone, FieldType.url):
        return None if isinstance(value, str) else f"{t} needs a str, got {type(value).__name__}"
    if t == FieldType.integer:
        return None if isinstance(value, int) and not isinstance(value, bool) else f"integer needs an int, got {value!r}"
    if t in (FieldType.float, FieldType.slider):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{t} needs a number, got {value!r}"
        if t == FieldType.slider and spec.range and not spec.range[0] <= value <= spec.range[1]:
            return f"slider value {value} is outside {spec.range}"
        return None
    if t == FieldType.checkbox:
        return None if isinstance(value, bool) else f"checkbox needs true/false, got {value!r}"
    if t == FieldType.select:
        return None if value in spec.options else f"select needs one of the options, got {value!r}"
    if t == FieldType.multiselect:
        if not isinstance(value, list):
            return f"multiselect needs a list of options, got {value!r}"
        wrong = [v for v in value if v not in spec.options]
        return f"multiselect values not among the options: {wrong}" if wrong else None
    if t == FieldType.date:
        return None if isinstance(value, str) and _DATE.match(value) and _parses(value, "%Y-%m-%d") \
            else f"date needs YYYY-MM-DD, got {value!r}"
    if t == FieldType.time:
        return None if isinstance(value, str) and _TIME.match(value) else f"time needs HH:MM (24 h), got {value!r}"
    if t == FieldType.datetime:
        return None if isinstance(value, str) and _DATETIME.match(value) and _parses(value, "%Y-%m-%dT%H:%M") \
            else f"datetime needs YYYY-MM-DDTHH:MM, got {value!r}"
    if t == FieldType.template:
        return None if isinstance(value, dict) else f"template needs a dict, got {type(value).__name__}"
    return f"unknown field type {t}"


def _parses(value: str, pattern: str) -> bool:
    try:
        dt.datetime.strptime(value, pattern)
    except ValueError:
        return False
    return True
