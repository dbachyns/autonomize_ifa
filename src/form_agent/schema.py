"""Core data contract: every reader produces a FormRecord, every later stage consumes one.

The universal part (`UniversalForm`) can describe any form: labeled fields, checkbox
groups, tables, and free text, each tied to a page and section. Form types we know
(see `form_agent.forms`) add a strict typed view on top, stored in `FormRecord.typed`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from form_agent.normalize import VALUE_TYPES

ValueType = Literal[tuple(VALUE_TYPES)]  # type: ignore[valid-type]
BBox = tuple[float, float, float, float]  # x0, y0, x1, y1 in source-page pixels


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def kind(k: str) -> dict[str, Any]:
    """Tag a typed-schema field with its value type; drives normalization and per-type eval."""
    return {"json_schema_extra": {"kind": k}}


# --- Universal record -----------------------------------------------------------------


class FormField(_Model):
    label: str  # as printed on the form, e.g. "Member or Medicaid ID #:"
    key: str | None = None  # normalized, e.g. "member_id"; filled by normalize.label_to_key if absent
    value: str | None = None  # verbatim; None when the field is blank
    value_type: ValueType = "text"
    source: Literal["filled", "printed"] = "filled"  # entered by the person vs pre-printed on the form
    section: str | None = None
    page: int = 1
    confidence: float | None = Field(None, ge=0, le=1)
    bbox: BBox | None = None


class CheckboxOption(_Model):
    label: str
    checked: bool
    confidence: float | None = Field(None, ge=0, le=1)
    method: str | None = None  # "llm", "pixel", "llm+pixel"
    bbox: BBox | None = None


class CheckboxGroup(_Model):
    label: str  # e.g. "Review Type"
    key: str | None = None
    section: str | None = None
    page: int = 1
    options: list[CheckboxOption] = Field(default_factory=list)

    @property
    def selected(self) -> list[str]:
        return [o.label for o in self.options if o.checked]


class Table(_Model):
    title: str | None = None
    section: str | None = None
    page: int = 1
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str | None]] = Field(default_factory=list)
    column_types: list[ValueType] = Field(default_factory=list)


class TextBlock(_Model):
    text: str
    label: str | None = None
    section: str | None = None
    page: int = 1
    bbox: BBox | None = None


class UniversalForm(_Model):
    title: str | None = None
    page_count: int = 1
    fields: list[FormField] = Field(default_factory=list)
    checkboxes: list[CheckboxGroup] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    text_blocks: list[TextBlock] = Field(default_factory=list)


# --- Envelope -------------------------------------------------------------------------


class Flag(_Model):
    rule: str
    severity: Literal["info", "warning", "error"]
    message: str
    fields: list[str] = Field(default_factory=list)


class RunInfo(_Model):
    reader: str  # "claude", "ollama", "hybrid", "ground_truth", ...
    model: str | None = None
    prompt_version: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    seconds: float | None = None
    cost_usd: float | None = None


class FormRecord(_Model):
    doc_id: str
    source_file: str  # relative to the data directory it was ingested from
    run: RunInfo
    form_type: str = "unknown"  # "tx_prior_auth", ... or "unknown"
    form_type_confidence: float | None = Field(None, ge=0, le=1)
    universal: UniversalForm
    typed: dict[str, Any] | None = None  # validated by the matching plugin in form_agent.forms
    flags: list[Flag] = Field(default_factory=list)


# --- Helpers ---------------------------------------------------------------------------


def flatten(model: BaseModel, prefix: str = "") -> dict[str, tuple[Any, str]]:
    """Flatten a typed model to {path: (value, kind)}; list items get an index (services.lines.0.code)."""
    out: dict[str, tuple[Any, str]] = {}
    for name, info in type(model).model_fields.items():
        value = getattr(model, name)
        path = f"{prefix}{name}"
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
        if isinstance(value, BaseModel):
            out.update(flatten(value, f"{path}."))
        elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
            for i, item in enumerate(value):
                out.update(flatten(item, f"{path}.{i}."))
        elif name != "form_type":
            out[path] = (value, extra.get("kind", "text"))
    return out
