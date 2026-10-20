"""Form-type plugins: optional strict schemas layered on top of the universal record.

A plugin knows how to recognize its form, which typed schema it fills, how to render
a typed record back into the universal shape, and which typed tables it adds to the
index. Unknown forms simply have no plugin and stay universal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from form_agent.schema import UniversalForm


class FormPlugin(ABC):
    form_type: str
    title: str
    schema: type[BaseModel]

    @abstractmethod
    def detect(self, universal: UniversalForm) -> float:
        """Confidence 0..1 that this universal record is this form type."""

    @abstractmethod
    def to_universal(self, typed: BaseModel) -> UniversalForm:
        """Render a typed record in the universal shape (used for ground truth and indexing)."""

    @abstractmethod
    def table_rows(self, doc_id: str, typed: BaseModel) -> dict[str, list[dict[str, Any]]]:
        """Rows for this form type's typed tables in the index, keyed by table name."""

    def checkbox_groups(self) -> list[tuple[str, list[str], str]]:
        """(typed path, option names, printed group label) for each checkbox group.

        A bool field is a single-option group. Used to verify typed values against pixels.
        """
        return []

    def single_choice_groups(self) -> list[str]:
        """Typed paths of checkbox groups where at most one option may be ticked."""
        return []

    def validate(self, typed: BaseModel) -> list:
        """Form-specific validation rules; returns a list of schema.Flag."""
        return []

    def extraction_hints(self) -> str:
        """Form-specific guidance appended to the typed-extraction prompt (allowed options, formats)."""
        return ""

    def field_paths(self) -> dict[str, str]:
        """Universal key (as produced by `to_universal`) -> typed path, for fields and checkbox groups.

        Lets the UI edit typed values and attach flags to what it displays.
        """
        return {}

    def table_paths(self) -> dict[int, tuple[str, list[str]]]:
        """Universal table index -> (typed list path, typed attribute for each column)."""
        return {}

    def parse(self, data: dict[str, Any]) -> BaseModel:
        return self.schema.model_validate(data)


_PLUGINS: dict[str, FormPlugin] = {}


def register_form(plugin: FormPlugin) -> FormPlugin:
    _PLUGINS[plugin.form_type] = plugin
    return plugin


def get_plugin(form_type: str) -> FormPlugin | None:
    return _PLUGINS.get(form_type)


def all_plugins() -> list[FormPlugin]:
    return list(_PLUGINS.values())


def identify(universal: UniversalForm, threshold: float = 0.6) -> tuple[str, float]:
    """Best-matching known form type, or ("unknown", best score)."""
    best, score = "unknown", 0.0
    for plugin in _PLUGINS.values():
        s = plugin.detect(universal)
        if s > score:
            best, score = plugin.form_type, s
    return (best, score) if score >= threshold else ("unknown", score)
