"""Reader contract: image in, FormRecord out. Pipeline code only talks to this interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from form_agent.schema import FormRecord


class Reader(ABC):
    name: str

    @abstractmethod
    def read(self, image_path: Path) -> FormRecord:
        """Read one form image and return a structured record."""


_REGISTRY: dict[str, Callable[..., Reader]] = {}


def register(name: str):
    """Class decorator: `@register("claude")` makes the reader selectable by name."""

    def wrap(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return wrap


def get_reader(name: str, **kwargs) -> Reader:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown reader {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_readers() -> list[str]:
    return sorted(_REGISTRY)
