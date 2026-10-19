"""Hybrid reader: a vision-LLM reader, verified by local OCR and pixel checkbox detection.

The base reader's raw output is cached under its own name, so re-running the hybrid
step (e.g. after tuning the verifier) costs no API calls.
"""

from __future__ import annotations

import time
from pathlib import Path

from form_agent.config import settings
from form_agent.ingest import doc_id_for, load_pages
from form_agent.readers.base import Reader, get_reader, register
from form_agent.schema import FormRecord
from form_agent.store import load_record, record_path, save_record
from form_agent.verify import verify


@register("hybrid")
class HybridReader(Reader):
    def __init__(self, base: str = "openai", cache_dir: Path | None = None, **base_kwargs):
        self.base_name = base
        self.base_kwargs = base_kwargs
        self.cache_dir = cache_dir or settings.cache_dir
        self._base: Reader | None = None

    def read(self, image_path: Path) -> FormRecord:
        cached = record_path(self.cache_dir, doc_id_for(image_path), self.base_name)
        if cached.exists():
            base_record = load_record(cached)
        else:
            self._base = self._base or get_reader(self.base_name, **self.base_kwargs)
            base_record = self._base.read(image_path)
            save_record(self.cache_dir, base_record)

        start = time.monotonic()
        record, flags = verify(base_record, load_pages(image_path))
        record.run = record.run.model_copy(update={
            "reader": self.name,
            "model": f"{base_record.run.model}+ocr+pixel",
            "seconds": round((base_record.run.seconds or 0) + time.monotonic() - start, 2),
        })
        record.flags = flags
        return record
