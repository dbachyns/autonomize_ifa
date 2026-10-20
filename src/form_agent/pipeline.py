"""Upload -> ready: read a new form, check it, save it, and make it searchable in one call."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from form_agent.config import settings
from form_agent.ingest import load_pages
from form_agent.layout import compute_layout
from form_agent.readers import get_reader
from form_agent.schema import FormRecord
from form_agent.store import build_index, save_record
from form_agent.validate import validate

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}


def process_file(path: Path, reader: str | None = None, cache_dir: Path | None = None, db_path: Path | None = None,
                 uploads_dir: Path | None = None, on_step: Callable[[str], None] | None = None) -> FormRecord:
    """Copy the file into the uploads folder, extract it, verify and validate it, cache it, and re-index.

    Returns the saved record. `on_step` receives a short progress message before each step.
    """
    step = on_step or (lambda _msg: None)
    cache_dir = Path(cache_dir or settings.cache_dir)
    db_path = Path(db_path or settings.db_path)
    uploads_dir = Path(uploads_dir or settings.data_dir / "uploads")
    reader = reader or "hybrid"
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported file type {path.suffix!r}; use one of {', '.join(sorted(SUPPORTED))}")

    target = _unique_target(path, uploads_dir, cache_dir)
    if path.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    step(f"Reading the form with the {reader} reader (vision model; this takes a minute or two)…")
    kwargs = {"cache_dir": cache_dir} if reader == "hybrid" else {}
    record = get_reader(reader, **kwargs).read(target)
    step("Running validation rules…")
    record.flags = [*record.flags, *validate(record)]
    save_record(cache_dir, record)
    step("Locating values on the scan (local OCR)…")
    compute_layout(cache_dir, record.doc_id, load_pages(target))
    step("Updating the search index…")
    build_index(cache_dir, db_path)
    return record


def _unique_target(path: Path, uploads_dir: Path, cache_dir: Path) -> Path:
    """A file name whose stem (the doc id) is not taken yet."""
    stem, suffix = _safe(path.stem), path.suffix.lower()
    candidate, n = stem, 1
    while (cache_dir / candidate).exists() or (uploads_dir / f"{candidate}{suffix}").exists():
        n += 1
        candidate = f"{stem}-{n}"
    return uploads_dir / f"{candidate}{suffix}"


def _safe(stem: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem).strip("._")
    return keep or "form"
