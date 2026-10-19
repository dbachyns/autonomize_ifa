"""Load any supported input (PNG/JPG/TIFF/PDF, single or multi-page) as page images."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageSequence

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED = IMAGE_SUFFIXES | {".pdf"}


def doc_id_for(path: Path) -> str:
    return path.stem


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pages(path: Path, dpi: int = 200) -> list[Image.Image]:
    """Return one RGB image per page. PDFs are rasterized at `dpi`."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pymupdf

        with pymupdf.open(path) as doc:
            pages = []
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
            return pages
    if suffix in IMAGE_SUFFIXES:
        with Image.open(path) as im:
            return [frame.convert("RGB") for frame in ImageSequence.Iterator(im)]
    raise ValueError(f"Unsupported file type {suffix!r}; supported: {sorted(SUPPORTED)}")


def find_documents(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
