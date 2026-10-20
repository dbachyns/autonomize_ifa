"""Local OCR (RapidOCR: PaddleOCR models on ONNX Runtime). No API, no GPU.

Used as an independent witness next to the vision LLM: word positions for checkbox
labeling and text for cross-checking extracted values.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class OCRLine:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    score: float

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


SMALL_PAGE = 1600  # pixels wide; below this (low-res faxes, ~100 dpi) text is too small for the OCR models


def ocr_page(image: Image.Image, offset: tuple[float, float] = (0, 0)) -> list[OCRLine]:
    """OCR lines in page coordinates. Small pages are also read at 2x and both readings are kept:
    upscaling recovers small or faint text, while the original size keeps a few fragments upscaling loses."""
    lines = _ocr(image, offset)
    if offset == (0, 0) and image.width < SMALL_PAGE:
        big = image.convert("RGB").resize((image.width * 2, image.height * 2), Image.LANCZOS)
        lines += [OCRLine(l.text, l.x0 / 2, l.y0 / 2, l.x1 / 2, l.y1 / 2, l.score) for l in _ocr(big, (0, 0))]
    return lines


def _ocr(image: Image.Image, offset: tuple[float, float]) -> list[OCRLine]:
    arr = np.asarray(image.convert("RGB"))[:, :, ::-1]  # RGB -> BGR for OpenCV-style engines
    result, _ = _engine()(arr)
    dx, dy = offset
    lines = []
    for box, text, score in result or []:
        pts = np.asarray(box, dtype=float)
        lines.append(OCRLine(text, pts[:, 0].min() + dx, pts[:, 1].min() + dy, pts[:, 0].max() + dx,
                             pts[:, 1].max() + dy, float(score)))
    return lines


def ocr_region(image: Image.Image, x0: float, y0: float, x1: float, y1: float) -> list[OCRLine]:
    """OCR a crop; small regions often recover text the full-page detector skipped."""
    box = (max(0, int(x0)), max(0, int(y0)), min(image.width, int(x1)), min(image.height, int(y1)))
    return ocr_page(image.crop(box), offset=(box[0], box[1]))
