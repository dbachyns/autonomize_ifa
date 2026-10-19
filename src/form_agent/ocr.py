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


def ocr_page(image: Image.Image, offset: tuple[float, float] = (0, 0)) -> list[OCRLine]:
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
