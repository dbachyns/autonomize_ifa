"""Pixel-level checkbox verification, independent of the LLM.

1. Find checkbox squares: thin, hollow, square outlines of plausible size.
2. Measure ink inside each square (fill ratio). Empty boxes are ~0, stray specks a few
   percent, real ticks well above `TICK_THRESHOLD`.
3. Label each square with the OCR text to its right.
4. For a checkbox group (a set of option names), pick the cluster of labeled squares that
   matches all options and sits closest together.

The LLM's reading is then confirmed or overridden when the pixels are decisive.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from form_agent.ocr import OCRLine, ocr_region

TICK_THRESHOLD = 0.08  # fill ratio separating ticks from empty boxes / stray marks
CLEAR_EMPTY = 0.06  # below this the pixels clearly say "not ticked"
CLEAR_TICK = 0.10  # above this the pixels clearly say "ticked"


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    fill: float
    label: str = ""  # normalized OCR text to the right of the box

    @property
    def checked(self) -> bool:
        return self.fill >= TICK_THRESHOLD

    @property
    def decisive(self) -> bool:
        return self.fill <= CLEAR_EMPTY or self.fill >= CLEAR_TICK

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def detect_boxes(image: Image.Image) -> list[Box]:
    gray = np.asarray(image.convert("L"))
    width = gray.shape[1]
    lo, hi = int(width * 0.010), int(width * 0.032)  # ~0.25-0.8 inch-ish on letter paper
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[Box] = []
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] != -1:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if not (lo <= w <= hi and lo <= h <= hi and 0.75 <= w / h <= 1.33):
            continue
        if cv2.contourArea(contour) < 0.85 * w * h:  # must be square-cornered
            continue
        hole, child = 0.0, hierarchy[0][i][2]
        while child != -1:
            hole = max(hole, cv2.contourArea(contours[child]))
            child = hierarchy[0][child][0]
        if hole < 0.45 * w * h:  # thin outline with a big interior, unlike letters such as D or O
            continue
        m = max(4, int(w * 0.18))
        fill = float((bw[y + m : y + h - m, x + m : x + w - m] > 0).mean())
        if any(abs(b.x - x) < w / 2 and abs(b.y - y) < h / 2 for b in boxes):
            continue
        boxes.append(Box(x, y, w, h, round(fill, 4)))
    return boxes


def label_boxes(boxes: list[Box], lines: list[OCRLine], image: Image.Image | None = None) -> None:
    """Attach to each box the OCR text that starts at (or runs through) its right edge.

    Boxes left unlabeled get a targeted OCR pass on the strip to their right, if `image` is given.
    """
    _label_from_lines(boxes, lines)
    if image is None:
        return
    for box in boxes:
        if not box.label:
            strip = ocr_region(image, box.x + box.w * 0.8, box.y - box.h * 0.6, box.x + box.w * 12, box.y + box.h * 1.6)
            _label_from_lines([box], strip)


def _label_from_lines(boxes: list[Box], lines: list[OCRLine]) -> None:
    for box in boxes:
        best, best_dx = "", None
        cy = box.y + box.h / 2
        right = box.x + box.w
        for line in lines:
            if abs((line.y0 + line.y1) / 2 - cy) > box.h or line.x1 <= right:
                continue
            if line.x0 >= box.x - 0.2 * box.w:
                # Line starts at the box (OCR often includes the box glyph) or after it.
                dx, text = max(0.0, line.x0 - right), line.text
            else:
                # Box sits inside a merged OCR line: estimate where its label starts, with slack.
                frac = (right - line.x0) / (line.x1 - line.x0)
                dx, text = 0.0, line.text[max(0, int(frac * len(line.text)) - 3) :]
            if dx <= box.w * 6 and (best_dx is None or dx < best_dx):
                best, best_dx = text, dx
        box.label = _norm(best)


def _label_score(option: str, label: str) -> float:
    opt = _norm(option)
    if not opt or not label:
        return 0.0
    if label.startswith(opt):
        return 1.0
    pos = label.find(opt)
    if 0 < pos <= 4:  # stray leading characters from a merged OCR line
        return 0.95
    # OCR noise: best alignment of the option against the label's first few characters.
    best = 0.0
    for start in range(0, min(4, len(label)) + 1):
        window = label[start : start + len(opt)]
        same = sum(a == b for a, b in zip(opt, window)) / len(opt)
        if len(opt) <= 3 and sorted(window) == sorted(opt):
            same = max(same, 0.8)  # short words OCR'd scrambled, e.g. "No)" -> "(oN"
        best = max(best, same - 0.05 * start)
    return best if best >= 0.75 else 0.0


def _hint_score(hint: str | None, context: str) -> float:
    """Fraction of the hint's words found in the text to the left of the group (disambiguates Yes/No groups)."""
    words = [_norm(w) for w in re.split(r"\W+", hint or "") if len(w) >= 3]
    words = [w for w in words if w not in ("yes", "the", "and", "attached")]
    return sum(w in context for w in words) / len(words) if words else 0.0


def locate_group(options: list[str], boxes: list[Box], used: set[int] | None = None,
                 hint: str | None = None, contexts: list[str] | None = None) -> dict[str, Box] | None:
    """Find one box per option, choosing the tightest spatial cluster. None if any option is unmatched.

    `hint` (the group's printed label) is matched against `contexts[i]`, the text left of box i.
    """
    used = used or set()
    candidates = []
    for option in options:
        scored = [(i, _label_score(option, b.label)) for i, b in enumerate(boxes) if i not in used]
        scored = [(i, s) for i, s in scored if s > 0]
        if not scored:
            return None
        candidates.append(sorted(scored, key=lambda t: -t[1])[:5])

    best, best_cost = None, None
    for combo in itertools.product(*candidates):
        idx = [i for i, _ in combo]
        if len(set(idx)) != len(idx):
            continue
        xs = [boxes[i].center[0] for i in idx]
        ys = [boxes[i].center[1] for i in idx]
        spread = (max(xs) - min(xs)) + 3 * (max(ys) - min(ys))  # groups are mostly horizontal
        cost = spread - 500 * sum(s for _, s in combo)
        if hint and contexts:
            first = min(idx, key=lambda i: (boxes[i].y // boxes[i].h, boxes[i].x))
            cost -= 2000 * _hint_score(hint, contexts[first])
        if best_cost is None or cost < best_cost:
            best, best_cost = idx, cost
    return {opt: boxes[i] for opt, i in zip(options, best)} if best else None


@dataclass
class GroupCheck:
    options: dict[str, Box]  # option -> located box

    def states(self) -> dict[str, bool]:
        return {o: b.checked for o, b in self.options.items()}


class PageCheckboxes:
    """Boxes found on one page, with bookkeeping so each box serves only one group."""

    def __init__(self, image: Image.Image, lines: list[OCRLine]):
        self.boxes = detect_boxes(image)
        label_boxes(self.boxes, lines, image)
        self.contexts = [self._left_context(b, lines) for b in self.boxes]
        self.used: set[int] = set()

    @staticmethod
    def _left_context(box: Box, lines: list[OCRLine]) -> str:
        """Normalized text on the box's row to its left (where a group's question usually is)."""
        cy = box.y + box.h / 2
        row = [l for l in lines if abs((l.y0 + l.y1) / 2 - cy) <= box.h and l.x0 < box.x and box.x - l.x0 < box.w * 25]
        return _norm(" ".join(l.text for l in sorted(row, key=lambda l: l.x0)))

    def locate(self, options: list[str], hint: str | None = None) -> GroupCheck | None:
        found = locate_group(options, self.boxes, self.used, hint, self.contexts)
        if not found:
            return None
        self.used.update(self.boxes.index(b) for b in found.values())
        return GroupCheck(found)
