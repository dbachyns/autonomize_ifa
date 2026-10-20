"""Pixel-level checkbox verification, independent of the LLM.

1. Find checkbox squares: thin, hollow, square (or somewhat wide) outlines of plausible size. Short
   blank lines that options are marked on ("_X_ Change", "Personal Service _X_") count as boxes too:
   the space just above the line is the box.
2. Measure ink inside each square (fill ratio). Empty boxes are ~0, stray specks a few
   percent, real ticks well above `TICK_THRESHOLD`.
3. Label each square with the OCR text to its right and to its left.
4. For a checkbox group (a set of option names), pick the cluster of labeled squares that
   matches all options and sits closest together, with all labels on the same side.

The LLM's reading is then confirmed or overridden when the pixels are decisive.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from rapidfuzz import fuzz

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
    left: str = ""  # normalized OCR text ending at the box's left edge ("JUNE 30 [X]")
    kind: str = "box"  # "box", or "line" for a blank line marked with an X or tick

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
        if not _box_size(w, h, lo, hi, width):
            if w <= 0.2 * width and h <= 0.2 * gray.shape[0]:
                boxes += _grid_cells(contours, hierarchy, i, bw, lo, hi, width, boxes)
            continue
        wide = w / h > 1.33  # some forms tick in wide boxes: "JUNE 30 [  X  ]"
        solid = cv2.contourArea(contour) >= (0.75 if wide else 0.85) * w * h  # square-cornered
        hole, inside, child = 0.0, None, hierarchy[0][i][2]
        while child != -1:
            if (area := cv2.contourArea(contours[child])) > hole:
                hole, inside = area, cv2.boundingRect(contours[child])
            child = hierarchy[0][child][0]
        m = max(3, int(h * 0.18)) if wide else max(4, int(w * 0.18))
        inner = bw[y + m : y + h - m, x + m : x + w - m] > 0
        if not solid or hole < 0.45 * w * h:
            # Not an obvious hollow square: a letter such as D or O, or a small box whose thick border, tick or a
            # touching letter hides its interior. A box has four straight, fully inked edges; letters don't.
            if min(w, h) < lo + 3 or not _framed(bw, x, y, w, h):
                continue  # (dots and bold specks are small solid blobs)
            t = max(3, round(min(w, h) * 0.25))
            inner = bw[y + t : y + h - t, x + t : x + w - t] > 0
            if inner.size == 0:
                continue
            fill = float(inner.mean())
        elif wide:  # a tick fills only part of a wide box: measure its densest box-sized stretch
            ix, iy, iw, ih = inside  # inside the border, however thick it is on a low-res scan
            inner = bw[iy + 2 : iy + ih - 2, ix + 2 : ix + iw - 2] > 0
            if inner.size == 0:
                continue
            win = min(inner.shape[1], int(inner.shape[0] * 1.5))
            fill = max(float(inner[:, j : j + win].mean()) for j in range(inner.shape[1] - win + 1))
        else:
            fill = float(inner.mean())
        if any(abs(b.x - x) < w / 2 and abs(b.y - y) < h / 2 for b in boxes):
            continue
        boxes.append(Box(x, y, w, h, round(fill, 4)))
    return boxes


def _box_size(w: int, h: int, lo: int, hi: int, width: int, max_aspect: float = 6) -> bool:
    wide = w / h > 1.33
    return lo <= h <= hi and (lo <= w <= hi and 0.75 <= w / h or wide and w / h <= max_aspect and w <= 0.15 * width)


def _grid_cells(contours, hierarchy, parent: int, bw: np.ndarray, lo: int, hi: int, width: int,
                boxes: list[Box]) -> list[Box]:
    """Cells of a small grid of boxes sharing their borders (three dates stacked in one frame)."""
    if not _framed(bw, *cv2.boundingRect(contours[parent])):
        return []  # letters with two holes (B, 8) are not a rectangular frame
    cells, child = [], hierarchy[0][parent][2]
    while child != -1:
        x, y, w, h = cv2.boundingRect(contours[child])
        child = hierarchy[0][child][0]
        if not _box_size(w + 2, h + 2, lo, hi, width, max_aspect=8):
            continue
        inner = bw[y + 2 : y + h - 2, x + 2 : x + w - 2] > 0
        if inner.size == 0 or any(abs(b.x - x) < w / 2 and abs(b.y - y) < h / 2 for b in boxes):
            continue
        win = min(inner.shape[1], int(inner.shape[0] * 1.5))
        fill = max(float(inner[:, j : j + win].mean()) for j in range(inner.shape[1] - win + 1))
        cells.append(Box(x - 1, y - 1, w + 2, h + 2, round(fill, 4)))
    return cells if len(cells) >= 2 else []


def detect_blanks(image: Image.Image, lines: list[OCRLine], boxes: list[Box]) -> list[Box]:
    """Short blank lines ("____ Summons", "Fed-X _x_") as boxes covering the space just above each line.

    Their fill is the ink of the densest mark-sized stretch above the line, so an X or tick on the blank reads
    as ticked. Lines under printed words (an OCR line sits there) and the edges of detected boxes are skipped.
    """
    gray = np.asarray(image.convert("L"))
    width = gray.shape[1]
    heights = sorted(l.y1 - l.y0 for l in lines)
    text_h = int(heights[len(heights) // 2] * 0.75) if heights else max(8, width // 80)  # OCR boxes have padding
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(6, int(width * 0.01)), 1))
    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    def under_text(x: int, y: int, w: int) -> bool:  # bottom stroke of printed letters, not an underscore
        above = bw[max(0, y - text_h) : y - 1, x : x + w] > 0
        return above.size == 0 or above.any(axis=0).mean() > 0.7

    pieces = [r for r in (_line_extent(horizontal, cv2.boundingRect(c)) for c in contours)
              if not under_text(r[0], r[1], r[2])]
    blanks: list[Box] = []
    for x, y, w, h in _join_segments(pieces, gap=text_h):
        if not (width * 0.025 <= w <= width * 0.2 and h <= max(4, text_h // 4)) or y < text_h:
            continue
        below = bw[y + h + 1 : y + h + 4, x : x + w] > 0
        if below.size == 0 or below.mean() > 0.15:
            continue  # bottom strokes of bold letters have more ink right under them
        top = max(0, y - text_h)
        region = (x, top, x + w, y)
        if any(b.x - 3 <= x <= b.x + b.w + 3 and b.y - 3 <= y <= b.y + b.h + 3 for b in boxes):
            continue  # top or bottom edge of a checkbox
        if all((bw[y - text_h : y, c] > 0).mean() > 0.6 for c in (x, x + w - 1)):
            continue  # both ends rise into vertical strokes: the bottom of a table cell, not a blank
        if any(_inside(l, region) > 0.6 and len(_norm(l.text)) >= 3 for l in lines):
            continue  # underlined printed text or a written-in value, not a mark
        if under_text(x, y, w):
            continue  # ink across the whole width: text sitting on a line, not a mark on a blank
        above = bw[top : y - 1, x : x + w] > 0
        win = min(above.shape[1], above.shape[0])  # a handwritten x can be small
        fill = max(float(above[:, j : j + win].mean()) for j in range(above.shape[1] - win + 1))
        blanks.append(Box(x, top, w, y - top, round(fill, 4), kind="line"))
    return blanks


def _line_extent(horizontal: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """The rect narrowed to its longest row of ink: the underline itself, without the feet of letters
    resting on it ("FULL____")."""
    x, y, w, h = rect
    best = (0, 0, 0)  # run length, start, row
    for r in range(y, y + h):
        row = np.concatenate(([0], (horizontal[r, x : x + w] > 0).astype(np.int8), [0]))
        edges = np.flatnonzero(np.diff(row))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a > best[0]:
                best = (b - a, a, r)
    length, start, _ = best
    return x + int(start), y, int(length), h


def _join_segments(rects: list[tuple[int, int, int, int]], gap: int) -> list[tuple[int, int, int, int]]:
    """Join pieces of one line split by a mark: a typed "_X_" is underscore, X, underscore."""
    joined: list[list[int]] = []
    for x, y, w, h in sorted(rects, key=lambda r: (r[0], r[1])):
        for j in joined:
            dx, dy = x - (j[0] + j[2]), abs(j[1] - y)
            # a mark leaves a gap of a few pixels; touching pieces on another row are letters' feet
            if dy <= 2 and 3 <= dx <= gap or dy == 0 and 0 <= dx < 3:
                j[2] = x + w - j[0]
                j[3] = max(j[3], h)
                break
        else:
            joined.append([x, y, w, h])
    return [tuple(j) for j in joined]


def _inside(line: OCRLine, region: tuple[int, int, int, int]) -> float:
    """Share of the OCR line's box that lies inside the region."""
    w = min(line.x1, region[2]) - max(line.x0, region[0])
    h = min(line.y1, region[3]) - max(line.y0, region[1])
    area = (line.x1 - line.x0) * (line.y1 - line.y0)
    return w * h / area if w > 0 and h > 0 and area > 0 else 0.0


def _framed(bw: np.ndarray, x: int, y: int, w: int, h: int, need: float = 0.9) -> bool:
    """All four sides of the rectangle are (nearly) continuous ink lines."""
    band = max(2, round(min(w, h) * 0.2))  # a stray pixel can widen the bounding rect by one
    region = bw[y : y + h, x : x + w] > 0
    if region.shape != (h, w) or min(w, h) < 2 * band + 2:
        return False
    sides = [region[:band].any(axis=0), region[-band:].any(axis=0),  # top, bottom: per column
             region[:, :band].any(axis=1), region[:, -band:].any(axis=1)]  # left, right: per row
    return all(side.mean() >= need for side in sides)


def label_boxes(boxes: list[Box], lines: list[OCRLine], image: Image.Image | None = None) -> None:
    """Attach to each box the OCR text that starts at (or runs through) its right edge.

    Boxes left unlabeled get a targeted OCR pass on the strip to their right, if `image` is given.
    """
    _label_from_lines(boxes, lines)
    _left_labels(boxes, lines)
    if image is None:
        return
    for box in boxes:
        if not box.label and box.kind == "box":
            strip = ocr_region(image, box.x + box.w * 0.8, box.y - box.h * 0.6, box.x + box.w * 12, box.y + box.h * 1.6)
            _label_from_lines([box], strip)


def _label_from_lines(boxes: list[Box], lines: list[OCRLine]) -> None:
    for box in boxes:
        best, best_dx = "", None
        cy = box.y + box.h / 2
        right = box.x + box.w
        for line in lines:
            if abs((line.y0 + line.y1) / 2 - cy) > box.h or line.x1 <= right or _is_mark(line.text):
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


def _is_mark(text: str) -> bool:
    """OCR text that is just a tick mark ("X", "x", "V", "✓"), not a label."""
    return text.strip(" _-.,:;|[]()") in {"X", "x", "V", "v", "✓", "✔", "/", "XX", "xx"}


def _left_labels(boxes: list[Box], lines: list[OCRLine]) -> None:
    for box in boxes:
        cy = box.y + box.h / 2
        near = [l for l in lines if abs((l.y0 + l.y1) / 2 - cy) <= box.h and l.x0 < box.x
                and box.x - box.w * 4 <= l.x1 <= box.x + box.w * 1.2 and not _is_mark(l.text)]
        if near:
            # the nearest line on the box's own row, not a heading just above it
            line = min(near, key=lambda l: (abs((l.y0 + l.y1) / 2 - cy) > box.h * 0.6, -l.x1))
            # a line running into the box may include its glyph ("SEP 22]"): _norm drops it
            box.left = _norm(line.text)


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
    if best < 0.75 and len(opt) >= 8:
        if len(label) >= 8 and opt.startswith(label):
            return 0.9  # a long option wrapped onto the next line: OCR's label is its first part
        if fuzz.ratio(opt, label[: len(opt)]) >= 85:
            return 0.8  # OCR slips inside a long word ("Compatibity")
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
    best, best_cost = None, None
    for side, penalty in (("right", 0), ("left", 250)):  # labels usually sit right of their boxes
        found = _best_cluster(options, boxes, used, hint, contexts, side)
        if found and (best_cost is None or found[1] + penalty < best_cost):
            found = (found[0], found[1] + penalty)
            best, best_cost = found
    return {opt: boxes[i] for opt, i in zip(options, best)} if best else None


def _best_cluster(options, boxes, used, hint, contexts, side):
    """(box indexes, cost) of the best cluster with every label on `side` of its box, or None."""
    def score(option: str, box: Box) -> float:
        if side == "right":
            return _label_score(option, box.label)
        # labels on the left end right at the box, so compare from the end
        return _label_score(_norm(option)[::-1], box.left[::-1])

    candidates = []
    for option in options:
        scored = [(i, score(option, b)) for i, b in enumerate(boxes) if i not in used]
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
        if max(ys) - min(ys) > max(boxes[i].h for i in idx):
            spread += 100  # options on one row beat a mix of rows ("12 MOS." vs "12 MOS. WHSE." below)
        cost = spread - 500 * sum(s for _, s in combo)
        if hint and contexts:
            first = min(idx, key=lambda i: (boxes[i].y // boxes[i].h, boxes[i].x))
            cost -= 2000 * _hint_score(hint, contexts[first])
        if best_cost is None or cost < best_cost:
            best, best_cost = idx, cost
    return (best, best_cost) if best else None


@dataclass
class GroupCheck:
    options: dict[str, Box]  # option -> located box

    def states(self) -> dict[str, bool]:
        return {o: b.checked for o, b in self.options.items()}


class PageCheckboxes:
    """Boxes found on one page, with bookkeeping so each box serves only one group."""

    def __init__(self, image: Image.Image, lines: list[OCRLine]):
        self.boxes = detect_boxes(image)
        self.boxes += detect_blanks(image, lines, self.boxes)
        label_boxes(self.boxes, lines, image)
        self.contexts = [self._left_context(b, lines) for b in self.boxes]
        self.used: set[int] = set()

    @classmethod
    def from_cache(cls, boxes: list[Box], contexts: list[str]) -> "PageCheckboxes":
        """Rebuild from previously detected and labeled boxes (skips detection and OCR)."""
        page = cls.__new__(cls)
        page.boxes, page.contexts, page.used = boxes, contexts, set()
        return page

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
