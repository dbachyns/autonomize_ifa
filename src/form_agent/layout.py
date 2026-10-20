"""Where things are on the page: OCR lines and checkbox squares, cached per document.

The layout is computed locally (RapidOCR + OpenCV, no API) once per scan and saved as
`data/cache/<doc_id>/_layout.json`. It lets the UI draw extracted values on the scan,
gives each value an OCR-based confidence, and locates checkbox groups by their pixels.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image
from rapidfuzz import fuzz

from form_agent.checkboxes import Box, PageCheckboxes
from form_agent.config import settings
from form_agent.ocr import OCRLine, ocr_page

BBox = tuple[float, float, float, float]
LAYOUT_VERSION = 4  # 2: small pages OCR'd at 1x and 2x; 3: wide boxes, labels left; 4: blank-line marks, small boxes


def _loose(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


_CONFUSABLE = str.maketrans({"o": "0", "i": "1", "l": "1", "s": "5", "b": "8", "z": "2", "q": "0"})


@dataclass
class Hit:
    page: int  # 1-based
    bbox: BBox
    score: float  # 1.0 exact, lower = fuzzy
    reading: str  # the OCR text that matched


class PageLayout:
    def __init__(self, page: int, size: tuple[int, int], lines: list[OCRLine], boxes: list[Box], contexts: list[str]):
        self.page, self.size, self.lines, self.boxes, self.contexts = page, size, lines, boxes, contexts
        self.loose = [_loose(l.text) for l in lines]

    @classmethod
    def compute(cls, page: int, image: Image.Image) -> "PageLayout":
        lines = ocr_page(image)
        checks = PageCheckboxes(image, lines)
        return cls(page, image.size, lines, checks.boxes, checks.contexts)

    def checkboxes(self) -> PageCheckboxes:
        return PageCheckboxes.from_cache(self.boxes, self.contexts)

    def find(self, value: str, labels: list[BBox] | None = None, code_like: bool = False,
             exclude: list[BBox] | None = None) -> Hit | None:
        """Best occurrence of `value` on this page.

        Tries, in order: exact substring of an OCR line, a value wrapped onto the next line, a paragraph
        assembled from several lines, a fuzzy line match, and finally a partial reading next to the field's
        label (OCR saw "Direct" where the value is "Direct Express"). Among equally good hits, the one
        closest to one of the field's `labels` wins. Regions in `exclude` already confirmed another value,
        so they can't confirm this one too. `code_like` values ignore OCR look-alikes such as O/0 and I/1.
        """
        fold = (lambda t: t.translate(_CONFUSABLE)) if code_like else (lambda t: t)
        target = fold(_loose(value))
        if not target:
            return None
        labels, exclude = labels or [], exclude or []
        free = lambda box: not any(_overlap(box, e) > 0.5 for e in exclude)
        hits: list[Hit] = []
        for line, loose in zip(self.lines, self.loose):
            loose = fold(loose)
            start = loose.find(target)
            while start >= 0:
                box = _sub_box(line, loose, start, len(target))
                if free(box):
                    hits.append(Hit(self.page, box, 1.0, line.text))
                start = loose.find(target, start + 1)
        if not hits:  # values wrapped onto the next line ("… Apt." / "Texas 83462")
            for a, b, la, lb in zip(self.lines, self.lines[1:], self.loose, self.loose[1:]):
                box = (min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1))
                if len(target) > max(len(la), len(lb)) and target in fold(la + lb) and free(box):
                    hits.append(Hit(self.page, box, 1.0, f"{a.text} {b.text}"))
        if not hits and len(target) > 30 and (para := self._paragraph(target)) and free(para.bbox):
            hits.append(para)
        if not hits and len(target) >= 4:
            for line, loose in zip(self.lines, self.loose):
                score = (fuzz.partial_ratio(target, loose) if len(loose) >= len(target) else fuzz.ratio(target, loose)) / 100
                box = (line.x0, line.y0, line.x1, line.y1)
                if score >= 0.8 and free(box):
                    hits.append(Hit(self.page, box, score, line.text))
        if not hits and labels:
            hits = self._partial_near(target, labels, free)
        if not hits:
            return None
        best_score = max(h.score for h in hits)
        best = [h for h in hits if h.score == best_score]
        if labels and len(best) > 1:
            best.sort(key=lambda h: min(_distance(h.bbox, lb) for lb in labels))
        return best[0]

    def _partial_near(self, target: str, labels: list[BBox], free) -> list[Hit]:
        """OCR lines right next to the label that read part of the value (a word, or the value with slips)."""
        width, height = self.size
        hits = []
        for line, loose in zip(self.lines, self.loose):
            box = (line.x0, line.y0, line.x1, line.y1)
            if len(loose) < 4 or not free(box):
                continue
            near = min(_distance(box, lb) for lb in labels) < width * 0.35 and \
                any(abs((box[1] + box[3]) / 2 - (lb[1] + lb[3]) / 2) < height * 0.025 for lb in labels)
            if near and len(loose) >= 0.3 * len(target) and (loose in target or fuzz.partial_ratio(loose, target) >= 85):
                hits.append(Hit(self.page, box, 0.75, line.text))
        return hits

    def _paragraph(self, target: str) -> Hit | None:
        """Long text wrapped over several OCR lines: a block of consecutive lines that are pieces of it, if they
        cover most of it. Lines elsewhere on the page that happen to share a phrase ("at room temperature")
        are not part of the block."""
        pieces = {}  # line index -> exact piece?
        for i, (line, loose) in enumerate(zip(self.lines, self.loose)):
            if len(loose) >= 6 and loose in target:
                pieces[i] = True
            elif len(loose) >= 12 and fuzz.partial_ratio(loose, target) >= 90:
                pieces[i] = False
        best = None
        for block in self._blocks(sorted(pieces, key=lambda i: (self.lines[i].y0, self.lines[i].x0))):
            exact = [i for i in block if pieces[i]]
            covered, score = sum(len(self.loose[i]) for i in exact), 1.0
            if covered < 0.8 * len(target):  # allow a few lines with OCR slips, but then it is not an exact match
                exact, score = block, 0.9
                covered = sum(len(self.loose[i]) for i in block)
            if exact and covered >= 0.8 * len(target) and (best is None or (score, covered) > best[:2]):
                best = (score, covered, exact)
        if best is None:
            return None
        lines = [self.lines[i] for i in best[2]]
        box = (min(l.x0 for l in lines), min(l.y0 for l in lines), max(l.x1 for l in lines), max(l.y1 for l in lines))
        return Hit(self.page, box, best[0], " ".join(l.text for l in sorted(lines, key=lambda l: (l.y0, l.x0))))

    def _blocks(self, order: list[int]) -> list[list[int]]:
        """Group lines into blocks of touching lines: one right below another, or side by side on a row."""
        def near(a: OCRLine, b: OCRLine) -> bool:
            h = max(a.y1 - a.y0, b.y1 - b.y0)
            x_gap = max(a.x0, b.x0) - min(a.x1, b.x1)  # negative when they overlap horizontally
            y_gap = max(a.y0, b.y0) - min(a.y1, b.y1)
            return (x_gap < 0 and y_gap <= 1.2 * h) or (y_gap < 0 and x_gap <= 3 * h)

        blocks: list[list[int]] = []
        for i in order:
            touching = [b for b in blocks if any(near(self.lines[i], self.lines[j]) for j in b)]
            merged = [i] + [j for b in touching for j in b]
            blocks = [b for b in blocks if b not in touching] + [merged]
        return blocks

    def find_labels(self, label: str) -> list[BBox]:
        """Positions of a printed label. Lines that start with it come first ("DATE:" before "EXPIRATION DATE:")."""
        target = _loose(label)
        if len(target) < 3:
            return []
        starts, inside, fuzzy = [], [], []
        for line, loose in zip(self.lines, self.loose):
            start = loose.find(target)
            if start >= 0:
                (starts if start == 0 else inside).append(_sub_box(line, loose, start, len(target)))
            elif len(target) >= 6 and len(loose) >= 0.7 * len(target):
                # OCR slips in the label itself ("SHLPPED-VIA:"): align the label to the start of the line
                head = loose[: len(target)]
                if fuzz.ratio(target, head) >= 80:
                    fuzzy.append(_sub_box(line, loose, 0, min(len(target), len(loose))))
        return starts or inside or fuzzy

    def to_dict(self) -> dict:
        return {"page": self.page, "size": list(self.size), "lines": [asdict(l) for l in self.lines],
                "boxes": [asdict(b) for b in self.boxes], "contexts": self.contexts}

    @classmethod
    def from_dict(cls, d: dict) -> "PageLayout":
        return cls(d["page"], tuple(d["size"]), [OCRLine(**l) for l in d["lines"]], [Box(**b) for b in d["boxes"]],
                   d["contexts"])


def _sub_box(line: OCRLine, loose: str, start: int, length: int) -> BBox:
    """Approximate the box of a substring by its share of the line's characters."""
    if not loose:
        return line.x0, line.y0, line.x1, line.y1
    width = line.x1 - line.x0
    x0 = line.x0 + width * start / len(loose)
    x1 = line.x0 + width * (start + length) / len(loose)
    return x0, line.y0, x1, line.y1


def _overlap(a: BBox, b: BBox) -> float:
    """Intersection over the smaller box (1.0 = one contains the other)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / smaller if smaller > 0 else 0.0


def _distance(a: BBox, b: BBox) -> float:
    """Distance from label b to value a; values sit right of or below their label, so penalize left/above."""
    dx = a[0] - b[2]
    dy = (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2
    return (abs(dx) * (3 if dx < -20 else 1)) + abs(dy) * (3 if dy < -10 else 1.5)


# --- cache ---------------------------------------------------------------------------------


def layout_path(cache_dir: Path, doc_id: str) -> Path:
    return cache_dir / doc_id / "_layout.json"


def source_path(source_file: str, data_dir: Path | None = None) -> Path:
    path = Path(source_file)
    return path if path.is_absolute() or path.exists() else (data_dir or settings.data_dir) / path


def load_layout(cache_dir: Path, doc_id: str) -> list[PageLayout] | None:
    path = layout_path(cache_dir, doc_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("version") != LAYOUT_VERSION:
        return None
    return [PageLayout.from_dict(p) for p in data["pages"]]


def compute_layout(cache_dir: Path, doc_id: str, pages: list[Image.Image]) -> list[PageLayout]:
    layouts = [PageLayout.compute(i, page) for i, page in enumerate(pages, 1)]
    path = layout_path(cache_dir, doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": LAYOUT_VERSION, "pages": [l.to_dict() for l in layouts]}) + "\n")
    return layouts


def get_layout(cache_dir: Path, doc_id: str, source_file: str, compute: bool = True) -> list[PageLayout] | None:
    """Cached layout, computing it from the scan if allowed and the scan is available."""
    cached = load_layout(cache_dir, doc_id)
    if cached is not None or not compute:
        return cached
    path = source_path(source_file)
    if not path.exists():
        return None
    from form_agent.ingest import load_pages

    return compute_layout(cache_dir, doc_id, load_pages(path))
