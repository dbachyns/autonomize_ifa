"""Cross-check an LLM-extracted record against independent local evidence.

- Checkboxes: pixel ink inside each located box. Decisive pixels override the LLM.
- Text values: does local OCR see the same text on the page? Mismatches are flagged
  (with what OCR read) but values are never silently rewritten.

Returns the updated record plus verification flags.
"""

from __future__ import annotations

import re

from PIL import Image
from rapidfuzz import fuzz

from form_agent.checkboxes import CLEAR_EMPTY, CLEAR_TICK, PageCheckboxes
from form_agent.forms import get_plugin
from form_agent.ocr import OCRLine, ocr_page
from form_agent.schema import Flag, FormRecord, flatten

OCR_KINDS = {"text", "name", "date", "phone", "id", "npi", "code", "address", "number", "money", "email"}
CODE_KINDS = {"date", "phone", "id", "npi", "code", "number", "money"}  # where O/0, I/1 confusions are noise


def _loose(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


# Characters OCR engines routinely confuse; equal after this mapping means "no real disagreement".
_CONFUSABLE = str.maketrans({"o": "0", "i": "1", "l": "1", "s": "5", "b": "8", "z": "2", "q": "0"})


def _unconfuse(text: str) -> str:
    return _loose(text).translate(_CONFUSABLE)


class PageEvidence:
    def __init__(self, image: Image.Image):
        self.lines: list[OCRLine] = ocr_page(image)
        self.checkboxes = PageCheckboxes(image, self.lines)
        self.loose_lines = [_loose(l.text) for l in self.lines]
        self.loose_all = "".join(self.loose_lines)
        self.unconfused_all = self.loose_all.translate(_CONFUSABLE)

    def reset_boxes(self) -> None:
        self.checkboxes.used = set()

    def find_text(self, value: str, value_kind: str = "text") -> tuple[float, str | None]:
        """(similarity 0..1, best OCR reading) for a value on this page."""
        target = _loose(value)
        if not target:
            return 1.0, None
        if target in self.loose_all or (value_kind in CODE_KINDS and _unconfuse(value) in self.unconfused_all):
            return 1.0, value
        best, reading = 0.0, None
        for line, loose in zip(self.lines, self.loose_lines):
            score = fuzz.partial_ratio(target, loose) / 100 if len(loose) >= len(target) else fuzz.ratio(target, loose) / 100
            if score > best:
                best, reading = score, line.text
        return best, reading


def verify(record: FormRecord, pages: list[Image.Image]) -> tuple[FormRecord, list[Flag]]:
    evidence = [PageEvidence(p) for p in pages]
    record = record.model_copy(deep=True)
    flags: list[Flag] = []
    plugin = get_plugin(record.form_type)

    # --- typed checkbox groups (known form types) -----------------------------------
    if plugin and record.typed:
        typed = plugin.parse(record.typed)
        for ev in evidence:
            ev.reset_boxes()
        for path, options, hint in plugin.checkbox_groups():
            group = next((g for ev in evidence if (g := ev.checkboxes.locate(options, hint))), None)
            if group is None:
                continue
            current = _get(typed, path)
            is_bool = isinstance(current, bool)
            ticked = {options[0]} if (is_bool and current) else (set() if is_bool else set(current))
            corrected = set(ticked)
            for option, box in group.options.items():
                llm, px = option in ticked, box.checked
                if llm == px:
                    continue
                if box.decisive:
                    corrected.symmetric_difference_update({option})
                    flags.append(Flag(
                        rule="checkbox_corrected", severity="info", fields=[path],
                        message=f"{path}: '{option}' box has {box.fill:.1%} ink, so it is "
                                f"{'ticked' if px else 'not ticked'}; the vision model said otherwise. Corrected.",
                    ))
                else:
                    flags.append(Flag(
                        rule="checkbox_uncertain", severity="warning", fields=[path],
                        message=f"{path}: '{option}' box has {box.fill:.1%} ink (ambiguous); kept the vision "
                                f"model's reading ({'ticked' if llm else 'not ticked'}).",
                    ))
            if corrected != ticked:
                value = bool(corrected) if is_bool else [o for o in options if o in corrected]
                _set(typed, path, value)
        record.typed = typed.model_dump()

        # OCR cross-check of typed text values
        for path, (value, value_kind) in flatten(typed).items():
            if isinstance(value, str) and value_kind in OCR_KINDS:
                flags += _ocr_check(path, value, value_kind, evidence)

    # --- universal checkbox groups ------------------------------------------------------
    for ev in evidence:
        ev.reset_boxes()
    for g in sorted(record.universal.checkboxes, key=lambda g: -len(g.options)):
        if not 1 <= g.page <= len(evidence):
            continue
        group = evidence[g.page - 1].checkboxes.locate([o.label for o in g.options], g.label)
        if group is None:
            continue
        for option in g.options:
            box = group.options[option.label]
            option.bbox = (box.x, box.y, box.x + box.w, box.y + box.h)
            if box.decisive:
                option.method = "llm+pixel" if option.checked == box.checked else "pixel"
                option.checked = box.checked
                option.confidence = 0.97 if option.method == "llm+pixel" else 0.9
            else:
                option.method = "llm"
                option.confidence = 0.6

    # --- universal field confidence from OCR agreement -----------------------------------
    for f in record.universal.fields:
        if f.value and f.value_type in OCR_KINDS and 1 <= f.page <= len(evidence):
            score, _ = evidence[f.page - 1].find_text(f.value, f.value_type)
            f.confidence = 0.95 if score == 1.0 else 0.7 if score >= 0.85 else 0.4
    if not (plugin and record.typed):
        for f in record.universal.fields:
            if f.value and f.value_type in OCR_KINDS:
                flags += _ocr_check(f.key or f.label, f.value, f.value_type, evidence[f.page - 1:f.page] or evidence)
    return record, flags


def _ocr_check(name: str, value: str, value_kind: str, evidence: list[PageEvidence]) -> list[Flag]:
    if len(_loose(value)) < 4:  # too short to verify meaningfully ("1", "10", "M")
        return []
    score, reading = max((ev.find_text(value, value_kind) for ev in evidence), key=lambda t: t[0])
    if score == 1.0:
        return []
    if score >= 0.85 and reading and len(_loose(reading)) < len(_loose(value)) and fuzz.partial_ratio(
        _loose(reading), _loose(value)
    ) == 100:
        # OCR saw only part of the value (dropped characters); that is not evidence against it.
        return []
    if score >= 0.85:
        return [Flag(rule="ocr_disagrees", severity="warning", fields=[name],
                     message=f"{name}: extracted '{value}' but local OCR reads '{reading}'. Check the spelling/digits.")]
    return [Flag(rule="ocr_not_found", severity="info", fields=[name],
                 message=f"{name}: '{value}' could not be confirmed by local OCR.")]


def _get(model, path: str):
    for part in path.split("."):
        model = getattr(model, part)
    return model


def _set(model, path: str, value) -> None:
    *parents, last = path.split(".")
    for part in parents:
        model = getattr(model, part)
    setattr(model, last, value)


__all__ = ["verify", "CLEAR_EMPTY", "CLEAR_TICK"]
