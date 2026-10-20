"""One form as a list of reviewable items: value, position on the scan, confidence, flags.

This is what the agent reads (`render_text`) and what the UI shows. Each item has an
`id` that says where its value lives in the record, so a reviewer's edit can be written
back: `typed:<path>` for known form types, `universal:<path>` otherwise.

Confidence is about *reading*: did we transcribe the form correctly? It comes from local
evidence (does OCR see the same text, are the checkbox pixels decisive) and from human
review. Problems with the *content* (an NPI that fails its check digit) are flags, shown
separately: a value can be read perfectly and still be wrong on the form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from form_agent.forms import get_plugin
from form_agent.layout import BBox, Hit, PageLayout, _overlap
from form_agent.normalize import parse_date
from form_agent.schema import CheckboxGroup, Flag, FormRecord, flag_id
from form_agent.validate import READER_RULES

CODE_LIKE = {"date", "phone", "id", "npi", "code", "number", "money"}  # where OCR look-alikes (O/0) are noise
Level = Literal["reviewed", "high", "medium", "low", "blank", "unknown"]
LEVELS: dict[str, str] = {
    "reviewed": "checked or corrected by a person",
    "high": "local OCR / checkbox pixels confirm the reading",
    "medium": "not confirmed: OCR could not find it, or reads it slightly differently",
    "low": "evidence disagrees or is ambiguous; check the scan",
    "blank": "left empty on the form",
    "unknown": "could not be checked",
}


@dataclass
class ViewItem:
    id: str
    kind: Literal["field", "checkbox", "cell", "text"]
    label: str
    value: str | None
    section: str | None = None
    value_type: str = "text"
    source: str = "filled"
    page: int = 1
    options: list[tuple[str, bool]] | None = None  # checkbox groups: (option, ticked)
    boxes: list[tuple[int, BBox]] = field(default_factory=list)  # (page, box) where the value is on the scan
    confidence: float | None = None
    level: Level = "unknown"
    reasons: list[str] = field(default_factory=list)
    flags: list[Flag] = field(default_factory=list)
    keys: set[str] = field(default_factory=set)  # names flags use for this item
    cell: tuple[int, int] | None = None  # table cells: (table index, row index)
    table: str | None = None  # table cells: the table's name

    @property
    def issues(self) -> list[Flag]:
        """Problems with the form's content (bad NPI, missing reason, ...), as opposed to reading problems."""
        return [f for f in self.flags if f.rule not in READER_RULES]

    @property
    def reading(self) -> str:
        """One plain sentence on how well the reading is confirmed."""
        return self.reasons[0] if self.reasons else ""

    @property
    def editable(self) -> bool:
        return self.id.startswith(("typed:", "universal:"))


@dataclass
class DocView:
    record: FormRecord
    short_id: str
    items: list[ViewItem]
    layout: list[PageLayout] | None

    def item(self, item_id: str) -> ViewItem:
        return next(i for i in self.items if i.id == item_id)

    @property
    def flags(self) -> list[Flag]:
        return self.record.flags

    def acknowledged(self, flag: Flag) -> bool:
        return bool(self.record.review and flag_id(flag) in self.record.review.acknowledged)


# --- building ------------------------------------------------------------------------------


def _norm_key(name: str) -> str:
    return re.sub(r"\[(\d+)\]", r".\1", name)


def build_view(record: FormRecord, layout: list[PageLayout] | None = None, short_id: str | None = None) -> DocView:
    items = _typed_items(record) if (get_plugin(record.form_type) and record.typed) else _universal_items(record)
    view = DocView(record, short_id or record.doc_id, items, layout)
    _attach_flags(view)
    _locate_and_score(view)
    return view


def _typed_items(record: FormRecord) -> list[ViewItem]:
    plugin = get_plugin(record.form_type)
    typed = plugin.parse(record.typed)
    tv = plugin.to_universal(typed)
    paths, table_paths = plugin.field_paths(), plugin.table_paths()
    items: list[ViewItem] = []

    # Pre-printed values come from the generic pass (the typed schema holds only filled-in values).
    typed_values = {f.value for f in tv.fields if f.value}
    for i, f in enumerate(record.universal.fields):
        if f.source == "printed" and f.value and f.value not in typed_values:
            items.append(ViewItem(f"universal:fields.{i}", "field", f.label, f.value, f.section, f.value_type,
                                  "printed", f.page, keys={f.key or "", f.label}))
    for f in tv.fields:
        path = paths.get(f.key or "")
        items.append(ViewItem(f"typed:{path}" if path else f"view:{f.key}", "field", f.label, f.value, f.section,
                              f.value_type, keys={path or f.key or ""}))
    for g in tv.checkboxes:
        path = paths.get(g.key or "")
        items.append(_checkbox_item(f"typed:{path}" if path else f"view:{g.key}", g, {path or g.key or ""}))
    for t_index, t in enumerate(tv.tables):
        list_path, attrs = table_paths.get(t_index, (None, []))
        for r, row in enumerate(t.rows):
            for c, cell in enumerate(row):
                path = f"{list_path}.{r}.{attrs[c]}" if list_path and c < len(attrs) else None
                header = t.headers[c] if c < len(t.headers) else f"col{c + 1}"
                items.append(ViewItem(f"typed:{path}" if path else f"view:{t_index}.{r}.{c}", "cell",
                                      f"{t.title or 'Table'} · row {r + 1} · {header}", cell, t.section,
                                      t.column_types[c] if c < len(t.column_types) else "text", keys={path or ""},
                                      cell=(t_index, r), table=t.name))
    for i, b in enumerate(record.universal.text_blocks):
        if b.text not in typed_values:
            items.append(ViewItem(f"universal:text_blocks.{i}", "text", b.label or "Free text", b.text, b.section,
                                  page=b.page))
    return items


def _universal_items(record: FormRecord) -> list[ViewItem]:
    u = record.universal
    items: list[ViewItem] = []
    for i, f in enumerate(u.fields):
        items.append(ViewItem(f"universal:fields.{i}", "field", f.label, f.value, f.section, f.value_type,
                              f.source, f.page, keys={f.key or "", f.label}))
    for i, g in enumerate(u.checkboxes):
        items.append(_checkbox_item(f"universal:checkboxes.{i}", g, {g.key or "", g.label}))
    for t_index, t in enumerate(u.tables):
        for r, row in enumerate(t.rows):
            for c, cell in enumerate(row):
                header = t.headers[c] if c < len(t.headers) else f"col{c + 1}"
                items.append(ViewItem(f"universal:tables.{t_index}.rows.{r}.{c}", "cell",
                                      f"{t.title or 'Table'} · row {r + 1} · {header}", cell, t.section,
                                      t.column_types[c] if c < len(t.column_types) else "text", page=t.page,
                                      keys={f"{t.title or 'table'}[{r}].{header}"}, cell=(t_index, r), table=t.name))
    for i, b in enumerate(u.text_blocks):
        items.append(ViewItem(f"universal:text_blocks.{i}", "text", b.label or "Free text", b.text, b.section,
                              page=b.page))
    return items


def _checkbox_item(item_id: str, g: CheckboxGroup, keys: set[str]) -> ViewItem:
    item = ViewItem(item_id, "checkbox", g.label, ", ".join(g.selected) or None, g.section, "checkbox",
                    page=g.page, options=[(o.label, o.checked) for o in g.options], keys=keys)
    if all(o.bbox for o in g.options) and g.options:  # positions from the hybrid reader
        item.boxes = [(g.page, o.bbox) for o in g.options]
    return item


def _attach_flags(view: DocView) -> None:
    for flag in view.record.flags:
        names = {_norm_key(n) for n in flag.fields}
        for item in view.items:
            if names & {_norm_key(k) for k in item.keys if k}:
                item.flags.append(flag)


# --- positions and confidence --------------------------------------------------------------


def _locate_and_score(view: DocView) -> None:
    record, layout = view.record, view.layout
    review = record.review
    reviewed = set(review.confirmed) | {c.item for c in review.corrections} if review else set()
    group_hits = _locate_checkbox_groups(view) if layout else {}
    claimed: dict[int, list[BBox]] = {}  # page -> regions that already confirmed a value
    rows: dict[tuple[int, int], tuple[int, BBox]] = {}  # table row -> (page, box) of its located cells

    for item in view.items:
        reader_flags = [f for f in item.flags if f.rule in READER_RULES]
        if item.kind == "checkbox":
            _score_checkbox(item, group_hits.get(item.id), reader_flags, layout)
        elif not item.value:
            item.level, item.reasons = "blank", ["Left empty on the form"]
        elif layout is None:
            item.level, item.reasons = "unknown", ["Not checked: the scan's layout isn't available"]
        else:
            _score_text(item, layout, claimed, rows.get(item.cell) if item.cell else None)
            if item.cell and item.boxes:
                page, box = item.boxes[0]
                if item.cell in rows and rows[item.cell][0] == page:
                    old = rows[item.cell][1]
                    box = (min(old[0], box[0]), min(old[1], box[1]), max(old[2], box[2]), max(old[3], box[3]))
                rows[item.cell] = (page, box)
        for f in reader_flags:
            if f.rule == "checkbox_corrected":
                item.reasons.insert(0, "Corrected from the checkbox ink: the vision model misread a stray mark")
            elif f.rule == "ocr_not_found" and item.boxes:
                continue  # found now (e.g. text wrapped over several lines), so the old flag doesn't apply
            else:
                item.confidence = min(item.confidence or 0.5, 0.5)
                if f.rule == "checkbox_uncertain":
                    item.reasons.insert(0, "Some boxes have an ambiguous amount of ink; check the scan")
                # ocr_disagrees / ocr_not_found repeat what the OCR comparison above already says
        if item.id in reviewed:
            item.confidence, item.level = 1.0, "reviewed"
            item.reasons.insert(0, "Checked by a reviewer")
        elif item.level != "blank" and item.confidence is not None:
            item.level = "high" if item.confidence >= 0.9 else "medium" if item.confidence >= 0.6 else "low"


def _score_text(item: ViewItem, layout: list[PageLayout], claimed: dict[int, list[BBox]],
                row: tuple[int, BBox] | None = None) -> None:
    """Find the value on the scan and say how well local OCR confirms it.

    A region of the page can confirm only one value: two "DATE:" fields can't both lean on the same
    "03/01/90" when the second one actually says "03/02/90". `row` is where the other cells of a table
    row were found: a short cell ("21") is only trusted on that line.
    """
    pages = sorted(layout, key=lambda p: p.page != item.page)  # the item's own page first
    label = item.label.split("·")[-1]
    if (item.value or "").startswith("[signed]"):
        # OCR can't read a signature; point at the area right of the label so a person can look.
        for page in pages:
            if labels := page.find_labels(label):
                x0, y0, x1, y1 = labels[0]
                h = y1 - y0
                item.boxes = [(page.page, (x1, y0 - 1.5 * h, min(page.size[0], x1 + page.size[0] * 0.35), y1 + h))]
                break
        item.confidence = 0.65
        item.reasons = ["A signature: OCR can't check it, so compare with the scan"]
        return
    short = len(re.sub(r"[^A-Za-z0-9]", "", item.value or "")) < 3
    best = None
    if short and row and (hit := _find_in_row(item, layout, row, claimed)):
        best, pages = hit, []
    for page in pages:
        labels = page.find_labels(label) if item.kind != "text" else []
        if short and not labels:
            continue  # "4" or "10" is everywhere; only trust it next to its label
        hit = page.find(item.value, labels, code_like=item.value_type in CODE_LIKE, exclude=claimed.get(page.page))
        if hit and short and all(_far(hit.bbox, lb, page) for lb in labels):
            hit = None
        if hit and (best is None or hit.score > best.score):
            best = hit
        if best and best.score == 1.0:
            break
    if (best is None or best.score < 1.0) and (parts := _number_parts(item.value)):
        # A field holding several numbers ("Phone: UM 855-322-4080 Imaging 855-714-2415"): find each. The numbers
        # are what matters, so this beats an approximate match of the whole text (a dropped "UM" before a label).
        found = [h for part in parts for page in pages if (h := page.find(part, code_like=True))]
        if len(found) >= len(parts):
            item.confidence, item.boxes = 0.9, [(h.page, h.bbox) for h in found]
            item.reasons = [f"Confirmed: local OCR finds all {len(parts)} numbers"]
            return
    if best is None:  # no evidence either way: unconfirmed, not contradicted
        item.confidence = 0.65
        item.reasons = ["Not confirmed: local OCR couldn't find this text (handwriting, faint or clipped print)"]
        return
    claimed.setdefault(best.page, []).append(best.bbox)
    item.boxes = [(best.page, best.bbox)]
    if best.score == 1.0:
        item.confidence, item.reasons = 0.95, ["Confirmed: local OCR reads the same text"]
    elif best.score == 0.75:
        item.confidence = 0.65
        item.reasons = [f"Partly confirmed: next to the label, OCR only reads '{best.reading}'"]
    else:
        item.confidence = 0.7 if best.score >= 0.85 else 0.5
        item.reasons = [f"OCR reads it differently: '{best.reading}'. Compare with the scan"]


def _find_in_row(item: ViewItem, layout: list[PageLayout], row: tuple[int, BBox],
                 claimed: dict[int, list[BBox]]) -> Hit | None:
    """An OCR line reading exactly this cell's value on the same line as the rest of its table row."""
    page = next((p for p in layout if p.page == row[0]), None)
    if page is None:
        return None
    _, (_, y0, _, y1) = row
    target, exclude = re.sub(r"[^a-z0-9]", "", item.value.lower()), claimed.get(page.page, [])
    headers = page.find_labels(item.label.split("·")[-1])
    found = []
    for line, loose in zip(page.lines, page.loose):
        box = (line.x0, line.y0, line.x1, line.y1)
        same_line = abs((line.y0 + line.y1) / 2 - (y0 + y1) / 2) < max(y1 - y0, line.y1 - line.y0) * 0.6
        if loose == target and same_line and not any(_overlap(box, e) > 0.5 for e in exclude):
            found.append(Hit(page.page, box, 1.0, line.text))
    if headers:  # under its column's header
        found.sort(key=lambda h: min(abs((h.bbox[0] + h.bbox[2]) - (b[0] + b[2])) for b in headers))
    return found[0] if found else None


def _number_parts(value: str) -> list[str]:
    parts = re.findall(r"\(?\d[\d\-() ]{5,}\d", value)
    return parts if len(parts) >= 2 else []


def _far(box: BBox, near: BBox | None, page: PageLayout) -> bool:
    if near is None:
        return True
    return abs(box[0] - near[2]) > page.size[0] * 0.35 or abs(box[1] - near[1]) > page.size[1] * 0.05


def _locate_option_texts(item: ViewItem, layout: list[PageLayout]) -> list[tuple[int, BBox]]:
    chosen = [o for o, c in item.options or [] if c]
    found = []
    for page in sorted(layout, key=lambda p: p.page != item.page):
        labels = page.find_labels(item.label)
        hits = [page.find(o, labels) for o in chosen]
        if chosen and all(hits):
            return [(h.page, h.bbox) for h in hits]
    return found


def _locate_checkbox_groups(view: DocView) -> dict[str, list[tuple[int, object]]]:
    """item id -> [(page, Box) per option], using the same pixel matching as the hybrid reader."""
    plugin = get_plugin(view.record.form_type)
    plugin_groups = plugin.checkbox_groups() if plugin and view.record.typed else []
    hints = {f"typed:{p}": h for p, _, h in plugin_groups}
    rank = {f"typed:{p}": i for i, (p, _, _) in enumerate(plugin_groups)}
    pages = [(p.page, p.checkboxes()) for p in view.layout]
    groups = [i for i in view.items if i.kind == "checkbox" and i.options and not i.boxes]
    # Bigger groups first, as in verification: they are the least ambiguous.
    order = sorted([i for i in groups if i.id in hints], key=lambda i: rank[i.id]) + sorted([i for i in groups if i.id not in hints],
                                                          key=lambda i: -len(i.options))
    found: dict[str, list[tuple[int, object]]] = {}
    for item in order:
        names = [o for o, _ in item.options]
        for page_no, checks in sorted(pages, key=lambda p: p[0] != item.page):
            group = checks.locate(names, hints.get(item.id, item.label))
            if group:
                found[item.id] = [(page_no, group.options[n]) for n in names]
                break
    return found


def _score_checkbox(item: ViewItem, located, reader_flags: list[Flag], layout: list[PageLayout] | None) -> None:
    if located:
        item.boxes = [(page, (b.x, b.y, b.x + b.w, b.y + b.h)) for page, b in located]
        states = [checked for _, checked in item.options]
        agree = all(b.checked == s for (_, b), s in zip(located, states))
        decisive = all(b.decisive for _, b in located)
        if decisive and agree:
            item.confidence, item.reasons = 0.97, ["Confirmed: the ink in each box agrees"]
        elif not decisive:
            item.confidence, item.reasons = 0.6, ["Some boxes have an ambiguous amount of ink; check the scan"]
        else:
            item.confidence, item.reasons = 0.3, ["The ink in the boxes disagrees with the extracted ticks"]
    elif item.boxes:  # positions from the hybrid reader, which already verified the pixels
        item.confidence, item.reasons = 0.95, ["Confirmed against the checkbox ink when the form was read"]
    elif layout and (texts := _locate_option_texts(item, layout)):
        # No box or blank found for every option (circled options, faded boxes): point at the chosen option.
        item.boxes = texts
        item.confidence, item.reasons = 0.65, ["Not double-checked: the pixel check couldn't find a box or blank "
                                               "for every option (faded, circled or unusual marks). The highlighted "
                                               "option is the model's reading; compare with the scan"]
    else:
        item.confidence, item.reasons = 0.6, ["Not double-checked: the pixel check couldn't find this group's "
                                              "boxes on the scan, so compare the ticks with the scan"]


# --- text for the agent --------------------------------------------------------------------


def render_text(view: DocView) -> str:
    record = view.record
    plugin = get_plugin(record.form_type)
    u = record.universal
    out = [f"Document [{view.short_id}] (full id {record.doc_id})",
           f"Form type: {record.form_type}" + (f" — {plugin.title}" if plugin else ""),
           f"Title: {u.title or '-'} · pages: {u.page_count} · extracted by {record.run.reader}"
           f" ({record.run.model or '-'})"]
    if record.review:
        r = record.review
        out.append(f"Human review: {len(r.corrections)} correction(s), {len(r.confirmed)} value(s) confirmed")
        out += [f"  corrected {c.label}: '{_show(c.old)}' -> '{_show(c.new)}'" for c in r.corrections]
    out += ["", "FIELDS (verbatim; '(blank)' = left empty on the form; {printed} = pre-printed on the form;",
            "        {uncertain: ...} = the reading could not be confirmed; {reviewed} = checked by a person)"]

    section = object()
    for item in view.items:
        if item.kind != "field":
            continue
        if item.section != section:
            section = item.section
            out.append(f"  [{section or 'no section'}]")
        out.append(f"    {item.label.rstrip(':')}: {item.value or '(blank)'}{_tags(item)}")

    boxes = [i for i in view.items if i.kind == "checkbox"]
    if boxes:
        out += ["", "CHECKBOXES ([x] ticked, [ ] not ticked)"]
        for item in boxes:
            opts = "  ".join(f"[{'x' if c else ' '}] {o}" for o, c in item.options or [])
            out.append(f"  {(item.section + ' / ') if item.section else ''}{item.label}: {opts}{_tags(item)}")

    for t, (name, headers, rows) in enumerate(_tables(view), 1):
        out += ["", f"TABLE {t}: {name}", f"  columns: {' | '.join(headers)} (empty cells are left out)"]
        out += [f"  row {r}: " + " | ".join(f"{h}: {c.value}{_tags(c)}" for h, c in row if c.value)
                for r, row in rows if any(c.value for _, c in row)]

    texts = [i for i in view.items if i.kind == "text"]
    if texts:
        out += ["", "FREE TEXT"] + [f"  {i.value}{_tags(i)}" for i in texts]

    out += ["", "REVIEW FLAGS (automatic checks)"]
    out += [f"  [{f.severity}] {f.rule}: {f.message}" + (" [acknowledged by reviewer]" if view.acknowledged(f) else "")
            for f in record.flags] or ["  none"]
    return "\n".join(out)


def _show(value) -> str:
    if isinstance(value, list):
        return ", ".join(value) or "(none ticked)"
    return "(blank)" if value in (None, "") else str(value)


def _tags(item: ViewItem) -> str:
    tags = []
    if item.source == "printed":
        tags.append("{printed}")
    if item.level == "reviewed":
        tags.append("{reviewed}")
    elif item.level == "low":
        tags.append("{uncertain: " + item.reading[:100] + "}")
    return (" " + " ".join(tags)) if tags else ""


def _tables(view: DocView) -> list[tuple[str, list[str], list[tuple[int, list[tuple[str, ViewItem]]]]]]:
    """Per table: its name, its headers, and its rows as (header, cell) pairs."""
    tables: dict[int, tuple[str, dict[int, list[tuple[str, ViewItem]]]]] = {}
    for item in view.items:
        if item.kind == "cell" and item.cell:
            t, r = item.cell
            header = item.label.split("·")[-1].strip()
            tables.setdefault(t, (item.table or "Table", {}))[1].setdefault(r + 1, []).append((header, item))
    out = []
    for _, (name, rows) in sorted(tables.items()):
        headers = [h for h, _ in next(iter(rows.values()))] if rows else []
        out.append((name, headers, sorted(rows.items())))
    return out


# --- evidence for an answer ----------------------------------------------------------------


def evidence_for_answer(view: DocView, answer: str) -> list[ViewItem]:
    """Items of this form whose values appear in an answer, i.e. what to highlight on the scan."""
    text = answer.lower().replace("’", "'")
    loose = re.sub(r"[^a-z0-9]", "", text)
    digit_runs = set(re.findall(r"\d{5,}", re.sub(r"(?<=\d)[\s\-().]+(?=\d)", "", text)))
    found = []
    for item in view.items:
        if not item.value or not item.boxes:
            continue
        if item.kind == "checkbox":
            hit = any(c and (len(o) >= 5 or item.label.lower().rstrip(":") in text)
                      and re.search(rf"\b{re.escape(o.lower())}\b", text) for o, c in item.options or [])
        elif item.value_type == "date" and (d := parse_date(item.value)):
            hit = any(s.lower() in text for s in _date_spellings(d))
        else:
            v = re.sub(r"[^a-z0-9]", "", item.value.lower())
            digits = re.sub(r"\D", "", item.value)
            hit = (len(v) >= 4 and v in loose) or (len(digits) >= 5 and digits in digit_runs)
        if hit:
            found.append(item)
    return found


def _date_spellings(d) -> list[str]:
    month, short = d.strftime("%B"), d.strftime("%b")
    return [d.isoformat(), f"{d.month:02d}/{d.day:02d}/{d.year}", f"{d.month}/{d.day}/{d.year}",
            f"{month} {d.day}, {d.year}", f"{short} {d.day}, {d.year}", f"{d.day} {month} {d.year}",
            f"{month} {d.day}–", f"{month} {d.day}-"]
