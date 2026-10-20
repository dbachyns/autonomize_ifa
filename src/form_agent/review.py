"""Human review: correct values, confirm values, acknowledge flags.

A review never edits the original extraction. It saves a new record under the reader name
`reviewed` (`data/cache/<doc_id>/reviewed.json`), which the index prefers over every
machine reading. The record keeps a log of every correction (old -> new, who, when).

After an edit the validation rules run again on the new values. Reading flags about an
edited or confirmed value (OCR disagreement, ambiguous checkbox) are dropped, because a
person has now looked at it.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from form_agent.forms import get_plugin
from form_agent.normalize import clean
from form_agent.schema import Correction, FormRecord, Review, RunInfo, UniversalForm
from form_agent.validate import READER_RULES, validate
from form_agent.view import build_view

REVIEWED = "reviewed"


def apply_review(record: FormRecord, edits: dict[str, Any] | None = None, confirm: list[str] | None = None,
                 acknowledge: list[str] | None = None, reviewer: str | None = None) -> FormRecord:
    """Return the reviewed version of `record`.

    edits: item id -> new value (a string, None for blank, or a list of ticked options for a checkbox group).
    confirm: item ids a person checked and found correct. acknowledge: flag ids (schema.flag_id) a person has seen.
    """
    edits, confirm, acknowledge = edits or {}, confirm or [], acknowledge or []
    view = build_view(record)
    items = {i.id: i for i in view.items}
    data = record.model_dump(mode="json")
    review = record.review.model_copy(deep=True) if record.review else Review(based_on=record.run.reader)

    touched: set[str] = set()
    for item_id, new in edits.items():
        item = items.get(item_id)
        if item is None or not item.editable:
            raise ValueError(f"{item_id!r} is not an editable item of {record.doc_id}")
        new = _coerce(new)
        old = _write(data, item_id, new)
        if _same(old, new):
            continue
        previous = next((c for c in review.corrections if c.item == item_id), None)
        if previous:  # keep the original machine value, update the latest
            previous.new, previous.at, previous.reviewer = new, datetime.now(timezone.utc), reviewer
            if _same(previous.old, new):
                review.corrections.remove(previous)
        else:
            review.corrections.append(Correction(item=item_id, label=item.label, old=old, new=new, reviewer=reviewer))
        touched |= item.keys
    for item_id in confirm:
        if item_id not in items:
            raise ValueError(f"{item_id!r} is not an item of {record.doc_id}")
        if item_id not in review.confirmed:
            review.confirmed.append(item_id)
        touched |= items[item_id].keys
    review.acknowledged += [a for a in acknowledge if a not in review.acknowledged]
    review.updated_at = datetime.now(timezone.utc)

    new_record = FormRecord.model_validate(data)
    if edits and (plugin := get_plugin(new_record.form_type)) and new_record.typed:
        new_record.universal = _resync_universal(new_record, plugin)
    touched = {_norm(k) for k in touched if k}
    kept = [f for f in record.flags if f.rule in READER_RULES and not {_norm(n) for n in f.fields} & touched]
    new_record.flags = [*kept, *validate(new_record)]
    new_record.review = review
    new_record.run = RunInfo(reader=REVIEWED, model=f"{record.run.model or record.run.reader} + human review",
                             prompt_version=record.run.prompt_version)
    return new_record


def _resync_universal(record: FormRecord, plugin) -> UniversalForm:
    """After a typed edit, rebuild the generic view from the typed values so both agree in the index."""
    tv = plugin.to_universal(plugin.parse(record.typed))
    u = record.universal
    printed = [f for f in u.fields if f.source == "printed" and f.value]
    return UniversalForm(title=u.title, page_count=u.page_count, fields=[*printed, *tv.fields],
                         checkboxes=tv.checkboxes, tables=tv.tables, text_blocks=u.text_blocks)


# --- reading and writing values by item id -------------------------------------------------


def _write(data: dict, item_id: str, new: Any) -> Any:
    """Set the value an item id points at; returns the old value."""
    root, path = item_id.split(":", 1)
    node: Any = data[root]
    parts = path.split(".")
    if root == "universal" and parts[0] == "checkboxes":  # a whole group: new = list of ticked option names
        group = data["universal"]["checkboxes"][int(parts[1])]
        old = [o["label"] for o in group["options"] if o["checked"]]
        ticked = set(new or [])
        for o in group["options"]:
            o["checked"] = o["label"] in ticked
        return old
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node[part]
    last: Any = int(parts[-1]) if isinstance(node, list) else parts[-1]
    if root == "universal" and parts[0] in ("fields", "text_blocks"):  # fields.3 -> fields[3]["value"]
        node, last = node[last], "value" if parts[0] == "fields" else "text"
    old = node[last]
    if isinstance(old, bool):  # single-box group stored as a flag (e.g. "DME requested")
        new = bool(new)
    elif isinstance(old, list) and not isinstance(new, list):
        new = [] if new is None else [new]
    node[last] = new
    return old


def _coerce(value: Any) -> Any:
    if isinstance(value, str):
        return clean(value)
    if isinstance(value, (list, tuple)):
        return [v for v in value if v]
    return value


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return sorted(a) == sorted(b)
    return (a or None) == (b or None)


def _norm(name: str) -> str:
    return re.sub(r"\[(\d+)\]", r".\1", name)
