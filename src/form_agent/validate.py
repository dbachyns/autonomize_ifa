"""Validation: turn extracted records into review flags.

Generic rules key off value types (date, phone, npi, code, ...) so they apply to any
form. Known form types add their own rules through their plugin. Rules never change
values; they only report.
"""

from __future__ import annotations

import re
from datetime import date

from form_agent.forms import get_plugin
from form_agent.normalize import clean, digits, parse_date
from form_agent.schema import Flag, FormRecord, flatten

ICD10 = re.compile(r"^[A-TV-Z][0-9][0-9A-Z](\.[0-9A-Z]{1,4})?$")
CPT = re.compile(r"^\d{4}[0-9FTU]$")  # Category I (5 digits), II (####F), III (####T)
HCPCS = re.compile(r"^[A-V]\d{4}$")


def npi_is_valid(npi: str) -> bool:
    """10 digits with a Luhn check digit computed over the '80840' prefix + first 9 digits."""
    if not re.fullmatch(r"\d{10}", npi):
        return False
    total = 0
    for i, ch in enumerate(reversed("80840" + npi[:9])):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            d = d - 9 if d > 9 else d
        total += d
    return (10 - total % 10) % 10 == int(npi[9])


def check_value(name: str, value: str | None, value_kind: str, label: str = "", today: date | None = None) -> list[Flag]:
    """Format/plausibility checks for one value, chosen by its type."""
    today = today or date.today()
    value = clean(value)
    if value is None:
        return []
    flags: list[Flag] = []
    if value_kind == "npi":
        d = digits(value) or ""
        if len(d) != 10:
            flags.append(Flag(rule="npi_length", severity="error", fields=[name],
                              message=f"NPI '{value}' has {len(d)} digits; NPIs have 10."))
        elif not npi_is_valid(d):
            flags.append(Flag(rule="npi_checksum", severity="error", fields=[name],
                              message=f"NPI '{value}' fails the NPI check-digit test."))
    elif value_kind == "date":
        parsed = parse_date(value)
        if parsed is None:
            flags.append(Flag(rule="date_unreadable", severity="warning", fields=[name],
                              message=f"'{value}' is not a valid date."))
        elif _is_birth_date(name, label) and (parsed > today or today.year - parsed.year > 120):
            flags.append(Flag(rule="dob_implausible", severity="error", fields=[name],
                              message=f"Date of birth {parsed.isoformat()} is in the future or >120 years ago."))
        elif parsed.year < 1900 or (parsed - today).days > 5 * 365:
            flags.append(Flag(rule="date_implausible", severity="warning", fields=[name],
                              message=f"Date {parsed.isoformat()} is implausible."))
    elif value_kind == "phone":
        d = digits(value) or ""
        # 7 digits is a local number; more than 11 is usually several numbers in one field.
        if len(d) < 7 or len(d) in (8, 9) or (len(d) == 11 and d[0] != "1"):
            flags.append(Flag(rule="phone_length", severity="warning", fields=[name],
                              message=f"Phone/fax '{value}' has {len(d)} digits; US numbers have 10."))
    elif value_kind == "code":
        code = re.sub(r"\s", "", value).upper()
        context = f"{name} {label}".lower()
        if "icd" in context or "diagnos" in context:
            if not ICD10.match(code):
                flags.append(Flag(rule="icd10_format", severity="warning", fields=[name],
                                  message=f"'{value}' is not a valid ICD-10 code format."))
        elif "cpt" in context or "hcpcs" in context or "procedure" in context or context.split(".")[-1].strip() == "code":
            if not (CPT.match(code) or HCPCS.match(code)):
                flags.append(Flag(rule="procedure_code_format", severity="warning", fields=[name],
                                  message=f"'{value}' is not a valid CPT/HCPCS code format."))
    return flags


def _is_birth_date(name: str, label: str) -> bool:
    text = f"{name} {label}".lower()
    return "dob" in text or "birth" in text


# Flags produced while reading (pixel/OCR verification); everything else comes from `validate`.
READER_RULES = {"checkbox_corrected", "checkbox_uncertain", "ocr_disagrees", "ocr_not_found"}


def revalidate(record: FormRecord, today: date | None = None) -> FormRecord:
    """Re-run the validation rules on a cached record, keeping the reader's own flags."""
    kept = [f for f in record.flags if f.rule in READER_RULES]
    return record.model_copy(update={"flags": [*kept, *validate(record, today)]})


def validate(record: FormRecord, today: date | None = None) -> list[Flag]:
    """All flags for a record (typed rules if a plugin applies, universal rules otherwise)."""
    flags: list[Flag] = []
    plugin = get_plugin(record.form_type)
    u = record.universal

    if not any(f.value for f in u.fields) and not any(t.rows for t in u.tables) and not u.text_blocks:
        flags.append(Flag(rule="empty_extraction", severity="error", message="No values were extracted from this form."))

    if plugin and record.typed:
        typed = plugin.parse(record.typed)
        for path, (value, value_kind) in flatten(typed).items():
            if isinstance(value, str):
                flags += check_value(path, value, value_kind, today=today)
        for path in plugin.single_choice_groups():
            ticked = _get(typed, path)
            if len(ticked) > 1:
                flags.append(Flag(rule="multiple_ticks", severity="error", fields=[path],
                                  message=f"{path}: {len(ticked)} options ticked ({', '.join(ticked)}) in a pick-one group."))
        flags += plugin.validate(typed, today=today)
    else:
        for f in u.fields:
            if f.source == "printed":
                continue
            flags += check_value(f.key or f.label, f.value, f.value_type, f.label, today=today)
        for t in u.tables:
            for r, row in enumerate(t.rows):
                for c, cell in enumerate(row):
                    kind = t.column_types[c] if c < len(t.column_types) else "text"
                    header = t.headers[c] if c < len(t.headers) else ""
                    flags += check_value(f"{t.title or 'table'}[{r}].{header}", cell, kind, header, today=today)
        for g in u.checkboxes:
            names = {o.label.strip().lower() for o in g.options}
            if names == {"yes", "no"} and len(g.selected) > 1:
                flags.append(Flag(rule="multiple_ticks", severity="error", fields=[g.label],
                                  message=f"'{g.label}': both Yes and No are ticked."))
    return _dedupe(flags)


def _get(model, path: str):
    for part in path.split("."):
        model = getattr(model, part)
    return model


def _dedupe(flags: list[Flag]) -> list[Flag]:
    seen, out = set(), []
    for f in flags:
        key = (f.rule, tuple(f.fields), f.message)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out
