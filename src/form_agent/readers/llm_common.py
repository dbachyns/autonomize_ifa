"""Pieces shared by every vision-LLM reader: output schema, prompts, image encoding, pricing.

The output models here are "strict" (every field required, no defaults) so they can be
used as a structured-output schema by any provider. They are converted to the richer
`UniversalForm` afterwards.
"""

from __future__ import annotations

import base64
import io
from typing import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict

from form_agent.normalize import VALUE_TYPES, clean, label_to_key
from form_agent.schema import CheckboxGroup, CheckboxOption, FormField, Table, TextBlock, UniversalForm

PROMPT_VERSION = "v4"

ValueType = Literal[tuple(VALUE_TYPES)]  # type: ignore[valid-type]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMField(_Strict):
    page: int
    section: str | None
    label: str
    value: str | None
    value_type: ValueType
    source: Literal["filled", "printed"]


class LLMOption(_Strict):
    label: str
    checked: bool


class LLMCheckboxGroup(_Strict):
    page: int
    section: str | None
    label: str
    options: list[LLMOption]


class LLMTable(_Strict):
    page: int
    section: str | None
    title: str | None
    headers: list[str]
    column_types: list[ValueType]
    rows: list[list[str | None]]


class LLMTextBlock(_Strict):
    page: int
    section: str | None
    label: str | None
    text: str


class LLMUniversal(_Strict):
    title: str | None
    fields: list[LLMField]
    checkboxes: list[LLMCheckboxGroup]
    tables: list[LLMTable]
    text_blocks: list[LLMTextBlock]


UNIVERSAL_PROMPT = f"""You are a meticulous document-transcription system. Extract EVERYTHING filled in on this form
into the JSON schema. The pages are given in order; set `page` to the page number (1-based).

Rules:
- Copy values VERBATIM, exactly as written: same spelling (even typos), punctuation, and digits.
  Never correct, complete, or guess. If text is cut off ("Laceration of"), keep it cut off.
- Distinguish look-alike characters carefully (O vs 0, I vs 1, S vs 5, B vs 8), especially in codes and IDs.
- Every labeled blank on the form becomes a field, even if empty: use value null for empty, blank, dashes
  ("-----"), redacted, or unreadable values. Do not invent values for empty fields.
- Signatures: if a signature (or initials) is written on a signature line, the value is "[signed]"; if a
  name is also printed or written legibly there, use "[signed] <name>". Use null only when the line is empty.
- Handwriting: read handwritten characters one by one, especially digits in dates and numbers; do not copy a
  value from a nearby field that looks similar.
- `label` is the printed label as it appears (e.g. "Member or Medicaid ID #:"); `section` is the printed
  section heading the field sits under, or null.
- `source`: "filled" for anything entered on the form (typed into blanks, handwritten, stamped);
  "printed" for values that are part of the form template itself.
- Also capture PRE-PRINTED values that answer a label, with source "printed": prices and fees, phone/fax
  numbers, mailing or submission addresses, deadlines and dates, contact names, form/version numbers.
  Make one field per printed value (never lump several labeled numbers into one field), labeled with the
  text printed next to it. Do not capture pre-printed instructions, legal text, or headings as fields.
  A blank meant to be filled in is always source "filled", even when it is empty.
- `value_type` is one of: {", ".join(VALUE_TYPES)}. Use "npi" for National Provider Identifiers, "code" for
  CPT/HCPCS/ICD/other codes, "id" for member/account/reference numbers, "name" for person names.
- Checkbox groups: list EVERY option printed in the group, with checked=true only if the box is clearly ticked
  (check mark, X, or fill). Stray dots or smudges are not ticks. A lone checkbox is a group with one option.
- Tables: one entry per table with its printed headers; include only rows that contain data (skip empty rows).
- text_blocks: free-form content entered on the form that is not a labeled field (notes, descriptions,
  addresses written in open areas). Do not transcribe pre-printed instructions or boilerplate.
- `title`: the form's printed title."""


def to_universal(out: LLMUniversal, page_count: int) -> UniversalForm:
    return UniversalForm(
        title=clean(out.title),
        page_count=page_count,
        fields=[
            FormField(label=f.label, key=label_to_key(f.label), value=clean(f.value), value_type=f.value_type,
                      source=f.source if clean(f.value) else "filled", section=f.section, page=f.page)
            for f in out.fields
        ],
        checkboxes=[
            CheckboxGroup(label=g.label, key=label_to_key(g.label), section=g.section, page=g.page,
                          options=[CheckboxOption(label=o.label, checked=o.checked, method="llm") for o in g.options])
            for g in out.checkboxes
        ],
        tables=[
            Table(title=t.title, section=t.section, page=t.page, headers=t.headers,
                  column_types=t.column_types, rows=[[clean(c) for c in row] for row in t.rows]).tidy()
            for t in out.tables
        ],
        text_blocks=[TextBlock(text=b.text, label=b.label, section=b.section, page=b.page) for b in out.text_blocks],
    )


def typed_prompt(title: str, hints: str) -> str:
    return f"""This is a "{title}". Fill the JSON schema from the form.

Rules:
- Copy values VERBATIM, exactly as written (same spelling, punctuation, digits, date format). Never correct,
  complete, or guess; keep cut-off text cut off. Distinguish O/0, I/1, S/5, B/8 carefully in codes and IDs.
- Use null for empty, blank, dashed ("-----"), redacted, or unreadable fields. Never invent values.
- Checkbox lists contain ONLY the options that are clearly ticked, using the exact option names below.
  Stray dots or smudges are not ticks. Use an empty list if nothing is ticked.
- Include only service lines/rows that contain data.
{hints}"""


def encode_image(image: Image.Image, max_side: int = 2048, quality: int = 90) -> str:
    """JPEG data URL, downscaled so the long side is at most `max_side`."""
    im = image.convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# USD per 1M tokens (input, output). Source: provider pricing pages, checked 2026-10-19.
PRICES = {
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6-sol": (2.00, 10.00),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-astra": (10.00, 50.00),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o": (2.50, 10.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if model not in PRICES:
        return None
    pin, pout = PRICES[model]
    return (input_tokens * pin + output_tokens * pout) / 1_000_000
