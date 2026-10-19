"""Value and label normalization shared by storage, validation, and evaluation.

Every function takes the raw string as read off the form and returns a canonical
string (or None when there is nothing usable), so comparisons and SQL filters work
the same across readers and form types.
"""

from __future__ import annotations

import re
from datetime import date, datetime

_WS = re.compile(r"\s+")
_EMPTY = {"", "-", "--", "n/a", "na", "none", "null"}

VALUE_TYPES = [
    "text", "name", "date", "phone", "id", "npi", "code", "address",
    "number", "money", "email", "checkbox",
]


def clean(text: str | None) -> str | None:
    """Collapse whitespace; treat blanks and placeholder dashes as missing."""
    if text is None:
        return None
    text = _WS.sub(" ", str(text)).strip()
    if text.lower() in _EMPTY or re.fullmatch(r"[-_.\s]+", text):
        return None
    return text


_DATE_FORMATS = [
    "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y", "%Y-%m-%d", "%m.%d.%Y", "%m.%d.%y",
    "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y",
    "%d-%b-%y", "%d-%b-%Y", "%b %d %y", "%B %d, %y", "%b %d, %y",
    "%B %Y", "%b %Y",  # month and year only: the 1st of the month
]


def parse_date(text: str | None) -> date | None:
    text = clean(text)
    if not text:
        return None
    # OCR often inserts spaces around separators: "8/ 31/ 98", "12 /10 /98".
    text = re.sub(r"\s+\d{1,2}:\d{2}.*$", "", text)  # trailing time: "2/18/80 7:00 P.M."
    text = re.sub(r"\b([A-Za-z]{3,4})\.", r"\1", text)  # "Jan. 5, 1998", "Sept. 1998"
    text = re.sub(r"\bSept\b", "Sep", text, flags=re.I)
    text = re.sub(r"\b(1[89]|20) (\d{2})$", r"\1\2", text)  # split year: "October 16, 19 92"
    compact = re.sub(r"\s*([/\-])\s*", r"\1", text).rstrip(".")
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(compact, fmt).date()
        except ValueError:
            continue
        if "%y" in fmt and parsed.year > date.today().year:
            parsed = parsed.replace(year=parsed.year - 100)
        return parsed
    return None


def digits(text: str | None) -> str | None:
    text = clean(text)
    if not text:
        return None
    out = re.sub(r"\D", "", text)
    return out or None


def normalize_value(value: str | None, value_type: str = "text") -> str | None:
    """Canonical form used for comparison and querying (dates -> ISO, phones -> digits)."""
    value = clean(value)
    if value is None:
        return None
    if value_type == "date":
        parsed = parse_date(value)
        return parsed.isoformat() if parsed else value
    if value_type in ("phone", "npi"):
        return digits(value) or value
    if value_type in ("id", "code"):
        return re.sub(r"\s", "", value).upper()
    if value_type in ("number", "money"):
        num = re.sub(r"[^\d.\-]", "", value)
        return num or value
    if value_type == "email":
        return value.lower()
    if value_type == "name":
        return value.title()
    return value


# --- labels -> keys ---------------------------------------------------------------

# Synonyms that should land on the same key across different form types.
_KEY_SYNONYMS = {
    "dob": "date_of_birth",
    "birth_date": "date_of_birth",
    "date_of_birth": "date_of_birth",
    "npi": "npi",
    "npi_number": "npi",
    "phone_number": "phone",
    "telephone": "phone",
    "telephone_number": "phone",
    "tel": "phone",
    "fax_number": "fax",
    "fax_no": "fax",
    "member_or_medicaid_id": "member_id",
    "member_id_number": "member_id",
    "e_mail": "email",
    "email_address": "email",
    "zip_code": "zip",
    "postal_code": "zip",
}


def label_to_key(label: str | None) -> str | None:
    """'Member or Medicaid ID #:' -> 'member_id'; 'DATE WANTED' -> 'date_wanted'."""
    label = clean(label)
    if not label:
        return None
    key = label.lower().replace("#", " number ").replace("&", " and ")
    key = re.sub(r"\(.*?\)", " ", key)  # drop parentheticals like "(if different)"
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    key = re.sub(r"_number$", "", key) if key not in ("number",) else key
    return _KEY_SYNONYMS.get(key, key) or None
