import pytest

from form_agent.normalize import clean, label_to_key, normalize_value, parse_date


@pytest.mark.parametrize(
    "raw, iso",
    [
        ("04/02/1977", "1977-04-02"),
        ("8/ 31/ 98", "1998-08-31"),
        ("12 /10 /98", "1998-12-10"),
        ("April 19, 1988", "1988-04-19"),
        ("2023-10-14", "2023-10-14"),
    ],
)
def test_dates(raw, iso):
    assert parse_date(raw).isoformat() == iso


def test_unparseable_date_kept_verbatim():
    assert normalize_value("B-17-88", "date") == "B-17-88"


@pytest.mark.parametrize("raw", ["", "  ", "-------", "--", "N/A", None])
def test_placeholders_are_empty(raw):
    assert clean(raw) is None


def test_values():
    assert normalize_value("(519)-490-4052", "phone") == "5194904052"
    assert normalize_value(" s66.326d ", "code") == "S66.326D"
    assert normalize_value("$135 00", "money") == "13500"
    assert normalize_value("richard  MCKEE", "name") == "Richard Mckee"


@pytest.mark.parametrize(
    "label, key",
    [
        ("Member or Medicaid ID #:", "member_id"),
        ("DOB:", "date_of_birth"),
        ("Date of Birth", "date_of_birth"),
        ("NPI #:", "npi"),
        ("Subscriber Name (if different):", "subscriber_name"),
        ("DATE WANTED", "date_wanted"),
        ("FAX NO.", "fax"),
        ("Phone #:", "phone"),
    ],
)
def test_label_to_key(label, key):
    assert label_to_key(label) == key


@pytest.mark.parametrize("text, iso", [
    ("2-Dec-97", "1997-12-02"),
    ("DEC 14 89", "1989-12-14"),
    ("JUNE 1995", "1995-06-01"),
    ("2/18/80 7:00 P.M.", "1980-02-18"),
    ("October 16, 19 92", "1992-10-16"),
    ("Jan. 5, 1998", "1998-01-05"),
    ("12.10.98", "1998-12-10"),
])
def test_parse_date_scanned_formats(text, iso):
    assert parse_date(text).isoformat() == iso
