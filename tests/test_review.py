from pathlib import Path

import pytest
from PIL import Image

from form_agent.forms import get_plugin
from form_agent.layout import load_layout
from form_agent.readers.base import Reader, register
from form_agent.review import apply_review
from form_agent.schema import CheckboxGroup, CheckboxOption, FormField, FormRecord, RunInfo, UniversalForm, flag_id, flatten
from form_agent.store import build_index, connect_readonly, load_record, save_record
from form_agent.view import build_view, evidence_for_answer, render_text

CACHE = Path(__file__).parent.parent / "data" / "cache"
DOC = "0a18d138-5f03-46dd-9cb8-d10b486a10fe_TX_page_1"  # scan says "laparascopy"; the model wrote "laparoscopy"


@pytest.fixture(scope="module")
def record():
    return load_record(CACHE / DOC / "hybrid.json")


@pytest.fixture(scope="module")
def view(record):
    return build_view(record, load_layout(CACHE, DOC), "0a18d138")


def _universal_record():
    return FormRecord(
        doc_id="fax", source_file="x.png", run=RunInfo(reader="openai"),
        universal=UniversalForm(
            fields=[FormField(label="TO:", value="Jack Reily", value_type="name"),
                    FormField(label="DATE:", value="2-Dec-97", value_type="date")],
            checkboxes=[CheckboxGroup(label="Urgent?", options=[CheckboxOption(label="Yes", checked=True),
                                                                 CheckboxOption(label="No", checked=True)])],
        ),
    )


# --- plugin path mapping ------------------------------------------------------------------


def test_texas_plugin_maps_every_typed_value():
    plugin = get_plugin("tx_prior_auth")
    typed = plugin.parse({})
    keys = {f.key for f in plugin.to_universal(typed).fields} | {g.key for g in plugin.to_universal(typed).checkboxes}
    assert keys <= set(plugin.field_paths())
    paths = {p for p in flatten(typed) if not p.startswith("services.lines")}
    assert paths == set(plugin.field_paths().values())


# --- view: positions and confidence ---------------------------------------------------------


def test_view_confidence_levels(view):
    item = view.item("typed:services.lines.1.service")
    assert item.level == "low" and "laparascopy" in " ".join(item.reasons)
    npi = view.item("typed:requesting_provider.npi")
    assert npi.level == "high" and npi.boxes  # read correctly ...
    assert [f.rule for f in npi.flags] == ["npi_checksum"]  # ... but wrong on the form: a flag, not low confidence
    assert view.item("typed:general.clinical_reason_for_urgency").level == "blank"
    review_type = view.item("typed:general.review_type")
    assert review_type.level == "high" and len(review_type.boxes) == 2  # one box per option


def test_text_ocr_cannot_find_is_medium_not_low():
    # "Apt. Texas 83462" is clipped at the top edge of the scan: the vision model read it, OCR missed it.
    record = load_record(CACHE / "00c33de5-461e-4642-966b-73ee20ef0d27_TX_page_1" / "hybrid.json")
    v = build_view(record, load_layout(CACHE, record.doc_id))
    clipped = next(i for i in v.items if i.kind == "text" and i.value == "Apt. Texas 83462")
    assert clipped.level == "medium" and not clipped.boxes


def test_view_without_layout_is_unknown_not_high(record):
    v = build_view(record)
    assert v.item("typed:patient.name").level == "unknown"
    assert v.item("typed:general.clinical_reason_for_urgency").level == "blank"


def test_render_text_marks_uncertain_values(view):
    text = render_text(view)
    assert "Surgical laparoscopy {uncertain:" in text
    assert "Donna Lee" in text


def test_evidence_for_answer(view):
    answer = "The requesting NPI 8041191212 fails the check digit; the form is dated April 15, 2022, review type Urgent."
    ids = {i.id for i in evidence_for_answer(view, answer)}
    assert {"typed:requesting_provider.npi", "typed:submission.date", "typed:general.review_type"} <= ids
    assert "typed:service_provider.npi" not in ids


# --- review --------------------------------------------------------------------------------


def test_review_edit_confirm_acknowledge(record):
    ack = flag_id(next(f for f in record.flags if f.rule == "npi_checksum"))
    reviewed = apply_review(
        record,
        edits={"typed:services.lines.1.service": "Surgical laparascopy", "typed:general.review_type": ["Non-Urgent"]},
        confirm=["typed:patient.name"], acknowledge=[ack], reviewer="tester",
    )
    assert reviewed.run.reader == "reviewed"
    assert reviewed.typed["services"]["lines"][1]["service"] == "Surgical laparascopy"
    assert {c.item for c in reviewed.review.corrections} == {"typed:services.lines.1.service", "typed:general.review_type"}
    rules = {f.rule for f in reviewed.flags}
    assert "ocr_disagrees" not in rules  # a person fixed that value
    assert "urgent_without_reason" not in rules  # re-validated: no longer urgent
    assert "npi_checksum" in rules  # content problems stay, marked as seen
    assert ack in reviewed.review.acknowledged
    # The generic view follows the typed edit, so cross-form SQL sees it too.
    assert next(g for g in reviewed.universal.checkboxes if g.label == "Review Type").selected == ["Non-Urgent"]
    v = build_view(reviewed)
    assert v.item("typed:patient.name").level == "reviewed"
    assert v.acknowledged(next(f for f in reviewed.flags if flag_id(f) == ack))
    assert record.typed["services"]["lines"][1]["service"] == "Surgical laparoscopy"  # original untouched


def test_review_revert_drops_correction(record):
    once = apply_review(record, edits={"typed:patient.member_id": "35947"})
    back = apply_review(once, edits={"typed:patient.member_id": "35946"})
    assert once.review.corrections and not back.review.corrections


def test_review_universal_record():
    rec = _universal_record()
    reviewed = apply_review(rec, edits={"universal:fields.0": "Jack Reilly", "universal:checkboxes.0": ["Yes"]})
    assert reviewed.universal.fields[0].value == "Jack Reilly"
    assert reviewed.universal.checkboxes[0].selected == ["Yes"]
    assert "multiple_ticks" not in {f.rule for f in reviewed.flags}
    with pytest.raises(ValueError):
        apply_review(rec, edits={"universal:fields.99": "x"})


def test_reviewed_record_wins_in_index(tmp_path, record):
    cache = tmp_path / "cache"
    save_record(cache, record)
    save_record(cache, apply_review(record, edits={"typed:patient.name": "Donna Leigh"}))
    build_index(cache, tmp_path / "forms.db")
    con = connect_readonly(tmp_path / "forms.db")
    assert con.execute("SELECT reader FROM extractions WHERE active = 1").fetchone()[0] == "reviewed"
    assert con.execute("SELECT patient_name FROM tx_prior_auth").fetchone()[0] == "Donna Leigh"


# --- upload pipeline ------------------------------------------------------------------------


@register("fake")
class FakeReader(Reader):
    def read(self, image_path):
        rec = _universal_record()
        return rec.model_copy(update={"doc_id": image_path.stem, "source_file": str(image_path),
                                      "run": RunInfo(reader="fake")})


def test_process_file(tmp_path):
    from form_agent.pipeline import process_file

    scan = tmp_path / "my scan.png"
    Image.new("RGB", (400, 300), "white").save(scan)
    steps = []
    kwargs = dict(reader="fake", cache_dir=tmp_path / "cache", db_path=tmp_path / "forms.db",
                  uploads_dir=tmp_path / "uploads", on_step=steps.append)
    first = process_file(scan, **kwargs)
    second = process_file(scan, **kwargs)  # same name again: gets a new id instead of overwriting
    assert first.doc_id == "my_scan" and second.doc_id == "my_scan-2"
    assert (tmp_path / "cache" / "my_scan" / "_layout.json").exists()
    assert {f.rule for f in first.flags} == {"multiple_ticks"}  # validation ran
    con = connect_readonly(tmp_path / "forms.db")
    assert con.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
    assert steps
    with pytest.raises(ValueError):
        process_file(tmp_path / "notes.txt", **kwargs)


def test_one_ocr_region_confirms_only_one_value():
    # Two "DATE:" fields; the scan shows 03/01/90 (typed) and a handwritten 03/02/90 OCR can't read.
    # The reader wrote 03/01/90 for both: only the first may be confirmed by the single "03/01/90" on the page.
    from form_agent.layout import PageLayout
    from form_agent.ocr import OCRLine

    lines = [OCRLine("DATE:", 500, 100, 560, 120, 0.9), OCRLine("03/01/90", 600, 100, 680, 120, 0.9),
             OCRLine("DATE:", 100, 400, 160, 420, 0.9), OCRLine("SHLPPED-VIA:", 100, 200, 200, 220, 0.9),
             OCRLine("Direct", 220, 200, 280, 220, 0.9)]
    layout = [PageLayout(1, (800, 1000), lines, [], [])]
    rec = FormRecord(doc_id="x", source_file="x.png", run=RunInfo(reader="openai"), universal=UniversalForm(fields=[
        FormField(label="DATE:", value="03/01/90", value_type="date"),
        FormField(label="DATE:", value="03/01/90", value_type="date"),
        FormField(label="SHIPPED VIA:", value="Direct Express")]))
    first, second, shipped = build_view(rec, layout).items
    assert first.level == "high" and first.boxes
    assert second.level == "medium" and not second.boxes
    assert shipped.level == "medium" and shipped.boxes and "only reads 'Direct'" in shipped.reading


FUNSD = Path(__file__).parent.parent / "data" / "cache_funsd"
STYLE = "82253245_3247"  # two untitled side-by-side tables; "JUNE 30 [ X ]" ticked in a wide box


@pytest.fixture(scope="module")
def style_view():
    return build_view(load_record(FUNSD / STYLE / "openai.json"), load_layout(FUNSD, STYLE))


def test_untitled_tables_stay_separate_and_named(style_view):
    text = render_text(style_view)
    within, outside = text.split("TABLE 1: ")[1], text.split("TABLE 2: ")[1]
    assert within.startswith("DIRECT ACCOUNTS AND CHAINS HEADQUARTERED WITHIN")
    assert outside.startswith("DIRECT ACCOUNTS AND CHAINS HEADQUARTERED OUTSIDE")
    assert "Kroger" not in within.split("TABLE 2")[0]  # rows of the two tables are not merged


def test_short_table_cells_confirmed_on_their_row(style_view):
    kroger_stores = next(i for i in style_view.items if i.cell == (1, 0) and i.value == "21")
    assert kroger_stores.level == "high" and kroger_stores.boxes


def test_wide_checkbox_with_label_on_the_left(style_view):
    group = next(i for i in style_view.items if i.kind == "checkbox")
    assert group.level == "high" and "ink" in group.reading


def _funsd_view(doc):
    return build_view(load_record(FUNSD / doc / "openai.json"), load_layout(FUNSD, doc))


def test_marks_on_blank_lines_are_checked():
    # "_X_ Change", "Personal Service _X_", "STATUTORY ___X___": an X written on a blank line, no printed box
    for doc, label in [("87528321", "Type of Change"), ("92380595", "via"), ("92380595", "Service type")]:
        groups = [i for i in _funsd_view(doc).items if i.kind == "checkbox" and i.label == label]
        assert groups and all(g.level == "high" for g in groups), (doc, label)


def test_small_boxes_and_rows():
    # 15px boxes whose ticks hide the interior; "12 MOS." must not take the "12 MOS. WHSE." box of the next row
    levels = {i.label: i.level for i in _funsd_view("87528380").items if i.kind == "checkbox"}
    assert levels["CORNERS:"] == levels["BILLING:"] == levels["WAREHOUSING:"] == "high"


def test_paragraphs_do_not_swallow_other_sections():
    # "at room temperature" appears in four sections; the first long value must not claim them all
    v = _funsd_view("87428306")
    state = next(i for i in v.items if i.label == "PHYSICAL STATE")
    (_, (_, y0, _, y1)), = state.boxes
    assert y1 - y0 < 60
    assert all(next(i for i in v.items if i.label == lab).level == "high"
               for lab in ["SPECIFIC GRAVITY", "MUTAGENICITY SOLVENTS", "PARENTERAL", "ORAL"])


def test_contact_block_confirmed_by_its_numbers():
    # The model wrote the printed fax block as one value and dropped a word ("UM Outpatient" -> "Outpatient"):
    # the numbers are all on the scan, so it is confirmed rather than "reads differently".
    doc = "0adf81aa-71ff-4a88-be58-35c9feb76e05_TX_page_1"
    v = build_view(load_record(CACHE / doc / "hybrid.json"), load_layout(CACHE, doc))
    fax = next(i for i in v.items if i.label == "Fax:" and i.value and "833-994-1960" in i.value)
    assert fax.level == "high" and "numbers" in fax.reading


def test_reader_not_found_flag_does_not_make_a_value_red():
    # The hybrid reader flags values its OCR pass couldn't find; that means "unconfirmed" (medium), not "wrong".
    from form_agent.layout import PageLayout
    from form_agent.ocr import OCRLine
    from form_agent.schema import Flag

    rec = FormRecord(doc_id="x", source_file="x.png", run=RunInfo(reader="hybrid"), universal=UniversalForm(
        fields=[FormField(label="Date Received:", key="date_received", value="12/14/89", value_type="date")]),
        flags=[Flag(rule="ocr_not_found", severity="info", message="not found", fields=["date_received"])])
    layout = [PageLayout(1, (800, 1000), [OCRLine("Date Received:", 50, 900, 200, 920, 0.9)], [], [])]
    item, = build_view(rec, layout).items
    assert item.level == "medium"
