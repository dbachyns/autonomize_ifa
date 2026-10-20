"""Checkbox detection and OCR verification on real scans (local only, no API)."""

import json
from pathlib import Path

import pytest
from PIL import Image

from form_agent.checkboxes import PageCheckboxes, _label_score
from form_agent.forms import get_plugin
from form_agent.ocr import ocr_page
from form_agent.schema import FormRecord
from form_agent.verify import verify

DATA = Path(__file__).parent.parent / "data"
DOC = "0a01f77b-85f9-48c9-89bf-4b095bebb438_TX_page_1"  # sex = Unknown, stray mark in the Female box


@pytest.fixture(scope="module")
def page():
    return Image.open(DATA / "forms" / f"{DOC}.png")


@pytest.fixture(scope="module")
def gt():
    return FormRecord.model_validate(json.loads((DATA / "ground_truth" / f"{DOC}.json").read_text()))


def test_label_score_handles_ocr_noise():
    assert _label_score("Non-Urgent", "nonurgent") == 1.0
    assert _label_score("Occupational Therapy", "qccupationaltherapy") > 0.9
    assert _label_score("No", "on") > 0  # "No)" read as "(oN"
    assert _label_score("Male", "female") == 0.95  # contained, but not at the start
    assert _label_score("Urgent", "speechtherapy") == 0


def test_detects_every_checkbox_and_reads_ticks(page, gt):
    boxes = PageCheckboxes(page, ocr_page(page))
    assert len([b for b in boxes.boxes if b.kind == "box"]) == 30  # plus blank lines, which aren't boxes
    for group in sorted(gt.universal.checkboxes, key=lambda g: -len(g.options)):
        found = boxes.locate([o.label for o in group.options], group.label)
        assert found, group.label
        assert found.states() == {o.label: o.checked for o in group.options}, group.label


def test_stray_mark_is_not_a_tick(page):
    boxes = PageCheckboxes(page, ocr_page(page))
    female = boxes.locate(["Male", "Female", "Other", "Unknown"]).options["Female"]
    assert 0 < female.fill < 0.06 and not female.checked


def test_verify_corrects_llm_checkbox_and_flags_text_mismatch(page, gt):
    llm = gt.model_copy(deep=True)
    llm.typed["patient"]["sex"] = ["Female", "Unknown"]  # what gpt-6-luna actually returned
    llm.typed["patient"]["name"] = "Daniel Jarvls"  # a misread name
    plugin = get_plugin("tx_prior_auth")
    llm.universal = plugin.to_universal(plugin.parse(llm.typed))

    fixed, flags = verify(llm, [page])
    assert fixed.typed["patient"]["sex"] == ["Unknown"]
    rules = {(f.rule, f.fields[0]) for f in flags}
    assert ("checkbox_corrected", "patient.sex") in rules
    assert ("ocr_disagrees", "patient.name") in rules
    sex = next(g for g in fixed.universal.checkboxes if g.key == "sex")
    assert sex.selected == ["Unknown"] and all(o.method for o in sex.options)


def test_group_prefers_labels_right_of_boxes():
    from form_agent.checkboxes import Box, locate_group

    # "Q? [ ] Yes [ ] No": the No box also has "Yes" on its left; the right-hand labels must win.
    yes, no = Box(100, 10, 12, 12, 0.0, label="yes", left="q"), Box(150, 10, 12, 12, 0.3, label="no", left="yes")
    assert locate_group(["Yes", "No"], [yes, no]) == {"Yes": yes, "No": no}
    # "JUNE 30 [X]  SEP 22 [ ]": labels only on the left
    june, sep = Box(100, 10, 60, 14, 0.2, label="sep22", left="june30"), Box(250, 10, 20, 10, 0.0, left="sep22")
    assert locate_group(["JUNE 30", "SEP 22"], [june, sep]) == {"JUNE 30": june, "SEP 22": sep}
