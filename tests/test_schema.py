import json
from pathlib import Path

import pytest

from form_agent.forms import get_plugin, identify
from form_agent.forms.tx_prior_auth import ServiceLine, TexasPriorAuth
from form_agent.readers import Reader, available_readers, get_reader, register
from form_agent.schema import FormRecord, RunInfo, UniversalForm, flatten

DATA = Path(__file__).parent.parent / "data"
GROUND_TRUTH = sorted((DATA / "ground_truth").glob("*.json"))


def _record(universal: UniversalForm | None = None) -> FormRecord:
    return FormRecord(doc_id="f1", source_file="f1.png", run=RunInfo(reader="test"), universal=universal or UniversalForm())


def test_roundtrip_record():
    rec = _record()
    assert FormRecord.model_validate_json(rec.model_dump_json()) == rec


def test_unknown_fields_rejected():
    with pytest.raises(ValueError):
        TexasPriorAuth.model_validate({"patient": {"nickname": "x"}})


def test_flatten_paths_and_kinds():
    data = TexasPriorAuth()
    data.services.lines.append(ServiceLine(code="61641"))
    flat = flatten(data)
    assert flat["patient.dob"] == (None, "date")
    assert flat["general.review_type"] == ([], "checkbox")
    assert flat["services.lines.0.code"] == ("61641", "code")
    assert "form_type" not in flat


def test_tx_to_universal_keeps_unticked_options():
    data = TexasPriorAuth()
    data.general.review_type = ["Urgent"]
    universal = get_plugin("tx_prior_auth").to_universal(data)
    review = next(g for g in universal.checkboxes if g.key == "review_type")
    assert [(o.label, o.checked) for o in review.options] == [("Non-Urgent", False), ("Urgent", True)]


def test_identify_by_title_and_unknown():
    assert identify(UniversalForm(title="TEXAS STANDARD PRIOR AUTHORIZATION REQUEST FORM"))[0] == "tx_prior_auth"
    assert identify(UniversalForm(title="PURCHASE REQUISITION"))[0] == "unknown"


def test_reader_registry():
    @register("dummy")
    class Dummy(Reader):
        def read(self, image_path):
            return _record()

    assert "dummy" in available_readers()
    assert get_reader("dummy").read(Path("x.png")).run.reader == "test"
    with pytest.raises(ValueError):
        get_reader("nope")


@pytest.mark.parametrize("path", GROUND_TRUTH, ids=lambda p: p.stem[:8])
def test_ground_truth_files_validate(path):
    rec = FormRecord.model_validate(json.loads(path.read_text()))
    assert rec.run.reader == "ground_truth"
    assert (DATA / rec.source_file).exists()
    typed = get_plugin(rec.form_type).parse(rec.typed)
    assert get_plugin(rec.form_type).to_universal(typed) == rec.universal
