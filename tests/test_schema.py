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


def test_table_tidy_unfolds_side_by_side_and_numbers_repeats():
    from form_agent.schema import Table

    side = Table(section="Accounts within the region", headers=["NAME OF ACCOUNT", "IND/OR VOLUME", "NO. OF STORES",
                                                                "NAME OF ACCOUNT", "IND/LOR VOLUME", "NO. OF STORES"],
                 rows=[["Sheetz", "521/42", "150", "Kroger", None, "21"], ["CVS", None, "87", None, None, None]],
                 column_types=["text", "text", "number"] * 2).tidy()
    assert side.headers == ["NAME OF ACCOUNT", "IND/OR VOLUME", "NO. OF STORES"]
    assert side.rows == [["Sheetz", "521/42", "150"], ["CVS", None, "87"], ["Kroger", None, "21"]]
    assert side.name == "Accounts within the region"
    # Different left/right categories are two real columns, not a repeat: keep them, number the repeats.
    split = Table(headers=["NATIONAL CHAIN ACCOUNTS", "NO. OF STORES", "REGIONAL CHAIN ACCOUNTS", "NO. OF STORES"],
                  rows=[["A", "1", "B", "2"]]).tidy()
    assert split.headers == ["NATIONAL CHAIN ACCOUNTS", "NO. OF STORES", "REGIONAL CHAIN ACCOUNTS", "NO. OF STORES (2)"]
    assert split.rows == [["A", "1", "B", "2"]]
