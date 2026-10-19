import shutil
import sqlite3
from pathlib import Path

import pytest

from form_agent.schema import CheckboxGroup, CheckboxOption, FormField, FormRecord, RunInfo, TextBlock, UniversalForm
from form_agent.store import build_index, connect_readonly, iter_records, load_record, save_record

GROUND_TRUTH = Path(__file__).parent.parent / "data" / "ground_truth"


@pytest.fixture()
def db(tmp_path):
    cache = tmp_path / "cache"
    for path in GROUND_TRUTH.glob("*.json"):
        save_record(cache, load_record(path))
    # A non-Texas form and a second reader run for one Texas form.
    save_record(cache, FormRecord(
        doc_id="purchase_req", source_file="external/x.png", run=RunInfo(reader="ollama"),
        universal=UniversalForm(
            title="PURCHASE REQUISITION",
            fields=[FormField(label="DATE", value="April 19, 1988", value_type="date"),
                    FormField(label="VENDOR", value="Piedmont Research Laboratory")],
            checkboxes=[CheckboxGroup(label="Type", options=[CheckboxOption(label="PURCHASING", checked=False)])],
            text_blocks=[TextBlock(text="Preparation of smoke condensate for mouse skin bioassay")],
        ),
    ))
    gt = load_record(next(GROUND_TRUTH.glob("0ab19462*.json")))
    save_record(cache, gt.model_copy(update={"run": RunInfo(reader="ollama")}))
    path = tmp_path / "forms.db"
    counts = build_index(cache, path)
    return path, counts


def test_counts(db):
    _, counts = db
    assert counts["documents"] == 12
    assert counts["extractions"] == 13
    assert counts["tx_prior_auth"] == 11
    assert counts["tx_services"] == 26


def test_one_active_run_per_document(db):
    con = connect_readonly(db[0])
    rows = con.execute("SELECT doc_id, SUM(active) a FROM extractions GROUP BY doc_id").fetchall()
    assert all(r["a"] == 1 for r in rows)
    # ollama ranks above ground_truth in the default preference, so it is active for the duplicated form.
    reader = con.execute(
        "SELECT reader FROM extractions WHERE active = 1 AND doc_id LIKE '0ab19462%'"
    ).fetchone()["reader"]
    assert reader == "ollama"


def test_cross_form_queries(db):
    con = connect_readonly(db[0])
    urgent = con.execute("SELECT COUNT(*) FROM tx_prior_auth WHERE general_review_type = 'Urgent'").fetchone()[0]
    assert urgent == 5
    born_before_1960 = con.execute(
        "SELECT COUNT(*) FROM active_fields WHERE key = 'date_of_birth' AND value_norm < '1960-01-01'"
    ).fetchone()[0]
    assert born_before_1960 == 2
    any_form_dates = con.execute(
        "SELECT COUNT(DISTINCT doc_id) FROM active_fields WHERE value_type = 'date' AND value_norm LIKE '1988-%'"
    ).fetchone()[0]
    assert any_form_dates == 1  # the non-Texas form is queryable the same way
    fracture = con.execute(
        "SELECT COUNT(DISTINCT doc_id) FROM tx_services WHERE diagnosis LIKE '%fracture%'"
    ).fetchone()[0]
    assert fracture == 3


def test_full_text_search(db):
    con = connect_readonly(db[0])
    hits = {r["doc_id"] for r in con.execute("SELECT doc_id FROM search WHERE search MATCH 'bioassay'")}
    assert hits == {"purchase_req"}


def test_readonly_connection_rejects_writes(db):
    con = connect_readonly(db[0])
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM documents")


def test_iter_records_roundtrip(tmp_path):
    shutil.copytree(GROUND_TRUTH, tmp_path / "x" / "doc")
    assert len(list(iter_records(tmp_path / "x"))) == 11


def test_printed_template_values_are_collected_once(tmp_path):
    cache = tmp_path / "cache"
    for i, path in enumerate(sorted(GROUND_TRUTH.glob("*.json"))):
        rec = load_record(path)
        rec.universal.fields.append(FormField(label="Utilization Management:", value="855-322-4080",
                                              value_type="phone", source="printed"))
        if i == 0:  # a printed value unique to one form is not a template constant
            rec.universal.fields.append(FormField(label="Issuer Name:", value="Molina Healthcare of Texas",
                                                  source="printed"))
        save_record(cache, rec)
    db = tmp_path / "forms.db"
    counts = build_index(cache, db)
    con = connect_readonly(db)
    assert counts["form_type_constants"] == 1
    row = con.execute("SELECT label, value, documents FROM form_type_constants").fetchone()
    assert tuple(row) == ("Utilization Management:", "855-322-4080", 11)
    assert con.execute("SELECT COUNT(*) FROM fields WHERE template = 1").fetchone()[0] == 11
    assert con.execute("SELECT COUNT(*) FROM fields WHERE source = 'printed' AND template = 0").fetchone()[0] == 1
