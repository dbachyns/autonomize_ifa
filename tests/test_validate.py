from datetime import date

import pytest

from form_agent.forms import get_plugin
from form_agent.forms.tx_prior_auth import ServiceLine, TexasPriorAuth
from form_agent.schema import CheckboxGroup, CheckboxOption, FormField, FormRecord, RunInfo, UniversalForm
from form_agent.validate import check_value, npi_is_valid, validate

TODAY = date(2026, 10, 19)


def rules(flags):
    return {f.rule for f in flags}


@pytest.mark.parametrize("npi, ok", [("1234567893", True), ("1234567890", False), ("4336428227", False)])
def test_npi_checksum(npi, ok):
    assert npi_is_valid(npi) is ok


@pytest.mark.parametrize(
    "name, value, kind, rule",
    [
        ("npi", "7660952", "npi", "npi_length"),
        ("npi", "1234567890", "npi", "npi_checksum"),
        ("patient.dob", "13/45/2001", "date", "date_unreadable"),
        ("patient.dob", "04/02/2030", "date", "dob_implausible"),
        ("phone", "(519)-490-405", "phone", "phone_length"),
        ("services.lines.0.icd_code", "S6.326D", "code", "icd10_format"),
        ("services.lines.0.code", "6164", "code", "procedure_code_format"),
    ],
)
def test_value_rules_fire(name, value, kind, rule):
    assert rule in rules(check_value(name, value, kind, today=TODAY))


@pytest.mark.parametrize(
    "name, value, kind",
    [
        ("npi", "1234567893", "npi"),
        ("dob", "04/02/1977", "date"),
        ("phone", "(519)-490-4052", "phone"),
        ("services.lines.0.icd_code", "O36.8391", "code"),
        ("services.lines.0.icd_code", "W89.0XXA", "code"),
        ("services.lines.0.code", "0441T", "code"),
        ("services.lines.0.code", "J1100", "code"),
        ("anything", None, "date"),
    ],
)
def test_valid_values_pass(name, value, kind):
    assert check_value(name, value, kind, today=TODAY) == []


def _tx_record(typed: TexasPriorAuth) -> FormRecord:
    plugin = get_plugin("tx_prior_auth")
    return FormRecord(doc_id="d", source_file="d.png", run=RunInfo(reader="test"), form_type="tx_prior_auth",
                      universal=plugin.to_universal(typed), typed=typed.model_dump())


def test_tx_rules():
    t = TexasPriorAuth()
    t.submission.date = "01/10/2022"
    t.general.review_type = ["Non-Urgent", "Urgent"]
    t.services.lines = [
        ServiceLine(code="61641", icd_code="S66.326D", start_date="05/22/2022", end_date="05/17/2022"),
        ServiceLine(code="33822", icd_code="W89.0XXA", start_date="05/08/2023", end_date="05/13/2023"),
    ]
    found = rules(validate(_tx_record(t), today=TODAY))
    assert {"multiple_ticks", "end_before_start", "service_far_after_request", "missing_required",
            "urgent_without_reason", "no_selection"} <= found


def test_generic_rules_on_unknown_form():
    rec = FormRecord(
        doc_id="x", source_file="x.png", run=RunInfo(reader="test"),
        universal=UniversalForm(
            fields=[FormField(label="Provider NPI", value="123", value_type="npi"),
                    FormField(label="Date of Birth", value="01/01/2090", value_type="date")],
            checkboxes=[CheckboxGroup(label="Smoker?", options=[CheckboxOption(label="Yes", checked=True),
                                                                CheckboxOption(label="No", checked=True)])],
        ),
    )
    assert {"npi_length", "dob_implausible", "multiple_ticks"} <= rules(validate(rec, today=TODAY))


def test_empty_extraction_is_an_error():
    rec = FormRecord(doc_id="x", source_file="x.png", run=RunInfo(reader="test"), universal=UniversalForm())
    assert "empty_extraction" in rules(validate(rec, today=TODAY))
