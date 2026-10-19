from form_agent.evaluate import score_pairs, score_typed, summarize_typed
from form_agent.forms.tx_prior_auth import ServiceLine, TexasPriorAuth


def _form(**patient) -> TexasPriorAuth:
    f = TexasPriorAuth()
    for k, v in patient.items():
        setattr(f.patient, k, v)
    f.services.lines = [ServiceLine(code="61641", icd_code="S66.326D"), ServiceLine(code="33822", icd_code="W89.0XXA")]
    return f


def _by_path(results):
    return {r.path: r for r in results}


def test_identical_is_perfect():
    gt = _form(name="Richard Mckee", dob="04/02/1977", sex=["Female"])
    summary, _, errors = summarize_typed(score_typed("d", gt, gt.model_copy(deep=True)))
    assert summary["all_fields"]["accuracy"] == 1.0 and not errors


def test_normalization_forgives_format_not_content():
    gt = _form(name="Richard Mckee", dob="04/02/1977", phone="(519)-490-4052")
    pred = _form(name="richard  mckee", dob="4/2/1977", phone="519 490 4052")
    assert all(r.correct for r in score_typed("d", gt, pred))
    pred.patient.dob = "04/02/1971"
    assert not _by_path(score_typed("d", gt, pred))["patient.dob"].correct


def test_hallucinated_value_in_empty_field_is_an_error():
    gt, pred = _form(), _form(group_number="12345")
    r = _by_path(score_typed("d", gt, pred))["patient.group_number"]
    assert r.expected_empty and not r.correct


def test_extra_checkbox_tick_is_an_error():
    gt, pred = _form(sex=["Unknown"]), _form(sex=["Female", "Unknown"])
    assert not _by_path(score_typed("d", gt, pred))["patient.sex"].correct


def test_table_rows_aligned_even_if_reordered_or_missing():
    gt = _form()
    pred = _form()
    pred.services.lines = [ServiceLine(code="33822", icd_code="W89.0XXA")]  # first row missing
    results = _by_path(score_typed("d", gt, pred))
    assert results["services.lines[0].code"].correct is False
    assert results["services.lines[1].code"].correct is True


def test_extra_table_row_is_reported():
    gt, pred = _form(), _form()
    pred.services.lines.append(ServiceLine(code="99999"))
    assert any(r.path.startswith("services.lines[extra]") and not r.correct for r in score_typed("d", gt, pred))


def test_pair_recall_tolerates_ocr_spacing_and_extra_predictions():
    expected = [("FAX NUMBER:", "(336) 335- 7392"), ("DATE:", "12 /10 /98")]
    predicted = [("Fax Number", "(336) 335-7392"), ("Date", "12/10/98"), ("Pages", "3")]
    assert all(r.found for r in score_pairs("d", expected, predicted))


def test_pair_recall_requires_matching_label():
    results = score_pairs("d", [("To:", "George Baroody")], [("From:", "George Baroody")])
    assert not results[0].found
