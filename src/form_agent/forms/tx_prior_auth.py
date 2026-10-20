"""Texas Standard Prior Authorization Request Form for Health Care Services (TDI NOFR001)."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from form_agent.forms.base import FormPlugin, register_form
from form_agent.normalize import normalize_value, parse_date
from form_agent.schema import (
    CheckboxGroup,
    CheckboxOption,
    Flag,
    FormField,
    Table,
    UniversalForm,
    flatten,
    kind,
)

REVIEW_TYPES = ["Non-Urgent", "Urgent"]
REQUEST_TYPES = ["Initial Request", "Extension/Renewal/Amendment"]
SEXES = ["Male", "Female", "Other", "Unknown"]
SETTINGS = ["Inpatient", "Outpatient", "Provider Office", "Observation", "Home", "Day Surgery", "Other"]
THERAPIES = [
    "Physical Therapy",
    "Occupational Therapy",
    "Speech Therapy",
    "Cardiac Rehab",
    "Mental Health/Substance Abuse",
]
YES_NO = ["Yes", "No"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Submission(_Model):
    issuer_name: str | None = Field(None, **kind("text"))
    date: str | None = Field(None, **kind("date"))


class GeneralInfo(_Model):
    review_type: list[str] = Field(default_factory=list, **kind("checkbox"))
    clinical_reason_for_urgency: str | None = Field(None, **kind("text"))
    request_type: list[str] = Field(default_factory=list, **kind("checkbox"))
    prev_auth_number: str | None = Field(None, **kind("id"))


class Patient(_Model):
    name: str | None = Field(None, **kind("name"))
    phone: str | None = Field(None, **kind("phone"))
    dob: str | None = Field(None, **kind("date"))
    sex: list[str] = Field(default_factory=list, **kind("checkbox"))
    subscriber_name: str | None = Field(None, **kind("name"))
    member_id: str | None = Field(None, **kind("id"))
    group_number: str | None = Field(None, **kind("id"))


class RequestingProvider(_Model):
    name: str | None = Field(None, **kind("name"))
    npi: str | None = Field(None, **kind("npi"))
    specialty: str | None = Field(None, **kind("text"))
    phone: str | None = Field(None, **kind("phone"))
    fax: str | None = Field(None, **kind("phone"))
    contact_name: str | None = Field(None, **kind("name"))
    contact_phone: str | None = Field(None, **kind("phone"))
    signature_and_date: str | None = Field(None, **kind("text"))


class ServiceProvider(_Model):
    name: str | None = Field(None, **kind("name"))
    npi: str | None = Field(None, **kind("npi"))
    specialty: str | None = Field(None, **kind("text"))
    phone: str | None = Field(None, **kind("phone"))


class PrimaryCareProvider(_Model):
    name: str | None = Field(None, **kind("name"))
    phone: str | None = Field(None, **kind("phone"))
    fax: str | None = Field(None, **kind("phone"))


class ServiceLine(_Model):
    service: str | None = Field(None, **kind("text"))
    code: str | None = Field(None, **kind("code"))
    start_date: str | None = Field(None, **kind("date"))
    end_date: str | None = Field(None, **kind("date"))
    diagnosis: str | None = Field(None, **kind("text"))
    icd_code: str | None = Field(None, **kind("code"))


class Therapy(_Model):
    types: list[str] = Field(default_factory=list, **kind("checkbox"))
    sessions: str | None = Field(None, **kind("number"))
    duration: str | None = Field(None, **kind("text"))
    frequency: str | None = Field(None, **kind("text"))
    other: str | None = Field(None, **kind("text"))


class HomeHealth(_Model):
    requested: bool = Field(False, **kind("checkbox"))
    md_order_attached: list[str] = Field(default_factory=list, **kind("checkbox"))
    nursing_assessment_attached: list[str] = Field(default_factory=list, **kind("checkbox"))
    visits: str | None = Field(None, **kind("number"))
    duration: str | None = Field(None, **kind("text"))
    frequency: str | None = Field(None, **kind("text"))
    other: str | None = Field(None, **kind("text"))


class DME(_Model):
    requested: bool = Field(False, **kind("checkbox"))
    md_order_attached: list[str] = Field(default_factory=list, **kind("checkbox"))
    title19_attached: list[str] = Field(default_factory=list, **kind("checkbox"))
    equipment: str | None = Field(None, **kind("text"))
    duration: str | None = Field(None, **kind("text"))


class Services(_Model):
    icd_version: str | None = Field(None, **kind("id"))
    lines: list[ServiceLine] = Field(default_factory=list)
    setting: list[str] = Field(default_factory=list, **kind("checkbox"))
    setting_other: str | None = Field(None, **kind("text"))
    therapy: Therapy = Field(default_factory=Therapy)
    home_health: HomeHealth = Field(default_factory=HomeHealth)
    dme: DME = Field(default_factory=DME)


class TexasPriorAuth(_Model):
    form_type: Literal["tx_prior_auth"] = "tx_prior_auth"
    submission: Submission = Field(default_factory=Submission)
    general: GeneralInfo = Field(default_factory=GeneralInfo)
    patient: Patient = Field(default_factory=Patient)
    requesting_provider: RequestingProvider = Field(default_factory=RequestingProvider)
    service_provider: ServiceProvider = Field(default_factory=ServiceProvider)
    pcp: PrimaryCareProvider = Field(default_factory=PrimaryCareProvider)
    services: Services = Field(default_factory=Services)
    facility_address: str | None = Field(None, **kind("address"))


S1 = "Section I — Submission"
S2 = "Section II — General Information"
S3 = "Section III — Patient Information"
S4R = "Section IV — Provider Information / Requesting Provider or Facility"
S4S = "Section IV — Provider Information / Service Provider or Facility"
S4P = "Section IV — Provider Information / Primary Care Provider"
S5 = "Section V — Services Requested and Supporting Diagnoses"
S6 = "Section VI — Clinical Documentation"

SERVICE_HEADERS = [
    "Planned Service or Procedure", "Code", "Start Date", "End Date",
    "Diagnosis Description", "ICD Code",
]
SERVICE_TYPES = ["text", "code", "date", "date", "text", "code"]


class TexasPriorAuthPlugin(FormPlugin):
    form_type = "tx_prior_auth"
    title = "Texas Standard Prior Authorization Request Form for Health Care Services"
    schema = TexasPriorAuth

    _TITLE = re.compile(r"texas\s+standard\s+prior\s+authori[sz]ation", re.I)
    _MARKERS = ["prior authorization", "npi", "icd", "utilization management", "service provider or facility"]

    def detect(self, universal: UniversalForm) -> float:
        if universal.title and self._TITLE.search(universal.title):
            return 1.0
        text = " ".join(
            [universal.title or ""]
            + [f.label for f in universal.fields]
            + [g.label for g in universal.checkboxes]
            + [t.title or "" for t in universal.tables]
        ).lower()
        return sum(m in text for m in self._MARKERS) / len(self._MARKERS) * 0.9

    def checkbox_groups(self) -> list[tuple[str, list[str], str]]:
        return [
            ("services.setting", SETTINGS, "Inpatient Outpatient"),
            ("services.therapy.types", THERAPIES, "Therapy"),
            ("patient.sex", SEXES, "Sex"),
            ("general.review_type", REVIEW_TYPES, "Review Type"),
            ("general.request_type", REQUEST_TYPES, "Request Type"),
            ("services.home_health.requested", ["Home Health"], "Home Health"),
            ("services.dme.requested", ["DME"], "DME"),
            ("services.home_health.md_order_attached", YES_NO, "Home Health MD Signed Order"),
            ("services.home_health.nursing_assessment_attached", YES_NO, "Nursing Assessment"),
            ("services.dme.md_order_attached", YES_NO, "DME MD Signed Order"),
            ("services.dme.title19_attached", YES_NO, "Medicaid Title 19 Certification"),
        ]

    def validate(self, typed: TexasPriorAuth, today=None) -> list[Flag]:
        flags: list[Flag] = []

        def flag(rule, severity, message, *fields):
            flags.append(Flag(rule=rule, severity=severity, message=message, fields=list(fields)))

        required = {
            "patient.name": typed.patient.name,
            "patient.dob": typed.patient.dob,
            "patient.member_id": typed.patient.member_id,
            "requesting_provider.name": typed.requesting_provider.name,
            "requesting_provider.npi": typed.requesting_provider.npi,
            "submission.date": typed.submission.date,
        }
        for path, value in required.items():
            if not value:
                flag("missing_required", "error", f"Required field {path} is empty.", path)
        if not typed.services.lines:
            flag("missing_required", "error", "No services are listed in Section V.", "services.lines")
        for path, label in [("general.review_type", "review type"), ("general.request_type", "request type")]:
            if not _get(typed, path):
                flag("no_selection", "warning", f"No {label} is ticked.", path)
        if "Urgent" in typed.general.review_type and not typed.general.clinical_reason_for_urgency:
            flag("urgent_without_reason", "info", "Marked Urgent but no clinical reason for urgency is given.",
                 "general.review_type", "general.clinical_reason_for_urgency")
        if "Extension/Renewal/Amendment" in typed.general.request_type and not typed.general.prev_auth_number:
            flag("extension_without_prev_auth", "warning", "Extension/renewal requested but no previous auth # given.",
                 "general.prev_auth_number")
        if typed.services.icd_version and typed.services.icd_version.strip() != "10":
            flag("icd_version", "warning", f"ICD version is '{typed.services.icd_version}', expected 10.",
                 "services.icd_version")

        submitted = parse_date(typed.submission.date)
        for i, line in enumerate(typed.services.lines):
            p = f"services.lines[{i}]"
            for attr, what in [("code", "procedure code"), ("icd_code", "ICD code")]:
                if not getattr(line, attr):
                    flag("missing_required", "warning", f"Service line {i + 1} has no {what}.", f"{p}.{attr}")
            start, end = parse_date(line.start_date), parse_date(line.end_date)
            if start and end and end < start:
                flag("end_before_start", "error", f"Service line {i + 1} ends ({end}) before it starts ({start}).",
                     f"{p}.start_date", f"{p}.end_date")
            if start and submitted:
                gap = (start - submitted).days
                if gap > 365:
                    flag("service_far_after_request", "warning",
                         f"Service line {i + 1} starts {gap} days after the request date ({submitted}).",
                         f"{p}.start_date", "submission.date")
                elif gap < 0:
                    flag("service_before_request", "info",
                         f"Service line {i + 1} started {-gap} days before the request date (retroactive).",
                         f"{p}.start_date", "submission.date")
        return flags

    def single_choice_groups(self) -> list[str]:
        return [
            "general.review_type", "general.request_type", "patient.sex",
            "services.home_health.md_order_attached", "services.home_health.nursing_assessment_attached",
            "services.dme.md_order_attached", "services.dme.title19_attached",
        ]

    def extraction_hints(self) -> str:
        return f"""
Checkbox option names:
- general.review_type: {REVIEW_TYPES}
- general.request_type: {REQUEST_TYPES}
- patient.sex: {SEXES}
- services.setting: {SETTINGS}
- services.therapy.types: {THERAPIES}
- yes/no attachment questions: {YES_NO}
- services.home_health.requested / services.dme.requested: true only if that section's own box is ticked.
Field notes:
- Dates as printed (MM/DD/YYYY); phone and fax numbers as printed, e.g. "(519)-490-4052".
- NPI and member IDs: copy every digit as printed, even if the count looks wrong.
- services.icd_version: the number written after "ICD version".
- services.lines: one per filled row of the Section V table (service, CPT/HCPCS code, start/end dates,
  diagnosis description, ICD code).
- facility_address: the address written in Section VI.
- submission.issuer_name: null if blurred or redacted."""

    def to_universal(self, typed: TexasPriorAuth) -> UniversalForm:
        fields: list[FormField] = []
        groups: list[CheckboxGroup] = []

        def field(section, label, key, value, value_type="text"):
            fields.append(FormField(label=label, key=key, value=value, value_type=value_type, section=section))

        def group(section, label, key, options, selected):
            groups.append(CheckboxGroup(
                label=label, key=key, section=section,
                options=[CheckboxOption(label=o, checked=o in selected) for o in options],
            ))

        t = typed
        field(S1, "Issuer Name:", "issuer_name", t.submission.issuer_name)
        field(S1, "Date:", "submission_date", t.submission.date, "date")
        group(S2, "Review Type", "review_type", REVIEW_TYPES, t.general.review_type)
        field(S2, "Clinical Reason for Urgency:", "clinical_reason_for_urgency", t.general.clinical_reason_for_urgency)
        group(S2, "Request Type", "request_type", REQUEST_TYPES, t.general.request_type)
        field(S2, "Prev. Auth. #:", "prev_auth_number", t.general.prev_auth_number, "id")

        p = t.patient
        field(S3, "Name:", "patient_name", p.name, "name")
        field(S3, "Phone:", "patient_phone", p.phone, "phone")
        field(S3, "DOB:", "date_of_birth", p.dob, "date")
        group(S3, "Sex", "sex", SEXES, p.sex)
        field(S3, "Subscriber Name (if different):", "subscriber_name", p.subscriber_name, "name")
        field(S3, "Member or Medicaid ID #:", "member_id", p.member_id, "id")
        field(S3, "Group #:", "group_number", p.group_number, "id")

        r = t.requesting_provider
        field(S4R, "Name:", "requesting_provider_name", r.name, "name")
        field(S4R, "NPI #:", "requesting_provider_npi", r.npi, "npi")
        field(S4R, "Specialty:", "requesting_provider_specialty", r.specialty)
        field(S4R, "Phone:", "requesting_provider_phone", r.phone, "phone")
        field(S4R, "Fax:", "requesting_provider_fax", r.fax, "phone")
        field(S4R, "Contact Name:", "contact_name", r.contact_name, "name")
        field(S4R, "Phone:", "contact_phone", r.contact_phone, "phone")
        field(S4R, "Requesting Provider's Signature and Date (if required):", "signature_and_date", r.signature_and_date)

        s = t.service_provider
        field(S4S, "Name:", "service_provider_name", s.name, "name")
        field(S4S, "NPI #:", "service_provider_npi", s.npi, "npi")
        field(S4S, "Specialty:", "service_provider_specialty", s.specialty)
        field(S4S, "Phone:", "service_provider_phone", s.phone, "phone")

        field(S4P, "Primary Care Provider Name (see instructions):", "pcp_name", t.pcp.name, "name")
        field(S4P, "Phone:", "pcp_phone", t.pcp.phone, "phone")
        field(S4P, "Fax:", "pcp_fax", t.pcp.fax, "phone")

        sv = t.services
        field(S5, "ICD version", "icd_version", sv.icd_version, "id")
        table = Table(
            title="Services Requested", section=S5, headers=SERVICE_HEADERS, column_types=SERVICE_TYPES,
            rows=[[l.service, l.code, l.start_date, l.end_date, l.diagnosis, l.icd_code] for l in sv.lines],
        )
        group(S5, "Setting", "setting", SETTINGS, sv.setting)
        field(S5, "Other:", "setting_other", sv.setting_other)
        group(S5, "Therapy", "therapy_type", THERAPIES, sv.therapy.types)
        field(S5, "Number of Sessions:", "therapy_sessions", sv.therapy.sessions, "number")
        field(S5, "Duration:", "therapy_duration", sv.therapy.duration)
        field(S5, "Frequency:", "therapy_frequency", sv.therapy.frequency)
        field(S5, "Other:", "therapy_other", sv.therapy.other)

        hh = sv.home_health
        group(S5, "Home Health", "home_health", ["Home Health"], ["Home Health"] if hh.requested else [])
        group(S5, "Home Health: MD Signed Order Attached?", "home_health_md_order", YES_NO, hh.md_order_attached)
        group(S5, "Home Health: Nursing Assessment Attached?", "home_health_nursing_assessment", YES_NO,
              hh.nursing_assessment_attached)
        field(S5, "Number of Visits:", "home_health_visits", hh.visits, "number")
        field(S5, "Duration:", "home_health_duration", hh.duration)
        field(S5, "Frequency:", "home_health_frequency", hh.frequency)
        field(S5, "Other:", "home_health_other", hh.other)

        d = sv.dme
        group(S5, "DME", "dme", ["DME"], ["DME"] if d.requested else [])
        group(S5, "DME: MD Signed Order Attached?", "dme_md_order", YES_NO, d.md_order_attached)
        group(S5, "DME: Title 19 Certification Attached?", "dme_title19", YES_NO, d.title19_attached)
        field(S5, "Equipment/Supplies (include any HCPCS Codes):", "dme_equipment", d.equipment)
        field(S5, "Duration:", "dme_duration", d.duration)

        field(S6, "Physical address of service provider or facility", "facility_address", t.facility_address, "address")
        return UniversalForm(title=self.title, fields=fields, checkboxes=groups, tables=[table])

    def field_paths(self) -> dict[str, str]:
        return {
            "issuer_name": "submission.issuer_name", "submission_date": "submission.date",
            "review_type": "general.review_type", "request_type": "general.request_type",
            "clinical_reason_for_urgency": "general.clinical_reason_for_urgency",
            "prev_auth_number": "general.prev_auth_number",
            "patient_name": "patient.name", "patient_phone": "patient.phone", "date_of_birth": "patient.dob",
            "sex": "patient.sex", "subscriber_name": "patient.subscriber_name", "member_id": "patient.member_id",
            "group_number": "patient.group_number",
            "requesting_provider_name": "requesting_provider.name", "requesting_provider_npi": "requesting_provider.npi",
            "requesting_provider_specialty": "requesting_provider.specialty",
            "requesting_provider_phone": "requesting_provider.phone", "requesting_provider_fax": "requesting_provider.fax",
            "contact_name": "requesting_provider.contact_name", "contact_phone": "requesting_provider.contact_phone",
            "signature_and_date": "requesting_provider.signature_and_date",
            "service_provider_name": "service_provider.name", "service_provider_npi": "service_provider.npi",
            "service_provider_specialty": "service_provider.specialty", "service_provider_phone": "service_provider.phone",
            "pcp_name": "pcp.name", "pcp_phone": "pcp.phone", "pcp_fax": "pcp.fax",
            "icd_version": "services.icd_version", "setting": "services.setting",
            "setting_other": "services.setting_other",
            "therapy_type": "services.therapy.types", "therapy_sessions": "services.therapy.sessions",
            "therapy_duration": "services.therapy.duration", "therapy_frequency": "services.therapy.frequency",
            "therapy_other": "services.therapy.other",
            "home_health": "services.home_health.requested",
            "home_health_md_order": "services.home_health.md_order_attached",
            "home_health_nursing_assessment": "services.home_health.nursing_assessment_attached",
            "home_health_visits": "services.home_health.visits", "home_health_duration": "services.home_health.duration",
            "home_health_frequency": "services.home_health.frequency", "home_health_other": "services.home_health.other",
            "dme": "services.dme.requested", "dme_md_order": "services.dme.md_order_attached",
            "dme_title19": "services.dme.title19_attached", "dme_equipment": "services.dme.equipment",
            "dme_duration": "services.dme.duration",
            "facility_address": "facility_address",
        }

    def table_paths(self) -> dict[int, tuple[str, list[str]]]:
        return {0: ("services.lines", ["service", "code", "start_date", "end_date", "diagnosis", "icd_code"])}

    def table_rows(self, doc_id: str, typed: TexasPriorAuth) -> dict[str, list[dict[str, Any]]]:
        """Typed tables hold normalized values (ISO dates, digit-only phones) so SQL can sort and filter."""

        def norm(value, value_kind):
            if isinstance(value, list):
                return ", ".join(value)
            return normalize_value(value, value_kind) if isinstance(value, str) else value

        row: dict[str, Any] = {"doc_id": doc_id}
        for path, (value, value_kind) in flatten(typed).items():
            if not path.startswith("services.lines."):
                row[path.replace(".", "_")] = norm(value, value_kind)
        services = []
        for i, line in enumerate(typed.services.lines):
            out: dict[str, Any] = {"doc_id": doc_id, "line": i + 1}
            for name, (value, value_kind) in flatten(line).items():
                out[name] = norm(value, value_kind)
            services.append(out)
        return {"tx_prior_auth": [row], "tx_services": services}


register_form(TexasPriorAuthPlugin())


def _get(model, path: str):
    for part in path.split("."):
        model = getattr(model, part)
    return model
