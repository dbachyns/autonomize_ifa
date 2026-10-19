import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from form_agent.agent import FormAgent
from form_agent.qa_eval import contains, grade
from form_agent.schema import FormField, FormRecord, RunInfo, UniversalForm
from form_agent.store import build_index, load_record, save_record
from form_agent.tools import FormTools

GROUND_TRUTH = Path(__file__).parent.parent / "data" / "ground_truth"


@pytest.fixture()
def paths(tmp_path):
    cache = tmp_path / "cache"
    for path in GROUND_TRUTH.glob("*.json"):
        save_record(cache, load_record(path))
    save_record(cache, FormRecord(
        doc_id="fax_cover", source_file="x.png", run=RunInfo(reader="openai"),
        universal=UniversalForm(title="FAX COVER", fields=[
            FormField(label="TO:", value="Jack Reilly", value_type="name"),
            FormField(label="Phone:", value="(212) 555-0199", value_type="phone", source="printed")]),
    ))
    db = tmp_path / "forms.db"
    build_index(cache, db)
    return db, cache


@pytest.fixture()
def tools(paths):
    return FormTools(*paths)


# --- tools -------------------------------------------------------------------------------


def test_resolve_short_and_prefix_ids(tools):
    full = tools.resolve("0a01f77b")
    assert full.startswith("0a01f77b-") and tools.short[full] == "0a01f77b"
    assert tools.resolve("[0a01f]") == full
    with pytest.raises(Exception, match="ambiguous"):
        tools.resolve("0a")
    assert "No document" in tools.call("get_document", {"doc_id": "nope"})


def test_get_document_renders_typed_values_blanks_and_flags(tools):
    text = tools.get_document("0a01f77b")
    assert "Name: Daniel Jarvis" in text
    assert "Clinical Reason for Urgency: (blank)" in text
    assert "[x] Unknown" in text and "[ ] Female" in text
    assert "Open placement | 44300" in text
    assert "REVIEW FLAGS" in text


def test_get_document_universal_form(tools):
    text = tools.get_document("fax_cover")
    assert "TO: Jack Reilly" in text
    assert "Phone: (212) 555-0199 {printed}" in text


def test_sql_is_read_only_and_capped(tools):
    out = tools.sql("SELECT general_review_type, COUNT(*) FROM tx_prior_auth GROUP BY 1 ORDER BY 1")
    assert out["rows"] == [["Non-Urgent", 6], ["Urgent", 5]]
    for bad in ["DELETE FROM documents", "SELECT 1; DROP TABLE documents", "PRAGMA writable_schema=1"]:
        assert "error" in json.loads(tools.call("sql", {"query": bad}))
    # A WITH that tries to write still hits the read-only connection.
    assert "error" in json.loads(tools.call("sql", {"query": "WITH x AS (SELECT 1) INSERT INTO documents SELECT * FROM x"}))
    tools.max_rows = 3
    assert "truncated" in tools.sql("SELECT * FROM active_fields")["note"]


def test_sql_errors_are_returned_to_the_model(tools):
    assert "no such column" in tools.call("sql", {"query": "SELECT nope FROM documents"})


def test_search_all_words_then_any(tools):
    hits = tools.search("Daniel Jarvis")
    assert {h["id"] for h in hits} == {"0a01f77b"}
    assert tools.search("Jarvis zzzunknown")  # falls back to any word
    assert tools.search("Reilly", doc_id="fax_cover")[0]["label"] == "TO:"


def test_list_documents_and_schema(tools):
    docs = tools.list_documents()
    assert len(docs) == 12
    tx = tools.list_documents(form_type="tx_prior_auth")
    assert len(tx) == 11 and {"id", "title", "errors", "warnings"} <= set(tx[0])
    schema = tools.schema_description()
    assert "general_review_type [" in schema and "Urgent" in schema
    assert "form_type_constants" in schema


# --- agent loop (fake OpenAI client) ------------------------------------------------------


class FakeResponses:
    """Replays scripted responses: a tool call, then a final answer."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.script.pop(0)


def _response(output, text=""):
    return SimpleNamespace(output=output, output_text=text,
                           usage=SimpleNamespace(input_tokens=1000, output_tokens=100))


def _call(name, arguments, call_id="c1"):
    return SimpleNamespace(type="function_call", name=name, arguments=json.dumps(arguments), call_id=call_id)


def test_agent_runs_tools_and_returns_answer(paths):
    responses = FakeResponses([
        _response([_call("sql", {"query": "SELECT COUNT(*) FROM tx_prior_auth WHERE general_review_type='Urgent'"})]),
        _response([], "5 requests are urgent."),
    ])
    agent = FormAgent(*paths, model="gpt-6-luna", client=SimpleNamespace(responses=responses))
    result = agent.ask("How many urgent?")
    assert result.answer == "5 requests are urgent."
    assert [s.tool for s in result.steps] == ["sql"]
    assert '"rows": [[5]]' in result.steps[0].result_preview
    # The tool output went back to the model in the second request.
    second_input = responses.calls[1]["input"]
    assert second_input[-1]["type"] == "function_call_output" and "[[5]]" in second_input[-1]["output"]
    assert result.cost_usd == pytest.approx(2 * (1000 * 0.10 + 100 * 0.50) / 1e6)


def test_agent_forces_answer_after_max_steps(paths):
    loop = [_response([_call("list_documents", {"form_type": None})]) for _ in range(2)]
    responses = FakeResponses([*loop, _response([], "done")])
    agent = FormAgent(*paths, model="gpt-6-luna", client=SimpleNamespace(responses=responses), max_steps=2)
    assert agent.ask("loop forever").answer == "done"
    assert responses.calls[-1]["tool_choice"] == "none"


def test_single_form_question_preloads_document(paths):
    responses = FakeResponses([_response([], "62106 [0a01f77b]")])
    agent = FormAgent(*paths, model="gpt-6-luna", client=SimpleNamespace(responses=responses))
    agent.ask("Member ID?", doc_id="0a01f77b")
    assert "Daniel Jarvis" in responses.calls[0]["input"][0]["content"]


def test_summary_uses_no_tools(paths):
    responses = FakeResponses([_response([], "**Summary** ...")])
    agent = FormAgent(*paths, model="gpt-6-luna", client=SimpleNamespace(responses=responses))
    agent.summarize("0a01f77b")
    assert "tools" not in responses.calls[0]
    assert "[0a01f77b]" in responses.calls[0]["instructions"]


# --- Q&A grading --------------------------------------------------------------------------


@pytest.mark.parametrize("answer, alt, expected", [
    ("Fax to **866-617-4967**.", "8666174967", True),
    ("The sex is Female.", "male", False),
    ("The sex is Male.", "male", True),
    ("Both NPIs fail", "npi", True),
    ("30 forms", "3", False),
    ("average 4.18 sessions", "4.18", True),
    ("It was left blank.", "blank", True),
])
def test_contains(answer, alt, expected):
    assert contains(answer, alt) is expected


def test_grade_expect_and_forbid():
    item = {"id": "x", "kind": "single", "question": "q", "expect": [["male"]], "forbid": ["female"]}
    ok = SimpleNamespace(answer="Male", steps=[], seconds=1.0, cost_usd=0.0)
    bad = SimpleNamespace(answer="Male or female", steps=[], seconds=1.0, cost_usd=0.0)
    assert grade(item, ok).passed and not grade(item, bad).passed
