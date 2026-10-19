"""The Q&A agent: answers questions about one form or many, and writes form summaries.

An LLM (OpenAI Responses API with function calling) decides which read-only tools to call (see
`form_agent.tools`): list documents, read one document, full-text search, or SQL over the
index. Counting and filtering happen in SQL, so cross-form numbers are exact; the model
only plans the lookups and writes the answer, citing the documents it used.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from form_agent.config import settings
from form_agent.readers.llm_common import cost_usd
from form_agent.tools import FormTools

TOOL_SPECS = [
    {
        "type": "function", "name": "list_documents",
        "description": "List the forms in the database with their short id, form type, title and number of "
                       "review flags. Optionally filter by form_type.",
        "parameters": {"type": "object", "properties": {"form_type": {"type": ["string", "null"]}},
                       "required": ["form_type"], "additionalProperties": False},
        "strict": True},
    {
        "type": "function", "name": "get_document",
        "description": "Read one whole form: every field (blank ones too), checkboxes, tables, free text and "
                       "its automatic review flags. Use for questions about a specific form.",
        "parameters": {"type": "object", "properties": {"doc_id": {"type": "string", "description": "short or full id"}},
                       "required": ["doc_id"], "additionalProperties": False},
        "strict": True},
    {
        "type": "function", "name": "search",
        "description": "Full-text search over all field values, labels, table cells and free text. Use to find "
                       "which forms mention a name, code, word or number. Optionally restrict to one doc_id.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"},
                                                        "doc_id": {"type": ["string", "null"]}},
                       "required": ["query", "doc_id"], "additionalProperties": False},
        "strict": True},
    {
        "type": "function", "name": "sql",
        "description": "Run one read-only SQLite SELECT over the index (schema in the system prompt). Use for "
                       "counting, filtering, grouping, comparing and listing across forms.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"], "additionalProperties": False},
        "strict": True},
]

SYSTEM_PROMPT = """You answer questions about filled-in forms that were scanned and extracted into a database.
Today is {today}. The database holds: {overview}.

How to work:
- Always look things up with the tools; never answer from memory or guess.
- Questions about one form: get_document. Counting, filtering, grouping or comparing across forms: sql
  (prefer the typed tables when the form type has one; they hold normalized values). Finding where a
  name/code/word appears: search. Pre-printed template info (e.g. the insurer's phone/fax numbers):
  form_type_constants, which is the same on every form of that type.
- If a query returns nothing or errors, fix it and try again (check spelling, use LIKE, try another table).

How to answer:
- Answer the question directly first, then the supporting details. Be concise.
- Cite every fact with the form's short id in brackets, e.g. [0a01f77b]. For lists, cite each item.
- Quote values as written on the form (names, codes, IDs). Dates may be written in a readable format.
- If a field exists but was left blank, say it was left blank. If the form has no such field, say so.
  Never invent a value.
- Values tagged {{printed}} are pre-printed on the form (part of the template), not entered by the person;
  say so when it matters.
- Mention review flags that affect the answer (e.g. an NPI that fails its check digit, OCR disagreement,
  dates that look wrong), so a human knows what to double-check.
- For counts, state the number and list the forms it covers (unless there are more than 15).

Database schema (SQLite; the active_* views hold the current extraction of each form):
{schema}
"""

SUMMARY_PROMPT = """You summarize one filled-in form for a reviewer who has not seen it. Today is {today}.
Use only the document below; never invent details. Write in plain language, in this structure:

**Summary** — 2–3 sentences: what the form is, who submitted it for whom, what is being requested or
reported, and when.
**Key details** — bullet points with the important filled-in values (people and their roles, IDs,
dates, selected options, table rows such as services/codes/diagnoses). Skip blank fields and pre-printed
template text unless it matters.
**Needs review** — bullet points for each error or warning flag, explained in plain words, plus anything
missing that the form clearly requires. Write "Nothing flagged." if there is nothing.

Cite the form as [{short_id}] once at the end of the Summary paragraph."""


@dataclass
class ToolStep:
    tool: str
    arguments: dict[str, Any]
    result_preview: str


@dataclass
class AgentAnswer:
    question: str
    answer: str
    model: str
    steps: list[ToolStep] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def trace(self) -> str:
        """The tool calls made, one per line (for demos and debugging)."""
        return "\n".join(f"  → {s.tool}({json.dumps(s.arguments, ensure_ascii=False)})" for s in self.steps)


class FormAgent:
    def __init__(self, db_path: Path | None = None, cache_dir: Path | None = None, model: str | None = None,
                 client: Any = None, max_steps: int = 10, today: date | None = None):
        self.tools = FormTools(db_path, cache_dir)
        self.model = model or settings.agent_model
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self.max_steps = max_steps
        self.today = today or date.today()
        self._system = SYSTEM_PROMPT.format(today=self.today.isoformat(), overview=self.tools.overview(),
                                            schema=self.tools.schema_description())

    # --- public API ------------------------------------------------------------------------

    def ask(self, question: str, doc_id: str | None = None) -> AgentAnswer:
        """Answer a question about one form (doc_id given) or about all forms."""
        user = question
        if doc_id:
            short = self.tools.short[self.tools.resolve(doc_id)]
            user = (f"This question is about form [{short}]. Its content:\n\n{self.tools.get_document(short)}"
                    f"\n\nQuestion: {question}")
        return self._run(question, self._system, user)

    def summarize(self, doc_id: str) -> AgentAnswer:
        """Readable summary of one form, with what needs human review."""
        full = self.tools.resolve(doc_id)
        short = self.tools.short[full]
        instructions = SUMMARY_PROMPT.format(today=self.today.isoformat(), short_id=short)
        return self._run(f"Summarize [{short}]", instructions, self.tools.get_document(full), use_tools=False)

    # --- loop ------------------------------------------------------------------------------

    def _run(self, question: str, instructions: str, user: str, use_tools: bool = True) -> AgentAnswer:
        start = time.monotonic()
        result = AgentAnswer(question=question, answer="", model=self.model)
        items: list[Any] = [{"role": "user", "content": user}]
        for step in range(self.max_steps + 1):
            kwargs: dict[str, Any] = {}
            if use_tools:
                # On the last step, force a written answer from what has been gathered.
                kwargs = {"tools": TOOL_SPECS, "tool_choice": "none" if step == self.max_steps else "auto"}
            response = self.client.responses.create(model=self.model, instructions=instructions, input=items, **kwargs)
            if response.usage:
                result.input_tokens += response.usage.input_tokens
                result.output_tokens += response.usage.output_tokens
            calls = [o for o in response.output if o.type == "function_call"]
            if not calls:
                result.answer = (response.output_text or "").strip()
                break
            items += response.output  # keeps reasoning items and the calls in the conversation
            for call in calls:
                try:
                    arguments = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    arguments, output = {}, json.dumps({"error": "arguments were not valid JSON"})
                else:
                    arguments = {k: v for k, v in arguments.items() if v is not None}
                    output = self.tools.call(call.name, arguments)
                result.steps.append(ToolStep(call.name, arguments, output[:300]))
                items.append({"type": "function_call_output", "call_id": call.call_id, "output": _cap(output)})
        result.seconds = round(time.monotonic() - start, 2)
        result.cost_usd = cost_usd(self.model, result.input_tokens, result.output_tokens)
        return result


def _cap(text: str, limit: int = 30_000) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... [truncated {len(text) - limit} characters]"
