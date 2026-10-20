"""Read-only tools the Q&A agent uses to look at extracted forms.

They work on any form type: everything goes through the universal tables, and known
form types add their typed tables (e.g. tx_prior_auth). No tool can change data: SQL runs
on a read-only connection, and only a single SELECT/WITH statement is accepted.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from form_agent.config import settings
from form_agent.layout import load_layout
from form_agent.schema import FormRecord
from form_agent.store import connect_readonly, load_record, record_path
from form_agent.view import DocView, build_view, render_text

MAX_SQL_STEPS = 5_000_000  # SQLite VM steps before a query is aborted (runaway joins)

# Descriptions of the core tables for the agent's system prompt. Typed tables are described
# automatically from their columns.
CORE_TABLES = {
    "documents": "one row per form. form_type is 'unknown' for forms without a typed schema",
    "active_fields": "labeled fields, verbatim `value` plus `value_norm` (ISO dates, digit-only phones/NPIs). "
                     "source='filled' was entered on the form, 'printed' is pre-printed on it; template=1 marks "
                     "printed values that are identical on every form of the type",
    "active_checkboxes": "one row per checkbox option; checked is 0/1",
    "active_table_cells": "one row per table cell. (doc_id, table_index) is one table, table_name its printed "
                          "title or heading, row_index the row, column its header. To read whole rows, self-join on "
                          "(doc_id, table_index, row_index); value_norm holds numbers/dates normalized",
    "text_blocks": "free text written on forms (join extractions on run_id with active=1)",
    "active_flags": "automatic review flags: rule, severity (info|warning|error), message",
    "form_type_constants": "printed values shared by all forms of a type (e.g. the insurer's fax numbers)",
}


class ToolError(Exception):
    """An error the model should see and recover from (bad doc id, bad SQL, ...)."""


class FormTools:
    def __init__(self, db_path: Path | None = None, cache_dir: Path | None = None, max_rows: int = 100):
        self.db_path = Path(db_path or settings.db_path)
        self.cache_dir = Path(cache_dir or settings.cache_dir)
        if not self.db_path.exists():  # a fresh clone: the index is rebuilt from the JSON records in seconds
            if not self.cache_dir.exists():
                raise FileNotFoundError(f"{self.db_path} not found and no records in {self.cache_dir}; "
                                        "read some forms first: python -m form_agent extract <folder>")
            from form_agent.store import build_index

            build_index(self.cache_dir, self.db_path)
        self._local = threading.local()
        self.max_rows = max_rows
        self.doc_ids = [r[0] for r in self.con.execute("SELECT doc_id FROM documents ORDER BY doc_id")]
        self.short = _short_ids(self.doc_ids)

    @property
    def con(self) -> sqlite3.Connection:
        """One read-only connection per thread (sqlite connections can't be shared across threads)."""
        if not hasattr(self._local, "con"):
            self._local.con = connect_readonly(self.db_path)
        return self._local.con

    # --- doc ids ---------------------------------------------------------------------------

    def resolve(self, doc_id: str) -> str:
        """Full doc id from a full id, short id, or unique prefix."""
        doc_id = doc_id.strip().strip("[]")
        if doc_id in self.doc_ids:
            return doc_id
        matches = [d for d in self.doc_ids if d.startswith(doc_id)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ToolError(f"No document '{doc_id}'. Use list_documents to see the ids.")
        raise ToolError(f"'{doc_id}' is ambiguous: {', '.join(self.short[d] for d in matches[:10])}")

    # --- tools -----------------------------------------------------------------------------

    def list_documents(self, form_type: str | None = None) -> list[dict[str, Any]]:
        sql = """SELECT d.doc_id, d.form_type, d.title, d.page_count,
                        SUM(g.severity = 'error') AS errors, SUM(g.severity = 'warning') AS warnings
                 FROM documents d LEFT JOIN active_flags g USING (doc_id)
                 WHERE (? IS NULL OR d.form_type = ?) GROUP BY d.doc_id ORDER BY d.doc_id"""
        return [
            {"id": self.short[r["doc_id"]], "form_type": r["form_type"], "title": r["title"],
             "pages": r["page_count"], "errors": r["errors"] or 0, "warnings": r["warnings"] or 0}
            for r in self.con.execute(sql, (form_type, form_type))
        ]

    def get_document(self, doc_id: str) -> str:
        """The whole form as readable text: fields, checkboxes, tables, free text, review flags."""
        return render_text(self.view(doc_id))

    def view(self, doc_id: str) -> DocView:
        """The active record of a form with positions and confidence (layout from the cache, if computed)."""
        record = self.load_active(self.resolve(doc_id))
        layout = load_layout(self.cache_dir, record.doc_id)
        return build_view(record, layout, self.short.get(record.doc_id, record.doc_id))

    def search(self, query: str, doc_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text search over values, labels and free text of all forms (or one)."""
        words = re.findall(r"\w+", query)
        if not words:
            raise ToolError("Empty search query.")
        doc = self.resolve(doc_id) if doc_id else None
        rows: list[sqlite3.Row] = []
        for joiner in (" ", " OR "):  # all words first, then any word
            match = joiner.join(f'"{w}"' for w in words)
            rows = self.con.execute(
                "SELECT doc_id, kind, label, text FROM search WHERE search MATCH ? AND (? IS NULL OR doc_id = ?)"
                " ORDER BY rank LIMIT ?", (match, doc, doc, limit)).fetchall()
            if rows or len(words) == 1:
                break
        return [{"id": self.short[r["doc_id"]], "kind": r["kind"], "label": r["label"], "text": r["text"][:300]}
                for r in rows]

    def sql(self, query: str) -> dict[str, Any]:
        """Run one read-only SELECT. Returns columns, rows (capped at max_rows) and a truncation note."""
        q = query.strip().rstrip(";").strip()
        if not re.match(r"(?is)^(select|with)\b", q) or ";" in q:
            raise ToolError("Only a single SELECT (or WITH ... SELECT) statement is allowed.")
        steps = 0

        def guard() -> int:
            nonlocal steps
            steps += 1000
            return int(steps > MAX_SQL_STEPS)  # non-zero aborts the query

        self.con.set_progress_handler(guard, 1000)
        try:
            cur = self.con.execute(q)
            rows = cur.fetchmany(self.max_rows + 1)
        except sqlite3.Error as exc:
            raise ToolError(f"SQL error: {exc}") from None
        finally:
            self.con.set_progress_handler(None, 0)
        columns = [c[0] for c in cur.description or []]
        out: dict[str, Any] = {"columns": columns, "rows": [list(r) for r in rows[: self.max_rows]]}
        if len(rows) > self.max_rows:
            out["note"] = f"truncated to {self.max_rows} rows; aggregate or add LIMIT/WHERE"
        return out

    # --- dispatch for the LLM -------------------------------------------------------------

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool by name; errors come back as text so the model can correct itself."""
        fn = {"list_documents": self.list_documents, "get_document": self.get_document,
              "search": self.search, "sql": self.sql}.get(name)
        try:
            if fn is None:
                raise ToolError(f"Unknown tool '{name}'.")
            result = fn(**arguments)
        except ToolError as exc:
            return json.dumps({"error": str(exc)})
        except TypeError as exc:  # wrong/missing arguments
            return json.dumps({"error": f"Bad arguments for {name}: {exc}"})
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)

    # --- helpers ---------------------------------------------------------------------------

    def load_active(self, doc_id: str) -> FormRecord:
        row = self.con.execute("SELECT reader FROM extractions WHERE doc_id = ? AND active = 1", (doc_id,)).fetchone()
        if row is None:
            raise ToolError(f"No extraction for '{doc_id}'.")
        return load_record(record_path(self.cache_dir, doc_id, row["reader"]))

    def overview(self) -> str:
        counts = self.con.execute("SELECT form_type, COUNT(*) FROM documents GROUP BY 1 ORDER BY 2 DESC").fetchall()
        return ", ".join(f"{n} × {t}" for t, n in counts) or "no documents"

    def schema_description(self) -> str:
        """Tables and columns for the system prompt; typed tables list their small value sets."""
        lines = []
        for table, note in CORE_TABLES.items():
            lines.append(f"- {table}({', '.join(self._columns(table))}): {note}")
        typed = [r[0] for r in self.con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT IN "
            "('documents','extractions','fields','checkboxes','table_cells','text_blocks','flags',"
            "'form_type_constants') AND name NOT LIKE 'search%'")]
        for table in typed:
            cols = []
            for col in self._columns(table):
                values = [r[0] for r in self.con.execute(
                    f'SELECT DISTINCT "{col}" FROM {table} WHERE "{col}" IS NOT NULL AND "{col}" != "" LIMIT 7')]
                cols.append(f"{col} [{' | '.join(map(str, values))}]" if 0 < len(values) <= 6 and col != "doc_id"
                            else col)
            lines.append(f"- {table}({', '.join(cols)}): typed values for a known form type; dates ISO, "
                         "phones/NPIs digits only, multi-select checkboxes joined with ', '")
        return "\n".join(lines)

    def _columns(self, table: str) -> list[str]:
        return [r[1] for r in self.con.execute(f"PRAGMA table_info({table})")]


def _short_ids(doc_ids: list[str]) -> dict[str, str]:
    """Random-looking ids (uuids, hashes) are shortened to their first 8 characters when unique;
    readable names ("expense_report") are kept whole."""
    prefixes: dict[str, int] = {}
    for d in doc_ids:
        prefixes[d[:8]] = prefixes.get(d[:8], 0) + 1
    return {d: d[:8] if len(d) > 12 and re.fullmatch(r"[0-9a-f]{8}", d[:8]) and prefixes[d[:8]] == 1 else d
            for d in doc_ids}
