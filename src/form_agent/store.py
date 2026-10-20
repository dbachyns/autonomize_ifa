"""Storage: per-document JSON records (source of truth) and a rebuildable SQLite index.

Layout:
    data/cache/<doc_id>/<reader>.json   one FormRecord per document per reader run
    data/forms.db                       SQLite index built from the cache

The index is disposable: `build_index` drops and recreates it from the JSON files, so
schema changes never require re-reading scans.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from form_agent.forms import get_plugin
from form_agent.normalize import label_to_key, normalize_value
from form_agent.schema import FormRecord

# Which reader's output the agent answers from when several exist for a document.
# A human-reviewed version always wins; ground truth is only a fallback (it is for evaluation).
DEFAULT_PREFERENCE = ["reviewed", "hybrid", "openai", "claude", "ollama", "ground_truth"]


# --- JSON cache --------------------------------------------------------------------------


def record_path(cache_dir: Path, doc_id: str, reader: str) -> Path:
    return cache_dir / doc_id / f"{reader}.json"


def save_record(cache_dir: Path, record: FormRecord) -> Path:
    path = record_path(cache_dir, record.doc_id, record.run.reader)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + "\n")
    return path


def load_record(path: Path) -> FormRecord:
    return FormRecord.model_validate_json(path.read_text())


def iter_records(cache_dir: Path) -> Iterable[FormRecord]:
    for path in sorted(cache_dir.glob("*/*.json")):
        if not path.name.startswith("_"):  # _layout.json etc. are not records
            yield load_record(path)


def pick_active(records: list[FormRecord], preference: list[str]) -> FormRecord:
    rank = {r: i for i, r in enumerate(preference)}
    return min(records, key=lambda r: (rank.get(r.run.reader, len(rank)), -r.run.created_at.timestamp()))


# --- SQLite index ------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    source_file TEXT NOT NULL,
    form_type TEXT NOT NULL,
    form_type_confidence REAL,
    title TEXT,
    page_count INTEGER
);
CREATE TABLE extractions (
    run_id INTEGER PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    reader TEXT NOT NULL,
    model TEXT,
    prompt_version TEXT,
    created_at TEXT,
    seconds REAL,
    cost_usd REAL,
    active INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE fields (
    run_id INTEGER NOT NULL REFERENCES extractions(run_id),
    doc_id TEXT NOT NULL,
    page INTEGER,
    section TEXT,
    label TEXT NOT NULL,
    key TEXT,
    value TEXT,
    value_norm TEXT,
    value_type TEXT,
    source TEXT,              -- 'filled' (entered on the form) or 'printed' (part of the template)
    confidence REAL,
    template INTEGER DEFAULT 0 -- 1: printed value identical across this form type (see form_type_constants)
);
CREATE TABLE checkboxes (
    run_id INTEGER NOT NULL REFERENCES extractions(run_id),
    doc_id TEXT NOT NULL,
    page INTEGER,
    section TEXT,
    group_label TEXT NOT NULL,
    group_key TEXT,
    option TEXT NOT NULL,
    checked INTEGER NOT NULL,
    confidence REAL,
    method TEXT
);
CREATE TABLE table_cells (
    run_id INTEGER NOT NULL REFERENCES extractions(run_id),
    doc_id TEXT NOT NULL,
    page INTEGER,
    table_index INTEGER NOT NULL,
    table_name TEXT,
    row_index INTEGER NOT NULL,
    column TEXT,
    value TEXT,
    value_norm TEXT,
    value_type TEXT
);
CREATE TABLE text_blocks (
    run_id INTEGER NOT NULL REFERENCES extractions(run_id),
    doc_id TEXT NOT NULL,
    page INTEGER,
    section TEXT,
    label TEXT,
    text TEXT NOT NULL
);
CREATE TABLE flags (
    run_id INTEGER NOT NULL REFERENCES extractions(run_id),
    doc_id TEXT NOT NULL,
    rule TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    fields TEXT
);
CREATE INDEX idx_fields_key ON fields(key, value_norm);
CREATE INDEX idx_fields_doc ON fields(doc_id);
CREATE INDEX idx_checkboxes ON checkboxes(group_key, option, checked);
CREATE VIRTUAL TABLE search USING fts5(doc_id UNINDEXED, kind UNINDEXED, label, text);

-- Convenience views over the active extraction of each document.
CREATE VIEW active_fields AS
    SELECT f.* FROM fields f JOIN extractions e USING (run_id) WHERE e.active = 1;
CREATE VIEW active_checkboxes AS
    SELECT c.* FROM checkboxes c JOIN extractions e USING (run_id) WHERE e.active = 1;
CREATE VIEW active_table_cells AS
    SELECT t.* FROM table_cells t JOIN extractions e USING (run_id) WHERE e.active = 1;
CREATE VIEW active_flags AS
    SELECT g.* FROM flags g JOIN extractions e USING (run_id) WHERE e.active = 1;
"""


# Printed values repeated on (nearly) every form of a known type are template constants,
# e.g. the insurer's fax numbers. They are marked and collected once per form type so
# cross-form statistics don't count them as data.
TEMPLATE_SQL = """
CREATE TEMP TABLE _printed AS
    SELECT f.rowid AS rid, d.form_type, f.label, f.value, f.doc_id
    FROM fields f JOIN extractions e USING (run_id) JOIN documents d ON d.doc_id = f.doc_id
    WHERE e.active = 1 AND f.source = 'printed' AND d.form_type != 'unknown' AND f.value IS NOT NULL;
-- Matched by value, not label: the same printed number gets labeled differently form to form.
CREATE TEMP TABLE _constants AS
    SELECT p.form_type, p.value, COUNT(DISTINCT p.doc_id) AS documents FROM _printed p
    GROUP BY p.form_type, p.value
    HAVING COUNT(DISTINCT p.doc_id) >= 3
       AND COUNT(DISTINCT p.doc_id) >= 0.5 * (SELECT COUNT(*) FROM documents d WHERE d.form_type = p.form_type);
-- A printed field is template text if it is, or contains, a constant (some readers lump several together).
UPDATE fields SET template = 1 WHERE rowid IN (
    SELECT p.rid FROM _printed p JOIN _constants c
      ON c.form_type = p.form_type AND (p.value = c.value OR instr(p.value, c.value) > 0));
CREATE TABLE form_type_constants AS
    SELECT c.form_type,
           (SELECT p.label FROM _printed p WHERE p.form_type = c.form_type AND p.value = c.value
            GROUP BY p.label ORDER BY COUNT(*) DESC, LENGTH(p.label) DESC LIMIT 1) AS label,
           c.value, c.documents
    FROM _constants c ORDER BY c.form_type, c.documents DESC;
"""


def build_index(cache_dir: Path, db_path: Path, preference: list[str] | None = None) -> dict[str, int]:
    """(Re)build the SQLite index from every record in the cache. Returns row counts."""
    preference = preference or DEFAULT_PREFERENCE
    by_doc: dict[str, list[FormRecord]] = {}
    for rec in iter_records(cache_dir):
        by_doc.setdefault(rec.doc_id, []).append(rec)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    typed_rows: dict[str, list[dict[str, Any]]] = {}

    for doc_id, recs in sorted(by_doc.items()):
        active = pick_active(recs, preference)
        con.execute(
            "INSERT INTO documents VALUES (?,?,?,?,?,?)",
            (doc_id, active.source_file, active.form_type, active.form_type_confidence,
             active.universal.title, active.universal.page_count),
        )
        for rec in recs:
            _insert_record(con, rec, rec is active)
        if active.typed and (plugin := get_plugin(active.form_type)):
            for table, rows in plugin.table_rows(doc_id, plugin.parse(active.typed)).items():
                typed_rows.setdefault(table, []).extend(rows)

    for table, rows in typed_rows.items():
        _write_typed_table(con, table, rows)
    con.executescript(TEMPLATE_SQL)
    con.commit()
    counts = {
        t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ["documents", "extractions", "fields", "checkboxes", "table_cells", "text_blocks", "flags",
                  "form_type_constants", *typed_rows]
    }
    con.close()
    return counts


def _insert_record(con: sqlite3.Connection, rec: FormRecord, active: bool) -> int:
    run = rec.run
    cur = con.execute(
        "INSERT INTO extractions (doc_id, reader, model, prompt_version, created_at, seconds, cost_usd, active)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (rec.doc_id, run.reader, run.model, run.prompt_version, run.created_at.isoformat(), run.seconds,
         run.cost_usd, int(active)),
    )
    run_id = cur.lastrowid
    u, doc = rec.universal, rec.doc_id

    for f in u.fields:
        key = f.key or label_to_key(f.label)
        source = f.source if f.value else "filled"  # an empty blank is always a fill-in field
        con.execute(
            "INSERT INTO fields VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, doc, f.page, f.section, f.label, key, f.value, normalize_value(f.value, f.value_type),
             f.value_type, source, f.confidence, 0),
        )
        if active and f.value:
            kind = "printed" if source == "printed" else "field"
            con.execute("INSERT INTO search VALUES (?,?,?,?)", (doc, kind, f.label, f.value))

    for g in u.checkboxes:
        key = g.key or label_to_key(g.label)
        for o in g.options:
            con.execute(
                "INSERT INTO checkboxes VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run_id, doc, g.page, g.section, g.label, key, o.label, int(o.checked), o.confidence, o.method),
            )
        if active and g.selected:
            con.execute("INSERT INTO search VALUES (?,?,?,?)", (doc, "checkbox", g.label, ", ".join(g.selected)))

    for ti, t in enumerate(u.tables):
        for ri, row in enumerate(t.rows):
            for ci, value in enumerate(row):
                vtype = t.column_types[ci] if ci < len(t.column_types) else "text"
                column = t.headers[ci] if ci < len(t.headers) else f"col{ci + 1}"
                con.execute(
                    "INSERT INTO table_cells VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (run_id, doc, t.page, ti, t.name, ri, column, value, normalize_value(value, vtype), vtype),
                )
                if active and value:
                    con.execute("INSERT INTO search VALUES (?,?,?,?)", (doc, "table", column, value))

    for b in u.text_blocks:
        con.execute("INSERT INTO text_blocks VALUES (?,?,?,?,?,?)", (run_id, doc, b.page, b.section, b.label, b.text))
        if active:
            con.execute("INSERT INTO search VALUES (?,?,?,?)", (doc, "text", b.label or "", b.text))

    for fl in rec.flags:
        con.execute(
            "INSERT INTO flags VALUES (?,?,?,?,?,?)",
            (run_id, doc, fl.rule, fl.severity, fl.message, json.dumps(fl.fields)),
        )
    return run_id


def _write_typed_table(con: sqlite3.Connection, table: str, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(c for row in rows for c in row))
    quoted = ", ".join('"' + c + '"' for c in columns)
    con.execute(f"CREATE TABLE {table} ({quoted})")
    placeholders = ",".join("?" for _ in columns)
    con.executemany(
        f"INSERT INTO {table} VALUES ({placeholders})",
        [tuple(_sql_value(row.get(c)) for c in columns) for row in rows],
    )


def _sql_value(value: Any) -> Any:
    return int(value) if isinstance(value, bool) else value


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Read-only connection for the agent's SQL tool: writes raise sqlite3.OperationalError."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con
