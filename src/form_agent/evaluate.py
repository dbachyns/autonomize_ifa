"""Score reader output against ground truth.

Two scorers:
- `score_typed`: strict field-by-field comparison of a known form type's typed record
  (our hand-labeled Texas forms). Every field counts, including ones that should be
  empty, so hallucinated values are caught.
- `score_pairs`: lenient label->value recall for any form, used for the universal record
  and for external datasets (FUNSD) whose labels are partial. Only labeled pairs count,
  so extra fields the reader finds are not penalized.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from form_agent.normalize import clean, normalize_value
from form_agent.schema import FormRecord, UniversalForm, flatten

# --- value comparison -----------------------------------------------------------------


def _text_key(value: str) -> str:
    return re.sub(r"[\s.,;:]+$", "", re.sub(r"\s+", " ", value)).casefold()


def canonical(value: Any, value_kind: str) -> Any:
    if isinstance(value, list):
        return frozenset(v.casefold() for v in value)
    if isinstance(value, bool) or value is None:
        return value
    value = clean(value)
    if value is None:
        return None
    if value_kind in ("text", "name", "address"):
        return _text_key(value)
    return normalize_value(value, value_kind)


def is_empty(value: Any) -> bool:
    return value in (None, False) or value == frozenset()


# --- typed scoring ----------------------------------------------------------------------


@dataclass
class FieldResult:
    doc_id: str
    path: str
    kind: str
    expected: Any
    predicted: Any
    correct: bool
    expected_empty: bool
    flagged: bool = False


def _align_lines(gt: list[dict], pred: list[dict], kinds: dict[str, str]) -> list[tuple[dict | None, dict | None]]:
    """Greedy one-to-one alignment of table rows by number of matching cells."""

    def matches(a, b):
        return sum(canonical(a[k], kinds[k]) == canonical(b.get(k), kinds[k]) and a[k] is not None for k in a)

    pairs, used = [], set()
    for g in gt:
        best = max((i for i in range(len(pred)) if i not in used), key=lambda i: matches(g, pred[i]), default=None)
        if best is not None and matches(g, pred[best]) > 0:
            used.add(best)
            pairs.append((g, pred[best]))
        else:
            pairs.append((g, None))
    pairs += [(None, pred[i]) for i in range(len(pred)) if i not in used]
    return pairs


def score_typed(doc_id: str, expected: BaseModel, predicted: BaseModel | None, line_attr: str = "services.lines") -> list[FieldResult]:
    """Compare two typed records; table rows under `line_attr` are aligned before comparison."""
    exp_flat = {p: v for p, v in flatten(expected).items() if not p.startswith(line_attr + ".")}
    pred_flat = flatten(predicted) if predicted is not None else {}
    results = []
    for path, (value, value_kind) in exp_flat.items():
        got = pred_flat.get(path, (None, value_kind))[0]
        e, p = canonical(value, value_kind), canonical(got, value_kind)
        results.append(FieldResult(doc_id, path, value_kind, value, got, e == p or (is_empty(e) and is_empty(p)), is_empty(e)))

    def lines(model):
        obj = model
        for part in line_attr.split("."):
            obj = getattr(obj, part)
        return obj

    exp_lines = lines(expected)
    pred_lines = lines(predicted) if predicted is not None else []
    if exp_lines or pred_lines:
        line_kinds = {k: v[1] for k, v in flatten((exp_lines or pred_lines)[0]).items()}
        aligned = _align_lines([l.model_dump() for l in exp_lines], [l.model_dump() for l in pred_lines], line_kinds)
        for i, (g, p) in enumerate(aligned):
            for k, value_kind in line_kinds.items():
                exp_v = g.get(k) if g else None
                got_v = p.get(k) if p else None
                e, q = canonical(exp_v, value_kind), canonical(got_v, value_kind)
                path = f"{line_attr}[{i}].{k}" if g else f"{line_attr}[extra].{k}"
                results.append(FieldResult(doc_id, path, f"table:{value_kind}", exp_v, got_v,
                                           e == q or (is_empty(e) and is_empty(q)), is_empty(e)))
    return results


def _path_key(path: str) -> str:
    """'services.lines[1].service' and 'services.lines.1.service' -> same key."""
    return re.sub(r"\[(\d+)\]", r".\1", path)


# --- lenient pair recall (any form) --------------------------------------------------------


CHECK_GLYPHS = {"☑", "☒", "✓", "✔", "✗", "■", "x", "X"}


def _loose(s: str | None) -> str:
    if s is not None and s.strip() in CHECK_GLYPHS:
        return "x"  # a ticked box, however it was written
    return re.sub(r"[^a-z0-9]", "", (s or "").casefold())


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def universal_pairs(u: UniversalForm, include_printed: bool = True) -> list[tuple[str, str]]:
    """Every (label, value) a universal record asserts: fields, ticked checkboxes, table cells, text."""
    pairs = [(f.label, f.value) for f in u.fields if f.value and (include_printed or f.source == "filled")]
    pairs += [(f"{g.label} {o.label}", "x") for g in u.checkboxes for o in g.options if o.checked]
    for t in u.tables:
        for row in t.rows:
            pairs += [(t.headers[i] if i < len(t.headers) else "", c) for i, c in enumerate(row) if c]
    pairs += [(b.label or "", b.text) for b in u.text_blocks]
    return pairs


@dataclass
class PairResult:
    doc_id: str
    label: str
    expected: str
    matched_label: str | None
    matched_value: str | None
    value_similarity: float
    found: bool


def score_pairs(doc_id: str, expected: list[tuple[str, str]], predicted: list[tuple[str, str]],
                value_threshold: float = 0.85, label_threshold: float = 0.5) -> list[PairResult]:
    """A labeled pair is found if some predicted pair has a similar value AND a similar label.

    Values may also be found inside a longer predicted value (e.g. text blocks).
    """
    pred = [(_loose(l), _loose(v), l, v) for l, v in predicted]
    results = []
    for label, value in expected:
        el, ev = _loose(label), _loose(value)
        best = (0.0, None, None)
        for pl, pv, raw_l, raw_v in pred:
            vs = 1.0 if ev and len(ev) >= 6 and ev in pv else _sim(ev, pv)
            ls = _sim(el, pl) if el else 1.0
            if ls >= label_threshold or (not pl and vs >= 0.95) or (el and el in pl):
                if vs > best[0]:
                    best = (vs, raw_l, raw_v)
        results.append(PairResult(doc_id, label, value, best[1], best[2], round(best[0], 3), best[0] >= value_threshold))
    return results


# --- reports -----------------------------------------------------------------------------


@dataclass
class Report:
    dataset: str
    reader: str
    model: str | None
    documents: int
    missing: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    by_kind: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0
    seconds: float = 0.0

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str) + "\n")
        return path


def _rate(hits: int, total: int) -> dict[str, Any]:
    return {"correct": hits, "total": total, "accuracy": round(hits / total, 4) if total else None}


def summarize_typed(results: list[FieldResult]) -> tuple[dict, dict, list[dict]]:
    filled = [r for r in results if not r.expected_empty]
    empty = [r for r in results if r.expected_empty]
    summary = {
        "all_fields": _rate(sum(r.correct for r in results), len(results)),
        "filled_fields": _rate(sum(r.correct for r in filled), len(filled)),
        "empty_fields_left_empty": _rate(sum(r.correct for r in empty), len(empty)),
        "documents_perfect": None,
    }
    per_doc = defaultdict(list)
    for r in results:
        per_doc[r.doc_id].append(r.correct)
    summary["documents_perfect"] = _rate(sum(all(v) for v in per_doc.values()), len(per_doc))
    by_kind = defaultdict(lambda: [0, 0])
    for r in filled:
        by_kind[r.kind][0] += r.correct
        by_kind[r.kind][1] += 1
    wrong = [r for r in results if not r.correct]
    summary["errors_flagged_for_review"] = _rate(sum(r.flagged for r in wrong), len(wrong))
    errors = [asdict(r) for r in wrong]
    return summary, {k: _rate(*v) for k, v in sorted(by_kind.items())}, errors


def summarize_pairs(results: list[PairResult]) -> tuple[dict, list[dict]]:
    exact = sum(r.value_similarity == 1.0 for r in results)
    summary = {
        "pairs_found": _rate(sum(r.found for r in results), len(results)),
        "pairs_exact": _rate(exact, len(results)),
    }
    return summary, [asdict(r) for r in results if not r.found]


# --- datasets ------------------------------------------------------------------------------


def evaluate_tx(gt_dir: Path, cache_dir: Path, reader: str) -> Report:
    """Typed accuracy + universal pair recall on the hand-labeled Texas forms."""
    from form_agent.forms import get_plugin

    plugin = get_plugin("tx_prior_auth")
    typed_results, pair_results, detected = [], [], 0
    report = Report(dataset="tx_prior_auth", reader=reader, model=None, documents=0)
    for gt_path in sorted(gt_dir.glob("*.json")):
        gt = FormRecord.model_validate_json(gt_path.read_text())
        pred_path = cache_dir / gt.doc_id / f"{reader}.json"
        if not pred_path.exists():
            report.missing.append(gt.doc_id)
            continue
        pred = FormRecord.model_validate_json(pred_path.read_text())
        report.documents += 1
        report.model = pred.run.model
        report.cost_usd += pred.run.cost_usd or 0
        report.seconds += pred.run.seconds or 0
        detected += pred.form_type == plugin.form_type
        typed_pred = plugin.parse(pred.typed) if pred.typed and pred.form_type == plugin.form_type else None
        doc_results = score_typed(gt.doc_id, plugin.parse(gt.typed), typed_pred)
        flagged_paths = {_path_key(f) for fl in pred.flags if fl.severity in ("warning", "error") or
                         fl.rule == "checkbox_corrected" for f in fl.fields}
        for r in doc_results:
            r.flagged = _path_key(r.path) in flagged_paths
        typed_results += doc_results
        pair_results += score_pairs(gt.doc_id, universal_pairs(gt.universal), universal_pairs(pred.universal))

    summary, by_kind, errors = summarize_typed(typed_results)
    pair_summary, _ = summarize_pairs(pair_results)
    report.summary = {**summary, "form_type_detected": _rate(detected, report.documents), "universal": pair_summary}
    report.by_kind = by_kind
    report.errors = errors
    report.cost_usd = round(report.cost_usd, 4)
    report.seconds = round(report.seconds, 1)
    return report


def funsd_pairs(annotation: Path) -> list[tuple[str, str]]:
    """Linked question->answer pairs from a FUNSD annotation file."""
    form = json.loads(annotation.read_text())["form"]
    by_id = {e["id"]: e for e in form}
    pairs = []
    for e in form:
        for q, a in e["linking"]:
            if q == e["id"] and by_id[q]["label"] == "question" and by_id[a]["label"] == "answer":
                if by_id[a]["text"].strip():
                    pairs.append((by_id[q]["text"], by_id[a]["text"]))
    return pairs


def evaluate_funsd(funsd_dir: Path, cache_dir: Path, reader: str) -> Report:
    pair_results, filled_results = [], []
    report = Report(dataset="funsd", reader=reader, model=None, documents=0)
    for ann in sorted((funsd_dir / "annotations").glob("*.json")):
        pred_path = cache_dir / ann.stem / f"{reader}.json"
        if not pred_path.exists():
            report.missing.append(ann.stem)
            continue
        pred = FormRecord.model_validate_json(pred_path.read_text())
        report.documents += 1
        report.model = pred.run.model
        report.cost_usd += pred.run.cost_usd or 0
        report.seconds += pred.run.seconds or 0
        pair_results += score_pairs(ann.stem, funsd_pairs(ann), universal_pairs(pred.universal))
        filled_results += score_pairs(ann.stem, funsd_pairs(ann), universal_pairs(pred.universal, include_printed=False))
    report.summary, report.errors = summarize_pairs(pair_results)
    report.summary["filled_only"] = summarize_pairs(filled_results)[0]
    report.cost_usd = round(report.cost_usd, 4)
    report.seconds = round(report.seconds, 1)
    return report


def format_report(report: Report) -> str:
    lines = [f"== {report.dataset} · reader={report.reader} · model={report.model} · docs={report.documents}"
             f" · cost=${report.cost_usd:.4f} · {report.seconds:.0f}s"]
    if report.missing:
        lines.append(f"   missing predictions for {len(report.missing)} docs")

    def fmt(name, r):
        return f"   {name:<28} {r['accuracy'] * 100:6.1f}%  ({r['correct']}/{r['total']})" if r["total"] else f"   {name:<28} n/a"

    for name, r in report.summary.items():
        if isinstance(r, dict) and "accuracy" in r:
            lines.append(fmt(name, r))
        elif isinstance(r, dict):
            for sub, rr in r.items():
                lines.append(fmt(f"{name}.{sub}", rr))
    if report.by_kind:
        lines.append("   by value type (filled fields):")
        for k, r in report.by_kind.items():
            lines.append("  " + fmt(k, r))
    if report.errors:
        lines.append(f"   first errors ({len(report.errors)} total):")
        for e in report.errors[:15]:
            if "path" in e:
                lines.append(f"     {e['doc_id'][:8]} {e['path']:<34} expected={e['expected']!r} got={e['predicted']!r}")
            else:
                lines.append(f"     {e['doc_id'][:12]} {e['label'][:30]!r:32} expected={e['expected'][:40]!r} "
                             f"best={str(e['matched_value'])[:40]!r} sim={e['value_similarity']}")
    return "\n".join(lines)
