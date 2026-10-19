"""Score the Q&A agent on a question set with known answers (e.g. data/qa/tx_questions.json).

Grading is deterministic: every requirement in `expect` must appear in the answer (any of
its alternatives), and nothing in `forbid` may appear. Long digit strings are compared
against the answer's digits only, so "866-617-4967" and "(866) 617-4967" both match.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from form_agent.agent import AgentAnswer, FormAgent


@dataclass
class QAResult:
    id: str
    kind: str
    question: str
    passed: bool
    missing: list[list[str]]
    forbidden: list[str]
    answer: str
    tools: list[str]
    seconds: float
    cost_usd: float | None


@dataclass
class QAReport:
    model: str
    results: list[QAResult] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for kind in ["single", "multi", "summary", "all"]:
            rs = [r for r in self.results if kind == "all" or r.kind == kind]
            if rs:
                out[kind] = {"passed": sum(r.passed for r in rs), "total": len(rs)}
        costs = [r.cost_usd for r in self.results if r.cost_usd is not None]
        out["cost_usd"] = round(sum(costs), 4) if costs else None
        out["avg_seconds"] = round(sum(r.seconds for r in self.results) / max(len(self.results), 1), 1)
        out["avg_tool_calls"] = round(sum(len(r.tools) for r in self.results) / max(len(self.results), 1), 1)
        return out

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": self.model, "summary": self.summary(),
                                    "results": [asdict(r) for r in self.results]}, indent=2, ensure_ascii=False) + "\n")
        return path

    def format(self) -> str:
        s = self.summary()
        lines = [f"== Q&A · model={self.model} · questions={len(self.results)} · cost=${s['cost_usd'] or 0:.4f}"
                 f" · avg {s['avg_seconds']}s, {s['avg_tool_calls']} tool calls"]
        for kind in ["single", "multi", "summary", "all"]:
            if kind in s:
                p, t = s[kind]["passed"], s[kind]["total"]
                lines.append(f"   {kind:<10} {p}/{t}  ({100 * p / t:.0f}%)")
        for r in self.results:
            if not r.passed:
                why = [f"missing {m}" for m in r.missing] + [f"contains '{f}'" for f in r.forbidden]
                lines.append(f"   FAIL {r.id} {r.question[:60]!r}: {'; '.join(why)}")
                lines.append(f"        answer: {r.answer[:300]!r}")
        return "\n".join(lines)


def _normalize(text: str) -> str:
    text = text.lower().replace("’", "'").replace("**", "").replace("`", "")
    return re.sub(r"\s+", " ", text)


def contains(answer: str, alternative: str) -> bool:
    text, alt = _normalize(answer), alternative.lower()
    if alt.isdigit() and len(alt) >= 5:
        return alt in re.sub(r"\D", "", text) or alt in text
    if re.fullmatch(r"[\w.]+", alt):
        # Word start must match ("male" never matches "female"); numbers must also end there ("3" not "30"),
        # words may continue ("npi" matches "NPIs").
        end = r"(?![\w])" if alt[-1].isdigit() else ""
        return re.search(rf"(?<![\w.]){re.escape(alt)}{end}", text) is not None
    return alt in text


def grade(item: dict[str, Any], result: AgentAnswer) -> QAResult:
    missing = [alts for alts in item["expect"] if not any(contains(result.answer, a) for a in alts)]
    forbidden = [f for f in item.get("forbid", []) if contains(result.answer, f)]
    return QAResult(
        id=item["id"], kind=item["kind"], question=item.get("question") or f"summarize {item['doc']}",
        passed=not missing and not forbidden, missing=missing, forbidden=forbidden, answer=result.answer,
        tools=[s.tool for s in result.steps], seconds=result.seconds, cost_usd=result.cost_usd,
    )


def run_qa_eval(questions_path: Path, agent: FormAgent, workers: int = 4, only: list[str] | None = None) -> QAReport:
    items = json.loads(questions_path.read_text())["questions"]
    if only:
        items = [q for q in items if q["id"] in only]

    def run(item: dict[str, Any]) -> QAResult:
        if item["kind"] == "summary":
            return grade(item, agent.summarize(item["doc"]))
        return grade(item, agent.ask(item["question"], doc_id=item.get("doc")))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, items))
    return QAReport(model=agent.model, results=results)
