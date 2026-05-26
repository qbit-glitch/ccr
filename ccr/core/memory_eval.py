"""Local memory evaluation for CCR projects.

The evaluator is intentionally deterministic and project-local. It does not
judge natural-language answer quality with an LLM; instead it checks whether
recall returns the expected evidence IDs and handles empty/abstention cases.
"""

from __future__ import annotations

import html
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ccr.core.facts import FactLedger
from ccr.core.memory import MemoryManager
from ccr.core.recall import RecallEngine
from ccr.core.types import CCRConfig


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class EvalCase:
    """One deterministic memory recall test case."""

    id: str
    query: str
    expected_source_ids: list[str] = field(default_factory=list)
    min_confidence: float = 0.0
    should_abstain: bool = False
    require_stale_note: bool = False
    require_conflict_note: bool = False
    expected_sources: list[str] = field(default_factory=list)
    category: str = "recall"


@dataclass
class EvalCaseResult:
    """Result for one memory eval case."""

    id: str
    query: str
    passed: bool
    reason: str
    confidence: float
    evidence_ids: list[str]
    category: str = "recall"


@dataclass
class EvalReport:
    """Aggregate memory eval report."""

    project: str
    suite: str
    generated_at: str
    total: int
    passed: int
    failed: int
    accuracy: float
    results: list[EvalCaseResult]
    category_metrics: dict[str, dict[str, float | int]] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    meets_thresholds: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "suite": self.suite,
            "generated_at": self.generated_at,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "accuracy": self.accuracy,
            "category_metrics": self.category_metrics,
            "thresholds": self.thresholds,
            "meets_thresholds": self.meets_thresholds,
            "results": [asdict(r) for r in self.results],
        }

    def to_markdown(self) -> str:
        lines = [
            "# CCR Memory Eval",
            f"Project: {self.project}",
            f"Suite: {self.suite}",
            f"Accuracy: {self.passed}/{self.total} ({self.accuracy:.1%})",
            f"Thresholds: {'PASS' if self.meets_thresholds else 'FAIL'}",
            "",
            "## Category Metrics",
            "| Category | Passed | Total | Accuracy |",
            "|----------|--------|-------|----------|",
        ]
        for category, metrics in sorted(self.category_metrics.items()):
            lines.append(
                f"| {category} | {metrics['passed']} | {metrics['total']} | "
                f"{float(metrics['accuracy']):.1%} |"
            )
        lines.extend([
            "",
            "| Case | Category | Pass | Confidence | Evidence | Reason |",
            "|------|----------|------|------------|----------|--------|",
        ])
        for r in self.results:
            mark = "yes" if r.passed else "no"
            evidence = ", ".join(r.evidence_ids) or "-"
            lines.append(
                f"| {r.id} | {r.category} | {mark} | {r.confidence:.2f} | "
                f"{evidence} | {r.reason} |"
            )
        return "\n".join(lines)

    def to_html(self) -> str:
        rows = []
        for r in self.results:
            rows.append(
                "<tr>"
                f"<td>{html.escape(r.id)}</td>"
                f"<td>{html.escape(r.category)}</td>"
                f"<td>{'PASS' if r.passed else 'FAIL'}</td>"
                f"<td>{r.confidence:.2f}</td>"
                f"<td>{html.escape(', '.join(r.evidence_ids) or '-')}</td>"
                f"<td>{html.escape(r.reason)}</td>"
                "</tr>"
            )
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CCR Memory Eval</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f6f8fa; }}
  </style>
</head>
<body>
  <h1>CCR Memory Eval</h1>
  <p><strong>Project:</strong> {html.escape(self.project)}</p>
  <p><strong>Suite:</strong> {html.escape(self.suite)}</p>
  <p><strong>Accuracy:</strong> {self.passed}/{self.total} ({self.accuracy:.1%})</p>
  <p><strong>Thresholds:</strong> {'PASS' if self.meets_thresholds else 'FAIL'}</p>
  <table>
    <thead><tr><th>Case</th><th>Category</th><th>Pass</th><th>Confidence</th><th>Evidence</th><th>Reason</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""


class MemoryEvalRunner:
    """Run deterministic recall-quality checks against a CCR project."""

    def __init__(self, project_root: str):
        self.project_root = os.path.abspath(project_root)
        storage_backend = os.environ.get("CCR_STORAGE_BACKEND", CCRConfig().storage_backend)
        self.mem = MemoryManager(self.project_root, CCRConfig(storage_backend=storage_backend))
        self.recall = RecallEngine(self.mem)

    def run(
        self,
        suite: str = "smoke",
        limit: int = 10,
        thresholds: dict[str, float] | None = None,
    ) -> EvalReport:
        cases = self._build_cases(suite=suite, limit=limit)
        results = [self._run_case(case) for case in cases]
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        failed = total - passed
        accuracy = passed / total if total else 0.0
        category_metrics = _category_metrics(results)
        thresholds = thresholds or {}
        meets_thresholds = _meets_thresholds(accuracy, category_metrics, thresholds)
        return EvalReport(
            project=self.project_root,
            suite=suite,
            generated_at=_utc_stamp(),
            total=total,
            passed=passed,
            failed=failed,
            accuracy=accuracy,
            results=results,
            category_metrics=category_metrics,
            thresholds=thresholds,
            meets_thresholds=meets_thresholds,
        )

    def _build_cases(self, suite: str, limit: int) -> list[EvalCase]:
        suite = suite.lower()
        if suite not in {"smoke", "temporal", "v2"}:
            raise ValueError("suite must be one of: smoke, temporal, v2")

        cases: list[EvalCase] = []
        branch = self.mem.get_active_branch()
        commits = self.mem._storage.commit_list(branch, limit=max(1, limit))
        for commit in commits:
            cid = str(commit.get("id", ""))
            title = str(commit.get("title", "")).strip()
            if cid and title:
                cases.append(EvalCase(
                    id=f"commit-{cid}",
                    query=f"What memory records {title}?",
                    expected_source_ids=[cid],
                    min_confidence=0.1,
                    category="commit-citation",
                ))

        # Add a temporal-style case for facts when available.
        facts_path = os.path.join(self.mem.ccr_root, "facts.json")
        if suite in {"temporal", "v2"} and os.path.isfile(facts_path):
            ledger = FactLedger(self.mem.ccr_root)
            facts = ledger.list_facts(include_inactive=True, limit=limit)
            for fact in facts[:limit]:
                fid = fact.id
                key = fact.key
                if fid and key:
                    cases.append(EvalCase(
                        id=f"fact-{fid}",
                        query=f"What is the current memory for {key}?",
                        expected_source_ids=[fid],
                        min_confidence=0.1,
                        expected_sources=["fact", "graph"] if suite == "v2" else [],
                        category="fact-citation",
                    ))
                    if suite == "v2" and (fact.valid_to or fact.superseded_by):
                        cases.append(EvalCase(
                            id=f"stale-{fid}",
                            query=f"Is the old fact {fid} stale for {key}?",
                            expected_source_ids=[fid, f"G:{fid}"],
                            min_confidence=0.1,
                            require_stale_note=True,
                            expected_sources=["fact", "graph"],
                            category="stale-handling",
                        ))
            if suite == "v2":
                for conflict in ledger.detect_conflicts(limit=limit):
                    cases.append(EvalCase(
                        id=f"conflict-{conflict.fact_a}-{conflict.fact_b}",
                        query=f"What conflict exists for {conflict.key}?",
                        expected_source_ids=[conflict.fact_a, conflict.fact_b, f"G:{conflict.fact_a}", f"G:{conflict.fact_b}"],
                        min_confidence=0.0,
                        require_conflict_note=True,
                        expected_sources=["fact", "graph"],
                        category="conflict-detection",
                    ))

        cases.append(EvalCase(
            id="abstain-empty-evidence",
            query="zzzz_unlikely_memory_eval_query_zzzz",
            should_abstain=True,
            category="abstention",
        ))
        return cases[: max(1, limit + 1)]

    def _run_case(self, case: EvalCase) -> EvalCaseResult:
        result = self.recall.recall(case.query, limit=5)
        evidence_ids = [ev.id for ev in result.evidence]
        evidence_sources = [ev.source for ev in result.evidence]
        if case.should_abstain:
            passed = not evidence_ids and result.confidence == 0.0
            reason = "abstained with no evidence" if passed else "expected abstention"
        else:
            expected = set(case.expected_source_ids)
            actual = set(evidence_ids)
            has_expected = bool(expected & actual)
            confident = result.confidence >= case.min_confidence
            has_source = True
            if case.expected_sources:
                has_source = bool(set(case.expected_sources) & set(evidence_sources))
            has_stale_note = True
            if case.require_stale_note:
                has_stale_note = bool(result.stale_notes)
            has_conflict_note = True
            if case.require_conflict_note:
                has_conflict_note = bool(result.conflict_notes)
            passed = has_expected and confident and has_source and has_stale_note and has_conflict_note
            if passed:
                reason = "expected evidence returned"
            elif not has_expected:
                reason = f"missing expected evidence: {', '.join(sorted(expected))}"
            elif not has_source:
                reason = f"missing expected source: {', '.join(case.expected_sources)}"
            elif not has_stale_note:
                reason = "missing stale warning"
            elif not has_conflict_note:
                reason = "missing conflict warning"
            else:
                reason = f"confidence below {case.min_confidence:.2f}"
        return EvalCaseResult(
            id=case.id,
            query=case.query,
            passed=passed,
            reason=reason,
            confidence=result.confidence,
            evidence_ids=evidence_ids,
            category=case.category,
        )


def _category_metrics(results: list[EvalCaseResult]) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for result in results:
        row = metrics.setdefault(result.category, {"passed": 0, "total": 0, "accuracy": 0.0})
        row["total"] = int(row["total"]) + 1
        if result.passed:
            row["passed"] = int(row["passed"]) + 1
    for row in metrics.values():
        total = int(row["total"])
        row["accuracy"] = (int(row["passed"]) / total) if total else 0.0
    return metrics


def _meets_thresholds(
    accuracy: float,
    category_metrics: dict[str, dict[str, float | int]],
    thresholds: dict[str, float],
) -> bool:
    for name, threshold in thresholds.items():
        if name in {"accuracy", "overall"}:
            if accuracy < threshold:
                return False
            continue
        category = category_metrics.get(name)
        if category is None:
            return False
        if float(category.get("accuracy", 0.0)) < threshold:
            return False
    return True
