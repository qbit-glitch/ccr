"""Benchmark reports: our measured table and a separate published-numbers table."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import yaml

from ccr.bench.results import QueryResult

_FLAGS = {"excluded", "not_run"}
_ORDER = re.compile(r"^(recall|ndcg)@(\d+)$")


@dataclass(frozen=True)
class SystemSummary:
    system: str
    n_queries: int
    n_excluded: int
    n_not_run: int
    empty_rate: float
    mean_tokens: float
    ingest_s: float
    mean_retrieve_s: float
    metrics: dict[str, float]
    by_category: dict[str, dict[str, float]]


@dataclass(frozen=True)
class PublishedEntry:
    system: str
    dataset: str
    metric: str
    value: float
    source: str
    locator: str
    protocol_note: str


def _means(rows: Sequence[QueryResult]) -> dict[str, float]:
    sums: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.scores.items():
            if key not in _FLAGS:
                sums[key].append(value)
    return {k: sum(v) / len(v) for k, v in sums.items()}


def _summarize(system: str, rows: Sequence[QueryResult]) -> SystemSummary:
    scored = [r for r in rows if "excluded" not in r.scores]
    by_cat: dict[str, list[QueryResult]] = defaultdict(list)
    for r in scored:
        by_cat[r.category].append(r)
    n = len(scored)
    return SystemSummary(
        system=system,
        n_queries=n,
        n_excluded=len(rows) - n,
        n_not_run=sum(1 for r in scored if "not_run" in r.scores),
        empty_rate=sum(r.empty for r in scored) / n if n else 0.0,
        mean_tokens=sum(r.retrieved_tokens for r in scored) / n if n else 0.0,
        ingest_s=sum(r.ingest_s for r in scored),
        mean_retrieve_s=sum(r.retrieve_s for r in scored) / n if n else 0.0,
        metrics=_means(scored),
        by_category={c: _means(rs) for c, rs in sorted(by_cat.items())},
    )


def _metric_sort(name: str) -> tuple[int, int, str]:
    match = _ORDER.match(name)
    if match:
        return (0 if match.group(1) == "recall" else 1, int(match.group(2)), name)
    return (2, 0, name)


@dataclass(frozen=True)
class BenchReport:
    meta: dict[str, str]
    systems: tuple[SystemSummary, ...]

    def to_json(self) -> str:
        return json.dumps(
            {"meta": self.meta, "systems": [asdict(s) for s in self.systems]}, indent=2
        )

    def to_markdown(self, published: Sequence[PublishedEntry] = ()) -> str:
        pilot = self.meta.get("pilot") == "true"
        title = f"# Benchmark: {self.meta.get('dataset', '?')}"
        title += " — pilot (10%)" if pilot else ""
        cols = sorted({m for s in self.systems for m in s.metrics}, key=_metric_sort)
        head = ["system", "n", *cols, "tokens", "empty%"]
        lines = [title, "", "## Table 1: measured under identical conditions", "",
                 "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for s in self.systems:
            cells = [s.system, str(s.n_queries)]
            cells += [f"{s.metrics[c]:.3f}" if c in s.metrics else "-" for c in cols]
            cells += [f"{s.mean_tokens:.0f}", f"{100 * s.empty_rate:.1f}"]
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", "Run details: " + ", ".join(f"{k}={v}" for k, v in self.meta.items())]
        dataset = self.meta.get("dataset")
        rows = [p for p in published if p.dataset == dataset]
        if rows:
            lines += ["", "## Table 2: reported by the dataset authors (different protocol)",
                      "", "| system | metric | value | source | locator | note |",
                      "|---|---|---|---|---|---|"]
            for p in rows:
                lines.append(f"| {p.system} | {p.metric} | {p.value} | {p.source} | "
                             f"{p.locator} | {p.protocol_note} |")
        return "\n".join(lines) + "\n"


def build_report(results: Sequence[QueryResult], meta: dict[str, str]) -> BenchReport:
    """Aggregate per-query results into one summary per system."""
    grouped: dict[str, list[QueryResult]] = defaultdict(list)
    for r in results:
        grouped[r.system].append(r)
    return BenchReport(meta=dict(meta),
                       systems=tuple(_summarize(s, rows) for s, rows in sorted(grouped.items())))


def load_published(path: str) -> list[PublishedEntry]:
    """Load published numbers; every entry must name its source and locator."""
    with open(path, encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    entries: list[PublishedEntry] = []
    for raw in data.get("entries") or []:
        for field in ("source", "locator"):
            if not str(raw.get(field, "")).strip():
                raise ValueError(f"published entry {raw.get('system')!r} missing {field}")
        entries.append(PublishedEntry(
            str(raw["system"]), str(raw["dataset"]), str(raw["metric"]),
            float(raw["value"]), str(raw["source"]), str(raw["locator"]),
            str(raw.get("protocol_note", "")),
        ))
    return entries
