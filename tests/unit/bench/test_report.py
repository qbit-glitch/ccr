"""Tests for report building and the published-numbers table."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ccr.bench.report import PublishedEntry, build_report, load_published
from ccr.bench.results import QueryResult


def _r(qid: str, system: str, scores: dict[str, float], *, cat: str = "c1",
       tokens: int = 10, empty: bool = False, world: str = "w") -> QueryResult:
    return QueryResult(qid, world, system, cat, ("d1",), tokens, None, scores,
                       empty, 0.5, 0.1)


META = {"dataset": "locomo", "pilot": "true", "reader": "qwen3.8:27b", "seed": "1"}


def test_means_empty_rate_and_tokens_per_system() -> None:
    results = [
        _r("q1", "bm25", {"recall@5": 1.0}, tokens=10),
        _r("q2", "bm25", {"recall@5": 0.0}, tokens=30, empty=True),
        _r("q1", "ccr", {"recall@5": 1.0}, tokens=5),
    ]
    rep = build_report(results, META)
    bm25 = {s.system: s for s in rep.systems}["bm25"]
    assert bm25.metrics["recall@5"] == 0.5
    assert bm25.n_queries == 2
    assert bm25.empty_rate == 0.5
    assert bm25.mean_tokens == 20.0


def test_excluded_and_not_run_are_counted_not_averaged() -> None:
    results = [
        _r("q1", "bm25", {"recall@5": 1.0}),
        _r("q2", "bm25", {"excluded": 1.0}),
        _r("q3", "bm25", {"not_run": 1.0, "recall@5": 0.0}),
    ]
    s = build_report(results, META).systems[0]
    assert s.n_excluded == 1 and s.n_not_run == 1
    assert s.n_queries == 2
    assert "excluded" not in s.metrics and "not_run" not in s.metrics
    assert s.metrics["recall@5"] == 0.5


def test_per_category_means() -> None:
    results = [
        _r("q1", "bm25", {"recall@5": 1.0}, cat="a"),
        _r("q2", "bm25", {"recall@5": 0.0}, cat="b"),
    ]
    s = build_report(results, META).systems[0]
    assert s.by_category["a"]["recall@5"] == 1.0
    assert s.by_category["b"]["recall@5"] == 0.0


def test_pilot_label_in_title_only_when_pilot() -> None:
    results = [_r("q1", "bm25", {"recall@5": 1.0})]
    assert "pilot (10%)" in build_report(results, META).to_markdown()
    full = build_report(results, {**META, "pilot": "false"}).to_markdown()
    assert "pilot" not in full.splitlines()[0]


def test_json_round_trips_and_carries_meta() -> None:
    data = json.loads(build_report([_r("q1", "bm25", {"recall@5": 1.0})], META).to_json())
    assert data["meta"]["reader"] == "qwen3.8:27b"
    assert data["systems"][0]["system"] == "bm25"


def test_published_table_is_separate_from_ours() -> None:
    pub = [PublishedEntry("Contriever", "locomo", "recall@5", 0.42,
                          "Paper X", "Table 3", "different reader")]
    md = build_report([_r("q1", "bm25", {"recall@5": 1.0})], META).to_markdown(pub)
    assert md.index("Table 1") < md.index("Table 2")
    table1, table2 = md.split("Table 2")
    assert "Contriever" not in table1 and "Contriever" in table2
    assert "Paper X" in table2 and "Table 3" in table2


def test_published_for_other_datasets_are_not_shown() -> None:
    pub = [PublishedEntry("Other", "beir-scifact", "ndcg@10", 0.6, "P", "T1", "")]
    md = build_report([_r("q1", "bm25", {"recall@5": 1.0})], META).to_markdown(pub)
    assert "Other" not in md


def test_load_published_requires_source_and_locator(tmp_path) -> None:
    good = tmp_path / "ok.yaml"
    good.write_text(
        "entries:\n- {system: S, dataset: d, metric: m, value: 0.5, source: P, "
        "locator: T1, protocol_note: n}\n"
    )
    assert load_published(str(good))[0].value == 0.5
    for missing in ("source", "locator"):
        bad = tmp_path / f"bad_{missing}.yaml"
        fields = {"system": "S", "dataset": "d", "metric": "m", "value": 0.5,
                  "source": "P", "locator": "T1", "protocol_note": "n"}
        fields[missing] = ""
        bad.write_text("entries:\n- " + json.dumps(fields) + "\n")
        with pytest.raises(ValueError, match=missing):
            load_published(str(bad))


def test_shipped_published_file_loads_and_every_entry_is_sourced() -> None:
    path = Path(__file__).resolve().parents[3] / "docs" / "bench" / "published.yaml"
    entries = load_published(str(path))
    assert isinstance(entries, list)
    assert all(e.source and e.locator for e in entries)
