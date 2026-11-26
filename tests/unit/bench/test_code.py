"""Tests for the code-retrieval adapter (mteb/CodeSearchNetRetrieval, python)."""

from __future__ import annotations

import pytest

from ccr.bench import datasets
from ccr.bench.datasets import code
from ccr.bench.datasets.code import LOAD_STATS, parse_code, world_gains

CORPUS = [
    {"id": "https://github.com/a/b/blob/1/x.py", "text": "def f():\n    pass", "title": ""},
    {"id": "https://github.com/a/b/blob/1/y.py", "text": "def g():\n    pass", "title": "T"},
    {"id": "https://github.com/a/b/blob/1/z.py", "text": "def h(): pass", "title": ""},
]
QUERIES = [{"id": "0", "text": "does f"}, {"id": "1", "text": "does g"}, {"id": "2", "text": "orphan"}]
QRELS = [
    {"query-id": "0", "corpus-id": "https://github.com/a/b/blob/1/x.py", "score": "1"},
    {"query-id": "1", "corpus-id": "https://github.com/a/b/blob/1/y.py", "score": "1"},
    {"query-id": "1", "corpus-id": "https://github.com/a/b/blob/1/z.py", "score": "0"},
    {"query-id": "9", "corpus-id": "https://github.com/a/b/blob/1/ghost.py", "score": "1"},
]


def test_docs_keep_ids_and_join_nonempty_titles() -> None:
    world = parse_code("code", CORPUS, QUERIES, QRELS)
    docs = {d.id: d.text for d in world.docs}
    assert docs["https://github.com/a/b/blob/1/x.py"] == "def f():\n    pass"
    assert docs["https://github.com/a/b/blob/1/y.py"] == "T\ndef g():\n    pass"


def test_gold_from_positive_qrels_and_unjudged_queries_dropped() -> None:
    world = parse_code("code", CORPUS, QUERIES, QRELS)
    by_id = {q.id: q for q in world.queries}
    assert set(by_id) == {"0", "1"}
    assert by_id["1"].gold_ids == frozenset({"https://github.com/a/b/blob/1/y.py"})
    assert by_id["0"].category == "code" and by_id["0"].answer is None


def test_ghost_qrels_counted_and_gains_exposed() -> None:
    world = parse_code("code", CORPUS, QUERIES, QRELS)
    assert LOAD_STATS["dropped_qrels"] == 1
    assert world_gains(world.id)["0"] == {"https://github.com/a/b/blob/1/x.py": 1}


def test_registered_with_query_sampler_that_keeps_corpus() -> None:
    assert datasets.ADAPTERS["code"] is code.load_code
    assert "code" in datasets.SAMPLERS and "code" in datasets.GAINS
    world = parse_code("code", CORPUS, QUERIES, QRELS)
    from ccr.bench.types import BenchConfig

    cfg = BenchConfig(ks=(10,), seed=1, sample_fraction=0.5, chunk_tokens=1, cache_dir="/x")
    sampled = datasets.SAMPLERS["code"]([world], cfg)[0]
    assert len(sampled.queries) == 1 and sampled.docs == world.docs


def test_parquet_round_trip_through_loader(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from ccr.bench.types import BenchConfig

    def write(name: str, rows: list[dict]) -> None:
        folder = tmp_path / name
        folder.mkdir()
        pq.write_table(pa.Table.from_pylist(rows), folder / "test-00000-of-00001.parquet")

    write("python-corpus", CORPUS)
    write("python-queries", QUERIES)
    write("python-qrels", QRELS)
    cfg = BenchConfig(ks=(10,), seed=1, sample_fraction=1.0, chunk_tokens=1,
                      cache_dir=str(tmp_path / "cache"))
    worlds = code.load_code(cfg, base_url=tmp_path.as_uri())
    assert len(worlds) == 1 and len(worlds[0].queries) == 2
