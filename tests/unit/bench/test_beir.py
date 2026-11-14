"""Tests for the BEIR adapter with tiny JSONL/TSV fixtures."""

from __future__ import annotations

import json

import pytest

from ccr.bench import datasets
from ccr.bench.datasets import beir
from ccr.bench.datasets.beir import LOAD_STATS, parse_beir, sample_queries
from ccr.bench.types import BenchConfig


@pytest.fixture()
def files(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps(r) for r in [
        {"_id": "d1", "title": "Quokkas", "text": "They live in Australia."},
        {"_id": "d2", "title": "", "text": "Rain in Spain."},
        {"_id": "d3", "title": "Weather", "text": "Mostly sunny."},
    ]))
    queries = tmp_path / "queries.jsonl"
    queries.write_text("\n".join(json.dumps(r) for r in [
        {"_id": "q1", "text": "where do quokkas live"},
        {"_id": "q2", "text": "spain rain"},
        {"_id": "q3", "text": "no judgments for this one"},
    ]))
    qrels = tmp_path / "test.tsv"
    qrels.write_text(
        "query-id\tcorpus-id\tscore\n"
        "q1\td1\t2\nq1\td3\t1\nq1\td2\t0\nq2\td2\t1\nq2\tghost\t1\n"
    )
    return str(corpus), str(queries), str(qrels)


def test_docs_join_title_and_text_and_skip_empty_title(files) -> None:
    world = parse_beir("beir-x", *files)
    docs = {d.id: d.text for d in world.docs}
    assert docs["d1"] == "Quokkas\nThey live in Australia."
    assert docs["d2"] == "Rain in Spain."


def test_header_skipped_zero_scores_not_gold_and_queries_without_qrels_dropped(files) -> None:
    world = parse_beir("beir-x", *files)
    by_id = {q.id: q for q in world.queries}
    assert set(by_id) == {"q1", "q2"}
    assert by_id["q1"].gold_ids == frozenset({"d1", "d3"})
    assert by_id["q1"].category == "beir" and by_id["q1"].answer is None


def test_graded_gains_exposed_and_ghost_docs_dropped_and_counted(files) -> None:
    world = parse_beir("beir-x", *files)
    gains = beir.world_gains(world.id)
    assert gains["q1"] == {"d1": 2, "d3": 1}
    assert gains["q2"] == {"d2": 1}
    assert LOAD_STATS["dropped_qrels"] == 1


def test_sample_queries_keeps_every_doc_and_is_seeded(files) -> None:
    world = parse_beir("beir-x", *files)
    small = sample_queries(world, 0.5, seed=3)
    assert len(small.queries) == 1 and small.docs == world.docs
    assert [q.id for q in sample_queries(world, 0.5, seed=3).queries] == [q.id for q in small.queries]
    assert len(sample_queries(world, 1.0, seed=3).queries) == 2


def test_registered_adapters_samplers_and_gains() -> None:
    for name in ("beir-scifact", "beir-nfcorpus"):
        assert name in datasets.ADAPTERS
        assert name in datasets.SAMPLERS and name in datasets.GAINS
    cfg = BenchConfig(ks=(10,), seed=1, sample_fraction=0.5, chunk_tokens=1, cache_dir="/x")
    assert callable(datasets.SAMPLERS["beir-scifact"])
    assert cfg.sample_fraction == 0.5
