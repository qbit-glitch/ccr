"""Tests for the resumable benchmark runner (fake retriever and reader)."""

from __future__ import annotations

import math

import pytest

from ccr.bench.metrics import count_tokens
from ccr.bench.results import ResultStore
from ccr.bench.retrievers.fullcontext import FullContextRetriever
from ccr.bench.runner import run_world
from ccr.bench.types import BenchConfig, Doc, Query, World

CFG = BenchConfig(ks=(1, 2), seed=1, sample_fraction=1.0, chunk_tokens=512, cache_dir="/tmp/x")


class FakeRetriever:
    name = "fake"
    max_k: int | None = None

    def __init__(self, ranking: list[str], fail: bool = False) -> None:
        self.ranking = ranking
        self.fail = fail
        self.empty_count = 0
        self.index_calls = 0
        self.closed = 0

    def index(self, docs, workdir: str) -> None:
        self.index_calls += 1

    def retrieve(self, query: str, k: int) -> list[str]:
        if self.fail:
            raise RuntimeError("boom")
        return self.ranking[:k]

    def close(self) -> None:
        self.closed += 1


class FakeReader:
    tag = "fake-reader"

    def __init__(self, reply: str = "Paris") -> None:
        self.reply = reply
        self.contexts: list[str] = []

    def answer(self, question: str, context: str) -> str:
        self.contexts.append(context)
        return self.reply


def _world() -> World:
    docs = (Doc("d1", "alpha text"), Doc("d2", "beta text"), Doc("d3", "gamma text"))
    queries = (
        Query("q1", "where?", frozenset({"d1"}), "Paris", "cat1"),
        Query("q2", "nothing", frozenset(), None, "cat5"),
    )
    return World("w1", docs, queries, "g")


def test_retrieval_scores_computed_for_each_k(tmp_path) -> None:
    store = ResultStore(str(tmp_path / "r.jsonl"))
    r = FakeRetriever(["d2", "d1", "d3"])
    run_world(_world(), r, CFG, None, store, None)
    res = {x.query_id: x for x in store.all()}["q1"]
    assert res.scores["recall@1"] == 0.0
    assert res.scores["recall@2"] == 1.0
    assert res.scores["mrr"] == 0.5
    assert res.scores["ndcg@2"] == pytest.approx(1 / math.log2(3))
    assert res.ranked_ids == ("d2", "d1")


def test_queries_without_gold_or_answer_are_excluded_and_counted(tmp_path) -> None:
    store = ResultStore(str(tmp_path / "r.jsonl"))
    run_world(_world(), FakeRetriever(["d1"]), CFG, None, store, None)
    excluded = [x for x in store.all() if x.scores.get("excluded") == 1.0]
    assert [x.query_id for x in excluded] == ["q2"]


def test_rerun_resumes_with_no_new_work_and_no_reindex(tmp_path) -> None:
    store = ResultStore(str(tmp_path / "r.jsonl"))
    r = FakeRetriever(["d1"])
    first = run_world(_world(), r, CFG, None, store, None)
    again = run_world(_world(), r, CFG, None, ResultStore(str(tmp_path / "r.jsonl")), None)
    assert first == 2 and again == 0
    assert r.index_calls == 1


def test_k_above_retriever_cap_raises_value_error(tmp_path) -> None:
    r = FakeRetriever(["d1"])
    r.max_k = 1
    with pytest.raises(ValueError, match="max_k"):
        run_world(_world(), r, CFG, None, ResultStore(str(tmp_path / "r.jsonl")), None)


def test_reader_gets_full_text_of_top_docs_and_tokens_match(tmp_path) -> None:
    cfg = BenchConfig(ks=(2,), seed=1, sample_fraction=1.0, chunk_tokens=8,
                      cache_dir="/x", reader_k=2)
    reader = FakeReader("Paris")
    store = ResultStore(str(tmp_path / "r.jsonl"))
    run_world(_world(), FakeRetriever(["d2", "d1", "d3"]), cfg, reader, store, None)
    assert reader.contexts == ["beta text\n\nalpha text"]
    res = {x.query_id: x for x in store.all()}["q1"]
    assert res.retrieved_tokens == count_tokens("beta text\n\nalpha text")
    assert res.answer == "Paris"
    assert res.scores["f1"] == 1.0 and res.scores["em"] == 1.0


def test_empty_reader_answer_scores_zero_and_is_kept(tmp_path) -> None:
    store = ResultStore(str(tmp_path / "r.jsonl"))
    run_world(_world(), FakeRetriever(["d1"]), CFG, FakeReader(""), store, None)
    res = {x.query_id: x for x in store.all()}["q1"]
    assert res.answer == "" and res.scores["f1"] == 0.0


def test_context_over_limit_is_not_run_and_reader_not_called(tmp_path) -> None:
    reader = FakeReader()
    store = ResultStore(str(tmp_path / "r.jsonl"))
    run_world(_world(), FullContextRetriever(), CFG, reader, store, 1)
    res = {x.query_id: x for x in store.all()}["q1"]
    assert res.scores == {"not_run": 1.0}
    assert res.answer is None and reader.contexts == []


def test_full_context_reports_no_retrieval_scores(tmp_path) -> None:
    store = ResultStore(str(tmp_path / "r.jsonl"))
    run_world(_world(), FullContextRetriever(), CFG, FakeReader(), store, None)
    res = {x.query_id: x for x in store.all()}["q1"]
    assert "recall@1" not in res.scores and res.scores["f1"] == 1.0


def test_graded_gains_drive_ndcg_when_provided(tmp_path) -> None:
    store = ResultStore(str(tmp_path / "r.jsonl"))
    gains = {"q1": {"d1": 2, "d2": 1}}
    run_world(_world(), FakeRetriever(["d2", "d1"]), CFG, None, store, None,
              gains_for=lambda world_id: gains)
    res = {x.query_id: x for x in store.all()}["q1"]
    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    ideal = 2 / math.log2(2) + 1 / math.log2(3)
    assert res.scores["ndcg@2"] == pytest.approx(dcg / ideal)


def test_retriever_is_closed_even_when_retrieval_fails(tmp_path) -> None:
    r = FakeRetriever(["d1"], fail=True)
    with pytest.raises(RuntimeError):
        run_world(_world(), r, CFG, None, ResultStore(str(tmp_path / "r.jsonl")), None)
    assert r.closed == 1


def test_alternative_gold_answers_take_the_best_score(tmp_path) -> None:
    world = World("w", (Doc("d1", "alpha"),),
                  (Query("q", "city?", frozenset(), "Paris", "c", alt_answers=("City of Paris",)),), "g")
    store = ResultStore(str(tmp_path / "r.jsonl"))
    run_world(world, FakeRetriever(["d1"]), CFG, FakeReader("City of Paris"), store, None)
    res = store.all()[0]
    assert res.scores["em"] == 1.0 and res.scores["sub_em"] == 1.0
