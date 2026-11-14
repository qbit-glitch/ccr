"""Regression tests for the whole-branch review findings (I1-I4, M1, M3)."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from ccr.bench import datasets
from ccr.bench.metrics import tokenizer_name
from ccr.bench.reader import OllamaReader
from ccr.bench.report import build_report
from ccr.bench.results import QueryResult, ResultStore
from ccr.bench.retrievers.ccr_memory import CcrMemoryRetriever
from ccr.bench.runner import run_world
from ccr.bench.types import BenchConfig, Doc, Query, World
from ccr.cli import cli

CFG = BenchConfig(ks=(1, 2), seed=1, sample_fraction=1.0, chunk_tokens=512, cache_dir="/tmp/x")


class _Ret:
    name = "fake"
    max_k: int | None = None
    empty_count = 0

    def __init__(self, ranking: list[str]) -> None:
        self.ranking = ranking

    def index(self, docs, workdir: str) -> None: ...

    def retrieve(self, query: str, k: int) -> list[str]:
        return self.ranking[:k]

    def close(self) -> None: ...


class _FlakyReader:
    """Times out on the first call, answers afterwards; mirrors OllamaReader counters."""

    tag = "flaky"

    def __init__(self) -> None:
        self.timeout_count = 0
        self.empty_count = 0
        self.truncated_count = 0
        self.calls = 0

    def answer(self, question: str, context: str) -> str:
        self.calls += 1
        if self.calls == 1:
            self.timeout_count += 1
            return ""
        return "Paris"


def _world() -> World:
    docs = (Doc("d1", "alpha"), Doc("d2", "beta"))
    return World("w", docs, (Query("q1", "where?", frozenset({"d1"}), "Paris", "c"),), "g")


# I3: timeouts are flagged, counted, and retried on resume instead of cached as zero.
def test_reader_timeout_is_flagged_and_retried_on_resume(tmp_path) -> None:
    path = str(tmp_path / "r.jsonl")
    reader = _FlakyReader()
    run_world(_world(), _Ret(["d1"]), CFG, reader, ResultStore(path), None)
    first = ResultStore(path).all()[0]
    assert first.scores["reader_timeout"] == 1.0 and first.scores["f1"] == 0.0
    run_world(_world(), _Ret(["d1"]), CFG, reader, ResultStore(path), None)
    final = ResultStore(path).all()[0]
    assert "reader_timeout" not in final.scores and final.scores["f1"] == 1.0
    assert reader.calls == 2


def test_report_counts_timeouts_and_empty_answers() -> None:
    def row(qid, scores):
        return QueryResult(qid, "w", "s", "c", ("d1",), 5, "", scores, False, 0.0, 0.0)

    rep = build_report([row("a", {"f1": 0.0, "reader_timeout": 1.0}),
                        row("b", {"f1": 0.0, "reader_empty": 1.0}),
                        row("c", {"f1": 1.0})], {"dataset": "d"})
    s = rep.systems[0]
    assert (s.n_reader_timeout, s.n_reader_empty) == (1, 1)
    assert "reader_timeout" not in s.metrics and "reader_empty" not in s.metrics


# I2: the markdown report surfaces excluded / not-run counts and the answered subset.
def test_markdown_shows_excluded_not_run_and_answered_counts() -> None:
    def row(qid, system, scores):
        return QueryResult(qid, "w", system, "c", (), 5, None, scores, False, 0.0, 0.0)

    rows = [row("a", "full_context", {"not_run": 1.0}),
            row("b", "full_context", {"f1": 1.0}),
            row("x", "full_context", {"excluded": 1.0})]
    md = build_report(rows, {"dataset": "d", "pilot": "false"}).to_markdown()
    assert "| excluded |" in md and "| not_run |" in md and "| answered |" in md
    assert "not run: exceeds reader context (1/2)" in md


# I4: truncated prompts are detected; the tokenizer is nameable for the report.
def test_prompt_filling_the_window_counts_as_truncated() -> None:
    t = lambda url, payload: {"response": "x", "prompt_eval_count": 16384}  # noqa: E731
    reader = OllamaReader(host="http://h:1", transport=t, num_ctx=16384)
    reader.answer("q", "c")
    assert reader.truncated_count == 1


def test_prompt_inside_the_window_is_not_truncated() -> None:
    t = lambda url, payload: {"response": "x", "prompt_eval_count": 500}  # noqa: E731
    reader = OllamaReader(host="http://h:1", transport=t)
    reader.answer("q", "c")
    assert reader.truncated_count == 0


def test_tokenizer_name_is_a_known_label() -> None:
    assert tokenizer_name() in {"tiktoken-cl100k", "chars/4"}


def test_bench_extra_installs_tiktoken() -> None:
    import tomllib
    from pathlib import Path

    data = tomllib.loads((Path(__file__).parents[3] / "pyproject.toml").read_text())
    assert any(d.startswith("tiktoken") for d in data["project"]["optional-dependencies"]["bench"])


# M3: duplicate ids in a ranking must not burn top-k slots.
def test_duplicate_ranked_ids_are_collapsed(tmp_path) -> None:
    store = ResultStore(str(tmp_path / "r.jsonl"))
    cfg = BenchConfig(ks=(3,), seed=1, sample_fraction=1.0, chunk_tokens=8, cache_dir="/x")
    run_world(_world(), _Ret(["d2", "d2", "d1"]), cfg, None, store, None)
    res = store.all()[0]
    assert res.ranked_ids == ("d2", "d1") and res.scores["recall@3"] == 1.0


def test_longmemeval_duplicate_session_ids_get_unique_doc_ids() -> None:
    from ccr.bench.datasets.longmemeval import LOAD_STATS, parse_longmemeval

    item = {"question_id": "q1", "question": "?", "answer": "a", "question_type": "t",
            "answer_session_ids": ["s1"], "haystack_session_ids": ["s1", "s2", "s1"],
            "haystack_dates": ["d", "d", "d"],
            "haystack_sessions": [[{"role": "user", "content": "x"}]] * 3}
    world = parse_longmemeval([item])[0]
    ids = [d.id for d in world.docs]
    assert len(set(ids)) == 3 and ids[0] == "s1"
    assert world.queries[0].gold_ids == frozenset({"s1"})
    assert LOAD_STATS["duplicate_session_ids"] == 1


# M1: doc ids (which can encode gold labels) must not be searchable in ccr_memory.
def test_doc_id_is_not_searchable_text_in_ccr_memory(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    r = CcrMemoryRetriever()
    r.index([Doc("zzgoldlabel", "completely unrelated sentence about weather")], str(tmp_path))
    try:
        assert r.retrieve("zzgoldlabel", 5) == []
        assert r.retrieve("weather sentence", 5) == ["zzgoldlabel"]
    finally:
        r.close()


# I1: the output directory is pinned to one configuration.
def _fake_worlds(cfg):
    docs = (Doc("d1", "the quokka lives in australia"), Doc("d2", "rain in spain"))
    qs = (Query("q1", "quokka", frozenset({"d1"}), "australia", "cat1"),)
    return [World(f"w{i}", docs, qs, f"g{i}") for i in range(10)]


@pytest.fixture()
def fake(monkeypatch):
    monkeypatch.setitem(datasets.ADAPTERS, "fake", _fake_worlds)
    monkeypatch.setitem(datasets.STATS, "fake", {"dropped_evidence_ids": 7})


def _run(tmp_path, *args):
    out = tmp_path / "out"
    res = CliRunner().invoke(cli, ["bench", "run", "fake", "--systems", "bm25", "--no-reader",
                                   "--output", str(out), "--cache-dir", str(tmp_path / "c"),
                                   *args])
    return res, out


def test_rerun_with_different_settings_into_same_dir_is_refused(tmp_path, fake) -> None:
    ok, out = _run(tmp_path, "--seed", "1")
    assert ok.exit_code == 0, ok.output
    before = (out / "results.jsonl").read_text()
    bad, _ = _run(tmp_path, "--seed", "2")
    assert bad.exit_code != 0 and "different configuration" in bad.output
    assert (out / "results.jsonl").read_text() == before


def test_rerun_with_same_settings_still_resumes(tmp_path, fake) -> None:
    _run(tmp_path)
    again, _ = _run(tmp_path)
    assert again.exit_code == 0, again.output


def test_meta_records_tokenizer_and_dataset_load_stats(tmp_path, fake) -> None:
    _, out = _run(tmp_path)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["tokenizer"] == tokenizer_name()
    assert meta["load.dropped_evidence_ids"] == "7"
