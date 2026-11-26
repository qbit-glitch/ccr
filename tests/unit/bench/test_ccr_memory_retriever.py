"""Tests for the CCR memory retriever (isolated temp project, no network)."""

from __future__ import annotations

import os

import pytest

from ccr.bench.retrievers import REGISTRY
from ccr.bench.retrievers.ccr_memory import CcrMemoryRetriever
from ccr.bench.types import Doc


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Fake HOME and an unrelated cwd so we can prove nothing leaks into either."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)
    return home, cwd, work


def _build(work, docs):
    r = CcrMemoryRetriever()
    r.index(docs, str(work))
    return r


def test_identical_turns_are_not_merged_by_admission(isolated) -> None:
    _, _, work = isolated
    docs = [Doc(f"D1:{i}", "Okay, thanks!") for i in range(1, 6)]
    r = _build(work, docs)
    try:
        assert set(r.retrieve("okay thanks", 5)) == {d.id for d in docs}
    finally:
        r.close()


def test_unique_word_ranks_its_doc_first(isolated) -> None:
    _, _, work = isolated
    docs = [
        Doc("D1:1", "we talked about the weather"),
        Doc("D1:2", "my quokka escaped yesterday"),
        Doc("D1:3", "the weather was fine again"),
    ]
    r = _build(work, docs)
    try:
        assert r.retrieve("quokka", 3)[0] == "D1:2"
    finally:
        r.close()


def test_special_character_queries_do_not_raise_or_vanish(isolated) -> None:
    _, _, work = isolated
    r = _build(work, [Doc("D1:3", "the quokka is cute")])
    try:
        out = r.retrieve('what "is" *quokka* -y: D1:3?', 3)
        assert out == ["D1:3"]
    finally:
        r.close()


def test_k_above_recall_cap_fails_loudly(isolated) -> None:
    _, _, work = isolated
    r = _build(work, [Doc("D1:1", "text")])
    try:
        assert r.max_k == 25
        with pytest.raises(ValueError, match="25"):
            r.retrieve("text", 26)
    finally:
        r.close()


def test_empty_results_are_counted_not_hidden(isolated) -> None:
    _, _, work = isolated
    r = _build(work, [Doc("D1:1", "alpha beta")])
    try:
        assert r.retrieve("zzzzunmatchable", 5) == []
        assert r.empty_count == 1
    finally:
        r.close()


def test_run_writes_only_inside_workdir(isolated) -> None:
    home, cwd, work = isolated
    r = _build(work, [Doc("D1:1", "alpha beta")])
    r.retrieve("alpha", 5)
    r.close()
    assert os.listdir(cwd) == []
    assert not (home / ".ccr").exists()
    assert (work / ".ccr").exists()


def test_registered_under_expected_name() -> None:
    assert REGISTRY["ccr_memory"]().name == "ccr_memory"


def test_variants_are_registered_with_their_own_names_and_options() -> None:
    assert {"ccr_memory", "ccr_memory_legacy", "ccr_memory_bm25", "ccr_memory_dense",
            "ccr_memory_norerank"} <= set(REGISTRY)
    legacy = REGISTRY["ccr_memory_legacy"]()
    assert legacy.name == "ccr_memory_legacy" and legacy.options.commit_search == "legacy"
    assert REGISTRY["ccr_memory"]().options.commit_search == "bm25"
    assert REGISTRY["ccr_memory_hybrid"]().options.commit_search == "hybrid"
    assert REGISTRY["ccr_memory_norerank"]().options.rerank is False


def test_hybrid_variant_answers_a_question_the_legacy_variant_caps(isolated) -> None:
    _, _, work = isolated
    docs = [Doc(f"d{n}", f"Weekly lake maintenance chores report number {n}.") for n in range(12)]
    docs.append(Doc("target", "Caroline asked the adoption agency about the process."))
    hybrid = REGISTRY["ccr_memory"]()
    hybrid.index(docs, str(work))
    ids = hybrid.retrieve("What did Caroline ask the adoption agency?", 10)
    hybrid.close()
    assert ids[0] == "target"


def test_technique_variants_each_switch_on_exactly_one_flag() -> None:
    flags = {"ccr_memory+links": "link_expansion", "ccr_memory+scenes": "scene_expansion",
             "ccr_memory+triples": "triple_boost", "ccr_memory+chunks": "chunk_dense"}
    for name, flag in flags.items():
        options = REGISTRY[name]().options
        assert getattr(options, flag) is True
        assert [f for f in flags.values() if getattr(options, f)] == [flag]
