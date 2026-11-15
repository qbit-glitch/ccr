"""Hybrid commit recall: natural-language questions, no 5-cap, legacy mode kept."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import ccr.core.commit_recall as commit_recall
from ccr.core.commit_search import tokenize
from ccr.core.memory import MemoryManager
from ccr.core.recall import RecallEngine, RecallOptions

DIM = 64


class FakeModel:
    def __init__(self) -> None:
        self.embedded = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_batch(self, texts):
        self.embedded += len(texts)
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def _commit(mem: MemoryManager, title: str, what: str) -> None:
    mem.commit(title=title, what=what, why="note", files_changed=[], next_step="",
               admission_threshold=1.0, rejection_threshold=0.0)


@pytest.fixture
def mem(tmp_path) -> MemoryManager:
    manager = MemoryManager(str(tmp_path))
    _commit(manager, "doc-0", "Caroline visited the adoption agency to ask about the process.")
    _commit(manager, "doc-1", "Melanie painted a sunrise over the lake for the charity auction.")
    _commit(manager, "doc-2", "The api gateway got a new rate limiting configuration.")
    for n in range(3, 12):
        _commit(manager, f"doc-{n}", f"Weekly status number {n} covering lake maintenance chores.")
    return manager


def _titles(result) -> list[str]:
    return [ev.title for ev in result.evidence]


def test_legacy_mode_keeps_the_five_result_cap(mem) -> None:
    engine = RecallEngine(mem, options=RecallOptions(commit_search="legacy"))
    question = "Which lake maintenance chores happened?"
    result = engine.recall(question, limit=10, sources=["commits"], include_sessions=False)
    assert 0 < len(result.evidence) <= 5
    fused = RecallEngine(mem).recall(question, limit=10, sources=["commits"],
                                     include_sessions=False)
    assert len(fused.evidence) >= 9


def test_hybrid_mode_answers_a_natural_language_question(mem) -> None:
    result = RecallEngine(mem).recall("What did Caroline ask the adoption agency?",
                                      sources=["commits"], include_sessions=False)
    assert _titles(result)[0] == "doc-0"


def test_question_marks_and_quotes_do_not_break_recall(mem) -> None:
    result = RecallEngine(mem).recall('Who painted "a sunrise" at the lake??', limit=3,
                                      sources=["commits"], include_sessions=False)
    assert _titles(result)[0] == "doc-1"


def test_more_than_five_results_come_back(mem) -> None:
    result = RecallEngine(mem).recall("lake", limit=10, sources=["commits"],
                                      include_sessions=False)
    assert len(result.evidence) >= 8


def test_exact_phrase_hit_still_ranks_first(mem) -> None:
    result = RecallEngine(mem).recall("rate limiting configuration", limit=5,
                                      sources=["commits"], include_sessions=False)
    assert _titles(result)[0] == "doc-2"


def test_new_commits_become_searchable_after_the_first_recall(mem) -> None:
    engine = RecallEngine(mem)
    engine.recall("lake", sources=["commits"], include_sessions=False)
    _commit(mem, "doc-new", "Priya presented the quarterly zebra migration plan.")
    result = engine.recall("zebra migration", sources=["commits"], include_sessions=False)
    assert _titles(result)[0] == "doc-new"


def test_dense_hits_without_shared_words_are_not_filtered_out(mem, monkeypatch) -> None:
    class Synonyms(FakeModel):
        def _vec(self, text: str) -> np.ndarray:
            return super()._vec(text.replace("sunrise", "dawn"))

    monkeypatch.setattr(commit_recall, "get_embedding_model", lambda: Synonyms())
    # "dawn" never appears in any commit; the dense side still maps it to "sunrise".
    result = RecallEngine(mem).recall("dawn", limit=3, sources=["commits"],
                                      include_sessions=False)
    assert "doc-1" in _titles(result)


def test_engines_on_one_manager_share_the_index_and_embed_once(mem, monkeypatch) -> None:
    model = FakeModel()
    monkeypatch.setattr(commit_recall, "get_embedding_model", lambda: model)
    RecallEngine(mem).recall("lake", sources=["commits"], include_sessions=False)
    first = model.embedded
    RecallEngine(mem).recall("adoption", sources=["commits"], include_sessions=False)
    assert first >= 12 and model.embedded == first


def test_rerank_off_keeps_the_fused_order(mem) -> None:
    plain = RecallEngine(mem, options=RecallOptions(rerank=False))
    result = plain.recall("lake maintenance chores", limit=5, sources=["commits"],
                          include_sessions=False)
    scores = [ev.score for ev in result.evidence]
    assert scores == sorted(scores, reverse=True)
    assert all(ev.metadata.get("reranker") is None for ev in result.evidence)


def test_unknown_commit_search_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="commit_search"):
        RecallOptions(commit_search="magic")


def _file_commit(mem: MemoryManager, title: str, what: str, files: list[str]) -> None:
    mem.commit(title=title, what=what, why="note", files_changed=files, next_step="",
               admission_threshold=1.0, rejection_threshold=0.0)


@pytest.fixture
def linked(tmp_path) -> MemoryManager:
    manager = MemoryManager(str(tmp_path))
    _file_commit(manager, "design", "Chose the quokka scheduler for nightly jobs.",
                 ["src/sched.py", "src/queue.py"])
    _file_commit(manager, "followup", "Retry budget raised after the incident review.",
                 ["src/sched.py", "src/queue.py"])
    _file_commit(manager, "unrelated", "Updated the office coffee machine rota.", ["docs/rota.md"])
    return manager


def test_link_expansion_pulls_in_commits_linked_to_a_top_hit(linked) -> None:
    ask = "quokka scheduler"
    base = RecallEngine(linked).recall(ask, limit=5, sources=["commits"], include_sessions=False)
    assert "followup" not in _titles(base)
    expanded = RecallEngine(linked, options=RecallOptions(link_expansion=True)).recall(
        ask, limit=5, sources=["commits"], include_sessions=False)
    titles = _titles(expanded)
    assert titles[0] == "design" and "followup" in titles and "unrelated" not in titles


def test_link_expansion_ranks_linked_commits_below_direct_hits(linked) -> None:
    result = RecallEngine(linked, options=RecallOptions(link_expansion=True)).recall(
        "quokka scheduler", limit=5, sources=["commits"], include_sessions=False)
    scores = {ev.title: ev.score for ev in result.evidence}
    assert scores["design"] > scores["followup"]
