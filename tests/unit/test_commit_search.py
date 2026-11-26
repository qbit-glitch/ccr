"""Tests for the hybrid commit search index (BM25 + dense + reciprocal-rank fusion)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from ccr.core.commit_search import CommitSearchIndex, rrf, tokenize

DIM = 64


class FakeEmbedder:
    """Deterministic bag-of-words embedder; counts calls so tests can assert reuse."""

    def __init__(self) -> None:
        self.texts_embedded = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        self.texts_embedded += len(texts)
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


DOCS = [
    ("C1", "Caroline visited the adoption agency on Tuesday to ask about the process."),
    ("C2", "Melanie painted a sunrise over the lake for the charity auction."),
    ("C3", "Deploy the api gateway with the new rate limiting configuration."),
    ("C4", "Caroline researched adoption agencies that support single parents."),
]


def test_tokenize_lowercases_and_drops_stop_words_and_punctuation() -> None:
    assert tokenize("What did Caroline research?") == ["caroline", "research"]


def test_rrf_rewards_agreement_and_breaks_ties_by_id() -> None:
    fused = dict(rrf([["a", "b", "c"], ["a", "c", "b"]]))
    assert fused["a"] > fused["b"] and fused["a"] > fused["c"]
    order = [i for i, _ in rrf([["x"], ["y"]])]
    assert order == ["x", "y"]


def test_bm25_uses_or_semantics_so_a_question_matches_partial_terms() -> None:
    idx = CommitSearchIndex()
    idx.sync(DOCS)
    hits = [cid for cid, _ in idx.search("What did Caroline research about adoption agencies?", 4)]
    assert hits[0] == "C4" and set(hits[:2]) == {"C4", "C1"}


def test_query_with_punctuation_and_unknown_words_does_not_raise() -> None:
    idx = CommitSearchIndex()
    idx.sync(DOCS)
    assert idx.search('"unbalanced (quote ??', 3) == []
    assert idx.search("   ", 3) == []


def test_dense_finds_a_neighbor_that_hybrid_also_returns() -> None:
    emb = FakeEmbedder()
    idx = CommitSearchIndex(embedder=emb)
    idx.sync(DOCS)
    assert idx.dense_available
    dense = [c for c, _ in idx.search("sunrise lake painting", 2, mode="dense")]
    hybrid = [c for c, _ in idx.search("sunrise lake painting", 2, mode="hybrid")]
    assert dense[0] == "C2" and hybrid[0] == "C2"


def test_hybrid_without_an_embedder_falls_back_to_bm25() -> None:
    idx = CommitSearchIndex()
    assert not idx.dense_available
    assert idx.search("adoption", 2, mode="hybrid") == []
    idx.sync(DOCS)
    assert [c for c, _ in idx.search("adoption", 2, mode="hybrid")] == [
        c for c, _ in idx.search("adoption", 2, mode="bm25")
    ]


def test_sync_is_incremental_and_does_not_reembed_known_commits() -> None:
    emb = FakeEmbedder()
    idx = CommitSearchIndex(embedder=emb)
    idx.sync(DOCS[:2])
    assert emb.texts_embedded == 2
    idx.sync(DOCS)
    assert emb.texts_embedded == 4
    assert len(idx) == 4


def test_sync_rebuilds_when_a_commit_disappears() -> None:
    idx = CommitSearchIndex()
    idx.sync(DOCS)
    idx.sync(DOCS[:2])
    assert len(idx) == 2
    assert all(c in {"C1", "C2"} for c, _ in idx.search("caroline adoption lake", 4))


def test_no_result_cap_beyond_five() -> None:
    idx = CommitSearchIndex()
    idx.sync([(f"C{i}", f"shared topic note number {i} alpha") for i in range(300)])
    assert len(idx.search("alpha topic", 50)) == 50


def test_vectors_persist_and_are_reused_until_the_text_changes(tmp_path) -> None:
    path = str(tmp_path / "vecs.npz")
    first = FakeEmbedder()
    idx = CommitSearchIndex(embedder=first, vecs_path=path)
    idx.sync(DOCS)
    idx.save()
    assert first.texts_embedded == 4

    second = FakeEmbedder()
    again = CommitSearchIndex(embedder=second, vecs_path=path)
    again.sync(DOCS)
    assert second.texts_embedded == 0

    edited = [DOCS[0], ("C2", "Melanie moved to a new city."), *DOCS[2:]]
    third = FakeEmbedder()
    CommitSearchIndex(embedder=third, vecs_path=path).sync(edited)
    assert third.texts_embedded == 1


def test_corrupt_vector_file_is_ignored(tmp_path) -> None:
    path = tmp_path / "vecs.npz"
    path.write_bytes(b"not an npz")
    emb = FakeEmbedder()
    idx = CommitSearchIndex(embedder=emb, vecs_path=str(path))
    idx.sync(DOCS)
    assert emb.texts_embedded == 4


@pytest.mark.parametrize("mode", ["hybrid", "bm25", "dense"])
def test_every_mode_returns_scored_pairs_best_first(mode) -> None:
    idx = CommitSearchIndex(embedder=FakeEmbedder())
    idx.sync(DOCS)
    hits = idx.search("caroline adoption", 4, mode=mode)
    scores = [s for _, s in hits]
    assert hits and scores == sorted(scores, reverse=True)


def test_rrf_weights_let_one_ranking_count_less() -> None:
    lexical, dense = ["a", "b"], ["b", "a"]
    even = [i for i, _ in rrf([lexical, dense])]
    assert even == ["a", "b"]  # tie on score, first seen wins
    light_dense = [i for i, _ in rrf([lexical, dense], weights=[1.0, 0.3])]
    heavy_dense = [i for i, _ in rrf([lexical, dense], weights=[0.3, 1.0])]
    assert light_dense == ["a", "b"] and heavy_dense == ["b", "a"]


def test_search_dense_weight_changes_the_hybrid_order() -> None:
    class Opposed(FakeEmbedder):
        """BM25 prefers L (alpha twice); this embedder makes D the dense neighbor."""

        def _vec(self, text: str) -> np.ndarray:
            v = np.zeros(DIM, dtype=np.float32)
            v[0 if ("beta" in text or text == "alpha beta") else 1] = 1.0
            return v

    idx = CommitSearchIndex(embedder=Opposed())
    idx.sync([("L", "alpha alpha"), ("D", "beta")])
    low = [c for c, _ in idx.search("alpha beta", 2, dense_weight=0.05)]
    high = [c for c, _ in idx.search("alpha beta", 2, dense_weight=3.0)]
    assert low == ["L", "D"] and high == ["D", "L"]


FILLER = " ".join(f"filler{i}" for i in range(60))
LONG = [("L1", f"{FILLER} zebra migration plan {FILLER}"), ("L2", f"{FILLER} tax forms {FILLER}")]


def test_chunk_dense_finds_a_passage_buried_in_a_long_commit() -> None:
    whole = CommitSearchIndex(embedder=FakeEmbedder())
    whole.sync(LONG)
    chunked = CommitSearchIndex(embedder=FakeEmbedder(), chunk_words=10)
    chunked.sync(LONG)
    q = "zebra migration"
    top_whole = whole.search(q, 2, mode="dense")
    top_chunk = chunked.search(q, 2, mode="dense")
    assert top_chunk[0][0] == "L1"
    assert top_chunk[0][1] > top_whole[0][1] * 2  # best chunk is far closer than the diluted whole


def test_chunked_vectors_persist_and_a_chunk_size_change_reembeds(tmp_path) -> None:
    path = str(tmp_path / "vecs.npz")
    first = FakeEmbedder()
    idx = CommitSearchIndex(embedder=first, vecs_path=path, chunk_words=10)
    idx.sync(LONG)
    idx.save()
    assert first.texts_embedded > 2
    again = FakeEmbedder()
    CommitSearchIndex(embedder=again, vecs_path=path, chunk_words=10).sync(LONG)
    assert again.texts_embedded == 0
    other = FakeEmbedder()
    CommitSearchIndex(embedder=other, vecs_path=path, chunk_words=20).sync(LONG)
    assert other.texts_embedded > 0
