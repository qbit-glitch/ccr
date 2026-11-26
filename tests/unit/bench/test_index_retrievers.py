"""Tests for the CCR index retrievers and the dense MiniLM baseline."""

from __future__ import annotations

import pytest

import ccr.bench.retrievers.ccr_index as ccr_index_mod
import ccr.bench.retrievers.dense as dense_mod
from ccr.bench.retrievers import REGISTRY, RetrieverUnavailable
from ccr.bench.retrievers.ccr_index import CcrIndexRetriever
from ccr.bench.retrievers.dense import DenseMiniLmRetriever
from ccr.bench.types import Doc

DOCS = [
    Doc("MED-1/a b", "the weather is nice today"),
    Doc("MED-2", "quokka lives in western australia"),
    Doc("MED-3", "weather forecast mentions rain"),
]


class FakeModel:
    """Bag-of-words hashing 'embedding' so tests need no ONNX or network."""

    DIM = 64

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.DIM
            for word in text.lower().split():
                vec[hash(word) % self.DIM] += 1.0
            out.append(vec)
        return out


def test_bm25_mode_returns_original_doc_ids_and_indexes_txt_files(tmp_path) -> None:
    r = CcrIndexRetriever("bm25")
    r.index(DOCS, str(tmp_path))
    try:
        assert r.indexed_count == 3
        assert r.retrieve("quokka", 3)[0] == "MED-2"
        assert "MED-1/a b" in r.retrieve("nice", 3)
    finally:
        r.close()


def test_keyword_mode_works_and_is_registered(tmp_path) -> None:
    r = REGISTRY["ccr_index_keyword"]()
    r.index(DOCS, str(tmp_path))
    try:
        assert r.retrieve("quokka", 3) == ["MED-2"]
    finally:
        r.close()


def test_top_k_above_cap_fails_loudly(tmp_path) -> None:
    r = CcrIndexRetriever("bm25")
    r.index(DOCS, str(tmp_path))
    try:
        assert r.max_k == 100
        with pytest.raises(ValueError, match="100"):
            r.retrieve("weather", 101)
    finally:
        r.close()


def test_special_characters_do_not_raise_and_count_empty(tmp_path) -> None:
    r = CcrIndexRetriever("bm25")
    r.index(DOCS, str(tmp_path))
    try:
        assert r.retrieve('"*" -: ?', 3) == []
        assert r.empty_count == 1
    finally:
        r.close()


def test_hybrid_without_embeddings_is_unavailable_not_silent_bm25(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(ccr_index_mod, "get_embedding_model", lambda: None)
    r = CcrIndexRetriever("hybrid")
    with pytest.raises(RetrieverUnavailable):
        r.index(DOCS, str(tmp_path))


def test_dense_without_onnx_is_unavailable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dense_mod, "get_embedding_model", lambda: None)
    with pytest.raises(RetrieverUnavailable):
        DenseMiniLmRetriever().index(DOCS, str(tmp_path))


def test_dense_ranks_by_cosine_with_fake_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dense_mod, "get_embedding_model", lambda: FakeModel())
    r = DenseMiniLmRetriever()
    r.index(DOCS, str(tmp_path))
    assert r.retrieve("quokka australia", 2)[0] == "MED-2"
    assert len(r.retrieve("weather", 1)) == 1
    assert REGISTRY["dense_minilm"]().name == "dense_minilm"
