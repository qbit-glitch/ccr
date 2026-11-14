"""Tests for the retriever protocol, registry and the two simple baselines."""

from __future__ import annotations

import pytest

from ccr.bench.retrievers import REGISTRY, RetrieverUnavailable, register
from ccr.bench.retrievers.bm25 import Bm25Retriever
from ccr.bench.retrievers.fullcontext import FullContextRetriever
from ccr.bench.types import Doc

DOCS = [
    Doc("d1", "the weather is nice today"),
    Doc("d2", "quokka lives in western australia"),
    Doc("d3", "the weather forecast mentions rain"),
]


def test_bm25_ranks_doc_with_rare_query_term_first(tmp_path) -> None:
    r = Bm25Retriever()
    r.index(DOCS, str(tmp_path))
    assert r.retrieve("quokka", 3)[0] == "d2"


def test_bm25_empty_index_returns_empty_list(tmp_path) -> None:
    r = Bm25Retriever()
    r.index([], str(tmp_path))
    assert r.retrieve("anything", 5) == []


def test_bm25_special_characters_do_not_raise(tmp_path) -> None:
    r = Bm25Retriever()
    r.index(DOCS, str(tmp_path))
    assert isinstance(r.retrieve('what "is" *x* -y: D1:3?', 3), list)


def test_bm25_respects_k_and_has_no_cap(tmp_path) -> None:
    r = Bm25Retriever()
    r.index(DOCS, str(tmp_path))
    assert len(r.retrieve("weather", 1)) == 1
    assert r.max_k is None


def test_full_context_returns_every_doc_regardless_of_k(tmp_path) -> None:
    r = FullContextRetriever()
    r.index(DOCS, str(tmp_path))
    assert r.retrieve("q", 1) == ["d1", "d2", "d3"]
    assert r.name == "full_context"


def test_registry_holds_both_baselines() -> None:
    assert REGISTRY["bm25"]().name == "bm25"
    assert REGISTRY["full_context"]().name == "full_context"


def test_register_decorator_adds_factory_and_unavailable_is_exception() -> None:
    @register("tmp_test_system")
    class _Tmp(Bm25Retriever):
        name = "tmp_test_system"

    try:
        assert REGISTRY["tmp_test_system"]().name == "tmp_test_system"
    finally:
        REGISTRY.pop("tmp_test_system", None)
    with pytest.raises(RetrieverUnavailable):
        raise RetrieverUnavailable("missing dep")
