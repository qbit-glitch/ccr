"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from ccr.context import embeddings


@pytest.fixture
def no_semantic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run a test with the embedding backend off, whatever extras are installed.

    get_embedding_model() reads SEMANTIC_AVAILABLE at call time, so every caller
    (A-MAC admission, pattern dedup, scratchpad search, ...) falls back to the
    keyword path and nothing downloads the ONNX model.
    """
    monkeypatch.setattr(embeddings, "SEMANTIC_AVAILABLE", False)
    monkeypatch.setattr(embeddings, "_cached_model", None)
