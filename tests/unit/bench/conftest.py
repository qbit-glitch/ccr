"""Shared fixtures for benchmark tests.

CCR's memory path loads the ONNX embedding model when onnxruntime is installed,
which downloads ~90 MB on first use. Benchmark unit tests must never touch the
network, so semantic features are switched off for every test in this package.
"""

from __future__ import annotations

import pytest

import ccr.context.embeddings as embeddings


@pytest.fixture(autouse=True)
def _no_semantic_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embeddings, "SEMANTIC_AVAILABLE", False)
    monkeypatch.setattr(embeddings, "_cached_model", None)
