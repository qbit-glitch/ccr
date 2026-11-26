"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from ccr.context import embeddings


@pytest.fixture(autouse=True)
def _isolate_hook_error_log(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Keep hook error logging out of the real repo.

    The hooks append failures to <CCR_PROJECT_ROOT or cwd>/.ccr/.hook_errors.log. Tests that
    mock a failing commit were writing into this repo's own log, so `ccr doctor` reported them
    as real hook errors. A root without a .ccr/ makes the logger a no-op; tests that care set
    their own root.
    """
    monkeypatch.setenv("CCR_PROJECT_ROOT", str(tmp_path_factory.mktemp("no_ccr_root")))


@pytest.fixture
def no_semantic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run a test with the embedding backend off, whatever extras are installed.

    get_embedding_model() reads SEMANTIC_AVAILABLE at call time, so every caller
    (A-MAC admission, pattern dedup, scratchpad search, ...) falls back to the
    keyword path and nothing downloads the ONNX model.
    """
    monkeypatch.setattr(embeddings, "SEMANTIC_AVAILABLE", False)
    monkeypatch.setattr(embeddings, "_cached_model", None)
