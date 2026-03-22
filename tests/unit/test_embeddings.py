"""Tests for the embedding backend (ccr/context/embeddings.py)."""

import gzip
import json
import os
import tempfile

import pytest

from ccr.context.embeddings import (
    SEMANTIC_AVAILABLE,
    load_embeddings,
    save_embeddings,
)


class TestSemanticAvailableFlag:
    def test_flag_is_bool(self):
        assert isinstance(SEMANTIC_AVAILABLE, bool)


class TestSaveLoadEmbeddings:
    def test_roundtrip(self):
        data = {"file_a.py": [0.1, 0.2, 0.3], "file_b.py": [0.4, 0.5, 0.6]}
        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
            path = f.name
        try:
            save_embeddings(data, path)
            loaded = load_embeddings(path)
            assert loaded == data
        finally:
            os.unlink(path)

    def test_load_missing_file(self):
        result = load_embeddings("/tmp/nonexistent_ccr_embeddings.json.gz")
        assert result == {}

    def test_load_corrupt_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
            f.write(b"not valid gzip data")
            path = f.name
        try:
            result = load_embeddings(path)
            assert result == {}
        finally:
            os.unlink(path)

    def test_save_atomic_cleanup_on_error(self):
        """save_embeddings cleans up .tmp on error."""
        # Use a non-serializable value to trigger error
        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(TypeError):
                save_embeddings({"bad": object()}, path)
            # .tmp file should be cleaned up
            assert not os.path.isfile(path + ".tmp")
        finally:
            if os.path.isfile(path):
                os.unlink(path)

    def test_gzip_compressed(self):
        data = {"x.py": [1.0, 2.0]}
        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
            path = f.name
        try:
            save_embeddings(data, path)
            # Verify it's actually gzip
            with gzip.open(path, "rt") as f:
                content = json.loads(f.read())
            assert content == data
        finally:
            os.unlink(path)


@pytest.mark.skipif(not SEMANTIC_AVAILABLE, reason="onnxruntime/tokenizers not installed")
class TestEmbeddingModel:
    """Tests for the EmbeddingModel class — only run if ONNX deps are available."""

    def test_get_model_returns_instance(self):
        from ccr.context.embeddings import get_embedding_model

        model = get_embedding_model()
        assert model is not None

    def test_get_model_cached(self):
        from ccr.context.embeddings import get_embedding_model

        m1 = get_embedding_model()
        m2 = get_embedding_model()
        assert m1 is m2
