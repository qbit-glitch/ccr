"""Optional ONNX embedding backend for semantic code search.

Uses all-MiniLM-L6-v2 (384-dim) via onnxruntime + tokenizers.
Auto-downloads model from HuggingFace CDN on first use.

Install: pip install onnxruntime tokenizers numpy
Or:      pip install ccr[semantic]

A-RAG §3.1: Hierarchical index construction with dense embeddings.
A-RAG §3.2 Eq 3: Score_sem(s, q) = cosine(v_s, v_q).
"""

from __future__ import annotations

import gzip
import json
import logging
import os

logger = logging.getLogger(__name__)

# Soft dependencies — semantic search degrades gracefully without these
try:
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    np = None  # type: ignore[assignment]
    ort = None  # type: ignore[assignment]
    Tokenizer = None  # type: ignore[assignment]

# HuggingFace CDN URLs for all-MiniLM-L6-v2
_HF_BASE = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
_MODEL_URL = f"{_HF_BASE}/onnx/model.onnx"
_TOKENIZER_URL = f"{_HF_BASE}/tokenizer.json"

_cached_model: "EmbeddingModel | None" = None


def get_embedding_model() -> "EmbeddingModel | None":
    """Return a cached EmbeddingModel instance, or None if deps unavailable."""
    global _cached_model
    if not SEMANTIC_AVAILABLE:
        return None
    if _cached_model is None:
        _cached_model = EmbeddingModel()
    return _cached_model


if SEMANTIC_AVAILABLE:

    class EmbeddingModel:
        """Manages ONNX embedding model for semantic code search.

        Uses all-MiniLM-L6-v2 (384-dim, ~90 MB ONNX).
        Auto-downloads to ~/.cache/ccr/models/ on first use via httpx.

        A-RAG §3.1: sentence-level embeddings for semantic matching.
        CCR adaptation: file-summary embeddings for code search.
        """

        MODEL_NAME = "all-MiniLM-L6-v2"
        MODEL_DIM = 384
        MAX_TOKENS = 256

        def __init__(self, cache_dir: str | None = None):
            self._cache_dir = cache_dir or os.path.join(
                os.path.expanduser("~"), ".cache", "ccr", "models", self.MODEL_NAME
            )
            self._session: ort.InferenceSession | None = None
            self._tokenizer: Tokenizer | None = None

        def _ensure_model(self) -> None:
            """Download model files if not cached, then load."""
            if self._session is not None:
                return

            os.makedirs(self._cache_dir, exist_ok=True)
            model_path = os.path.join(self._cache_dir, "model.onnx")
            tokenizer_path = os.path.join(self._cache_dir, "tokenizer.json")

            if not os.path.isfile(model_path):
                logger.info("Downloading %s ONNX model (~90 MB)...", self.MODEL_NAME)
                self._download_file(_MODEL_URL, model_path)

            if not os.path.isfile(tokenizer_path):
                logger.info("Downloading %s tokenizer...", self.MODEL_NAME)
                self._download_file(_TOKENIZER_URL, tokenizer_path)

            self._session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            self._tokenizer.enable_truncation(max_length=self.MAX_TOKENS)
            self._tokenizer.enable_padding(
                length=self.MAX_TOKENS, pad_id=0, pad_token="[PAD]"
            )

        @staticmethod
        def _download_file(url: str, dest: str) -> None:
            """Download a file from URL to dest using httpx (already a CCR dep)."""
            import httpx

            tmp = dest + ".tmp"
            try:
                with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_bytes(chunk_size=65536):
                            f.write(chunk)
                os.replace(tmp, dest)
            except Exception:
                if os.path.isfile(tmp):
                    os.unlink(tmp)
                raise

        def embed_batch(self, texts: list[str]) -> np.ndarray:
            """Embed a batch of texts. Returns (N, 384) float32 array."""
            self._ensure_model()
            assert self._tokenizer is not None
            assert self._session is not None

            encodings = self._tokenizer.encode_batch(texts)

            input_ids = np.array(
                [e.ids for e in encodings], dtype=np.int64
            )
            attention_mask = np.array(
                [e.attention_mask for e in encodings], dtype=np.int64
            )
            token_type_ids = np.zeros_like(input_ids)

            outputs = self._session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )

            # outputs[0] = last_hidden_state: (batch, seq_len, 384)
            last_hidden = outputs[0]

            # Mean pooling with attention mask
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            summed = (last_hidden * mask_expanded).sum(axis=1)
            counts = mask_expanded.sum(axis=1).clip(min=1e-9)
            pooled = summed / counts

            # L2 normalize
            norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
            return (pooled / norms).astype(np.float32)

        def embed_query(self, query: str) -> np.ndarray:
            """Embed a single query. Returns (384,) float32 vector."""
            return self.embed_batch([query])[0]

        @staticmethod
        def cosine_similarity(
            query_vec: np.ndarray, doc_vecs: np.ndarray
        ) -> np.ndarray:
            """Cosine similarity between query (384,) and docs (N, 384).

            Both must be L2-normalized. Returns (N,) scores in [-1, 1].
            """
            return doc_vecs @ query_vec


def quick_cosine(text_a: str, text_b: str) -> float | None:
    """Compute cosine similarity between two texts using ONNX embeddings.

    Returns float in [-1, 1] if ONNX available, None otherwise.
    Callers should fall back to word Jaccard when None is returned.
    """
    model = get_embedding_model()
    if model is None:
        return None
    try:
        vecs = model.embed_batch([text_a, text_b])
        return float(vecs[0] @ vecs[1])
    except Exception:
        return None


def save_embeddings(embeddings: dict[str, list[float]], path: str) -> None:
    """Save embeddings to gzip-compressed JSON."""
    tmp = path + ".tmp"
    try:
        data = json.dumps(embeddings, separators=(",", ":"))
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        if os.path.isfile(tmp):
            os.unlink(tmp)
        raise


def load_embeddings(path: str) -> dict[str, list[float]]:
    """Load embeddings from gzip-compressed JSON. Returns empty dict on failure."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.loads(f.read())
    except (OSError, json.JSONDecodeError, gzip.BadGzipFile):
        return {}


def migrate_gzip_to_sqlite(
    gzip_path: str, db_path: str, namespace: str = "commit", dim: int = 384
) -> int:
    """Migrate embeddings from gzip JSON to sqlite-vec store.

    Returns number of vectors migrated. Skips if sqlite-vec unavailable.
    """
    from ccr.context.vec_store import SQLITE_VEC_AVAILABLE, get_vec_store

    if not SQLITE_VEC_AVAILABLE:
        return 0
    store = get_vec_store(db_path, dim=dim)
    if store is None:
        return 0
    cache = load_embeddings(gzip_path)
    if not cache:
        return 0
    count = 0
    for id_, vec in cache.items():
        store.upsert(id_, vec if isinstance(vec, list) else list(vec), namespace)
        count += 1
    return count
