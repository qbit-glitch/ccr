"""Dense baseline: all-MiniLM-L6-v2 (CCR's shipped ONNX model) with cosine ranking.

The ONNX model truncates each text at 256 tokens, so long docs are embedded from
their first 256 tokens only. This is a property of the baseline, documented in
``docs/benchmarks.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ccr.bench.retrievers.base import RetrieverUnavailable, register
from ccr.bench.types import Doc
from ccr.context.embeddings import get_embedding_model

BATCH = 64


def _as_lists(vectors: object) -> list[list[float]]:
    tolist = getattr(vectors, "tolist", None)
    return tolist() if callable(tolist) else [list(v) for v in vectors]  # type: ignore[union-attr]


def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


@register("dense_minilm")
class DenseMiniLmRetriever:
    """Cosine similarity over MiniLM embeddings of whole docs."""

    name = "dense_minilm"
    max_k: int | None = None

    def __init__(self) -> None:
        self.empty_count = 0
        self._ids: list[str] = []
        self._vecs: list[list[float]] = []
        self._model = None

    def index(self, docs: Sequence[Doc], workdir: str) -> None:
        self._model = get_embedding_model()
        if self._model is None:
            raise RetrieverUnavailable(
                'dense_minilm needs onnxruntime/tokenizers/numpy: pip install -e ".[semantic]"'
            )
        self._ids = [d.id for d in docs]
        texts = [d.text for d in docs]
        vecs: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            vecs.extend(_as_lists(self._model.embed_batch(texts[i : i + BATCH])))
        self._vecs = [_unit(v) for v in vecs]

    def retrieve(self, query: str, k: int) -> list[str]:
        if self._model is None:
            raise RuntimeError("index() must be called before retrieve()")
        if not self._ids:
            self.empty_count += 1
            return []
        q = _unit(_as_lists(self._model.embed_batch([query]))[0])
        scored = sorted(
            ((sum(a * b for a, b in zip(q, v)), i) for i, v in enumerate(self._vecs)),
            key=lambda pair: (-pair[0], pair[1]),
        )
        return [self._ids[i] for _, i in scored[:k]]

    def close(self) -> None:
        self._vecs = []
        self._model = None
