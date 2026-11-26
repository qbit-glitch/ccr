"""CCR ``RepoIndex`` retriever: each doc becomes a ``.txt`` file in a temp corpus."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

from ccr.bench.retrievers.base import REGISTRY, RetrieverUnavailable
from ccr.bench.types import Doc
from ccr.context.embeddings import get_embedding_model
from ccr.context.indexer import RepoIndex

logger = logging.getLogger(__name__)

INDEX_CAP = 100
MODES = ("keyword", "bm25", "hybrid")


class CcrIndexRetriever:
    """Wraps ``RepoIndex`` search in one of three modes.

    ``hybrid`` needs ONNX embeddings; without them it raises
    ``RetrieverUnavailable`` instead of silently degrading to BM25.
    """

    max_k: int | None = INDEX_CAP

    def __init__(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        self.mode = mode
        self.name = f"ccr_index_{mode}"
        self.empty_count = 0
        self.indexed_count = 0
        self._idx: RepoIndex | None = None
        self._model = None
        self._file_to_id: dict[str, str] = {}

    def index(self, docs: Sequence[Doc], workdir: str) -> None:
        if self.mode == "hybrid":
            self._model = get_embedding_model()
            if self._model is None:
                raise RetrieverUnavailable(
                    "ccr_index_hybrid needs onnxruntime/tokenizers/numpy: "
                    'pip install -e ".[semantic]"'
                )
        corpus = os.path.join(workdir, "corpus")
        os.makedirs(corpus, exist_ok=True)
        self._file_to_id = {}
        for n, doc in enumerate(docs):
            rel = f"docs/{n}.txt"
            os.makedirs(os.path.join(corpus, "docs"), exist_ok=True)
            with open(os.path.join(corpus, rel), "w", encoding="utf-8") as fh:
                fh.write(doc.text)
            self._file_to_id[rel] = doc.id
        self._idx = RepoIndex.build(corpus, extensions={".txt"})
        self.indexed_count = len(self._idx.files)
        if self.mode == "hybrid":
            self._idx.build_embeddings(self._model)

    def retrieve(self, query: str, k: int) -> list[str]:
        if k > INDEX_CAP:
            raise ValueError(f"{self.name} supports k <= {INDEX_CAP}, got {k}")
        if self._idx is None:
            raise RuntimeError("index() must be called before retrieve()")
        if self.mode == "keyword":
            hits = self._idx.search(query)
        elif self.mode == "bm25":
            hits = self._idx.bm25_search(query, top_k=k)
        else:
            hits = self._idx.hybrid_search(query, self._model, top_k=k)
        ranked: list[str] = []
        for hit in hits:
            doc_id = self._file_to_id.get(str(hit.get("path", "")))
            if doc_id is not None and doc_id not in ranked:
                ranked.append(doc_id)
        ranked = ranked[:k]
        if not ranked:
            self.empty_count += 1
        return ranked

    def close(self) -> None:
        self._idx = None
        self._model = None


for _mode in MODES:
    REGISTRY[f"ccr_index_{_mode}"] = lambda m=_mode: CcrIndexRetriever(m)  # type: ignore[assignment]
