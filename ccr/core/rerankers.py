"""Optional local reranker provider layer for evidence recall."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Protocol

from ccr.core.facts import lexical_score


class EvidenceLike(Protocol):
    title: str
    snippet: str
    score: float
    metadata: dict[str, Any]


@dataclass
class RerankerStatus:
    provider: str
    available: bool
    detail: str = ""


class BaseReranker:
    provider = "lexical"

    def status(self) -> RerankerStatus:
        return RerankerStatus(self.provider, True)

    def rerank(self, query: str, evidence: list[EvidenceLike]) -> list[EvidenceLike]:
        return evidence


class LexicalReranker(BaseReranker):
    provider = "lexical"

    def rerank(self, query: str, evidence: list[EvidenceLike]) -> list[EvidenceLike]:
        for ev in evidence:
            boost = lexical_score(query, f"{ev.title} {ev.snippet}") * 0.25
            ev.score = min(1.0, ev.score + boost)
            ev.metadata["reranker"] = self.provider
        return sorted(evidence, key=lambda ev: ev.score, reverse=True)


class SentenceTransformersReranker(BaseReranker):
    provider = "sentence-transformers"

    def __init__(self, model_name: str = ""):
        self.model_name = model_name or os.environ.get(
            "CCR_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self._model: Any | None = None
        self._error = ""
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._model = CrossEncoder(self.model_name)
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"

    def status(self) -> RerankerStatus:
        return RerankerStatus(self.provider, self._model is not None, self._error)

    def rerank(self, query: str, evidence: list[EvidenceLike]) -> list[EvidenceLike]:
        if self._model is None:
            return LexicalReranker().rerank(query, evidence)
        pairs = [(query, f"{ev.title}\n{ev.snippet}") for ev in evidence]
        scores = self._model.predict(pairs)
        for ev, raw in zip(evidence, scores):
            normalized = 1.0 / (1.0 + math.exp(-float(raw)))
            ev.score = min(1.0, (ev.score * 0.4) + (normalized * 0.6))
            ev.metadata["reranker"] = self.provider
        return sorted(evidence, key=lambda ev: ev.score, reverse=True)


class BGEReranker(SentenceTransformersReranker):
    provider = "bge-reranker"

    def __init__(self):
        super().__init__(os.environ.get("CCR_RERANKER_MODEL", "BAAI/bge-reranker-base"))


class FastEmbedReranker(BaseReranker):
    provider = "fastembed"

    def __init__(self):
        self._model: Any | None = None
        self._error = ""
        try:
            from fastembed import TextEmbedding  # type: ignore
            self._model = TextEmbedding(
                model_name=os.environ.get("CCR_RERANKER_MODEL", "BAAI/bge-small-en-v1.5")
            )
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"

    def status(self) -> RerankerStatus:
        return RerankerStatus(self.provider, self._model is not None, self._error)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def rerank(self, query: str, evidence: list[EvidenceLike]) -> list[EvidenceLike]:
        if self._model is None:
            return LexicalReranker().rerank(query, evidence)
        texts = [query] + [f"{ev.title}\n{ev.snippet}" for ev in evidence]
        vectors = [list(v) for v in self._model.embed(texts)]
        q = vectors[0]
        for ev, vec in zip(evidence, vectors[1:]):
            sim = (self._cosine(q, vec) + 1.0) / 2.0
            ev.score = min(1.0, (ev.score * 0.5) + (sim * 0.5))
            ev.metadata["reranker"] = self.provider
        return sorted(evidence, key=lambda ev: ev.score, reverse=True)


class OllamaEmbeddingReranker(BaseReranker):
    provider = "ollama"

    def status(self) -> RerankerStatus:
        return RerankerStatus(
            self.provider,
            False,
            "Ollama reranking is declared as a provider hook; use lexical fallback unless integrated.",
        )

    def rerank(self, query: str, evidence: list[EvidenceLike]) -> list[EvidenceLike]:
        return LexicalReranker().rerank(query, evidence)


def get_reranker(provider: str | None = None) -> BaseReranker:
    name = (provider or os.environ.get("CCR_RERANKER", "lexical")).strip().lower()
    if name in {"", "none", "lexical"}:
        return LexicalReranker()
    if name in {"sentence-transformers", "sentence_transformers", "cross-encoder"}:
        rr = SentenceTransformersReranker()
        return rr if rr.status().available else LexicalReranker()
    if name in {"bge", "bge-reranker"}:
        rr = BGEReranker()
        return rr if rr.status().available else LexicalReranker()
    if name == "fastembed":
        rr = FastEmbedReranker()
        return rr if rr.status().available else LexicalReranker()
    if name == "ollama":
        return OllamaEmbeddingReranker()
    return LexicalReranker()


def provider_status() -> list[RerankerStatus]:
    providers: list[BaseReranker] = [
        LexicalReranker(),
        FastEmbedReranker(),
        SentenceTransformersReranker(),
        BGEReranker(),
        OllamaEmbeddingReranker(),
    ]
    return [p.status() for p in providers]
