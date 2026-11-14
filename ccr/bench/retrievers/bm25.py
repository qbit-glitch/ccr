"""Standalone Okapi BM25 baseline, independent of CCR's own indexer."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from ccr.bench.retrievers.base import register
from ccr.bench.types import Doc

_WORD = re.compile(r"\w+")
K1 = 1.5
B = 0.75


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@register("bm25")
class Bm25Retriever:
    """Okapi BM25 (k1=1.5, b=0.75) over lowercase word tokens."""

    name = "bm25"
    max_k: int | None = None

    def __init__(self) -> None:
        self.empty_count = 0
        self._ids: list[str] = []
        self._tf: list[Counter[str]] = []
        self._len: list[int] = []
        self._df: Counter[str] = Counter()
        self._avg_len = 0.0

    def index(self, docs: Sequence[Doc], workdir: str) -> None:
        self._ids = [d.id for d in docs]
        self._tf = [Counter(_tokens(d.text)) for d in docs]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._df = Counter(term for tf in self._tf for term in tf)
        self._avg_len = sum(self._len) / len(self._len) if self._len else 0.0

    def retrieve(self, query: str, k: int) -> list[str]:
        terms = _tokens(query)
        if not self._ids or not terms:
            self.empty_count += 1
            return []
        n_docs = len(self._ids)
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                df = self._df[term]
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                norm = freq + K1 * (1 - B + B * self._len[i] / (self._avg_len or 1.0))
                score += idf * freq * (K1 + 1) / norm
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        if not scored:
            self.empty_count += 1
        return [self._ids[i] for _, i in scored[:k]]

    def close(self) -> None:
        self._tf = []
