"""Full-context baseline: hand the reader every document (accuracy ceiling)."""

from __future__ import annotations

from collections.abc import Sequence

from ccr.bench.retrievers.base import register
from ccr.bench.types import Doc


@register("full_context")
class FullContextRetriever:
    """Returns every doc id in corpus order, ignoring the query and ``k``."""

    name = "full_context"
    max_k: int | None = None
    ranks_docs = False  # returns everything; recall/nDCG would be meaningless

    def __init__(self) -> None:
        self.empty_count = 0
        self._ids: list[str] = []

    def index(self, docs: Sequence[Doc], workdir: str) -> None:
        self._ids = [d.id for d in docs]

    def retrieve(self, query: str, k: int) -> list[str]:
        return list(self._ids)

    def close(self) -> None:
        self._ids = []
