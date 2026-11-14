"""CCR memory retriever: one commit per doc, recalled via ``RecallEngine``."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from ccr.bench.retrievers.base import register
from ccr.bench.types import Doc
from ccr.core.memory import MemoryManager
from ccr.core.recall import RecallEngine

logger = logging.getLogger(__name__)

RECALL_CAP = 25
_NON_WORD = re.compile(r"[^\w\s]")


@register("ccr_memory")
class CcrMemoryRetriever:
    """Ingest each doc as a commit titled with its id, retrieve via recall.

    Commit admission merging is disabled (threshold 1.0) so no gold evidence is
    merged away. Only commit-sourced evidence is mapped back to doc ids because
    facts and graph evidence do not carry a doc id.
    """

    name = "ccr_memory"
    max_k: int | None = RECALL_CAP

    def __init__(self) -> None:
        self.empty_count = 0
        self._mem: MemoryManager | None = None
        self._engine: RecallEngine | None = None

    def index(self, docs: Sequence[Doc], workdir: str) -> None:
        self._mem = MemoryManager(workdir)
        for doc in docs:
            self._mem.commit(
                title=doc.id,
                what=doc.text,
                why="bench",
                files_changed=[],
                next_step="",
                admission_threshold=1.0,
                rejection_threshold=0.0,
            )
        self._engine = RecallEngine(self._mem)

    def retrieve(self, query: str, k: int) -> list[str]:
        if k > RECALL_CAP:
            raise ValueError(f"ccr_memory supports k <= {RECALL_CAP}, got {k}")
        if self._engine is None:
            raise RuntimeError("index() must be called before retrieve()")
        ids = self._recall(query, k)
        if not ids:
            cleaned = _NON_WORD.sub(" ", query).strip()
            if cleaned and cleaned != query:
                ids = self._recall(cleaned, k)
        if not ids:
            self.empty_count += 1
        return ids

    def _recall(self, query: str, k: int) -> list[str]:
        assert self._engine is not None
        if not query.strip():
            return []
        try:
            result = self._engine.recall(
                query, limit=k, sources=["commits"], include_sessions=False
            )
        except Exception as exc:  # recall surfaces storage/FTS errors untyped
            logger.warning("recall failed for query %r: %s", query[:60], exc)
            return []
        seen: set[str] = set()
        ranked: list[str] = []
        for ev in result.evidence:
            if ev.source == "commit" and ev.title not in seen:
                seen.add(ev.title)
                ranked.append(ev.title)
        return ranked

    def close(self) -> None:
        self._engine = None
        self._mem = None
