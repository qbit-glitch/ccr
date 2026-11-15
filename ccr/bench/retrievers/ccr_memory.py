"""CCR memory retriever: one commit per doc, recalled via ``RecallEngine``."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from functools import partial

from ccr.bench.retrievers.base import register
from ccr.bench.types import Doc
from ccr.core.memory import MemoryManager
from ccr.core.recall import RecallEngine, RecallOptions

logger = logging.getLogger(__name__)

RECALL_CAP = 25
_NON_WORD = re.compile(r"[^\w\s]")


class CcrMemoryRetriever:
    """Ingest each doc as a commit titled with an opaque handle, retrieve via recall.

    Commit admission merging is disabled (threshold 1.0) so no gold evidence is
    merged away. Only commit-sourced evidence is mapped back to doc ids because
    facts and graph evidence do not carry a doc id.
    """

    max_k: int | None = RECALL_CAP

    def __init__(self, options: RecallOptions | None = None, name: str = "ccr_memory") -> None:
        self.name = name
        self.options = options or RecallOptions()
        self.empty_count = 0
        self._mem: MemoryManager | None = None
        self._engine: RecallEngine | None = None
        self._ids: dict[str, str] = {}

    def index(self, docs: Sequence[Doc], workdir: str) -> None:
        self._mem = MemoryManager(workdir)
        # Titles are opaque: real doc ids can encode gold labels (e.g. LongMemEval
        # ``answer_*``) and commit titles are searchable text.
        self._ids = {}
        for n, doc in enumerate(docs):
            self._ids[f"doc-{n}"] = doc.id
            self._mem.commit(
                title=f"doc-{n}",
                what=doc.text,
                why="bench",
                files_changed=[],
                next_step="",
                admission_threshold=1.0,
                rejection_threshold=0.0,
            )
        self._engine = RecallEngine(self._mem, options=self.options)

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
            doc_id = self._ids.get(ev.title)
            if ev.source == "commit" and doc_id is not None and doc_id not in seen:
                seen.add(doc_id)
                ranked.append(doc_id)
        return ranked

    def close(self) -> None:
        self._engine = None
        self._mem = None


# One system per retrieval configuration. ``ccr_memory`` is the shipped default;
# the others exist so a report can show before/after and each ablation.
VARIANTS: dict[str, RecallOptions] = {
    "ccr_memory": RecallOptions(),
    "ccr_memory_legacy": RecallOptions(commit_search="legacy"),
    "ccr_memory_bm25": RecallOptions(commit_search="bm25"),
    "ccr_memory_dense": RecallOptions(commit_search="dense"),
    "ccr_memory_norerank": RecallOptions(rerank=False),
    "ccr_memory+links": RecallOptions(link_expansion=True),
    "ccr_memory+scenes": RecallOptions(scene_expansion=True),
}

for _name, _options in VARIANTS.items():
    register(_name)(partial(CcrMemoryRetriever, _options, _name))  # type: ignore[arg-type]
