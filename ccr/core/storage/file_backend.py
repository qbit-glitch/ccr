"""Flat-file storage backend — delegates to existing CCR I/O classes.

This is a thin adapter: all behaviour remains identical to the pre-v4
flat-file approach. It exists so that the storage abstraction layer has
a concrete FileStorageBackend that can be swapped for SqliteStorageBackend.

Method implementations are split across phase-specific mixin files:
  _file_phase1.py  — scratchpad, metrics, log, metadata
  _file_phase2.py  — playbook bullets, sections, failure lessons, schema, audit
  _file_phase3a.py — commits, rolling summaries, branches
  _file_phase3b.py — links, patterns, triples, evolved summaries, clusters
  _file_phase3c.py — discussions, session/phase summaries, summary meta, project state
"""

from __future__ import annotations

import logging
import threading

from ccr.core.storage._file_phase1 import FilePhase1Mixin
from ccr.core.storage._file_phase2 import FilePhase2Mixin
from ccr.core.storage._file_phase3a import FilePhase3aMixin
from ccr.core.storage._file_phase3b import FilePhase3bMixin
from ccr.core.storage._file_phase3c import FilePhase3cMixin
from ccr.core.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class FileStorageBackend(
    FilePhase1Mixin, FilePhase2Mixin, FilePhase3aMixin, FilePhase3bMixin, FilePhase3cMixin,
    StorageBackend,
):
    """Delegates to flat-file I/O (existing .ccr/ layout)."""

    def __init__(self, ccr_root: str, global_ccr_root: str | None = None):
        super().__init__(ccr_root, global_ccr_root)
        self._lock = threading.Lock()

    @property
    def backend_type(self) -> str:
        return "files"
