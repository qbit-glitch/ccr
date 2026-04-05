"""GCC-style version-controlled memory management.

Thin composition facade over ccr.core.memory_pkg mixins.
All method implementations live in the individual mixin modules under memory_pkg/.

GCC paper features:
- COMMIT/BRANCH/MERGE/CONTEXT operations
- metadata.yaml (file trees, deps, config)
- summary.md per branch
- OTA logging (Observation-Thought-Action triples)
- Context windowing (scrollable K-window)
- Auto-CONTEXT before MERGE
- Log rotation
"""

from __future__ import annotations

from ccr.core.memory_pkg import (
    AdmissionMixin,
    BranchOpsMixin,
    CommitMixin,
    ConsolidationMixin,
    ContextMixin,
    DiscussionsMixin,
    EmbeddingsMixin,
    EvolvedSummary,
    EvolutionMixin,
    ExperimentsMixin,
    FileIOMixin,
    InitMixin,
    LinksMixin,
    OTAMixin,
    PatternsMixin,
    RegistryMixin,
    RollingSummaryMixin,
)
from ccr.core.types import CCRConfig

__all__ = ["MemoryManager", "EvolvedSummary"]


class MemoryManager(
    FileIOMixin,
    InitMixin,
    RegistryMixin,
    EmbeddingsMixin,
    EvolutionMixin,
    OTAMixin,
    PatternsMixin,
    AdmissionMixin,
    LinksMixin,
    RollingSummaryMixin,
    CommitMixin,
    BranchOpsMixin,
    ConsolidationMixin,
    ContextMixin,
    ExperimentsMixin,
    DiscussionsMixin,
):
    """Manages the .ccr/ directory for a project.

    All file I/O is synchronous. Thread-safe via per-file locks.
    Method implementations are in ccr/core/memory_pkg/ mixin modules.
    """

    def __init__(self, project_root: str, config: CCRConfig | None = None):
        self._init_state(project_root, config)
