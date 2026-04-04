"""memory_pkg — Mixin-based decomposition of MemoryManager."""

from __future__ import annotations

from ccr.core.memory_pkg.memory_types import (
    COMMITS_TEMPLATE,
    MAIN_COMMITS_TEMPLATE,
    MAIN_MD_TEMPLATE,
    METADATA_TEMPLATE,
    REGISTRY_TEMPLATE,
    SUMMARY_TEMPLATE,
    EvolvedSummary,
)
from ccr.core.memory_pkg.memory_file_io import FileIOMixin
from ccr.core.memory_pkg.memory_init import InitMixin
from ccr.core.memory_pkg.memory_registry import RegistryMixin
from ccr.core.memory_pkg.memory_embeddings import EmbeddingsMixin
from ccr.core.memory_pkg.memory_evolution import EvolutionMixin
from ccr.core.memory_pkg.memory_ota import OTAMixin
from ccr.core.memory_pkg.memory_patterns import PatternsMixin
from ccr.core.memory_pkg.memory_admission import AdmissionMixin
from ccr.core.memory_pkg.memory_links import LinksMixin
from ccr.core.memory_pkg.memory_rolling_summary import RollingSummaryMixin
from ccr.core.memory_pkg.memory_commit import CommitMixin
from ccr.core.memory_pkg.memory_branch_ops import BranchOpsMixin
from ccr.core.memory_pkg.memory_consolidation import ConsolidationMixin
from ccr.core.memory_pkg.memory_context import ContextMixin

__all__ = [
    "COMMITS_TEMPLATE",
    "MAIN_COMMITS_TEMPLATE",
    "MAIN_MD_TEMPLATE",
    "METADATA_TEMPLATE",
    "REGISTRY_TEMPLATE",
    "SUMMARY_TEMPLATE",
    "EvolvedSummary",
    "FileIOMixin",
    "InitMixin",
    "RegistryMixin",
    "EmbeddingsMixin",
    "EvolutionMixin",
    "OTAMixin",
    "PatternsMixin",
    "AdmissionMixin",
    "LinksMixin",
    "RollingSummaryMixin",
    "CommitMixin",
    "BranchOpsMixin",
    "ConsolidationMixin",
    "ContextMixin",
]
