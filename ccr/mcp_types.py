"""TypedDict return types for all MCP tools — enables outputSchema (MCP spec 2025-06-18).

FastMCP auto-generates ``outputSchema`` from TypedDict return annotations and
produces ``structuredContent`` alongside the legacy text ``content``.  Each
TypedDict includes a ``message`` field (str) that carries the same human-readable
text previously returned as a plain string, ensuring backward compatibility.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


# ===========================================================================
# GCC Memory Tools
# ===========================================================================


class GccCommitResult(TypedDict):
    commit_id: str
    branch: str
    title: str
    admission_decision: str  # "created" | "merged" | "rejected"
    message: str


class GccBranchResult(TypedDict):
    branch: str
    message: str


class GccMergeResult(TypedDict):
    source: str
    target: str
    message: str


class GccContextResult(TypedDict):
    level: int
    branch: str
    message: str


class GccLinksResult(TypedDict):
    commit_id: str
    links_found: int
    message: str


class GccEvolveMemoryResult(TypedDict):
    evolutions: int
    message: str


class GccLogOtaResult(TypedDict):
    message: str


class GccStatusResult(TypedDict):
    branch: str
    total_commits: int
    message: str


class GccConsolidateResult(TypedDict):
    tier: str
    message: str




class GccPatternsResult(TypedDict):
    total: int
    matching: int
    message: str


class GccClustersResult(TypedDict):
    cluster_count: int
    message: str


class GccTriplesResult(TypedDict):
    count: int
    message: str


class GccScratchpadResult(TypedDict):
    mode: str  # "get" | "set" | "clear"
    key: NotRequired[str]
    cleared: NotRequired[int]
    message: str


# ===========================================================================
# ACE Playbook Tools
# ===========================================================================


class AcePlaybookResult(TypedDict):
    global_bullet_count: int
    project_bullet_count: int
    message: str


class AceApplyDeltaResult(TypedDict):
    applied: int
    scope: str
    message: str
    failed_ids: NotRequired[list[str]]
    delta_history_path: NotRequired[str]
    author: NotRequired[str]


class AceUpdateCountersResult(TypedDict):
    updated: int
    scope: str
    missing_ids: NotRequired[list[str]]
    message: str




class AceFindSimilarResult(TypedDict):
    pairs_found: int
    scope: str
    pairs: NotRequired[list[dict]]
    message: str


class AcePruneResult(TypedDict):
    removed: int
    evolved: int
    scope: str
    removed_ids: NotRequired[list[str]]
    message: str


class AceGenerateBulletsResult(TypedDict):
    decisions: int
    applied: int
    message: str
    pending_decisions: NotRequired[list[dict]]


class AceEvolveFromFailuresResult(TypedDict):
    evolved: int
    synthesized: int
    message: str


class AceEvolveSchemaResult(TypedDict):
    version: int
    health: NotRequired[float]
    message: str


# ===========================================================================
# RLM Sandbox Tools
# ===========================================================================


class RlmInitResult(TypedDict):
    session_id: str
    file_count: int
    session_replaced: bool
    message: str


class RlmExecuteResult(TypedDict):
    has_error: bool
    has_final_answer: bool
    message: str


class RlmFinalizeResult(TypedDict):
    variable_name: str
    message: str
    variables: NotRequired[dict]


# ===========================================================================
# Repo Index Tools
# ===========================================================================


class IndexBuildResult(TypedDict):
    files_indexed: int
    message: str


class IndexSearchResult(TypedDict):
    result_count: int
    mode: str
    message: str


class IndexStatusResult(TypedDict):
    built_at: str           # ISO timestamp or "never"
    file_count: int
    embeddings_available: bool
    bm25_cache_built: bool
    chunk_embeddings_available: bool
    chunk_count: int
    is_stale: bool
    message: str
