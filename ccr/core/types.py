"""CCR shared types — all inter-layer data contracts."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ComplexityTier(str, Enum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None


@dataclass
class ModelUsage:
    model: str = ""
    calls: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)

    def record(self, input_tokens: int, output_tokens: int, cost: float | None = None) -> None:
        self.calls += 1
        self.usage.input_tokens += input_tokens
        self.usage.output_tokens += output_tokens
        self.usage.total_tokens += input_tokens + output_tokens
        if cost is not None:
            self.usage.cost_usd = (self.usage.cost_usd or 0.0) + cost


@dataclass
class SessionUsage:
    models: dict[str, ModelUsage] = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: datetime = field(default_factory=datetime.now)

    @property
    def total_cost(self) -> float:
        return sum(
            m.usage.cost_usd for m in self.models.values() if m.usage.cost_usd is not None
        )

    @property
    def total_tokens(self) -> int:
        return sum(m.usage.total_tokens for m in self.models.values())

    @property
    def total_calls(self) -> int:
        return sum(m.calls for m in self.models.values())

    def record(self, model: str, input_tokens: int, output_tokens: int, cost: float | None = None) -> None:
        if model not in self.models:
            self.models[model] = ModelUsage(model=model)
        self.models[model].record(input_tokens, output_tokens, cost)


@dataclass
class TaskClassification:
    tier: ComplexityTier
    confidence: float
    reasoning: str
    estimated_tokens: int = 0
    packed_tokens: int | None = None


@dataclass
class ContextPack:
    task_description: str
    files: list[tuple[str, str]]  # (rel_path, content_slice)
    symbols: list[str]
    memory_context: str
    total_tokens: int
    pack_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_prompt_text(self) -> str:
        parts = []
        if self.memory_context:
            parts.append(f"<project_context>\n{self.memory_context}\n</project_context>")
        for path, content in self.files:
            parts.append(f"<file path=\"{path}\">\n{content}\n</file>")
        return "\n\n".join(parts)


@dataclass
class CCRRequest:
    original_messages: list[dict[str, Any]]
    system_prompt: str | None
    model_requested: str
    max_tokens: int
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    received_at: datetime = field(default_factory=datetime.now)
    raw_body: bytes = b""

    @property
    def last_user_message(self) -> str:
        for msg in reversed(self.original_messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        b.get("text", "") for b in content if b.get("type") == "text"
                    )
        return ""


@dataclass
class CCRResponse:
    content: str
    model_used: str
    classification: TaskClassification
    context_pack: ContextPack | None
    usage: TokenUsage
    request_id: str
    routed_to: str  # "qwen_direct", "qwen_with_context", "claude_with_pack", "passthrough"


@dataclass
class RouteDecision:
    target: str
    context_pack: ContextPack | None = None
    memory_context_level: int = 0
    should_autocommit: bool = False
    use_rlm: bool = False


# --- RLM types ---

class HookEvent(str, Enum):
    SESSION_START = "SessionStart"
    POST_TOOL_USE = "PostToolUse"
    PRE_COMPACT = "PreCompact"
    POST_TASK_COMPLETE = "PostTaskComplete"
    STOP = "Stop"


@dataclass
class REPLResult:
    stdout: str = ""
    stderr: str = ""
    locals_snapshot: dict[str, str] = field(default_factory=dict)
    execution_time: float = 0.0
    final_answer: str | None = None
    error: str | None = None


@dataclass
class RLMConfig:
    max_depth: int = 2
    max_iterations: int = 15
    max_timeout_seconds: float = 120.0
    max_budget_usd: float | None = None
    max_consecutive_errors: int = 5
    compaction_threshold: int = 0  # 0 = auto (80% of model context)
    max_total_tokens: int = 0  # 0 = no limit


@dataclass
class RLMResult:
    response: str
    iterations_used: int = 0
    depth: int = 0
    execution_time: float = 0.0
    usage: TokenUsage = field(default_factory=TokenUsage)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    final_answer_source: str = ""  # "FINAL_VAR", "FINAL", "default", "text"


@dataclass
class CommitLink:
    """A heuristic cross-link between two GCC commits.

    Taxonomy inspired by A-MEM (bidirectional links) and MAGMA (typed edges),
    but uses mechanical heuristics instead of the papers' LLM inference and
    dense vector embeddings. See CLAUDE.md "Limitations vs. Paper" for details.

    Link types:
        entity: Shared files touched (file-set Jaccard; cf. MAGMA entity graph
                which uses LLM-extracted abstract entity nodes)
        causal: Explicit commit ID reference in text (regex detection; cf. MAGMA
                causal graph which uses LLM-inferred logical entailment)
        supersession: Replacement language + commit ID (heuristic; no paper analog)
        semantic: Keyword overlap above threshold (word Jaccard; cf. MAGMA semantic
                  graph which uses dense vector cosine similarity)
    """
    target: str           # Commit ID (e.g., "C005")
    link_type: str        # "entity" | "causal" | "supersession" | "semantic"
    score: float = 0.0    # Similarity/relevance score (0-1)
    shared_files: list[str] = field(default_factory=list)  # Entity links only
    snippet: str = ""     # Triggering text for causal/supersession

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"target": self.target, "score": self.score}
        if self.shared_files:
            d["shared_files"] = self.shared_files
        if self.snippet:
            d["snippet"] = self.snippet
        return d

    @classmethod
    def from_dict(cls, link_type: str, d: dict) -> "CommitLink":
        return cls(
            target=d["target"], link_type=link_type,
            score=d.get("score", 0.0),
            shared_files=d.get("shared_files", []),
            snippet=d.get("snippet", ""),
        )


@dataclass
class PatternEntry:
    """A transferable decision-making pattern (CER S_i / skill).

    Inspired by CER (arXiv:2506.06698) §3.1: abstract, parameterized,
    reusable skills distilled from task trajectories. Uses {curly_braces}
    for non-fixed elements (CER Fig 4).

    Unlike CER's LLM distiller, patterns here are provided by Claude Code
    at commit time (zero additional LLM calls).
    """
    text: str
    first_seen: str           # Commit ID (e.g., "C003")
    commit_ids: list[str] = field(default_factory=list)
    occurrence_count: int = 1
    created_at: str = ""      # ISO-8601
    last_seen: str = ""       # ISO-8601 timestamp of most recent occurrence
    promoted: bool = False    # True after promotion suggestion delivered
    # Quality scoring (EvolveR-inspired, arXiv:2510.16079)
    success_count: int = 0       # Times associated promoted bullet tagged helpful
    failure_count: int = 0       # Times associated promoted bullet tagged harmful
    quality_score: float = 0.5   # Bayesian: (success+1) / (success+failure+2)
    last_quality_update: str = ""  # ISO-8601

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "first_seen": self.first_seen,
            "commit_ids": list(self.commit_ids),
            "occurrence_count": self.occurrence_count,
            "created_at": self.created_at,
            "last_seen": self.last_seen,
            "promoted": self.promoted,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "quality_score": self.quality_score,
            "last_quality_update": self.last_quality_update,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PatternEntry":
        return cls(
            text=d["text"],
            first_seen=d["first_seen"],
            commit_ids=d.get("commit_ids", []),
            occurrence_count=d.get("occurrence_count", 1),
            created_at=d.get("created_at", ""),
            last_seen=d.get("last_seen", ""),
            promoted=d.get("promoted", False),
            success_count=d.get("success_count", 0),
            failure_count=d.get("failure_count", 0),
            quality_score=d.get("quality_score", 0.5),
            last_quality_update=d.get("last_quality_update", ""),
        )


@dataclass
class SchemaMetrics:
    """Health metrics for a playbook schema (MCE evaluation function J).

    All mechanical — computed from bullet data, no LLM calls.
    MCE §3.2 (arXiv:2601.21557): J(s, D) evaluates skill s on dataset D.
    """
    section_balance: float = 0.0      # Normalized Shannon entropy [0,1]
    utilization_rate: float = 0.0     # Fraction of bullets with helpful+harmful > 0
    harmful_ratio: float = 0.0        # Of utilized, fraction net-harmful
    unused_ratio: float = 0.0         # Fraction with helpful+harmful == 0
    decay_impact: float = 0.0         # Fraction with effective_score < 50% of raw
    empty_sections: list[str] = field(default_factory=list)
    overflow_sections: list[str] = field(default_factory=list)
    overall_health: float = 0.0       # Weighted composite [0,1]
    total_bullets: int = 0
    total_sections: int = 0
    timestamp: str = ""               # ISO-8601
    # ALMA-inspired memory retrieval metrics (MCE §3.9)
    search_zero_rate: float = 0.0     # Fraction of searches returning zero results
    link_density: float = 0.0         # Average links per commit
    embedding_coverage: float = 0.0   # Fraction of commits with embeddings

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_balance": self.section_balance,
            "utilization_rate": self.utilization_rate,
            "harmful_ratio": self.harmful_ratio,
            "unused_ratio": self.unused_ratio,
            "decay_impact": self.decay_impact,
            "empty_sections": list(self.empty_sections),
            "overflow_sections": list(self.overflow_sections),
            "overall_health": self.overall_health,
            "total_bullets": self.total_bullets,
            "total_sections": self.total_sections,
            "timestamp": self.timestamp,
            "search_zero_rate": self.search_zero_rate,
            "link_density": self.link_density,
            "embedding_coverage": self.embedding_coverage,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SchemaMetrics":
        return cls(
            section_balance=d.get("section_balance", 0.0),
            utilization_rate=d.get("utilization_rate", 0.0),
            harmful_ratio=d.get("harmful_ratio", 0.0),
            unused_ratio=d.get("unused_ratio", 0.0),
            decay_impact=d.get("decay_impact", 0.0),
            empty_sections=d.get("empty_sections", []),
            overflow_sections=d.get("overflow_sections", []),
            overall_health=d.get("overall_health", 0.0),
            total_bullets=d.get("total_bullets", 0),
            total_sections=d.get("total_sections", 0),
            timestamp=d.get("timestamp", ""),
            search_zero_rate=d.get("search_zero_rate", 0.0),
            link_density=d.get("link_density", 0.0),
            embedding_coverage=d.get("embedding_coverage", 0.0),
        )


@dataclass
class SchemaProposal:
    """A single schema change proposal (MCE offspring s_k in (1+1)-ES).

    Part of the (1+1)-ES flow (MCE §3.1): generate one proposal,
    compare metrics before/after. Claude Code decides whether to apply.
    """
    change_type: str          # ADD_SECTION, REMOVE_SECTION, ADJUST_DECAY, etc.
    description: str          # Human-readable explanation
    details: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0   # 0-1 heuristic confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_type": self.change_type,
            "description": self.description,
            "details": self.details,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SchemaProposal":
        return cls(
            change_type=d["change_type"],
            description=d.get("description", ""),
            details=d.get("details", {}),
            confidence=d.get("confidence", 0.0),
        )


@dataclass
class PlaybookSchema:
    """Versioned playbook schema (MCE skill s_k).

    Contains sections + thresholds governing playbook behavior.
    Part of schema history H = {(s_i, metrics_i)} in playbook_schema.json.
    MCE Algorithm 1 (arXiv:2601.21557): evolve skill → execute → evaluate → update H.
    """
    version: int = 1
    sections: list[str] = field(default_factory=list)
    slug_map: dict[str, str] = field(default_factory=dict)
    decay_rate: float = 0.95
    prune_min_harmful: int = 3
    evolution_threshold: int = 3
    token_budget: int = 80000
    parent_version: int | None = None
    change_description: str = ""
    created_at: str = ""
    baseline_metrics: SchemaMetrics | None = None
    # ALMA-inspired memory retrieval parameters (MCE §3.9)
    link_scan_window: int = 20
    link_semantic_threshold: float = 0.3
    context_level_default: int = 2
    search_result_limit: int = 5

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": self.version,
            "sections": list(self.sections),
            "slug_map": dict(self.slug_map),
            "decay_rate": self.decay_rate,
            "prune_min_harmful": self.prune_min_harmful,
            "evolution_threshold": self.evolution_threshold,
            "token_budget": self.token_budget,
            "parent_version": self.parent_version,
            "change_description": self.change_description,
            "created_at": self.created_at,
            "link_scan_window": self.link_scan_window,
            "link_semantic_threshold": self.link_semantic_threshold,
            "context_level_default": self.context_level_default,
            "search_result_limit": self.search_result_limit,
        }
        if self.baseline_metrics is not None:
            d["baseline_metrics"] = self.baseline_metrics.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlaybookSchema":
        bm = d.get("baseline_metrics")
        return cls(
            version=d.get("version", 1),
            sections=d.get("sections", []),
            slug_map=d.get("slug_map", {}),
            decay_rate=d.get("decay_rate", 0.95),
            prune_min_harmful=d.get("prune_min_harmful", 3),
            evolution_threshold=d.get("evolution_threshold", 3),
            token_budget=d.get("token_budget", 80000),
            parent_version=d.get("parent_version"),
            change_description=d.get("change_description", ""),
            created_at=d.get("created_at", ""),
            baseline_metrics=SchemaMetrics.from_dict(bm) if bm else None,
            link_scan_window=d.get("link_scan_window", 20),
            link_semantic_threshold=d.get("link_semantic_threshold", 0.3),
            context_level_default=d.get("context_level_default", 2),
            search_result_limit=d.get("search_result_limit", 5),
        )

    @classmethod
    def default(cls) -> "PlaybookSchema":
        """Create default schema matching current hardcoded values."""
        from ccr.ace.playbook import DEFAULT_SECTIONS, _SLUG_MAP
        return cls(
            version=1,
            sections=list(DEFAULT_SECTIONS),
            slug_map=dict(_SLUG_MAP),
        )


@dataclass
class CCRConfig:
    """Memory layer config."""
    recent_commit_count: int = 3
    milestones_kept: int = 5
    log_max_lines: int = 500
    proactive_commits: bool = True
    index_max_file_size_kb: int = 500
    # Hierarchical summary configuration (TiMem §3.1-3.2 adapted)
    session_summary_interval: int = 5       # commits between session summaries (TiMem L2)
    phase_summary_interval: int = 20        # commits on main before auto phase summary (TiMem L3-4)
    session_summary_max_chars: int = 500    # max chars per session summary
    phase_summary_max_items: int = 5        # max key accomplishments per phase summary
    overview_staleness_threshold: int = 5   # phase summaries before suggesting overview regen
    # Heuristic commit cross-linking (A-MEM/MAGMA inspired taxonomy)
    link_scan_window: int = 20            # recent commits to scan for links
    link_semantic_threshold: float = 0.3  # min keyword Jaccard for semantic links
    link_entity_threshold: float = 0.0    # min file Jaccard for entity links (any shared file)
    link_max_results: int = 10            # max results from BFS link traversal
    # CER-inspired pattern buffer (arXiv:2506.06698)
    pattern_dedup_threshold: float = 0.7     # word Jaccard for dedup
    pattern_promotion_count: int = 3          # commits before suggesting ACE promotion
    pattern_max_buffer_size: int = 200        # max patterns in buffer
    # Phase 2: Auto-extract transferable patterns via sub-model (CER §3.2, opt-in)
    auto_extract_patterns: bool = False  # must opt in — adds latency to every commit
    # Session Logger: persist Q&A turns to SQLite for replay/debug/training-data
    session_logging_enabled: bool = True
    session_db_path: str = ""          # defaults to .ccr/sessions.db when empty
    session_fts_enabled: bool = True   # FTS5 full-text search across turns


@dataclass
class RouterConfig:
    trivial_token_threshold: int = 500
    simple_token_threshold: int = 2000
    pack_token_budget: int = 8000
    auto_commit_threshold: ComplexityTier = ComplexityTier.MODERATE
    complex_keywords: list[str] = field(default_factory=lambda: [
        "architecture", "refactor", "cross-file", "entire codebase",
        "design", "integrate", "all files", "system-wide", "restructure",
    ])
    # Regex patterns that indicate COMPLEX exploration tasks even if short
    complex_patterns: list[str] = field(default_factory=lambda: [
        r"understand\s+(the\s+|this\s+)?(codebase|project|repo|code|system|app)",
        r"(walk|go)\s+(me\s+)?through\s+(the\s+|this\s+)?(codebase|project|repo|code)",
        r"(explain|describe|summarize|overview)\s+(the\s+|this\s+)?(entire\s+)?(codebase|project|repo|architecture|system)",
        r"how\s+does\s+(the\s+|this\s+)?(entire\s+)?(project|codebase|system|app)\s+(work|function|operate)",
        r"(map|trace|analyze)\s+(the\s+|this\s+)?(codebase|project|architecture|dependencies|data\s*flow)",
        r"(what|how)\s+is\s+(the\s+|this\s+)?(project|codebase|app)\s+(structured|organized|laid\s+out)",
        r"give\s+(me\s+)?(a\s+)?(full|complete|high[- ]level)\s+(overview|summary|picture)",
    ])


@dataclass
class ACEConfig:
    """ACE (Agentic Context Engineering) config."""
    enabled: bool = True
    playbook_token_budget: int = 80000
    curator_frequency: int = 1
    max_reflection_rounds: int = 3
    prune_min_harmful: int = 3
    max_tokens: int = 4096
    playbook_path: str = ""  # auto-set to .ccr/playbook.txt
    refinement_frequency: int = 10  # run deduplication every N steps
    dedup_similarity_threshold: float = 0.6  # Jaccard threshold for candidate pairs
    # MCE-inspired schema evolution (arXiv:2601.21557)
    schema_evolution_enabled: bool = True
    overflow_threshold: float = 0.5       # Section fraction triggering overflow
    min_cluster_size: int = 3             # Min bullets in OTHERS to propose new section
    stop_health_threshold: float = 0.8    # "Healthy enough" → no proposals
    rollback_health_delta: float = -0.05  # Health drop triggering rollback suggestion
    schema_max_history: int = 20          # Max schema versions to retain


@dataclass
class CCREngineConfig:
    memory: CCRConfig = field(default_factory=CCRConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    rlm: RLMConfig = field(default_factory=RLMConfig)
    ace: ACEConfig = field(default_factory=ACEConfig)
    claude_model: str = "claude-sonnet-4-5-20250514"
    sub_model: str = "openai/gpt-oss-20b"
    sub_model_base_url: str = "http://localhost:8000/v1"
    pack_token_budget: int = 8000
    anthropic_api_key: str = ""
    anthropic_real_base_url: str = "https://api.anthropic.com"
    sub_model_api_key: str | None = None
    gateway_host: str = "127.0.0.1"
    gateway_port: int = 7447
    passthrough_on_error: bool = True
    pre_compact_char_threshold: int = 50000  # ~12K tokens — triggers PreCompact hook
    max_search_candidates: int = 50  # max files from index search
    min_relevance_score: float = 0.3  # min file relevance for packing
    index_extensions: list[str] = field(default_factory=lambda: [
        ".py", ".ts", ".js", ".tsx", ".jsx", ".go", ".rs", ".java",
        ".md", ".yaml", ".yml", ".toml", ".json", ".sh", ".c", ".cpp", ".h",
    ])

    def validate(self) -> list[str]:
        """Validate configuration and return list of error messages.

        Returns empty list if config is valid.
        Raises ConfigError if there are fatal (non-recoverable) issues.
        """
        from ccr.core.exceptions import ConfigError

        errors: list[str] = []
        warnings: list[str] = []

        # --- Required fields ---
        if not self.anthropic_api_key:
            errors.append(
                "anthropic_api_key: Required. Set ANTHROPIC_API_KEY env var or config file."
            )

        if not self.sub_model:
            errors.append("sub_model: Required. Specify a sub-model name.")

        if not self.sub_model_base_url:
            errors.append("sub_model_base_url: Required. Specify the sub-model API URL.")

        # --- Format checks ---
        if self.anthropic_api_key and not self.anthropic_api_key.startswith(("sk-ant-", "sk-")):
            warnings.append(
                "anthropic_api_key: Doesn't start with 'sk-ant-' — may be invalid."
            )

        if self.sub_model_base_url and not self.sub_model_base_url.startswith(("http://", "https://")):
            errors.append(
                f"sub_model_base_url: Must start with http:// or https://, got '{self.sub_model_base_url}'."
            )

        if self.anthropic_real_base_url and not self.anthropic_real_base_url.startswith(("http://", "https://")):
            errors.append(
                f"anthropic_real_base_url: Must start with http:// or https://, got '{self.anthropic_real_base_url}'."
            )

        # --- Range checks ---
        if not (1 <= self.gateway_port <= 65535):
            errors.append(
                f"gateway_port: Must be 1-65535, got {self.gateway_port}."
            )

        if self.pack_token_budget < 100:
            errors.append(
                f"pack_token_budget: Must be >= 100, got {self.pack_token_budget}."
            )

        if self.pack_token_budget > 200000:
            warnings.append(
                f"pack_token_budget: {self.pack_token_budget} is very large — may exceed model context window."
            )

        # --- Router config ---
        r = self.router
        if r.trivial_token_threshold < 0:
            errors.append(f"router.trivial_token_threshold: Must be >= 0, got {r.trivial_token_threshold}.")
        if r.simple_token_threshold < r.trivial_token_threshold:
            errors.append(
                f"router.simple_token_threshold ({r.simple_token_threshold}) must be >= "
                f"trivial_token_threshold ({r.trivial_token_threshold})."
            )

        # --- RLM config ---
        rlm = self.rlm
        if rlm.max_depth < 0:
            errors.append(f"rlm.max_depth: Must be >= 0, got {rlm.max_depth}.")
        if rlm.max_iterations < 1:
            errors.append(f"rlm.max_iterations: Must be >= 1, got {rlm.max_iterations}.")
        if rlm.max_timeout_seconds <= 0:
            errors.append(f"rlm.max_timeout_seconds: Must be > 0, got {rlm.max_timeout_seconds}.")
        if rlm.max_consecutive_errors < 1:
            errors.append(f"rlm.max_consecutive_errors: Must be >= 1, got {rlm.max_consecutive_errors}.")

        # --- ACE config ---
        ace = self.ace
        if ace.enabled:
            if ace.playbook_token_budget < 100:
                errors.append(f"ace.playbook_token_budget: Must be >= 100, got {ace.playbook_token_budget}.")
            if ace.curator_frequency < 1:
                errors.append(f"ace.curator_frequency: Must be >= 1, got {ace.curator_frequency}.")
            if ace.max_reflection_rounds < 1:
                errors.append(f"ace.max_reflection_rounds: Must be >= 1, got {ace.max_reflection_rounds}.")
            if not (0.0 < ace.dedup_similarity_threshold <= 1.0):
                errors.append(
                    f"ace.dedup_similarity_threshold: Must be (0, 1], got {ace.dedup_similarity_threshold}."
                )

        # --- Memory config ---
        mem = self.memory
        if mem.recent_commit_count < 0:
            errors.append(f"memory.recent_commit_count: Must be >= 0, got {mem.recent_commit_count}.")
        if mem.index_max_file_size_kb < 1:
            errors.append(f"memory.index_max_file_size_kb: Must be >= 1, got {mem.index_max_file_size_kb}.")

        # Fatal errors → raise
        if errors:
            raise ConfigError(
                f"Configuration has {len(errors)} error(s):\n  - " + "\n  - ".join(errors),
                field="multiple" if len(errors) > 1 else errors[0].split(":")[0],
            )

        return warnings
