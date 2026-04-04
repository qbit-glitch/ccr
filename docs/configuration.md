# CCR Configuration

## MCP Server Config (.mcp.json)

The MCP server is configured via `.mcp.json` in the project root. Claude Code reads this to start the CCR server.

```json
{
  "mcpServers": {
    "ccr": {
      "command": ".venv/bin/python",
      "args": ["-m", "ccr.mcp_server", "--project", "."]
    }
  }
}
```

**Arguments:**

| Arg | Description |
|-----|-------------|
| `--project` | Path to the project root. Determines where `.ccr/` is created. Use `.` for the current project or an absolute path. |

For global availability across all projects, add CCR to `~/.claude/.mcp.json` with `--project .` (resolves to the current working directory at runtime).

---

## Memory Config (CCRConfig)

Defined in `ccr/core/types.py`. Controls GCC memory layer behavior.

```python
@dataclass
class CCRConfig:
    recent_commit_count: int = 3
    milestones_kept: int = 5
    log_max_lines: int = 500
    proactive_commits: bool = True
    index_max_file_size_kb: int = 500
    # Hierarchical summaries (TiMem-inspired)
    session_summary_interval: int = 5
    phase_summary_interval: int = 20
    session_summary_max_chars: int = 500
    phase_summary_max_items: int = 5
    overview_staleness_threshold: int = 5
    # Commit cross-linking (A-MEM/MAGMA-inspired)
    link_scan_window: int = 20
    link_semantic_threshold: float = 0.3
    link_entity_threshold: float = 0.0
    link_max_results: int = 10
    # Pattern buffer (CER-inspired)
    pattern_dedup_threshold: float = 0.7
    pattern_promotion_count: int = 3
    pattern_max_buffer_size: int = 200
    auto_extract_patterns: bool = False
```

### Parameter reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recent_commit_count` | `3` | Number of recent commits shown in level-2 context. |
| `milestones_kept` | `5` | Max milestones tracked in metadata. |
| `log_max_lines` | `500` | Maximum OTA log entries before truncation. |
| `proactive_commits` | `True` | Whether to auto-suggest commits at intervals. |
| `index_max_file_size_kb` | `500` | Skip files larger than this during indexing. |
| `session_summary_interval` | `5` | Commits between automatic session summaries. |
| `phase_summary_interval` | `20` | Commits on main before auto phase summary. |
| `session_summary_max_chars` | `500` | Max characters per session summary. |
| `phase_summary_max_items` | `5` | Max key accomplishments per phase summary. |
| `overview_staleness_threshold` | `5` | Phase summaries before suggesting overview regeneration. |
| `link_scan_window` | `20` | Recent commits to scan for cross-links during commit. |
| `link_semantic_threshold` | `0.3` | Minimum keyword Jaccard for semantic links. |
| `link_entity_threshold` | `0.0` | Minimum file Jaccard for entity links (0.0 = any shared file). |
| `link_max_results` | `10` | Max results from BFS link traversal. |
| `pattern_dedup_threshold` | `0.7` | Word Jaccard threshold for pattern deduplication. |
| `pattern_promotion_count` | `3` | Commits before suggesting pattern promotion to ACE playbook. |
| `pattern_max_buffer_size` | `200` | Max patterns in the CER buffer (evicts lowest occurrence). |
| `auto_extract_patterns` | `False` | Auto-extract patterns via sub-model on each commit (adds latency). |

---

## Playbook Schema (PlaybookSchema)

Defined in `ccr/core/types.py`. Versioned schema controlling playbook structure and evolution. Stored in `.ccr/playbook_schema.json`. Evolved via `ace_evolve_schema`.

```python
@dataclass
class PlaybookSchema:
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
    # Memory retrieval parameters (ALMA-inspired)
    link_scan_window: int = 20
    link_semantic_threshold: float = 0.3
    context_level_default: int = 2
    search_result_limit: int = 5
```

### Parameter reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `version` | `1` | Schema version number (increments on each change). |
| `sections` | (from Playbook) | List of playbook section names. Default: STRATEGIES & INSIGHTS, CODE SNIPPETS & TEMPLATES, COMMON MISTAKES TO AVOID, PROBLEM-SOLVING HEURISTICS, CONTEXT CLUES & INDICATORS, OTHERS. |
| `slug_map` | (from Playbook) | Maps normalized section names to slug prefixes. |
| `decay_rate` | `0.95` | Temporal decay factor. `effective_score = raw * decay_rate^days`. At 0.95: unused 30d = 21%, unused 90d = 1%. |
| `prune_min_harmful` | `3` | Minimum harmful count before a bullet is eligible for pruning. |
| `evolution_threshold` | `3` | Minimum harmful-with-lessons count to trigger failure evolution. |
| `token_budget` | `80000` | Character budget for the playbook. Budget pruning trims lowest-scoring bullets. |
| `parent_version` | `None` | Previous schema version (for rollback). |
| `link_scan_window` | `20` | How many recent commits to scan for cross-links. Synced to memory manager on schema change. |
| `link_semantic_threshold` | `0.3` | Minimum similarity for semantic links. Lowered by `ace_evolve_schema` when search zero-rate is high. |
| `context_level_default` | `2` | Default context level for `gcc_context`. |
| `search_result_limit` | `5` | Default max results for commit search. |

### Schema health metrics (SchemaMetrics)

Computed by `ace_evolve_schema` evaluation mode:

| Metric | Description |
|--------|-------------|
| `section_balance` | Normalized Shannon entropy of bullet distribution across sections [0,1]. |
| `utilization_rate` | Fraction of bullets with any helpful+harmful counter > 0. |
| `harmful_ratio` | Of utilized bullets, fraction that are net-harmful. |
| `unused_ratio` | Fraction with helpful+harmful == 0. |
| `decay_impact` | Fraction with effective_score < 50% of raw score. |
| `overall_health` | Weighted composite [0,1]. When >= 0.8, no proposals are generated. |
| `search_zero_rate` | Fraction of searches returning zero results (ALMA-inspired). |
| `link_density` | Average cross-links per commit. |
| `embedding_coverage` | Fraction of commits with cached embeddings. |

---

## ACE Config (ACEConfig)

Defined in `ccr/core/types.py`. Controls ACE playbook evolution behavior. Used by the legacy engine; MCP mode uses `PlaybookSchema` for the same parameters.

```python
@dataclass
class ACEConfig:
    enabled: bool = True
    playbook_token_budget: int = 80000
    curator_frequency: int = 1
    max_reflection_rounds: int = 3
    prune_min_harmful: int = 3
    max_tokens: int = 4096
    playbook_path: str = ""
    refinement_frequency: int = 10
    dedup_similarity_threshold: float = 0.6
    schema_evolution_enabled: bool = True
    overflow_threshold: float = 0.5
    min_cluster_size: int = 3
    stop_health_threshold: float = 0.8
    rollback_health_delta: float = -0.05
    schema_max_history: int = 20
```

---

## YAML Configuration (config/default.yaml)

Legacy configuration file for the proxy architecture (not used in MCP mode). Preserved for backward compatibility with `ccr start` CLI.

Key sections: `gateway`, `models`, `context`, `memory`, `router`, `ace`.

---

## Optional Dependencies

CCR has three optional dependency groups:

| Group | Install | Packages | Purpose |
|-------|---------|----------|---------|
| `semantic` | `pip install ccr-memory[semantic]` | `onnxruntime`, `tokenizers`, `numpy` | Dense embeddings for semantic search (all-MiniLM-L6-v2 ONNX). |
| `vector` | `pip install ccr-memory[vector]` | `sqlite-vec` | Persistent vector store for embeddings (sqlite-vec). |
| `watch` | `pip install ccr-memory[watch]` | `watchdog` | File watching (unused in MCP mode). |
| `full` | `pip install ccr-memory[full]` | All of `semantic` + `vector` | Everything. |

Without optional dependencies:
- Semantic search falls back to BM25 (zero-dep, Okapi BM25, k1=1.5, b=0.75).
- Embeddings fall back to gzip JSON storage instead of sqlite-vec.
- All core functionality (memory, playbook, REPL, hooks) works without any optional deps.

---

## Environment Variables

| Variable | Description | Used by |
|----------|-------------|---------|
| `CCR_PROJECT_ROOT` | Override project root directory. | All hooks, CLI. |
| `CCR_OLLAMA_MODEL` | Ollama model name for sub-model (e.g., `"qwen2.5:7b"`). Free, local. | `_get_sub_client()` in MCP server. |
| `CCR_OLLAMA_BASE_URL` | Ollama API base URL. Default: `http://localhost:11434/v1`. | MCP server. |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude sub-model. | MCP server (fallback if no Ollama), legacy CLI. |
| `ANTHROPIC_API_KEY_SUB` | Separate API key for sub-model only (takes priority over `ANTHROPIC_API_KEY`). | MCP server. |
| `CCR_SUB_MODEL_API_KEY` | Legacy alias for sub-model API key. | Legacy CLI. |
| `CLAUDE_TOOL_NAME` | Tool name passed by Claude Code to hooks. | `on_tool_use.py`. |
| `CLAUDE_TOOL_INPUT` | JSON tool input passed by Claude Code to hooks. | `on_tool_use.py`. |

### Sub-model priority

The MCP server checks for a sub-model in this order:
1. **Ollama** (`CCR_OLLAMA_MODEL` set) -- free, local, recommended.
2. **Anthropic Haiku** (`ANTHROPIC_API_KEY_SUB` or `ANTHROPIC_API_KEY` set).
3. **None** -- all LLM-dependent features silently degrade (no ACE pipeline, no A-MEM evolution, no auto-pattern extraction).

Most CCR functionality (all 30 tools) works without any sub-model. The sub-model is only needed for:
- `ace_generate_bullets` (3-agent pipeline)
- `gcc_evolve_memory` (A-MEM rewriting)
- Auto-pattern extraction in `gcc_commit` (when `auto_extract_patterns=True`)
- Cross-lesson synthesis in `ace_evolve_from_failures`

---

## Storage Layout (.ccr/)

All CCR state lives in the `.ccr/` directory at the project root:

```
.ccr/
  metadata.yaml              # Project metadata (name, description, milestones)
  main/                      # Main branch
    commits/                 # Commit files (C001.md, C002.md, ...)
    rolling_summary.md       # Rolling summary of all commits
    ota_log.md               # Observation-Thought-Action log
  playbook.txt               # Project ACE playbook
  playbook_schema.json       # Versioned playbook schema + history
  playbook_companion.json    # Per-bullet metadata (counters, timestamps, triggers)
  patterns.json              # CER pattern buffer
  commit_links.json          # Cross-link graph
  commit_embeddings.json.gz  # Commit embedding cache (gzip JSON)
  embeddings.db              # sqlite-vec persistent vector store (optional)
  index.json                 # Repo index cache
  summaries.json             # Hierarchical summaries
  failure_lessons.json       # Extended bullet failure lesson data
  scratchpad.json            # Working memory (ephemeral)
  triples.json               # Semantic triples
  clusters.json              # Thematic clusters
  evolved_summaries.json     # A-MEM evolved commit summaries
  memory_metrics.json        # Memory retrieval usage metrics (ALMA)
  .session_state.json        # Accumulated hook session state (auto-deleted)
  .session_active            # Session marker file (auto-deleted)
```

Global state lives in `~/.ccr/`:

```
~/.ccr/
  global_playbook.txt        # Global ACE playbook
  playbook_companion.json    # Global per-bullet metadata
  playbook_schema.json       # Global schema
  failure_lessons.json       # Global failure lessons
```
