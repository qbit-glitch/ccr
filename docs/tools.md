# CCR Tool Reference

30 MCP tools exposed via stdio transport. All tools are pure logic (zero LLM calls unless a sub-model is explicitly configured). Claude Code drives all reasoning.

Tools are organized into four groups: GCC (memory), ACE (playbook), RLM (sandbox), and Index (search).

---

## GCC -- Memory Tools

### gcc_commit

Commit progress to project memory.

Creates a structured commit record tracking what was done, why, which files changed, and what to do next. Similar commits are auto-deduplicated via A-MAC admission control.

Use after meaningful progress -- completing a feature, fixing a bug, reaching a milestone, or before context gets large.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `title` | `str` | Yes | -- | Short title for the commit. |
| `what` | `str` | Yes | -- | What was done. |
| `why` | `str` | Yes | -- | Why it was done. |
| `files_changed` | `list[str]` | Yes | -- | Files touched in this work. |
| `next_step` | `str` | Yes | -- | What to do next. |
| `patterns_learned` | `list[str]` | No | `None` | Transferable lessons, e.g., `"When adding {feature_type}, update tests + docs together"`. Deduped automatically. Recurring patterns (3+ commits) are suggested for ACE playbook promotion. |
| `admission_threshold` | `float` | No | `0.85` | Similarity threshold for dedup (0-1). Set to 1.0 to disable dedup. |
| `rejection_threshold` | `float` | No | `0.0` | Score below which commits are rejected (0.0 = disabled). |
| `compressed_summary` | `str` | No | `None` | Compressed rolling summary. Provide when prompted -- compress into a concise paragraph of project context, milestones, and direction. Max 1500 chars. |

**Returns:** `GccCommitResult` with `commit_id`, `branch`, `title`, `admission_decision` ("created", "merged", or "rejected"), and `message`.

**Annotations:** readOnly=False, destructive=False, idempotent=False

---

### gcc_context

Retrieve project memory at the specified depth.

**Levels:**

| Level | Content |
|-------|---------|
| 1 | Project overview only (~200 tokens). |
| 2 | + rolling summary + last 3 commits. Use at session start. |
| 3 | + branch summary + thematic clusters + triples. |
| 4 | + last 10 commits. For finding related work. |
| 5 | + specific commit search + cross-links. For tracing history. |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `level` | `int` | No | `2` | Depth of context retrieval (1-5). |
| `search_term` | `str` | No | `None` | Keyword to search commits (level 5 only). |
| `commit_id` | `str` | No | `None` | Specific commit ID to retrieve (level 5 only). |
| `log_window` | `int` | No | `0` | Number of recent OTA log entries to include. |
| `follow_links` | `bool` | No | `False` | If True and level >= 5, include linked commit summaries (1-hop BFS). |
| `include_summaries` | `bool` | No | `False` | If True, append hierarchical summaries (replaces standalone `gcc_summaries`). |
| `summaries_tier` | `str` | No | `"all"` | Which summary tier: "session", "phase", "project", or "all". |
| `summaries_count` | `int` | No | `5` | Max summaries per tier. |

**Returns:** `GccContextResult` with `level`, `branch`, and `message`.

**Annotations:** readOnly=True, destructive=False, idempotent=True

---

### gcc_branch

Create an exploration branch for experimental work.

Branches isolate experimental changes from the main line. Must be on main to create a branch.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `name` | `str` | Yes | -- | Branch name in kebab-case (e.g., `try-new-parser`). |
| `purpose` | `str` | Yes | -- | What this branch explores. |
| `hypothesis` | `str` | Yes | -- | What you expect to learn or achieve. |

**Returns:** `GccBranchResult` with `branch` and `message`.

**Annotations:** readOnly=False, destructive=False, idempotent=False

---

### gcc_merge

Merge an exploration branch back into main.

Must be on the branch being merged. Integrates the branch's rolling summary and OTA log into main.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `branch` | `str` | Yes | -- | Branch name to merge. |
| `outcome` | `str` | Yes | -- | One of `"success"`, `"failure"`, or `"partial"`. |
| `conclusion` | `str` | Yes | -- | Summary of what was learned. |

**Returns:** `GccMergeResult` with `source`, `target`, and `message`.

**Annotations:** readOnly=False, destructive=True, idempotent=False

---

### gcc_links

Retrieve cross-links for a commit.

Shows which other commits are related via shared files (entity), explicit C### references (causal), replacement language (supersession), or keyword overlap (semantic). Uses mechanical heuristics (A-MEM/MAGMA inspired taxonomy).

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `commit_id` | `str` | Yes | -- | The commit to look up (e.g., `"C012"`). |
| `link_types` | `str` | No | `None` | Comma-separated link types to filter. Options: `entity`, `causal`, `supersession`, `semantic`. |
| `max_hops` | `int` | No | `1` | How many link hops to traverse (1-5). |
| `query` | `str` | No | `None` | Natural language search intent. When provided, BFS traversal is weighted by query relevance. |

**Returns:** `GccLinksResult` with `commit_id`, `links_found`, and `message`.

**Annotations:** readOnly=True, destructive=False, idempotent=True

---

### gcc_clusters

Compute or retrieve thematic commit clusters.

Clusters related commits using connected components over the cross-link graph (entity + semantic links). Inspired by EverMemOS.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `min_size` | `int` | No | `2` | Minimum commits per cluster. |
| `recompute` | `bool` | No | `True` | If True, recompute clusters from current links. |

**Returns:** `GccClustersResult` with `cluster_count` and `message`.

**Annotations:** readOnly=False, destructive=False, idempotent=False

---

### gcc_triples

Search the semantic knowledge graph for entity relationships.

Triples are automatically extracted from commits using 13 regex patterns (Memori-inspired). Zero LLM calls. Format: `subject --predicate--> object (source_commit)`

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | `str` | No | `""` | Free-text search (word Jaccard similarity). |
| `commit_id` | `str` | No | `""` | Get all triples from a specific commit. |
| `entity` | `str` | No | `""` | Get all triples involving an entity (subject or object). |
| `top_k` | `int` | No | `10` | Maximum results to return (1-200). |

**Returns:** `GccTriplesResult` with `count` and `message`.

**Annotations:** readOnly=True, destructive=False, idempotent=True

---

### gcc_evolve_memory

Manually trigger A-MEM evolution for a commit or all recent commits.

When a sub-model is available, rewrites commit summaries to incorporate context from related commits. Requires `CCR_OLLAMA_MODEL` or `ANTHROPIC_API_KEY` environment variable.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `commit_id` | `str` | No | `None` | Specific commit to evolve (e.g., `"C001"`). If None, evolves all recent commits with eligible links. |

**Returns:** `GccEvolveMemoryResult` with `evolutions` count and `message`.

**Annotations:** readOnly=False, destructive=False, idempotent=False

---

### gcc_log_ota

Log an Observation-Thought-Action triple to the project log.

OTA triples track reasoning process. They are attached to commits and preserved across sessions.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `observation` | `str` | Yes | -- | What you observed (e.g., `"Test failure in router.py"`). |
| `thought` | `str` | Yes | -- | Your reasoning (e.g., `"The regex pattern is too greedy"`). |
| `action` | `str` | Yes | -- | What you did (e.g., `"Fixed pattern to use non-greedy match"`). |

**Returns:** `GccLogOtaResult` with `message`.

**Annotations:** readOnly=False, destructive=False, idempotent=False

---

### gcc_status

Show current project memory status.

Returns the active branch, recent milestones, open branches, and metadata summary. No parameters.

**Returns:** `GccStatusResult` with `branch`, `total_commits`, and `message`.

**Annotations:** readOnly=True, destructive=False, idempotent=True

---

### gcc_consolidate

Generate or save a hierarchical memory summary (TiMem-inspired).

Three tiers of consolidation:

| Tier | Behavior |
|------|----------|
| `"session"` | Mechanical. Consolidates last N commits into a structured paragraph. |
| `"phase"` | Mechanical. Aggregates session summaries + branch metadata into a strategic view. |
| `"project"` | Two-step. First call returns a prompt; second call with `content=` saves the overview. |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `tier` | `str` | No | `"session"` | `"session"`, `"phase"`, or `"project"`. |
| `content` | `str` | No | `None` | For tier="project" only -- the generated overview text to save. |

**Returns:** `GccConsolidateResult` with `tier` and `message`.

**Annotations:** readOnly=False, destructive=False, idempotent=False

---

### gcc_patterns

Query the CER-inspired pattern buffer.

Patterns are transferable decision-making skills observed across commits. Deduped by word similarity (Jaccard 0.7), tracked by occurrence count. Patterns in 3+ commits are suggested for ACE playbook promotion.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `min_occurrences` | `int` | No | `1` | Minimum occurrence count to include. |
| `include_promoted` | `bool` | No | `True` | Whether to include already-promoted patterns. |
| `search_term` | `str` | No | `None` | Optional keyword filter on pattern text. |

**Returns:** `GccPatternsResult` with `total`, `matching`, and `message`.

**Annotations:** readOnly=True, destructive=False, idempotent=True

---

### gcc_scratchpad

Working memory for temporary reasoning state (ephemeral, within session).

Inspired by AgeMem. Entries appear in `gcc_context` at level 2+. Use for hypotheses, debug focus, or intermediate results that do not warrant a permanent `gcc_commit`.

**Modes:**

| Mode | Description |
|------|-------------|
| `"get"` | Retrieve a key, or list all if key is None. |
| `"set"` | Store a key-value pair. Requires both key and value. |
| `"clear"` | Delete a key, or clear all if key is None. |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `mode` | `str` | No | `"get"` | Operation: `"get"`, `"set"`, or `"clear"`. |
| `key` | `str` | No | `None` | The key to operate on. None means all entries (for get/clear). |
| `value` | `str` | No | `None` | The value to store (required for `"set"` mode). |

**Returns:** `GccScratchpadResult` with `mode`, optional `key`, optional `cleared` count, and `message`.

**Annotations:** readOnly=False, destructive=False, idempotent=False

---

## ACE -- Playbook Tools

### ace_get_playbook

Get the current ACE playbook -- both global and project-specific strategies.

Returns two tiers:
- **GLOBAL**: Universal heuristics that transfer across all projects (`~/.ccr/`)
- **PROJECT**: Project-specific strategies (`.ccr/`)

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `task_context` | `str` | No | `""` | Optional task description. Prepends a policy-ranked top-5 section weighted by relevance. |
| `include_stats` | `bool` | No | `False` | If True, append per-bullet stats (counts, decay, section breakdown). Replaces standalone `ace_get_stats`. |

**Returns:** `AcePlaybookResult` with `global_bullet_count`, `project_bullet_count`, and `message`.

**Annotations:** readOnly=True, destructive=False, idempotent=True

---

### ace_apply_delta

Apply delta operations to the playbook.

**Supported operations:**

| Type | Required fields | Description |
|------|----------------|-------------|
| `ADD` | `section`, `content` | Add a new bullet. Optional `trigger`/`action` fields for ERL-inspired structured rules. |
| `UPDATE` | `bullet_id`, `content` | Update an existing bullet's content. |
| `MERGE` | `bullet_id`, `merge_target`, `content` | Merge two bullets into one. |
| `REMOVE` | `bullet_id` | Delete a bullet. |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `operations` | `list[dict]` | Yes | -- | List of delta operation dicts. |
| `scope` | `str` | No | `"project"` | `"project"` or `"global"`. |

**Returns:** `AceApplyDeltaResult` with `applied` count, `scope`, and `message`.

**Annotations:** readOnly=False, destructive=False, idempotent=False

---

### ace_update_counters

Update helpful/harmful counters for playbook bullets.

After completing a task, reflect on which strategies helped or hurt, then update their counters. High-scoring bullets persist; low-scoring ones get pruned.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `bullet_tags` | `list[dict]` | Yes | -- | List of dicts, each with `"id"` (bullet ID), `"tag"` (`"helpful"` or `"harmful"`), optional `"weight"` (0.0-1.0 for proportional credit), and optional `"failure_lesson"` dict. |
| `scope` | `str` | No | `"project"` | `"project"` or `"global"`. |

**failure_lesson dict structure** (optional, for harmful tags):

```json
{
    "failure_point": "Where the strategy broke down",
    "flawed_reasoning": "What incorrect assumption was made",
    "counterfactual": "What should have been done instead",
    "prevention_principle": "General rule to avoid this failure"
}
```

**Returns:** `AceUpdateCountersResult` with `updated` count, `scope`, and `message`.

**Annotations:** readOnly=False, destructive=False, idempotent=False

---

### ace_find_similar

Find similar bullet pairs that may be candidates for merging.

Uses Jaccard + trigram similarity (or ONNX cosine when available).

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `threshold` | `float` | No | `0.6` | Similarity threshold (0.0-1.0). |
| `scope` | `str` | No | `"project"` | `"project"`, `"global"`, or `"cross"` (find duplicates between tiers). |

**Returns:** `AceFindSimilarResult` with `pairs_found`, `scope`, and `message`.

**Annotations:** readOnly=True, destructive=False, idempotent=True

---

### ace_prune

Prune problematic bullets and enforce token budget.

Execution order:
1. Evolves failure lessons into new skills (aggressive threshold).
2. Removes bullets where `harmful >= helpful` and `harmful >= 3`.
3. Trims lowest-scoring bullets if playbook exceeds token budget.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `scope` | `str` | No | `"project"` | `"project"` or `"global"`. |

**Returns:** `AcePruneResult` with `removed` count, `evolved` count, `scope`, and `message`.

**Annotations:** readOnly=False, destructive=True, idempotent=False

---

### ace_evolve_from_failures

Evolve new skills from accumulated failure lessons (SkillRL-inspired).

**Two-step workflow:**
1. Call with no `synthesized_skills` -- returns failure lessons and a synthesis prompt.
2. Call again with your synthesized skills -- saves them as new playbook bullets.

Falls back to mechanical extraction (copies `prevention_principle` verbatim) if `synthesized_skills` is not provided.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `threshold` | `int` | No | `3` | Minimum harmful-with-lessons bullets to trigger evolution. |
| `scope` | `str` | No | `"project"` | `"project"` or `"global"`. |
| `synthesized_skills` | `list[dict]` | No | `None` | List of dicts with `"content"` and `"when_to_apply"` keys. |

**Returns:** `AceEvolveFromFailuresResult` with `evolved` count, `synthesized` count, and `message`.

**Annotations:** readOnly=False, destructive=False, idempotent=False

---

### ace_evolve_schema

Evolve playbook structure -- sections, thresholds, decay parameters (MCE-inspired).

**Three modes:**

| Mode | Call | Description |
|------|------|-------------|
| Evaluate | `ace_evolve_schema()` | Show health metrics and improvement proposals. |
| Apply | `ace_evolve_schema(apply_proposal=N)` | Apply the Nth proposed change. |
| Rollback | `ace_evolve_schema(rollback=True)` | Revert to previous schema version. |

Stops proposing changes when overall health >= 0.8.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `scope` | `str` | No | `"project"` | `"project"` or `"global"`. |
| `apply_proposal` | `int` | No | `None` | 1-indexed proposal to apply (None = evaluate only). |
| `rollback` | `bool` | No | `False` | Revert to parent schema version. |

**Schema change types:** `ADD_SECTION`, `REMOVE_SECTION`, `ADJUST_DECAY`, `ADJUST_PRUNING`, `ADJUST_EVOLUTION`, `ADJUST_BUDGET`, `REBALANCE`, `ADJUST_SEARCH_THRESHOLD`, `ADJUST_SCAN_WINDOW`, `ROLLBACK`.

**Returns:** `AceEvolveSchemaResult` with `version`, optional `health`, and `message`.

**Annotations:** readOnly=False, destructive=False, idempotent=False

---

### ace_generate_bullets

Generate and optionally apply strategy bullets via the ACE 3-agent pipeline.

Runs Generator -> Reflector -> Curator to produce candidate playbook bullets from a task context. **Requires a sub-model** (`CCR_OLLAMA_MODEL` or `ANTHROPIC_API_KEY`).

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `context` | `str` | Yes | -- | Task context or trajectory to generate bullets from. |
| `auto_apply` | `bool` | No | `False` | If True, automatically apply ADD decisions. Default is preview only. |

**Returns:** `AceGenerateBulletsResult` with `decisions` count, `applied` count, and `message`.

**Annotations:** readOnly=False, destructive=False, idempotent=False

---

## RLM -- Sandbox Tools

### rlm_init

Initialize a sandboxed Python REPL for structured problem-solving.

Provides the REPL execution component from the RLM paper. Claude Code drives the iteration loop manually via `rlm_execute` calls.

**Pre-loaded variables and tools in the REPL:**

| Name | Type | Description |
|------|------|-------------|
| `task_prompt` | `str` | Your problem statement. |
| `context` | `dict` | Repo index metadata (file paths, symbols, imports). |
| `get_file(path)` | function | Fetch full content of any indexed file. |
| `search_repo(query)` | function | Search files by content/symbol/path. |
| `estimate_tokens(text)` | function | Estimate token count. |
| `FINAL_VAR(name)` | function | Signal completion and return a variable's value. |
| `SHOW_VARS()` | function | List all user-created variables. |
| `playbook` | `str` | ACE playbook text (if available). |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `task_prompt` | `str` | Yes | -- | The problem or question to solve. |

**Returns:** `RlmInitResult` with `session_id`, `file_count`, and `message`.

**Annotations:** readOnly=False, destructive=False, idempotent=False

---

### rlm_execute

Execute Python code in the sandboxed REPL.

The REPL persists variables across calls. By default, long stdout is summarized to metadata per the RLM paper ("Metadata-only stdout") to save tokens.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `code` | `str` | Yes | -- | Python code to execute. |
| `metadata_only` | `bool` | No | `True` | If True, stdout exceeding 1000 chars is replaced with a metadata summary. Set to False for full stdout. |

**Returns:** `RlmExecuteResult` with `has_error`, `has_final_answer`, and `message`.

**Annotations:** readOnly=False, destructive=False, idempotent=False

---

### rlm_finalize

Finalize the REPL session and return a variable's value as the result.

Extracts and serializes the named variable, then cleans up the REPL.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `variable_name` | `str` | Yes | -- | Name of the variable to return. Must be a valid Python identifier. |

**Returns:** `RlmFinalizeResult` with `variable_name` and `message`.

**Annotations:** readOnly=False, destructive=True, idempotent=False

---

## Index -- Search Tools

### index_build

Build or rebuild the repo index.

Scans the project directory for source files, extracts symbols (classes, functions) and imports per file. If `onnxruntime` + `tokenizers` are installed, also computes dense embeddings for semantic search and chunk-level embeddings for snippet extraction.

**Parameters:** None.

**Returns:** `IndexBuildResult` with `files_indexed` and `message`.

**Annotations:** readOnly=False, destructive=False, idempotent=False

---

### index_search

Search the repo index for files matching a query.

**Three search modes:**

| Mode | Best for | Mechanism |
|------|----------|-----------|
| `"keyword"` | Specific file/symbol names | Fast substring matching on paths, symbols, content. |
| `"semantic"` | Conceptual queries | Meaning-based search using embeddings (or BM25 fallback). |
| `"hybrid"` (default) | General-purpose | Combines keyword + semantic scores. |

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | `str` | Yes | -- | Search term or natural language description. |
| `mode` | `str` | No | `"hybrid"` | `"keyword"`, `"semantic"`, or `"hybrid"`. |
| `top_k` | `int` | No | `10` | Maximum results to return (1-100). |

**Returns:** `IndexSearchResult` with `result_count`, `mode`, and `message`.

**Annotations:** readOnly=True, destructive=False, idempotent=True

---

## Internal Helpers

These functions are not exposed as MCP tools but provide important functionality used by the tool layer.

### _summarize_stdout (rlm_tools.py)

Summarize long stdout per the RLM paper Section 3 ("Metadata-only stdout"). If stdout exceeds 1000 chars, replaces with a metadata summary showing line/char counts plus the first and last 5 lines.

### _ace_generator / _ace_reflector / _ace_curator (ace_tools.py)

The ACE 3-agent pipeline. Generator produces candidate bullets from a task trajectory, Reflector filters by quality score (>= 3 out of 5), and Curator decides ADD/MERGE/SKIP against existing playbook. All three require a sub-model.

### _run_ace_pipeline (ace_tools.py)

Fire-and-forget execution of the 3-agent pipeline after each `gcc_commit`. Failures are silently swallowed to never affect the commit result.

### _auto_synthesize_skills (ace_tools.py)

Cross-synthesize generalized skills from related failure lessons (SkillRL). Groups candidates by task_context similarity, then calls the sub-model per group to generate "When X, do Y" skills.

### _word_jaccard / _semantic_or_jaccard (ace_tools.py)

Similarity utilities. ONNX cosine similarity with word Jaccard fallback for bullet comparison.
