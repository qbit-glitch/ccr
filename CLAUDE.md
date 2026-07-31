# CCR — Claude Context Reducer

MCP server giving Claude Code persistent memory (GCC), self-evolving playbooks (ACE), and sandboxed REPL (RLM). No API keys needed — works with Claude Max.

## How To Use CCR (MCP Tools)

### Memory (GCC)
- **Session start**: `gcc_context(level=2)` to load history
- **After progress**: `gcc_commit` with what/why/files/next
- **Before compaction**: Commit to preserve reasoning state
- **Alternatives**: `gcc_branch` to isolate, `gcc_merge` when decided
- **Search**: `gcc_context(level=5, search_term="...")`
- **Patterns**: Include `patterns_learned` in commits; query with `gcc_patterns`

### Playbook (ACE)
- Review: `ace_get_playbook` then `ace_update_counters` (helpful/harmful tags)
- Add insights: `ace_apply_delta` ADD
- Maintain: `ace_find_similar` + MERGE duplicates, `ace_prune` harmful
- Optional `weight` (0.0-1.0) for proportional credit on counters
- Failure lessons: include `failure_lesson` dict when tagging harmful

### REPL (RLM)
- `rlm_init` → `rlm_execute` (search_repo, get_file) → `rlm_finalize`

### Index
- `index_search(query)` — keyword/semantic/hybrid search
- `index_build` — rebuild after code changes

### Session Logger (SL)
- **After each response**: `session_log_turn(assistant_message="<your full response>")` — logs this Q&A turn to `.ccr/sessions.db`
- **Review session**: `session_get_history()` — last 20 turns of current session
- **Search past sessions**: `session_search(query="...")` — full-text search all Q&A logs
- **Export for training**: `session_export(format="jsonl")` — OpenAI fine-tuning format

### Research & Experiment Tools (v6)
- **Log experiment**: `gcc_experiments(experiment_id=..., hypothesis_contains=..., metric_filter={...})` — query/filter experiment records stored in commits
- **Log decision**: `gcc_discuss(topic=..., hypothesis=..., decision=..., rationale=...)` — persist reasoning behind design choices across sessions
- **List discussions**: `gcc_discussions(limit=20)` — retrieve stored decision log
- **Semantic search**: `gcc_search(query=..., mode="hybrid")` — keyword/semantic/hybrid search across all memory

## Project Structure

```
ccr/
  mcp_server.py     # Backward-compat shim → ccr/mcp/
  mcp/              # FastMCP tools: server.py, gcc_tools.py, ace_tools.py, rlm_tools.py, index_tools.py, session_tools.py
  hooks/            # on_session_start.py, on_stop.py, on_compact.py, on_tool_use.py
  core/
    memory.py       # GCC .ccr/ directory: commit/branch/merge/context
    scratchpad.py   # Ephemeral KV working memory
    triples.py      # Regex triple extraction from commits
    types.py        # Shared dataclasses, enums, configs
  context/
    indexer.py      # Repo indexing + keyword/BM25/ONNX search
    embeddings.py   # ONNX all-MiniLM-L6-v2 (optional)
    vec_store.py    # sqlite-vec vector store (optional)
  rlm/repl.py       # Sandboxed Python REPL
  ace/playbook.py   # Bullet library: sections, delta ops, decay, GRPO
  models/           # Legacy (not used in MCP mode)
  utils/            # tokens.py, parsing.py
tests/unit/         # ~1993 tests
```

## Key Patterns

- **44 MCP tools** always active (gcc×26, ace×8, rlm×3, index×3, session×4)
  (Note: broad `@mcp.tool` text searches also match comment/docstring mentions.
  Actual registered tool-function count = 44.)
- **A-MAC admission**: S(m) = 0.50*TypePrior + 0.35*Novelty + 0.15*Recency; three-way admit/merge/reject
- **Cross-linking**: 4 link types (entity, causal, supersession, semantic); BFS traversal via `gcc_links`
- **Two-tier playbook**: Global (~/.ccr/global.db) + project (.ccr/memory.db) via SQLite; `scope="global"|"project"` on ACE tools. WAL locking makes concurrent Claude Code sessions on different projects safe for global-scope writes.
- **Temporal decay**: effective_score = raw * 0.95^days
- **Pattern buffer**: Dedup at 0.7 Jaccard; 200-cap; 3+ occurrences suggests ACE promotion
- **Schema evolution**: Rule-based proposals; version history + rollback
- **Semantic search**: ONNX → BM25 → keyword fallback chain
- **Failure lessons**: prevention_principle → new bullets at harmful >= 3

## Development

```bash
pytest tests/unit/ tests/integration/ -x -q  # ~1993 tests
python -m ccr.mcp_server                     # MCP server (stdio)
```

## MCP Config

`.mcp.json`:
```json
{"mcpServers": {"ccr": {"command": ".venv/bin/python", "args": ["-m", "ccr.mcp_server", "--project", "."]}}}
```

## Research Papers (16)

1. **GCC** (2508.00031) — version-controlled memory + A-MAC admission
2. **RLM** (2512.24601) — REPL execution, FINAL_VAR termination
3. **ACE** (2510.04618) — evolving playbooks, delta operations
4. **SkillRL** (2602.08234) — failure-side skill distillation
5. **CER** (2506.06698) — transferable pattern buffer
6. **A-MEM/MAGMA** (2502.12110, 2601.03236) — cross-linking + memory evolution
7. **MCE** (2601.21557) — meta-level schema evolution
8. **A-RAG** (2602.03442) — hierarchical retrieval (keyword/semantic/hybrid)
9. **ExpRAG** (2603.18272) — embedding-based commit search
10. **ERL** (2603.24639) — trigger/action bullets
11. **EvolveR** (2510.16079) — quality-scored patterns (Bayesian)
12. **Memori** (2603.19935) — semantic triples for token-efficient memory
13. **EverMemOS** (2601.02163) — thematic clustering via connected components
14. **AgeMem** (2601.01885) — working memory scratchpad (ephemeral KV)
15. **AgentEvolver** (2511.10395) — contribution-weighted proportional credit
16. **ALMA** (2602.07755) — meta-learned memory retrieval parameter evolution

Full details and limitation tables: `docs/papers.md`

## ACE Playbook Format

```
## SECTION_NAME
[slug-00001] helpful=5 harmful=0 :: Strategy content
```

Sections: STRATEGIES & INSIGHTS, CODE SNIPPETS & TEMPLATES, COMMON MISTAKES TO AVOID, PROBLEM-SOLVING HEURISTICS, CONTEXT CLUES & INDICATORS, OTHERS.
