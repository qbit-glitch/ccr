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

## Project Structure

```
ccr/
  mcp_server.py     # Backward-compat shim → ccr/mcp/
  mcp/              # FastMCP tools: server.py, gcc_tools.py, ace_tools.py, rlm_tools.py, index_tools.py
  hooks/            # on_session_start.py, on_stop.py, on_compact.py
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
tests/unit/         # ~1566 tests
```

## Key Patterns

- **12 core MCP tools** via stdio; 10 more with `CCR_EXTENDED=1`
- **A-MAC admission**: S(m) = 0.50*TypePrior + 0.35*Novelty + 0.15*Recency; three-way admit/merge/reject
- **Cross-linking**: 4 link types (entity, causal, supersession, semantic); BFS traversal via `gcc_links`
- **Two-tier playbook**: Global (~/.ccr/) + project (.ccr/); `scope="global"|"project"` on ACE tools
- **Temporal decay**: effective_score = raw * 0.95^days
- **Pattern buffer**: Dedup at 0.7 Jaccard; 200-cap; 3+ occurrences suggests ACE promotion
- **Schema evolution**: Rule-based proposals; version history + rollback
- **Semantic search**: ONNX → BM25 → keyword fallback chain
- **Failure lessons**: prevention_principle → new bullets at harmful >= 3

## Development

```bash
pytest tests/unit/ tests/integration/ -x -q  # ~1566 tests
python -m ccr.mcp_server                     # MCP server (stdio)
```

## MCP Config

`.mcp.json`:
```json
{"mcpServers": {"ccr": {"command": ".venv/bin/python", "args": ["-m", "ccr.mcp_server", "--project", "."]}}}
```

## Research Papers (11)

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

Full details and limitation tables: `docs/papers.md`

## ACE Playbook Format

```
## SECTION_NAME
[slug-00001] helpful=5 harmful=0 :: Strategy content
```

Sections: STRATEGIES & INSIGHTS, CODE SNIPPETS & TEMPLATES, COMMON MISTAKES TO AVOID, PROBLEM-SOLVING HEURISTICS, CONTEXT CLUES & INDICATORS, OTHERS.
