# CCR — Claude Context Reducer

MCP server that gives Claude Code persistent memory (GCC), self-evolving strategy playbooks (ACE), and a sandboxed Python REPL (RLM). No API keys or sub-models needed — works with Claude Max subscription.

## How To Use CCR (MCP Tools)

### Memory (GCC) — Use these to persist knowledge across sessions

- **At session start**: Call `gcc_context(level=2)` to ground yourself in project history
- **After meaningful progress**: Call `gcc_commit` with what/why/files/next to save state (auto-merges if too similar to last commit)
- **Before context gets large**: Commit to avoid losing reasoning state on compaction
- **When exploring alternatives**: Use `gcc_branch` to isolate experiments, `gcc_merge` when decided
- **To search past work**: Use `gcc_context(level=5, search_term="...")` to find specific commits
- **When committing patterns**: Include `patterns_learned` with abstract, reusable patterns you noticed
  - Good: `"When adding {tool_type}, update annotations + tests + docs together"`
  - Bad: `"Updated test_mcp_server.py"` (too specific, not transferable)
- **To query patterns**: Use `gcc_patterns` to see the pattern buffer, filter by occurrences or search
- **Pattern promotion**: When a pattern appears in 3+ commits, `gcc_commit` suggests promoting it to the ACE playbook via `ace_apply_delta ADD`

### Self-Evolution (ACE) — Use these to learn from experience

- **After completing a significant task**, self-reflect:
  1. Call `ace_get_playbook` to review current strategies
  2. Evaluate: which strategies helped? Which hurt?
  3. Call `ace_update_counters` with helpful/harmful tags for relevant bullets
     - For harmful tags, include a `failure_lesson` dict: `{failure_point, flawed_reasoning, counterfactual, prevention_principle, task_context}`
  4. If you discovered a new insight, call `ace_apply_delta` with an ADD operation
  5. Periodically call `ace_find_similar` and MERGE duplicate bullets
  6. Call `ace_prune` to remove strategies that have proven harmful
  7. Call `ace_evolve_from_failures` to generate NEW skills from accumulated failure lessons (triggers when ≥3 harmful bullets have structured lessons)
  8. Periodically call `ace_evolve_schema` to evaluate playbook health and evolve structure (MCE-inspired)

### Schema Evolution (MCE) — Evolve playbook structure itself

- Call `ace_evolve_schema()` to see health metrics and structural proposals
- Call `ace_evolve_schema(apply_proposal=N)` to apply a proposed change
- Call `ace_evolve_schema(rollback=True)` to revert to parent schema version
- Schema tracks: sections, decay rate, pruning thresholds, token budget, evolution threshold
- All mechanical (zero LLM calls). You review and decide. Rule-based: one deterministic proposal per call.

### Problem-Solving (RLM) — Use the sandbox for complex analysis

- For tasks requiring iterative exploration:
  1. `rlm_init` with the problem statement
  2. `rlm_execute` to explore: `search_repo()`, `get_file()`, process data
  3. `rlm_finalize` to return structured output
- Prefer metadata-only observations: note lengths, not full file contents

### Repo Index — Search the codebase

- `index_search(query)` — find files by symbol, path, or content keyword
- `index_build` — rebuild after significant code changes

## Project Structure

```
ccr/
  mcp_server.py     # MCP server: all 23 tools (GCC + ACE + RLM + Index)
  hooks/
    on_session_start.py  # Injects playbook + context on UserPromptSubmit
    on_session_end.py    # Logs session end OTA
    on_compact.py        # Reminds to commit before compaction
  core/
    memory.py       # GCC-style .ccr/ directory with COMMIT/BRANCH/MERGE/CONTEXT
    hooks.py        # Internal lifecycle hooks
    types.py        # All shared dataclasses, enums, configs
    exceptions.py   # Exception hierarchy: CCRError → ModelError, ConfigError, etc.
    engine.py       # Legacy proxy orchestrator (not used in MCP mode)
    router.py       # Legacy task classification (not used in MCP mode)
  context/
    indexer.py      # Zero-token repo indexing with per-language symbol extraction
    packer.py       # Legacy context packing (not used in MCP mode)
    prompts.py      # System prompts (legacy)
  rlm/
    orchestrator.py # Legacy RLM completion loop (not used in MCP mode)
    repl.py         # Sandboxed Python REPL with FINAL_VAR termination
  ace/
    playbook.py     # Playbook data structure: bullets, sections, delta ops
    agents.py       # Legacy agent wrappers (not used in MCP mode)
    engine.py       # Legacy ACE orchestrator (not used in MCP mode)
    prompts.py      # ACE-specific prompts (reference)
  models/           # Legacy model clients (not used in MCP mode)
  utils/
    tokens.py       # Token counting (tiktoken + LRU cache + heuristic fallback)
    costs.py        # Cost tracking (legacy)
    parsing.py      # Shared JSON extraction helpers
  gateway.py        # Legacy HTTP proxy (not used in MCP mode)
  cli.py            # Click CLI: start, init, index, context, status
config/
  default.yaml      # Default configuration
tests/
  unit/             # Unit tests (~600 tests)
  integration/      # Integration tests (mock backends)
vendor/             # Reference implementations (rlm, open-gcc, git-context-controller)
```

## Tech Stack

- Python 3.11+ (dev uses 3.14), venv at `.venv/`
- Dependencies: anthropic, openai, click, httpx, tiktoken, pyyaml, rich, mcp
- Tests: `pytest` — run with `source .venv/bin/activate && pytest tests/unit/ tests/integration/ -x -q`

## Key Patterns

- **MCP server**: 23 tools exposed via stdio transport — GCC memory, ACE playbook, RLM sandbox, repo index
- **CER-inspired pattern buffer** (arXiv:2506.06698): `gcc_commit` accepts optional `patterns_learned` list of transferable skills. Patterns deduped via word Jaccard (0.7 threshold), tracked in `.ccr/patterns.json` with occurrence counts. Patterns in 3+ commits → suggested for ACE playbook promotion (suggestion-only, not auto-add). Buffer capped at 200 entries with eviction by lowest occurrence. `gcc_patterns` tool queries the buffer. Zero LLM calls.
- **Memory**: GCC-inspired `.ccr/` directory. Commits track what was done/learned/planned. Branches for experiments. 5-level context retrieval. A-MAC admission control with correct polarity: S(m) = 0.50·TypePrior + 0.35·Novelty + 0.15·Recency. Algorithm 1 with S(m) vs S(m_conflict) comparison, FindConflict with recency-dampened similarity (CCR adaptation; paper uses pure cosine threshold 0.85 per §3.3), three-way admit/merge/reject, structural bypass.
- **Heuristic commit cross-linking** (A-MEM/MAGMA inspired taxonomy): When `gcc_commit` creates a new entry, scans recent commits for overlap and stores bidirectional cross-links in `.ccr/commit_links.json`. Four link types: entity (file-set Jaccard), causal (regex C### detection), supersession (replacement language), semantic (word Jaccard). `gcc_links` tool traverses via BFS; `gcc_context` level 5 shows links. All mechanical — zero LLM calls.
- **Two-tier playbook**: Global (`~/.ccr/global_playbook.txt`) + project (`.ccr/playbook.txt`). All ACE tools accept `scope="global"|"project"`. Cross-tier similarity detection via `ace_find_similar(scope="cross")`.
- **Temporal decay**: Bullet counters decay via `effective_score = raw * 0.95^days`. Unused 30d → 21%, 90d → 1%. `last_updated` persisted in companion JSON. Budget pruning uses effective scores.
- **Playbook evolution**: ACE playbook with helpful/harmful counters. Claude Code reflects and curates — no sub-model needed.
- **Structured Failure Lessons** (SkillRL-inspired): When tagging harmful, include a failure_lesson dict explaining why. Lessons accumulate, then `ace_evolve_from_failures` copies prevention_principle verbatim as new bullets when harmful count >= 3 (no cross-lesson synthesis, no teacher model). Companion data in `.ccr/failure_lessons.json`.
- **MCE-inspired schema evolution** (arXiv:2601.21557): Playbook structure (sections, decay rate, pruning thresholds, token budget) is itself evolvable. `ace_evolve_schema` computes health metrics (entropy-normalized section balance, utilization, harmful ratio, decay impact), proposes one structural change per call (rule-based, deterministic — not true (1+1)-ES), tracks schema versions with baseline comparison for rollback. Schema stored in `.ccr/playbook_schema.json`.
- **Semantic search** (A-RAG-inspired): `index_search` supports three modes — keyword (substring), semantic (ONNX embeddings or BM25 fallback), hybrid (default, combines both). Fallback chain: ONNX dense embeddings → BM25 term frequency → keyword substring. BM25 zero-dep fallback (CCR's own; not from paper). ONNX requires optional `ccr[semantic]` extras. File-level embeddings use path + symbols + first 10 lines per file.
- **Sandboxed REPL**: RLM-inspired REPL with repo tools (search_repo, get_file, FINAL_VAR). Claude Code drives the iteration loop. Note: MCP mode uses the Algorithm 2 architecture (LLM drives loop) which the paper identifies as less expressive than Algorithm 1 (loop drives LLM). The full Algorithm 1 loop exists in legacy code (rlm/orchestrator.py) but is inactive in MCP mode.
- **Exception hierarchy**: `CCRError` base with `recoverable` flag.
- **Shared JSON extraction**: `extract_json_from_llm()` and `extract_json_string()` in `utils/parsing.py`.

## Development Commands

```bash
source .venv/bin/activate
pytest tests/unit/ tests/integration/ -x -q  # Run all tests (1215 pass)
python -m ccr.mcp_server                     # Start MCP server (stdio)
ccr init                                     # Init .ccr/ in current dir
ccr index                                    # Build repo index
ccr status                                   # Show memory state
```

## MCP Configuration

`.mcp.json` in project root configures the MCP server for Claude Code:
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

## Research Papers (Implemented or Inspired By)

1. **GCC** (arXiv:2508.00031): Git Context Controller — version-controlled agent memory
   - COMMIT/BRANCH/MERGE/CONTEXT operations, metadata.yaml, OTA triples, rolling summary. Rolling summary defaults to mechanical concatenation in MCP mode (no sub-model for LLM synthesis); opt-in compressed_summary parameter available.
   - Admission control (A-MAC, arXiv:2603.04549): Correct polarity (higher score = more valuable). 3 of 5 factors: Novelty N(m)=1-max_sim (Eq. 3, word Jaccard proxy for SBERT), Recency R=exp(-0.01·hours) (Eq. 4, λ=0.01/hour, half-life 69h), Type Prior T(m) (§3.2, rule-based 6-type classifier). U(m) and C(m) omitted (require LLM). FindConflict with recency-dampened similarity (CCR adaptation; paper uses pure cosine threshold 0.85 per §3.3). Algorithm 1 three-way: admit/merge/reject. Type-prior bypass for structural ops.
   - See "Limitations vs. Paper (A-MAC Admission Control)" below for gaps
2. **RLM** (arXiv:2512.24601): Recursive Language Models — REPL-based execution
   - Metadata-only stdout, prompt as REPL variable, FINAL_VAR termination
3. **ACE** (arXiv:2510.04618): Agentic Context Engineering — inspired by ACE's evolving playbooks
   - Structured bullets with helpful/harmful counters, mechanical pruning (no LLM agents)
   - Delta operations (ADD/UPDATE/MERGE/REMOVE), deterministic merge
   - Playbook stored at `.ccr/playbook.txt`
   - See "Limitations vs. Paper (ACE Playbook Evolution)" below for gaps
4. **SkillRL** (arXiv:2602.08234): Inspired by SkillRL's failure-side skill distillation
   - Structured failure lessons: failure_point, flawed_reasoning, counterfactual, prevention_principle
   - Threshold-triggered verbatim extraction: when harmful count >= 3, copies prevention_principle as new bullets. No cross-lesson synthesis, no teacher model.
   - Hierarchical scope (general/task_specific), when_to_apply per SkillRL Table 5
   - Idempotent evolution with `evolved` flag, dedup against existing bullets
   - Extended data in `.ccr/failure_lessons.json` (backward-compatible format)
   - See "Limitations vs. Paper (SkillRL Failure Distillation)" below for gaps

5. **CER** (arXiv:2506.06698): Inspired by CER's transferable pattern extraction
   - Pattern buffer stores skill strings from gcc_commit. No dynamics (D_i), no trajectory distillation, no VLM retrieval — only the buffer data structure is adapted.
   - Dynamic Experience Buffer with dedup (existing buffer checked before adding)
   - CCR adaptation: Claude Code provides patterns at commit time (no VLM distiller)
   - Word Jaccard dedup (0.7 threshold), occurrence tracking, suggestion-based ACE promotion
   - See "Limitations vs. Paper (CER Pattern Extraction)" below for gaps

6. **A-MEM** (arXiv:2502.12110) + **MAGMA** (arXiv:2601.03236): Commit cross-linking taxonomy
   - Bidirectional commit cross-references using file-overlap and regex heuristics. Borrows link-type taxonomy from MAGMA; no A-MEM operations (LLM enrichment, link generation, memory evolution) implemented.
   - Four link types inspired by MAGMA's multi-graph edge taxonomy
   - See "Limitations vs. Paper (Commit Cross-Linking — A-MEM/MAGMA)" below for what is NOT implemented

7. **MCE** (arXiv:2601.21557): Meta Context Engineering — inspired by MCE's meta-level evolution
   - Bi-level optimization: meta-level evolves "skills" (playbook structure), base-level executes
   - PlaybookSchema versioning: sections, decay rate, pruning thresholds, token budget
   - Rule-based schema proposals (one per call, deterministic) with baseline comparison. Not true (1+1)-ES — proposals are from hardcoded rules, not stochastic/intelligent mutation.
   - 8 change types: ADD/REMOVE section, ADJUST_DECAY/PRUNING/EVOLUTION/BUDGET, REBALANCE, ROLLBACK
   - Entropy-normalized section balance, utilization/harmful/decay metrics, overall health composite
   - Stop criteria (MCE Appendix D.3.2): no proposals when overall_health ≥ 0.8
   - Schema history in `.ccr/playbook_schema.json` (backward-compatible: default schema if absent)
   - See "Limitations vs. Paper (MCE Schema Evolution)" below for gaps

8. **A-RAG** (arXiv:2602.03442): Inspired by A-RAG's hierarchical retrieval interfaces
   - Three search modes: keyword (§3.2 Eq 1), semantic (§3.2 Eq 3), hybrid
   - Dense embeddings via all-MiniLM-L6-v2 ONNX (384-dim, optional deps)
   - BM25 fallback (Okapi BM25, k1=1.5, b=0.75) when ONNX unavailable — CCR's own zero-dep fallback, not from A-RAG paper
   - Paragraph/function-level chunk embeddings (§3.1): `_split_into_chunks()` + `build_chunk_embeddings()` + `chunk_semantic_search()` (800-token max, Python def/class boundaries, paragraph breaks for other files)
   - Snippet extraction (§3.2 Eq 2): `extract_snippet()` — sentences containing query keywords, up to 3 matches
   - `hybrid_search(return_snippets=True)` delegates semantic component to chunk-level search
   - File-level summaries still used for file-level fallback and keyword mode
   - Claude Code IS the agent — no ReAct loop implementation needed
   - See "Limitations vs. Paper (A-RAG Semantic Search)" below for gaps

## Limitations vs. Paper (A-RAG Semantic Search)

Semantic search uses **chunk-level BM25/ONNX** (zero LLM calls). Key gaps vs. paper:

| A-RAG Feature | Paper Mechanism | CCR Implementation |
|---|---|---|
| Hierarchical Index (§3.1) | Corpus → 1000-token chunks → sentence embeddings | Files → paragraph/function-level chunks (800-token max) + chunk embeddings; file-level summaries for keyword mode |
| keyword_search (§3.2, Eq 1) | Keyword frequency × length scoring | Substring matching (path=3, symbol=5, content=1) |
| semantic_search (§3.2, Eq 3) | Sentence-level dense cosine (Qwen3-Embedding-0.6B) | Chunk-level dense cosine (all-MiniLM-L6-v2 ONNX) / BM25 fallback |
| chunk_read (§3.2) | Full content + context tracker C^read | get_file() in REPL / Read tool (no tracker) |
| Agent Loop (§3.3, Alg 1) | ReAct with interleaved tool use | Claude Code IS the agent (no implementation needed) |
| Context Tracker (§3.3) | C^read prevents re-reading same chunks | Not implemented (Claude Code tracks its own context) |
| Dynamic strategy selection | Agent chooses keyword vs semantic per step | `mode` parameter — user/agent chooses at call time |
| Snippet extraction (Eq 2) | Sentences containing keywords | `extract_snippet()` — keyword-matching sentences, up to 3, with " ... " separator |
| BM25 fallback | Not in paper | CCR's own zero-dep fallback (not from A-RAG) |

## Limitations vs. Paper (A-MAC Admission Control)

Admission control uses **3 of 5 factors with hand-tuned weights** (zero LLM calls). Key gaps vs. paper:

| A-MAC Feature | Paper Mechanism | CCR Implementation |
|---|---|---|
| Utility U(m) (§3.2 Factor 3) | LLM rates future usefulness on 1-5 scale | Not implemented (requires LLM) |
| Confidence C(m) (§3.2 Factor 4) | ROUGE-L overlap against source turns | Not implemented (requires reference corpus) |
| Similarity φ(m) (Eq. 3) | Sentence-BERT cosine similarity | ONNX cosine (all-MiniLM-L6-v2 dot product on L2-normalized vectors) when available; Word Jaccard (0.50·file + 0.50·keyword) fallback |
| Weight learning (Table 2) | 5-fold cross-validation on dialogue corpora | Hand-tuned: w_T=0.50, w_N=0.35, w_R=0.15 (informed by Table 2 ablation ranking) |
| Rejection threshold (Alg. 1 line 11) | Score-based threshold learned from data | Configurable parameter, default 0.0 (disabled) |
| Type Prior taxonomy (§3.2 Factor 5) | Classifies dialogue acts (question, instruction, etc.) | Classifies developer actions (progress, refactor, fix, etc.) — different domain |

## Limitations vs. Paper (ACE Playbook Evolution)

Playbook management uses **mechanical delta operations** driven by Claude Code (zero LLM agents). Key gaps vs. paper:

| ACE Feature | Paper Mechanism | CCR Implementation |
|---|---|---|
| Generator agent (§3.1) | LLM generates candidate bullets from task trajectory | Not implemented — Claude Code manually adds bullets via `ace_apply_delta ADD` |
| Reflector agent (§3.2) | LLM evaluates bullet quality and relevance | Not implemented — Claude Code manually reviews via `ace_get_playbook` |
| Curator agent (§3.3) | LLM prunes/merges/refines bullet library | Mechanical pruning by harmful ratio + temporal decay. No LLM-driven refinement |
| Automatic trajectory analysis | Task execution trace → bullet extraction | No automatic extraction. Claude Code must explicitly provide insights |
| Grow-and-refine loop (§4) | Iterative Generator→Reflector→Curator pipeline | Approximated by manual `ace_update_counters` + `ace_prune` + `ace_find_similar` |
| Bullet quality scoring | LLM-assessed relevance and specificity | Counter-based: helpful/harmful tallies with temporal decay |

## Limitations vs. Paper (SkillRL Failure Distillation)

Failure-side skill learning uses **threshold-triggered verbatim extraction** (zero LLM calls). Key gaps vs. paper:

| SkillRL Feature | Paper Mechanism | CCR Implementation |
|---|---|---|
| Teacher model M_T (§3.1) | LLM analyzes full trajectory to identify failure points | Manual tagging — Claude Code provides `failure_lesson` dict at counter-update time |
| GRPO RL training (§3.2, Eq. 2-4) | Group Relative Policy Optimization on skill-conditioned policy | Not implemented (N/A — no trainable model in MCP architecture) |
| Cross-lesson synthesis (§3.3) | LLM synthesizes general skill from multiple related failures | `evolve_from_failures` copies `prevention_principle` verbatim per lesson — no cross-lesson generalization |
| Cold-start SFT (§3.1) | Supervised fine-tuning on teacher-generated skills | Not implemented (N/A — MCP tools, not a trainable model) |
| Skill scoring (§3.4, Table 3) | Reward-weighted skill selection during inference | Counter-based helpful/harmful scoring with temporal decay |
| Trajectory replay buffer | Full (state, action, reward) trajectories stored | Only structured failure lessons stored — no full trajectories |

## Limitations vs. Paper (CER Pattern Extraction)

Pattern buffer uses **manual skill annotation** at commit time (zero VLM/LLM calls). Key gaps vs. paper:

| CER Feature | Paper Mechanism | CCR Implementation |
|---|---|---|
| VLM distillation (§3.2, Eq. 3) | Vision-Language Model extracts skills from (observation, action) pairs | Manual — Claude Code provides `patterns_learned` list at `gcc_commit` time |
| Dynamics D_i (§3.1, Eq. 1) | Environmental state transitions stored per experience | Not captured — only skills S_i (pattern strings) stored in buffer |
| Retrieval module (§3.3, Eq. 5) | Automatic similarity-based retrieval conditioned on current state | Manual — Claude Code queries `gcc_patterns` explicitly |
| Experience weighting (§3.4) | Recency + relevance weighted retrieval | Occurrence count only — no recency weighting on pattern retrieval |
| Buffer management (§3.2) | Priority replay with TD-error weighting | FIFO eviction by lowest occurrence count. Cap at 200 entries |
| Dedup mechanism | Embedding-based similarity | Word Jaccard (0.7 threshold) — sound data structure, lower discriminative power |

## Limitations vs. Paper (MCE Schema Evolution)

All schema evolution is **mechanical heuristics** (zero LLM calls). Key gaps vs. paper:

| MCE Feature | Paper Mechanism | CCR Implementation |
|---|---|---|
| Agentic crossover (§3.1) | LLM synthesizes new skill from τ + H_{k-1} | Rule-based heuristic proposals |
| Multi-metric evaluation (§3.2) | LLM evaluates context quality | Entropy, utilization, harmful ratio |
| Skill = methodology folder | Processing pipeline + scripts + templates | Schema = sections + thresholds |
| (1+1)-ES search (§3.1) | Full evolutionary search with LLM offspring | Single mechanical proposal per call |
| Stop criteria (Appendix D.3.2) | Skill learns dynamically when to stop | Fixed threshold: overall_health ≥ 0.8 |
| Skill database H (§3.1) | H = {(s_i, c_i, J_train, J_val)} | Version history with SchemaMetrics |
| Context function c(x) (§3.1) | Executable retrieval function + context files | Playbook serialize() under schema params |

## Limitations vs. Paper (Commit Cross-Linking — A-MEM/MAGMA)

All commit linking is **mechanical heuristics** (zero LLM calls). Key gaps vs. papers:

| Paper Feature | Paper Mechanism | CCR Implementation |
|---|---|---|
| A-MEM link generation (§3.2, Eq. 4-6) | Dense vector cosine similarity + LLM analysis | Word Jaccard + regex |
| A-MEM memory evolution (§3.3, Eq. 7) | LLM rewrites existing memories on new info | Not implemented (commits immutable) |
| MAGMA temporal graph (§3.2) | Immutable chronological chain | Implicit in sequential C### IDs, not stored as links |
| MAGMA causal graph (§3.2, Eq. 8) | LLM-inferred logical entailment | Regex detection of explicit C### references only |
| MAGMA semantic graph (§3.2) | Dense vector cosine (all-MiniLM-L6-v2) | Word Jaccard with ~100-word stop list |
| MAGMA entity graph (§3.2) | Abstract entity nodes (people, orgs) via LLM | File-path overlap (Jaccard) |
| MAGMA adaptive traversal (§3.3, Alg. 1) | Intent-aware beam search with Eq. 5-6 | Query-weighted BFS: when `query` provided, edge scores = cosine(query_vec, target_vec); candidates sorted by edge_score per hop. Not true beam search (no adaptive cutoff). **Largest ablation impact in MAGMA (Table 3)** |
| MAGMA fast/slow paths (§3.4) | Sync ingestion + async LLM consolidation | Sync only (all in commit path) |
| Link scan scope | Global retrieval across all memories | Fixed sliding window (`link_scan_window`, default 20 most recent commits). Older commits are never scanned for links at commit time |

## ACE Playbook Format

```
## SECTION_NAME
[slug-00001] helpful=5 harmful=0 :: Strategy or insight content here
```

Sections: STRATEGIES & INSIGHTS, CODE SNIPPETS & TEMPLATES, COMMON MISTAKES TO AVOID, PROBLEM-SOLVING HEURISTICS, CONTEXT CLUES & INDICATORS, OTHERS.
