# NeurIPS Re-Audit: CCR Implementation Fidelity (2026-03-22)

**Reviewer**: NeurIPS-style adversarial audit (re-audit after C020 fixes)
**Previous audit**: C018 (2026-03-21), average score 2.3/10
**Scope**: 8 papers, 10 implementations, line-by-line code review

---

## Scoring Criteria

- **9-10**: Core algorithm implemented correctly, honest documentation, minor gaps only
- **7-8**: Most of the algorithm correct, some gaps, documentation mostly honest
- **5-6**: Key mechanisms present but significant gaps or simplifications
- **3-4**: Inspired-by but missing core algorithm components
- **1-2**: Name/concept only, core mechanism absent or wrong

---

### GCC (arXiv:2508.00031) — Git Context Controller

**Score: 6/10** (was 5/10 in C018)

**What matches**:
- COMMIT/BRANCH/MERGE/CONTEXT operations fully implemented (`memory.py:264-462`, `205-261`, `1676-1737`, `1760-1907`)
- `.ccr/` directory structure with `branches/`, `commits.md`, `log.md`, `summary.md`, `metadata.yaml` — faithful to open-gcc layout
- OTA logging (Observation-Thought-Action triples) in `_format_ota_log` used throughout
- Rolling summary S_t = f(S_{t-1}, D_t) (`memory.py:1537-1598`). Three-tier: caller-provided compressed summary (restores paper's LLM compression), legacy sub-client, and mechanical concatenation with structured truncation. The two-call pattern for MCP mode is an honest workaround.
- Context windowing: 5-level retrieval with offset parameter (`memory.py:1760-1907`), windowed commit reading
- Branch creation enforces kebab-case, requires being on main, writes summary.md with purpose/hypothesis/conclusion — matches GCC paper
- Merge performs auto-CONTEXT before merge (`memory.py:1688`), integrates branch summary into main, copies branch log with provenance header — all per GCC paper
- Commit references OTA log slice (`memory.py:384-386`)

**Key gaps**:
- Rolling summary in MCP mode defaults to semicolon concatenation (Strategy 3, `memory.py:1588-1598`). The paper requires LLM-regenerated summaries for the S_t = f(S_{t-1}, D_t) property. The `compressed_summary` parameter is opt-in and relies on the caller (Claude Code) to manually do a two-call pattern — this is not automatic.
- No metadata.yaml auto-population of file trees, dependencies, or config — the template exists but fields are never updated from actual project state
- The context K-window from the paper (scrollable window over commit history) is implemented via `offset` parameter, but there is no automatic window sizing based on token budget

**vs. C018**: Improved (+1). Rolling summary now has the `compressed_summary` path and structured truncation instead of blind tail-truncation. Documentation is honest about the concatenation default.

---

### A-MAC (arXiv:2603.04549) — Admission Memory Controller

**Score: 5/10** (was 3/10 in C018)

**What matches**:
- **Algorithm 1 logic is now correct** (`memory.py:306-354`):
  - Step 3: Rejection when `score < rejection_threshold` — correct polarity (low = reject)
  - Step 4: FindConflict via `similarity >= admission_threshold` — correct
  - Step 5: `S(m) > S(m_conflict)` triggers REPLACE via merge — **FIXED from C018** (was inverted)
  - Step 6: `S(m) <= S(m_conflict)` falls through to normal add — correct coexistence
- Three of five factors implemented (`memory.py:1199-1383`):
  - N(m) = 1 - max_raw_similarity (Eq. 3) — correctly uses raw sim, not recency-modulated
  - R(m) = exp(-0.01 * hours) (Eq. 4) — lambda=0.01/hour correctly implemented
  - T(m) = type_prior(classify(m)) (Section 3.2 Factor 5) — 6-type classifier with prior values
- **Novelty/recency separation fixed**: `best_raw_similarity` used for N(m), `effective_sim = raw * R_conflict` used only for FindConflict threshold — correct separation per paper intent
- S(m_conflict) recomputed with current R(m') at query time (`memory.py:1326-1343`) so old commits properly decay — thoughtful implementation
- Structural bypass for merge/branch types (`memory.py:1271-1279`)
- Type Prior weights informed by Table 2 ablation (T most impactful: w_T=0.50)
- Documentation in `compute_admission_score` docstring is detailed and mostly accurate

**Key gaps**:
- **U(m) and C(m) omitted** (2 of 5 factors). The paper's Eq. 1 has 5 factors. Missing Utility (requires LLM) and Confidence (requires ROUGE-L). Honestly documented.
- **Similarity computation uses word Jaccard** (0.50*file + 0.50*keyword) instead of Sentence-BERT cosine (paper Eq. 3). Lower discriminative power. This is a significant approximation.
- **S(m_conflict) extraction from stored score** (`memory.py:1336-1341`) uses algebra to back-derive novelty from the composite score, which introduces error. The paper would store factors separately.
- The recency-modulated similarity for FindConflict (`effective_sim = raw * R_conflict`) is a CCR adaptation — the paper uses a pure cosine threshold of 0.85 per Section 3.3. The modulation changes the semantics: old duplicates are harder to detect.
- Weight learning: paper uses 5-fold cross-validation (Table 2). CCR hand-tunes weights based on the ablation ranking.
- Rejection threshold default is 0.0 (disabled). The paper uses a learned threshold.

**vs. C018**: Significantly improved (+2). The critical merge logic inversion is fixed. Novelty/recency conflation is fixed. Documentation is honest about limitations.

---

### ACE (arXiv:2510.04618) — Agentic Context Engineering

**Score: 3/10** (was 2/10 in C018)

**What matches**:
- Playbook data structure with sections and structured bullets (`playbook.py:237-322`) matches Figure 3 format
- Delta operations (ADD, UPDATE, MERGE, REMOVE) in `DeltaOperation` dataclass (`playbook.py:205-213`)
- `apply_delta` correctly handles all four op types
- `update_bullet_counts` with helpful/harmful tagging (`playbook.py:357-399`)
- `prune_problematic` removes net-harmful bullets above threshold (`playbook.py:556-574`)
- `find_similar_pairs` for deduplication with combined word Jaccard + character trigram (`playbook.py:513-546`)
- `enforce_token_budget` removes lowest effective-score bullets — efficient O(n log n) (`playbook.py:576-604`)
- Temporal decay on effective_score: `score * decay_rate^days` (`playbook.py:155-171`)
- Two-tier playbook (global + project) with cross-tier similarity detection

**Key gaps**:
- **No Generator agent** (Section 3.1): The paper's core contribution is an LLM agent that generates candidate bullets from task trajectories. CCR relies entirely on manual `ace_apply_delta ADD` calls from Claude Code. This is the single largest gap.
- **No Reflector agent** (Section 3.2): No LLM evaluates bullet quality or relevance. Claude Code does this manually via `ace_get_playbook` and human judgment.
- **No Curator agent** (Section 3.3): The paper's Curator uses LLM to prune, merge, and refine. CCR uses mechanical pruning (harmful ratio + decay) and mechanical similarity detection. No refinement.
- **No automatic trajectory analysis**: The paper extracts bullets from execution traces. CCR requires manual bullet creation.
- **No Grow-and-refine loop** (Section 4): The paper's iterative Generator-Reflector-Curator pipeline is approximated by manual tool calls, but there is no automatic iteration.

The fundamental issue: ACE is about *agentic* context engineering — LLM agents curate the playbook. CCR implements the *data structure* (playbook with bullets, sections, counters) but not the *agents*. This is honestly documented in the limitations table.

**vs. C018**: Improved (+1). Documentation is now honest about what is and is not implemented. The code itself was already functional; the improvement is in disclosure.

---

### SkillRL (arXiv:2602.08234) — Skill-based RL from Failure

**Score: 3/10** (was 1/10 in C018)

**What matches**:
- `FailureLesson` dataclass (`playbook.py:63-123`) captures the 4 structured fields the paper describes: failure_point, flawed_reasoning, counterfactual, prevention_principle
- Extended with `task_context` mapping to SkillRL's trajectory context (P0-2/P0-3)
- `evolved` flag for idempotent processing (N3) (`playbook.py:84`)
- `evolve_from_failures` (`playbook.py:676-745`) implements threshold-triggered skill extraction: when harmful count >= 3, copies prevention_principle as new bullets. Deduplicates against existing bullet content (N2, line 703).
- `when_to_apply` field on bullets per Table 5 format
- `scope` field (general/task_specific) per Section 3.2 hierarchical SkillBank
- Failure lessons persisted in companion JSON with backward-compatible format (`playbook.py:773-874`)
- Docstrings now honestly state "manual annotation" vs paper's teacher model M_T

**Key gaps**:
- **No teacher model M_T** (Section 3.1): The paper's core contribution is a teacher model that analyzes full trajectories to identify failure points: s- = M_T(tau-, d). CCR relies on Claude Code manually providing failure_lesson dicts. Honestly documented.
- **No cross-lesson synthesis** (Section 3.3): `evolve_from_failures` copies each lesson's prevention_principle verbatim — no synthesis across related failures. The paper uses M_T to generalize.
- **No GRPO RL training** (Section 3.2): N/A for MCP architecture, but this is the paper's core technical contribution.
- **No cold-start SFT** (Section 3.1): N/A.
- **No trajectory replay buffer**: Only structured failure lessons are stored, not full (state, action, reward) trajectories.

The implementation captures the *failure lesson data structure* from SkillRL faithfully, including the skill format from Table 5. The evolution mechanism (threshold-triggered extraction) is a reasonable zero-LLM approximation. But the paper's core contributions (teacher model, GRPO training, cross-lesson synthesis) are absent.

**vs. C018**: Improved (+2). Documentation fixes removed overclaiming. The `FailureLesson` docstring (`playbook.py:65-76`) now honestly says "manual annotation" not "teacher model". The `evolve_from_failures` docstring (`playbook.py:677-684`) now says "verbatim extraction" not "cross-lesson synthesis".

---

### MCE (arXiv:2601.21557) — Meta Context Engineering

**Score: 3/10** (was 2/10 in C018)

**What matches**:
- `PlaybookSchema` versioning with parent tracking (schema history)
- `compute_metrics` (`playbook.py:909-1008`) computes structural health: section balance (normalized Shannon entropy), utilization rate, harmful ratio, unused ratio, decay impact, overall health composite
- `propose_schema_changes` (`playbook.py:1010-1224+`) proposes ONE change per call (matches (1+1)-ES structure). 8 change types: ROLLBACK, ADD_SECTION, REMOVE_SECTION, ADJUST_DECAY, ADJUST_PRUNING, REBALANCE, ADJUST_BUDGET (+ADJUST_EVOLUTION implied). Priority ordering.
- Stop criteria: health >= 0.8 threshold (`playbook.py:1066-1067`) per Appendix D.3.2
- Rollback check against baseline metrics (`playbook.py:1046-1063`)
- Schema stored in `.ccr/playbook_schema.json` with version history (backward-compatible default)
- Confidence scores on proposals
- Section clustering in OTHERS bullets for ADD_SECTION proposals (`playbook.py:1069-1088`)

**Key gaps**:
- **No agentic crossover** (Section 3.1): The paper uses an LLM to synthesize new "skills" (methodology folders). CCR uses hardcoded rule-based heuristic conditions — no LLM, no stochastic mutation. Honestly documented.
- **Not true (1+1)-ES**: The paper's evolutionary search generates stochastic offspring. CCR's proposals are deterministic condition checks. Same conditions always produce same proposal.
- **No skill = methodology folder**: The paper's "skill" is a complete processing pipeline (scripts + templates + retrieval function). CCR's "schema" is a parameter set (sections, decay, thresholds, budget). Different abstraction level.
- **No J_train/J_val evaluation**: Paper evaluates context quality on actual task performance. CCR uses structural metrics (entropy, utilization).
- **No multi-metric LLM evaluation**: Paper uses LLM to evaluate context quality (Section 3.2).
- Docstring in `compute_metrics` (`playbook.py:910-916`) now honestly states "structural metrics" not "task-performance metrics". `propose_schema_changes` docstring (`playbook.py:1020-1024`) now says "deterministic condition checks" not "LLM-generated offspring".

**vs. C018**: Improved (+1). Documentation fixes in code comments are honest about the gap between rule-based heuristics and MCE's LLM-driven evolutionary search.

---

### RLM (arXiv:2512.24601) — Recursive Language Models

**Score: 4/10** (was 3/10 in C018)

**What matches**:
- Sandboxed REPL with restricted builtins, AST validation, module allowlist (`repl.py:32-283`)
- FINAL_VAR termination pattern (`repl.py:397-417`) — matches paper's completion signal
- Metadata-only stdout: `_summarize_stdout` (`mcp_server.py:1246-1272`) replaces long output with line/char counts + head/tail — per Section 3
- Prompt as REPL variable: `task_prompt` loaded into REPL namespace (`mcp_server.py:1222`)
- Repo tools: `get_file()`, `search_repo()`, `estimate_tokens()`, `SHOW_VARS()` — per paper's REPL scaffolding
- `llm_query` / `rlm_query` / batched variants available as REPL tools (per paper's recursive sub-call mechanism)
- Kernel sandbox option (macOS Seatbelt, `repl.py:320-342`) — defense in depth beyond paper
- Context loading: large payloads written to temp file for disk-based access (`repl.py:674-685`) — per paper

**Key gaps**:
- **Algorithm 2 architecture, not Algorithm 1**: The `rlm_init` docstring (`mcp_server.py:1189-1191`) now honestly states: "The paper's autonomous generate-execute loop (Algorithm 1) is NOT active in MCP mode — Claude Code drives the iteration loop manually via rlm_execute calls." This is the fundamental gap. The paper's Algorithm 1 has a *loop-driven* architecture where the orchestrator decides when to generate code vs. finalize. MCP mode is Claude Code manually calling rlm_execute, which the paper explicitly identifies as less expressive (Algorithm 2).
- **No autonomous iteration**: No generate->observe->decide->generate loop. Claude Code must manually decide each step.
- **No step budget management**: Paper has explicit step limits and termination conditions. CCR relies on Claude Code's judgment.
- The legacy `rlm/orchestrator.py` contains a more faithful Algorithm 1 implementation but is explicitly unused in MCP mode.

The REPL substrate is well-implemented with good security (AST validation, module allowlist, Seatbelt sandbox, bounded stdout). The metadata-only stdout is a faithful implementation. But the paper's core contribution is the *loop architecture*, which is absent in MCP mode.

**vs. C018**: Improved (+1). The `rlm_init` docstring now honestly discloses Algorithm 2 vs Algorithm 1 gap. Code was already functional; improvement is in honesty of disclosure.

---

### CER (arXiv:2506.06698) — Contextual Experience Replay

**Score: 3/10** (was 2/10 in C018)

**What matches**:
- Pattern buffer data structure (`memory.py:1020-1095`): stores skill strings from gcc_commit with deduplication
- Word Jaccard dedup at 0.7 threshold (`_find_matching_pattern`)
- Occurrence tracking: `occurrence_count`, `commit_ids` list per pattern
- Buffer size enforcement: cap at 200 entries, evict by lowest occurrence count (`memory.py:1075-1094`)
- Promotion suggestions when occurrence >= threshold (default 3 commits) — suggestion-only, not auto-add
- `gcc_patterns` tool queries buffer with filtering (min_occurrences, search_term, include_promoted)
- Recurring patterns displayed in context level 2+ (`memory.py:1819-1837`)
- Pattern union during commit merging (`memory.py:1492-1507`)

**Key gaps**:
- **No VLM distillation** (Section 3.2, Eq. 3): Paper uses Vision-Language Model to extract skills from (observation, action) pairs. CCR relies on Claude Code manually providing `patterns_learned` at commit time.
- **No dynamics D_i** (Section 3.1, Eq. 1): Paper stores environmental state transitions per experience. CCR stores only skill strings.
- **No automatic retrieval module** (Section 3.3, Eq. 5): Paper uses similarity-based retrieval conditioned on current state. CCR requires manual `gcc_patterns` queries.
- **No experience weighting** (Section 3.4): Paper uses recency + relevance weighted retrieval. CCR only has occurrence count.
- **No TD-error weighting**: Paper uses priority replay with temporal-difference error. CCR uses FIFO eviction by lowest occurrence.
- **Dedup uses word Jaccard (0.7)**: Sound but lower discriminative power than embedding-based similarity.

The pattern buffer is a reasonable adaptation of CER's Dynamic Experience Buffer data structure, but the paper's core contributions (VLM distillation, trajectory-based extraction, automatic retrieval) are absent.

**vs. C018**: Improved (+1). Documentation no longer overclaims.

---

### A-MEM (arXiv:2502.12110) — Agentic Memory

**Score: 2/10** (was 1/10 in C018)

**What matches**:
- Bidirectional commit cross-linking (`memory.py:681-701`) — `_add_link` stores links in both directions, similar to A-MEM's Zettelkasten-inspired bidirectional references
- BFS traversal of linked commits (`get_linked_commits`, `memory.py:858+`) with configurable max_hops and result cap — inspired by A-MEM's memory graph traversal
- Link deduplication by higher score (`memory.py:688-696`)
- Docstring (`memory.py:683`) correctly labels this as "A-MEM Zettelkasten" inspiration

**Key gaps**:
- **No LLM link generation** (Section 3.2, Eq. 4-6): A-MEM uses dense vector cosine + LLM analysis to generate links. CCR uses word Jaccard + regex (with optional ONNX cosine for semantic links only since C020).
- **No memory evolution** (Section 3.3, Eq. 7): A-MEM's key contribution is that existing memories are LLM-rewritten when new related information arrives. CCR commits are immutable.
- **No memory enrichment**: A-MEM generates structured "memory notes" with concepts, emotions, implications. CCR stores raw commit data.
- The ONNX embedding wiring for semantic links (`memory.py:825-829`) is a genuine improvement since C020 — dense cosine similarity between commit embeddings when available, falling back to word Jaccard. This moves closer to A-MEM's embedding-based linking.

A-MEM's core contribution (agentic memory evolution — memories that rewrite themselves) is fundamentally absent. The bidirectional linking structure is a reasonable starting point but misses the "agentic" part entirely.

**vs. C018**: Improved (+1). ONNX embeddings now wired into `_compute_links` for semantic links. Still far from the paper.

---

### MAGMA (arXiv:2601.03236) — Multi-graph Agent Memory Architecture

**Score: 3/10** (was 1.5/10 in C018)

**What matches**:
- Four link types matching MAGMA's multi-graph taxonomy (`memory.py:552`): entity, causal, supersession, semantic
- **Entity links** (`memory.py:793-802`): file-set Jaccard > threshold. MAGMA uses LLM-extracted abstract entities; CCR uses file paths. Different domain but analogous structure.
- **Causal links** (`memory.py:804-821`): regex detection of C### IDs in commit text, validated against existing commits. MAGMA uses LLM-inferred logical entailment; CCR detects explicit references only.
- **Supersession links** (`memory.py:805-811`): replacement language detection near commit IDs via `_SUPERSESSION_KEYWORDS` regex — a heuristic unique to CCR with no direct MAGMA analog.
- **Semantic links** (`memory.py:823-842`): Dense cosine similarity via ONNX embeddings when available (`memory.py:825-829`), word Jaccard fallback. MAGMA uses all-MiniLM-L6-v2 — CCR now uses the same model for ONNX path. This is a genuine match when ONNX is available.
- Sliding window scan (`memory.py:763`, `config.link_scan_window` default 20) — not global but bounded and documented
- Comment at `memory.py:869` correctly notes this is "plain BFS, not MAGMA's intent-aware beam search"

**Key gaps**:
- **No adaptive traversal** (Section 3.3, Algorithm 1): MAGMA's biggest contribution per ablation (Table 3) — intent-aware beam search with relevance scoring (Eq. 5-6). CCR uses plain BFS. The docstring (`memory.py:869`) is honest about this.
- **No temporal graph**: MAGMA maintains an immutable chronological chain. CCR relies on sequential commit IDs implicitly.
- **No LLM-inferred causal links**: CCR detects only explicit C### references, not implicit logical entailment.
- **No fast/slow paths** (Section 3.4): MAGMA has sync ingestion + async LLM consolidation. CCR is sync only.
- **Scan window limitation**: Only scans last 20 commits for links. Older commits are never linked at commit time. Honestly documented.
- Entity links use file paths, not abstract entity nodes (people, organizations, concepts).

The ONNX embedding wiring for semantic links is a meaningful improvement — when ONNX is available, semantic links use the same model (all-MiniLM-L6-v2) as MAGMA's paper. But the core contribution (adaptive traversal) is absent.

**vs. C018**: Improved (+1.5). ONNX embeddings for semantic links is a genuine code improvement. Documentation is honest about gaps. The four-link taxonomy is well-implemented within the heuristic constraints.

---

### A-RAG (arXiv:2602.03442) — Agentic RAG

**Score: 5/10** (was 3/10 in C018)

**What matches**:
- Three search modes: keyword, semantic, hybrid (`indexer.py:98-108`)
- **Keyword search** (`indexer.py:210-243`): substring matching on paths (score 3), symbols (score 5), content (score 1). The docstring (`indexer.py:102-103`) honestly notes this uses "substring matching not frequency*length scoring" vs paper's Eq 1.
- **Semantic search** via ONNX embeddings (`indexer.py:398-445`, `embeddings.py`): all-MiniLM-L6-v2, 384-dim, mean pooling + L2 normalization, cosine similarity. This is a genuine dense retrieval implementation. The paper uses Qwen3-Embedding-0.6B but the approach (dense cosine) is the same.
- **File-level summaries** (`indexer.py:260-272`): path + symbols + first 10 lines per file. The docstring notes this is "A-RAG Section 3.1 adaptation". The paper uses sentence-level chunks; CCR uses file-level summaries.
- **BM25 fallback** (`indexer.py:306-364`): Okapi BM25 with k1=1.5, b=0.75 — correct formula. The docstring (`indexer.py:309-313`) now correctly states "CCR's own zero-dep fallback" and "Not from the A-RAG paper (arXiv:2602.03442), which does not mention BM25." **FIXED from C018.**
- **Hybrid search** (`indexer.py:447-511`): combined keyword + semantic with configurable weights. The docstring (`indexer.py:456-463`) honestly notes "CCR's own score-fusion design" and "A-RAG uses agent-driven tool selection."
- **Batch embedding** (`embeddings.py:120-156`): efficient batch processing with mean pooling and L2 normalization
- **Embedding persistence** (`embeddings.py:173-193`): gzip-compressed JSON for reuse
- Auto-download from HuggingFace CDN (`embeddings.py:85-118`)

**Key gaps**:
- **File-level, not sentence-level**: Paper uses 1000-token chunks with sentence embeddings. CCR uses file-level summaries. Coarser granularity.
- **No chunk_read with context tracker** (Section 3.2): Paper tracks C^read to prevent re-reading. CCR returns file paths, not content snippets.
- **No snippet extraction** (Eq 2): Paper extracts sentences containing keywords. CCR returns whole file paths.
- **Hybrid is score fusion, not agent-driven**: Paper's agent dynamically chooses keyword vs semantic per step. CCR fuses scores mechanically.
- **No ReAct agent loop** (Section 3.3, Algorithm 1): Paper has a ReAct loop with interleaved tool use. CCR correctly notes "Claude Code IS the agent."

**vs. C018**: Improved (+2). The BM25 false attribution is fixed across all docstrings (6 fixes in indexer.py per C019). Hybrid search docstring is honest. The ONNX embedding implementation is solid and well-tested.

---

## Overall

| Paper | C018 Score | C022 Score | Delta | Key Improvement |
|-------|-----------|-----------|-------|-----------------|
| GCC | 5/10 | 6/10 | +1 | Rolling summary compression path, structured truncation |
| A-MAC | 3/10 | 5/10 | +2 | Merge logic inversion fixed, novelty/recency separation |
| ACE | 2/10 | 3/10 | +1 | Honest documentation of agent gaps |
| SkillRL | 1/10 | 3/10 | +2 | Honest docstrings, data structure is well-implemented |
| MCE | 2/10 | 3/10 | +1 | Honest distinction: structural metrics vs task-performance |
| RLM | 3/10 | 4/10 | +1 | Algorithm 1 vs 2 gap honestly disclosed |
| CER | 2/10 | 3/10 | +1 | Documentation no longer overclaims |
| A-MEM | 1/10 | 2/10 | +1 | ONNX embeddings wired for semantic links |
| MAGMA | 1.5/10 | 3/10 | +1.5 | ONNX semantic links use same model as paper |
| A-RAG | 3/10 | 5/10 | +2 | BM25 false attribution fixed, ONNX implementation solid |

**Average score: 3.7/10** (was 2.3/10 in C018, improvement of +1.4)

---

## Analysis of Score Improvement

### What improved (documentation honesty, +~0.8 average):
The C020 fixes addressed systematic overclaiming. False attributions removed (BM25 to A-RAG, equation numbers), misleading comments fixed, and CLAUDE.md limitations tables are now comprehensive and accurate. Every docstring I checked was honest about what is and is not from the paper. This is significant — a 2/10 implementation with honest docs is more useful than a 2/10 implementation with misleading docs.

### What improved (code, +~0.6 average):
1. A-MAC merge logic inversion: fixing `S(m) > S(m_conflict)` polarity was the single most critical code bug. This alone moved A-MAC from 3 to 5.
2. ONNX embeddings wired into commit cross-linking: semantic links now use dense cosine via all-MiniLM-L6-v2 when available, with word Jaccard fallback. This genuinely improves A-MEM and MAGMA fidelity.
3. Rolling summary `compressed_summary` parameter: enables the GCC paper's S_t = f(S_{t-1}, D_t) via a two-call pattern.

### What did NOT improve (fundamental architecture gaps):
- ACE, SkillRL, MCE, CER all require LLM agents that generate/curate/synthesize. CCR's MCP architecture has no sub-model, so these will never exceed ~4/10 without architectural change.
- RLM's Algorithm 1 (loop-driven) requires an orchestrator that CCR's MCP mode cannot provide — Claude Code is the loop.
- A-MEM's core contribution (memories that evolve via LLM rewriting) is fundamentally incompatible with immutable commits.
- MAGMA's adaptive traversal (biggest ablation impact per Table 3) would require an LLM scoring function.

### Honest assessment:
The 3.7/10 average is appropriate for a zero-LLM-call adaptation of 8 papers that fundamentally require LLM components. The project has made the best possible use of mechanical heuristics and data structures from these papers. The documentation is now honest about every gap. The code quality is high — well-tested, thread-safe, defensive. The issue is not code quality but architectural limitation: an MCP server without a sub-model cannot implement algorithms that require LLM calls.

---

## Recommendations

1. **Do not claim higher fidelity** — 3.7/10 is an honest score for zero-LLM adaptations of LLM-dependent papers.
2. **Consider a "CCR concepts" framing** instead of "implements paper X" — the data structures are sound, the algorithms are not.
3. **The ONNX embedding path is a genuine win** — expand its use to admission control (replace word Jaccard in A-MAC similarity), pattern buffer dedup, and playbook similarity detection.
4. **Rolling summary compression** — the two-call pattern is clever but fragile. Consider making the compression prompt appear automatically after every N commits, not just when length exceeds threshold.
5. **A-RAG is the highest-fidelity implementation** (5/10) — consider investing here: sentence-level chunking + snippet extraction would push it to 7/10.

---

*Audit conducted 2026-03-22 by NeurIPS-style adversarial reviewer. All line numbers reference the working tree at audit time.*
