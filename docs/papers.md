# CCR Paper Fidelity Report

CCR draws from 16 research papers. This document classifies each by implementation fidelity and lists what was implemented versus what was not.

---

## Tier 1: Implemented (>70%)

### GCC -- Git Context Controller

**Paper:** arXiv:2508.00031
**Fidelity:** ~75%

**Implemented:**
- COMMIT/BRANCH/MERGE/CONTEXT operations with structured commit records.
- `metadata.yaml` project metadata.
- OTA (Observation-Thought-Action) triples per commit.
- Rolling summary on main branch.
- 5-level context retrieval.
- Branch-level isolation with purpose/hypothesis/outcome tracking.
- Merge with outcome classification (success/failure/partial).

**NOT implemented:**
- LLM-synthesized rolling summaries (uses mechanical concatenation; opt-in `compressed_summary` parameter for Claude Code to compress).
- Paper's full metadata schema (milestones, dependencies).

---

### ACE -- Agentic Context Engineering

**Paper:** arXiv:2510.04618
**Fidelity:** ~70%

**Implemented:**
- Structured bullets with helpful/harmful counters.
- Delta operations: ADD, UPDATE, MERGE, REMOVE.
- Temporal decay (`effective_score = raw * 0.95^days`).
- Token budget enforcement (prune lowest-scoring bullets).
- Similarity-based deduplication (Jaccard + trigram).
- Two-tier playbook (global + project).
- GRPO group-relative advantages for policy ranking.
- Optional 3-agent pipeline (Generator -> Reflector -> Curator) when sub-model available.

**NOT implemented:**
- Automatic trajectory analysis (Claude Code must explicitly provide insights).
- LLM-driven Reflector and Curator quality scoring in the core loop (only available as optional pipeline).
- Paper's grow-and-refine iterative loop (approximated by manual counter updates + prune + find_similar).

---

### RLM -- Recursive Language Models

**Paper:** arXiv:2512.24601
**Fidelity:** ~70%

**Implemented:**
- Sandboxed Python REPL with AST validation and restricted builtins.
- `FINAL_VAR` termination pattern.
- Metadata-only stdout (truncate long output to line/char counts + head/tail).
- Prompt as REPL variable (`task_prompt`).
- Repo tools: `search_repo()`, `get_file()`, `estimate_tokens()`, `SHOW_VARS()`.
- macOS Seatbelt kernel sandbox (optional, for standalone execution).

**NOT implemented:**
- Algorithm 1 autonomous generate-execute loop (loop drives LLM). The paper's more expressive architecture exists in legacy code (`rlm/orchestrator.py`) but is inactive in MCP mode.
- MCP mode uses Algorithm 2 (LLM drives loop via `rlm_execute` calls), which the paper identifies as less expressive.
- Recursive depth management.

---

## Tier 2: Substantially Adapted (30-70%)

### A-MAC -- Admission Control

**Paper:** arXiv:2603.04549
**Fidelity:** ~50%

**Implemented:**
- Scoring function: `S(m) = 0.50*TypePrior + 0.35*Novelty + 0.15*Recency`.
- Novelty `N(m) = 1 - max_sim` (word Jaccard proxy for SBERT, Eq. 3).
- Recency `R = exp(-0.01*hours)` (Eq. 4, half-life 69h).
- Type Prior `T(m)` (rule-based 6-type classifier for developer actions).
- FindConflict with recency-dampened similarity (CCR adaptation).
- Algorithm 1 three-way: admit/merge/reject.
- Type-prior bypass for structural operations.
- Configurable thresholds.

**NOT implemented:**
- Utility `U(m)` (requires LLM to rate future usefulness, 1-5 scale).
- Confidence `C(m)` (requires ROUGE-L overlap against source turns).
- Weight learning via 5-fold cross-validation (hand-tuned weights).
- Learned rejection threshold (uses configurable parameter, default 0.0 disabled).
- Dense vector similarity uses ONNX when available, word Jaccard fallback.

---

### A-RAG -- Hierarchical Retrieval

**Paper:** arXiv:2602.03442
**Fidelity:** ~45%

**Implemented:**
- Three search modes: keyword, semantic, hybrid.
- Dense embeddings via all-MiniLM-L6-v2 ONNX (384-dim).
- BM25 zero-dep fallback (Okapi BM25, k1=1.5, b=0.75) -- CCR's own, not from paper.
- Paragraph/function-level chunk embeddings (800-token max, Python def/class boundaries).
- Snippet extraction: keyword-matching sentences, up to 3 per file.
- File-level summaries for keyword mode.

**NOT implemented:**
- Paper's sentence-level embedding hierarchy (uses file/chunk level).
- Context tracker `C^read` preventing re-reading same chunks.
- ReAct agent loop (Claude Code IS the agent).
- Dynamic strategy selection (mode is chosen by caller).
- Paper uses Qwen3-Embedding-0.6B; CCR uses all-MiniLM-L6-v2.

---

### CER -- Contextual Experience Replay

**Paper:** arXiv:2506.06698
**Fidelity:** ~35%

**Implemented:**
- Pattern buffer storing skill strings from `gcc_commit`.
- Word Jaccard deduplication (0.7 threshold).
- Occurrence tracking with FIFO eviction at 200 entries.
- Promotion suggestion when patterns reach 3+ occurrences.
- Quality scoring (EvolveR-inspired Bayesian scoring).

**NOT implemented:**
- VLM distillation of skills from (observation, action) pairs.
- Dynamics `D_i` (environmental state transitions).
- Automatic similarity-based retrieval conditioned on state.
- Experience weighting (recency + relevance).
- Priority replay with TD-error weighting.
- Embedding-based dedup (uses word Jaccard).

---

### MCE -- Meta Context Engineering

**Paper:** arXiv:2601.21557
**Fidelity:** ~35%

**Implemented:**
- PlaybookSchema versioning: sections, decay rate, pruning thresholds, token budget.
- Rule-based schema proposals (one per call, deterministic).
- 8+ change types: ADD/REMOVE_SECTION, ADJUST_DECAY/PRUNING/EVOLUTION/BUDGET, REBALANCE, ROLLBACK, ADJUST_SEARCH_THRESHOLD, ADJUST_SCAN_WINDOW.
- Health metrics: entropy-normalized section balance, utilization, harmful ratio, decay impact.
- Baseline comparison for rollback decisions.
- Stop criteria: no proposals when `overall_health >= 0.8`.
- Schema history with version tracking.

**NOT implemented:**
- Agentic crossover (LLM synthesizes new skill from trajectory + history).
- Multi-metric LLM evaluation.
- True (1+1)-ES evolutionary search (proposals are hardcoded rules, not stochastic mutation).
- Skill = executable methodology folder (schema = threshold parameters).
- Dynamic stop criteria learning.

---

### SkillRL -- Failure Distillation

**Paper:** arXiv:2602.08234
**Fidelity:** ~30%

**Implemented:**
- Structured failure lessons: `failure_point`, `flawed_reasoning`, `counterfactual`, `prevention_principle`.
- Threshold-triggered verbatim extraction (copies `prevention_principle` as new bullet when harmful count >= 3).
- Hierarchical scope (`general`/`task_specific`), `when_to_apply` per SkillRL Table 5.
- Idempotent evolution with `evolved` flag, dedup against existing bullets.
- Extended data in `.ccr/failure_lessons.json`.

**NOT implemented:**
- Teacher model `M_T` analyzing full trajectory to identify failure points (manual tagging).
- GRPO RL training (N/A -- no trainable model in MCP architecture).
- Cross-lesson synthesis via teacher model (mechanical path copies verbatim; optional sub-model synthesis available).
- Cold-start SFT.
- Full (state, action, reward) trajectory replay buffer.

---

## Tier 3: Inspired By (<30%)

### MAGMA -- Multi-Graph Memory Architecture

**Paper:** arXiv:2601.03236
**Fidelity:** ~25%

**Implemented:** Four link types (entity, causal, supersession, semantic) inspired by MAGMA's multi-graph edge taxonomy. Query-weighted BFS traversal when `query=` provided.

**NOT implemented:** LLM-inferred logical entailment for causal links (uses regex). Dense vector cosine for semantic links (uses word Jaccard). Adaptive beam search (MAGMA Alg. 1, Table 3 -- largest ablation impact). Fast/slow async consolidation paths. Abstract entity nodes via LLM.

---

### A-MEM -- Adaptive Memory Evolution

**Paper:** arXiv:2502.12110
**Fidelity:** ~25%

**Implemented:** Bidirectional commit cross-references. `EvolvedSummary` overlay for LLM-rewritten commit summaries (opt-in, requires sub-model). `_trigger_memory_evolution()` fires on semantic/supersession links.

**NOT implemented:** Dense vector cosine similarity for link generation (uses word Jaccard + regex). LLM analysis for link inference. Paper's full memory evolution algorithm (Eq. 4-7).

---

### ExpRAG -- Experience Retrieval

**Paper:** arXiv:2603.18272
**Fidelity:** ~25%

**Implemented:** 3-phase `_search_commits()`: exact substring match -> ONNX cosine similarity (threshold 0.3) -> BM25 fallback. Reuses existing commit embedding cache.

**NOT implemented:** Paper's full experience-augmented retrieval pipeline. Zero new storage or architecture -- just reuses A-RAG infrastructure.

---

### ERL -- Event-Driven Rule Learning

**Paper:** arXiv:2603.24639
**Fidelity:** ~20%

**Implemented:** Bullet `trigger` and `action` fields for structured "When X, do Y" format. `get_policy_ranked()` weights trigger match 1.5x higher than content match.

**NOT implemented:** Paper's event-driven rule learning loop. Automatic trigger extraction from execution traces. Rule confidence scoring from execution outcomes.

---

### Memori -- Semantic Triple Memory

**Paper:** arXiv:2603.19935
**Fidelity:** ~20%

**Implemented:** 13 regex patterns extract (subject, predicate, object) triples from commit text. `TripleStore` with word Jaccard search. `gcc_triples` tool. Triples appear in `gcc_context` level 3+.

**NOT implemented:** Paper's LLM-based triple extraction (uses regex). Triple reasoning and inference chains. Relationship-aware retrieval.

---

### EverMemOS -- Memory Operating System

**Paper:** arXiv:2601.02163
**Fidelity:** ~15%

**Implemented:** BFS connected components over entity + semantic cross-links for thematic clustering. Keyword extraction for cluster naming. `gcc_clusters` tool.

**NOT implemented:** MemScene hierarchical organization. Memory consolidation policies. Temporal graph maintenance. The paper's full operating system abstraction.

---

### EvolveR -- Evolving Rule Quality

**Paper:** arXiv:2510.16079
**Fidelity:** ~15%

**Implemented:** Bayesian quality scoring on `PatternEntry`: `quality_score = (success+1) / (success+failure+2)`. Quality propagated from promoted bullets via `ace_update_counters`. Patterns sorted by quality DESC.

**NOT implemented:** Paper's full evolving rule system. Rule competition and selection. Fitness-proportionate sampling.

---

### AgeMem -- Unified Long/Short-Term Memory

**Paper:** arXiv:2601.01885
**Fidelity:** ~15%

**Implemented:** Ephemeral key-value scratchpad (`.ccr/scratchpad.json`). Three operations: set, get, clear. Appears in `gcc_context` at level 2+. Thread-safe atomic writes.

**NOT implemented:** Paper's unified LTM/STM architecture. Memory consolidation from STM to LTM. Attention-based memory retrieval.

---

### AgentEvolver -- Contribution-Weighted Rewards

**Paper:** arXiv:2511.10395
**Fidelity:** ~10%

**Implemented:** Optional `weight` parameter (0.0-1.0) on `ace_update_counters` for proportional credit/blame. `effective_score()` uses weighted values when present, falls back to integer counts.

**NOT implemented:** Paper's full agent evolution framework. Population-based training. Fitness evaluation.

---

### ALMA -- Meta-Learned Memory Retrieval

**Paper:** arXiv:2602.07755
**Fidelity:** ~10%

**Implemented:** `PlaybookSchema` includes `link_scan_window`, `link_semantic_threshold`, `context_level_default`, `search_result_limit`. Usage metrics in `.ccr/memory_metrics.json`. `ace_evolve_schema` proposes ADJUST_SEARCH_THRESHOLD and ADJUST_SCAN_WINDOW based on metrics.

**NOT implemented:** Paper's meta-learning framework for retrieval parameters. Gradient-based parameter optimization. Multi-task evaluation.
