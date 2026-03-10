# CCR v2: A Survey and Roadmap for Persistent Agent Memory, Self-Evolving Strategies, and Sandboxed Execution in LLM-Based Coding Agents

**CCR Research Team** | March 2026

---

## Abstract

Large language model (LLM) based coding agents face three fundamental challenges: (1) maintaining coherent memory across sessions, (2) evolving task strategies from experience, and (3) safely executing code for iterative reasoning. We present a comprehensive survey of 50+ recent papers (2025-2026), frameworks, and production systems across these three domains, conducted to inform the next version of CCR (Claude Context Reducer)---an MCP server that unifies Git-style versioned memory (GCC), self-evolving playbooks (ACE), and sandboxed REPL execution (RLM) for Claude Code. Our analysis identifies 14 high-impact improvements organized into three priority tiers, backed by evidence from ICLR 2026, ACL 2025, NeurIPS 2025, and industry systems. We find that (a) graph-based memory with admission control outperforms flat stores by up to 45.5% on reasoning benchmarks, (b) hierarchical skill libraries with structured failure distillation achieve 10-20x compression over raw trajectories, (c) kernel-level sandboxing (Landlock/Seatbelt) provides zero-overhead isolation superior to application-level blocklists, and (d) CCR's unified architecture remains unique in the ecosystem---no competing system combines versioned memory, self-evolving strategies, and sandboxed execution in a single zero-LLM-call MCP server. We propose a concrete roadmap for CCR v2 that integrates these findings while preserving the system's zero-infrastructure, zero-API-key design philosophy.

---

## 1. Introduction

LLM-based coding agents---systems such as Claude Code [31], Cursor, Devin [36], and OpenHands [34]---have rapidly advanced from single-turn code completion to autonomous multi-step software engineering. However, three persistent limitations constrain their effectiveness in long-horizon, multi-session workflows:

**Memory amnesia.** Most coding agents begin each session with no knowledge of prior work. Claude Code's native Auto Memory (MEMORY.md) and Cursor's community-built Memory Banks offer flat markdown persistence, but lack version control, branching for experimental isolation, or structured retrieval beyond keyword search.

**Strategy stasis.** Agents repeatedly encounter the same failure modes across sessions without learning from them. While reinforcement learning from human feedback (RLHF) adapts model weights, it cannot capture project-specific heuristics, coding conventions, or debugging insights that vary across codebases.

**Unsafe execution.** Code execution for iterative reasoning---validated by the RLM paradigm [3]---requires sandboxing that balances security with zero-overhead performance. Current approaches range from heavyweight microVMs (E2B, Firecracker) to fragile application-level blocklists.

CCR (Claude Context Reducer) addresses all three challenges through a unified MCP server exposing 17 tools built on three research papers: GCC (Git Context Controller) [1] for versioned memory, ACE (Agentic Context Engineering) [2] for self-evolving playbooks, and RLM (Recursive Language Models) [3] for sandboxed REPL execution. CCR requires zero LLM calls, zero infrastructure, and zero API keys---operating entirely through Claude Code's native tool-calling interface on a Claude Max subscription.

This report surveys the state of the art across all three domains as of March 2026, identifies gaps in CCR's current implementation relative to recent advances, and proposes a prioritized roadmap for CCR v2. Our contributions are:

1. A structured survey of 50+ papers, frameworks, and production systems across agent memory, context engineering, and sandboxed execution (Sections 2-4).
2. A competitive analysis of the agent memory ecosystem including Claude Code native features, IDE agents, and the MCP marketplace (Section 5).
3. A three-tier improvement roadmap with impact/effort ratings backed by empirical evidence (Section 6).

---

## 2. Related Work: Agent Memory Systems

The study of persistent memory for LLM agents has matured from scattered proposals to a recognized research area, with a dedicated ICLR 2026 workshop (MemAgents) [20] and comprehensive surveys [18, 19].

### 2.1 Taxonomies and Theoretical Foundations

The most comprehensive taxonomy is proposed by [18], who organize agent memory along three dimensions: **Forms** (token-level, parametric, latent), **Functions** (factual, experiential, working), and **Dynamics** (formation, evolution, retrieval). Experiential memory is further divided into case-based (specific episodes), strategy-based (general approaches), and skill-based (reusable procedures). CCR's GCC commits correspond to case-based memory, while ACE playbook bullets correspond to strategy-based memory. The survey identifies skill-based memory as underexplored---CCR's CODE SNIPPETS section partially addresses this but lacks structured preconditions, expected outcomes, and failure modes.

The Continuum Memory Architecture (CMA) [16] formalizes five architectural requirements for cross-session state: **persistent storage**, **selective retention**, **associative routing**, **temporal chaining**, and **consolidation into higher-order abstractions**. Evaluating CCR against these requirements reveals two gaps: no admission control (selective retention) and limited graph-based retrieval (associative routing).

### 2.2 Graph-Based Memory Architectures

**MAGMA** [7] decouples memory into four orthogonal graphs---semantic, temporal, causal, and entity---with policy-guided traversal at retrieval time. A dual-stream write mechanism separates fast ingestion from asynchronous consolidation. Results show up to 45.5% higher reasoning accuracy, 95% token reduction, and 40% faster query latency versus flat retrieval. This directly informs a potential upgrade to GCC's commit structure: maintaining parallel graph indexes over commits would enable queries like "what caused this bug?" (causal graph) or "what have we done to this file?" (entity graph).

**SYNAPSE** [8] applies spreading activation from the ACT-R cognitive architecture [42] to agent memory. Relevance emerges dynamically through activation propagation rather than pre-computed static links, with lateral inhibition suppressing weakly related memories and temporal decay aging out stale information. This solves the "Contextual Tunneling" problem where vector similarity retrieval fixates on surface-level matches, missing deeper causal connections.

**A-MEM** [9] applies the Zettelkasten method to agent memory, where each note carries structured attributes (keywords, tags, contextual descriptions) and explicit bidirectional links. Critically, new experiences *retroactively refine* existing notes' attributes, making memory a living graph rather than an append-only log. GCC commits are currently write-once; A-MEM suggests that new commits should trigger metadata updates to related earlier commits.

**Graphiti/Zep** [10] introduces a bi-temporal data model tracking both event occurrence time and ingestion time, enabling point-in-time queries ("what did the agent know at time T?"). With P95 latency of 300ms and zero LLM calls at retrieval time, this approach aligns with CCR's zero-LLM-call philosophy.

**PlugMem** [11] (Microsoft Research, ICML 2026 under review) structures episodic memories into a compact knowledge-centric graph where propositional and prescriptive knowledge are the units of access. Information-theoretic analysis confirms highest information density across task types. This suggests extracting atomic knowledge from GCC commits into a separate lightweight index.

### 2.3 Memory Consolidation and Forgetting

**A-MAC** [12] treats memory admission as a structured decision problem with five scoring factors: future utility, factual confidence, semantic novelty, temporal recency, and content type prior. Ablation reveals content type prior as the single most influential factor. F1 of 0.583 on LoCoMo with 31% latency reduction. This directly motivates adding a novelty gate to `gcc_commit`.

**TiMem** [13] organizes memory in a Temporal Memory Tree (TMT) with 5 hierarchical levels, from raw conversation to progressively abstracted representations. Semantic-guided consolidation promotes important details upward. Achieves **52.2% reduction in recalled memory length** while maintaining 75.3% accuracy on LoCoMo. This informs an upgrade to GCC's single rolling summary.

**HiMem** [14] introduces *conflict-aware memory reconsolidation*: when retrieval surfaces old memories that contradict current state, the system detects and revises stored knowledge. This addresses a real problem in long-running projects where stale information misleads future sessions.

Temporal decay mechanisms inspired by cognitive architectures [43] provide principled forgetting without deletion. Under exponential decay (score_eff = score_raw * 0.95^days), a strategy unused for 30 days retains ~21% of its original weight, naturally deprioritizing stale entries.

### 2.4 Benchmarks

The agent memory evaluation landscape has consolidated around several benchmarks:

| Benchmark | Venue | Focus |
|-----------|-------|-------|
| **MemoryAgentBench** [21] | ICLR 2026 | Retrieval, test-time learning, long-range understanding, selective forgetting |
| **MemBench** [22] | ACL 2025 Findings | Effectiveness, efficiency, capacity across factual and reflective memory |
| **LoCoMo** [23] | Snap Research | Long-term conversational memory (de facto standard) |
| **AMA-Bench** [24] | arXiv 2026 | Long-horizon agentic memory |

---

## 3. Context Engineering and Self-Evolving Strategies

### 3.1 Beyond ACE: Meta Context Engineering

The most significant advance beyond ACE is **MCE** (Meta Context Engineering via Agentic Skill Evolution) [4]. While ACE evolves playbook *content* (add, update, merge, remove bullets within a fixed schema), MCE evolves the context engineering *process itself*---the instructions and code that govern how context is constructed.

MCE operates at two levels: a **meta-level agent** that maintains a library of CE "skills" (executable instructions + code) and performs *agentic crossover*---deliberative search over the history of skills, their executions, and evaluations---and a **base-level agent** that executes evolved skills to construct context as programmatic artifacts. Results show **5.6-53.8% relative improvement** over state-of-the-art agentic CE methods (mean 16.9%), with superior transferability across task families.

The implication for CCR is profound: the playbook format (sections with slugged bullets and counters) could itself be treated as an evolvable artifact, with a meta-level process periodically evaluating whether the current structure serves well.

### 3.2 Hierarchical Skill Libraries

**SkillRL** [5] introduces a hierarchical SkillBank with three key mechanisms:

1. **Distillation**: Successful trajectories are compressed into strategic patterns (10-20x compression). Failed trajectories are distilled into structured *failure lessons* with four components: failure point, flawed reasoning, correct counterfactual, and general prevention principle.
2. **Hierarchy**: General skills (universal strategies) versus task-specific skills (procedural guides with nuanced preconditions).
3. **Recursive evolution**: Validation checkpoints detect when accumulated failures expose knowledge gaps, triggering targeted skill refinement.

This directly informs CCR's ACE system. Currently, a harmful increment (`harmful=3`) conveys no information about *why* a strategy failed or *what should have been done instead*. SkillRL's structured failure lessons would make harmful entries actionable.

**EXIF** [6] introduces *exploration-first* skill discovery using dual agents: an explorer that proactively generates skill datasets by interacting with environments, and a learner that trains on discovered skills. When both roles are filled by the same model, the system becomes self-evolving. This contrasts with CCR's current reactive approach where insights are accumulated only when they arise during normal work.

### 3.3 Context Compression

**ACON** [25] (ICLR 2025) learns compression guidelines from paired trajectory analysis---comparing where full context succeeds versus compressed context fails. An LLM analyzes failure causes and iteratively updates compression guidelines in natural language. Results: **26-54% peak token reduction**, with distilled compressors retaining 95% accuracy.

Anthropic's own context engineering principles [31] emphasize that "every token competes for attention" and recommend the smallest possible set of high-signal tokens. The multi-agent summarization pattern---subagents explore extensively but return condensed 1-2K token summaries---aligns with CCR's RLM design.

**CER** (Contextual Experience Replay) [26] (ACL 2025) demonstrates training-free self-improvement through a dynamic memory buffer of synthesized experiences, achieving **51% relative improvement** over GPT-4o. CER's training-free approach validates CCR's zero-LLM-call philosophy while suggesting that `gcc_commit` could include a `patterns_learned` field capturing transferable decision patterns.

### 3.4 Retrieval-Augmented Generation for Code

**A-RAG** [27] presents hierarchical retrieval with three tools (keyword search, semantic search, chunk read) where models dynamically choose retrieval strategies. This outperforms fixed pipelines because different queries benefit from different strategies. CCR's `index_search` currently supports only keyword/symbol search; adding semantic search would enable conceptually related code discovery.

---

## 4. Sandboxed Execution and REPL-Based Reasoning

### 4.1 The RLM Paradigm and Its Validation

The RLM (Recursive Language Models) paradigm [3] treats long prompts as an external environment, giving the LLM a Python REPL with context loaded as a string variable. Key results include handling inputs up to 2 orders of magnitude beyond context windows, 29% gain on BrowseComp-Plus with GPT-5, and 58% F1 on OOLONG-Pairs.

Critically, the *REPL-only ablation* (no recursive sub-model calls) still improves performance on several tasks, validating CCR's architecture where Claude Code---the model itself---drives the iteration loop. Post-paper advances include **RLM-Qwen3-8B** [33] (the first natively recursive model, +28.3% over vanilla Qwen3-8B) and Prime Intellect's **RLMEnv** [34] (confirming consistent gains across four benchmarks).

### 4.2 The Sandbox Security Hierarchy

The sandboxing landscape has stratified into a clear hierarchy:

| Tier | Technology | Boot Time | Overhead | Representative |
|------|-----------|-----------|----------|---------------|
| 1 | Firecracker microVM | 125ms | 3-5 MB | E2B, AWS Lambda |
| 2 | Kata Containers | 200ms | ~10 MB | Northflank |
| 3 | gVisor | 50ms | ~15 MB | Modal, GKE |
| 4 | Wasm (Wasmtime) | <13ms | <5 MB | Wassette [38] |
| 5 | Kernel filters | <1ms | Zero | nono [39] |
| **6** | **Process blocklists** | **<1ms** | **Zero** | **CCR (current)** |

**CCR currently operates at the weakest tier (Tier 6).**

**Nono** [39] uses Landlock (Linux) and Seatbelt (macOS) for kernel-level restriction with deny-by-default semantics. It auto-blocks SSH keys, cloud credentials, and shell history, provides atomic rollback with cryptographic audit chains (Sigstore attestation, in-toto/SLSA compliance), and critically, "once restrictions are applied, there is no API to escape them---not even for nono itself." This is directly applicable to CCR on macOS with zero performance overhead.

**Wassette** [38] (Microsoft) runs WebAssembly Components via MCP with a deny-by-default capability system. Sub-13ms startup. The capability model is formally similar to browser sandboxes, providing strong isolation without virtualization overhead.

**CaMeL** [37] (Google DeepMind) addresses prompt injection via data flow control. A Privileged LLM orchestrates tasks while a Quarantined LLM (stripped of tool-calling) processes untrusted data. Every value carries provenance tags. CaMeL solves 77% of tasks with provable security versus 84% undefended (only 7% capability cost).

The **OWASP Top 10 for Agentic Applications** (2026) [40] identifies Agent Goal Hijacking (ASI01) and Tool Misuse (ASI02) as the top risks, emphasizing the Least Agency principle: autonomy should be earned, not default.

### 4.3 MCP Server Patterns

The **Cloudflare Code Mode** [41] pattern is particularly relevant: instead of exposing every API operation as a separate MCP tool (consuming >1M tokens for large APIs), expose just `search()` and `execute()`. The agent discovers capabilities via search and writes code against a typed SDK. Result: **81% fewer tokens** versus direct tool calling. This aligns with CCR's REPL philosophy.

Anthropic's **Programmatic Tool Calling** [32] achieves **98.7% token reduction** (150K to 2K tokens) by shifting from sequential tool calling to code-based orchestration. Their **Tool Search Tool** enables searching thousands of tools without loading definitions upfront (85% token reduction).

---

## 5. Competitive Landscape Analysis

We analyze the competitive positioning of CCR against six categories of systems:

| Capability | Claude Native | IDE Agents | Devin/OH | LangGraph | MCP Servers | **CCR** |
|-----------|--------------|------------|----------|-----------|-------------|---------|
| Cross-session memory | Flat markdown | Via MCP | Event logs | Document store | Various | **Git-style commits** |
| Version control | No | No | No | No | No | **Yes (branch/merge)** |
| Self-evolving strategies | No | No | No | No | No | **Yes (ACE playbooks)** |
| Sandboxed REPL | No | No | Sandbox env | No | No | **Yes (RLM)** |
| Zero LLM calls | Yes | N/A | No | No | Mostly no | **Yes** |
| Zero infrastructure | Yes | Varies | No | No (DB) | Varies | **Yes (file-based)** |

**Key findings:**

- **Claude Code Native Memory** (MEMORY.md): Zero setup, auditable, but no version history, no branching, no self-evolving strategies, 200-line limit.
- **Cursor/Windsurf/Cline/Roo Code**: Windsurf has best native memory among IDEs. Roo Code Memory Bank has good structure (decisionLog, systemPatterns). None offer versioned memory.
- **Devin/OpenHands**: OpenHands' context condensation achieves ~50% cost reduction. Devin Wiki auto-indexes repos. Neither has self-evolving strategies.
- **LangGraph/CrewAI/AutoGen**: LangGraph offers namespaced stores with TTL and multiple backends. All require infrastructure.
- **MCP Servers** (8,500+ in marketplace): claude-mem, memsearch, engram, mcp-memory-service. None combine versioned memory + self-evolution + REPL.
- **Convergence signal**: Letta/MemGPT independently introduced "Context Repositories" (git-based versioning) in February 2026, validating GCC's design.

**CCR's unique moats:** (1) Only system implementing all three papers (GCC + ACE + RLM) as a unified MCP server; (2) Zero LLM calls; (3) Git-style memory with branch/merge; (4) Self-evolving playbooks (zero competitors); (5) 17 tools in one server vs. competitors' 3-5; (6) Pure file-based, zero infrastructure.

---

## 6. CCR v2 Roadmap

Based on the survey, we propose 14 improvements organized into three priority tiers by impact-to-effort ratio. Before diving into each tier, here is the full priority matrix showing how every item ranks across the dimensions that matter most:

### Priority Matrix

| # | Item | Subsystem | Impact | Effort | Priority Score | Tier |
|---|------|-----------|--------|--------|---------------|------|
| 1 | Structured Failure Lessons | ACE | Very High | Low | **10** | 1 |
| 6 | MCP Tool Annotations | MCP | High | Very Low | **9** | 1 |
| 3 | Temporal Decay | ACE | High | Low | **9** | 1 |
| 4 | Two-Tier Playbook | ACE | High | Low | **8** | 1 |
| 2 | Memory Admission Control | GCC | High | Medium | **8** | 1 |
| 5 | Kernel Sandboxing | RLM | High | Medium | **7** | 1 |
| 10 | Patterns Field in Commits | GCC | Medium | Low | **7** | 2 |
| 7 | Hierarchical Summaries | GCC | Very High | Medium | **7** | 2 |
| 8 | Retroactive Commit Linking | GCC | High | Medium-High | **6** | 2 |
| 9 | Evolving Compression Guidelines | GCC/ACE | Medium | Medium | **6** | 2 |
| 11 | Meta-Level Playbook Evolution | ACE | Very High | High | **5** | 3 |
| 14 | Knowledge Atom Extraction | GCC | High | High | **5** | 3 |
| 13 | Semantic Search in Index | Index | Medium | High | **4** | 3 |
| 12 | Wasm Sandbox Migration | RLM | Medium | Very High | **3** | 3 |

> **How to read the score**: Priority Score = Impact / Effort on a 1-10 scale. Items with identical scores are ordered by implementation dependency (do prerequisites first).

---

### 6.1 Tier 1: The Essentials --- Build These First

These six items deliver the highest return for the least work. None of them require architectural changes to CCR's core. Most are additions to existing data structures or configuration. If you implement nothing else from this report, implement these.

**Why this tier matters:** Tier 1 fixes CCR's most visible gaps---the ones users hit in every session. A playbook that says `harmful=5` without explaining *why* is useless. A memory that stores 50 near-identical commits wastes tokens. A sandbox that only uses Python-level blocklists is one `subprocess.call` away from disaster. These items close those gaps with surgical changes.

---

**1. Structured Failure Lessons (ACE)** | Impact: Very High | Effort: Low

*The problem:* When a playbook strategy fails, CCR increments a counter (`harmful=3`). That counter tells future sessions "this strategy is bad" but not *why* it's bad, *when* it fails, or *what to do instead*. It's like a code review that says "this is wrong" without explaining the fix.

*The fix:* Replace bare counters with structured failure records containing four fields:
- **Failure point**: Where exactly the strategy broke down
- **Flawed reasoning**: What incorrect assumption led to applying it
- **Correct counterfactual**: What should have been done instead
- **Prevention principle**: A generalizable rule to avoid this class of failure

*Why it matters:* This is the single highest-impact item on the roadmap. A harmful counter is a signal; a structured failure lesson is knowledge. The information density per entry increases dramatically, and every harmful entry becomes immediately actionable for future sessions.

*Source:* SkillRL [5] --- their failure distillation achieves 10-20x compression over raw trajectories while preserving all actionable information.

---

**2. Memory Admission Control (GCC)** | Impact: High | Effort: Medium

*The problem:* Every `gcc_commit` call creates a new commit, regardless of whether it adds meaningful new information. In practice, this produces runs of 10-20 commits that all say variations of "continued working on X" with overlapping file lists and similar observations. These redundant commits waste tokens at retrieval time and dilute the signal-to-noise ratio.

*The fix:* Before writing a new commit, compute semantic overlap with the last *k* commits using file intersection and keyword similarity. If overlap exceeds a threshold (e.g., 0.8), auto-merge the new information into the previous commit instead of creating a new one. This is admission control---not every observation deserves its own commit.

*Why it matters:* Reduces the commit history to meaningful state transitions. Retrieval becomes faster and more informative. Token budget for `gcc_context` goes further.

*Source:* A-MAC [12] --- structured admission scoring with five factors (future utility, factual confidence, semantic novelty, temporal recency, content type).

---

**3. Temporal Decay for Playbook Counters (ACE)** | Impact: High | Effort: Low

*The problem:* A strategy that was helpful 6 months ago but hasn't been used since carries the same weight as one that was helpful yesterday. Old strategies accumulate like geological layers, and the playbook never naturally forgets.

*The fix:* Apply exponential decay to helpful/harmful counters:

```
score_eff = score_raw * 0.95^d
```

where *d* is days since last update. A strategy unused for 30 days retains ~21% of its score. One unused for 90 days retains ~1%. The strategy isn't deleted---it just naturally fades unless reinforced by continued use.

*Why it matters:* Implements principled forgetting inspired by cognitive science (ACT-R memory decay). The playbook becomes self-cleaning: strategies that remain relevant get reinforced, while stale ones fade to near-zero without manual pruning.

*Source:* SYNAPSE [8] (spreading activation with temporal decay), ACT-R [42] (cognitive architecture memory model).

---

**4. Two-Tier Playbook (ACE)** | Impact: High | Effort: Low

*The problem:* Strategies learned in one project don't transfer to another. If you learn "always check for off-by-one errors in loop bounds" while working on Project A, that insight vanishes when you switch to Project B. Meanwhile, project-specific knowledge ("ChatMessage is a struct, not an enum in ironclaw") shouldn't pollute other projects.

*The fix:* Split the playbook into two layers:
- **Global** (`~/.ccr/global_playbook.txt`): Universal heuristics that transfer across all projects (e.g., "validate assumptions at each step", "read the file before editing it")
- **Project** (`.ccr/playbook.txt`): Project-specific strategies that only apply to the current codebase

Both are injected via the session start hook. `ace_apply_delta` gains an optional `scope` parameter (global/project). Promotion from project to global happens when a strategy proves helpful across multiple projects.

*Why it matters:* Cross-project transfer learning. The agent gets smarter with every project it works on, not just within each one.

*Source:* SkillRL [5] (general vs. task-specific skill hierarchy), Memory Survey [18] (experiential memory categories).

---

**5. Kernel Sandboxing (RLM)** | Impact: High | Effort: Medium

*The problem:* CCR's REPL currently uses Python-level blocklists to restrict dangerous operations. This is Tier 6 (weakest) on the sandbox security hierarchy. Python-level restrictions can be bypassed through `subprocess`, `ctypes`, dynamic imports, or other escape hatches. On a developer machine with SSH keys, cloud credentials, and API tokens, this is a real risk.

*The fix:* Wrap the REPL subprocess in macOS Seatbelt (or Linux Landlock) enforcement with deny-by-default semantics. The kernel itself blocks filesystem access outside the project directory, blocks all network access, and auto-protects `~/.ssh`, `~/.aws`, `~/.config/gcloud`, and shell history. Once the kernel applies restrictions, there is no API to escape them---not even for the sandboxing tool itself.

*Why it matters:* Moves CCR from Tier 6 to Tier 5 on the sandbox hierarchy with zero performance overhead. This is the difference between "we asked Python nicely not to do bad things" and "the operating system kernel enforces it."

*Source:* nono [39] (kernel-level agent sandboxing), OWASP Top 10 for Agentic Applications [40] (Least Agency principle).

---

**6. MCP Tool Annotations** | Impact: High | Effort: Very Low

*The problem:* Claude Code doesn't know which CCR tools are read-only, which are destructive, and which are safe to retry (idempotent). This means it can't make smart decisions about execution ordering or confirmation requirements.

*The fix:* Add three metadata fields to all 17 tool definitions: `readOnlyHint` (true for `gcc_context`, `gcc_status`, `ace_get_playbook`, etc.), `destructiveHint` (true for `gcc_branch` delete operations), and `idempotentHint` (true for `gcc_context`, `index_search`, etc.). This is pure metadata---zero changes to tool logic.

*Why it matters:* Lowest effort item on the entire roadmap. Takes minutes to implement, immediately improves Claude Code's ability to reason about tool safety. The MCP spec (2025-11-25) explicitly recommends these annotations.

*Source:* MCP Specification [45].

---

### 6.2 Tier 2: Significant Upgrades --- Build These Next

These four items require moderate refactoring but deliver meaningful capability improvements. They move CCR from "persistent memory" to "intelligent memory"---memory that understands relationships, learns compression, and captures patterns.

**Why this tier matters:** Tier 1 fixes what's broken. Tier 2 builds what's missing. After Tier 1, CCR has clean data (admission control), actionable failures (structured lessons), and real security (kernel sandbox). Tier 2 makes that data *smarter*---summaries that compress intelligently, commits that link to each other, and patterns that flow from memory to playbook automatically.

---

**7. Multi-Level Hierarchical Summaries (GCC)** | Impact: Very High | Effort: Medium

*The problem:* GCC maintains a single rolling summary that tries to capture everything. As a project grows, this summary either becomes too long (wasting tokens) or too compressed (losing important details). There's no middle ground between "give me the full history" and "give me the one-paragraph overview."

*The fix:* Upgrade to three summary tiers:
- **Session summaries** (generated every 5-10 commits): What was accomplished in this work session
- **Phase summaries** (generated on branch merges or milestones): What was accomplished in this development phase
- **Project overview** (updated monthly or on major milestones): High-level project status

`gcc_context(level=1)` returns the project overview + recent phase summaries. Higher levels add session summaries and individual commits progressively. The token budget naturally scales with the detail level requested.

*Why it matters:* The research (TiMem) shows this approach can reduce retrieved memory length by roughly half while maintaining accuracy. More importantly, it gives Claude the right level of detail for each situation---broad context for new tasks, deep history for debugging.

*Source:* TiMem [13] (Temporal Memory Tree with 5 hierarchical levels and semantic-guided consolidation).

---

**8. Retroactive Commit Linking (GCC)** | Impact: High | Effort: Medium-High

*The problem:* GCC commits are isolated entries in a flat log. When you need to answer "what caused this bug?" or "what have we done to this module?", the only option is full-text search through commit messages. There's no structural way to traverse relationships between commits.

*The fix:* When `gcc_commit` creates a new entry, scan recent commits for semantic overlap (shared files, referenced concepts) and store cross-links as metadata. Three link types:
- **Entity links**: Same files touched (e.g., commit C005 and C012 both modified `mcp_server.py`)
- **Causal links**: The "why" field references a prior problem (e.g., "fixing the bug discovered in C008")
- **Supersession links**: Replaces an earlier approach (e.g., "replaced the approach from C003")

`gcc_context` can then traverse these links for targeted retrieval rather than scanning everything.

*Why it matters:* Transforms flat memory into a graph. Instead of "show me the last 20 commits," you can ask "show me everything related to the MCP server refactoring" and get a connected subgraph of relevant commits.

*Source:* A-MEM [9] (bidirectional Zettelkasten links), MAGMA [7] (multi-graph with entity, causal, semantic, and temporal dimensions).

---

**9. Evolving Compression Guidelines (GCC/ACE)** | Impact: Medium | Effort: Medium

*The problem:* When Claude Code's context window compacts, some information is lost. Currently there's no feedback loop---if the compaction drops something important, it'll drop it again next time too. The system doesn't learn what matters.

*The fix:* Maintain `.ccr/compression_guidelines.txt` with rules about what information must survive compaction. When an agent can't find previously committed data after compaction (detectable when `gcc_context` is called and the agent asks about something it previously committed), update the guidelines. Over time, compaction learns what each project needs to preserve.

*Why it matters:* Turns compaction from a dumb truncation into a learned, project-specific process. A project with complex architecture needs to preserve structural decisions; a project with tricky bugs needs to preserve debugging context. The guidelines capture this.

*Source:* ACON [25] (learning compression guidelines from paired trajectory analysis).

---

**10. Patterns Field in Commits (GCC)** | Impact: Medium | Effort: Low

*The problem:* Transferable patterns discovered during work (e.g., "tokio async tests need a special Mutex setup", "this API returns 429 if you exceed 10 req/s") are buried in commit observations. There's no structured way to extract and reuse them.

*The fix:* Add an optional `patterns_learned` field to `gcc_commit`. This field captures decision patterns that might apply beyond the current task. Patterns that appear in multiple commits are candidates for automatic promotion to ACE playbook bullets.

*Why it matters:* Creates a pipeline from experience to strategy. Instead of manually calling `ace_apply_delta`, the system can detect recurring patterns and suggest (or automatically apply) playbook additions.

*Source:* CER [26] (extracting transferable decision patterns from experience replay).

---

### 6.3 Tier 3: Transformative Changes --- The Long Game

These four items represent fundamental capability upgrades that would position CCR at the frontier of agent memory research. They require significant engineering effort and, in some cases, new dependencies. Build these when Tiers 1 and 2 are stable and battle-tested.

**Why this tier matters:** Tier 1 and 2 make CCR excellent at what it already does. Tier 3 makes CCR do things no other system can. Meta-level evolution means the playbook format itself adapts. Knowledge atoms mean memory retrieval is surgical rather than bulk. These are the features that would make CCR a research contribution, not just a tool.

---

**11. Meta-Level Playbook Evolution (ACE)** | Impact: Very High | Effort: High

*The problem:* ACE evolves playbook *content* (adding, updating, merging, removing bullets), but the playbook *structure*---six fixed sections, slug-based bullets, helpful/harmful counters---never changes. What if the current format isn't optimal? What if some projects need different sections, or counters should weight differently?

*The fix:* Treat the playbook format itself as an evolvable artifact. A meta-level process periodically evaluates whether the current structure serves well (based on how often strategies are used, how accurate they are, whether sections are balanced) and proposes structural changes. This is MCE applied to ACE.

*Why it matters:* This is the deepest form of self-evolution---not just learning *what* to do, but learning *how to organize what you learn*. The research shows significant improvement when the context engineering process itself evolves rather than just the content.

*Source:* MCE [4] (Meta Context Engineering via Agentic Skill Evolution).

---

**12. Wasm Sandbox Migration (RLM)** | Impact: Medium | Effort: Very High

*The problem:* Even with kernel sandboxing (Tier 1, item 5), the REPL runs native Python. WebAssembly provides stronger isolation guarantees with a formally verified capability model, at the cost of significant architectural change.

*The fix:* Replace the native Python REPL with a WebAssembly runtime (Wasmtime). Code executes inside a Wasm sandbox with deny-by-default capabilities. Filesystem, network, and environment access are explicitly granted per-invocation. Sub-13ms startup, browser-engine-level isolation.

*Why it matters:* Future-proofs security. However, this is the lowest priority item because kernel sandboxing (item 5) already provides strong isolation with zero effort. Only pursue this if CCR needs to run untrusted third-party code or on platforms where Seatbelt/Landlock aren't available.

*Source:* Wassette [38] (Microsoft's WebAssembly-based MCP tool sandbox).

---

**13. Semantic Search in Index** | Impact: Medium | Effort: High

*The problem:* `index_search` currently finds code by keyword and symbol name. If you search for "authentication logic," you'll find functions named `authenticate` but miss a function called `verify_user_token` that does the same thing.

*The fix:* Add embedding-based retrieval alongside keyword/symbol search. Expose as `index_search(query, mode="keyword"|"semantic"|"hybrid")`. Semantic mode computes embeddings for the query and matches against pre-computed code embeddings.

*Why it matters:* Enables conceptual code discovery. However, this potentially requires either a local embedding model or an API key, which risks breaking CCR's zero-infrastructure principle. The mitigation is to use a lightweight local model (e.g., all-MiniLM-L6-v2 via ONNX runtime) or gracefully fall back to keyword search when embeddings aren't available.

*Source:* A-RAG [27] (hierarchical retrieval with dynamic strategy selection).

---

**14. Knowledge Atom Extraction (GCC)** | Impact: High | Effort: High

*The problem:* When `gcc_context` retrieves commits, it returns full commit text---observations, thoughts, actions, file lists. For broad queries ("what do I know about the test suite?"), this means loading many full commits when only a few atomic facts are needed ("600 tests pass", "5 skip", "run with pytest -x -q").

*The fix:* Extract atomic knowledge units from each commit into a separate lightweight index. Each atom is a single fact with a source link back to its parent commit. `gcc_context` returns atoms for broad queries (saving tokens) and full commits for deep dives.

*Why it matters:* Dramatically reduces token usage for overview queries. The knowledge atom index acts as a fast lookup layer over the full commit history.

*Source:* PlugMem [11] (knowledge-centric graph with propositional and prescriptive atoms).

---

### 6.4 Recommended Build Order

Within each tier, implementation order matters due to dependencies:

**Tier 1 (weeks 1-4):**
1. MCP Tool Annotations (#6) --- do this first, it takes minutes
2. Structured Failure Lessons (#1) --- foundational for all future ACE work
3. Temporal Decay (#3) --- builds on the counter system that #1 just improved
4. Two-Tier Playbook (#4) --- extends playbook structure, benefits from #1 and #3
5. Memory Admission Control (#2) --- requires analyzing commit overlap, independent of ACE changes
6. Kernel Sandboxing (#5) --- independent, can be done in parallel with any of the above

**Tier 2 (weeks 5-10):**
1. Patterns Field (#10) --- smallest change, immediately useful
2. Hierarchical Summaries (#7) --- builds the infrastructure for smarter retrieval
3. Retroactive Commit Linking (#8) --- benefits from #7's summary tiers as link targets
4. Evolving Compression Guidelines (#9) --- needs #7 and #8 data to learn from

**Tier 3 (ongoing):**
- Start with #11 (Meta-Level Evolution) once ACE changes from Tier 1 are battle-tested
- #14 (Knowledge Atoms) can begin once #8 (commit linking) provides the graph structure
- #13 (Semantic Search) and #12 (Wasm) are independent and can be explored opportunistically

---

## 7. Discussion

### 7.1 Architectural Validation

The survey provides strong external validation for CCR's core design choices:

1. **Git-style memory is converging as standard.** Letta/MemGPT independently introduced "Context Repositories" (git-based versioning) in February 2026, and GCC v2 reports state-of-the-art on SWE-Bench Verified.
2. **Zero-LLM-call architecture is viable.** CER [26] demonstrates training-free self-improvement, and Codified Context [28] validates hot/cold memory across 283 sessions without LLM calls for memory management.
3. **REPL-as-environment is validated.** The RLM REPL-only ablation [3] confirms gains even without recursive sub-model calls.
4. **Self-evolving strategies are unique.** No competing system implements self-evolving playbooks with helpful/harmful counters.

### 7.2 Key Risks

**Claude Code native memory evolution.** If Anthropic adds versioning, search, or strategy evolution to Auto Memory, CCR's advantage narrows. Mitigation: stay structurally superior.

**MCP marketplace fragmentation.** Users may prefer assembling focused tools. Mitigation: emphasize GCC + ACE + RLM integration value.

**Embedding dependency.** Semantic search (Tier 3) may require API keys, breaking zero-infrastructure. Mitigation: local models (all-MiniLM-L6-v2 via ONNX) or graceful fallback to keyword search.

### 7.3 Limitations

This analysis covers publicly available papers and documentation as of March 10, 2026. Commercial systems may have unpublished capabilities. Benchmark comparisons across papers use differing evaluation protocols. The proposed roadmap has not been empirically validated; impact estimates are extrapolated from source papers.

---

## 8. Conclusion

We surveyed 50+ papers, frameworks, and production systems across three domains to inform CCR v2. Our analysis reveals that:

1. **Graph-based memory with admission control** (MAGMA, A-MAC) outperforms flat stores by up to 45.5% on reasoning benchmarks, motivating commit linking and novelty gating for GCC.
2. **Hierarchical skill libraries with structured failure distillation** (SkillRL) achieve 10-20x compression while making failures actionable, directly informing ACE playbook evolution.
3. **Temporal decay** (SYNAPSE, ACT-R) and **multi-level summarization** (TiMem, 52% token reduction) provide principled mechanisms for managing growing memory.
4. **Kernel-level sandboxing** (nono/Seatbelt) provides zero-overhead isolation categorically stronger than CCR's current application-level blocklist.
5. **CCR's unified architecture remains unique**---no competing system combines versioned memory, self-evolving playbooks, and sandboxed REPL in a single zero-LLM-call MCP server.

We propose a 14-item roadmap: Tier 1 (6 items, highest ROI, no architectural changes), Tier 2 (4 items, moderate refactoring), Tier 3 (4 items, transformative). The convergence of independent research toward git-style memory (Letta), self-evolving strategies (MCE, SkillRL), and REPL-based reasoning (RLM-Qwen3) validates CCR's foundational bet that coding agents need structured, versioned, evolving memory---not just flat text files.

---

## References

[1] Git Context Controller (GCC). *arXiv:2508.00031*, 2025 (revised March 2026).

[2] Agentic Context Engineering (ACE). *arXiv:2510.04618*, 2025.

[3] A. Zhang, T. Kraska, O. Khattab. Recursive Language Models. *arXiv:2512.24601*, December 2025.

[4] H. Ye, X. He, et al. MCE: Meta Context Engineering via Agentic Skill Evolution. *arXiv:2601.21557*, January 2026.

[5] SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning. *arXiv:2602.08234*, February 2026.

[6] EXIF: Exploration-First Automated Skill Discovery for Language Agents. *arXiv:2506.04287*, June 2025.

[7] MAGMA: Multi-Graph Agentic Memory Architecture. *arXiv:2601.03236*, January 2026.

[8] SYNAPSE: Episodic-Semantic Memory via Spreading Activation. *arXiv:2601.02744*, January 2026.

[9] A-MEM: Agentic Memory for LLM Agents. *arXiv:2502.12110*, February 2025.

[10] Graphiti: Temporal Knowledge Graphs for AI Agents. *arXiv:2501.13956*, January 2025.

[11] PlugMem: Task-Agnostic Plugin Memory. *arXiv:2603.03296*, March 2026 (Microsoft Research).

[12] A-MAC: Adaptive Memory Admission Control. *arXiv:2603.04549*, March 2026.

[13] TiMem: Temporal-Hierarchical Memory Consolidation. *arXiv:2601.02845*, January 2026.

[14] HiMem: Hierarchical Long-Term Memory for LLM Agents. *arXiv:2601.06377*, January 2026.

[15] MemRL: Self-Evolving Agents via Reinforcement Learning on Episodic Memory. *arXiv:2601.03192*, January 2026.

[16] Continuum Memory Architecture. *arXiv:2601.09913*, January 2026.

[17] Aeon: Neuro-Symbolic Memory OS for LLM Agents. *arXiv:2601.15311*, January 2026.

[18] Memory in the Age of AI Agents: A Survey. *arXiv:2512.13564*, December 2025.

[19] Graph-based Agent Memory: Taxonomy, Techniques, and Applications. *arXiv:2602.05665*, February 2026.

[20] ICLR 2026 MemAgents Workshop: Memory for LLM-Based Agentic Systems. Vienna, May 2026.

[21] MemoryAgentBench. *arXiv:2507.05257*, ICLR 2026.

[22] MemBench: Evaluating Agent Memory. ACL 2025 Findings.

[23] LoCoMo: Long-term Conversational Memory Benchmark. Snap Research.

[24] AMA-Bench: Evaluating Long-Horizon Agentic Memory. *arXiv:2602.22769*, February 2026.

[25] ACON: Optimizing Context Compression for Long-Horizon Agents. *arXiv:2510.00615*, ICLR 2025.

[26] CER: Contextual Experience Replay for Self-Improvement. *arXiv:2506.06698*, ACL 2025.

[27] A-RAG: Scaling Agentic RAG via Hierarchical Retrieval Interfaces. *arXiv:2602.03442*, February 2026.

[28] Codified Context: Infrastructure for AI Agents in Complex Codebases. *arXiv:2602.20478*, February 2026.

[29] AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery. *arXiv:2506.13131*, DeepMind, May 2025.

[30] From Experience to Strategy: Trainable Graph Memory. *arXiv:2511.07800*, ICLR 2026 under review.

[31] Effective Context Engineering for AI Agents. *Anthropic Engineering Blog*, September 2025.

[32] Code Execution with MCP. *Anthropic Engineering Blog*, November 2025.

[33] RLM-Qwen3-8B: A Natively Recursive Language Model. 2026.

[34] Prime Intellect. RLM as the Paradigm of 2026. *Blog post*, 2026.

[35] OpenHands: An Open Platform for AI Software Developers. *arXiv:2407.16741*, 2024.

[36] Devin: AI Software Engineer. Cognition AI, 2024-2026.

[37] CaMeL: Defeating Prompt Injections by Design. *arXiv:2503.18813*, Google DeepMind, 2025.

[38] Wassette: WebAssembly-based Tools for AI Agents. Microsoft, August 2025.

[39] nono: Kernel-level Agent Sandboxing. GitHub/HuggingFace, 2026.

[40] OWASP Top 10 for Agentic Applications. 2026 Edition.

[41] Cloudflare. Code Mode MCP: Give Agents an Entire API in 1,000 Tokens. 2026.

[42] J. R. Anderson. How Can the Human Mind Occur in the Physical Universe? Oxford University Press, 2007.

[43] Human-Like Remembering and Forgetting for LLM Agents. ACM HAI 2025.

[44] AI Agentic Programming: A Survey. *arXiv:2508.11126*, 2025.

[45] MCP Specification, Version 2025-11-25. modelcontextprotocol.io.

[46] AgeMem: Unified LTM/STM Management. *arXiv:2601.01885*, January 2026.

[47] OpenHands Context Condensation. *OpenHands Blog*, 2025.

[48] Agentic RAG Survey. *arXiv:2501.09136*, January 2025.

[49] Repository-Level Code Generation Survey. *arXiv:2510.04905*, October 2025.

[50] Letta Memory Benchmark. letta.com/blog, 2026.
