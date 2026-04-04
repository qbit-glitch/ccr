# CCR — Claude Context Reducer

Persistent memory, self-evolving strategy playbooks, and a sandboxed Python REPL for Claude Code. No API keys required — works with a Claude Max subscription alone.

## What CCR Does

CCR is an MCP server that gives Claude Code three capabilities it doesn't have natively:

1. **Persistent Memory (GCC)** — Git-style version-controlled memory that survives across sessions. Branch, merge, and search your project's decision history.
2. **Self-Evolving Playbooks (ACE)** — Strategy bullets that track what works and what doesn't, with temporal decay and automatic pruning.
3. **Sandboxed REPL (RLM)** — An isolated Python environment for iterative analysis, with repo search and structured output.

All tools run as pure logic with zero LLM calls. Claude Code itself provides the reasoning.

## Quick Start

```bash
# 1. Install
pip install ccr-memory  # or: pip install -e . (from source)

# 2. Configure hooks for automatic memory management
ccr install

# 3. Use Claude Code normally — CCR auto-commits your session progress
```

### Manual Setup (without `ccr install`)

Add to your project's `.mcp.json`:

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

Then in your Claude Code session, call `gcc_context(level=2)` to load memory and `gcc_commit` after completing tasks.

## Features

### Persistent Memory (GCC)

- **Commits**: Save what you did, why, files changed, and what's next
- **Branches**: Isolate experiments with `gcc_branch`, merge when decided
- **Context levels**: 5 levels of detail retrieval (summary → full history)
- **Pattern buffer**: Transferable skills extracted from commits, with quality scoring
- **Cross-linking**: Automatic bidirectional links between related commits
- **Semantic search**: Find past work by meaning, not just keywords (ONNX embeddings)

### Self-Evolving Playbooks (ACE)

- **Strategy bullets**: "When X, do Y" rules with helpful/harmful counters
- **Temporal decay**: Unused strategies fade (30 days → 21% weight, 90 days → 1%)
- **Two-tier scope**: Global strategies (all projects) + project-specific strategies
- **Failure lessons**: Structured analysis of what went wrong and prevention principles
- **Schema evolution**: The playbook structure itself evolves based on usage metrics

### Sandboxed REPL (RLM)

- **Kernel-level isolation**: macOS Seatbelt sandbox (deny-default + allowlist)
- **Module allowlist**: Only safe standard library modules permitted
- **Repo tools**: `search_repo()`, `get_file()`, `estimate_tokens()` available in REPL
- **Structured output**: `FINAL_VAR` termination pattern for clean results

### Repo Indexing

- **Hybrid search**: Keyword + semantic + combined modes
- **Per-language parsing**: Symbol extraction for Python, TypeScript, Rust, Go, and more
- **ONNX embeddings**: Optional dense embeddings (all-MiniLM-L6-v2, 384-dim)
- **Zero-config**: Works immediately; semantic search available with `pip install ccr-memory[semantic]`

## Architecture

```
Claude Code ──stdio──> CCR MCP Server
                         ├── GCC Memory    (.ccr/commits, branches, patterns)
                         ├── ACE Playbook  (.ccr/playbook.txt, failure_lessons.json)
                         ├── RLM Sandbox   (isolated Python subprocess)
                         └── Repo Index    (.ccr/index.json, embeddings)
```

CCR stores all data in a `.ccr/` directory within your project (like `.git/`). Global strategies live in `~/.ccr/`.

## Tools

### Core (used in every session)

| Tool | Purpose |
|------|---------|
| `gcc_commit` | Save progress with what/why/files/next |
| `gcc_context` | Retrieve memory at 5 detail levels |
| `gcc_status` | Show current memory state |
| `ace_get_playbook` | View strategies with stats |
| `ace_update_counters` | Rate strategies helpful/harmful |
| `ace_apply_delta` | Add/update/merge/remove strategies |

### Extended

| Tool | Purpose |
|------|---------|
| `gcc_branch` / `gcc_merge` | Experiment isolation |
| `gcc_links` | Trace commit relationships |
| `gcc_patterns` | Query transferable patterns |
| `gcc_scratchpad` | Ephemeral working memory |
| `gcc_consolidate` | Generate hierarchical summaries |
| `ace_find_similar` | Find duplicate strategies |
| `ace_prune` | Remove harmful strategies |
| `rlm_init` / `rlm_execute` / `rlm_finalize` | Sandboxed REPL |
| `index_build` / `index_search` | Repo search |

## Research Foundation

CCR draws on 16 research papers across three tiers of implementation fidelity:

### Implemented (>70% fidelity)
- **GCC** (arXiv:2508.00031) — Git-style version-controlled agent memory
- **ACE** (arXiv:2510.04618) — Evolving playbooks with structured bullets and delta operations
- **RLM** (arXiv:2512.24601) — REPL-based execution with metadata-only stdout

### Substantially Adapted (30-70% fidelity)
- **A-MAC** (arXiv:2603.04549) — Admission control with 3 of 5 scoring factors
- **A-RAG** (arXiv:2602.03442) — Hierarchical retrieval with keyword/semantic/hybrid modes
- **CER** (arXiv:2506.06698) — Pattern buffer with dedup and quality scoring
- **MCE** (arXiv:2601.21557) — Schema evolution with rule-based structural proposals
- **SkillRL** (arXiv:2602.08234) — Failure-side skill distillation via structured lessons

### Inspired By (<30% fidelity)
- **A-MEM/MAGMA** — Commit cross-linking taxonomy
- **ERL** — Trigger/action bullet structure
- **Memori** — Semantic triple extraction
- **EverMemOS** — Thematic commit clustering
- **EvolveR** — Bayesian quality scoring for patterns
- **AgeMem** — Working memory scratchpad
- **AgentEvolver** — Contribution-weighted counters
- **ALMA** — Meta-learned retrieval parameters

All implementations use mechanical heuristics (zero LLM calls). See `CLAUDE.md` for detailed limitation tables comparing CCR's implementation vs. each paper.

## vs. Alternatives

| Feature | CCR | Mem0 | Letta/MemGPT | Graphiti |
|---------|-----|------|--------------|---------|
| Auto-manages memory | Yes (hooks) | Yes | Yes | Yes |
| Version control (branch/merge) | Yes | No | No | No |
| Self-evolving strategies | Yes | No | No | No |
| Sandboxed REPL | Yes | No | No | No |
| Zero LLM calls | Yes | No | No | No |
| Zero infrastructure | Yes | No | No (DB) | No (Neo4j) |
| Works with Claude Max only | Yes | No | No | No |
| Open source | MIT | Yes | Apache 2.0 | Apache 2.0 |

## Configuration

### Optional Dependencies

```bash
pip install ccr-memory[semantic]  # ONNX embeddings for semantic search
pip install ccr-memory[vector]    # sqlite-vec for persistent vector store
pip install ccr-memory[full]      # Both of the above
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `CCR_PROJECT_ROOT` | Override project root detection |
| `CCR_OLLAMA_MODEL` | Enable Ollama sub-model (e.g., `qwen2.5:7b`) |
| `ANTHROPIC_API_KEY_SUB` | Enable Anthropic Haiku sub-model |

Sub-models are optional — they enable LLM-powered features like rolling summary synthesis and automatic bullet generation.

### Diagnostics

```bash
ccr doctor   # Check CCR health (deps, config, hooks)
ccr status   # Show memory state
ccr context  # Print project context
```

## Development

```bash
git clone https://github.com/qbit-glitch/ccr.git
cd ccr
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/unit/ tests/integration/ -x -q
```

## License

MIT
