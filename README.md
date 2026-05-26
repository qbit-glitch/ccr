# CCR — Continuous Context Retention

> **Without CCR:** "Can you remind me what we decided about the dataset preprocessing last week?"  
> **With CCR:** Your AI agent already knows — months of decisions, experiments, and code reasoning recalled instantly.

CCR gives MCP-capable AI agents persistent project memory, strategy playbooks, and a sandboxed Python REPL. Full auto-context for **Claude Code**, **Kimi Code CLI**, and **Codex CLI** via `codex-ccr`; MCP tools for **Continue.dev**; SDK wrappers for **Ollama** and **OpenAI API**. **macOS/Linux only** (Windows support is not yet implemented).

> **New to CCR?** See the [Student & Researcher Quickstart](https://github.com/qbit-glitch/ccr/blob/main/docs/quickstart-students.md) — setup in 3 minutes, before/after examples, PhD workflow guide.

## Quick Start

> **Requirements**: macOS/Linux · Python 3.11+ · An AI agent (see cost table below)

```bash
# 0. Prerequisites (if not already installed)
# - Python 3.11+:  python3 --version
# - Claude Code:   npm install -g @anthropic-ai/claude-code  (requires paid Claude Pro or API key)
# - Kimi CLI:      pip install kimi-cli  (free tier available)
# - Ollama:        https://ollama.com  (free, runs locally)

# 1. Install CCR (free and open source)
pip install ccr-memory  # or: pip install -e . (from source)

# 2. Global setup — works across ALL projects automatically
ccr install-global

# 3. Open your agent from any project directory — CCR handles the rest
cd /your/project && claude   # or kimi, continue, ollama, etc.
cd /your/project && kimi     # Kimi Code CLI (free tier)
cd /your/project && codex-ccr # Codex with full CCR lifecycle
```

That's it. Your agent will automatically load project memory on every session start and auto-commit progress when you finish — in **every directory** without per-project setup. Memory is stored per-project in `./.ccr/` and auto-initialized on first use.

### What CCR Does

CCR is an MCP server that gives AI agents three capabilities they don't have natively:

1. **Persistent Memory (GCC)** — Git-style version-controlled memory that survives across sessions. Branch, merge, and search your project's decision history.
2. **Self-Evolving Playbooks (ACE)** — Strategy bullets that track what works and what doesn't, with temporal decay and automatic pruning.
3. **Sandboxed REPL (RLM)** — An isolated Python environment for iterative analysis, with repo search and structured output.

All core tools run with minimal overhead. The AI agent itself provides the reasoning; CCR provides the memory layer.

**Works across agents** — Claude Code, Kimi, and Codex share memory via hooks and wrappers; Continue.dev via MCP; Ollama and OpenAI via SDK wrappers. The same `.ccr/` directory is readable by all.

### For Researchers and Students

CCR is designed for long-running research projects where context loss is the main productivity bottleneck. A 3-month project means ~90 agent sessions. Without CCR, each starts from scratch. With CCR, each starts where the last left off.

**Researcher-specific features:**
- `gcc_commit(experiment={"metrics": {"val_loss": 0.23}})` — log ML runs with metrics and hypothesis
- `gcc_experiments(metric_filter={"val_loss": {"lt": 0.3}})` — find all runs meeting a metric threshold
- `gcc_discuss(topic=..., decision=..., rationale=...)` — persistent decision log for architecture choices
- `gcc_search("preprocessing decision")` — find any past decision across commits, discussions, and sessions

**CCR is free and open source.** The AI agents it connects to are not:

| Agent | Cost | Notes |
|-------|------|-------|
| Claude Code | $20/mo Pro or ~$2–8/mo API | Most capable; requires Claude Pro subscription or Anthropic API key |
| Kimi Code CLI | Free tier | No payment required for basic usage |
| Codex CLI | OpenAI account/API | Use `codex-ccr` for wrapper lifecycle plus native hooks and MCP tools |
| Continue | Free extension | But LLM backends (OpenAI/Anthropic) require paid API keys |
| Ollama | Free | Runs local models; needs RAM/GPU for larger models |
| OpenAI API | Pay-per-token | No subscription, but every API call costs money |

> **Global pricing note:** Agent subscriptions (e.g., Claude Pro at $20/mo) are US-priced. At PPP, this is $40–80/mo equivalent in many countries. The API-key path (~$2–8/mo actual usage) is the most accessible entry point for budget-constrained students.

See the [Student & Researcher Quickstart](https://github.com/qbit-glitch/ccr/blob/main/docs/quickstart-students.md) for setup, cost details, and a full PhD workflow guide.

### Global Setup (`ccr install-global`) — Recommended

Run once to enable CCR across **all** projects:

```bash
ccr install-global              # Claude Code + Kimi + Codex (default)
ccr install-global --agents auto # Auto-detect all installed agents
```

This configures:
- **Claude Code** global MCP + hooks (`~/.claude/.mcp.json`, `~/.claude/settings.json`)
- **Kimi Code CLI** global MCP + hooks (`~/.kimi/mcp.json`, `~/.kimi/config.toml`)
- **Codex CLI** global MCP + hooks (`~/.codex/config.toml`)
- **Continue.dev** MCP config (`~/.continue/config.json`)
- **Ollama** wrapper script (`~/.ccr/bin/ollama-ccr`)
- **OpenAI API** SDK wrapper + CLI prefix
- Helper scripts in `~/.ccr/bin/` and shell aliases

After installation, run your agent from any project directory. `.ccr/` is auto-created on first use. For Codex, use `codex-ccr` when you want the full CCR lifecycle wrapper; plain `codex` still gets the global MCP server and native hooks.

Generated MCP and hook configs set `CCR_STORAGE_BACKEND=sqlite`, so CCR tools
and lifecycle hooks use `.ccr/memory.db` by default while manual library use
without that env var remains file-backed for compatibility.

See [docs/AGENTS.md](docs/AGENTS.md) for per-agent setup details.

### Per-Project Setup (`ccr install`)

If you prefer per-project configuration (e.g., for team settings in version control):

```bash
cd /your/project
ccr install --agent claude-code
```

### Manual Setup

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

Then in your session, call `gcc_context(level=2)` to load memory and `gcc_commit` after completing tasks.

## Features

### Persistent Memory (GCC)

- **Commits**: Save what you did, why, files changed, and what's next
- **Branches**: Isolate experiments with `gcc_branch`, merge when decided
- **Context levels**: 5 levels of detail retrieval (summary → full history)
- **Pattern buffer**: Transferable skills extracted from commits, with quality scoring
- **Cross-linking**: Automatic bidirectional links between related commits
- **Semantic search**: Find past work by meaning, not just keywords (ONNX embeddings)

### Playbooks (ACE)

- **Strategy bullets**: "When X, do Y" rules with helpful/harmful counters
- **Temporal decay**: Unused strategies fade (30 days → 21% weight, 90 days → 1%)
- **Two-tier scope**: Global strategies (all projects) + project-specific strategies
- **Failure lessons**: Structured analysis of what went wrong and prevention principles
- **Optional LLM-powered evolution**: Automatic bullet generation, curation, and deduplication when a sub-model is configured

### Sandboxed REPL (RLM)

- **Python-level sandbox**: AST validation, restricted builtins, and module allowlist
- **Repo tools**: `search_repo()`, `get_file()`, `estimate_tokens()` available in REPL
- **Structured output**: `FINAL_VAR` termination pattern for clean results
- **Optional kernel sandbox**: macOS Seatbelt / Linux Landlock available for standalone execution (disabled in MCP path to preserve repo tool access)

### Repo Indexing

- **Hybrid search**: Keyword + semantic + combined modes
- **Per-language parsing**: Symbol extraction for Python, TypeScript, Rust, Go, and more
- **ONNX embeddings**: Optional dense embeddings (all-MiniLM-L6-v2, 384-dim)
- **Zero-config**: Works immediately; semantic search available with `pip install ccr-memory[semantic]`

### Session Logger

Every Q&A turn (user message + the agent's response) is persisted to `.ccr/sessions.db` (SQLite). Use it to replay any past session, debug unexpected agent behaviour, or export conversation pairs for fine-tuning. Logging is automatic when hooks are active — the agent calls `session_log_turn` after each response. See [docs/session-logger.md](https://github.com/qbit-glitch/ccr/blob/main/docs/session-logger.md) for the full reference.

## Architecture

```
AI Agent ──stdio──> CCR MCP Server
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

### Session Logger

| Tool | Purpose |
|------|---------|
| `session_log_turn` | Log the current Q&A turn (called automatically after each response) |
| `session_get_history` | Retrieve recent turns for a session (defaults to current session) |
| `session_search` | Full-text search across all session turns (FTS5) |
| `session_export` | Export a session as `json`, `jsonl` (OpenAI fine-tune), or `markdown` |

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

All implementations use mechanical heuristics where possible. See `CLAUDE.md` (project architecture notes) for detailed limitation tables comparing CCR's implementation vs. each paper.

## vs. Alternatives

| Feature | CCR | Mem0 | Letta/MemGPT | Graphiti |
|---------|-----|------|--------------|---------|
| Auto-manages memory | Yes (Claude + Kimi hooks, Codex wrapper/hooks) | Yes | Yes | Yes |
| Multi-agent support | Yes (shared `.ccr/`) | No | No | No |
| Version control (branch/merge) | Yes | No | No | No |
| Playbooks with optional LLM evolution | Yes | No | No | No |
| Sandboxed REPL | Yes | No | No | No |
| No external database server | Yes | No | No (DB) | No (Neo4j) |
| Core features work without LLM billing | Yes | No | No | No |
| Open source | Apache 2.0 | Yes | Apache 2.0 | Apache 2.0 |

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

Apache 2.0 — see [LICENSE](LICENSE) for full text.
