# CCR — Context Compression & Retrieval

> **Without CCR:** "Can you remind me what we decided about the dataset preprocessing last week?"  
> **With CCR:** Your AI agent already knows — months of decisions, experiments, and code reasoning recalled instantly.

CCR gives AI agents persistent memory, self-evolving strategy playbooks, and a sandboxed Python REPL. Works with **Claude Code**, **Kimi Code CLI**, **Continue.dev**, **Ollama**, **OpenAI API**, and any MCP-capable agent. **macOS/Linux only** (Windows support is not yet implemented).

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
cd /your/project && claude   # Claude Code (paid)
cd /your/project && kimi     # Kimi Code CLI (free tier)
```

That's it. Your agent will automatically load project memory on every session start and auto-commit progress when you finish — in **every directory** without per-project setup. Memory is stored per-project in `./.ccr/` and auto-initialized on first use.

### What CCR Does

CCR is an MCP server that gives AI agents three capabilities they don't have natively:

1. **Persistent Memory (GCC)** — Git-style version-controlled memory that survives across sessions. Branch, merge, and search your project's decision history.
2. **Self-Evolving Playbooks (ACE)** — Strategy bullets that track what works and what doesn't, with temporal decay and automatic pruning.
3. **Sandboxed REPL (RLM)** — An isolated Python environment for iterative analysis, with repo search and structured output.

All tools run as pure logic with zero LLM calls. The AI agent itself provides the reasoning.

**Works with any AI agent** — use Claude Code, Kimi, Continue.dev, Ollama, or OpenAI. CCR memory is shared across all of them.

### For Researchers and Students

CCR is designed for long-running research projects where context loss is the main productivity bottleneck. A 3-month project means ~90 Claude Code sessions. Without CCR, each starts from scratch. With CCR, each starts where the last left off.

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
| Continue | Free extension | But LLM backends (OpenAI/Anthropic) require paid API keys |
| Ollama | Free | Runs local models; needs RAM/GPU for larger models |
| OpenAI API | Pay-per-token | No subscription, but every API call costs money |

> **Global pricing note:** Claude Pro ($20/mo) is US-priced. At PPP, this is $40–80/mo equivalent in many countries. The API-key path (~$2–8/mo actual usage) is the most accessible entry point for budget-constrained students.

See the [Student & Researcher Quickstart](https://github.com/qbit-glitch/ccr/blob/main/docs/quickstart-students.md) for setup, cost details, and a full PhD workflow guide.

### Global Setup (`ccr install-global`) — Recommended

Run once to enable CCR across **all** projects:

```bash
ccr install-global              # Claude Code + Kimi (default)
ccr install-global --agents auto # Auto-detect all installed agents
```

This configures:
- **Claude Code** global MCP + hooks (`~/.claude/.mcp.json`, `~/.claude/settings.json`)
- **Kimi Code CLI** global MCP + hooks (`~/.kimi/mcp.json`, `~/.kimi/config.toml`)
- **Continue.dev** MCP config (`~/.continue/config.json`)
- **Ollama** wrapper script (`~/.ccr/bin/ollama-ccr`)
- **OpenAI API** SDK wrapper + CLI prefix
- Helper scripts in `~/.ccr/bin/` and shell aliases

After installation, simply run your agent from any project directory. `.ccr/` is auto-created on first use.

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

### Session Logger

Every Q&A turn (user message + Claude's response) is persisted to `.ccr/sessions.db` (SQLite). Use it to replay any past session, debug unexpected Claude behaviour, or export conversation pairs for fine-tuning. Logging is automatic when hooks are active — Claude calls `session_log_turn` after each response. See [docs/session-logger.md](https://github.com/qbit-glitch/ccr/blob/main/docs/session-logger.md) for the full reference.

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

All implementations use mechanical heuristics (zero LLM calls). See `CLAUDE.md` for detailed limitation tables comparing CCR's implementation vs. each paper.

## vs. Alternatives

| Feature | CCR | Mem0 | Letta/MemGPT | Graphiti |
|---------|-----|------|--------------|---------|
| Auto-manages memory | Yes (hooks) | Yes | Yes | Yes |
| Multi-agent (Claude + Kimi) | Yes | No | No | No |
| Version control (branch/merge) | Yes | No | No | No |
| Self-evolving strategies | Yes | No | No | No |
| Sandboxed REPL | Yes | No | No | No |
| Zero LLM calls | Yes | No | No | No |
| Zero infrastructure | Yes | No | No (DB) | No (Neo4j) |
| Works without per-call LLM billing | Yes | No | No | No |
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
