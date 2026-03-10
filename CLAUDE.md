# CCR — Claude Context Reducer

MCP server that gives Claude Code persistent memory (GCC), self-evolving strategy playbooks (ACE), and a sandboxed Python REPL (RLM). No API keys or sub-models needed — works with Claude Max subscription.

## How To Use CCR (MCP Tools)

### Memory (GCC) — Use these to persist knowledge across sessions

- **At session start**: Call `gcc_context(level=2)` to ground yourself in project history
- **After meaningful progress**: Call `gcc_commit` with what/why/files/next to save state (auto-merges if too similar to last commit)
- **Before context gets large**: Commit to avoid losing reasoning state on compaction
- **When exploring alternatives**: Use `gcc_branch` to isolate experiments, `gcc_merge` when decided
- **To search past work**: Use `gcc_context(level=5, search_term="...")` to find specific commits

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
  mcp_server.py     # MCP server: all 18 tools (GCC + ACE + RLM + Index)
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

- **MCP server**: 18 tools exposed via stdio transport — GCC memory, ACE playbook, RLM sandbox, repo index
- **Memory**: GCC-inspired `.ccr/` directory. Commits track what was done/learned/planned. Branches for experiments. 5-level context retrieval. A-MAC admission control with correct polarity: S(m) = 0.60·Novelty + 0.40·TypePrior. Algorithm 1 with S(m) vs S(m_conflict) comparison, recency-modulated FindConflict (sim threshold 0.85 per §3.3), three-way admit/merge/reject, structural bypass.
- **Playbook evolution**: ACE playbook with helpful/harmful counters. Claude Code reflects and curates — no sub-model needed.
- **Structured Failure Lessons** (SkillRL-inspired): When tagging harmful, include a failure_lesson dict explaining why. Lessons accumulate, then `ace_evolve_from_failures` generates NEW skill bullets from prevention principles. Companion data in `.ccr/failure_lessons.json`.
- **Sandboxed REPL**: RLM-inspired REPL with repo tools (search_repo, get_file, FINAL_VAR). Claude Code drives the iteration loop.
- **Exception hierarchy**: `CCRError` base with `recoverable` flag.
- **Shared JSON extraction**: `extract_json_from_llm()` and `extract_json_string()` in `utils/parsing.py`.

## Development Commands

```bash
source .venv/bin/activate
pytest tests/unit/ tests/integration/ -x -q  # Run all tests (728 pass, 5 skip)
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

## Research Papers Implemented

1. **GCC** (arXiv:2508.00031): Git Context Controller — version-controlled agent memory
   - COMMIT/BRANCH/MERGE/CONTEXT operations, metadata.yaml, OTA triples, rolling summary
   - Admission control (A-MAC, arXiv:2603.04549): Correct polarity (higher score = more valuable). 3 of 5 factors: Novelty N(m)=1-max_sim (Eq. 3, word Jaccard proxy for SBERT), Recency R=exp(-0.01·hours) (Eq. 4, λ=0.01/hour, half-life 69h), Type Prior T(m) (§3.2, rule-based 6-type classifier). U(m) and C(m) omitted (require LLM). Recency-modulated FindConflict. Algorithm 1 three-way: admit/merge/reject. Type-prior bypass for structural ops.
2. **RLM** (arXiv:2512.24601): Recursive Language Models — REPL-based execution
   - Metadata-only stdout, prompt as REPL variable, FINAL_VAR termination
3. **ACE** (arXiv:2510.04618): Agentic Context Engineering — evolving playbooks
   - Structured bullets with helpful/harmful counters, grow-and-refine
   - Delta operations (ADD/UPDATE/MERGE/REMOVE), deterministic merge
   - Playbook stored at `.ccr/playbook.txt`
4. **SkillRL** (arXiv:2602.08234): Experience-based skill distillation (failure side)
   - Structured failure lessons: failure_point, flawed_reasoning, counterfactual, prevention_principle
   - Threshold-triggered evolution: accumulated failures → NEW skill bullets
   - Hierarchical scope (general/task_specific), when_to_apply per SkillRL Table 5
   - Idempotent evolution with `evolved` flag, dedup against existing bullets
   - Extended data in `.ccr/failure_lessons.json` (backward-compatible format)

## ACE Playbook Format

```
## SECTION_NAME
[slug-00001] helpful=5 harmful=0 :: Strategy or insight content here
```

Sections: STRATEGIES & INSIGHTS, CODE SNIPPETS & TEMPLATES, COMMON MISTAKES TO AVOID, PROBLEM-SOLVING HEURISTICS, CONTEXT CLUES & INDICATORS, OTHERS.
