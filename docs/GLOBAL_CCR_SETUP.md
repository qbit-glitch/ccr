# Global CCR Setup Guide — Fully Automatic Memory Across All Projects

> **Status:** ✅ Active — CCR memory is fully automatic for **Claude Code**, **Kimi Code CLI**, and **Codex CLI**. Codex gets complete lifecycle parity through the `codex-ccr` wrapper plus Codex-native hooks.

---

## What Changed

Your CCR installation has been upgraded from **per-project** to **global**. You no longer need to run `ccr install` in every new repository. Memory management is **fully automatic** in Claude Code, Kimi, and Codex.

### Before (Per-Project)
```bash
cd ~/project-a && ccr install
cd ~/project-b && ccr install
cd ~/project-c && ccr install  # ... repeat for every project
# Inside session: memory only works after local hooks are installed
```

### After (Global + Fully Automatic)
```bash
cd ~/any-project && cclaude      # Claude Code — memory auto-managed
cd ~/any-project && kimi-ccr     # Kimi — memory auto-managed (same experience!)
cd ~/any-project && codex-ccr    # Codex — memory auto-managed with wrapper lifecycle
```

---

## Architecture

```
Any Directory ──cclaude / kimi-ccr / codex-ccr──► AI Agent
                                          ├── MCP Server (tools)
                                          │   └── gcc_commit, gcc_context,
                                          │       ace_get_playbook, rlm_execute,
                                          │       index_search, session_log_turn, ...
                                          │
                                          └── Hooks (auto-memory lifecycle)
                                              ├── UserPromptSubmit → inject context
                                              ├── PostToolUse      → track changes (Claude/Kimi, optional Codex)
                                              ├── PreCompact/sim  → save state
                                              └── Stop             → auto-commit
```

**Claude Code, Kimi, and Codex all use CCR MCP tools plus lifecycle hooks.** Codex uses native `SessionStart`, `UserPromptSubmit`, and turn-scoped `Stop` hooks by default. Because Codex shows hook stdout, `SessionStart` retrieves full memory but prints only a compact 1-2 line summary. The high-frequency Codex `PostToolUse` hook is opt-in with `CCR_CODEX_POST_TOOL_USE=1 ccr install-global --agents codex`. The `codex-ccr` wrapper adds process start/exit lifecycle so session finalization and transcript reconciliation match Claude/Kimi as closely as Codex allows.

---

## Quick Commands

### Claude Code (fully automatic)
```bash
cclaude                          # alias → auto-inits .ccr/ → launches Claude
```

### Kimi Code CLI (fully automatic)
```bash
kimi-ccr                         # auto-inits .ccr/ → launches Kimi with CCR
```

### Codex CLI (fully automatic)
```bash
codex-ccr                        # native Codex hooks + CCR wrapper lifecycle
```

Plain `codex` still receives global CCR MCP tools and native hooks from `~/.codex/config.toml`, but it cannot provide process-exit finalization because Codex does not expose that lifecycle event to hooks.

### CCR CLI (anywhere)
```bash
ccr init                         # init .ccr/ in cwd
ccr status                       # show memory state
ccr context                      # print project context
ccr doctor                       # health check
ccr clean --days 90              # archive old commits
```

---

## How It Works — Fully Automatic Lifecycle

### 1. Session Start (`UserPromptSubmit` hook)
When you type your first message, CCR automatically:
- Loads your project's memory context (recent commits, rolling summary)
- Injects the ACE playbook (global + project strategies)
- Outputs a directive reminding the agent to use CCR tools

**You do nothing.** The agent already knows the project history.

### 2. During Session (`PostToolUse` hook)
For Claude Code and Kimi, after every tool call (file write, shell command, etc.), CCR silently:
- Tracks which files were modified
- Accumulates session state for auto-commit

**You do nothing.** Changes are tracked automatically.

Codex does not install this high-frequency hook by default to avoid repeated
hook activity during normal work. Codex is instructed to call `gcc_commit` only
after meaningful milestones, before likely context loss, or when the user
explicitly asks to save/update CCR memory. Otherwise it relies on prompt/session
hooks, turn `Stop`, and wrapper exit finalization unless
`CCR_CODEX_POST_TOOL_USE=1` is set during install.

### 3. Context Compaction (`PreCompact` hook or Codex simulation)
Before Claude/Kimi compacts context to save tokens, CCR:
- Logs the compaction event to memory
- Ensures no state is lost during the reset

Codex does not expose a native `PreCompact` hook, so CCR simulates the safety check with prompt-time context pressure reminders and wrapper-managed session state.

### 4. Session End (`Stop` hook + Codex wrapper exit)
When the agent session or Codex turn ends, CCR automatically:
- Generates a commit title, summary, and file list
- Saves the commit to `.ccr/commits.md`
- Extracts transferable patterns for the playbook
- Cross-links related commits

For Codex, turn-level `Stop` saves meaningful progress, and `codex-ccr` finalizes the overall session when the Codex process exits.

---

## Agent Feature Matrix

| Feature | Claude Code | Kimi Code CLI | Codex CLI |
|---------|------------|---------------|-----------|
| **MCP tools** (`gcc_commit`, `gcc_context`, `ace_get_playbook`, `rlm_execute`, `index_search`, etc.) | ✅ Yes | ✅ Yes | ✅ Yes |
| **Auto memory injection** at session start | ✅ Yes | ✅ Yes | ✅ Yes |
| **Auto-commit on session end / turn stop** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Tool-use tracking** | ✅ Yes | ✅ Yes | Opt-in |
| **Pre-compact save** | ✅ Yes | ✅ Yes | ✅ Simulated |
| **Session logger** (`session_log_turn`, `session_search`) | ✅ Yes | ✅ Yes | ✅ Yes |
| **Fully automatic** — no manual `gcc_commit` | ✅ Yes | ✅ Yes | Quiet lifecycle + milestone saves |

**Bottom line:** all three primary CLIs now have automatic CCR memory. Use `codex-ccr` for full Codex lifecycle parity; plain `codex` has MCP and native hooks but not wrapper exit finalization.

---

## Files Created / Modified

### Global Configs
| File | Purpose |
|------|---------|
| `~/.claude/.mcp.json` | CCR MCP server for Claude Code |
| `~/.claude/settings.json` | CCR hooks for Claude Code |
| `~/.kimi/mcp.json` | CCR MCP server for Kimi (fallback) |
| `~/.kimi/config.toml` | CCR hooks for Kimi (UserPromptSubmit, PostToolUse, PreCompact, Stop) |
| `~/.codex/config.toml` | CCR MCP server and hooks for Codex CLI |

### Helper Scripts (`~/.ccr/bin/`)
| Script | Purpose |
|--------|---------|
| `claude-ccr` | Auto-inits `.ccr/` → launches `claude` |
| `kimi-ccr` | Auto-inits `.ccr/` → launches `kimi` with explicit project root + merges existing MCP servers |
| `codex-ccr` | Auto-inits `.ccr/` → launches `codex` with CCR lifecycle env + exit finalization |
| `ccr-global` | Runs `ccr` CLI from dev venv anywhere |

### Shell Aliases (`~/.zshrc`, `~/.bashrc`)
```bash
alias ccr='ccr-global'
alias cclaude='claude-ccr'
alias kimi-ccr='kimi-ccr'
alias ccodex='codex-ccr'
```

---

## Existing Projects with Memory

The following projects already have `.ccr/` memory and will continue working seamlessly:

```
~/Desktop/coding-projects/gpt_oss_20b_projects
~/Desktop/coding-projects/claude-code
~/Desktop/coding-projects/DeepCode
~/Desktop/coding-projects/finetune_nemotron
~/Desktop/coding-projects/webcrawler
~/Desktop/coding-projects/supervised_instance_segmentation
~/Desktop/coding-projects/ironclaw
~/Desktop/coding-projects/satyam_dissertation
~/Desktop/coding-projects/auth0_mobile_productivity_app
~/Desktop/coding-projects/flash-moe
~/Desktop/coding-projects/chatterbox
~/Desktop/coding-projects/claude_awesome
~/Desktop/coding-projects/communicate_between_claudes
~/Desktop/coding-projects/davinci_resolve_edits
~/Desktop/coding-projects/CLI-Anything
~/Desktop/coding-projects/eeg_epilepsy
~/Desktop/coding-projects/edge_audio_transcriber
~/Desktop/coding-projects/agentic_office_claude
~/Desktop/coding-projects/mbps_panoptic_segmentation
... and more
```

---

## Migration from Per-Project Setup

If you previously ran `ccr install` in some projects, those per-project configs still work and **override** the global config when you're in those directories. Claude Code merges `.claude/settings.local.json` (project) with `~/.claude/settings.json` (global).

You can leave old per-project configs in place, or clean them up:

```bash
# In a project with old per-project CCR config:
ccr uninstall                    # removes local hooks + .mcp.json entry
# Global hooks will still work after this
```

---

## Environment Variables

| Variable | Effect |
|----------|--------|
| `CCR_PROJECT_ROOT` | Override project root detection (hooks + MCP) |
| `CCR_STORAGE_BACKEND` | Generated MCP/hook configs set `sqlite`; unset defaults to `files` for backward-compatible manual use |
| `CCR_OLLAMA_MODEL` | Enable local sub-model (e.g. `qwen2.5:7b`) |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_API_KEY_SUB` | Enable Haiku sub-model for synthesis |

---

## Important Notes

1. **Python path dependency:** The global setup uses the venv at `powering_claude_with_less_tokens/.venv/`. If you delete or move that repo, update these paths:
   - `~/.claude/.mcp.json` → `command`
   - `~/.claude/settings.json` → all CCR hook `command` fields
   - `~/.kimi/config.toml` → all CCR hook `command` fields
   - `~/.kimi/mcp.json` → `command`
   - `~/.codex/config.toml` → MCP `command` and hook `command` fields
   - `~/.ccr/bin/*` → `PYTHON` and `REPO` variables

2. **Kimi hooks are Beta:** Kimi's hook system is marked Beta. If Kimi updates its hook format in a future release, the CCR hooks may need adjustment. The hooks use the exact same Python scripts as Claude Code, so any behavioral changes will be minimal.

3. **Doctor reports missing hooks (Claude only):** `ccr doctor` checks the **current project** for `.claude/settings.local.json` hooks. With global hooks, it reports `[!!] Hooks not configured` — this is expected and harmless. The global hooks in `~/.claude/settings.json` are active.

4. **Session logging:** Sessions are logged to `.ccr/sessions.db` per-project. `codex-ccr` reconciles Codex JSONL transcripts on wrapper exit; native Codex turn `Stop` also reconciles when Codex provides a transcript path.

5. **Launch from project directory:** For the most reliable project detection, always launch `cclaude`, `kimi-ccr`, or `codex-ccr` from inside your project directory. Hooks auto-initialize `.ccr/` if missing.

---

## Troubleshooting

### CCR tools not appearing
**Claude:** Restart Claude Code completely (quit and re-open).  
**Kimi:** Run `kimi mcp list` or `/mcp` inside Kimi to check connection status.
**Codex:** Run `codex mcp get ccr` to verify the global CCR server entry.

### Hooks not firing
```bash
# Check for hook errors
# Claude:
tail -60 .ccr/.hook_errors.log

# Kimi:
tail -60 ~/.kimi/logs/*.log 2>/dev/null | grep -i hook

# Verify global hooks exist
# Claude:
cat ~/.claude/settings.json | grep ccr

# Kimi:
cat ~/.kimi/config.toml | grep -A1 "\[\[hooks\]\]"

# Codex:
cat ~/.codex/config.toml | grep -A3 "\[\[hooks\."
```

### Kimi wrapper issues
```bash
# Test the wrapper in isolation
~/.ccr/bin/kimi-ccr --help

# If other MCP servers disappear, check ~/.kimi/mcp.json is intact
# The wrapper merges CCR with your existing servers
```

### Move CCR to a different Python/venv
```bash
# 1. Update MCP server paths in:
#    ~/.claude/.mcp.json
#    ~/.kimi/mcp.json
# 2. Update all hook commands in:
#    ~/.claude/settings.json
#    ~/.kimi/config.toml
# 3. Update ~/.ccr/bin/* scripts
```

---

## Backups

Backups of your original configs were saved before modification:
```bash
ls -la ~/.claude/*.backup.* ~/.kimi/*.backup.*
```

To restore:
```bash
cp ~/.claude/.mcp.json.backup.YYYYMMDD-HHMMSS ~/.claude/.mcp.json
cp ~/.claude/settings.json.backup.YYYYMMDD-HHMMSS ~/.claude/settings.json
cp ~/.kimi/config.toml.backup.YYYYMMDD-HHMMSS ~/.kimi/config.toml
```

---

*Generated on 2026-04-21. The global CCR setup uses the development repo at `~/Desktop/coding-projects/powering_claude_with_less_tokens` as the canonical CCR installation.*
