# CCR Hooks -- Automatic Memory Management

## Overview

CCR uses Claude Code hooks to automatically track work and commit to memory without requiring manual `gcc_commit` calls. Hooks fire at specific lifecycle events in a Claude Code session.

Three hooks work together as a pipeline:

1. **UserPromptSubmit** -- Initializes session, injects context and playbook.
2. **PostToolUse** -- Silently accumulates file edits, test runs, and git operations.
3. **Stop** -- Auto-commits accumulated state if meaningful progress was detected.

An additional **PreCompact** hook reminds Claude Code to commit before context window compaction.

## How It Works

### Session Start (UserPromptSubmit)

File: `ccr/hooks/on_session_start.py`

On the **first prompt** of a session:
- Creates a `.ccr/.session_active` marker file (atomic `O_CREAT|O_EXCL` to prevent race conditions).
- Initializes fresh session state via `state_accumulator.initialize_state()`.
- Creates a new session row in `.ccr/sessions.db` and writes the session ID to `.ccr/.current_session_id` (Session Logger).
- Reads level-1 memory context from `.ccr/`.
- Reads global playbook from `~/.ccr/global_playbook.txt`.
- Reads project playbook from `.ccr/playbook.txt`.
- Logs a "session start" OTA triple.
- Outputs context, playbook, and a `<MANDATORY_CCR_ACTIONS>` directive to stdout (Claude Code captures hook stdout as injected context).

On **every prompt** (first and subsequent):
- Writes the user's message to `.ccr/.pending_user_msg` using an atomic write (`tmp` + `os.replace`). The `session_log_turn` MCP tool reads and deletes this file when Claude logs the turn after responding.

On **subsequent prompts** in the same session:
- Outputs a lightweight `<ccr_reminder>` to commit after tasks.

### Tool Use Accumulation (PostToolUse)

File: `ccr/hooks/on_tool_use.py`

Fires after every tool use. **Silent** -- produces no stdout. Reads tool context from `CLAUDE_TOOL_NAME` and `CLAUDE_TOOL_INPUT` environment variables.

Detects and accumulates:

| Tool | Detection | What is recorded |
|------|-----------|-----------------|
| `Write`, `Edit` | `file_path` in input | File path added to `files_touched` |
| `Bash` (pytest) | `"pytest"` in command | `"Ran tests"` in `what_accumulated` |
| `Bash` (git commit) | `cmd.startswith("git commit")` | `"Git commit"` |
| `Bash` (git *) | `cmd.startswith("git ")` | `"Git: {subcommand}"` |

Appends each event to `.ccr/.session_state.json` via `state_accumulator.append_tool_use()`.

### Auto-Commit on Stop

File: `ccr/hooks/on_stop.py`

Fires when Claude Code session ends. Reads accumulated session state and auto-commits if meaningful:

- Loads `SessionState` from `.ccr/.session_state.json`.
- Checks `is_meaningful()`: total `what_accumulated` text >= 50 chars, OR >= 3 files touched.
- If meaningful: converts state to commit fields and calls `mem.commit()`.
- If not meaningful: logs a clean session-end OTA triple.
- Always clears session state file on exit.
- **Session Logger finalization**: reads `.ccr/.current_session_id`, sets `ended_at` on the session row in `sessions.db`, then deletes `.current_session_id`. Also deletes `.ccr/.pending_user_msg` if it exists (cleanup for any unread buffered prompt). Both operations are non-fatal — a failure here never blocks session termination.

### Pre-Compact Reminder

File: `ccr/hooks/on_compact.py`

Fires before context window compaction. Outputs a reminder to stdout:

```
REMINDER: Context is about to be compacted.
Use gcc_commit to save your progress before state is lost.
```

Also logs a "pre-compact" OTA triple.

### Legacy Session End

File: `ccr/hooks/on_session_end.py`

Simpler predecessor to `on_stop.py`. Only logs a session-end OTA triple without auto-commit logic. Superseded by `on_stop.py` for projects using `ccr install`.

## Session State Accumulator

File: `ccr/hooks/state_accumulator.py`

Shared module providing atomic read/write of session state across hooks.

### SessionState dataclass

```python
@dataclass
class SessionState:
    files_touched: list[str]       # Project-relative paths of modified files
    tasks_completed: list[str]     # Task descriptions
    what_accumulated: list[str]    # Summaries of detected tool operations
    patterns_observed: list[str]   # Transferable patterns (for gcc_commit)
    tool_calls: int                # Total tool call count
    start_time: float              # Session start timestamp
```

### Key functions

| Function | Description |
|----------|-------------|
| `initialize_state(ccr_root)` | Create fresh state with start timestamp. |
| `load_state(ccr_root)` | Load from `.ccr/.session_state.json`. Returns empty state if missing. |
| `save_state(ccr_root, state)` | Atomic write (write to `.tmp`, then `os.replace`). |
| `append_tool_use(ccr_root, tool_name, summary, files)` | Read-modify-write a tool use event. |
| `clear_state(ccr_root)` | Remove session state file. |

### Meaningfulness threshold

Auto-commit fires when either condition is met:
- Total accumulated text >= 50 characters.
- >= 3 distinct files touched.

These thresholds are hardcoded in `SessionState.is_meaningful()`.

## Installation

Run `ccr install` from the project directory (or `ccr install /path/to/project`):

```bash
ccr install .
```

This generates hook configuration in `.claude/settings.local.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [{
      "type": "command",
      "command": "/path/to/.venv/bin/python /path/to/ccr/hooks/on_session_start.py"
    }],
    "PostToolUse": [{
      "type": "command",
      "command": "/path/to/.venv/bin/python /path/to/ccr/hooks/on_tool_use.py"
    }],
    "Stop": [{
      "type": "command",
      "command": "/path/to/.venv/bin/python /path/to/ccr/hooks/on_stop.py"
    }],
    "PreCompact": [{
      "type": "command",
      "command": "/path/to/.venv/bin/python /path/to/ccr/hooks/on_compact.py"
    }]
  }
}
```

It also writes `.mcp.json` and initializes `.ccr/` if not already present.

## Customization

### Disabling auto-commit

Remove the `Stop` hook entry from `.claude/settings.local.json`. The session start hook can remain active for context injection without auto-commit.

### Adjusting meaningfulness threshold

Edit `ccr/hooks/state_accumulator.py`, method `SessionState.is_meaningful()`:

```python
def is_meaningful(self, min_chars: int = 50) -> bool:
    total = sum(len(w) for w in self.what_accumulated)
    return total >= min_chars or len(self.files_touched) >= 3
```

Change `min_chars` default or the file count threshold.

### Adding custom tool detectors

Edit `ccr/hooks/on_tool_use.py` `main()` to detect additional tools. The pattern is:

```python
elif tool_name == "YourTool":
    summary = "Description of what happened"
    files = [input_data.get("some_path", "")]
```

### Environment variables

All hooks respect `CCR_PROJECT_ROOT` to override the project directory. If not set, `os.getcwd()` is used.

## File Structure

| Path | Purpose |
|------|---------|
| `.ccr/.session_state.json` | Accumulated session state (auto-deleted on session end). |
| `.ccr/.session_active` | Marker file for first-prompt detection (atomic create). |
| `.ccr/.current_session_id` | Active Session Logger session ID (written on first prompt, deleted on Stop). |
| `.ccr/.pending_user_msg` | Buffered user prompt for `session_log_turn` to consume (written each prompt, deleted after logging). |
| `.ccr/sessions.db` | Session Logger SQLite database (`sessions` + `turns` + FTS5 index). |
| `ccr/hooks/on_session_start.py` | Session initialization + context injection + Session Logger setup. |
| `ccr/hooks/on_tool_use.py` | Silent tool use accumulator. |
| `ccr/hooks/on_stop.py` | Auto-committer on session end + Session Logger finalization. |
| `ccr/hooks/on_compact.py` | Pre-compaction reminder. |
| `ccr/hooks/on_session_end.py` | Legacy session-end logger (superseded by on_stop.py). |
| `ccr/hooks/state_accumulator.py` | Shared SessionState dataclass + atomic file I/O. |
| `ccr/hooks/__init__.py` | Empty package marker. |
