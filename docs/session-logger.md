# Session Logger

## Overview

The Session Logger persists every Q&A turn — user message plus Claude's full response — to `.ccr/sessions.db` (SQLite). Data is scoped per-project, stored alongside the rest of CCR's state. The three primary use cases are: replaying past sessions to understand what was decided and why, debugging unexpected Claude behaviour by examining the exact context at each turn, and exporting conversation pairs for fine-tuning or evaluation.

## How It Works

### Turn capture pipeline

1. **UserPromptSubmit hook** (`ccr/hooks/on_session_start.py`): Fires at the start of every prompt. On the first prompt of a session it creates a new row in `sessions.db` and writes the session ID to `.ccr/.current_session_id`. On every prompt (first and subsequent) it writes the user's message to `.ccr/.pending_user_msg` using an atomic write (`tmp` file + `os.replace`).

2. **`session_log_turn` MCP tool** (`ccr/mcp/session_tools.py`): Claude calls this after composing each response. The tool reads `.ccr/.pending_user_msg` (consuming and deleting it), pairs it with the `assistant_message` argument, and inserts a row into the `turns` table. Turn numbers are assigned under a threading write lock to prevent concurrent-insert races. If no active session ID is found in memory, the tool reads `.ccr/.current_session_id` from disk; if that is also absent it creates a transient session so no data is silently dropped.

3. **Stop hook** (`ccr/hooks/on_stop.py`): Fires when the Claude Code session ends. Reads `.ccr/.current_session_id`, sets `ended_at` on the session row, then deletes both `.current_session_id` and `.pending_user_msg`.

### Reliability note

> Session turns are logged when Claude calls `session_log_turn`. In practice this happens automatically via the `<MANDATORY_CCR_ACTIONS>` directive injected by the UserPromptSubmit hook. If Claude is heavily context-constrained, some turns may be missed. Missed turns appear as gaps in `session_get_history()` — `turn_number` values are sequential, so a jump from turn 3 to turn 5 means turn 4 was not logged.

### Storage details

| Item | Value |
|------|-------|
| Database file | `.ccr/sessions.db` (per-project, alongside `.ccr/commits`) |
| Journal mode | WAL — safe for concurrent readers |
| Write serialization | `threading.Lock` on turn inserts |
| FTS engine | FTS5 virtual table (compiled into CPython's `sqlite3` on macOS and major Linux distros); falls back to `LIKE` if unavailable |

#### Schema

```sql
sessions (
    id          TEXT PRIMARY KEY,        -- e.g. ses_20260405_143012_a3f9c1
    project     TEXT NOT NULL DEFAULT '',
    started_at  TEXT NOT NULL,           -- ISO-8601 UTC
    ended_at    TEXT,                    -- NULL until Stop hook fires
    turn_count  INTEGER NOT NULL DEFAULT 0
)

turns (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT NOT NULL REFERENCES sessions(id),
    turn_number         INTEGER NOT NULL,   -- 1-based, sequential per session
    timestamp           TEXT NOT NULL,      -- ISO-8601 UTC
    user_message        TEXT NOT NULL DEFAULT '',
    assistant_message   TEXT NOT NULL DEFAULT '',
    tool_calls_json     TEXT NOT NULL DEFAULT '[]',
    files_touched_json  TEXT NOT NULL DEFAULT '[]',
    token_estimate      INTEGER NOT NULL DEFAULT 0,  -- len // 4 approximation
    source              TEXT NOT NULL DEFAULT 'direct',
    UNIQUE(session_id, turn_number)
)

-- FTS5 virtual table (populated by AFTER INSERT trigger on turns)
turns_fts USING fts5(user_message, assistant_message, content='turns', content_rowid='id')
```

## MCP Tools

| Tool | Annotations | Description |
|------|-------------|-------------|
| `session_log_turn(assistant_message, user_message="", tool_calls=None, files_touched=None)` | write | Log the current Q&A turn. `user_message` is auto-read from `.pending_user_msg` if omitted. Called automatically after each response via the hook directive. |
| `session_get_history(session_id="", limit=20, offset=0)` | read-only | Retrieve turns for a session ordered by `turn_number`. Defaults to the current session. Supports pagination via `offset`. |
| `session_search(query, limit=10)` | read-only | Full-text search across all turns using FTS5 (or LIKE fallback). Returns `session_id`, `turn_number`, `timestamp`, and highlighted snippets. |
| `session_export(session_id="", format="jsonl")` | read-only | Export a session. `format` must be one of `"json"`, `"jsonl"`, or `"markdown"`. Defaults to the current session. |

### Export formats

| Format | Structure | Use case |
|--------|-----------|----------|
| `json` | `{"session": {...}, "turns": [...]}` — full object with all fields | Programmatic processing, archival |
| `jsonl` | One `{"messages": [{"role":"user",...},{"role":"assistant",...}]}` per line | OpenAI fine-tuning API, evaluation sets |
| `markdown` | Human-readable with `## Turn N` headings | Reading, sharing, code review of sessions |

## Example Workflows

Review the current session (last 20 turns):
```
session_get_history()
```

Page through a long session:
```
session_get_history(limit=20, offset=20)
```

Search all past sessions for a specific topic:
```
session_search(query="SQLite WAL")
```

Export the current session for fine-tuning:
```
session_export(format="jsonl")
```

Export a specific past session as markdown:
```
session_export(session_id="ses_20260405_143012_a3f9c1", format="markdown")
```

Export a past session as a full JSON object:
```
session_export(session_id="ses_20260405_143012_a3f9c1", format="json")
```

## Configuration

Both flags live in `CCRConfig` (`ccr/core/types.py`):

```python
session_logging_enabled: bool = True   # set False to disable the SessionStore entirely
session_fts_enabled: bool = True       # set False to skip FTS5 table creation
```

`session_db_path` defaults to `.ccr/sessions.db` when empty (the server sets it at startup in `ccr/mcp/server.py`). Override by setting `session_db_path` in your `CCRConfig` instance.

## Files

| Path | Purpose |
|------|---------|
| `ccr/core/session_store.py` | `SessionStore` class — all DB logic |
| `ccr/mcp/session_tools.py` | Four MCP tools wrapping `SessionStore` |
| `ccr/hooks/on_session_start.py` | Creates DB session + buffers user prompt |
| `ccr/hooks/on_stop.py` | Finalizes session + cleans up temp files |
| `.ccr/sessions.db` | SQLite database (per-project) |
| `.ccr/.current_session_id` | Active session ID (deleted on Stop) |
| `.ccr/.pending_user_msg` | Buffered user prompt (consumed by `session_log_turn`) |
