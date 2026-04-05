# Changelog

All notable changes to ccr-memory are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.7] — 2026-04-05

### Fixed
- `ace_tools.py` split into three submodules to satisfy 800-line coding-style limit (1642 → 718 lines):
  - `ace_llm_tools.py` (~603 lines): ACE 3-agent pipeline helpers + `ace_generate_bullets` + `ace_evolve_from_failures`
  - `ace_schema_tools.py` (~388 lines): `_apply_rollback` + `_build_retrieval_proposals` + `ace_evolve_schema`
  - `ace_tools.py` retains core tools (`ace_get_playbook`, `ace_apply_delta`, `ace_update_counters`, `ace_find_similar`, `ace_prune`) and re-exports from submodules for backward compat
- `test_ace_industry.py`: `_mock_sub_client` now patches both `ace_tools` and `ace_llm_tools` so monkeypatches survive the module split

---

## [0.2.6] — 2026-04-05

### Fixed
- `ace_evolve_from_failures` restored as a registered MCP tool (decorator was accidentally removed)
- `ace_find_similar` auto-merge path: playbook re-resolved under `_state_lock` before writes (TOCTOU race)
- `cli.py` split into `cli_doctor.py` + `cli_stats.py` to satisfy 800-line coding-style limit (902 → ~630 lines)
- `server.py` tool-module imports wrapped in try/except with clear stderr message on failure

### Removed
- `rich>=13.0.0` removed from mandatory dependencies (was never imported — inflated install by ~2 MB)

---

## [0.2.5] — 2026-04-05

### Fixed
- `session_export`: changed `except ValueError` to `except Exception` so SQLite `OperationalError` (disk I/O, corrupted DB) is caught and returned gracefully
- `ccr doctor` check #10: version parity between installed `ccr-memory` package and `ccr.__version__` — shows `[OK]` when matched, `[!!]` with pip fix command when skewed
- `pip install -e .` regenerated dist-info to sync 0.2.4 → 0.2.5

### Added
- 7 new tests in `test_session_tools.py`: DB-error paths for `session_get_history`, `session_search`, `session_export`; invalid format; monkeypatched SQLite errors

---

## [0.2.4] — 2026-04-04

### Fixed
- SQLite FTS5 content table: added missing DELETE, BEFORE UPDATE, and AFTER UPDATE triggers (`turns_ad`, `turns_bu`, `turns_au`). Without all 4 triggers, deleting or updating a turn left phantom rows in the full-text search index
- `ccr install`: changed `dict.update()` to per-event list-merge — pre-existing non-CCR hooks on the same event (e.g. another tool's `UserPromptSubmit`) are now preserved
- `session_get_history` and `session_search`: wrapped in `try/except` for consistency with `session_log_turn`
- `python-dotenv>=1.0.0` moved from mandatory `[project.dependencies]` to `[project.optional-dependencies].legacy` — only used by the legacy HTTP proxy gateway

### Added
- `ccr doctor` severity tiers: `[OK]` (passing), `[!!]` (real issues), `[--]` (optional notices). ONNX and sqlite-vec absence are now notices, not issues
- `ccr doctor` check #9: duplicate hook command detection — flags when `ccr install` was run twice before the dedup fix

---

## [0.2.3] — 2026-04-03

### Added
- **Session Logger** (4 new MCP tools): `session_log_turn`, `session_get_history`, `session_search`, `session_export` — persists Q&A turns to `.ccr/sessions.db` (SQLite); supports JSONL export for OpenAI fine-tuning
- `ccr stats` ROI dashboard: token savings, session duration, 30-session projection
- `ccr uninstall` command: removes CCR hooks from `.claude/settings.local.json`
- `ccr clean` command: archives old commits to `.ccr/archive/` and rewrites `commits.md` to keep only recent history, reducing memory file size
- Transcript reconciliation in `on_stop.py`: inserts any Q&A turns from the Claude Code transcript that `session_log_turn` missed

---

## [0.2.2] — 2026-04-02

### Added
- `ccr doctor` command: 9 health checks including Python version, `.ccr/` directory, `.mcp.json`, ONNX availability, hooks, hook error log, disk usage
- Hook error logging: all 4 hooks (`on_session_start`, `on_stop`, `on_compact`, `on_tool_use`) now catch exceptions and write to `.ccr/.hook_errors.log` — silent crashes are surfaced
- `ccr doctor` check #8: flags recent entries in `.ccr/.hook_errors.log` as `[!!]` issues
- `ccr install` now writes all 4 hooks + `.mcp.json` in a single command

---

## [0.2.1] — 2026-04-01

### Fixed
- `ModuleNotFoundError: No module named 'anthropic'` on fresh install: `ccr/__init__.py` eagerly imported `CCREngine` → `engine.py` → `anthropic_client.py`. Fixed by rewriting `__init__.py` to only export `MemoryManager` and `CCRConfig`
