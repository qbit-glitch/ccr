# CCR v6: Student & Researcher Usability — 5/10 → 9.5/10

**Date:** 2026-04-05  
**Goal:** Push CCR's usability score for students and researchers from 5/10 to 9.5/10 across all five rated dimensions.

---

## Context

CCR scored 4/10 for student usability in April 2026 (brutal agent evaluation). After round-1 fixes (ccr install, experiment field, quickstart doc), it reached 5/10. Three critical gaps remain:

| Dimension | Current | Target |
|---|---|---|
| Token cost reduction | 7 | 9 |
| "Claude forgot everything" | 8 | 9.5 |
| Non-expert setup | 5 | 9 |
| Research experiment tracking | 5 | 9.5 |
| Value prop clarity to students | 5 | 9 |

**Critical bug discovered:** `.ccr/.session_active` marker is never deleted on session end. Every session after the first sees only the 19-token reminder instead of full memory context. Second-session memory recall is completely broken for new users.

---

## Architecture Overview

Three parallel streams, Stream A unblocks everything:

```
Stream A (foundations, must-first)
  A1: Fix session marker bug          → second sessions now recall memory
  A2: Empty-project first-session UX → new users guided, not confused
  A3: ccr doctor platform checks      → students know if sandbox works
  A4: ccr uninstall command           → students can cleanly remove CCR
  A5: ccr clean command               → long-lived projects stay healthy
  A6: pyproject.toml polish           → better PyPI discoverability

Stream B (research power, after A)
  B1: gcc_experiments query tool      → filter/compare/rank ML runs
  B2: Context level heuristic         → suggest level=3 for big projects
  B3: gcc_discuss + gcc_discussions   → persistent decision/hypothesis log
  B4: Session Logger (SL)             → automatic Q&A history in SQLite

Stream C (prove the value, after A)
  C1: Session token tracking infra    → measure what CCR actually saves
  C2: ccr stats ROI dashboard         → "is CCR worth it?" has a number answer
  C3: README rewrite                  → before/after dialogue first, not arXiv
```

---

## Stream A — Fix Foundations

### A1: Session Marker Bug Fix (CRITICAL)

**Bug:** `.ccr/.session_active` is created atomically on first prompt but never deleted. On the second Claude Code session, `_is_first_prompt()` finds the marker, thinks the session is already active, and injects only the 19-token reminder instead of full memory context.

**Files:**
- `ccr/hooks/on_stop.py` — add deletion after `clear_state()`
- `ccr/hooks/on_session_start.py` — add PID validation (stale marker guard)

**`on_stop.py` change:** After line 53 (`clear_state(mem.ccr_root)`):
```python
# Delete session marker so next session gets full context injection
marker = os.path.join(mem.ccr_root, ".session_active")
try:
    os.unlink(marker)
except (FileNotFoundError, OSError):
    pass  # Non-fatal
```

**`on_session_start.py` — replace `_is_first_prompt()` (lines 20-33):**
```python
def _is_first_prompt(ccr_root: str) -> bool:
    marker = os.path.join(ccr_root, ".session_active")
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True  # Created fresh — definitely first prompt
    except FileExistsError:
        # Check if owning process is still alive (stale marker guard)
        try:
            with open(marker, "r") as f:
                stored_pid = int(f.read().strip())
            os.kill(stored_pid, 0)  # Raises OSError if dead
            return False  # Process alive — genuinely mid-session
        except (ValueError, OSError):
            # Stale: replace marker and treat as first prompt
            try:
                with open(marker, "w") as f:
                    f.write(str(os.getpid()))
            except OSError:
                pass
            return True
    except OSError:
        return False
```

**Crash recovery notice:** When a stale marker is replaced, `_handle_session_start` prepends:
```
⚡ CCR reloaded after unclean shutdown — previous session marker was stale. Memory context injected fresh.
```
This surfaces to the user via Claude's context so they know the prior session may not have auto-committed.

**Tests (new file `tests/unit/test_session_marker.py`):**
1. Marker created on first call, returns (True, False)
2. Marker exists with live PID → returns (False, False)
3. Marker exists with dead PID → returns (True, True), marker replaced
4. Marker with non-integer content → returns (True, True)
5. on_stop deletes marker → next call returns (True, False)
6. Missing `.ccr/` → no crash, returns (False, False)

---

### A2: Empty-Project First-Session UX

**Bug:** Fresh project with 0 commits still injects `<MANDATORY_CCR_ACTIONS>` — "Before responding: call gcc_context(level=2)" is meaningless when nothing exists. Students are confused.

**File:** `ccr/hooks/on_session_start.py`

**Add helper** (after `_is_first_prompt`):
```python
def _project_has_commits(mem) -> bool:
    """Return True if any commits exist on the active branch."""
    import re
    branch = mem.get_active_branch()
    recent = mem._read_commits_window(branch, 0, 1)
    return bool(re.search(r"## \[C\d{3,}\]", recent))
```

**Modify `_handle_session_start`** — add branch at top:
```python
def _handle_session_start(mem):
    if not _project_has_commits(mem):
        print("""<ccr_ready>
CCR is active. This project has no memory yet — that's normal on first use.

After finishing your first task, call:
  gcc_commit(title="...", what="...", why="...", files_changed=[...], next_step="...")

CCR will then remember your progress across all future sessions automatically.
See: docs/quickstart-students.md for a 5-minute guide.
</ccr_ready>""")
        return
    # ... existing logic ...
```

**Tests:**
1. 0 commits → prints `<ccr_ready>`, no `<MANDATORY_CCR_ACTIONS>`
2. 1+ commits → prints `<MANDATORY_CCR_ACTIONS>` + `<gcc_context>`
3. Template-only `main.md` (no `[C###]` markers) → treated as empty

---

### A2b: ML-File Commit Nudge (on_tool_use.py)

**Motivation:** Researchers forget to add `experiment=` to `gcc_commit` after training runs. The `on_tool_use.py` hook already sees every file written; we can detect ML-flavoured writes and nudge.

**File:** `ccr/hooks/on_tool_use.py` — add nudge check when Write/Edit tool fires on a likely results file.

**Detection heuristic** — file path matches any of:
- `*.log`, `results/*.json`, `outputs/**/*.json`, `runs/**`, `wandb/**`, `metrics.json`, `eval_results.json`

**Output** (appended to tool-use hook stdout, only once per session):
```
<ccr_nudge>
Looks like you wrote results/metrics.json.
If this is an experiment run, consider adding experiment= to your next gcc_commit:
  gcc_commit(..., experiment={"id": "run-N", "hypothesis": "...", "metrics": {"val_loss": 0.23}, "conclusion": "..."})
</ccr_nudge>
```

**Guard:** Only emit once per session (write `.ccr/.nudge_sent` flag). Don't spam every save.

**Tests:**
1. Write to `results/metrics.json` → nudge emitted
2. Write to `main.py` → no nudge
3. Second ML-file write in same session → no nudge (flag set)
4. Non-existent `.ccr/` → no crash

---

### A3: `ccr doctor` Platform Checks

**File:** `ccr/cli.py`, insert after line ~418 (disk usage check), before print block.

**Check 9 — RLM sandbox mode:**
```python
import platform, shutil
system = platform.system()
if system == "Darwin":
    if shutil.which("sandbox-exec"):
        ok_items.append("RLM sandbox: Seatbelt (macOS) — full kernel isolation")
    else:
        issues.append("RLM sandbox: sandbox-exec not found (unexpected on macOS)")
elif system == "Linux":
    try:
        major, minor = [int(x) for x in platform.release().split("-")[0].split(".")[:2]]
        if (major, minor) >= (5, 13):
            ok_items.append(f"RLM sandbox: Landlock available (kernel {major}.{minor})")
        else:
            issues.append(f"RLM sandbox: Landlock unavailable (kernel {major}.{minor} < 5.13) — RLM runs unsandboxed")
    except (ValueError, IndexError):
        issues.append("RLM sandbox: Cannot detect kernel version")
elif system == "Windows":
    issues.append("RLM sandbox: No kernel sandbox on Windows — RLM runs unsandboxed")
```

**Check 10 — Stale session marker:**
```python
session_marker = os.path.join(ccr_dir, ".session_active")
if os.path.isfile(session_marker):
    try:
        with open(session_marker) as f:
            stored_pid = int(f.read().strip())
        os.kill(stored_pid, 0)
        ok_items.append(f"Session marker: active session (PID {stored_pid})")
    except (ValueError, OSError):
        issues.append(f"Session marker: STALE — delete {session_marker} or run 'ccr install' to reset")
else:
    ok_items.append("Session marker: absent (clean state)")
```

---

### A4: `ccr uninstall` Command

**File:** `ccr/cli.py` — insert after `install` command (~line 306).

Removes CCR hooks from `.claude/settings.local.json` and removes `ccr` entry from `.mcp.json`. Does NOT touch `.ccr/` (memory preserved).

```python
@cli.command()
@click.argument("project", default=".")
@click.option("--yes", "-y", is_flag=True)
def uninstall(project: str, yes: bool) -> None:
    """Remove CCR hooks and MCP registration (memory is preserved)."""
    # ... confirm + remove "ccr"-referencing entries from hooks + mcp.json ...
```

Key behavior: only removes hook entries where `"ccr"` appears in the command path (preserves other hooks a student may have added).

**Install path clarity (A1b):** The `ccr install` success message now prints:
```
=== Next step ===
  Open Claude Code from this exact directory:
    cd /absolute/path/to/project && claude

  ⚠️  Claude Code must be launched from /absolute/path/to/project
     Hooks only fire when CWD matches the project root.
```
This prevents the most common student failure mode: installing into `/path/A` then opening Claude Code from `/path/B`.

**Tests:** install → uninstall → assert hooks/MCP removed, `.ccr/` intact; idempotent; nothing installed → graceful.

---

### A5: `ccr clean` Command

**File:** `ccr/cli.py` — insert after `uninstall`.

Prunes commits older than N days (default 90) to `.ccr/archive/YYYY-MM-DD_branch.md`. Rolling summary preserved in `commits.md`.

```python
@cli.command()
@click.argument("project", default=".")
@click.option("--days", "-d", default=90, type=int)
@click.option("--dry-run", is_flag=True)
@click.option("--yes", "-y", is_flag=True)
def clean(project: str, days: int, dry_run: bool, yes: bool) -> None:
    """Prune old commits to .ccr/archive/ (keeps rolling summary)."""
```

**Tests:** 10 old + 5 recent commits → 10 archived, 5 kept; dry-run shows preview; rolling summary survives; nothing to prune → graceful.

---

### A6: pyproject.toml Polish

- Add `"experiment-tracking"`, `"research-workflow"`, `"reproducibility"` to `keywords`
- Fix README content-type: `readme = {file = "README.md", content-type = "text/markdown"}`
- Minor: add Bug Reports URL

---

## Stream B — Research Power Features

### B1: `gcc_experiments` Query Tool

**New file:** `ccr/core/memory_pkg/memory_experiments.py` — `ExperimentsMixin`

Parses `**Experiment**:` blocks from commits (stored by `gcc_commit(experiment={...})`). Filters by metric ranges, hypothesis text, date, experiment ID. Sorts by metric.

**New MCP tool** in `ccr/mcp/gcc_tools.py`:
```python
def gcc_experiments(
    experiment_id: str | None = None,
    hypothesis_contains: str | None = None,
    metric_filter: dict | None = None,   # {"val_loss": {"lt": 0.3}}
    date_range: list[str] | None = None, # ["2026-01-01", "2026-04-01"]
    top_n: str | None = None,            # "val_loss:asc"
    compare: list[str] | None = None,    # ["exp-001", "exp-042"]
) -> GccExperimentsResult
```

**Output modes:**
- Default: markdown table of matching experiments (ID, commit, date, hypothesis, metrics, conclusion)
- `compare=[id1, id2]`: side-by-side comparison table

**New type** in `ccr/mcp_types.py`: `GccExperimentsResult`

**Integration:** Add `ExperimentsMixin` to `MemoryManager` MRO in `ccr/core/memory.py`, export from `ccr/core/memory_pkg/__init__.py`.

**Tests (7):** single result, metric filter (match + no-match), hypothesis filter, compare mode, top_n sort, no experiments → count=0, commit without experiment excluded.

---

### B2: Context Level Heuristic

**File:** `ccr/mcp/gcc_tools.py` — inside `gcc_context()`, after building `result`, before return.

```python
if level == 2:
    total = len(mem._build_commit_index(mem.get_active_branch()))
    if total >= 30:
        result["message"] += (
            f"\n\n[Hint: This project has {total} commits. "
            f"Try gcc_context(level=3) for richer history or "
            f"gcc_context(level=5, search_term='<topic>') for targeted search.]"
        )
```

**Tests:** 29 commits → no hint; 30 commits → hint; level=3 with 30 commits → no hint.

---

### B3: `gcc_discuss` + `gcc_discussions` — Persistent Decision Log

**New file:** `ccr/core/memory_pkg/memory_discussions.py` — `DiscussionsMixin`

Stores discussion/decision records to `.ccr/branches/{branch}/discussions.md`:

```markdown
## [D001] 2026-04-05 10:30 | dataset preprocessing approach
**Hypothesis**: Use TorchDataset for batch loading
**Alternatives**: pandas CSV loader, HDF5
**Decision**: TorchDataset
**Rationale**: 40% throughput gain in profiling (train.py benchmark)
**Uncertainty**: Not tested on datasets >10GB
**Linked Commit**: C045

---
```

**Two new MCP tools:**

```python
def gcc_discuss(
    topic: str,
    hypothesis: str,
    alternatives_considered: str,   # comma-separated
    decision: str,
    rationale: str,
    uncertainty: str = "",
    linked_commit: str | None = None,
) -> GccDiscussResult
```

```python
def gcc_discussions(
    search: str | None = None,       # full-text substring
    topic: str | None = None,        # exact topic match
    date_range: list[str] | None = None,
) -> GccDiscussionsResult
```

**Search integration:** When `gcc_context(level=5, search_term=X)` is called, also scan `discussions.md` and include matching blocks in the result under `# Matching Discussions`.

**New types:** `GccDiscussResult`, `GccDiscussionsResult` in `ccr/mcp_types.py`

**Integration:** Add `DiscussionsMixin` to `MemoryManager`, export from `__init__.py`.

**Tests (6):** create → D001 in file; create twice → D001+D002; search finds match; empty project → count=0; gcc_context level=5 search finds discussion; linked_commit stored correctly.

---

### B4: Session Logger (SL)

**New capability:** Automatically log every Q&A turn to SQLite for full conversation recall. Addresses the "what did we discuss last month?" gap.

**Storage:** `.ccr/sessions.db` (SQLite, append-only)

**Schema:**
```sql
CREATE TABLE turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    turn_index  INTEGER NOT NULL,
    timestamp   TEXT NOT NULL,           -- ISO 8601
    user_prompt TEXT NOT NULL,
    assistant   TEXT NOT NULL,
    tokens_est  INTEGER DEFAULT 0
);
CREATE INDEX idx_turns_session ON turns(session_id);
CREATE INDEX idx_turns_ts ON turns(timestamp);
```

**New MCP tools** (in new file `ccr/mcp/sl_tools.py`):

```python
def session_log_turn(
    assistant_message: str,        # full response text
    user_prompt: str = "",         # optional: inject user prompt for context
) -> SessionLogResult
# Called after EVERY assistant response (per CLAUDE.md instruction)

def session_get_history(
    limit: int = 20,               # last N turns of current session
    session_id: str | None = None, # default: current session
) -> SessionHistoryResult

def session_search(
    query: str,                    # full-text search across all sessions
    limit: int = 10,
    date_range: list[str] | None = None,
) -> SessionSearchResult

def session_export(
    format: str = "jsonl",         # "jsonl" (OpenAI fine-tune) or "markdown"
    session_id: str | None = None, # default: all sessions
) -> SessionExportResult
```

**New file:** `ccr/sl/logger.py` — `SessionLogger` class wrapping SQLite operations.

**Session ID:** Reuse `state.session_id` from `state_accumulator.py` (set in C1).

**New types:** `SessionLogResult`, `SessionHistoryResult`, `SessionSearchResult`, `SessionExportResult` in `ccr/mcp_types.py`.

**Tests (new file `tests/unit/test_sl_tools.py`):**
1. `session_log_turn` writes row to DB
2. `session_get_history` returns last N turns for current session
3. `session_search` returns turns containing query text
4. `session_export` returns valid JSONL (parse with `json.loads`)
5. Empty DB → history returns 0 turns, search returns 0
6. Multiple sessions → search spans all sessions, history is session-scoped

---

## Stream C — Prove the Value

### C1: Session Token Tracking Infrastructure

**Modified files:**
- `ccr/hooks/state_accumulator.py` — add `context_tokens: int = 0` and `session_id: str = ""` to `SessionState`; set `session_id = str(uuid.uuid4())[:8]` in `initialize_state()`
- `ccr/hooks/on_session_start.py` — after building `parts`, estimate tokens and save to state:
  ```python
  from ccr.utils.tokens import estimate_tokens
  ctx_tokens = estimate_tokens("\n\n".join(parts))
  state.context_tokens = ctx_tokens
  save_state(mem.ccr_root, state)
  ```
- `ccr/hooks/on_stop.py` — before `clear_state()`, write session record:
  ```python
  _write_session_metrics(mem, state)  # → .ccr/metrics/sessions.jsonl
  ```

**sessions.jsonl format** (one JSON per line):
```json
{"session_id": "abc12345", "start": "2026-04-05T10:00:00+00:00", "end": "2026-04-05T10:45:00+00:00", "context_tokens": 2707, "commits_in_session": 3, "branch": "main"}
```

**Tests:** on_stop writes record; two sessions = two lines; on_stop without prior on_session_start = context_tokens=0, skipped by ccr stats.

---

### C2: `ccr stats` ROI Dashboard

**File:** `ccr/cli.py` — new `stats` command.

```
$ ccr stats

=== CCR Stats Dashboard ===

Project memory (main)
  Commits:          31
  Rolling summary:  843/1500 chars (56%)
  Playbook:         2,428 tokens (47 bullets)

Session history (last 30 sessions)
  Avg context injected:  2,707 tokens/session
  Total injected:        81,210 tokens
  Est. tokens avoided:   324,840 tokens (re-typing heuristic: ×4)
  Avg session duration:  42 min

30-session projection
  Est. savings:  ~324,840 tokens
  Break-even:    6 sessions ✓ (already past)

Recent sessions
  Date         Context   Avoided   Duration
  ----------   -------   -------   --------
  2026-04-05    2,707    10,828      45 min
  2026-04-04    2,589    10,356      38 min
  ...
```

**Heuristic transparency:** The `×4` multiplier (tokens re-typed without CCR) is a rough heuristic, not measured data. Display it as:
```
Est. tokens avoided:   324,840 tokens (×4 re-typing heuristic*)
...
* Rough estimate: assumes you'd re-type ~4× the context without CCR.
  Pass --multiplier N to adjust (e.g., --multiplier 2 for conservative estimate).
```

Add `--multiplier` option:
```python
@click.option("--multiplier", "-m", default=4, type=float,
              help="Token re-typing multiplier (default 4). Use 2 for conservative, 6 for optimistic.")
```

**Edge cases:**
- No sessions.jsonl → "No session history yet."
- Omit sessions where `context_tokens == 0` from averages

**Tests (5):** no history → message; known data → correct totals; rolling summary >80% → warning; 30+ sessions → projection shown; broken jsonl line → skipped gracefully.

---

### C3: README Rewrite

**File:** `README.md` — rewrite first 60 lines.

**Current structure:** Title → "What CCR Does" (features) → Quick Start → Manual Setup
**New structure:** Hook (before/after dialogue) → Problem statement → Quick Start → features

**New first 30 lines:**
```markdown
# CCR — Claude Context Reducer

> **Without CCR:** "Can you remind me what we decided about my dataset preprocessing?"  
> **With CCR:** Claude already knows. No re-explaining. No copy-pasting. Months of decisions recalled instantly.

CCR gives Claude Code persistent memory, self-evolving strategy playbooks, and a sandboxed Python REPL — without API keys.
Works with Claude Max ($20/mo).

> **New to CCR?** See the [Student & Researcher Quickstart](docs/quickstart-students.md) — setup in 3 minutes.

## Quick Start

pip install ccr-memory  # or: pip install -e . from source
ccr install             # writes .mcp.json + all hooks automatically

Open Claude Code in your project. Done.
...
```

---

### B5: `gcc_search` — Unified Search Dispatcher

**Motivation:** CCR currently has 4 separate search surfaces (commits via `gcc_context(level=5, search_term=...)`, discussions, experiments, sessions). Users — especially new students — don't know which to call.

**New MCP tool** in `ccr/mcp/gcc_tools.py`:
```python
def gcc_search(
    query: str,
    sources: list[str] | None = None,   # ["commits", "discussions", "experiments", "sessions"]
                                          # default: all four
    limit: int = 10,
    date_range: list[str] | None = None,
) -> GccSearchResult
```

**Output:** Aggregated results grouped by source:
```
## Commits (2 matches)
[C041] LoRA fine-tuning with r=16...
[C053] Ablation: r=8 vs r=16...

## Experiments (1 match)
[exp-042] val_loss=0.23, accuracy=0.87

## Sessions (3 matches)
[2026-04-03] "We decided LoRA r=16 matches full FT at 12% params"
```

**Implementation:** Delegates to existing search functions in each subsystem. New `GccSearchResult` TypedDict with `matches: dict[str, list[dict]]` and `total: int`.

**Tests (3):**
1. Query matching only commits → returns commits, empty other sections
2. Query matching commits + sessions → returns both
3. Empty sources list → returns all four
4. No matches anywhere → `total=0`, graceful output

---

## Implementation Order

```
Phase 1 — BLOCKER (do first):
  A1  session marker fix        on_session_start.py + on_stop.py
  A2  empty project UX          on_session_start.py

Phase 2 — infrastructure:
  C1  session token tracking    state_accumulator.py + on_stop.py
  A6  pyproject.toml            pyproject.toml

Phase 3 — CLI (independent):
  A3  ccr doctor checks         cli.py
  A4  ccr uninstall             cli.py
  A5  ccr clean                 cli.py
  C2  ccr stats                 cli.py

Phase 4 — MCP tools (can be parallel):
  B1  gcc_experiments           memory_experiments.py + gcc_tools.py
  B3  gcc_discuss               memory_discussions.py + gcc_tools.py
  B4  Session Logger            sl/logger.py + sl_tools.py
  B2  context level heuristic   gcc_tools.py (1 line)

Phase 5 — content:
  C3  README rewrite            README.md
```

---

## Critical Files

| File | Changes |
|---|---|
| `ccr/hooks/on_session_start.py` | A1 PID check, A2 empty-project branch, C1 token count |
| `ccr/hooks/on_stop.py` | A1 marker deletion, C1 metrics write |
| `ccr/hooks/state_accumulator.py` | C1 add `context_tokens`, `session_id` fields |
| `ccr/cli.py` | A3 doctor checks, A4 uninstall, A5 clean, C2 stats |
| `ccr/core/memory_pkg/memory_experiments.py` | B1 new file |
| `ccr/core/memory_pkg/memory_discussions.py` | B3 new file |
| `ccr/sl/logger.py` | B4 new file |
| `ccr/mcp/sl_tools.py` | B4 new file |
| `ccr/mcp/gcc_tools.py` | B1 tool, B2 hint, B3 tools, B4 wiring |
| `ccr/mcp_types.py` | 5 new TypedDicts |
| `ccr/core/memory.py` | Add 2 new Mixins to MRO |
| `ccr/core/memory_pkg/__init__.py` | Export 2 new Mixins |
| `pyproject.toml` | A6 metadata |
| `README.md` | C3 rewrite |

---

## New Test Files

| File | Coverage |
|---|---|
| `tests/unit/test_session_marker.py` | A1 — 5 tests |
| `tests/unit/test_empty_project_ux.py` | A2 — 3 tests |
| `tests/unit/test_install_uninstall.py` | A4 — 4 tests |
| `tests/unit/test_ccr_clean.py` | A5 — 4 tests |
| `tests/unit/test_gcc_experiments.py` | B1 — 7 tests |
| `tests/unit/test_gcc_discuss.py` | B3 — 6 tests |
| `tests/unit/test_sl_tools.py` | B4 — 6 tests |
| `tests/unit/test_session_tracking.py` | C1 — 5 tests |
| `tests/unit/test_ccr_stats.py` | C2 — 5 tests |
| add to `test_mcp_server.py` | B2 context hint — 3 tests |

**Total new tests: ~48**

---

## Verification

After all streams ship:

```bash
# A1: Session marker works
ccr install && open Claude Code → ask "what do you know?" → close
→ reopen Claude Code → should load full memory context

# B1: Experiment query
gcc_commit(..., experiment={"id":"run-1","metrics":{"val_loss":0.2},...})
gcc_experiments(metric_filter={"val_loss":{"lt":0.3}})
→ returns run-1

# B4: Session search
session_log_turn(assistant_message="We decided to use LoRA...")
session_search(query="LoRA")
→ returns that turn

# C2: Stats dashboard
ccr stats
→ shows commits, tokens, projections

# Full test suite
pytest tests/unit/ tests/integration/ -x -q
→ 1873 + ~48 new tests pass
```

**Expected re-rating after all streams:** 9.0–9.5/10
- Setup: 5→9 (A1 fixes the critical blocker, A2-A5 polish the experience)
- "Claude forgot": 8→9.5 (A1 fix + B4 session logger = complete recall)
- Research tracking: 5→9.5 (B1+B3+B4 together)
- Token cost: 7→9 (C1+C2 give concrete numbers)
- Value prop: 5→9 (C2 dashboard + C3 README)
