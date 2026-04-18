# Phase 2: Playbook + Failure Lessons + Schema -> SQLite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize all ACE playbook data (bullets, failure lessons, schema, delta history, archived bullets) into SQLite tables for efficient CRUD, while retaining the Playbook class for complex analytics (GRPO, similarity, schema evolution).

**Architecture:** B3 hybrid — StorageBackend handles CRUD (insert/update/query/delete) via SQL. Playbook class loads from SQL for complex operations (find_similar, evolve_from_failures, schema evolution), then writes results back. All methods take a `scope` parameter routing to `memory.db` (project) or `global.db` (global).

**Tech Stack:** Python 3.12, SQLite WAL mode, pytest, existing `SqliteConnectionManager` from Phase 0.

**Spec:** `docs/superpowers/specs/2026-04-17-phase2-playbook-sqlite-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `ccr/core/storage/base.py` | Modify | +18 abstract methods (Phase 2 section) |
| `ccr/core/storage/sqlite_backend.py` | Modify | Phase 2 DDL + SQL implementations + `_get_scoped_conn()` |
| `ccr/core/storage/file_backend.py` | Modify | File-based implementations (full load/save per call) |
| `ccr/core/storage/migration.py` | Modify | `migrate_phase_2()` + `migrate_phase_2_global()` |
| `ccr/ace/playbook.py` | Modify | `from_backend()` classmethod + `save_to_backend()` |
| `tests/unit/test_sqlite_phase1.py` | Rename | -> `test_sqlite_storage.py`, add Phase 2 tests |
| `tests/unit/test_storage_backend.py` | Modify | Add Phase 2 dual-backend parity tests |

---

### Task 1: Add Phase 2 Abstract Methods to StorageBackend ABC

**Files:**
- Modify: `ccr/core/storage/base.py:93` (after Phase 1 Metadata section, before Lifecycle)

- [ ] **Step 1: Add 18 abstract methods to base.py**

Add a new Phase 2 section after the Phase 1 Metadata section (line 93) and before the Lifecycle section:

```python
    # ── Phase 2: Playbook Bullets ───────────────────────────────

    @abstractmethod
    def bullet_get(self, bullet_id: str, scope: str = "project") -> dict | None:
        """Get a single bullet by ID. Returns dict with all fields or None."""

    @abstractmethod
    def bullet_list(self, section: str | None = None, scope: str = "project") -> list[dict]:
        """List all bullets, optionally filtered by section."""

    @abstractmethod
    def bullet_insert(self, bullet: dict, scope: str = "project") -> None:
        """Insert a new bullet. Dict must have 'id', 'section', 'content', 'created_at'."""

    @abstractmethod
    def bullet_update(self, bullet_id: str, updates: dict, scope: str = "project") -> bool:
        """Update bullet fields. Returns True if bullet existed."""

    @abstractmethod
    def bullet_delete(self, bullet_id: str, scope: str = "project") -> bool:
        """Delete a bullet by ID. Returns True if existed."""

    @abstractmethod
    def bullet_update_counters(self, bullet_tags: list[dict], scope: str = "project") -> int:
        """Increment helpful/harmful counters. Returns count of updated bullets."""

    @abstractmethod
    def bullet_get_next_id(self, scope: str = "project") -> int:
        """Return the next available bullet ID number."""

    # ── Phase 2: Failure Lessons ────────────────────────────────

    @abstractmethod
    def failure_lessons_for_bullet(self, bullet_id: str, scope: str = "project") -> list[dict]:
        """Get all failure lessons for a bullet."""

    @abstractmethod
    def failure_lessons_insert(self, bullet_id: str, lesson: dict, scope: str = "project") -> None:
        """Insert a failure lesson for a bullet."""

    @abstractmethod
    def failure_lessons_mark_evolved(self, bullet_id: str, scope: str = "project") -> int:
        """Mark all lessons for a bullet as evolved. Returns count updated."""

    @abstractmethod
    def failure_lessons_all(self, scope: str = "project") -> dict[str, list[dict]]:
        """Get all failure lessons grouped by bullet_id."""

    # ── Phase 2: Playbook Sections ──────────────────────────────

    @abstractmethod
    def playbook_sections_get(self, scope: str = "project") -> list[str]:
        """Get ordered section names."""

    @abstractmethod
    def playbook_sections_set(self, sections: list[str], scope: str = "project") -> None:
        """Set ordered section names (replaces all)."""

    # ── Phase 2: Playbook Schema ────────────────────────────────

    @abstractmethod
    def playbook_schema_load(self, scope: str = "project") -> dict:
        """Load the current playbook schema as dict."""

    @abstractmethod
    def playbook_schema_save(self, schema: dict, history_entry: dict | None = None, scope: str = "project") -> None:
        """Save schema. If history_entry provided, append to version history."""

    @abstractmethod
    def playbook_schema_history(self, scope: str = "project") -> list[dict]:
        """Get schema version history."""

    # ── Phase 2: Audit ──────────────────────────────────────────

    @abstractmethod
    def delta_history_append(self, entry: dict, scope: str = "project") -> None:
        """Append a delta operation audit entry."""

    @abstractmethod
    def archived_bullets_insert(self, bullets: list[dict], reason: str, scope: str = "project") -> int:
        """Archive pruned bullets. Returns count archived."""
```

- [ ] **Step 2: Verify it compiles**

Run: `cd /Users/qbit-glitch/Desktop/coding-projects/powering_claude_with_less_tokens && .venv/bin/python -c "from ccr.core.storage.base import StorageBackend; print('OK')"`

Expected: `OK` (abstract class itself can be imported; concrete classes will fail until implemented)

- [ ] **Step 3: Commit**

```bash
git add ccr/core/storage/base.py
git commit -m "feat(storage): add 18 Phase 2 abstract methods to StorageBackend ABC"
```

---

### Task 2: Implement Phase 2 DDL + SQL Backend (Bullets + Sections)

**Files:**
- Modify: `ccr/core/storage/sqlite_backend.py`
- Test: `tests/unit/test_sqlite_phase1.py` (will be renamed in Task 6)

- [ ] **Step 1: Add Phase 2 DDL to sqlite_backend.py**

After the `_PHASE1_TABLES` constant (line 60), add:

```python
_PHASE2_TABLES = """
CREATE TABLE IF NOT EXISTS playbook_bullets (
    id                  TEXT PRIMARY KEY,
    section             TEXT NOT NULL,
    content             TEXT NOT NULL,
    helpful             INTEGER NOT NULL DEFAULT 0,
    harmful             INTEGER NOT NULL DEFAULT 0,
    scope               TEXT NOT NULL DEFAULT 'general',
    when_to_apply       TEXT NOT NULL DEFAULT '',
    trigger_text        TEXT NOT NULL DEFAULT '',
    action              TEXT NOT NULL DEFAULT '',
    weighted_helpful    REAL NOT NULL DEFAULT 0.0,
    weighted_harmful    REAL NOT NULL DEFAULT 0.0,
    personal_decay_rate REAL NOT NULL DEFAULT 0.0,
    grpo_advantage      REAL NOT NULL DEFAULT 0.0,
    last_updated        TEXT,
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS failure_lessons (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    bullet_id            TEXT NOT NULL REFERENCES playbook_bullets(id) ON DELETE CASCADE,
    failure_point        TEXT NOT NULL,
    flawed_reasoning     TEXT NOT NULL,
    counterfactual       TEXT NOT NULL,
    prevention_principle TEXT NOT NULL,
    task_context         TEXT NOT NULL DEFAULT '',
    timestamp            TEXT,
    evolved              INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_fl_bullet ON failure_lessons(bullet_id);

CREATE TABLE IF NOT EXISTS playbook_sections (
    position    INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS playbook_schema (
    version            INTEGER PRIMARY KEY,
    data_json          TEXT NOT NULL,
    parent_version     INTEGER,
    change_description TEXT,
    created_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS delta_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    author          TEXT NOT NULL DEFAULT '',
    ops_count       INTEGER NOT NULL DEFAULT 0,
    applied_count   INTEGER NOT NULL DEFAULT 0,
    scope           TEXT NOT NULL DEFAULT 'project',
    operations_json TEXT,
    failed_ids_json TEXT
);

CREATE TABLE IF NOT EXISTS archived_bullets (
    id          TEXT NOT NULL,
    section     TEXT,
    content     TEXT,
    helpful     INTEGER,
    harmful     INTEGER,
    archived_at TEXT NOT NULL,
    reason      TEXT NOT NULL DEFAULT ''
);
"""
```

- [ ] **Step 2: Add `_ensure_phase2_tables` and `_get_scoped_conn`**

In `SqliteStorageBackend.__init__`, call `_ensure_phase2_tables` after phase 1. Add the scope routing helper:

```python
def _ensure_phase2_tables(self) -> None:
    self.memory_conn.executescript(_PHASE2_TABLES)
    if self._global_mgr:
        self.global_conn.executescript(_PHASE2_TABLES)

def _get_scoped_conn(self, scope: str) -> sqlite3.Connection:
    if scope == "global":
        if self.global_conn is None:
            raise ValueError("No global database configured")
        return self.global_conn
    return self.memory_conn
```

In `__init__`, add after `self._ensure_phase1_tables()`:
```python
self._ensure_phase2_tables()
```

- [ ] **Step 3: Implement bullet CRUD methods**

Add to `SqliteStorageBackend` class:

```python
# ── Playbook Bullets ───────────────────────────────────────

def bullet_get(self, bullet_id: str, scope: str = "project") -> dict | None:
    conn = self._get_scoped_conn(scope)
    row = conn.execute(
        "SELECT * FROM playbook_bullets WHERE id = ?", (bullet_id,),
    ).fetchone()
    if row is None:
        return None
    return dict(row)

def bullet_list(self, section: str | None = None, scope: str = "project") -> list[dict]:
    conn = self._get_scoped_conn(scope)
    if section is not None:
        rows = conn.execute(
            "SELECT * FROM playbook_bullets WHERE section = ?", (section,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM playbook_bullets").fetchall()
    return [dict(r) for r in rows]

def bullet_insert(self, bullet: dict, scope: str = "project") -> None:
    conn = self._get_scoped_conn(scope)
    conn.execute(
        """INSERT INTO playbook_bullets
           (id, section, content, helpful, harmful, scope, when_to_apply,
            trigger_text, action, weighted_helpful, weighted_harmful,
            personal_decay_rate, grpo_advantage, last_updated, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            bullet["id"], bullet["section"], bullet["content"],
            bullet.get("helpful", 0), bullet.get("harmful", 0),
            bullet.get("scope", "general"), bullet.get("when_to_apply", ""),
            bullet.get("trigger_text", ""), bullet.get("action", ""),
            bullet.get("weighted_helpful", 0.0), bullet.get("weighted_harmful", 0.0),
            bullet.get("personal_decay_rate", 0.0), bullet.get("grpo_advantage", 0.0),
            bullet.get("last_updated"), bullet.get("created_at", _utcnow()),
        ),
    )
    conn.commit()

def bullet_update(self, bullet_id: str, updates: dict, scope: str = "project") -> bool:
    conn = self._get_scoped_conn(scope)
    if not updates:
        return False
    set_parts = []
    values = []
    for key, val in updates.items():
        set_parts.append(f"{key} = ?")
        values.append(val)
    values.append(bullet_id)
    conn.execute(
        f"UPDATE playbook_bullets SET {', '.join(set_parts)} WHERE id = ?",
        values,
    )
    changed = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    return changed > 0

def bullet_delete(self, bullet_id: str, scope: str = "project") -> bool:
    conn = self._get_scoped_conn(scope)
    conn.execute("DELETE FROM playbook_bullets WHERE id = ?", (bullet_id,))
    changed = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    return changed > 0

def bullet_update_counters(self, bullet_tags: list[dict], scope: str = "project") -> int:
    conn = self._get_scoped_conn(scope)
    updated = 0
    now = _utcnow()
    for tag in bullet_tags:
        bid = tag.get("id") or tag.get("bullet", "")
        if not bid:
            continue
        raw_weight = tag.get("weight", 1.0)
        try:
            weight = max(0.0, min(1.0, float(raw_weight)))
        except (TypeError, ValueError):
            weight = 1.0
        tag_val = tag.get("tag", "neutral")
        if tag_val == "helpful":
            conn.execute(
                """UPDATE playbook_bullets SET
                    helpful = helpful + 1,
                    weighted_helpful = weighted_helpful + ?,
                    last_updated = ?,
                    personal_decay_rate = max(0.90, min(0.99,
                        0.95 + (helpful + 1 - harmful) * 0.002))
                    WHERE id = ?""",
                (weight, now, bid),
            )
        elif tag_val == "harmful":
            conn.execute(
                """UPDATE playbook_bullets SET
                    harmful = harmful + 1,
                    weighted_harmful = weighted_harmful + ?,
                    last_updated = ?,
                    personal_decay_rate = max(0.90, min(0.99,
                        0.95 + (helpful - harmful - 1) * 0.002))
                    WHERE id = ?""",
                (weight, now, bid),
            )
            lesson = tag.get("failure_lesson")
            if isinstance(lesson, dict) and lesson:
                self.failure_lessons_insert(bid, lesson, scope)
        else:
            continue
        if conn.execute("SELECT changes()").fetchone()[0] > 0:
            updated += 1
    conn.commit()
    return updated

def bullet_get_next_id(self, scope: str = "project") -> int:
    conn = self._get_scoped_conn(scope)
    row = conn.execute(
        """SELECT MAX(CAST(SUBSTR(id, INSTR(id, '-') + 1) AS INTEGER))
           FROM playbook_bullets""",
    ).fetchone()
    max_num = row[0] if row[0] is not None else 0
    return max_num + 1
```

- [ ] **Step 4: Implement playbook_sections methods**

```python
# ── Playbook Sections ──────────────────────────────────────

def playbook_sections_get(self, scope: str = "project") -> list[str]:
    conn = self._get_scoped_conn(scope)
    rows = conn.execute(
        "SELECT name FROM playbook_sections ORDER BY position",
    ).fetchall()
    return [r["name"] for r in rows]

def playbook_sections_set(self, sections: list[str], scope: str = "project") -> None:
    conn = self._get_scoped_conn(scope)
    conn.execute("DELETE FROM playbook_sections")
    for i, name in enumerate(sections):
        conn.execute(
            "INSERT INTO playbook_sections (position, name) VALUES (?, ?)",
            (i, name),
        )
    conn.commit()
```

- [ ] **Step 5: Write tests for bullet CRUD and sections**

Create tests in `tests/unit/test_sqlite_phase1.py` (we'll rename in Task 6):

```python
class TestSqliteBullets:
    def test_insert_and_get(self, sqlite_backend):
        bullet = {
            "id": "str-00001", "section": "STRATEGIES & INSIGHTS",
            "content": "Test strategy", "created_at": "2026-04-17T00:00:00+00:00",
        }
        sqlite_backend.bullet_insert(bullet)
        got = sqlite_backend.bullet_get("str-00001")
        assert got is not None
        assert got["content"] == "Test strategy"
        assert got["helpful"] == 0

    def test_get_missing(self, sqlite_backend):
        assert sqlite_backend.bullet_get("nope") is None

    def test_list_all(self, sqlite_backend):
        for i in range(3):
            sqlite_backend.bullet_insert({
                "id": f"str-{i:05d}", "section": "STRATEGIES & INSIGHTS",
                "content": f"Strategy {i}", "created_at": "2026-04-17T00:00:00+00:00",
            })
        assert len(sqlite_backend.bullet_list()) == 3

    def test_list_by_section(self, sqlite_backend):
        sqlite_backend.bullet_insert({
            "id": "str-00001", "section": "STRATEGIES & INSIGHTS",
            "content": "A", "created_at": "2026-04-17T00:00:00+00:00",
        })
        sqlite_backend.bullet_insert({
            "id": "heu-00001", "section": "PROBLEM-SOLVING HEURISTICS",
            "content": "B", "created_at": "2026-04-17T00:00:00+00:00",
        })
        result = sqlite_backend.bullet_list(section="STRATEGIES & INSIGHTS")
        assert len(result) == 1
        assert result[0]["id"] == "str-00001"

    def test_update(self, sqlite_backend):
        sqlite_backend.bullet_insert({
            "id": "str-00001", "section": "STRATEGIES & INSIGHTS",
            "content": "Old", "created_at": "2026-04-17T00:00:00+00:00",
        })
        assert sqlite_backend.bullet_update("str-00001", {"content": "New"})
        assert sqlite_backend.bullet_get("str-00001")["content"] == "New"

    def test_update_missing(self, sqlite_backend):
        assert sqlite_backend.bullet_update("nope", {"content": "X"}) is False

    def test_delete(self, sqlite_backend):
        sqlite_backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        assert sqlite_backend.bullet_delete("str-00001") is True
        assert sqlite_backend.bullet_delete("str-00001") is False
        assert sqlite_backend.bullet_get("str-00001") is None

    def test_update_counters_helpful(self, sqlite_backend):
        sqlite_backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        updated = sqlite_backend.bullet_update_counters([
            {"id": "str-00001", "tag": "helpful", "weight": 0.8},
        ])
        assert updated == 1
        b = sqlite_backend.bullet_get("str-00001")
        assert b["helpful"] == 1
        assert abs(b["weighted_helpful"] - 0.8) < 0.001

    def test_update_counters_harmful_with_lesson(self, sqlite_backend):
        sqlite_backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        updated = sqlite_backend.bullet_update_counters([{
            "id": "str-00001", "tag": "harmful",
            "failure_lesson": {
                "failure_point": "broke here",
                "flawed_reasoning": "wrong assumption",
                "counterfactual": "should have done X",
                "prevention_principle": "always do X",
            },
        }])
        assert updated == 1
        b = sqlite_backend.bullet_get("str-00001")
        assert b["harmful"] == 1
        lessons = sqlite_backend.failure_lessons_for_bullet("str-00001")
        assert len(lessons) == 1
        assert lessons[0]["failure_point"] == "broke here"

    def test_update_counters_missing_id(self, sqlite_backend):
        assert sqlite_backend.bullet_update_counters([
            {"id": "nope", "tag": "helpful"},
        ]) == 0

    def test_get_next_id_empty(self, sqlite_backend):
        assert sqlite_backend.bullet_get_next_id() == 1

    def test_get_next_id(self, sqlite_backend):
        sqlite_backend.bullet_insert({
            "id": "str-00042", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        assert sqlite_backend.bullet_get_next_id() == 43


class TestSqliteSections:
    def test_set_and_get(self, sqlite_backend):
        sections = ["STRATEGIES & INSIGHTS", "OTHERS"]
        sqlite_backend.playbook_sections_set(sections)
        got = sqlite_backend.playbook_sections_get()
        assert got == sections

    def test_preserves_order(self, sqlite_backend):
        sections = ["C", "A", "B"]
        sqlite_backend.playbook_sections_set(sections)
        assert sqlite_backend.playbook_sections_get() == ["C", "A", "B"]

    def test_empty(self, sqlite_backend):
        assert sqlite_backend.playbook_sections_get() == []

    def test_replace(self, sqlite_backend):
        sqlite_backend.playbook_sections_set(["A", "B"])
        sqlite_backend.playbook_sections_set(["X", "Y", "Z"])
        assert sqlite_backend.playbook_sections_get() == ["X", "Y", "Z"]
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/test_sqlite_phase1.py::TestSqliteBullets tests/unit/test_sqlite_phase1.py::TestSqliteSections -v`

Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add ccr/core/storage/sqlite_backend.py tests/unit/test_sqlite_phase1.py
git commit -m "feat(storage): implement Phase 2 bullet CRUD + sections in SQLite backend"
```

---

### Task 3: Implement Phase 2 SQL Backend (Failure Lessons + Schema + Audit)

**Files:**
- Modify: `ccr/core/storage/sqlite_backend.py`
- Test: `tests/unit/test_sqlite_phase1.py`

- [ ] **Step 1: Implement failure_lessons methods**

```python
# ── Failure Lessons ────────────────────────────────────────

def failure_lessons_for_bullet(self, bullet_id: str, scope: str = "project") -> list[dict]:
    conn = self._get_scoped_conn(scope)
    rows = conn.execute(
        "SELECT * FROM failure_lessons WHERE bullet_id = ?", (bullet_id,),
    ).fetchall()
    return [dict(r) for r in rows]

def failure_lessons_insert(self, bullet_id: str, lesson: dict, scope: str = "project") -> None:
    conn = self._get_scoped_conn(scope)
    conn.execute(
        """INSERT INTO failure_lessons
           (bullet_id, failure_point, flawed_reasoning, counterfactual,
            prevention_principle, task_context, timestamp, evolved)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            bullet_id,
            lesson.get("failure_point", ""),
            lesson.get("flawed_reasoning", ""),
            lesson.get("counterfactual", ""),
            lesson.get("prevention_principle", ""),
            lesson.get("task_context", ""),
            lesson.get("timestamp", _utcnow()),
            int(lesson.get("evolved", False)),
        ),
    )
    conn.commit()

def failure_lessons_mark_evolved(self, bullet_id: str, scope: str = "project") -> int:
    conn = self._get_scoped_conn(scope)
    conn.execute(
        "UPDATE failure_lessons SET evolved = 1 WHERE bullet_id = ? AND evolved = 0",
        (bullet_id,),
    )
    changed = conn.execute("SELECT changes()").fetchone()[0]
    conn.commit()
    return changed

def failure_lessons_all(self, scope: str = "project") -> dict[str, list[dict]]:
    conn = self._get_scoped_conn(scope)
    rows = conn.execute("SELECT * FROM failure_lessons ORDER BY bullet_id").fetchall()
    result: dict[str, list[dict]] = {}
    for r in rows:
        bid = r["bullet_id"]
        if bid not in result:
            result[bid] = []
        result[bid].append(dict(r))
    return result
```

- [ ] **Step 2: Implement schema methods**

```python
# ── Playbook Schema ────────────────────────────────────────

def playbook_schema_load(self, scope: str = "project") -> dict:
    conn = self._get_scoped_conn(scope)
    row = conn.execute(
        "SELECT data_json FROM playbook_schema ORDER BY version DESC LIMIT 1",
    ).fetchone()
    if row is None:
        return {}
    try:
        return json.loads(row["data_json"])
    except json.JSONDecodeError:
        return {}

def playbook_schema_save(self, schema: dict, history_entry: dict | None = None, scope: str = "project") -> None:
    conn = self._get_scoped_conn(scope)
    version = schema.get("version", 1)
    conn.execute(
        """INSERT OR REPLACE INTO playbook_schema
           (version, data_json, parent_version, change_description, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (
            version,
            json.dumps(schema, default=str),
            schema.get("parent_version"),
            schema.get("change_description", ""),
            schema.get("created_at", _utcnow()),
        ),
    )
    if history_entry:
        h_version = history_entry.get("version", version - 1)
        conn.execute(
            """INSERT OR IGNORE INTO playbook_schema
               (version, data_json, parent_version, change_description, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                h_version,
                json.dumps(history_entry, default=str),
                history_entry.get("parent_version"),
                history_entry.get("change_description", ""),
                history_entry.get("created_at", _utcnow()),
            ),
        )
    conn.commit()

def playbook_schema_history(self, scope: str = "project") -> list[dict]:
    conn = self._get_scoped_conn(scope)
    rows = conn.execute(
        "SELECT * FROM playbook_schema ORDER BY version",
    ).fetchall()
    result = []
    for r in rows:
        entry = dict(r)
        try:
            entry["data"] = json.loads(entry.pop("data_json"))
        except (json.JSONDecodeError, KeyError):
            pass
        result.append(entry)
    return result
```

- [ ] **Step 3: Implement audit methods**

```python
# ── Audit ──────────────────────────────────────────────────

def delta_history_append(self, entry: dict, scope: str = "project") -> None:
    conn = self._get_scoped_conn(scope)
    conn.execute(
        """INSERT INTO delta_history
           (timestamp, author, ops_count, applied_count, scope,
            operations_json, failed_ids_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            entry.get("timestamp", _utcnow()),
            entry.get("author", ""),
            entry.get("ops_count", 0),
            entry.get("applied_count", 0),
            entry.get("scope", scope),
            json.dumps(entry.get("operations", []), default=str) if entry.get("operations") else None,
            json.dumps(entry.get("failed_ids", []), default=str) if entry.get("failed_ids") else None,
        ),
    )
    conn.commit()

def archived_bullets_insert(self, bullets: list[dict], reason: str, scope: str = "project") -> int:
    conn = self._get_scoped_conn(scope)
    now = _utcnow()
    for b in bullets:
        conn.execute(
            """INSERT INTO archived_bullets
               (id, section, content, helpful, harmful, archived_at, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                b.get("id", ""),
                b.get("section", ""),
                b.get("content", ""),
                b.get("helpful", 0),
                b.get("harmful", 0),
                now,
                reason,
            ),
        )
    conn.commit()
    return len(bullets)
```

- [ ] **Step 4: Write tests for failure lessons, schema, and audit**

```python
class TestSqliteFailureLessons:
    def test_insert_and_get(self, sqlite_backend):
        sqlite_backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        sqlite_backend.failure_lessons_insert("str-00001", {
            "failure_point": "broke",
            "flawed_reasoning": "wrong",
            "counterfactual": "do X",
            "prevention_principle": "always X",
        })
        lessons = sqlite_backend.failure_lessons_for_bullet("str-00001")
        assert len(lessons) == 1
        assert lessons[0]["failure_point"] == "broke"
        assert lessons[0]["evolved"] == 0

    def test_mark_evolved(self, sqlite_backend):
        sqlite_backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        sqlite_backend.failure_lessons_insert("str-00001", {
            "failure_point": "broke", "flawed_reasoning": "wrong",
            "counterfactual": "do X", "prevention_principle": "always X",
        })
        count = sqlite_backend.failure_lessons_mark_evolved("str-00001")
        assert count == 1
        lessons = sqlite_backend.failure_lessons_for_bullet("str-00001")
        assert lessons[0]["evolved"] == 1

    def test_all_grouped(self, sqlite_backend):
        for bid in ["str-00001", "str-00002"]:
            sqlite_backend.bullet_insert({
                "id": bid, "section": "S", "content": "X",
                "created_at": "2026-04-17T00:00:00+00:00",
            })
            sqlite_backend.failure_lessons_insert(bid, {
                "failure_point": f"broke-{bid}", "flawed_reasoning": "wrong",
                "counterfactual": "do X", "prevention_principle": "always X",
            })
        grouped = sqlite_backend.failure_lessons_all()
        assert len(grouped) == 2
        assert "str-00001" in grouped
        assert "str-00002" in grouped

    def test_cascade_delete(self, sqlite_backend):
        sqlite_backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        sqlite_backend.failure_lessons_insert("str-00001", {
            "failure_point": "broke", "flawed_reasoning": "wrong",
            "counterfactual": "do X", "prevention_principle": "always X",
        })
        sqlite_backend.bullet_delete("str-00001")
        assert sqlite_backend.failure_lessons_for_bullet("str-00001") == []


class TestSqlitePlaybookSchema:
    def test_save_and_load(self, sqlite_backend):
        schema = {"version": 1, "decay_rate": 0.95, "token_budget": 80000}
        sqlite_backend.playbook_schema_save(schema)
        loaded = sqlite_backend.playbook_schema_load()
        assert loaded["decay_rate"] == 0.95

    def test_load_empty(self, sqlite_backend):
        assert sqlite_backend.playbook_schema_load() == {}

    def test_history(self, sqlite_backend):
        sqlite_backend.playbook_schema_save({"version": 1, "decay_rate": 0.95})
        sqlite_backend.playbook_schema_save({"version": 2, "decay_rate": 0.90, "parent_version": 1})
        history = sqlite_backend.playbook_schema_history()
        assert len(history) == 2
        assert history[0]["version"] == 1
        assert history[1]["version"] == 2


class TestSqliteAudit:
    def test_delta_history_append(self, sqlite_backend):
        sqlite_backend.delta_history_append({
            "author": "claude", "ops_count": 3, "applied_count": 2,
            "scope": "project", "failed_ids": ["str-00099"],
        })
        conn = sqlite_backend.memory_conn
        row = conn.execute("SELECT * FROM delta_history").fetchone()
        assert row["author"] == "claude"
        assert row["ops_count"] == 3

    def test_archived_bullets_insert(self, sqlite_backend):
        count = sqlite_backend.archived_bullets_insert([
            {"id": "str-00001", "section": "S", "content": "X", "helpful": 1, "harmful": 5},
            {"id": "str-00002", "section": "S", "content": "Y", "helpful": 0, "harmful": 3},
        ], reason="pruned")
        assert count == 2
        rows = sqlite_backend.memory_conn.execute("SELECT * FROM archived_bullets").fetchall()
        assert len(rows) == 2
        assert rows[0]["reason"] == "pruned"
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/test_sqlite_phase1.py::TestSqliteFailureLessons tests/unit/test_sqlite_phase1.py::TestSqlitePlaybookSchema tests/unit/test_sqlite_phase1.py::TestSqliteAudit -v`

Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add ccr/core/storage/sqlite_backend.py tests/unit/test_sqlite_phase1.py
git commit -m "feat(storage): implement Phase 2 failure lessons, schema, and audit in SQLite backend"
```

---

### Task 4: Implement Phase 2 File Backend

**Files:**
- Modify: `ccr/core/storage/file_backend.py`
- Test: `tests/unit/test_storage_backend.py`

- [ ] **Step 1: Implement all 18 Phase 2 methods in FileStorageBackend**

The file backend does full load/save on each call (inefficient but correct for the legacy path). All methods follow the same pattern: load playbook.txt + failure_lessons.json, operate, save.

Add these imports at top of `file_backend.py`:
```python
import re
```

Add all 18 methods. Key pattern: `_load_playbook_data()` returns `(bullets_list, sections_list, next_id)` by parsing playbook.txt with `_BULLET_RE`. `_save_playbook_data()` writes back.

The full implementation should wrap the existing Playbook parse/serialize for bullets, and use JSON files for schema/history/archived. Each method acquires `self._lock`.

This is the longest implementation (~200 lines) but all methods are straightforward file I/O wrappers.

- [ ] **Step 2: Write dual-backend parity tests in test_storage_backend.py**

Add a `backend` fixture parameterized over `["files", "sqlite"]` and test all 18 methods produce identical results on both backends:

```python
class TestPhase2DualBackend:
    def test_bullet_insert_get(self, backend):
        backend.bullet_insert({
            "id": "str-00001", "section": "STRATEGIES & INSIGHTS",
            "content": "Test", "created_at": "2026-04-17T00:00:00+00:00",
        })
        got = backend.bullet_get("str-00001")
        assert got is not None
        assert got["content"] == "Test"

    def test_bullet_update_counters(self, backend):
        backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        updated = backend.bullet_update_counters([
            {"id": "str-00001", "tag": "helpful"},
        ])
        assert updated == 1
        b = backend.bullet_get("str-00001")
        assert b["helpful"] == 1

    def test_sections_roundtrip(self, backend):
        backend.playbook_sections_set(["A", "B", "C"])
        assert backend.playbook_sections_get() == ["A", "B", "C"]

    def test_failure_lessons(self, backend):
        backend.bullet_insert({
            "id": "str-00001", "section": "S", "content": "X",
            "created_at": "2026-04-17T00:00:00+00:00",
        })
        backend.failure_lessons_insert("str-00001", {
            "failure_point": "broke", "flawed_reasoning": "wrong",
            "counterfactual": "X", "prevention_principle": "always X",
        })
        lessons = backend.failure_lessons_for_bullet("str-00001")
        assert len(lessons) == 1

    def test_schema_roundtrip(self, backend):
        backend.playbook_schema_save({"version": 1, "decay_rate": 0.95})
        loaded = backend.playbook_schema_load()
        assert loaded["version"] == 1

    def test_delta_history(self, backend):
        backend.delta_history_append({"author": "test", "ops_count": 1})
        # Just verify no exception; audit is append-only

    def test_archived_bullets(self, backend):
        count = backend.archived_bullets_insert([
            {"id": "str-00001", "content": "X", "helpful": 0, "harmful": 5},
        ], reason="pruned")
        assert count == 1
```

- [ ] **Step 3: Run all storage tests**

Run: `.venv/bin/python -m pytest tests/unit/test_storage_backend.py tests/unit/test_sqlite_phase1.py -v`

Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add ccr/core/storage/file_backend.py tests/unit/test_storage_backend.py
git commit -m "feat(storage): implement Phase 2 methods in FileStorageBackend + dual-backend parity tests"
```

---

### Task 5: Add Playbook.from_backend() and save_to_backend()

**Files:**
- Modify: `ccr/ace/playbook.py`
- Test: `tests/unit/test_playbook.py`

- [ ] **Step 1: Add from_backend classmethod**

Add to the `Playbook` class in `ccr/ace/playbook.py`, after `__init__`:

```python
@classmethod
def from_backend(cls, backend, scope: str = "project") -> Playbook:
    """Load a Playbook from a StorageBackend (for complex analytics operations)."""
    sections = backend.playbook_sections_get(scope)
    bullets_data = backend.bullet_list(scope=scope)

    pb = cls.__new__(cls)
    pb._sections = sections or list(DEFAULT_SECTIONS)
    pb._bullets = []
    pb._id_index = {}
    pb._next_id = backend.bullet_get_next_id(scope)

    for bd in bullets_data:
        bullet = Bullet(
            id=bd["id"],
            helpful=bd.get("helpful", 0),
            harmful=bd.get("harmful", 0),
            content=bd.get("content", ""),
            section=bd.get("section", ""),
            scope=bd.get("scope", "general"),
            when_to_apply=bd.get("when_to_apply", ""),
            last_updated=bd.get("last_updated", ""),
            grpo_advantage=bd.get("grpo_advantage", 0.0),
            trigger=bd.get("trigger_text", ""),
            action=bd.get("action", ""),
            weighted_helpful=bd.get("weighted_helpful", 0.0),
            weighted_harmful=bd.get("weighted_harmful", 0.0),
            personal_decay_rate=bd.get("personal_decay_rate", 0.0),
        )
        pb._bullets.append(bullet)
        pb._id_index[bullet.id] = bullet

    all_lessons = backend.failure_lessons_all(scope)
    for bid, lessons in all_lessons.items():
        if bid in pb._id_index:
            pb._id_index[bid].failure_lessons = [
                FailureLesson(
                    failure_point=l.get("failure_point", ""),
                    flawed_reasoning=l.get("flawed_reasoning", ""),
                    counterfactual=l.get("counterfactual", ""),
                    prevention_principle=l.get("prevention_principle", ""),
                    task_context=l.get("task_context", ""),
                    timestamp=l.get("timestamp", ""),
                    evolved=bool(l.get("evolved", False)),
                )
                for l in lessons
            ]
    return pb

def save_to_backend(self, backend, scope: str = "project") -> None:
    """Save the current Playbook state back to a StorageBackend."""
    backend.playbook_sections_set(self._sections, scope)

    existing_ids = {b["id"] for b in backend.bullet_list(scope=scope)}
    current_ids = {b.id for b in self._bullets}

    for bid in existing_ids - current_ids:
        backend.bullet_delete(bid, scope)

    for bullet in self._bullets:
        bd = {
            "id": bullet.id, "section": bullet.section,
            "content": bullet.content, "helpful": bullet.helpful,
            "harmful": bullet.harmful, "scope": bullet.scope,
            "when_to_apply": bullet.when_to_apply,
            "trigger_text": bullet.trigger, "action": bullet.action,
            "weighted_helpful": bullet.weighted_helpful,
            "weighted_harmful": bullet.weighted_harmful,
            "personal_decay_rate": bullet.personal_decay_rate,
            "grpo_advantage": bullet.grpo_advantage,
            "last_updated": bullet.last_updated,
            "created_at": bullet.last_updated or _utcnow(),
        }
        if bullet.id in existing_ids:
            backend.bullet_update(bullet.id, bd, scope)
        else:
            backend.bullet_insert(bd, scope)

        for fl in bullet.failure_lessons:
            backend.failure_lessons_insert(bullet.id, fl.to_dict(), scope)
```

Add `from ccr.core.storage.sqlite_backend import _utcnow` at the top of the file (or inline the helper).

- [ ] **Step 2: Write roundtrip test**

In `tests/unit/test_playbook.py`, add:

```python
class TestPlaybookBackendRoundtrip:
    def test_from_backend_roundtrip(self, tmp_path):
        from ccr.core.storage.sqlite_backend import SqliteStorageBackend
        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        backend = SqliteStorageBackend(str(ccr))

        # Create playbook with bullets
        pb = Playbook("## STRATEGIES & INSIGHTS\n[str-00001] helpful=3 harmful=1 :: Test strategy")
        pb.save_to_backend(backend)

        # Reload and verify
        pb2 = Playbook.from_backend(backend)
        assert len(pb2.bullets) == 1
        assert pb2.bullets[0].helpful == 3
        assert pb2.bullets[0].content == "Test strategy"
        assert pb2.sections == pb.sections
        backend.close()
```

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/test_playbook.py::TestPlaybookBackendRoundtrip -v`

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add ccr/ace/playbook.py tests/unit/test_playbook.py
git commit -m "feat(playbook): add from_backend() and save_to_backend() for SQL round-trip"
```

---

### Task 6: Implement Migration Phase 2

**Files:**
- Modify: `ccr/core/storage/migration.py`
- Rename: `tests/unit/test_sqlite_phase1.py` -> `tests/unit/test_sqlite_storage.py`

- [ ] **Step 1: Add migrate_phase_2 to migration.py**

```python
def migrate_phase_2(ccr_root: str, db_path: str) -> dict[str, Any]:
    """Migrate playbook flat files to SQLite tables."""
    result: dict[str, Any] = {"migrated": 0, "errors": []}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("BEGIN")
        result["migrated"] += _migrate_playbook_txt(ccr_root, conn)
        result["migrated"] += _migrate_failure_lessons_json(ccr_root, conn)
        result["migrated"] += _migrate_playbook_schema_json(ccr_root, conn)
        result["migrated"] += _migrate_playbook_history_json(ccr_root, conn)
        result["migrated"] += _migrate_archived_bullets_json(ccr_root, conn)
        conn.commit()
        logger.info("Phase 2 migration: %d records migrated", result["migrated"])
    except Exception as exc:
        conn.rollback()
        result["errors"].append(f"Phase 2 failed: {exc}")
        logger.error("Phase 2 migration failed: %s", exc)
    finally:
        conn.close()
    return result
```

Add the 5 sub-functions: `_migrate_playbook_txt`, `_migrate_failure_lessons_json`, `_migrate_playbook_schema_json`, `_migrate_playbook_history_json`, `_migrate_archived_bullets_json`.

Key parsing logic for `_migrate_playbook_txt`:
```python
import re
_BULLET_RE = re.compile(r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)")
```

Parse sections from `## SECTION` headers and bullets from `_BULLET_RE` matches. INSERT into `playbook_bullets` and `playbook_sections`.

- [ ] **Step 2: Wire phase 2 into auto_migrate**

In `auto_migrate`, after phase 1 block:
```python
p2 = migrate_phase_2(ccr_root, db_path)
result["phases_run"].append(2)
result["total_migrated"] += p2["migrated"]
result["errors"].extend(p2["errors"])
```

Update `CURRENT_SCHEMA_VERSION = 2`.

- [ ] **Step 3: Rename test file and add migration tests**

Rename `tests/unit/test_sqlite_phase1.py` to `tests/unit/test_sqlite_storage.py`. Add:

```python
class TestMigratePhase2:
    def _setup_playbook_files(self, ccr_root):
        os.makedirs(ccr_root, exist_ok=True)
        with open(os.path.join(ccr_root, "playbook.txt"), "w") as f:
            f.write("## STRATEGIES & INSIGHTS\n")
            f.write("[str-00001] helpful=5 harmful=1 :: Test strategy\n")
            f.write("[str-00002] helpful=0 harmful=0 :: Another strategy\n")
            f.write("\n## PROBLEM-SOLVING HEURISTICS\n")
            f.write("[heu-00001] helpful=2 harmful=3 :: Heuristic\n")

        import json
        with open(os.path.join(ccr_root, "failure_lessons.json"), "w") as f:
            json.dump({
                "str-00001": {
                    "lessons": [{
                        "failure_point": "broke",
                        "flawed_reasoning": "wrong",
                        "counterfactual": "X",
                        "prevention_principle": "always X",
                    }],
                    "scope": "general",
                    "trigger": "when X",
                    "weighted_helpful": 2.5,
                }
            }, f)

    def test_playbook_migration(self, tmp_path):
        from ccr.core.storage.migration import migrate_phase_2

        ccr = tmp_path / ".ccr"
        ccr.mkdir()
        self._setup_playbook_files(str(ccr))

        backend = SqliteStorageBackend(str(ccr))
        result = migrate_phase_2(str(ccr), os.path.join(str(ccr), "memory.db"))
        assert result["errors"] == []
        assert result["migrated"] > 0

        bullets = backend.bullet_list()
        assert len(bullets) == 3

        b1 = backend.bullet_get("str-00001")
        assert b1["helpful"] == 5
        assert b1["harmful"] == 1

        lessons = backend.failure_lessons_for_bullet("str-00001")
        assert len(lessons) == 1

        sections = backend.playbook_sections_get()
        assert "STRATEGIES & INSIGHTS" in sections
        assert "PROBLEM-SOLVING HEURISTICS" in sections
        backend.close()
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/test_sqlite_storage.py -v`

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add ccr/core/storage/migration.py
git mv tests/unit/test_sqlite_phase1.py tests/unit/test_sqlite_storage.py
git add tests/unit/test_sqlite_storage.py
git commit -m "feat(storage): implement Phase 2 migration + rename test file"
```

---

### Task 7: Full Test Suite Verification

**Files:**
- All test files

- [ ] **Step 1: Run full unit test suite**

Run: `.venv/bin/python -m pytest tests/unit/ -x -q`

Expected: 2400+ passed, 0 failures

- [ ] **Step 2: Verify import paths after test rename**

Run: `.venv/bin/python -m pytest tests/unit/test_sqlite_storage.py tests/unit/test_storage_backend.py -v --tb=short`

Expected: All PASS

- [ ] **Step 3: Commit if any fixups needed**

```bash
git add -A
git commit -m "fix: resolve test regressions from Phase 2 implementation"
```

---

### Task 8 (Optional): Wire MCP Server to Use Backend for Simple CRUD

**Files:**
- Modify: `ccr/mcp/server.py`
- Modify: `ccr/mcp/ace_tools.py`

This task is the integration step where MCP tools start using the storage backend. It can be deferred to a follow-up session since all Phase 2 storage primitives and tests are complete after Tasks 1-7.

- [ ] **Step 1: Update `_resolve_playbook` to return backend + scope**

The existing `_resolve_playbook(scope)` returns `(Playbook, save_fn)`. For B3 hybrid, add a new function that returns the backend for direct CRUD:

```python
def _resolve_backend(scope: str) -> tuple[StorageBackend, str]:
    """Return (backend, scope) for direct CRUD operations."""
    mem = _ensure_memory()
    return mem._storage, scope
```

- [ ] **Step 2: Commit**

```bash
git add ccr/mcp/server.py ccr/mcp/ace_tools.py
git commit -m "feat(mcp): wire ace tools to use storage backend for simple CRUD"
```
