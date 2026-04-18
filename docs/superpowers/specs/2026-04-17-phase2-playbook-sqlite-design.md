# Phase 2: Playbook + Failure Lessons + Schema -> SQLite

## Context

Phase 0 (storage abstraction) and Phase 1 (scratchpad/metrics/log/metadata) are complete. The ACE playbook system currently stores data across 5 flat files per scope (project + global = 10 files total). Counter updates require full serialize/deserialize cycles. This phase normalizes all playbook data into SQLite tables for efficient CRUD while keeping the Playbook class for complex analytics.

## Decisions

- **Fully normalized SQL** (not blob storage) — bullets, failure lessons, schema as separate tables
- **B3 hybrid** — StorageBackend handles CRUD; Playbook class retained for GRPO, similarity, schema evolution
- **Everything migrated** — including delta_history and archived_bullets (total recall)
- **Same schema in both `memory.db` (project) and `global.db` (global)**

## Schema

Six new tables, created in both `memory.db` and `global.db`:

```sql
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
```

## Architecture

### B3 Hybrid Pattern

```
MCP Tool Call
  |
  +-- Simple CRUD (no full load) --> StorageBackend SQL methods
  |     ace_update_counters         --> bullet_update_counters()
  |     ace_apply_delta(ADD)        --> bullet_insert()
  |     ace_apply_delta(REMOVE)     --> bullet_delete()
  |     ace_apply_delta(UPDATE)     --> bullet_update()
  |     ace_get_playbook (read)     --> bullet_list() + sections_get()
  |
  +-- Complex logic (full load) --> Playbook.from_backend(backend, scope)
        ace_apply_delta(MERGE)       --> load, merge, save_to_backend()
        ace_find_similar             --> load, compute embeddings, return
        ace_prune                    --> load, score, delete via backend
        ace_evolve_from_failures     --> load, deduplicate, insert new
        ace_evolve_schema            --> load, compute health, propose
        ace_generate_bullets         --> load, check similarity, insert
```

### Scope Routing

All Phase 2 StorageBackend methods take `scope: str = "project"`. Internally:
- `scope="project"` -> `self.memory_conn` (`.ccr/memory.db`)
- `scope="global"` -> `self.global_conn` (`~/.ccr/global.db`)

## StorageBackend ABC Additions

18 new abstract methods in `ccr/core/storage/base.py`:

### Bullets (7 methods)

```python
def bullet_get(self, bullet_id: str, scope: str = "project") -> dict | None
def bullet_list(self, section: str | None = None, scope: str = "project") -> list[dict]
def bullet_insert(self, bullet: dict, scope: str = "project") -> None
def bullet_update(self, bullet_id: str, updates: dict, scope: str = "project") -> bool
def bullet_delete(self, bullet_id: str, scope: str = "project") -> bool
def bullet_update_counters(self, bullet_tags: list[dict], scope: str = "project") -> int
def bullet_get_next_id(self, scope: str = "project") -> int
```

### Failure Lessons (4 methods)

```python
def failure_lessons_for_bullet(self, bullet_id: str, scope: str = "project") -> list[dict]
def failure_lessons_insert(self, bullet_id: str, lesson: dict, scope: str = "project") -> None
def failure_lessons_mark_evolved(self, bullet_id: str, scope: str = "project") -> int
def failure_lessons_all(self, scope: str = "project") -> dict[str, list[dict]]
```

### Sections (2 methods)

```python
def playbook_sections_get(self, scope: str = "project") -> list[str]
def playbook_sections_set(self, sections: list[str], scope: str = "project") -> None
```

### Schema (3 methods)

```python
def playbook_schema_load(self, scope: str = "project") -> dict
def playbook_schema_save(self, schema: dict, history_entry: dict | None = None, scope: str = "project") -> None
def playbook_schema_history(self, scope: str = "project") -> list[dict]
```

### Audit (2 methods)

```python
def delta_history_append(self, entry: dict, scope: str = "project") -> None
def archived_bullets_insert(self, bullets: list[dict], reason: str, scope: str = "project") -> int
```

## File Changes

### Create

None. All changes modify existing files.

### Modify

| File | Changes | Est. Lines |
|------|---------|-----------|
| `ccr/core/storage/base.py` | +18 abstract methods (Phase 2 section) | +80 |
| `ccr/core/storage/sqlite_backend.py` | Phase 2 DDL + all 18 SQL implementations + `_get_scoped_conn()` helper | +250 |
| `ccr/core/storage/file_backend.py` | File-based implementations wrapping playbook.txt + JSON I/O (full load/save per call; inefficient but correct for legacy path) | +200 |
| `ccr/core/storage/migration.py` | `migrate_phase_2()` + `migrate_phase_2_global()` | +150 |
| `ccr/ace/playbook.py` | Add `from_backend(backend, scope)` classmethod + `save_to_backend(backend, scope)` | +60 |
| `ccr/mcp/server.py` | `_load_playbook`/`_save_playbook` delegate to backend; `_resolve_playbook` returns backend | +30 |
| `ccr/mcp/ace_tools.py` | Simple CRUD tools bypass Playbook, call backend directly | +40 |
| `tests/unit/test_storage_backend.py` | Phase 2 method tests for both backends | +100 |
| `tests/unit/test_sqlite_phase1.py` -> rename to `test_sqlite_storage.py` | Add Phase 2 SQL-specific tests + migration tests | +200 |

### Key Implementation Details

**`_get_scoped_conn(scope)`** in SqliteStorageBackend:
```python
def _get_scoped_conn(self, scope: str) -> sqlite3.Connection:
    if scope == "global":
        if self.global_conn is None:
            raise ValueError("No global database configured")
        return self.global_conn
    return self.memory_conn
```

**`bullet_update_counters()`** — the most important optimization:
```python
def bullet_update_counters(self, bullet_tags, scope="project"):
    conn = self._get_scoped_conn(scope)
    updated = 0
    for tag in bullet_tags:
        bid = tag["id"]
        weight = tag.get("weight", 1.0)
        if tag["tag"] == "helpful":
            conn.execute("""UPDATE playbook_bullets SET
                helpful = helpful + 1,
                weighted_helpful = weighted_helpful + ?,
                last_updated = ?,
                personal_decay_rate = max(0.90, min(0.99,
                    0.95 + (helpful + 1 - harmful) * 0.002))
                WHERE id = ?""", (weight, _utcnow(), bid))
        else:  # harmful
            conn.execute("""UPDATE playbook_bullets SET
                harmful = harmful + 1,
                weighted_harmful = weighted_harmful + ?,
                last_updated = ?,
                personal_decay_rate = max(0.90, min(0.99,
                    0.95 + (helpful - harmful - 1) * 0.002))
                WHERE id = ?""", (weight, _utcnow(), bid))
            lesson = tag.get("failure_lesson")
            if lesson:
                self.failure_lessons_insert(bid, lesson, scope)
        if conn.execute("SELECT changes()").fetchone()[0] > 0:
            updated += 1
    conn.commit()
    return updated
```

**`Playbook.from_backend()`** — loads all data from SQL into in-memory Playbook:
```python
@classmethod
def from_backend(cls, backend, scope="project"):
    sections = backend.playbook_sections_get(scope)
    bullets_data = backend.bullet_list(scope=scope)
    pb = cls.__new__(cls)
    pb._sections = sections or list(DEFAULT_SECTIONS)
    pb._bullets = [Bullet(**b) for b in bullets_data]
    pb._id_index = {b.id: b for b in pb._bullets}
    pb._next_id = backend.bullet_get_next_id(scope)
    # Load failure lessons
    all_lessons = backend.failure_lessons_all(scope)
    for bid, lessons in all_lessons.items():
        if bid in pb._id_index:
            pb._id_index[bid].failure_lessons = [
                FailureLesson(**l) for l in lessons
            ]
    return pb
```

## Migration

### `migrate_phase_2(ccr_root, db_path)`

Parses project-scope flat files:

1. **`playbook.txt`** -> `_BULLET_RE` regex -> INSERT into `playbook_bullets` + `playbook_sections`
2. **`failure_lessons.json`** -> INSERT into `failure_lessons` + UPDATE bullet extended fields (scope, trigger, weighted_*, personal_decay_rate)
3. **`playbook_schema.json`** -> INSERT `current` into `playbook_schema` + all `history` entries
4. **`playbook_history.json`** -> INSERT into `delta_history`
5. **`archived_bullets.json`** -> INSERT into `archived_bullets`

### `migrate_phase_2_global(global_ccr_root, global_db_path)`

Same logic for global-scope files:
- `~/.ccr/global_playbook.txt`
- `~/.ccr/global_failure_lessons.json`
- `~/.ccr/global_playbook_schema.json`

Both functions run in single transactions. Source files backed up to `.bak`.

## Testing

### Dual-backend parameterized tests
- All 18 new StorageBackend methods tested with `@pytest.fixture(params=["files", "sqlite"])`
- Bullet CRUD: insert, get, list, update, delete
- Counter updates: helpful/harmful increment, weight tracking, failure lesson attachment
- Section ordering: get/set roundtrip, preserve order
- Schema: load/save/history
- Delta history: append/query
- Archived bullets: insert/list

### Migration tests
- Parse playbook.txt with various bullet formats (with/without trigger, with/without failure lessons)
- Verify counter values preserved exactly
- Verify failure lesson FK integrity
- Verify section ordering matches original
- Idempotency: run migration twice, no duplicates

### Integration tests
- `Playbook.from_backend()` -> mutate -> `save_to_backend()` -> reload -> assert equal
- `ace_get_playbook` returns identical output on both backends
- `ace_update_counters` increments correctly on SQLite

## Verification

1. `pytest tests/unit/ -x -q` — full suite passes, zero regressions
2. `CCR_STORAGE_BACKEND=sqlite pytest tests/unit/test_playbook.py` — playbook tests pass on SQLite
3. Counter update benchmark: verify SQLite path doesn't require full serialize/deserialize
4. Global scope: bullets stored in `global.db`, not `memory.db`
5. Migration: create flat files -> migrate -> verify all data accessible via SQL

## Risk

**Medium**. The Playbook class has been stable since v3. Key risks:
- `_BULLET_RE` regex must exactly match all existing playbook.txt formats during migration
- FK cascade delete on failure_lessons means bullet deletion is irreversible
- Global scope routing adds a new failure mode (no global.db configured)

Mitigations:
- Migration runs in single transaction (all or nothing)
- `.bak` files preserved for manual recovery
- Feature flag (`storage_backend="files"`) allows instant rollback
