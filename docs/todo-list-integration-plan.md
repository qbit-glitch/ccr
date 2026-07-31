# CCR TODO List Integration Plan

Date: 2026-06-02

## Goal

Add a durable project TODO system to CCR so coding agents can see active work
without relying on noisy auto-commit titles, stale `Current Focus` text, or
one-off `next_step` strings.

The system should:

- Store project-scoped TODOs durably in CCR memory.
- Expose explicit MCP tools for create, list, update, complete, reorder, and
  delete operations.
- Surface a compact TODO digest whenever agents retrieve CCR memory context.
- Optionally attach the same compact digest to every CCR MCP tool response
  without causing token bloat.
- Preserve existing `gcc_commit(next_step=...)` behavior while giving it a real
  task destination.

## Research Summary

### MCP design constraints

MCP tools are model-controlled executable operations with typed inputs and
outputs. CCR already uses `TypedDict` result objects so FastMCP can expose
structured output schemas. TODO tools should follow the same pattern and return
both machine-readable fields and a human-readable `message`.

MCP also has resources. Resources are meant to expose contextual data to
clients, while hosts decide how and when to include that data. A project TODO
snapshot is a strong fit for a read-only resource such as:

```text
ccr://project/todos
```

Tool annotations matter for safety and client UX. TODO list reads should be
`readOnlyHint=True`, additive writes should be non-destructive, repeat-safe
updates should be marked idempotent where possible, and hard deletes should be
destructive.

### Work-management standards

RFC 5545 `VTODO` is the closest formal task standard. Its core fields map
cleanly to CCR: summary/title, description, status, due date, priority,
percent-complete, related items, and assignee-style ownership.

Jira and Linear both model work as items moving through explicit statuses.
Jira emphasizes status, priority, and resolution as high-signal fields for
workflow reporting. Linear uses ordered workflow states such as Backlog, Todo,
In Progress, Done, and Canceled, and keeps priority intentionally coarse.

GitHub now prefers sub-issues over retired tasklist blocks for work that needs
tracking, metadata, and hierarchy. That supports giving CCR TODOs parent-child
relationships instead of only markdown checkboxes.

Kanban guidance emphasizes visualizing workflow, defining start/finish points,
controlling work in progress, monitoring item age, and unblocking work. For CCR,
this argues for an `active` view that is small and sorted, not a giant backlog
injected into every agent turn.

Todoist-style task APIs reinforce common operational fields: parent/project,
order, priority, due date, assignee, labels, and comments.

## Current CCR Gap

CCR currently has:

- `gcc_commit(next_step=...)`, which stores one forward-looking string per
  milestone.
- `.ccr/main.md` sections for `Current Focus`, `Recent Milestones`, and
  `Open Branches`.
- `gcc_context(level=1/2)` and `gcc_status()` that show project overview state.
- `gcc_scratchpad`, which is ephemeral and not suitable for durable project
  tasks.

CCR does not currently have:

- Stable task IDs.
- Multiple open TODOs.
- Status transitions.
- Priority, due date, assignee, labels, or blockers.
- Parent-child work hierarchy.
- Task ordering or active-work limits.
- A read-only TODO resource.
- A single place for agents to check what remains to be done.

## Proposed Data Model

Use project-scoped TODO IDs:

```text
T001, T002, ...
```

Recommended fields:

```python
TodoItem = {
    "id": "T001",
    "title": "Implement todo storage backend",
    "description": "",
    "status": "todo",
    "priority": "normal",
    "order": 1000,
    "branch": "main",
    "scope": "project",
    "parent_id": "",
    "blocked_by": [],
    "labels": [],
    "assignee": "",
    "due_at": "",
    "started_at": "",
    "completed_at": "",
    "percent_complete": 0,
    "source": "manual",
    "source_commit": "",
    "created_at": "2026-06-02T00:00:00Z",
    "updated_at": "2026-06-02T00:00:00Z",
}
```

Status values:

```text
backlog, todo, in_progress, blocked, review, done, canceled
```

Priority values:

```text
urgent, high, normal, low
```

The model intentionally avoids many priority levels. Industry tools usually
benefit from coarse priority and stable ordering more than a large priority
taxonomy.

## Storage Plan

### SQLite backend

Add a Phase 3d storage section:

```sql
CREATE TABLE IF NOT EXISTS todos (
    id               TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    description      TEXT NOT NULL DEFAULT '',
    status           TEXT NOT NULL DEFAULT 'todo',
    priority         TEXT NOT NULL DEFAULT 'normal',
    order_index      INTEGER NOT NULL DEFAULT 1000,
    branch           TEXT NOT NULL DEFAULT 'main',
    scope            TEXT NOT NULL DEFAULT 'project',
    parent_id        TEXT NOT NULL DEFAULT '',
    blocked_by_json  TEXT NOT NULL DEFAULT '[]',
    labels_json      TEXT NOT NULL DEFAULT '[]',
    assignee         TEXT NOT NULL DEFAULT '',
    due_at           TEXT NOT NULL DEFAULT '',
    started_at       TEXT NOT NULL DEFAULT '',
    completed_at     TEXT NOT NULL DEFAULT '',
    percent_complete INTEGER NOT NULL DEFAULT 0,
    source           TEXT NOT NULL DEFAULT 'manual',
    source_commit    TEXT NOT NULL DEFAULT '',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_todos_status ON todos(status);
CREATE INDEX IF NOT EXISTS idx_todos_branch ON todos(branch);
CREATE INDEX IF NOT EXISTS idx_todos_due ON todos(due_at);
CREATE INDEX IF NOT EXISTS idx_todos_parent ON todos(parent_id);
```

Add methods to `StorageBackend`:

```python
todo_insert(data) -> dict
todo_get(todo_id) -> dict | None
todo_list(status=None, branch=None, include_done=False, limit=20) -> list[dict]
todo_update(todo_id, updates) -> bool
todo_delete(todo_id) -> bool
todo_next_id() -> str
todo_reorder(todo_id, before_id=None, after_id=None, order_index=None) -> bool
```

### File backend

Add `.ccr/todos.json`:

```json
{
  "version": 1,
  "next_id": 1,
  "todos": []
}
```

This keeps non-SQLite deployments functional and makes migration simple.

### Migration

Migration should not try to convert every historical `next_step` into a TODO.
That would recreate the current noise problem. Use conservative migration:

- If `.ccr/todos.json` exists, migrate it to SQLite.
- Optionally seed one TODO from current focus only if it is not noisy, not empty,
  and not already represented.
- Do not seed from `[auto] ...`, `Auto-commit: ...`, `status ??`, or empty
  `Next:` fragments.

## MCP Tool Plan

Add a new module:

```text
ccr/mcp/gcc_todo_tools.py
```

Register it in:

- `ccr/mcp/__init__.py`
- `ccr/mcp/server.py` `_TOOL_MODULES`
- `ccr/mcp_server.py` re-export shim if needed

Add result types in `ccr/mcp_types.py`:

```python
class GccTodoResult(TypedDict):
    action: str
    count: int
    todos: list[dict]
    message: str

class GccTodoDigestResult(TypedDict):
    count: int
    overdue: int
    blocked: int
    todos: list[dict]
    message: str
```

Recommended tools:

```text
gcc_todos(action="list", ...)
gcc_todo_add(title, description="", priority="normal", due_at="", labels=[], parent_id="")
gcc_todo_update(todo_id, ...)
gcc_todo_done(todo_id, note="", source_commit="")
gcc_todo_delete(todo_id, hard=False)
gcc_todo_digest(limit=5, include_blocked=True)
```

Annotation policy:

- `gcc_todos` list and `gcc_todo_digest`: read-only, non-destructive,
  idempotent.
- `gcc_todo_add`: write, non-destructive, non-idempotent by default.
- `gcc_todo_update`: write, non-destructive, idempotent if exact fields are set.
- `gcc_todo_done`: write, non-destructive, idempotent.
- `gcc_todo_delete(hard=False)`: soft cancel is non-destructive; hard delete is
  destructive. If MCP annotations cannot vary by argument, annotate the tool as
  destructive and document soft-delete behavior.

## "Visible Whenever CCR Tools Are Called"

There are three possible levels. Implement them in order.

### Level 1: Explicit and low-risk

Inject a compact TODO digest into:

- `gcc_context(level>=1)`
- `gcc_status()`
- Codex `SessionStart` compact summary
- `gcc_commit` result when open TODOs exist

This gives agents the TODO list at the places they already use to load memory.

### Level 2: Bounded digest on every CCR MCP result

Add a shared helper:

```python
def attach_todo_digest(result: dict, *, max_items: int = 3) -> dict:
    ...
```

The helper should:

- Add `todo_digest` to structured results.
- Append at most one short `## Active TODOs` block to `message`.
- Include only `in_progress`, `blocked`, and top sorted `todo` items.
- Skip on `gcc_todo_*` tools to avoid echo loops.
- Skip if no open TODOs.
- Obey `CCR_TODO_DIGEST=0` to disable.
- Obey `CCR_TODO_DIGEST_MAX=3` to tune size.

This satisfies the user's "whenever they call CCR MCP tools" requirement, but
keeps payloads bounded.

### Level 3: MCP resource

Expose:

```text
ccr://project/todos
```

Return a markdown and JSON-friendly TODO snapshot. Hosts that support resources
can display or include it separately from tool calls.

## Sorting and Digest Rules

Default digest sort:

1. `blocked` items first if they block in-progress work.
2. `in_progress`.
3. Overdue `due_at`.
4. Priority: urgent, high, normal, low.
5. `order_index`.
6. `updated_at` descending.

Digest format:

```text
## Active TODOs
- [T003] blocked/high: Fix hook TODO digest loop (blocked by T001)
- [T001] in_progress/normal: Add todo storage backend
- [T002] todo/normal: Add MCP tool tests
```

The digest should not exceed 600 characters by default.

## Integration With Existing GCC Memory

### `gcc_commit`

Add optional args later, after the core TODO tools are stable:

```python
related_todos: list[str] | None = None
complete_todos: list[str] | None = None
create_todo_from_next_step: bool = False
```

Default behavior should stay backward compatible:

- Keep storing `next_step`.
- Do not automatically create TODOs from every commit.
- If `complete_todos` is provided, mark those TODOs done and link
  `source_commit`.
- If `create_todo_from_next_step=True`, create a TODO only when `next_step` is
  clean and non-empty.

### `gcc_context`

Add TODO digest after project overview and before recent commits. That location
is high-signal and does not bury active work under history.

### `gcc_status`

Add counts:

```text
TODOs: 2 in progress, 1 blocked, 5 todo, 3 overdue
```

Then include the same compact digest.

### Hooks

Codex `SessionStart` should include only a one-line TODO count unless the focus
is empty:

```text
CCR retrieved full memory and 3 playbook rules. TODOs: 1 active, 1 blocked.
```

Do not print the full TODO list in hook stdout unless explicitly enabled. Codex
startup output is visible and should remain compact.

## Tests

Unit tests:

- Storage insert/list/update/delete for SQLite and file backends.
- ID allocation and ordering.
- Status validation.
- Priority validation.
- Parent-child relationship validation.
- Blocker validation.
- Soft delete versus hard delete.
- Digest sorting and max-size behavior.
- Noisy `next_step` strings are not migrated into TODOs.
- `gcc_context` includes a compact TODO digest.
- `gcc_status` includes TODO counts and digest.
- `gcc_commit(complete_todos=[...])` marks TODOs done after commit creation.

MCP tests:

- New tools are registered.
- Tool annotations are set correctly.
- `structuredContent` includes `todos` or `todo_digest`.
- Every CCR tool result can be wrapped with a bounded `todo_digest` when
  `CCR_TODO_DIGEST=1`.
- `CCR_TODO_DIGEST=0` disables global digest injection.

Hook tests:

- Codex `SessionStart` includes TODO count only.
- Empty TODO list does not add noise.
- Blocked TODO count appears.

Migration tests:

- Existing `.ccr/todos.json` migrates to SQLite.
- Empty projects stay empty.
- Noisy `[auto] status ??` current focus is ignored.

## Rollout Plan

### Phase A: Core TODO store

Add `ccr/core/todos.py`, storage methods, SQLite DDL, file backend support, and
tests.

### Phase B: First-class MCP tools

Add `gcc_todo_*` tools, result types, docs, output schema tests, and integration
test coverage.

### Phase C: Context integration

Inject TODO digest into `gcc_context`, `gcc_status`, and SessionStart summary.

### Phase D: Commit linkage

Add optional `related_todos`, `complete_todos`, and
`create_todo_from_next_step` support to `gcc_commit`.

### Phase E: Every-tool digest wrapper

Add the bounded digest helper and apply it across all MCP tool result returns.
Start disabled by default if token cost is a concern, or enabled only for GCC
tools first. Measure output growth before enabling across ACE/RLM/index/session
tools.

## Recommended Default Policy

Use:

```text
CCR_TODO_DIGEST=1
CCR_TODO_DIGEST_MAX=3
CCR_TODO_DIGEST_SCOPE=gcc
```

Reasoning:

- Agents see TODOs on normal memory/status calls immediately.
- GCC calls carry TODO context because they are memory-adjacent.
- ACE/RLM/index/session calls avoid unnecessary TODO injection until there is a
  measured reason to enable all-tool injection.

## Source References

- MCP tools and structured tool results:
  https://modelcontextprotocol.io/specification/2025-06-18/server/tools
- MCP schema `structuredContent` and `outputSchema`:
  https://modelcontextprotocol.io/specification/2025-11-25/schema
- MCP resources:
  https://modelcontextprotocol.io/specification/2025-06-18/server/resources
- MCP tool annotations:
  https://blog.modelcontextprotocol.io/posts/2026-03-16-tool-annotations/
- RFC 5545 VTODO:
  https://www.ietf.org/rfc/rfc5545
- Jira work item statuses, priorities, resolutions:
  https://support.atlassian.com/jira-cloud-administration/docs/what-are-issue-statuses-priorities-and-resolutions/
- Linear issue status:
  https://linear.app/docs/configuring-workflows
- Linear milestones:
  https://linear.app/docs/project-milestones
- GitHub tasklists and sub-issues:
  https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/about-tasklists
- Todoist REST task fields:
  https://developer.todoist.com/rest/v1/
- Kanban Guide:
  https://kanbanguides.org/the-kanban-guide/2020.7/
