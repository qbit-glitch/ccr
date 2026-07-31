# CCR MCP Manual E2E Verification - 2026-06-02

## Scope

Verified the current registered CCR MCP surface in an isolated temporary project
with an isolated temporary `HOME`, using `CCR_STORAGE_BACKEND=sqlite` and
`CCR_AUTO_INIT=1`. This avoided destructive effects on the real project memory
and global playbook.

Current registered tool count: 44.

## Manual E2E Result

The manual harness exercised all 44 registered MCP tools through the exported
tool functions:

- Registered tools covered: 44 / 44
- Valid workflow checks: 56 passed, 0 failed
- Edge/error probes: 17 passed, 0 failed
- Missing tools: none

Tool families covered:

- GCC memory/search/facts/discussions/branches: commit, context, status,
  branch, merge, links, search, recall, patterns, experiments, discussions,
  facts, conflicts, conflict resolution, consolidation, memory evolution,
  projects, scratchpad, scratchpad search
- GCC TODOs: list, add, update, done, soft delete/cancel, digest
- ACE: get playbook, apply delta, update counters, find similar, evolve schema,
  generate bullets, evolve from failures, prune
- RLM: init, execute, finalize
- Index: build, status, search
- Session logger: log turn, history, search, export

Edge probes covered invalid index modes, missing RLM sessions, invalid TODO
titles/priorities/status values, missing TODO deletion, invalid facts actions,
missing conflict IDs, duplicate branch creation before merge, missing branch
merge, unknown ACE operations, missing session history, invalid session export
formats, scratchpad TTL expiry, and invalid RLM variable names.

## Automated Verification

Focused verification:

```bash
.venv/bin/python -m pytest tests/unit/test_todos.py tests/unit/test_empty_project_ux.py tests/unit/test_output_schema.py tests/unit/test_mcp_server.py::TestToolAnnotations -q
```

Result: 117 passed.

Lint:

```bash
.venv/bin/python -m ruff check ccr/core/storage/_file_phase3d.py ccr/core/storage/_sqlite_phase3d.py ccr/core/memory_pkg/memory_todos.py ccr/mcp/gcc_todo_tools.py tests/unit/test_todos.py
```

Result: all checks passed.

Broad suite:

```bash
.venv/bin/python -m pytest tests/unit/ tests/integration/ -q
```

Sandbox result: 2982 passed, 56 skipped, 13 errors. All 13 errors came from
`tests/integration/test_gateway_e2e.py` failing to bind a local mock HTTP server
on `127.0.0.1` under the sandbox.

Escalated rerun of the failing integration file:

```bash
.venv/bin/python -m pytest tests/integration/test_gateway_e2e.py -q
```

Result: 13 passed.

## Notes

- No manual E2E failures remained after aligning the harness checks with the
  actual result contracts (`todos`, `data`, and message-based envelopes).
- `gcc_todo_delete(hard=False)` intentionally returns `count=1` plus a cancel
  message and an empty `todos` list.
- Unknown ACE delta operation types currently apply zero operations and return a
  successful zero-applied result rather than raising.
- Invalid session export formats return a non-fatal `Export failed` message with
  empty `data` rather than raising.
