# Contributing to CCR

## Development Setup

```bash
git clone https://github.com/qbit-glitch/ccr.git
cd ccr
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.11+. **Platform: macOS and Linux only.** Windows is not yet supported.

## Running Tests

```bash
pytest tests/unit/ -x -q              # unit tests only (~2000 tests, fast)
pytest tests/integration/ -x -q       # integration tests (slower)
pytest tests/ -x -q                   # full suite
```

Run a specific test file:
```bash
pytest tests/unit/test_gcc_search.py -v
```

## Code Style

**File size limit**: 800 lines maximum per file. If a file grows beyond this, split it.
Current split pattern: `gcc_tools.py` → `gcc_tools.py` + `gcc_search_tools.py` + `gcc_branch_tools.py`.

**Type hints**: all functions must have type hints.

**Imports**: stdlib → third-party → local. Use `from __future__ import annotations` at the top.

**No bare `except:`** — always catch specific exceptions or use `except Exception as exc:` and log it.

Lint check:
```bash
ruff check ccr/
```

## Adding a New MCP Tool

1. Add `@mcp.tool(annotations=...)` function to the appropriate tool module in `ccr/mcp/`
2. Add the return type to `ccr/mcp_types.py`
3. Write tests in `tests/unit/test_<module>.py`
4. If the module is new, add it to `ccr/mcp/__init__.py`
5. Update tool count in `CLAUDE.md`

## Referencing Research Papers

CCR implements ideas from 16 papers. When adding a feature inspired by a paper:

```python
# GCC (arXiv:2508.00031) — version-controlled memory
# ACE (arXiv:2510.04618) — evolving playbooks
```

Add the paper to `docs/papers.md` and the `CLAUDE.md` research list if it's new.

## Pull Request Checklist

- [ ] Tests pass: `pytest tests/unit/ -x -q`
- [ ] No ruff errors: `ruff check ccr/`
- [ ] File sizes under 800 lines: `wc -l ccr/mcp/*.py ccr/core/*.py`
- [ ] New MCP tools added to `ccr/mcp/__init__.py`
- [ ] Tool count in `CLAUDE.md` updated if tools added/removed
- [ ] `CHANGELOG.md` entry added for user-visible changes
