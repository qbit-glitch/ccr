#!/usr/bin/env python3
"""Claude Code hook: fires on PostToolUse.

Silently accumulates session state by detecting file writes,
test runs, and other meaningful tool operations. Appends to
.ccr/.session_state.json for later auto-commit by on_stop.py.

Outputs nothing to stdout (silent accumulator).
"""

import json
import os
import sys


def _log_hook_error(error_text: str) -> None:
    """Write error to .ccr/.hook_errors.log — non-fatal, never raises."""
    import datetime
    try:
        proj = os.environ.get("CCR_PROJECT_ROOT", os.getcwd())
        log_path = os.path.join(proj, ".ccr", ".hook_errors.log")
        if os.path.isdir(os.path.dirname(log_path)):
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n--- {ts} [on_tool_use] ---\n{error_text}\n")
    except Exception:
        pass


def main() -> None:
    project_root = os.environ.get("CCR_PROJECT_ROOT", os.getcwd())
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    ccr_root = os.path.join(project_root, ".ccr")
    if not os.path.isdir(ccr_root):
        return

    # Claude Code passes hook context via CLAUDE_HOOK_* env vars
    tool_name = os.environ.get("CLAUDE_TOOL_NAME", "")
    tool_input = os.environ.get("CLAUDE_TOOL_INPUT", "{}")

    if not tool_name:
        return

    try:
        input_data = json.loads(tool_input) if tool_input else {}
    except (json.JSONDecodeError, TypeError):
        input_data = {}

    summary = ""
    files: list[str] = []

    # Detect file operations
    if tool_name in ("Write", "Edit"):
        file_path = input_data.get("file_path", "")
        if file_path:
            rel = _relative_path(file_path, project_root)
            files.append(rel)
            summary = f"Modified {rel}"

    # Detect bash commands that indicate progress
    elif tool_name == "Bash":
        cmd = input_data.get("command", "")
        if "pytest" in cmd or "python -m pytest" in cmd:
            summary = "Ran tests"
        elif cmd.startswith("git commit"):
            summary = "Git commit"
        elif cmd.startswith("git "):
            summary = f"Git: {cmd.split()[1] if len(cmd.split()) > 1 else 'operation'}"

    # Skip if nothing meaningful detected
    if not summary and not files:
        return

    from ccr.hooks.state_accumulator import append_tool_use
    append_tool_use(ccr_root, tool_name, summary, files)


def _relative_path(file_path: str, project_root: str) -> str:
    """Convert absolute path to project-relative path."""
    try:
        return os.path.relpath(file_path, project_root)
    except ValueError:
        return file_path


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        _log_hook_error(traceback.format_exc())
