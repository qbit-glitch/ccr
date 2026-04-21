#!/usr/bin/env python3
"""Hook: fires after tool use (PostToolUse).

Silently accumulates session state by detecting file writes,
test runs, and other meaningful tool operations. Appends to
.ccr/.session_state.json for later auto-commit by on_stop.py.

Provider-agnostic: reads CanonicalEvent payload from stdin,
with fallback to Claude Code's native PostToolUse JSON format.
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


def _read_tool_data_from_stdin() -> tuple[str, dict]:
    """Read tool_name and tool_input from stdin.

    Tries canonical payload first, then falls back to Claude Code's
    native PostToolUse JSON format.
    """
    try:
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
            if raw.strip():
                data = json.loads(raw)
                # Canonical payload: {event: "tool_use", tool_name: "...", tool_input: {...}}
                if data.get("event") == "tool_use":
                    return data.get("tool_name", ""), data.get("tool_input", {}) or {}
                # Claude native: {tool_name: "Write", tool_input: {...}}
                if "tool_name" in data:
                    return data.get("tool_name", ""), data.get("tool_input", {}) or {}
    except Exception:
        pass
    return "", {}


def main() -> None:
    project_root = os.environ.get("CCR_PROJECT_ROOT", os.getcwd())
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    ccr_root = os.path.join(project_root, ".ccr")
    if not os.path.isdir(ccr_root):
        if os.environ.get("CCR_AUTO_INIT", "").lower() in ("1", "true", "yes"):
            from ccr.core.memory import MemoryManager  # noqa: PLC0415
            from ccr.core.types import CCRConfig  # noqa: PLC0415
            MemoryManager(project_root, CCRConfig()).ensure_structure()
        else:
            return

    # Primary: Claude Code sends PostToolUse hook data via stdin JSON.
    tool_name, input_data = _read_tool_data_from_stdin()

    # Fallback: env vars (backward compat / manual invocation / older Claude Code versions).
    if not tool_name:
        tool_name = os.environ.get("CLAUDE_TOOL_NAME", "")
        try:
            input_data = json.loads(os.environ.get("CLAUDE_TOOL_INPUT", "{}") or "{}")
        except (json.JSONDecodeError, TypeError):
            input_data = {}

    if not tool_name:
        return

    summary = ""
    files: list[str] = []

    # Detect file operations
    if tool_name in ("Write", "Edit"):
        file_path = input_data.get("file_path", "")
        if file_path:
            rel = _relative_path(file_path, project_root)
            files.append(rel)
            summary = f"Modified {rel}"

    # Track Read-accessed files (files_touched only, no summary to avoid noise in commits)
    elif tool_name == "Read":
        file_path = input_data.get("file_path", "")
        if file_path:
            rel = _relative_path(file_path, project_root)
            files.append(rel)

    # Detect bash commands that indicate progress
    elif tool_name == "Bash":
        cmd = input_data.get("command", "")
        if "pytest" in cmd or "python -m pytest" in cmd:
            summary = "Ran tests"
        elif cmd.startswith("git commit"):
            summary = "Git commit"
        elif cmd.startswith("git "):
            summary = f"Git: {cmd.split()[1] if len(cmd.split()) > 1 else 'operation'}"

    # Always increment tool_calls (for conditional reminder gate + is_meaningful accuracy).
    # Only append summary/files when meaningful content was detected.
    from ccr.hooks.state_accumulator import append_tool_use, load_state, save_state
    if summary or files:
        append_tool_use(ccr_root, tool_name, summary, files)
    else:
        state = load_state(ccr_root)
        state.tool_calls += 1
        save_state(ccr_root, state)


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
