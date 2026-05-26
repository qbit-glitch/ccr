#!/usr/bin/env python3
"""Codex hook: fires before a user prompt is submitted.

Codex has a real SessionStart hook, so this script does not inject full memory.
It only buffers the prompt for session logging and emits a lightweight reminder
after meaningful tool activity has occurred.
"""

from __future__ import annotations

import os
import sys


def _log_hook_error(error_text: str) -> None:
    """Write hook errors to .ccr/.hook_errors.log without failing Codex."""
    import datetime

    try:
        project = os.environ.get("CCR_PROJECT_ROOT", os.getcwd())
        log_path = os.path.join(project, ".ccr", ".hook_errors.log")
        if os.path.isdir(os.path.dirname(log_path)):
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n--- {ts} [on_user_prompt_submit] ---\n{error_text}\n")
    except Exception:
        pass


def _buffer_user_prompt(ccr_root: str, prompt: str) -> None:
    if not prompt:
        return
    try:
        tmp = os.path.join(ccr_root, ".pending_user_msg.tmp")
        dst = os.path.join(ccr_root, ".pending_user_msg")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(prompt)
        os.replace(tmp, dst)
    except OSError:
        pass


def _should_emit_compaction_reminder(prompt: str, state) -> bool:
    """Heuristic Codex PreCompact simulation.

    Codex has no native PreCompact hook, so the closest safe moment to nudge is
    before the next prompt is submitted when prior work has accumulated.
    """
    if not state.is_meaningful():
        return False
    try:
        threshold = int(os.environ.get("CCR_CODEX_PRECOMPACT_CHARS", "50000"))
    except ValueError:
        threshold = 50000
    work_chars = sum(len(item) for item in state.what_accumulated)
    work_chars += sum(len(path) for path in state.files_touched) * 4
    return (len(prompt or "") + work_chars) >= threshold


def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from ccr.hooks.codex import codex_ccr_config, read_json_stdin, resolve_project_root
    from ccr.core.memory import MemoryManager
    from ccr.hooks.state_accumulator import load_state

    payload = read_json_stdin()
    project_root = resolve_project_root(payload)
    mem = MemoryManager(project_root, codex_ccr_config())
    if not os.path.isdir(mem.ccr_root):
        if os.environ.get("CCR_AUTO_INIT", "").lower() in ("1", "true", "yes"):
            mem.ensure_structure()
        else:
            return

    prompt = payload.get("prompt", "")
    _buffer_user_prompt(mem.ccr_root, prompt)

    try:
        state = load_state(mem.ccr_root)
        if state.tool_calls == 0:
            return
    except Exception:
        return

    if _should_emit_compaction_reminder(prompt, state):
        reminder = (
            "CCR context is getting large and Codex has no native PreCompact hook. "
            "Call gcc_commit now for durable memory before continuing."
        )
    else:
        reminder = (
            "CCR save policy for Codex: call gcc_commit only after a meaningful "
            "milestone, before likely context loss, or when the user explicitly asks."
        )
    print(reminder)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        _log_hook_error(traceback.format_exc())
