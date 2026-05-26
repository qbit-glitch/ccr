#!/usr/bin/env python3
"""Codex Stop hook for CCR.

Codex Stop is turn-scoped, not process-exit scoped. This hook therefore saves
meaningful progress at the end of a turn without finalizing the whole CCR
session or deleting session-start markers used by other agents.
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
                f.write(f"\n--- {ts} [codex_stop] ---\n{error_text}\n")
    except Exception:
        pass


def _consume_explicit_commit_marker(ccr_root: str) -> bool:
    marker = os.path.join(ccr_root, ".session_explicit_commit")
    try:
        if os.path.isfile(marker):
            os.unlink(marker)
            return True
    except OSError:
        pass
    return False


def _commit_state(mem, state) -> bool:
    if not state.is_meaningful():
        return False

    fields = state.to_commit_fields()
    mem.commit(
        title=fields["title"],
        what=fields["what"],
        why="Auto-committed by Codex turn hook",
        files_changed=fields["files_changed"],
        next_step=fields["next_step"],
        patterns_learned=fields["patterns_learned"],
        author="[codex-auto]",
    )
    return True


def _read_current_session_id(ccr_root: str, payload: dict) -> str:
    id_file = os.path.join(ccr_root, ".current_session_id")
    try:
        if os.path.isfile(id_file):
            with open(id_file, "r", encoding="utf-8") as f:
                sid = f.read().strip()
                if sid:
                    return sid
    except OSError:
        pass
    sid = payload.get("session_id", "")
    return sid if isinstance(sid, str) else ""


def _reconcile_current_transcript(ccr_root: str, payload: dict) -> None:
    """Best-effort turn logging on each Codex turn Stop."""
    transcript_path = payload.get("transcript_path", "")
    if not isinstance(transcript_path, str) or not os.path.isfile(transcript_path):
        return
    session_id = _read_current_session_id(ccr_root, payload)
    if not session_id:
        return
    try:
        from ccr.core.session_store import SessionStore  # noqa: PLC0415
        from ccr.hooks.on_stop import _reconcile_transcript  # noqa: PLC0415

        store = SessionStore(os.path.join(ccr_root, "sessions.db"))
        _reconcile_transcript(store, session_id, transcript_path)
        store.close()
    except Exception:
        import traceback

        _log_hook_error(traceback.format_exc())


def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from ccr.hooks.codex import (
        codex_ccr_config,
        read_json_stdin,
        remember_stop_payload,
        resolve_project_root,
    )
    from ccr.core.memory import MemoryManager
    from ccr.hooks.state_accumulator import clear_work_state, load_state

    payload = read_json_stdin()
    project_root = resolve_project_root(payload)
    mem = MemoryManager(project_root, codex_ccr_config())
    if not os.path.isdir(mem.ccr_root):
        if os.environ.get("CCR_AUTO_INIT", "").lower() in ("1", "true", "yes"):
            mem.ensure_structure()
        else:
            return

    remember_stop_payload(mem.ccr_root, payload)
    _reconcile_current_transcript(mem.ccr_root, payload)
    state = load_state(mem.ccr_root)

    if _consume_explicit_commit_marker(mem.ccr_root):
        clear_work_state(mem.ccr_root)
        return

    try:
        if _commit_state(mem, state):
            clear_work_state(mem.ccr_root)
    except Exception:
        import traceback

        _log_hook_error(traceback.format_exc())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        _log_hook_error(traceback.format_exc())
