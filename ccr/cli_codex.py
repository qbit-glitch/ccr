"""Codex wrapper entry point for CCR lifecycle parity."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Sequence


def _log_error(project_root: str, text: str) -> None:
    import datetime

    try:
        ccr_root = os.path.join(project_root, ".ccr")
        log_path = os.path.join(ccr_root, ".hook_errors.log")
        if os.path.isdir(ccr_root):
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n--- {ts} [codex-ccr] ---\n{text}\n")
    except Exception:
        pass


def _find_codex_binary() -> str:
    override = os.environ.get("CCR_CODEX_BINARY")
    if override:
        return override
    binary = shutil.which("codex")
    if not binary:
        raise RuntimeError("codex binary not found on PATH")
    return binary


def _read_current_session_id(ccr_root: str) -> str:
    path = os.path.join(ccr_root, ".current_session_id")
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
    except OSError:
        pass
    return ""


def _remove_if_present(path: str) -> None:
    try:
        os.unlink(path)
    except (FileNotFoundError, OSError):
        pass


def prepare_codex_session(project_root: str, argv: Sequence[str] | None = None) -> str:
    """Initialize project CCR state before launching Codex."""
    from ccr.core.memory import MemoryManager
    from ccr.hooks.codex import codex_ccr_config, mark_wrapper_start

    project_root = os.path.abspath(project_root)
    mem = MemoryManager(project_root, codex_ccr_config())
    mem.ensure_structure()
    mark_wrapper_start(mem.ccr_root, project_root, list(argv or []))
    return mem.ccr_root


def finalize_codex_session(project_root: str, transcript_path: str = "") -> None:
    """Finalize the CCR session after the wrapped Codex process exits."""
    project_root = os.path.abspath(project_root)
    ccr_root = os.path.join(project_root, ".ccr")
    if not os.path.isdir(ccr_root):
        return

    try:
        from ccr.core.memory import MemoryManager
        from ccr.core.session_store import SessionStore
        from ccr.hooks.codex import (
            codex_ccr_config,
            clear_wrapper_state,
            find_latest_codex_transcript,
            read_last_stop_payload,
        )
        from ccr.hooks.on_stop import (
            _auto_baseline_commit,
            _reconcile_transcript,
            _write_session_metrics,
        )
        from ccr.hooks.state_accumulator import clear_state, load_state

        mem = MemoryManager(project_root, codex_ccr_config())
        if not os.path.isdir(mem.ccr_root):
            mem.ensure_structure()

        last_stop = read_last_stop_payload(mem.ccr_root)
        session_id = _read_current_session_id(mem.ccr_root)
        if not session_id and isinstance(last_stop.get("session_id"), str):
            session_id = last_stop["session_id"]

        if not transcript_path and isinstance(last_stop.get("transcript_path"), str):
            transcript_path = last_stop["transcript_path"]
        if not transcript_path:
            transcript_path = find_latest_codex_transcript(project_root)

        state = load_state(mem.ccr_root)
        store = None
        turn_count = 0
        try:
            if session_id:
                store = SessionStore(os.path.join(mem.ccr_root, "sessions.db"))
                if transcript_path and os.path.isfile(transcript_path):
                    _reconcile_transcript(store, session_id, transcript_path)
                store.finalize_session(session_id)
                try:
                    turn_count = len(store.get_session_turns(session_id, limit=10000))
                except Exception:
                    turn_count = 0

            _auto_baseline_commit(mem, state, store, session_id)
            _write_session_metrics(mem.ccr_root, state, turn_count=turn_count)
        finally:
            if store is not None:
                store.close()

        clear_state(mem.ccr_root)
        clear_wrapper_state(mem.ccr_root)
        _remove_if_present(os.path.join(mem.ccr_root, ".current_session_id"))
        _remove_if_present(os.path.join(mem.ccr_root, ".pending_user_msg"))
        _remove_if_present(os.path.join(mem.ccr_root, ".session_active"))
    except Exception:
        import traceback

        _log_error(project_root, traceback.format_exc())


def run_codex(argv: Sequence[str] | None = None) -> int:
    """Run Codex with CCR lifecycle environment and finalize on exit."""
    argv = list(argv if argv is not None else sys.argv[1:])
    project_root = os.path.abspath(os.getcwd())
    previous_storage_backend = os.environ.get("CCR_STORAGE_BACKEND")
    storage_backend = (previous_storage_backend or "").strip() or "sqlite"

    # prepare/finalize call codex_ccr_config(), which reads CCR_STORAGE_BACKEND
    # from the current process. Scope the Codex default to this wrapper call so
    # tests and embedding callers do not inherit a sticky SQLite backend.
    os.environ["CCR_STORAGE_BACKEND"] = storage_backend

    try:
        prepare_codex_session(project_root, argv)
        env = os.environ.copy()
        env.update(
            {
                "CCR_PROJECT_ROOT": project_root,
                "CCR_AGENT": "codex",
                "CCR_HOOK_AGENT": "codex",
                "CCR_AUTO_INIT": "1",
                "CCR_CODEX_WRAPPER": "1",
                "CCR_STORAGE_BACKEND": storage_backend,
            }
        )
        binary = _find_codex_binary()
        completed = subprocess.run([binary, *argv], env=env, check=False)
        return int(completed.returncode)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        _log_error(project_root, str(exc))
        print(f"[CCR] codex-ccr failed: {exc}", file=sys.stderr)
        return 127
    finally:
        finalize_codex_session(project_root)
        if previous_storage_backend is None:
            os.environ.pop("CCR_STORAGE_BACKEND", None)
        else:
            os.environ["CCR_STORAGE_BACKEND"] = previous_storage_backend


def main() -> None:
    raise SystemExit(run_codex())


if __name__ == "__main__":
    main()
