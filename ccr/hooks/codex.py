"""Shared helpers for Codex hook and wrapper integration."""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from typing import Any

_LAST_STOP_PAYLOAD = ".codex_last_stop.json"
_WRAPPER_STATE = ".codex_wrapper_state.json"
CODEX_DEFAULT_STORAGE_BACKEND = "sqlite"


def codex_storage_backend() -> str:
    """Return the storage backend Codex CCR should use by default."""
    return os.environ.get("CCR_STORAGE_BACKEND", "").strip() or CODEX_DEFAULT_STORAGE_BACKEND


def codex_ccr_config():
    """Return CCRConfig for Codex hooks/wrapper, defaulting to SQLite."""
    from ccr.core.types import CCRConfig  # noqa: PLC0415

    return CCRConfig(storage_backend=codex_storage_backend())


def read_json_stdin() -> dict[str, Any]:
    """Read a hook JSON payload from stdin without raising."""
    try:
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
            if raw.strip():
                data = json.loads(raw)
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def resolve_project_root(payload: dict[str, Any] | None = None) -> str:
    """Resolve the project root for Codex hooks.

    Codex provides ``cwd`` in native hook payloads.  Older/shared hooks mostly
    relied on process cwd, which is wrong if the hook subprocess is launched
    from a different directory.
    """
    payload = payload or {}
    candidates = (
        os.environ.get("CCR_PROJECT_ROOT"),
        payload.get("cwd"),
        payload.get("project_root"),
        os.environ.get("PWD"),
        os.getcwd(),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return os.path.abspath(os.path.expanduser(candidate))
    return os.getcwd()


def write_json_atomic(path: str, data: dict[str, Any]) -> None:
    """Write JSON with an atomic replace."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def remember_stop_payload(ccr_root: str, payload: dict[str, Any]) -> None:
    """Persist the latest Codex Stop payload for wrapper exit finalization."""
    if not payload:
        return
    keep = {
        "cwd": payload.get("cwd"),
        "session_id": payload.get("session_id"),
        "transcript_path": payload.get("transcript_path"),
        "hook_event_name": payload.get("hook_event_name"),
        "timestamp": time.time(),
    }
    try:
        write_json_atomic(os.path.join(ccr_root, _LAST_STOP_PAYLOAD), keep)
    except OSError:
        pass


def read_last_stop_payload(ccr_root: str) -> dict[str, Any]:
    """Read the latest stored Codex Stop payload."""
    path = os.path.join(ccr_root, _LAST_STOP_PAYLOAD)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def mark_wrapper_start(ccr_root: str, project_root: str, argv: list[str]) -> None:
    """Record wrapper lifecycle metadata for recovery/debugging."""
    try:
        write_json_atomic(
            os.path.join(ccr_root, _WRAPPER_STATE),
            {
                "agent": "codex",
                "project_root": project_root,
                "argv": argv,
                "pid": os.getpid(),
                "started_at": time.time(),
            },
        )
    except OSError:
        pass


def clear_wrapper_state(ccr_root: str) -> None:
    """Remove wrapper lifecycle metadata if present."""
    try:
        os.unlink(os.path.join(ccr_root, _WRAPPER_STATE))
    except (FileNotFoundError, OSError):
        pass


def _codex_transcript_project(path: str) -> str:
    """Return the cwd recorded in a Codex session transcript, if available."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for _ in range(25):
                line = f.readline()
                if not line:
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "session_meta":
                    payload = obj.get("payload", {})
                    cwd = payload.get("cwd")
                    return os.path.abspath(cwd) if isinstance(cwd, str) else ""
    except OSError:
        return ""
    return ""


def find_latest_codex_transcript(project_root: str) -> str:
    """Find the newest Codex JSONL transcript for a project, if one is available."""
    base = os.path.expanduser("~/.codex/sessions")
    if not os.path.isdir(base):
        return ""

    project_root = os.path.abspath(project_root)
    pattern = os.path.join(base, "**", "*.jsonl")
    try:
        candidates = sorted(
            glob.glob(pattern, recursive=True),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
    except OSError:
        return ""

    for path in candidates[:200]:
        if _codex_transcript_project(path) == project_root:
            return path
    return ""
