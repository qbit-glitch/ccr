"""Canonical cross-agent event contract for CCR.

The schema is intentionally small and JSONL-friendly so Claude, Codex, Kimi,
Continue, Ollama wrappers, and future agents can all emit the same event shape.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class AgentEvent:
    """One canonical event emitted by any CCR-integrated agent."""

    agent: str
    session_id: str
    project: str
    event_type: str
    user_intent: str = ""
    files_touched: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    outcome: str = ""
    memory_candidates: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        errors: list[str] = []
        required = {
            "agent": self.agent,
            "session_id": self.session_id,
            "project": self.project,
            "event_type": self.event_type,
        }
        for key, value in required.items():
            if not str(value).strip():
                errors.append(f"{key} is required")
        if not isinstance(self.files_touched, list):
            errors.append("files_touched must be a list")
        if not isinstance(self.tools_used, list):
            errors.append("tools_used must be a list")
        if not isinstance(self.memory_candidates, list):
            errors.append("memory_candidates must be a list")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentEvent":
        return cls(
            agent=str(data.get("agent", "")),
            session_id=str(data.get("session_id", "")),
            project=str(data.get("project", "")),
            event_type=str(data.get("event_type", "")),
            user_intent=str(data.get("user_intent", "")),
            files_touched=list(data.get("files_touched") or []),
            tools_used=list(data.get("tools_used") or []),
            outcome=str(data.get("outcome", "")),
            memory_candidates=list(data.get("memory_candidates") or []),
            timestamp=str(data.get("timestamp") or utc_now()),
            metadata=dict(data.get("metadata") or {}),
        )


def events_path(ccr_root: str) -> str:
    return os.path.join(ccr_root, "agent_events.jsonl")


def append_agent_event(ccr_root: str, event: AgentEvent) -> None:
    """Append an event to .ccr/agent_events.jsonl after schema validation."""
    errors = event.validate()
    if errors:
        raise ValueError("; ".join(errors))
    os.makedirs(ccr_root, exist_ok=True)
    path = events_path(ccr_root)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(event.to_dict(), sort_keys=True))
        fh.write("\n")


def load_agent_events(ccr_root: str, limit: int = 100) -> list[AgentEvent]:
    path = events_path(ccr_root)
    if not os.path.isfile(path):
        return []
    rows: list[AgentEvent] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(AgentEvent.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return rows[-max(1, limit):]


def validate_event_file(path: str) -> tuple[int, list[str]]:
    """Validate a JSONL event file. Returns (valid_count, errors)."""
    errors: list[str] = []
    valid = 0
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                event = AgentEvent.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                errors.append(f"line {line_no}: {exc}")
                continue
            ev_errors = event.validate()
            if ev_errors:
                errors.append(f"line {line_no}: {'; '.join(ev_errors)}")
            else:
                valid += 1
    return valid, errors


def atomic_write_json(path: str, data: dict[str, Any]) -> None:
    """Utility for schema/config writers."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".event.", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
