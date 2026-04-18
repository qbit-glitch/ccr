"""Phase 1 file-backend mixin: scratchpad, metrics, log, metadata."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from ccr.core.storage._sqlite_utils import _utcnow

logger = logging.getLogger(__name__)


class FilePhase1Mixin:
    """Scratchpad, metrics, log, and metadata methods.

    Requires self.ccr_root and self._lock from FileStorageBackend.
    """

    # ── Scratchpad ──────────────────────────────────────────────
    # Delegates to Scratchpad class (unchanged behaviour).
    # The MCP server wires Scratchpad directly when backend is "files",
    # so these methods are only called if something goes through the
    # storage layer explicitly.

    def _scratchpad_path(self) -> str:
        return os.path.join(self.ccr_root, "scratchpad.json")

    def _load_scratchpad(self) -> dict:
        path = self._scratchpad_path()
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("entries", {})
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_scratchpad(self, entries: dict) -> None:
        path = self._scratchpad_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "entries": entries}, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def scratchpad_set(
        self, key: str, value: str, ttl_seconds: int | None = None,
    ) -> dict:
        with self._lock:
            entries = self._load_scratchpad()
            now = _utcnow()
            existing = entries.get(key)
            entry = {
                "value": value,
                "created_at": existing["created_at"] if existing else now,
                "updated_at": now,
                "access_count": (existing.get("access_count", 0) if existing else 0),
                "expires_at": None,
            }
            if ttl_seconds is not None:
                from datetime import timedelta
                exp = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
                entry["expires_at"] = exp.isoformat()
            entries[key] = entry
            self._save_scratchpad(entries)
            return {"key": key, **entry}

    def scratchpad_get(self, key: str) -> dict | None:
        with self._lock:
            entries = self._load_scratchpad()
            entry = entries.get(key)
            if entry is None:
                return None
            if entry.get("expires_at"):
                if datetime.fromisoformat(entry["expires_at"]) < datetime.now(timezone.utc):
                    del entries[key]
                    self._save_scratchpad(entries)
                    return None
            entry["access_count"] = entry.get("access_count", 0) + 1
            entries[key] = entry
            self._save_scratchpad(entries)
            return {"key": key, **entry}

    def scratchpad_list(self) -> list[dict]:
        with self._lock:
            entries = self._load_scratchpad()
            now = datetime.now(timezone.utc)
            result = []
            for k, v in entries.items():
                if v.get("expires_at"):
                    if datetime.fromisoformat(v["expires_at"]) < now:
                        continue
                result.append({"key": k, **v})
            return result

    def scratchpad_delete(self, key: str) -> bool:
        with self._lock:
            entries = self._load_scratchpad()
            if key not in entries:
                return False
            del entries[key]
            self._save_scratchpad(entries)
            return True

    def scratchpad_clear(self) -> int:
        with self._lock:
            entries = self._load_scratchpad()
            count = len(entries)
            self._save_scratchpad({})
            return count

    def scratchpad_search(self, query: str, top_k: int = 5) -> list[dict]:
        entries = self.scratchpad_list()
        query_lower = query.lower()
        scored = []
        for e in entries:
            text = f"{e['key']} {e['value']}".lower()
            if query_lower in text:
                scored.append(e)
        return scored[:top_k]

    # ── Metrics ─────────────────────────────────────────────────

    def _metrics_path(self) -> str:
        return os.path.join(self.ccr_root, "memory_metrics.json")

    def metrics_increment(self, key: str, amount: int = 1) -> None:
        with self._lock:
            data = self.metrics_get()
            data[key] = data.get(key, 0) + amount
            data["last_updated"] = _utcnow()
            path = self._metrics_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def metrics_get(self) -> dict[str, Any]:
        path = self._metrics_path()
        if not os.path.isfile(path):
            return {"total_commits": 0, "search_calls": 0, "link_creations": 0}
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"total_commits": 0, "search_calls": 0, "link_creations": 0}

    # ── Log ─────────────────────────────────────────────────────

    def _log_path(self, branch: str) -> str:
        return os.path.join(self.ccr_root, "branches", branch, "log.md")

    def log_append(self, branch: str, line: str, max_lines: int = 500) -> None:
        with self._lock:
            path = self._log_path(branch)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            content = ""
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    content = f.read()
            lines = content.split("\n") if content else []
            lines.append(line)
            if len(lines) > max_lines:
                lines = lines[-max_lines:]
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

    def log_read(self, branch: str, count: int = 50) -> str:
        path = self._log_path(branch)
        if not os.path.isfile(path):
            return ""
        with open(path, encoding="utf-8") as f:
            content = f.read()
        lines = content.split("\n")
        return "\n".join(lines[-count:])

    # ── Metadata ────────────────────────────────────────────────

    def _metadata_path(self) -> str:
        return os.path.join(self.ccr_root, "metadata.yaml")

    def metadata_load(self) -> dict:
        import yaml

        path = self._metadata_path()
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def metadata_save(self, data: dict) -> None:
        import yaml

        path = self._metadata_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)
