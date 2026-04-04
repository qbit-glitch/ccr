"""Working memory scratchpad for ephemeral key-value storage.

Inspired by AgeMem (arXiv:2601.01885) unified LTM/STM as tool-based actions.
Provides ephemeral working memory distinct from permanent GCC commit memory.
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


@dataclass
class ScratchpadEntry:
    """A single scratchpad entry."""
    key: str
    value: str
    created_at: str       # ISO-8601
    updated_at: str       # ISO-8601
    access_count: int = 0
    expires_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, key: str, d: dict) -> "ScratchpadEntry":
        return cls(
            key=key,
            value=d["value"],
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            access_count=d.get("access_count", 0),
            expires_at=d.get("expires_at"),
        )


class Scratchpad:
    """Ephemeral key-value working memory. Stored in .ccr/scratchpad.json.

    Thread-safe. All public methods acquire a lock before read/write.
    """

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        self._entries: dict[str, ScratchpadEntry] = {}
        self._load()

    def _load(self) -> None:
        """Load entries from disk. Silent on missing/corrupt file."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, entry_data in data.get("entries", {}).items():
                self._entries[key] = ScratchpadEntry.from_dict(key, entry_data)
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning("Failed to load scratchpad %s: %s", self._path, exc)

    def _save(self) -> None:
        """Atomic write to disk (tmp + fsync + replace)."""
        data = {
            "version": 1,
            "entries": {k: v.to_dict() for k, v in self._entries.items()},
        }
        tmp_path = self._path + ".tmp"
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self._path)

    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> ScratchpadEntry:
        """Set a working memory key-value pair. Creates or updates.

        Args:
            key: The key to store under.
            value: The value to store.
            ttl_seconds: Optional time-to-live in seconds. After this duration,
                get() will return None and the entry will be deleted on access.
        """
        now = datetime.now(timezone.utc).isoformat()
        expires_at = (
            (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()
            if ttl_seconds is not None
            else None
        )
        with self._lock:
            if key in self._entries:
                entry = self._entries[key]
                entry.value = value
                entry.updated_at = now
                entry.expires_at = expires_at
            else:
                entry = ScratchpadEntry(
                    key=key,
                    value=value,
                    created_at=now,
                    updated_at=now,
                    access_count=0,
                    expires_at=expires_at,
                )
                self._entries[key] = entry
            self._save()
            return entry

    def get(self, key: str) -> ScratchpadEntry | None:
        """Get a working memory value by key. Increments access_count."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            # TTL expiry check
            if entry.expires_at:
                try:
                    exp = datetime.fromisoformat(entry.expires_at)
                    if datetime.now(timezone.utc) > exp:
                        del self._entries[key]
                        self._save()  # save the deletion
                        return None
                except (ValueError, TypeError):
                    pass
            entry.access_count += 1
            # Do NOT call _save() here — access_count is bookkeeping only
            return entry

    def list_entries(self) -> list[ScratchpadEntry]:
        """List all working memory entries, sorted by updated_at descending.

        Expired entries are deleted in batch from disk to prevent bloat.
        """
        now = datetime.now(timezone.utc)
        with self._lock:
            expired_keys = []
            active = []
            for e in self._entries.values():
                if e.expires_at:
                    try:
                        if datetime.fromisoformat(e.expires_at) <= now:
                            expired_keys.append(e.key)
                            continue
                    except (ValueError, TypeError):
                        pass
                active.append(e)
            if expired_keys:
                for k in expired_keys:
                    del self._entries[k]
                self._save()
            return sorted(active, key=lambda e: e.updated_at, reverse=True)

    def delete(self, key: str) -> bool:
        """Delete a working memory entry. Returns True if it existed."""
        with self._lock:
            if key in self._entries:
                del self._entries[key]
                self._save()
                return True
            return False

    def clear(self) -> int:
        """Clear all working memory entries. Returns count cleared."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._save()
            return count

    @property
    def size(self) -> int:
        """Number of entries in the scratchpad."""
        with self._lock:
            return len(self._entries)

    def format_for_context(self) -> str:
        """Format scratchpad entries for inclusion in gcc_context output.

        Expired entries are excluded from context output.
        """
        active = self.list_entries()
        if not active:
            return ""
        lines = ["# Working Memory"]
        for entry in active:
            lines.append(f"- **{entry.key}**: {entry.value}")
        return "\n".join(lines)
