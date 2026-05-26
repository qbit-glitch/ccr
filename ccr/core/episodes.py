"""Immutable episode and evidence store for CCR memory governance.

Episodes are append-only, hash-chained records of important memory events:
tool-observed evidence, recall traces, conflict resolutions, and promotion
decisions.  The store intentionally uses JSONL so it stays local-first,
inspectable, git-diffable, and safe to add to existing CCR projects without a
schema migration.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ccr.core.facts import lexical_score


def utc_now() -> str:
    """Return a stable ISO UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_hash(entry: dict[str, Any]) -> str:
    """Return the SHA-256 hash for an episode body without its hash field."""
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class EpisodeRecord:
    """One immutable evidence episode."""

    id: str
    event_type: str
    agent: str = "ccr"
    summary: str = ""
    content: str = ""
    source_ids: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    prev_hash: str = ""
    hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeRecord":
        return cls(
            id=str(data.get("id", "")),
            event_type=str(data.get("event_type", "")),
            agent=str(data.get("agent") or "ccr"),
            summary=str(data.get("summary", "")),
            content=str(data.get("content", "")),
            source_ids=list(data.get("source_ids") or []),
            files=list(data.get("files") or []),
            tools=list(data.get("tools") or []),
            metadata=dict(data.get("metadata") or {}),
            created_at=str(data.get("created_at") or utc_now()),
            prev_hash=str(data.get("prev_hash", "")),
            hash=str(data.get("hash", "")),
        )


@dataclass
class ChainVerification:
    """Episode hash-chain verification result."""

    ok: bool
    checked: int
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EpisodeStore:
    """Append-only JSONL episode/evidence log."""

    def __init__(self, ccr_root: str):
        self.ccr_root = ccr_root
        self.path = os.path.join(ccr_root, "episodes.jsonl")

    def append_episode(
        self,
        event_type: str,
        *,
        agent: str = "ccr",
        summary: str = "",
        content: str = "",
        source_ids: list[str] | None = None,
        files: list[str] | None = None,
        tools: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EpisodeRecord:
        """Append an immutable episode and return the stored record."""
        event_type = event_type.strip()
        if not event_type:
            raise ValueError("event_type must not be empty.")
        episodes = list(reversed(self.list_episodes(limit=100000)))
        prev_hash = episodes[-1].hash if episodes else ""
        episode = EpisodeRecord(
            id=self._next_id(episodes),
            event_type=event_type,
            agent=agent.strip() or "ccr",
            summary=summary.strip(),
            content=content.strip(),
            source_ids=list(source_ids or []),
            files=list(files or []),
            tools=list(tools or []),
            metadata=dict(metadata or {}),
            prev_hash=prev_hash,
        )
        data = episode.to_dict()
        data["hash"] = _json_hash(data)
        episode.hash = data["hash"]
        os.makedirs(self.ccr_root, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, sort_keys=True))
            fh.write("\n")
        return episode

    def list_episodes(self, query: str = "", limit: int = 50) -> list[EpisodeRecord]:
        """List recent episodes, optionally filtered by lexical query."""
        if not os.path.isfile(self.path):
            return []
        episodes: list[EpisodeRecord] = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    episodes.append(EpisodeRecord.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        if query.strip():
            scored = [
                (
                    lexical_score(
                        query,
                        " ".join([
                            e.id,
                            e.event_type,
                            e.summary,
                            e.content,
                            " ".join(e.source_ids),
                            " ".join(e.files),
                        ]),
                    ),
                    e,
                )
                for e in episodes
            ]
            episodes = [e for score, e in scored if score > 0 or query.lower() == e.id.lower()]
            episodes.sort(
                key=lambda e: lexical_score(
                    query,
                    f"{e.id} {e.event_type} {e.summary} {e.content} {' '.join(e.source_ids)}",
                ),
                reverse=True,
            )
        else:
            episodes = list(reversed(episodes))
        return episodes[: max(1, limit)]

    def get_episode(self, episode_id: str) -> EpisodeRecord | None:
        """Return a single episode by id."""
        wanted = episode_id.strip().lower()
        for episode in self.list_episodes(limit=100000):
            if episode.id.lower() == wanted:
                return episode
        return None

    def verify_chain(self) -> ChainVerification:
        """Verify IDs, prev_hash links, and body hashes."""
        errors: list[str] = []
        prev_hash = ""
        checked = 0
        if not os.path.isfile(self.path):
            return ChainVerification(ok=True, checked=0)
        with open(self.path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"line {line_no}: invalid json: {exc}")
                    continue
                episode = EpisodeRecord.from_dict(data)
                checked += 1
                if not re.match(r"^E\d{3,}$", episode.id):
                    errors.append(f"line {line_no}: invalid episode id {episode.id!r}")
                if episode.prev_hash != prev_hash:
                    errors.append(f"line {line_no}: prev_hash mismatch")
                expected = _json_hash(data)
                if episode.hash != expected:
                    errors.append(f"line {line_no}: hash mismatch")
                prev_hash = episode.hash
        return ChainVerification(ok=not errors, checked=checked, errors=errors)

    @staticmethod
    def _next_id(episodes: list[EpisodeRecord]) -> str:
        max_id = 0
        for episode in episodes:
            match = re.match(r"E(\d+)$", episode.id)
            if match:
                max_id = max(max_id, int(match.group(1)))
        return f"E{max_id + 1:03d}"
