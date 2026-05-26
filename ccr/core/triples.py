"""Semantic triple extraction from commit text (Memori-inspired).

Inspired by Memori (arXiv:2603.19935) semantic triple representation.
Uses regex patterns for zero-LLM extraction from structured commit fields.
"""

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class Triple:
    """A semantic triple (subject-predicate-object)."""
    subject: str
    predicate: str
    object: str
    source_commit: str
    confidence: float = 0.8
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "source_commit": self.source_commit,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Triple":
        return cls(
            subject=d["subject"],
            predicate=d["predicate"],
            object=d["object"],
            source_commit=d["source_commit"],
            confidence=d.get("confidence", 0.8),
            timestamp=d.get("timestamp", ""),
        )

    def format_compact(self) -> str:
        """Format as compact triple string."""
        return f"{self.subject} --{self.predicate}--> {self.object} ({self.source_commit})"


# Extraction patterns: (regex, predicate_name, confidence)
# Each regex should capture exactly 2 groups (subject, object)
_EXTRACTION_PATTERNS: list[tuple[str, str, float]] = [
    (r"[Aa]dded\s+(.+?)\s+to\s+(.+?)(?:\.|,|$)", "added_to", 0.8),
    (r"[Rr]efactored\s+(.+?)\s+into\s+(.+?)(?:\.|,|$)", "refactored_into", 0.8),
    (r"[Ff]ix(?:ed)?\s+(.+?)\s+in\s+(.+?)(?:\.|,|$)", "fixed_in", 0.8),
    (r"[Rr]eplaced\s+(.+?)\s+with\s+(.+?)(?:\.|,|$)", "replaced_by", 0.8),
    (r"[Cc]reated\s+(.+?)\s+for\s+(.+?)(?:\.|,|$)", "created_for", 0.8),
    (r"[Ii]mplemented\s+(.+?)\s+(?:in|for|using)\s+(.+?)(?:\.|,|$)", "implemented_in", 0.7),
    (r"[Uu]pdated\s+(.+?)\s+to\s+(.+?)(?:\.|,|$)", "updated_to", 0.7),
    (r"[Rr]emoved\s+(.+?)\s+from\s+(.+?)(?:\.|,|$)", "removed_from", 0.8),
    (r"[Mm]igrated\s+(.+?)\s+to\s+(.+?)(?:\.|,|$)", "migrated_to", 0.8),
    (r"[Ss]plit\s+(.+?)\s+into\s+(.+?)(?:\.|,|$)", "split_into", 0.7),
    (r"[Mm]erged\s+(.+?)\s+(?:into|with)\s+(.+?)(?:\.|,|$)", "merged_into", 0.7),
    (r"[Rr]enamed\s+(.+?)\s+to\s+(.+?)(?:\.|,|$)", "renamed_to", 0.9),
    (r"[Mm]oved\s+(.+?)\s+to\s+(.+?)(?:\.|,|$)", "moved_to", 0.8),
]

# Stop words for subject/object cleaning
_STOP_PREFIXES = {"the ", "a ", "an ", "all ", "some ", "new ", "old "}


def _clean_entity(text: str) -> str:
    """Clean extracted entity text."""
    text = text.strip()
    changed = True
    while changed:
        changed = False
        for prefix in sorted(_STOP_PREFIXES):  # sorted for determinism
            if text.lower().startswith(prefix):
                text = text[len(prefix):]
                changed = True
    return text.strip()


class TripleStore:
    """Zero-LLM triple store with regex extraction and word Jaccard search."""

    def __init__(self, path: str, max_buffer_size: int = 500):
        self._path = path
        self._max_buffer_size = max_buffer_size
        self._lock = threading.Lock()
        self._triples: list[Triple] = []
        self._triple_keys: set[tuple[str, str, str]] = set()
        self._load()

    def _load(self) -> None:
        """Load triples from disk."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._triples = [Triple.from_dict(t) for t in data.get("triples", [])]
            self._triple_keys = {
                (t.subject, t.predicate, t.object) for t in self._triples
            }
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning("Failed to load %s: %s", self._path, exc)

    def _save(self) -> None:
        """Atomic write to disk."""
        data = {
            "version": 1,
            "triples": [t.to_dict() for t in self._triples],
        }
        tmp_path = self._path + ".tmp"
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self._path)

    def extract_from_commit(
        self, commit_id: str, title: str, what: str, why: str, files: list[str]
    ) -> list[Triple]:
        """Extract triples from commit fields using regex patterns.

        Returns list of newly extracted triples.
        """
        now = datetime.now(timezone.utc).isoformat()
        candidates: list[Triple] = []

        # Extract from title + what + why
        for text in [title, what, why]:
            if not text:
                continue
            for pattern, predicate, confidence in _EXTRACTION_PATTERNS:
                for match in re.finditer(pattern, text):
                    subject = _clean_entity(match.group(1))
                    obj = _clean_entity(match.group(2))
                    if subject and obj and len(subject) > 1 and len(obj) > 1:
                        candidates.append(Triple(
                            subject=subject,
                            predicate=predicate,
                            object=obj,
                            source_commit=commit_id,
                            confidence=confidence,
                            timestamp=now,
                        ))

        # Extract file-level triples
        for filepath in files:
            candidates.append(Triple(
                subject=filepath,
                predicate="modified_in",
                object=commit_id,
                source_commit=commit_id,
                confidence=1.0,
                timestamp=now,
            ))

        # Dedup + extend + save under a single lock to prevent race conditions
        new_triples: list[Triple] = []
        with self._lock:
            for triple in candidates:
                if not self._is_duplicate(triple):
                    self._triples.append(triple)
                    self._triple_keys.add((triple.subject, triple.predicate, triple.object))
                    new_triples.append(triple)
            if new_triples:
                self._enforce_buffer_size()
                self._save()

        return new_triples

    def _is_duplicate(self, new: Triple) -> bool:
        """O(1) check: triple already exists (same subject+predicate+object)."""
        return (new.subject, new.predicate, new.object) in self._triple_keys

    def _enforce_buffer_size(self) -> None:
        """Evict lowest-value triples when buffer exceeds max size.

        Must be called under self._lock. Sorts by (confidence ASC, timestamp ASC)
        and evicts from the front (lowest value first).
        """
        if len(self._triples) <= self._max_buffer_size:
            return
        sorted_triples = sorted(
            self._triples, key=lambda t: (t.confidence, t.timestamp)
        )
        evict_count = len(sorted_triples) - self._max_buffer_size
        evicted = sorted_triples[:evict_count]
        self._triples = sorted_triples[evict_count:]
        for t in evicted:
            self._triple_keys.discard((t.subject, t.predicate, t.object))

    def get_recent(self, top_k: int = 10) -> list[Triple]:
        """Return most recent triples sorted by timestamp descending."""
        with self._lock:
            return sorted(
                self._triples, key=lambda t: t.timestamp, reverse=True
            )[:top_k]

    def search(self, query: str, top_k: int = 10) -> list[Triple]:
        """Search triples by word Jaccard similarity against query."""
        query_words = set(query.lower().split())
        if not query_words:
            return []

        scored = []
        with self._lock:
            for triple in self._triples:
                triple_text = f"{triple.subject} {triple.predicate} {triple.object}"
                triple_words = set(triple_text.lower().split())
                intersection = query_words & triple_words
                union = query_words | triple_words
                jaccard = len(intersection) / len(union) if union else 0
                if jaccard > 0.05:
                    scored.append((jaccard * triple.confidence, triple))

        scored.sort(key=lambda x: -x[0])
        return [t for _, t in scored[:top_k]]

    def get_by_commit(self, commit_id: str) -> list[Triple]:
        """Get all triples from a specific commit."""
        with self._lock:
            return [t for t in self._triples if t.source_commit == commit_id]

    def get_by_entity(self, entity: str) -> list[Triple]:
        """Get all triples involving an entity (as subject or object)."""
        entity_lower = entity.lower()
        with self._lock:
            return [
                t for t in self._triples
                if entity_lower in t.subject.lower() or entity_lower in t.object.lower()
            ]

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._triples)

    def format_for_context(self, top_k: int = 10) -> str:
        """Format top triples for inclusion in gcc_context output."""
        with self._lock:
            if not self._triples:
                return ""
            # Return most recent triples
            recent = sorted(self._triples, key=lambda t: t.timestamp, reverse=True)[:top_k]

        lines = ["# Knowledge Graph (recent relationships)"]
        for t in recent:
            lines.append(f"- {t.format_compact()}")
        return "\n".join(lines)
