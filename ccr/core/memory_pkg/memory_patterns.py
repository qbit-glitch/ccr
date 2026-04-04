"""PatternsMixin — CER-inspired pattern buffer with EvolveR quality scoring.

Methods: _get_patterns_path, _load_patterns, _save_patterns,
_find_matching_pattern, _process_patterns, _enforce_pattern_buffer_size,
_scan_pending_promotions, mark_pattern_promoted_by_content,
update_pattern_quality, get_patterns, _pattern_recency_weight_at.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone

from ccr.context.embeddings import quick_cosine

logger = logging.getLogger(__name__)


class PatternsMixin:
    """Pattern buffer management for MemoryManager.

    Implements CER S3.1 Dynamic Experience Buffer with EvolveR
    (arXiv:2510.16079) quality-scored patterns and Bayesian scoring.
    """

    def _get_patterns_path(self) -> str:
        return os.path.join(self.ccr_root, "patterns.json")

    def _load_patterns(self) -> dict:
        """Load pattern buffer from JSON. Returns default if missing/corrupt."""
        path = self._get_patterns_path()
        with self._locks[path], self._file_lock(path):
            raw = self._read_file_unlocked(path)
        if not raw:
            return {"version": 1, "patterns": {}, "next_id": 1}
        try:
            data = json.loads(raw)
            if not isinstance(data.get("patterns"), dict):
                return {"version": 1, "patterns": {}, "next_id": 1}
            return data
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to load %s: %s", path, exc)
            return {"version": 1, "patterns": {}, "next_id": 1}

    def _save_patterns(self, data: dict) -> None:
        """Atomically save the pattern buffer."""
        path = self._get_patterns_path()
        content = json.dumps(data, indent=2, ensure_ascii=False)
        with self._locks[path], self._file_lock(path):
            self._write_file_unlocked(path, content)

    def _find_matching_pattern(self, data: dict, new_text: str) -> str | None:
        """Find existing pattern matching new_text. Primary: ONNX cosine. Fallback: word Jaccard.

        CER S3.1: existing buffer shown to distiller to avoid repetition.
        Returns matching pattern ID or None.
        """
        new_words = {w.lower() for w in new_text.split()
                     if w.lower() not in self._STOP_WORDS and len(w) > 2}
        if len(new_words) < 2:
            return None

        # Try ONNX cosine similarity first
        best_id = None
        best_sim = 0.0

        for pid, entry in data.get("patterns", {}).items():
            existing_text = entry["text"]
            existing_words = {w.lower() for w in existing_text.split()
                              if w.lower() not in self._STOP_WORDS and len(w) > 2}
            if len(existing_words) < 2:
                continue

            # ONNX primary, Jaccard fallback
            onnx_sim = quick_cosine(new_text, existing_text)
            sim = onnx_sim if onnx_sim is not None else self._jaccard(new_words, existing_words)

            if sim >= self.config.pattern_dedup_threshold and sim > best_sim:
                best_sim = sim
                best_id = pid

        return best_id

    def _process_patterns(
        self,
        commit_id: str,
        patterns: list[str],
        timestamp: str,
    ) -> list[dict]:
        """Process new patterns: dedup, store, track occurrences, suggest promotions.

        CER S3.1 Dynamic Experience Buffer: new skills are deduped against existing
        buffer (existing experiences shown to distiller to avoid repetition).

        Returns list of promotion suggestion dicts for patterns that crossed
        the promotion threshold.
        """
        path = self._get_patterns_path()
        with self._locks[path], self._file_lock(path):
            raw = self._read_file_unlocked(path)
            if not raw:
                data: dict = {"version": 1, "patterns": {}, "next_id": 1}
            else:
                try:
                    data = json.loads(raw)
                    if not isinstance(data.get("patterns"), dict):
                        data = {"version": 1, "patterns": {}, "next_id": 1}
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning("Failed to load %s: %s", path, exc)
                    data = {"version": 1, "patterns": {}, "next_id": 1}

            promotion_suggestions: list[dict] = []

            for pattern_text in patterns:
                pattern_text = pattern_text.strip()
                if not pattern_text:
                    continue

                # Dedup: find existing pattern with word Jaccard >= threshold
                matched_id = self._find_matching_pattern(data, pattern_text)

                if matched_id:
                    # Update existing pattern's occurrence
                    entry = data["patterns"][matched_id]
                    if commit_id not in entry["commit_ids"]:
                        entry["commit_ids"].append(commit_id)
                        entry["occurrence_count"] = len(entry["commit_ids"])
                    # CER recency tracking: update last_seen timestamp
                    entry["last_seen"] = timestamp

                    # Check promotion threshold
                    if (entry["occurrence_count"] >= self.config.pattern_promotion_count
                            and not entry.get("promoted", False)):
                        promotion_suggestions.append({
                            "pattern_id": matched_id,
                            "text": entry["text"],
                            "count": entry["occurrence_count"],
                            "commit_ids": entry["commit_ids"],
                        })
                else:
                    # New pattern — add to buffer
                    pid = f"P{data['next_id']:03d}"
                    data["next_id"] = data["next_id"] + 1
                    data["patterns"][pid] = {
                        "text": pattern_text,
                        "first_seen": commit_id,
                        "commit_ids": [commit_id],
                        "occurrence_count": 1,
                        "created_at": timestamp,
                        "promoted": False,
                        "success_count": 0,
                        "failure_count": 0,
                        "quality_score": 0.5,
                        "last_quality_update": "",
                    }

            # Buffer size enforcement (CER S3.1 Dynamic Buffer)
            self._enforce_pattern_buffer_size(data)

            content = json.dumps(data, indent=2, ensure_ascii=False)
            self._write_file_unlocked(path, content)

        return promotion_suggestions

    def _enforce_pattern_buffer_size(self, data: dict) -> None:
        """Evict lowest-value patterns if buffer exceeds max size."""
        patterns = data.get("patterns", {})
        max_size = self.config.pattern_max_buffer_size
        if len(patterns) <= max_size:
            return

        # Sort by (quality_score ASC, occurrence_count ASC, created_at ASC)
        # Evict lowest quality + least frequent + oldest first
        sorted_ids = sorted(
            patterns.keys(),
            key=lambda pid: (
                patterns[pid].get("quality_score", 0.5),
                patterns[pid].get("occurrence_count", 1),
                patterns[pid].get("created_at", ""),
            ),
        )

        # Evict from the front (lowest value) until within budget
        evict_count = len(patterns) - max_size
        for pid in sorted_ids[:evict_count]:
            del patterns[pid]

    def _scan_pending_promotions(self) -> list[dict]:
        """Scan the pattern buffer for patterns that crossed the promotion threshold.

        CER S3.1 (CCR extension): surfaces ready-to-promote patterns even when no
        new patterns_learned are passed to commit(). This ensures promotable patterns
        are not silently ignored when the caller omits patterns_learned.

        Returns a list of promotion suggestion dicts (same shape as _process_patterns):
            {pattern_id, text, count, commit_ids}

        Capped at 5 results to avoid flooding the commit response.
        Does NOT mark patterns as promoted — that only happens when the user
        explicitly calls ace_apply_delta.
        """
        data = self._load_patterns()
        threshold = self.config.pattern_promotion_count

        suggestions: list[dict] = []
        for pid, entry in data.get("patterns", {}).items():
            if (entry.get("occurrence_count", 0) >= threshold
                    and not entry.get("promoted", False)):
                suggestions.append({
                    "pattern_id": pid,
                    "text": entry["text"],
                    "count": entry["occurrence_count"],
                    "commit_ids": entry.get("commit_ids", []),
                })

        # Sort by occurrence_count DESC for deterministic ordering
        suggestions.sort(key=lambda x: -x["count"])

        return suggestions[:5]

    def mark_pattern_promoted_by_content(self, text: str) -> int:
        """Mark patterns matching `text` as promoted (CER buffer management).

        Called after ace_apply_delta ADD to close the loop: if the user added
        a bullet matching a pending pattern, that pattern is now promoted.

        Returns: number of patterns marked promoted.
        """
        path = self._get_patterns_path()
        with self._locks[path], self._file_lock(path):
            raw = self._read_file_unlocked(path)
            if not raw:
                return 0
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Failed to load %s: %s", path, exc)
                return 0

            matched_id = self._find_matching_pattern(data, text)
            if matched_id is None:
                return 0

            entry = data["patterns"][matched_id]
            if entry.get("promoted", False):
                return 0  # Already promoted

            entry["promoted"] = True
            self._write_file_unlocked(path, json.dumps(data, indent=2))
            return 1

    def update_pattern_quality(self, pattern_text: str, success: bool) -> bool:
        """Update quality score for a pattern based on its promoted bullet's performance.

        Called when ace_update_counters tags a bullet that was promoted from a pattern.
        Uses Bayesian average: (success+1) / (success+failure+2).

        Inspired by EvolveR (arXiv:2510.16079) quality-scored pattern buffer.

        Args:
            pattern_text: The pattern text to find (uses word Jaccard matching).
            success: True if bullet was tagged helpful, False if harmful.

        Returns:
            True if a matching pattern was found and updated.
        """
        path = self._get_patterns_path()
        with self._locks[path], self._file_lock(path):
            raw = self._read_file_unlocked(path)
            if not raw:
                return False
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Failed to load %s: %s", path, exc)
                return False

            # Find matching pattern by word Jaccard
            match_pid = self._find_matching_pattern(data, pattern_text)
            if match_pid is None:
                return False

            p = data["patterns"][match_pid]
            # Ensure quality fields exist (backward compat with old format)
            p.setdefault("success_count", 0)
            p.setdefault("failure_count", 0)
            if success:
                p["success_count"] += 1
            else:
                p["failure_count"] += 1

            # Recompute Bayesian quality score
            sc = p["success_count"]
            fc = p["failure_count"]
            p["quality_score"] = (sc + 1) / (sc + fc + 2)
            p["last_quality_update"] = datetime.now(timezone.utc).isoformat()

            self._write_file_unlocked(
                path, json.dumps(data, indent=2, ensure_ascii=False)
            )
            return True

    def get_patterns(
        self,
        min_occurrences: int = 1,
        include_promoted: bool = True,
        search_term: str | None = None,
    ) -> dict:
        """Query the pattern buffer. Returns dict for MCP tool formatting.

        CER-inspired recency-weighted retrieval: patterns seen recently
        are ranked higher. Combines quality_score with temporal decay
        on last_seen timestamp (lambda=0.005/hour, half-life ~139h).
        """
        data = self._load_patterns()
        results = []
        # Snapshot current time once to avoid floating point drift between entries
        now = datetime.now(timezone.utc)

        for pid, entry in data.get("patterns", {}).items():
            if entry.get("occurrence_count", 1) < min_occurrences:
                continue
            if not include_promoted and entry.get("promoted", False):
                continue
            if search_term:
                if search_term.lower() not in entry["text"].lower():
                    continue
            # CER recency-weighted retrieval: compute effective score
            quality = entry.get("quality_score", 0.5)
            last_seen = entry.get("last_seen", entry.get("created_at", ""))
            recency_weight = self._pattern_recency_weight_at(last_seen, now)
            entry["effective_score"] = quality * recency_weight
            results.append({"id": pid, **entry})

        # Sort by effective_score DESC (quality * recency), then occurrence_count DESC
        # (EvolveR + CER: high-quality recent patterns surface first)
        results.sort(key=lambda x: (
            -x.get("effective_score", 0.5),
            -x.get("occurrence_count", 1),
        ))

        return {
            "total": len(data.get("patterns", {})),
            "matching": len(results),
            "patterns": results,
        }

    @staticmethod
    def _pattern_recency_weight_at(timestamp_str: str, now: datetime) -> float:
        """Compute recency weight for pattern retrieval (CER-inspired).

        Uses decay formula: exp(-0.005 * hours). Half-life ~139h.
        Returns 1.0 for recent patterns, decaying toward 0 for old ones.
        `now` is passed in to avoid floating point drift between entries.
        """
        if not timestamp_str:
            return 0.5  # Default moderate weight for undated patterns
        try:
            ts = datetime.strptime(timestamp_str.strip(), "%Y-%m-%d %H:%M")
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            hours = (now - ts).total_seconds() / 3600.0
            return math.exp(-0.005 * max(0.0, hours))
        except (ValueError, TypeError):
            return 0.5
