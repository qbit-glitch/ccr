"""EvolutionMixin — A-MEM evolved summary storage and trigger logic.

Methods: _evolved_path (property), _load_evolved_summaries, _save_evolved_summaries,
get_evolved_what, _evolve_commit_summary, _trigger_memory_evolution.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from ccr.core.memory_pkg.memory_types import EvolvedSummary
from ccr.utils.parsing import extract_json_string

logger = logging.getLogger(__name__)


class EvolutionMixin:
    """A-MEM evolved summary storage for MemoryManager.

    Implements A-MEM S3.3 Eq.7: evolved summaries are LLM-rewritten
    overlays on existing commits triggered by new related work.
    """

    # --- A-MEM Evolved Summary Storage (S3.3 Eq.7) ---

    @property
    def _evolved_path(self) -> str:
        return os.path.join(self.ccr_root, "evolved_summaries.json")

    def _load_evolved_summaries(self) -> None:
        """Load evolved summary overlays from JSON. Non-destructive if missing."""
        path = self._evolved_path
        with self._locks[path], self._file_lock(path):
            raw = self._read_file_unlocked(path)
        if not raw:
            return
        try:
            data = json.loads(raw)
            entries = data.get("evolved", {})
            for commit_id, ev in entries.items():
                self._evolved_summaries[commit_id] = EvolvedSummary(
                    commit_id=ev["commit_id"],
                    evolved_what=ev["evolved_what"],
                    evolution_reason=ev["evolution_reason"],
                    evolved_at=ev["evolved_at"],
                    source_commit_id=ev["source_commit_id"],
                    original_what=ev["original_what"],
                )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to load %s: %s", path, exc)

    def _save_evolved_summaries(self) -> None:
        """Persist the evolved_summaries dict to JSON."""
        path = self._evolved_path
        data = {
            "version": 1,
            "evolved": {
                cid: {
                    "commit_id": ev.commit_id,
                    "evolved_what": ev.evolved_what,
                    "evolution_reason": ev.evolution_reason,
                    "evolved_at": ev.evolved_at,
                    "source_commit_id": ev.source_commit_id,
                    "original_what": ev.original_what,
                }
                for cid, ev in self._evolved_summaries.items()
            },
        }
        content = json.dumps(data, indent=2, ensure_ascii=False)
        with self._locks[path], self._file_lock(path):
            self._write_file_unlocked(path, content)

    def get_evolved_what(self, commit_id: str) -> str | None:
        """Return evolved summary if available, else None (A-MEM S3.3 Eq.7)."""
        ev = self._evolved_summaries.get(commit_id)
        return ev.evolved_what if ev else None

    def _evolve_commit_summary(
        self, existing_commit: dict, new_commit: dict
    ) -> EvolvedSummary | None:
        """Rewrite existing commit's 'what' to incorporate context from new_commit.

        Implements A-MEM S3.3 Eq.7: m_tilde_i = f_LLM(m_i, m')
        Returns None when sub_client is unavailable or an error occurs.
        When sub_client is None, applies a text fallback: deduplicates sentences
        in the 'what' field, returning an EvolvedSummary only if dedup removed
        at least one sentence.
        """
        if self.sub_client is None:
            # Text fallback: dedup sentences in existing_commit's 'what' field
            what = existing_commit.get("what", "")
            sentences = what.split(". ")
            deduped = list(dict.fromkeys(s for s in sentences if s))
            if len(deduped) < len(sentences):
                evolved_what = ". ".join(deduped)
                now_iso = datetime.now(timezone.utc).isoformat()
                existing_id = existing_commit.get("id", "")
                new_id = new_commit.get("id", "")
                return EvolvedSummary(
                    commit_id=existing_id,
                    evolved_what=evolved_what,
                    evolution_reason="dedup-fallback: removed duplicate sentences",
                    evolved_at=now_iso,
                    source_commit_id=new_id,
                    original_what=what,
                )
            return None
        try:
            prompt = (
                "You are updating a project memory entry based on new related work.\n\n"
                f"Original entry:\n"
                f"Title: {existing_commit.get('title', '')}\n"
                f"What: {existing_commit.get('what', '')}\n"
                f"Why: {existing_commit.get('why', '')}\n\n"
                f"New related work arrived:\n"
                f"Title: {new_commit.get('title', '')}\n"
                f"What: {new_commit.get('what', '')}\n"
                f"Why: {new_commit.get('why', '')}\n\n"
                "Rewrite the original \"What\" field to incorporate relevant context "
                "from the new work.\n"
                "Keep it concise (1-3 sentences). Only update if the new work adds "
                "meaningful context.\n"
                'Respond with a JSON object: {"evolved_what": "...", "evolution_reason": "..."}'
            )
            response = self.sub_client.completion([{"role": "user", "content": prompt}])
            raw_json = extract_json_string(response)
            parsed = json.loads(raw_json)
            evolved_what = parsed.get("evolved_what", "").strip()
            evolution_reason = parsed.get("evolution_reason", "").strip()
            if not evolved_what:
                return None
            now_iso = datetime.now(timezone.utc).isoformat()
            existing_id = existing_commit.get("id", "")
            new_id = new_commit.get("id", "")
            return EvolvedSummary(
                commit_id=existing_id,
                evolved_what=evolved_what,
                evolution_reason=evolution_reason,
                evolved_at=now_iso,
                source_commit_id=new_id,
                original_what=existing_commit.get("what", ""),
            )
        except Exception:
            return None  # Never fail — evolution is supplementary

    def _trigger_memory_evolution(self, new_commit_id: str, links: list) -> None:
        """Evolve related commit summaries when a new commit arrives.

        Implements A-MEM S3.3: fires on semantic/supersession links with score > 0.5.
        Caps at 3 evolutions per commit to avoid LLM overuse.
        """
        try:
            branch = self.get_active_branch()
            new_block = self._find_commit_by_id(branch, new_commit_id)
            if not new_block:
                return
            new_commit = self._parse_commit_block(new_block)
            new_commit["id"] = new_commit_id

            evolution_count = 0
            for link in links:
                if evolution_count >= 3:
                    break
                # Only evolve on semantic or supersession links above threshold
                link_type = link.link_type if hasattr(link, "link_type") else link.get("link_type", "")
                score = link.score if hasattr(link, "score") else link.get("score", 0.0)
                if link_type not in ("semantic", "supersession"):
                    continue
                if score <= 0.5:
                    continue
                existing_id = link.target if hasattr(link, "target") else link.get("target", "")
                if not existing_id:
                    continue
                existing_block = self._find_commit_by_id(branch, existing_id)
                if not existing_block:
                    continue
                existing_commit = self._parse_commit_block(existing_block)
                existing_commit["id"] = existing_id

                result = self._evolve_commit_summary(existing_commit, new_commit)
                if result is not None:
                    self._evolved_summaries[existing_id] = result
                    self._save_evolved_summaries()
                    evolution_count += 1
        except Exception:
            pass  # Evolution is supplementary — never fail
