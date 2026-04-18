"""Playbook mutations mixin — failure lesson management and skill evolution.

Contains methods for evolving new skills from failure lessons, serializing
with failure data, and loading/saving failure lesson companion files.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ccr.ace.playbook_types import Bullet, DeltaOperation, FailureLesson, _get_slug


class MutationsMixin:
    """Mixin providing failure lesson and evolution methods for Playbook.

    Requires the host class to have:
        self._bullets: list[Bullet]
        self._sections: list[str]
        self._next_id: int
        self._id_index: dict[str, Bullet]
        self.get_bullet(bullet_id) -> Bullet | None
        self.check_evolution_needed(threshold) -> dict
    """

    def evolve_from_failures(self, threshold: int = 3) -> list[Bullet]:
        """Generate NEW standalone skill bullets from accumulated failure lessons.

        Inspired by SkillRL §3.3 (evolution triggered when Acc(C) < delta) and
        Prompt B.1 ("Generate 1-3 NEW actionable skills"). However, the paper's
        mechanism uses a teacher model M_T for cross-lesson synthesis across
        related failures. CCR does verbatim extraction: each failure lesson's
        prevention_principle becomes a new bullet directly — no cross-lesson
        generalization or LLM synthesis.

        Each new bullet is placed in PROBLEM-SOLVING HEURISTICS with scope="general"
        and when_to_apply derived from the task_context.

        Args:
            threshold: Minimum harmful-with-lessons bullets to trigger. If fewer
                       exist, returns empty list (evolution not needed yet).

        Returns:
            List of newly created Bullet objects added to the playbook.
        """
        check = self.check_evolution_needed(threshold)
        if not check["needed"]:
            return []

        new_bullets: list[Bullet] = []
        # N2: Seed with existing bullet content to avoid duplicating existing skills
        # Per SkillRL Prompt B.1: "existing_titles" are provided to avoid duplicates
        seen_principles: set[str] = {b.content.strip().lower() for b in self._bullets}

        for bullet in list(self._bullets):  # snapshot — we append during iteration
            if bullet.harmful <= 0 or not bullet.has_failure_lessons:
                continue
            for fl in bullet.failure_lessons:
                # N3: Skip already-evolved lessons for idempotency
                if fl.evolved:
                    continue
                principle = fl.prevention_principle.strip()
                if not principle:
                    continue
                # Deduplicate by normalized principle text
                norm = principle.lower()
                if norm in seen_principles:
                    # Still mark as evolved even if deduplicated
                    fl.evolved = True
                    continue
                seen_principles.add(norm)

                # Create new skill bullet from the prevention principle
                section = "PROBLEM-SOLVING HEURISTICS"
                if section not in self._sections:
                    self._sections.append(section)

                slug = _get_slug(section)
                bullet_id = f"{slug}-{self._next_id:05d}"
                self._next_id += 1

                new_bullet = Bullet(
                    id=bullet_id,
                    helpful=0,
                    harmful=0,
                    content=principle,
                    section=section,
                    scope="general",
                    when_to_apply=fl.task_context if fl.task_context else f"When facing issues similar to: {fl.failure_point[:80]}",
                )
                self._bullets.append(new_bullet)
                self._id_index[new_bullet.id] = new_bullet
                new_bullets.append(new_bullet)
                fl.evolved = True  # N3: Mark as evolved

        return new_bullets

    def serialize_with_failures(self) -> str:
        """Serialize playbook with inline failure lessons for display.

        Uses the standard format but appends failure lessons beneath
        harmful bullets for human/agent readability.
        """
        lines: list[str] = []

        for section in self._sections:
            lines.append(f"## {section}")
            section_bullets = [b for b in self._bullets if b.section == section]
            for bullet in section_bullets:
                lines.append(bullet.format_line_with_failures())
            lines.append("")  # blank line after section

        # Orphans
        orphans = [b for b in self._bullets if b.section not in self._sections]
        if orphans:
            if "OTHERS" not in self._sections:
                lines.append("## OTHERS")
            for bullet in orphans:
                lines.append(bullet.format_line_with_failures())
            lines.append("")

        return "\n".join(lines).rstrip()

    def load_failure_lessons(self, path: str) -> int:
        """Load extended bullet data from companion JSON file.

        Handles two formats for backward compatibility:
            Old format: {"bullet_id": [lesson_dict, ...]}
            New format: {"bullet_id": {"lessons": [...], "scope": "...", "when_to_apply": "..."}}

        Args:
            path: Path to failure_lessons.json.

        Returns:
            Number of lessons loaded.
        """
        if not os.path.isfile(path):
            return 0

        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                return 0

        loaded = 0
        for bullet in self._bullets:
            entry = data.get(bullet.id)
            if entry is None:
                continue
            # Backward compat: old format is a bare list of lessons
            if isinstance(entry, list):
                lessons_raw = entry
                # No extended fields in old format
            elif isinstance(entry, dict):
                lessons_raw = entry.get("lessons", [])
                # N1: Restore scope, when_to_apply, and last_updated
                if "scope" in entry:
                    bullet.scope = entry["scope"]
                if "when_to_apply" in entry:
                    bullet.when_to_apply = entry["when_to_apply"]
                if "last_updated" in entry:
                    bullet.last_updated = entry["last_updated"]
                # ERL: Restore trigger/action fields
                if "trigger" in entry:
                    bullet.trigger = entry["trigger"]
                if "action" in entry:
                    bullet.action = entry["action"]
                # Restore contribution-weighted counters (AgentEvolver-inspired)
                bullet.weighted_helpful = float(entry.get("weighted_helpful", 0.0))
                bullet.weighted_harmful = float(entry.get("weighted_harmful", 0.0))
                # Restore SM-2 adaptive decay rate
                bullet.personal_decay_rate = float(entry.get("personal_decay_rate", 0.0))
            else:
                continue
            if isinstance(lessons_raw, list):
                bullet.failure_lessons = [
                    FailureLesson.from_dict(d) for d in lessons_raw
                    if isinstance(d, dict)
                ]
                loaded += len(bullet.failure_lessons)
        return loaded

    def save_failure_lessons(self, path: str) -> int:
        """Save extended bullet data (failure lessons, scope, when_to_apply) to companion JSON.

        Uses new format that stores all extended fields per bullet:
            {"bullet_id": {"lessons": [...], "scope": "...", "when_to_apply": "..."}}

        Args:
            path: Path to failure_lessons.json.

        Returns:
            Number of lessons saved.
        """
        data: dict[str, dict[str, Any]] = {}
        total = 0
        for bullet in self._bullets:
            has_extended = (
                bullet.failure_lessons
                or bullet.scope != "general"
                or bullet.when_to_apply
                or bullet.last_updated
                or bullet.trigger
                or bullet.action
                or bullet.weighted_helpful > 0
                or bullet.weighted_harmful > 0
                or bullet.personal_decay_rate > 0.0
            )
            if has_extended:
                entry: dict[str, Any] = {}
                if bullet.failure_lessons:
                    entry["lessons"] = [fl.to_dict() for fl in bullet.failure_lessons]
                    total += len(bullet.failure_lessons)
                if bullet.scope != "general":
                    entry["scope"] = bullet.scope
                if bullet.when_to_apply:
                    entry["when_to_apply"] = bullet.when_to_apply
                if bullet.last_updated:
                    entry["last_updated"] = bullet.last_updated
                if bullet.trigger:
                    entry["trigger"] = bullet.trigger
                if bullet.action:
                    entry["action"] = bullet.action
                # Persist contribution-weighted counters (AgentEvolver-inspired)
                if bullet.weighted_helpful > 0:
                    entry["weighted_helpful"] = bullet.weighted_helpful
                if bullet.weighted_harmful > 0:
                    entry["weighted_harmful"] = bullet.weighted_harmful
                # Persist SM-2 adaptive decay rate
                if bullet.personal_decay_rate > 0.0:
                    entry["personal_decay_rate"] = bullet.personal_decay_rate
                if entry:
                    data[bullet.id] = entry

        dir_name = os.path.dirname(path)
        os.makedirs(dir_name, exist_ok=True)
        # H5: Atomic write via tmp + fsync + os.replace
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return total

    def get_failure_lessons_for_bullet(self, bullet_id: str) -> list[FailureLesson]:
        """Get all failure lessons for a specific bullet."""
        bullet = self.get_bullet(bullet_id)
        if bullet is None:
            return []
        return list(bullet.failure_lessons)

    def get_all_prevention_principles(self) -> list[tuple[str, str]]:
        """Extract all prevention principles across the playbook.

        Returns:
            List of (bullet_id, prevention_principle) tuples.
        """
        principles = []
        for bullet in self._bullets:
            for fl in bullet.failure_lessons:
                if fl.prevention_principle:
                    principles.append((bullet.id, fl.prevention_principle))
        return principles
