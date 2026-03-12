"""Playbook data structure — structured, itemized bullets with helpful/harmful counters.

Per ACE paper (§3.1): Context is a collection of structured, itemized bullets rather than
a monolithic prompt. Each bullet has metadata (unique ID, helpful/harmful counters) and
content (a reusable strategy, domain concept, or common failure mode).

Extended with Structured Failure Lessons (SkillRL-inspired): When a strategy is tagged
harmful, an optional failure lesson captures *why* it failed, *what* should have been done,
and a generalizable prevention principle. Stored in .ccr/failure_lessons.json.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# Standard playbook sections (from ACE paper Figure 3 + official repo)
DEFAULT_SECTIONS = [
    "STRATEGIES & INSIGHTS",
    "CODE SNIPPETS & TEMPLATES",
    "COMMON MISTAKES TO AVOID",
    "PROBLEM-SOLVING HEURISTICS",
    "CONTEXT CLUES & INDICATORS",
    "OTHERS",
]

# Section name → slug prefix mapping
_SLUG_MAP = {
    "strategies_&_insights": "str",
    "strategies_and_insights": "str",
    "code_snippets_&_templates": "code",
    "code_snippets_and_templates": "code",
    "common_mistakes_to_avoid": "mis",
    "problem-solving_heuristics": "heu",
    "problem_solving_heuristics": "heu",
    "context_clues_&_indicators": "ctx",
    "context_clues_and_indicators": "ctx",
    "formulas_&_calculations": "cal",
    "formulas_and_calculations": "cal",
    "others": "oth",
    "general": "gen",
}


def _normalize_section(name: str) -> str:
    """Normalize section name for lookup."""
    return name.lower().replace(" ", "_").replace("&", "and")


def _get_slug(section: str) -> str:
    """Get the ID slug prefix for a section."""
    normalized = _normalize_section(section)
    return _SLUG_MAP.get(normalized, normalized[:3])


@dataclass
class FailureLesson:
    """A structured failure record for a harmful strategy (SkillRL-inspired).

    Per SkillRL §3.1 (Eq. 3): Failed trajectories τ⁻ are distilled into concise
    failure lessons s⁻ = M_T(τ⁻, d) identifying: (1) the point of failure,
    (2) the flawed reasoning or action, (3) what should have been done, and
    (4) general principles to prevent similar failures.

    Extended with task_context to capture the trajectory context (SkillRL P0-2/P0-3).
    """

    failure_point: str  # Where exactly the strategy broke down (Table 6: "Failure Description")
    flawed_reasoning: str  # What incorrect assumption led to it (Table 6: "Root Cause")
    counterfactual: str  # What should have been done instead (Table 6: "Mitigation")
    prevention_principle: str  # Generalizable rule (→ becomes new skill via evolve)
    task_context: str = ""  # What task/trajectory this failure occurred in (P0-2/P0-3)
    timestamp: str = ""  # ISO-8601 when the lesson was recorded
    evolved: bool = False  # True after evolve_from_failures has processed this lesson (N3)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_point": self.failure_point,
            "flawed_reasoning": self.flawed_reasoning,
            "counterfactual": self.counterfactual,
            "prevention_principle": self.prevention_principle,
            "task_context": self.task_context,
            "timestamp": self.timestamp,
            "evolved": self.evolved,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FailureLesson:
        return cls(
            failure_point=d.get("failure_point", ""),
            flawed_reasoning=d.get("flawed_reasoning", ""),
            counterfactual=d.get("counterfactual", ""),
            prevention_principle=d.get("prevention_principle", ""),
            task_context=d.get("task_context", ""),
            timestamp=d.get("timestamp", ""),
            evolved=bool(d.get("evolved", False)),
        )

    def format_text(self) -> str:
        """Format as human-readable text for playbook display."""
        lines = [
            f"  FAILURE: {self.failure_point}",
            f"  FLAW: {self.flawed_reasoning}",
            f"  INSTEAD: {self.counterfactual}",
            f"  PRINCIPLE: {self.prevention_principle}",
        ]
        if self.task_context:
            lines.append(f"  CONTEXT: {self.task_context}")
        return "\n".join(lines)


# Regex for parsing bullet lines: [id] helpful=X harmful=Y :: content
_BULLET_RE = re.compile(
    r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)"
)


@dataclass
class Bullet:
    """A single playbook bullet point.

    Per SkillRL Table 5: Each skill has {ID, Skill Title, Principle, When to Apply}.
    Per SkillRL §3.2: Skills are organized into general (S_g) and task-specific (S_k).
    """

    id: str
    helpful: int
    harmful: int
    content: str
    section: str = ""
    scope: str = "general"  # "general" or "task_specific" (SkillRL §3.2 hierarchical SkillBank)
    when_to_apply: str = ""  # Applicability condition (SkillRL Table 5: "When to Apply")
    last_updated: str = ""  # ISO-8601 timestamp of last counter update (for temporal decay)
    failure_lessons: list[FailureLesson] = field(default_factory=list)

    @property
    def score(self) -> int:
        """Net score: helpful - harmful."""
        return self.helpful - self.harmful

    def effective_score(self, decay_rate: float = 0.95) -> float:
        """Net score with temporal decay: (helpful - harmful) * decay_rate^days_since_update.

        Inspired by ACT-R memory decay / SYNAPSE spreading activation.
        A bullet unused for 30 days retains ~21%, 90 days ~1%.
        """
        if not self.last_updated:
            return float(self.score)  # No timestamp = no decay
        try:
            updated = datetime.fromisoformat(self.last_updated)
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            days = (now - updated).total_seconds() / 86400
            return self.score * (decay_rate ** days)
        except (ValueError, TypeError):
            return float(self.score)

    @property
    def is_problematic(self) -> bool:
        """Bullet has more harmful than helpful tags."""
        return self.harmful > 0 and self.harmful >= self.helpful

    @property
    def is_unused(self) -> bool:
        """Bullet has never been tagged."""
        return self.helpful == 0 and self.harmful == 0

    @property
    def has_failure_lessons(self) -> bool:
        """Bullet has structured failure data beyond bare counters."""
        return len(self.failure_lessons) > 0

    def format_line(self) -> str:
        """Format as a playbook line (unchanged format for backward compatibility)."""
        return f"[{self.id}] helpful={self.helpful} harmful={self.harmful} :: {self.content}"

    def format_line_with_failures(self) -> str:
        """Format as a playbook line with inline failure lessons for display."""
        line = self.format_line()
        if self.failure_lessons:
            lessons_text = "\n".join(
                f"  [{i+1}/{len(self.failure_lessons)}]\n{fl.format_text()}"
                for i, fl in enumerate(self.failure_lessons)
            )
            line += f"\n{lessons_text}"
        return line


@dataclass
class DeltaOperation:
    """A single delta operation proposed by the Curator or Deduplicator."""

    op_type: str  # "ADD", "UPDATE", "MERGE", "REMOVE"
    section: str
    content: str
    bullet_id: str | None = None  # For UPDATE/REMOVE operations
    merge_target: str | None = None  # For MERGE: second bullet to merge into bullet_id


@dataclass
class PlaybookStats:
    """Statistics about a playbook."""

    total_bullets: int = 0
    high_performing: int = 0  # helpful > 5, harmful < 2
    problematic: int = 0  # harmful >= helpful and harmful > 0
    unused: int = 0  # helpful + harmful == 0
    by_section: dict[str, dict[str, int]] = field(default_factory=dict)
    total_chars: int = 0
    total_sections: int = 0
    # Failure lesson stats
    total_failure_lessons: int = 0
    harmful_with_lessons: int = 0  # bullets with harmful > 0 AND at least one lesson
    harmful_without_lessons: int = 0  # bullets with harmful > 0 but no lessons
    # Evolution trigger stats (SkillRL §3.3)
    evolution_needed: bool = False  # True when accumulated failures exceed threshold
    evolution_candidates: int = 0  # Number of bullets eligible for evolution
    # Temporal decay stats
    decayed_bullets: int = 0  # Bullets whose effective_score < 50% of raw score


class Playbook:
    """Structured playbook: a collection of sections containing bullets.

    The playbook format:
        ## SECTION NAME
        [slug-00001] helpful=X harmful=Y :: Content here
        [slug-00002] helpful=X harmful=Y :: More content

        ## ANOTHER SECTION
        ...
    """

    def __init__(self, text: str | None = None, sections: list[str] | None = None):
        self._sections: list[str] = sections or list(DEFAULT_SECTIONS)
        self._bullets: list[Bullet] = []
        self._next_id: int = 1

        if text:
            self._parse(text)
        else:
            # Initialize empty — _next_id stays at 1
            pass

    def _parse(self, text: str) -> None:
        """Parse playbook text into structured bullets."""
        lines = text.strip().split("\n")
        current_section = ""
        max_id_num = 0
        seen_ids: set[str] = set()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("##"):
                current_section = stripped[2:].strip()
                if current_section not in self._sections:
                    self._sections.append(current_section)
                continue

            match = _BULLET_RE.match(stripped)
            if match:
                bullet_id = match.group(1)
                # Skip duplicate IDs — keep the first occurrence
                if bullet_id in seen_ids:
                    continue
                seen_ids.add(bullet_id)

                bullet = Bullet(
                    id=bullet_id,
                    helpful=int(match.group(2)),
                    harmful=int(match.group(3)),
                    content=match.group(4),
                    section=current_section,
                )
                self._bullets.append(bullet)

                # Track max ID number
                id_num_match = re.search(r"-(\d+)$", bullet.id)
                if id_num_match:
                    max_id_num = max(max_id_num, int(id_num_match.group(1)))

        self._next_id = max_id_num + 1

    def serialize(self) -> str:
        """Serialize the playbook to text format."""
        lines: list[str] = []

        for section in self._sections:
            lines.append(f"## {section}")
            section_bullets = [b for b in self._bullets if b.section == section]
            for bullet in section_bullets:
                lines.append(bullet.format_line())
            lines.append("")  # blank line after section

        # Any bullets without a matching section go to OTHERS
        orphans = [b for b in self._bullets if b.section not in self._sections]
        if orphans:
            if "OTHERS" not in self._sections:
                lines.append("## OTHERS")
            for bullet in orphans:
                lines.append(bullet.format_line())
            lines.append("")

        return "\n".join(lines).rstrip()

    @property
    def bullets(self) -> list[Bullet]:
        """All bullets in the playbook."""
        return list(self._bullets)

    @property
    def sections(self) -> list[str]:
        """All section names."""
        return list(self._sections)

    @property
    def next_id(self) -> int:
        """Next available global ID number."""
        return self._next_id

    def get_bullet(self, bullet_id: str) -> Bullet | None:
        """Find a bullet by ID."""
        for b in self._bullets:
            if b.id == bullet_id:
                return b
        return None

    def get_section_bullets(self, section: str) -> list[Bullet]:
        """Get all bullets in a section."""
        return [b for b in self._bullets if b.section == section]

    def extract_bullets(self, bullet_ids: list[str]) -> str:
        """Extract specific bullets by ID, formatted for Reflector input."""
        found = [b for b in self._bullets if b.id in bullet_ids]
        if not found:
            return "(No matching bullets found)"
        return "\n".join(b.format_line() for b in found)

    def update_bullet_counts(self, bullet_tags: list[dict[str, Any]]) -> int:
        """Update helpful/harmful counts based on Reflector tags.

        When tag is "harmful", an optional "failure_lesson" dict can be included
        with structured failure data (SkillRL-inspired):
            {
                "failure_point": "Where the strategy broke down",
                "flawed_reasoning": "What assumption was wrong",
                "counterfactual": "What should have been done instead",
                "prevention_principle": "General rule to avoid this failure"
            }

        Args:
            bullet_tags: List of dicts with "id", "tag", and optional "failure_lesson".

        Returns:
            Number of bullets updated.
        """
        tag_map: dict[str, dict[str, Any]] = {}
        for tag in bullet_tags:
            bid = tag.get("id") or tag.get("bullet", "")
            tag_val = tag.get("tag", "neutral")
            if bid:
                tag_map[bid] = {"tag": tag_val, "failure_lesson": tag.get("failure_lesson")}

        now_iso = datetime.now(timezone.utc).isoformat()
        updated = 0
        for bullet in self._bullets:
            if bullet.id in tag_map:
                entry = tag_map[bullet.id]
                tag = entry["tag"]
                if tag == "helpful":
                    bullet.helpful += 1
                    bullet.last_updated = now_iso
                    updated += 1
                elif tag == "harmful":
                    bullet.harmful += 1
                    bullet.last_updated = now_iso
                    updated += 1
                    # Attach structured failure lesson if provided
                    lesson_data = entry.get("failure_lesson")
                    if isinstance(lesson_data, dict) and lesson_data:
                        lesson = FailureLesson.from_dict(lesson_data)
                        bullet.failure_lessons.append(lesson)
        return updated

    def apply_delta(self, operations: list[DeltaOperation]) -> int:
        """Apply delta operations to the playbook.

        Per ACE paper (§3.1): Operations are applied in safe order —
        ADDs first (fully independent/parallel), then UPDATEs, MERGEs, REMOVEs.

        Supports: ADD, UPDATE, MERGE, REMOVE.

        Args:
            operations: List of DeltaOperation to apply.

        Returns:
            Number of operations applied.
        """
        applied = 0
        # Sort by operation type for safe parallel-like processing
        adds = [op for op in operations if op.op_type == "ADD"]
        updates = [op for op in operations if op.op_type == "UPDATE"]
        merges = [op for op in operations if op.op_type == "MERGE"]
        removes = [op for op in operations if op.op_type == "REMOVE"]

        for op in adds:
            applied += self._apply_add(op)
        for op in updates:
            applied += self._apply_update(op)
        for op in merges:
            applied += self._apply_merge(op)
        for op in removes:
            applied += self._apply_remove(op)
        return applied

    def _apply_add(self, op: DeltaOperation) -> int:
        """Add a new bullet to a section."""
        # Resolve section
        section = op.section
        found = False
        for s in self._sections:
            if _normalize_section(s) == _normalize_section(section):
                section = s
                found = True
                break

        if not found:
            # Fall back to OTHERS
            section = "OTHERS"
            if section not in self._sections:
                self._sections.append(section)

        slug = _get_slug(section)
        bullet_id = f"{slug}-{self._next_id:05d}"
        self._next_id += 1

        bullet = Bullet(
            id=bullet_id,
            helpful=0,
            harmful=0,
            content=op.content,
            section=section,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )
        self._bullets.append(bullet)
        return 1

    def _apply_update(self, op: DeltaOperation) -> int:
        """Update an existing bullet's content."""
        if not op.bullet_id:
            return 0
        bullet = self.get_bullet(op.bullet_id)
        if bullet is None:
            return 0
        bullet.content = op.content
        if op.section:
            # Optionally move to a new section
            for s in self._sections:
                if _normalize_section(s) == _normalize_section(op.section):
                    bullet.section = s
                    break
        return 1

    def _apply_merge(self, op: DeltaOperation) -> int:
        """Merge two bullets: keep bullet_id with merged content, remove merge_target.

        Per P1-3: Combines failure_lessons from absorbed bullet into keeper,
        preserving all structured failure data.
        """
        if not op.bullet_id or not op.merge_target:
            return 0
        keeper = self.get_bullet(op.bullet_id)
        absorbed = self.get_bullet(op.merge_target)
        if keeper is None or absorbed is None:
            return 0
        # Combine counts
        keeper.helpful += absorbed.helpful
        keeper.harmful += absorbed.harmful
        # Combine failure lessons from both bullets
        keeper.failure_lessons.extend(absorbed.failure_lessons)
        # Use provided merged content
        keeper.content = op.content
        # Remove the absorbed bullet
        self._bullets = [b for b in self._bullets if b.id != op.merge_target]
        return 1

    def _apply_remove(self, op: DeltaOperation) -> int:
        """Remove a bullet by ID."""
        if not op.bullet_id:
            return 0
        before = len(self._bullets)
        self._bullets = [b for b in self._bullets if b.id != op.bullet_id]
        return 1 if len(self._bullets) < before else 0

    def find_similar_pairs(self, threshold: float = 0.6) -> list[tuple[Bullet, Bullet, float]]:
        """Find bullet pairs with high text similarity.

        Uses combined word-overlap Jaccard + character trigram similarity
        for better paraphrase detection than word-only Jaccard.

        Returns:
            List of (bullet_a, bullet_b, similarity_score) tuples.
        """
        pairs = []
        for i, a in enumerate(self._bullets):
            words_a = set(a.content.lower().split())
            trigrams_a = self._char_trigrams(a.content.lower())
            if len(words_a) < 2:
                continue
            for b in self._bullets[i + 1:]:
                words_b = set(b.content.lower().split())
                trigrams_b = self._char_trigrams(b.content.lower())
                if len(words_b) < 2:
                    continue
                # Word Jaccard
                word_inter = words_a & words_b
                word_union = words_a | words_b
                word_jaccard = len(word_inter) / len(word_union) if word_union else 0.0
                # Trigram Jaccard
                tri_inter = trigrams_a & trigrams_b
                tri_union = trigrams_a | trigrams_b
                tri_jaccard = len(tri_inter) / len(tri_union) if tri_union else 0.0
                # Combined score (weighted average)
                combined = 0.4 * word_jaccard + 0.6 * tri_jaccard
                if combined >= threshold:
                    pairs.append((a, b, combined))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    @staticmethod
    def _char_trigrams(text: str) -> set[str]:
        """Extract character trigrams from text."""
        text = text.strip()
        if len(text) < 3:
            return set()
        return {text[i:i+3] for i in range(len(text) - 2)}

    def prune_problematic(self, min_harmful: int = 3) -> list[Bullet]:
        """Remove bullets where harmful >= helpful and harmful >= min_harmful.

        Per ACE paper (§3.2): Grow-and-refine prunes redundancy.
        Per P1-3: Returned bullets retain their failure_lessons so callers
        can feed them into evolve_from_failures() before discarding.

        Returns:
            List of removed bullets (with failure_lessons intact).
        """
        removed = []
        kept = []
        for b in self._bullets:
            if b.harmful >= b.helpful and b.harmful >= min_harmful:
                removed.append(b)
            else:
                kept.append(b)
        self._bullets = kept
        return removed

    def enforce_token_budget(self, max_chars: int) -> list[Bullet]:
        """Remove lowest-scoring bullets until under budget (M7: O(n log n) not O(n^2)).

        Pre-sorts bullets by score once, then removes cheapest until under budget,
        estimating size reduction from format_line() length instead of re-serializing.

        Args:
            max_chars: Maximum character count for the serialized playbook.

        Returns:
            List of removed bullets.
        """
        removed = []
        current_size = len(self.serialize())
        if current_size <= max_chars:
            return removed
        # Sort ascending by score (worst first)
        ranked = sorted(self._bullets, key=lambda b: (b.effective_score(), -b.harmful))
        keep = set(id(b) for b in self._bullets)
        for bullet in ranked:
            if current_size <= max_chars:
                break
            keep.discard(id(bullet))
            removed.append(bullet)
            # Estimate size reduction: format_line + newline + section header overhead
            current_size -= len(bullet.format_line()) + 1
        self._bullets = [b for b in self._bullets if id(b) in keep]
        return removed

    def get_stats(self) -> PlaybookStats:
        """Get playbook statistics."""
        stats = PlaybookStats(
            total_chars=len(self.serialize()),
            total_sections=len(self._sections),
        )

        for bullet in self._bullets:
            stats.total_bullets += 1
            if bullet.helpful > 5 and bullet.harmful < 2:
                stats.high_performing += 1
            elif bullet.is_problematic:
                stats.problematic += 1
            elif bullet.is_unused:
                stats.unused += 1

            section = bullet.section or "general"
            if section not in stats.by_section:
                stats.by_section[section] = {"count": 0, "helpful": 0, "harmful": 0}
            stats.by_section[section]["count"] += 1
            stats.by_section[section]["helpful"] += bullet.helpful
            stats.by_section[section]["harmful"] += bullet.harmful

            # Failure lesson stats
            stats.total_failure_lessons += len(bullet.failure_lessons)
            if bullet.harmful > 0:
                if bullet.has_failure_lessons:
                    stats.harmful_with_lessons += 1
                else:
                    stats.harmful_without_lessons += 1

        # Temporal decay stats
        for bullet in self._bullets:
            raw = float(bullet.score)
            if raw != 0 and bullet.last_updated:
                eff = bullet.effective_score()
                if abs(eff) < abs(raw) * 0.5:
                    stats.decayed_bullets += 1

        # Evolution trigger check (SkillRL §3.3)
        evo = self.check_evolution_needed()
        stats.evolution_needed = evo["needed"]
        stats.evolution_candidates = evo["candidate_count"]

        return stats


    def check_evolution_needed(self, threshold: int = 3) -> dict[str, Any]:
        """Check if skill evolution should be triggered (SkillRL §3.3).

        Per SkillRL §3.3: Evolution is triggered when accuracy for a task category
        falls below threshold δ. In our adaptation, we trigger when the number of
        harmful bullets with failure lessons meets or exceeds the threshold.

        Args:
            threshold: Minimum number of harmful-with-lessons bullets to trigger evolution.

        Returns:
            Dict with 'needed' (bool), 'candidate_count' (int), 'candidate_ids' (list[str]).
        """
        candidates = [
            b for b in self._bullets
            if b.harmful > 0 and b.has_failure_lessons
        ]
        return {
            "needed": len(candidates) >= threshold,
            "candidate_count": len(candidates),
            "candidate_ids": [b.id for b in candidates],
        }

    def evolve_from_failures(self, threshold: int = 3) -> list[Bullet]:
        """Generate NEW standalone skill bullets from accumulated failure lessons.

        Per SkillRL Prompt B.1: "Generate 1-3 NEW actionable skills... each must have:
        skill_id, title, principle, when_to_apply." Failed strategies produce NEW skills
        via their prevention_principles, not mere annotations.

        Per SkillRL §3.3: Evolution is triggered when Acc(C) < δ. The teacher model
        analyzes failed validation trajectories and distills them into new skills.

        In our adaptation (no sub-model), each failure lesson's prevention_principle
        becomes a new bullet in PROBLEM-SOLVING HEURISTICS with scope="general" and
        when_to_apply derived from the task_context.

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


def create_empty_playbook(sections: list[str] | None = None) -> Playbook:
    """Create a new empty playbook with standard sections."""
    return Playbook(sections=sections or list(DEFAULT_SECTIONS))


def parse_delta_operations(curator_output: dict[str, Any]) -> list[DeltaOperation]:
    """Parse Curator/Deduplicator JSON output into DeltaOperation list.

    Expected format:
        {"reasoning": "...", "operations": [
            {"type": "ADD", "section": "...", "content": "..."},
            {"type": "UPDATE", "bullet_id": "str-00001", "content": "..."},
            {"type": "MERGE", "bullet_id": "str-00001", "merge_target": "str-00002", "content": "..."},
            {"type": "REMOVE", "bullet_id": "str-00001"},
        ]}
    """
    ops = []
    raw_ops = curator_output.get("operations", [])
    for raw in raw_ops:
        if not isinstance(raw, dict):
            continue
        op_type = raw.get("type", "")
        if op_type == "ADD":
            ops.append(DeltaOperation(
                op_type="ADD",
                section=raw.get("section", "OTHERS"),
                content=raw.get("content", ""),
            ))
        elif op_type == "UPDATE" and raw.get("bullet_id"):
            ops.append(DeltaOperation(
                op_type="UPDATE",
                section=raw.get("section", ""),
                content=raw.get("content", ""),
                bullet_id=raw.get("bullet_id"),
            ))
        elif op_type == "MERGE" and raw.get("bullet_id") and raw.get("merge_target"):
            ops.append(DeltaOperation(
                op_type="MERGE",
                section=raw.get("section", ""),
                content=raw.get("content", ""),
                bullet_id=raw.get("bullet_id"),
                merge_target=raw.get("merge_target"),
            ))
        elif op_type == "REMOVE" and raw.get("bullet_id"):
            ops.append(DeltaOperation(
                op_type="REMOVE",
                section="",
                content="",
                bullet_id=raw.get("bullet_id"),
            ))
    return ops
