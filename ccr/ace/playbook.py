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
import math
import os
import re
from collections import defaultdict
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

    Data structure inspired by SkillRL §3.1. The paper's distillation uses a
    teacher model M_T on full trajectories: s⁻ = M_T(τ⁻, d). CCR relies on
    manual annotation — Claude Code provides the failure_lesson dict when
    tagging a bullet as harmful. No teacher model or trajectory replay.

    Fields capture: (1) the point of failure, (2) the flawed reasoning or action,
    (3) what should have been done, and (4) general principles to prevent similar failures.

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
    grpo_advantage: float = 0.0  # group-relative advantage (SkillRL GRPO Eq.3)
    trigger: str = ""  # ERL: When condition (e.g., "when adding API endpoints")
    action: str = ""  # ERL: Then action (e.g., "add input validation first")
    weighted_helpful: float = 0.0  # contribution-weighted helpful (AgentEvolver-inspired)
    weighted_harmful: float = 0.0  # contribution-weighted harmful (AgentEvolver-inspired)
    personal_decay_rate: float = 0.0  # 0.0 = use schema default; set by update_bullet_counts (SM-2 inspired)
    failure_lessons: list[FailureLesson] = field(default_factory=list)

    @property
    def score(self) -> int:
        """Net score: helpful - harmful."""
        return self.helpful - self.harmful

    def effective_score(self, decay_rate: float = 0.95) -> float:
        """Net score with temporal decay: raw * rate^days_since_update.

        When contribution-weighted counters are present (weighted_helpful or
        weighted_harmful > 0), uses weighted values as raw score. Otherwise
        falls back to integer helpful - harmful (backward compat).

        Uses personal_decay_rate when set (> 0.0) — SM-2 inspired per-bullet
        adaptive decay based on helpful/harmful history. Falls back to the
        schema-level decay_rate parameter when personal_decay_rate is 0.0.

        Inspired by ACT-R memory decay / SYNAPSE spreading activation.
        AgentEvolver-inspired contribution weighting for proportional credit.
        A bullet unused for 30 days retains ~21%, 90 days ~1%.
        """
        # Use weighted counters when available, fall back to integer counts
        if self.weighted_helpful > 0 or self.weighted_harmful > 0:
            raw = self.weighted_helpful - self.weighted_harmful
        else:
            raw = float(self.score)

        if not self.last_updated:
            return raw  # No timestamp = no decay
        try:
            updated = datetime.fromisoformat(self.last_updated)
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            days = (now - updated).total_seconds() / 86400
            rate = self.personal_decay_rate if self.personal_decay_rate > 0.0 else decay_rate
            return raw * (rate ** days)
        except (ValueError, TypeError):
            return raw

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
    trigger: str = ""  # ERL: When condition for ADD operations
    action: str = ""  # ERL: Then action for ADD operations


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
        self._id_index: dict[str, Bullet] = {}
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
        self._id_index = {b.id: b for b in self._bullets}

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
        """Find a bullet by ID (O(1) via in-memory index, linear fallback)."""
        result = self._id_index.get(bullet_id)
        if result is not None:
            return result
        # Fallback: linear scan if index is stale or bypassed (e.g., tests)
        for b in self._bullets:
            if b.id == bullet_id:
                self._id_index[b.id] = b  # repair index
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

        An optional "weight" key (0.0-1.0) enables contribution-weighted counters
        (AgentEvolver-inspired). When multiple strategies were active, each receives
        proportional credit/blame. The integer counters always increment by 1
        regardless of weight; the weight accumulates in weighted_helpful/weighted_harmful.

        Args:
            bullet_tags: List of dicts with "id", "tag", optional "failure_lesson",
                and optional "weight" (float 0.0-1.0, default 1.0).

        Returns:
            Number of bullets updated.
        """
        tag_map: dict[str, dict[str, Any]] = {}
        for tag in bullet_tags:
            bid = tag.get("id") or tag.get("bullet", "")
            tag_val = tag.get("tag", "neutral")
            if bid:
                # Clamp weight to [0.0, 1.0], default 1.0
                raw_weight = tag.get("weight", 1.0)
                try:
                    weight = max(0.0, min(1.0, float(raw_weight)))
                except (TypeError, ValueError):
                    weight = 1.0
                tag_map[bid] = {
                    "tag": tag_val,
                    "failure_lesson": tag.get("failure_lesson"),
                    "weight": weight,
                }

        now_iso = datetime.now(timezone.utc).isoformat()
        updated = 0
        for bullet in self._bullets:
            if bullet.id in tag_map:
                entry = tag_map[bullet.id]
                tag = entry["tag"]
                weight = entry["weight"]
                if tag == "helpful":
                    bullet.helpful += 1
                    bullet.weighted_helpful += weight
                    bullet.last_updated = now_iso
                    updated += 1
                elif tag == "harmful":
                    bullet.harmful += 1
                    bullet.weighted_harmful += weight
                    bullet.last_updated = now_iso
                    updated += 1
                    # Attach structured failure lesson if provided
                    lesson_data = entry.get("failure_lesson")
                    if isinstance(lesson_data, dict) and lesson_data:
                        lesson = FailureLesson.from_dict(lesson_data)
                        bullet.failure_lessons.append(lesson)
                if tag in ("helpful", "harmful"):
                    # SM-2 inspired: per-bullet adaptive decay rate
                    # High helpful → slower decay (knowledge stays relevant longer)
                    # High harmful → faster decay (discourage stale bad advice)
                    bullet.personal_decay_rate = max(
                        0.90, min(0.99, 0.95 + (bullet.helpful - bullet.harmful) * 0.002)
                    )
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
            trigger=op.trigger,
            action=op.action,
        )
        self._bullets.append(bullet)
        self._id_index[bullet.id] = bullet
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
        # Combine trigger/action fields (ERL-inspired)
        if absorbed.trigger:
            if keeper.trigger and keeper.trigger != absorbed.trigger:
                keeper.trigger = f"{keeper.trigger}; {absorbed.trigger}"
            elif not keeper.trigger:
                keeper.trigger = absorbed.trigger
        if absorbed.action:
            if keeper.action and keeper.action != absorbed.action:
                keeper.action = f"{keeper.action}; {absorbed.action}"
            elif not keeper.action:
                keeper.action = absorbed.action
        # Use provided merged content
        keeper.content = op.content
        # Remove the absorbed bullet
        self._bullets = [b for b in self._bullets if b.id != op.merge_target]
        self._id_index.pop(op.merge_target, None)
        return 1

    def _apply_remove(self, op: DeltaOperation) -> int:
        """Remove a bullet by ID."""
        if not op.bullet_id:
            return 0
        before = len(self._bullets)
        self._bullets = [b for b in self._bullets if b.id != op.bullet_id]
        if len(self._bullets) < before:
            self._id_index.pop(op.bullet_id, None)
            return 1
        return 0

    def find_similar_pairs(self, threshold: float = 0.6) -> list[tuple[Bullet, Bullet, float]]:
        """Find bullet pairs with high text similarity.

        Primary: ONNX cosine similarity (all-MiniLM-L6-v2).
        Fallback: combined word-overlap Jaccard + character trigram similarity.

        Returns:
            List of (bullet_a, bullet_b, similarity_score) tuples.
        """
        from ccr.context.embeddings import get_embedding_model

        texts = [(a.content + " " + a.trigger).strip() for a in self._bullets]
        model = get_embedding_model()
        embeddings = None

        # Try ONNX batch embedding for O(n) embed + O(n^2) dot product
        if model is not None and len(texts) >= 2:
            try:
                non_empty = [t if len(t.split()) >= 2 else "empty" for t in texts]
                embeddings = model.embed_batch(non_empty)
            except Exception:
                embeddings = None

        pairs = []
        for i, a in enumerate(self._bullets):
            text_a = texts[i]
            if len(text_a.split()) < 2:
                continue
            for j in range(i + 1, len(self._bullets)):
                b = self._bullets[j]
                text_b = texts[j]
                if len(text_b.split()) < 2:
                    continue

                if embeddings is not None:
                    # ONNX cosine (primary)
                    combined = float(embeddings[i] @ embeddings[j])
                else:
                    # Jaccard + trigram fallback
                    words_a = set(text_a.lower().split())
                    words_b = set(text_b.lower().split())
                    trigrams_a = self._char_trigrams(text_a.lower())
                    trigrams_b = self._char_trigrams(text_b.lower())
                    word_inter = words_a & words_b
                    word_union = words_a | words_b
                    word_jaccard = len(word_inter) / len(word_union) if word_union else 0.0
                    tri_inter = trigrams_a & trigrams_b
                    tri_union = trigrams_a | trigrams_b
                    tri_jaccard = len(tri_inter) / len(tri_union) if tri_union else 0.0
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
        for b in removed:
            self._id_index.pop(b.id, None)
        return removed

    def enforce_token_budget(self, max_chars: int, decay_rate: float = 0.95) -> list[Bullet]:
        """Remove lowest-scoring bullets until under budget (M7: O(n log n) not O(n^2)).

        Pre-sorts bullets by score once, then removes cheapest until under budget,
        estimating size reduction from format_line() length instead of re-serializing.

        Args:
            max_chars: Maximum character count for the serialized playbook.
            decay_rate: Temporal decay rate for effective_score (MCE schema param).

        Returns:
            List of removed bullets.
        """
        removed = []
        current_size = len(self.serialize())
        if current_size <= max_chars:
            return removed
        # Sort ascending: (worst_score, highest_harmful, oldest_timestamp) pruned first.
        # ISO string comparison: "" < "2026-01-..." < "2026-04-..." so newly-evolved
        # bullets (recent last_updated) sort LATER and are spared when over budget.
        ranked = sorted(
            self._bullets,
            key=lambda b: (b.effective_score(decay_rate), -b.harmful, b.last_updated or ""),
        )
        keep = set(id(b) for b in self._bullets)
        for bullet in ranked:
            if current_size <= max_chars:
                break
            keep.discard(id(bullet))
            removed.append(bullet)
            # Estimate size reduction: format_line + newline + section header overhead
            current_size -= len(bullet.format_line()) + 1
        self._bullets = [b for b in self._bullets if id(b) in keep]
        for b in removed:
            self._id_index.pop(b.id, None)
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

    # ------------------------------------------------------------------
    # GRPO-inspired group-relative advantage scoring (SkillRL GRPO Eq.3)
    # ------------------------------------------------------------------

    @staticmethod
    def _word_jaccard_sim(a: str, b: str) -> float:
        """Word-level Jaccard similarity between two strings."""
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def recompute_grpo_advantages(self) -> int:
        """Recompute GRPO group-relative advantages for all bullets (SkillRL Eq.3).

        Groups bullets within each section by word-Jaccard similarity (>= 0.4).
        Reward r_i = effective_score (helpful * decay). Advantage A_i = (r_i - mean) / (std + epsilon).
        Single-bullet groups get advantage 0.0.
        Returns number of bullets updated.
        """
        updated = 0
        for section in self._sections:
            section_bullets = [b for b in self._bullets if b.section == section]
            if not section_bullets:
                continue

            # Greedy word-Jaccard clustering with threshold 0.4
            groups: list[list[Bullet]] = []
            for bullet in section_bullets:
                placed = False
                for group in groups:
                    anchor = group[0]
                    if self._word_jaccard_sim(bullet.content, anchor.content) >= 0.4:
                        group.append(bullet)
                        placed = True
                        break
                if not placed:
                    groups.append([bullet])

            for group in groups:
                rewards = [b.effective_score() for b in group]
                n = len(rewards)
                if n == 1:
                    group[0].grpo_advantage = 0.0
                    updated += 1
                    continue
                mean_r = sum(rewards) / n
                variance = sum((r - mean_r) ** 2 for r in rewards) / n
                std_r = math.sqrt(variance)
                for bullet, r_i in zip(group, rewards):
                    bullet.grpo_advantage = (r_i - mean_r) / (std_r + 1e-8)
                    updated += 1

        # Also handle bullets with no section or section not in self._sections
        orphans = [b for b in self._bullets if b.section not in self._sections]
        for bullet in orphans:
            bullet.grpo_advantage = 0.0
            updated += 1

        return updated

    def get_policy_ranked(self, task_context: str = "", top_k: int = 10) -> list[Bullet]:
        """Return bullets ranked by effective_score * (1 + grpo_advantage) — GRPO policy-weighted.

        When task_context is provided, filters to bullets with similarity >= 0.1
        against bullet content, when_to_apply, or trigger. Primary: ONNX cosine.
        Fallback: word-Jaccard. Trigger matches get a 1.5x weight boost
        (ERL-inspired: structured trigger/action enables context-aware retrieval).
        Returns top_k results.
        """
        candidates = list(self._bullets)
        retrieval_scores: dict[str, float] = {}

        if task_context.strip():
            from ccr.context.embeddings import quick_cosine

            filtered = []
            for b in candidates:
                # Try ONNX cosine first, fall back to word Jaccard
                trigger_text = b.trigger if b.trigger else ""
                content_text = b.content + " " + b.when_to_apply

                trigger_sim_onnx = quick_cosine(task_context, trigger_text) if trigger_text else None
                content_sim_onnx = quick_cosine(task_context, content_text)

                if trigger_sim_onnx is not None:
                    trigger_sim = trigger_sim_onnx
                else:
                    trigger_sim = self._word_jaccard_sim(task_context, trigger_text) if trigger_text else 0.0

                if content_sim_onnx is not None:
                    content_sim = content_sim_onnx
                else:
                    content_sim = self._word_jaccard_sim(task_context, content_text)

                combined = max(trigger_sim * 1.5, content_sim)
                if combined >= 0.1:
                    retrieval_scores[b.id] = combined
                    filtered.append(b)
            candidates = filtered if filtered else candidates

        def policy_score(b: Bullet) -> float:
            base = b.effective_score() * (1.0 + b.grpo_advantage)
            return base * (1.0 + retrieval_scores.get(b.id, 0.0))

        candidates.sort(key=policy_score, reverse=True)
        return candidates[:top_k]

    # ------------------------------------------------------------------
    # MCE-inspired schema evolution (arXiv:2601.21557)
    # ------------------------------------------------------------------

    # Stop words for clustering (minimal set for bullet content analysis)
    _META_STOP_WORDS: frozenset[str] = frozenset({
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "has", "have", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "that", "this", "it", "not", "no",
        "if", "when", "then", "than", "so", "as", "up", "out", "about",
    })

    def compute_metrics(self, schema: "PlaybookSchema | None" = None) -> "SchemaMetrics":
        """Compute mechanical health metrics for the playbook.

        Inspired by MCE §3.2 evaluation, but these are structural metrics
        (section balance, utilization, harmful ratio, decay impact) — not
        task-performance metrics as in the paper. MCE uses LLM-evaluated
        context quality; CCR uses mechanical heuristics only.

        Args:
            schema: Optional schema for decay_rate. Uses 0.95 default if None.

        Returns:
            SchemaMetrics with all fields populated.
        """
        from ccr.core.types import SchemaMetrics

        decay_rate = schema.decay_rate if schema else 0.95
        total = len(self._bullets)
        now_iso = datetime.now(timezone.utc).isoformat()

        if total == 0:
            return SchemaMetrics(
                empty_sections=list(self._sections),
                total_sections=len(self._sections),
                timestamp=now_iso,
            )

        # Section bullet counts
        section_counts: dict[str, int] = defaultdict(int)
        for b in self._bullets:
            section_counts[b.section] += 1

        # Section balance: normalized Shannon entropy
        non_empty = [c for c in section_counts.values() if c > 0]
        n_non_empty = len(non_empty)
        if n_non_empty <= 1:
            section_balance = 0.0
        else:
            entropy = 0.0
            for count in non_empty:
                p = count / total
                if p > 0:
                    entropy -= p * math.log2(p)
            max_entropy = math.log2(n_non_empty)
            section_balance = entropy / max_entropy if max_entropy > 0 else 0.0

        # Utilization, harmful, unused
        utilized = sum(1 for b in self._bullets if b.helpful + b.harmful > 0)
        utilization_rate = utilized / total

        harmful_count = sum(
            1 for b in self._bullets
            if b.helpful + b.harmful > 0 and b.harmful >= b.helpful and b.harmful > 0
        )
        harmful_ratio = harmful_count / utilized if utilized > 0 else 0.0

        unused_ratio = (total - utilized) / total

        # Decay impact: fraction where effective_score < 50% of raw score
        decay_affected = 0
        for b in self._bullets:
            raw = abs(b.score)
            if raw == 0:
                continue
            eff = abs(b.effective_score(decay_rate))
            if eff < 0.5 * raw:
                decay_affected += 1
        scored_bullets = sum(1 for b in self._bullets if b.score != 0)
        decay_impact = decay_affected / scored_bullets if scored_bullets > 0 else 0.0

        # Empty and overflow sections
        overflow_threshold = 0.5
        empty_sections = [s for s in self._sections if section_counts.get(s, 0) == 0]
        overflow_sections = [
            s for s in self._sections
            if section_counts.get(s, 0) > overflow_threshold * total
        ]

        # Overall health: weighted composite
        overall_health = (
            0.25 * section_balance
            + 0.25 * utilization_rate
            + 0.25 * (1.0 - harmful_ratio)
            + 0.15 * (1.0 - unused_ratio)
            + 0.10 * (1.0 - decay_impact)
        )

        return SchemaMetrics(
            section_balance=round(section_balance, 4),
            utilization_rate=round(utilization_rate, 4),
            harmful_ratio=round(harmful_ratio, 4),
            unused_ratio=round(unused_ratio, 4),
            decay_impact=round(decay_impact, 4),
            empty_sections=empty_sections,
            overflow_sections=overflow_sections,
            overall_health=round(overall_health, 4),
            total_bullets=total,
            total_sections=len(self._sections),
            timestamp=now_iso,
        )

    def propose_schema_changes(
        self,
        schema: "PlaybookSchema",
        metrics: "SchemaMetrics | None" = None,
        *,
        overflow_threshold: float = 0.5,
        min_cluster_size: int = 3,
        stop_health_threshold: float = 0.8,
        rollback_health_delta: float = -0.05,
    ) -> list["SchemaProposal"]:
        """Propose at most one schema change via rule-based heuristic proposals.

        Inspired by MCE's (1+1)-ES (single offspring per call), but MCE uses
        LLM-generated offspring via agentic crossover (§3.1). CCR uses
        deterministic condition checks — no LLM calls.

        Checks conditions in priority order, returns the first applicable
        proposal. Returns empty list if playbook is healthy enough.

        Args:
            schema: Current active schema.
            metrics: Pre-computed metrics (computed if None).
            overflow_threshold: Section fraction triggering overflow.
            min_cluster_size: Min bullets to propose new section from OTHERS.
            stop_health_threshold: Health above this → no proposals.
            rollback_health_delta: Health drop threshold for rollback.

        Returns:
            List with 0 or 1 SchemaProposal.
        """
        from ccr.core.types import SchemaProposal

        if metrics is None:
            metrics = self.compute_metrics(schema)

        # 1. ROLLBACK check: if health degraded from baseline
        if schema.baseline_metrics is not None:
            delta = metrics.overall_health - schema.baseline_metrics.overall_health
            if delta < rollback_health_delta:
                return [SchemaProposal(
                    change_type="ROLLBACK",
                    description=(
                        f"Playbook health degraded by {delta:+.3f} since schema v{schema.version} "
                        f"was adopted (from {schema.baseline_metrics.overall_health:.3f} to "
                        f"{metrics.overall_health:.3f}). Consider reverting to v{schema.parent_version}."
                    ),
                    details={
                        "current_health": metrics.overall_health,
                        "baseline_health": schema.baseline_metrics.overall_health,
                        "delta": round(delta, 4),
                        "parent_version": schema.parent_version,
                    },
                    confidence=min(1.0, abs(delta) / 0.1),
                )]

        # 2. Stop criteria (MCE Appendix D.3.2): healthy enough
        if metrics.overall_health >= stop_health_threshold:
            return []

        # 3. ADD_SECTION: clustered bullets in OTHERS
        clusters = self._cluster_others_bullets(min_cluster_size)
        if clusters:
            name, bullet_ids = clusters[0]  # Best cluster
            slug_prefix = _normalize_section(name)[:3]
            return [SchemaProposal(
                change_type="ADD_SECTION",
                description=(
                    f"OTHERS has {len(self.get_section_bullets('OTHERS'))} bullets. "
                    f"Found cluster of {len(bullet_ids)} related bullets. "
                    f"Propose new section '{name}'."
                ),
                details={
                    "name": name,
                    "slug_prefix": slug_prefix,
                    "from_section": "OTHERS",
                    "bullet_ids": bullet_ids,
                },
                confidence=min(1.0, len(bullet_ids) / (2 * min_cluster_size)),
            )]

        # 4. REMOVE_SECTION: persistently empty section
        if metrics.empty_sections:
            baseline_empty = (
                set(schema.baseline_metrics.empty_sections)
                if schema.baseline_metrics else set()
            )
            for sec in metrics.empty_sections:
                if sec == "OTHERS":
                    continue  # Never remove OTHERS
                if sec in baseline_empty or schema.baseline_metrics is None:
                    return [SchemaProposal(
                        change_type="REMOVE_SECTION",
                        description=f"Section '{sec}' has 0 bullets (persistently empty). Propose removal.",
                        details={"name": sec},
                        confidence=0.7,
                    )]

        # 5. ADJUST_DECAY
        if metrics.decay_impact > 0.5:
            new_rate = max(0.90, schema.decay_rate - 0.02)
            if new_rate != schema.decay_rate:
                return [SchemaProposal(
                    change_type="ADJUST_DECAY",
                    description=(
                        f"{metrics.decay_impact:.0%} of bullets heavily decayed. "
                        f"Propose lowering decay rate from {schema.decay_rate} to {new_rate} "
                        f"(slower decay, longer retention)."
                    ),
                    details={"old_rate": schema.decay_rate, "new_rate": new_rate},
                    confidence=min(1.0, (metrics.decay_impact - 0.5) / 0.3),
                )]
        elif metrics.decay_impact < 0.1 and metrics.unused_ratio > 0.5:
            new_rate = min(0.99, schema.decay_rate + 0.02)
            if new_rate != schema.decay_rate:
                return [SchemaProposal(
                    change_type="ADJUST_DECAY",
                    description=(
                        f"Only {metrics.decay_impact:.0%} decayed but {metrics.unused_ratio:.0%} unused. "
                        f"Propose raising decay rate from {schema.decay_rate} to {new_rate} "
                        f"(faster decay to clear stale bullets)."
                    ),
                    details={"old_rate": schema.decay_rate, "new_rate": new_rate},
                    confidence=0.5,
                )]

        # 6. ADJUST_PRUNING
        if metrics.harmful_ratio > 0.3:
            new_min = max(1, schema.prune_min_harmful - 1)
            if new_min != schema.prune_min_harmful:
                return [SchemaProposal(
                    change_type="ADJUST_PRUNING",
                    description=(
                        f"{metrics.harmful_ratio:.0%} of utilized bullets are net-harmful. "
                        f"Propose lowering prune threshold from {schema.prune_min_harmful} to {new_min}."
                    ),
                    details={
                        "old_min_harmful": schema.prune_min_harmful,
                        "new_min_harmful": new_min,
                    },
                    confidence=min(1.0, (metrics.harmful_ratio - 0.3) / 0.2),
                )]
        elif metrics.harmful_ratio < 0.05 and schema.prune_min_harmful < 5:
            new_min = schema.prune_min_harmful + 1
            return [SchemaProposal(
                change_type="ADJUST_PRUNING",
                description=(
                    f"Only {metrics.harmful_ratio:.0%} harmful. "
                    f"Propose raising prune threshold from {schema.prune_min_harmful} to {new_min} "
                    f"(less aggressive pruning)."
                ),
                details={
                    "old_min_harmful": schema.prune_min_harmful,
                    "new_min_harmful": new_min,
                },
                confidence=0.4,
            )]

        # 7. REBALANCE: overflow sections
        if metrics.overflow_sections:
            overflow_sec = metrics.overflow_sections[0]
            overflow_bullets = self.get_section_bullets(overflow_sec)
            if len(overflow_bullets) > 1:
                # Find most dissimilar bullet in overflow section
                avg_words: set[str] = set()
                for b in overflow_bullets:
                    avg_words |= set(b.content.lower().split())
                # Find bullet with lowest overlap to section average
                best_move = None
                lowest_sim = 1.0
                for b in overflow_bullets:
                    b_words = set(b.content.lower().split())
                    union = avg_words | b_words
                    inter = avg_words & b_words
                    sim = len(inter) / len(union) if union else 0.0
                    if sim < lowest_sim:
                        lowest_sim = sim
                        best_move = b
                if best_move:
                    # Find best target section
                    target = "OTHERS" if overflow_sec != "OTHERS" else (
                        self._sections[0] if self._sections else "OTHERS"
                    )
                    return [SchemaProposal(
                        change_type="REBALANCE",
                        description=(
                            f"Section '{overflow_sec}' has {len(overflow_bullets)} bullets "
                            f"(>{overflow_threshold:.0%} of total). "
                            f"Propose moving [{best_move.id}] to '{target}'."
                        ),
                        details={
                            "moves": [{
                                "bullet_id": best_move.id,
                                "from": overflow_sec,
                                "to": target,
                            }],
                        },
                        confidence=0.5,
                    )]

        # 8. ADJUST_BUDGET
        total_chars = len(self.serialize())
        if total_chars > 0.9 * schema.token_budget:
            new_budget = schema.token_budget + 20000
            return [SchemaProposal(
                change_type="ADJUST_BUDGET",
                description=(
                    f"Playbook using {total_chars}/{schema.token_budget} chars "
                    f"({total_chars/schema.token_budget:.0%}). "
                    f"Propose increasing budget to {new_budget}."
                ),
                details={"old_budget": schema.token_budget, "new_budget": new_budget},
                confidence=min(1.0, (total_chars / schema.token_budget - 0.9) / 0.1),
            )]
        elif total_chars < 0.3 * schema.token_budget and schema.token_budget > 40000:
            new_budget = max(40000, schema.token_budget - 20000)
            if new_budget != schema.token_budget:
                return [SchemaProposal(
                    change_type="ADJUST_BUDGET",
                    description=(
                        f"Playbook using only {total_chars}/{schema.token_budget} chars "
                        f"({total_chars/schema.token_budget:.0%}). "
                        f"Propose decreasing budget to {new_budget}."
                    ),
                    details={"old_budget": schema.token_budget, "new_budget": new_budget},
                    confidence=0.3,
                )]

        return []

    def _cluster_others_bullets(self, min_cluster: int = 3) -> list[tuple[str, list[str]]]:
        """Cluster OTHERS bullets by content similarity for section proposals.

        Uses single-linkage clustering via BFS on a word Jaccard similarity
        graph (threshold >= 0.3).

        Returns:
            List of (proposed_section_name, [bullet_ids]) sorted by cluster size desc.
        """
        others = self.get_section_bullets("OTHERS")
        if len(others) < min_cluster:
            return []

        # Build word sets (excluding stop words)
        bullet_words: dict[str, set[str]] = {}
        for b in others:
            words = {
                w for w in b.content.lower().split()
                if w not in self._META_STOP_WORDS and len(w) > 2
            }
            if words:
                bullet_words[b.id] = words

        if len(bullet_words) < min_cluster:
            return []

        # Build adjacency graph with word Jaccard >= 0.3
        adjacency: dict[str, set[str]] = defaultdict(set)
        ids = list(bullet_words.keys())
        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1:]:
                inter = bullet_words[id_a] & bullet_words[id_b]
                union = bullet_words[id_a] | bullet_words[id_b]
                jaccard = len(inter) / len(union) if union else 0.0
                if jaccard >= 0.3:
                    adjacency[id_a].add(id_b)
                    adjacency[id_b].add(id_a)

        # BFS connected components
        visited: set[str] = set()
        clusters: list[list[str]] = []
        for start in ids:
            if start in visited:
                continue
            component: list[str] = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                queue.extend(adjacency.get(node, set()) - visited)
            if len(component) >= min_cluster:
                clusters.append(component)

        # Generate proposed names from top-3 frequent words
        results: list[tuple[str, list[str]]] = []
        for cluster in sorted(clusters, key=len, reverse=True):
            word_freq: dict[str, int] = defaultdict(int)
            for bid in cluster:
                for w in bullet_words.get(bid, set()):
                    word_freq[w] += 1
            top_words = sorted(word_freq, key=word_freq.get, reverse=True)[:3]  # type: ignore[arg-type]
            name = " ".join(w.upper() for w in top_words) if top_words else "UNNAMED"
            results.append((name, cluster))

        return results

    def apply_schema(self, schema: "PlaybookSchema") -> int:
        """Apply a schema to the playbook, updating sections.

        Bullets in sections removed by the schema get moved to OTHERS.
        New sections are added. Existing bullets stay in place.

        Args:
            schema: The schema to apply.

        Returns:
            Number of bullets moved.
        """
        moved = 0
        schema_sections = set(schema.sections)

        # Move bullets from removed sections to OTHERS
        for bullet in self._bullets:
            if bullet.section and bullet.section not in schema_sections:
                bullet.section = "OTHERS"
                moved += 1

        # Update sections list
        self._sections = list(schema.sections)

        # Ensure OTHERS exists
        if "OTHERS" not in self._sections:
            self._sections.append("OTHERS")

        # Register new slug prefixes at runtime
        for section_name in schema.sections:
            normalized = _normalize_section(section_name)
            if normalized not in _SLUG_MAP and normalized in schema.slug_map:
                _SLUG_MAP[normalized] = schema.slug_map[normalized]

        return moved


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
                trigger=raw.get("trigger", ""),
                action=raw.get("action", ""),
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
