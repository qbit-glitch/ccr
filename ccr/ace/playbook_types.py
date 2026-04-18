"""Playbook types — constants, dataclasses, and helper functions.

Shared types used by the Playbook class and its mixins. Extracted from
playbook.py to keep each file within 200-400 lines.
"""

from __future__ import annotations

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
    "strategies_and_insights": "str",
    "code_snippets_and_templates": "code",
    "common_mistakes_to_avoid": "mis",
    "problem-solving_heuristics": "heu",
    "problem_solving_heuristics": "heu",
    "context_clues_and_indicators": "ctx",
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
