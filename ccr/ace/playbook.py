"""Playbook data structure — structured, itemized bullets with helpful/harmful counters.

Per ACE paper (§3.1): Context is a collection of structured, itemized bullets rather than
a monolithic prompt. Each bullet has metadata (unique ID, helpful/harmful counters) and
content (a reusable strategy, domain concept, or common failure mode).

Extended with Structured Failure Lessons (SkillRL-inspired): When a strategy is tagged
harmful, an optional failure lesson captures *why* it failed, *what* should have been done,
and a generalizable prevention principle. Stored in .ccr/failure_lessons.json.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

# Re-export all public types for backward compatibility
from ccr.ace.playbook_types import (  # noqa: F401
    Bullet,
    DeltaOperation,
    FailureLesson,
    PlaybookStats,
    DEFAULT_SECTIONS,
    _SLUG_MAP,
    _BULLET_RE,
    _normalize_section,
    _get_slug,
)
from ccr.ace.playbook_analytics import AnalyticsMixin
from ccr.ace.playbook_mutations import MutationsMixin
from ccr.ace.playbook_schema import SchemaMixin


class Playbook(AnalyticsMixin, SchemaMixin, MutationsMixin):
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

    @classmethod
    def from_backend(cls, backend: Any, scope: str = "project") -> Playbook:
        """Load a Playbook from a StorageBackend (for complex analytics)."""
        sections = backend.playbook_sections_get(scope)
        bullets_data = backend.bullet_list(scope=scope)

        pb = cls.__new__(cls)
        pb._sections = sections or list(DEFAULT_SECTIONS)
        pb._bullets = []
        pb._id_index = {}
        pb._next_id = backend.bullet_get_next_id(scope)

        for bd in bullets_data:
            bullet = Bullet(
                id=bd["id"],
                helpful=bd.get("helpful", 0),
                harmful=bd.get("harmful", 0),
                content=bd.get("content", ""),
                section=bd.get("section", ""),
                scope=bd.get("scope", "general"),
                when_to_apply=bd.get("when_to_apply", ""),
                last_updated=bd.get("last_updated", ""),
                grpo_advantage=bd.get("grpo_advantage", 0.0),
                trigger=bd.get("trigger_text", ""),
                action=bd.get("action", ""),
                weighted_helpful=bd.get("weighted_helpful", 0.0),
                weighted_harmful=bd.get("weighted_harmful", 0.0),
                personal_decay_rate=bd.get("personal_decay_rate", 0.0),
            )
            pb._bullets.append(bullet)
            pb._id_index[bullet.id] = bullet

        all_lessons = backend.failure_lessons_all(scope)
        for bid, lessons in all_lessons.items():
            if bid in pb._id_index:
                pb._id_index[bid].failure_lessons = [
                    FailureLesson(
                        failure_point=lesson.get("failure_point", ""),
                        flawed_reasoning=lesson.get("flawed_reasoning", ""),
                        counterfactual=lesson.get("counterfactual", ""),
                        prevention_principle=lesson.get("prevention_principle", ""),
                        task_context=lesson.get("task_context", ""),
                        timestamp=lesson.get("timestamp", ""),
                        evolved=bool(lesson.get("evolved", False)),
                    )
                    for lesson in lessons
                ]
        return pb

    def save_to_backend(self, backend: Any, scope: str = "project") -> None:
        """Save the current Playbook state back to a StorageBackend.

        Phase 5a review fixes:
          C1 — persists failure_lessons (was previously silent-dropped; from_backend
               loaded them but save never wrote, causing asymmetric round-trip).
          C2 — wraps the full body in a single outer txn on the scoped SQLite
               connection, calling `_nc` (no-commit) variants of the playbook
               primitives so a mid-save exception rolls the whole change back.
               The file backend falls back to per-call commits (its primitives
               have their own per-file atomic writes, so partial rollback is
               not achievable but also not a concurrent-reader problem — file
               writes are already atomic via rename).
        """
        now = datetime.now(timezone.utc).isoformat()

        # Detect SQLite backend by duck-typing on the _nc helpers we added.
        # Inside a `with conn:` block, all mutations either commit on clean
        # exit or roll back on exception, so concurrent readers never see
        # partial state. File-backend path keeps the pre-existing per-call
        # commits (its primitives are already atomic at the file level).
        has_nc = hasattr(backend, "_bullet_insert_nc")
        conn = None
        if has_nc and hasattr(backend, "_get_scoped_conn"):
            try:
                conn = backend._get_scoped_conn(scope)
            except Exception:
                conn = None

        existing_bullets = backend.bullet_list(scope=scope)
        existing_ids = {b["id"] for b in existing_bullets}
        current_ids = {b.id for b in self._bullets}

        # Load lessons once up-front so we can dedup and compute evolution deltas.
        # failure_lessons_all returns dict[bullet_id -> list[lesson_dict]].
        existing_lessons = backend.failure_lessons_all(scope)

        def _commit_body(c: Any) -> None:
            # Sections
            if has_nc and c is not None:
                backend._playbook_sections_set_nc(c, self._sections)
            else:
                backend.playbook_sections_set(self._sections, scope)

            # Deletes (bullets removed from memory). Cascade via FK should clean
            # failure_lessons; no explicit lesson delete needed.
            for bid in existing_ids - current_ids:
                if has_nc and c is not None:
                    backend._bullet_delete_nc(c, bid)
                else:
                    backend.bullet_delete(bid, scope)

            # Upserts
            for bullet in self._bullets:
                bd = {
                    "id": bullet.id, "section": bullet.section,
                    "content": bullet.content, "helpful": bullet.helpful,
                    "harmful": bullet.harmful, "scope": bullet.scope,
                    "when_to_apply": bullet.when_to_apply,
                    "trigger_text": bullet.trigger, "action": bullet.action,
                    "weighted_helpful": bullet.weighted_helpful,
                    "weighted_harmful": bullet.weighted_harmful,
                    "personal_decay_rate": bullet.personal_decay_rate,
                    "grpo_advantage": bullet.grpo_advantage,
                    "last_updated": bullet.last_updated,
                    "created_at": bullet.last_updated or now,
                }
                if bullet.id in existing_ids:
                    if has_nc and c is not None:
                        backend._bullet_update_nc(c, bullet.id, bd)
                    else:
                        backend.bullet_update(bullet.id, bd, scope)
                else:
                    if has_nc and c is not None:
                        backend._bullet_insert_nc(c, bd)
                    else:
                        backend.bullet_insert(bd, scope)

            # Failure lessons: insert NEW lessons (dedup by (failure_point, timestamp)).
            # Also propagate evolved=True for any existing lesson now marked evolved
            # in-memory that wasn't marked evolved in the DB. We use the existing
            # per-bullet `failure_lessons_mark_evolved` API (it marks ALL unevolved
            # lessons for a bullet). If the in-memory copy has any evolved lesson
            # that corresponds to an unevolved DB row, mark all evolved for that
            # bullet. Coarser than ideal (API takes only bullet_id), but matches
            # the existing contract and keeps the migration minimal.
            for bullet in self._bullets:
                if not bullet.failure_lessons:
                    continue
                db_lessons = existing_lessons.get(bullet.id, [])
                db_keys = {
                    (lesson.get("failure_point", ""), lesson.get("timestamp", ""))
                    for lesson in db_lessons
                }
                mem_any_evolved = False
                mem_evolved_keys: set[tuple[str, str]] = set()
                for mem_lesson in bullet.failure_lessons:
                    lesson_dict = (
                        mem_lesson.to_dict()
                        if hasattr(mem_lesson, "to_dict") else dict(mem_lesson)
                    )
                    key = (
                        lesson_dict.get("failure_point", ""),
                        lesson_dict.get("timestamp", ""),
                    )
                    if lesson_dict.get("evolved"):
                        mem_any_evolved = True
                        mem_evolved_keys.add(key)
                    if key not in db_keys:
                        if has_nc and c is not None:
                            backend._failure_lessons_insert_nc(c, bullet.id, lesson_dict)
                        else:
                            backend.failure_lessons_insert(bullet.id, lesson_dict, scope)

                # Propagate evolution: if any in-memory lesson is evolved AND any
                # existing DB lesson is NOT yet evolved, mark the bullet's unevolved
                # DB lessons as evolved. (The existing API is per-bullet, not per-
                # lesson; accept this coarser granularity.)
                if mem_any_evolved and any(
                    not bool(lesson.get("evolved", False)) for lesson in db_lessons
                ):
                    if has_nc and c is not None and hasattr(
                        backend, "_failure_lessons_mark_evolved_nc"
                    ):
                        backend._failure_lessons_mark_evolved_nc(c, bullet.id)
                    elif hasattr(backend, "failure_lessons_mark_evolved"):
                        backend.failure_lessons_mark_evolved(bullet.id, scope)

        if has_nc and conn is not None:
            # Single outer transaction. On exception Python rolls back the
            # whole body, so concurrent readers never see partial state.
            with conn:
                _commit_body(conn)
        else:
            # File backend path (or SQLite fallback if _get_scoped_conn failed):
            # per-call commits; primitives are atomic at the file/row level but
            # not collectively atomic across the whole save.
            _commit_body(None)

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
        if op.bullet_id == op.merge_target:
            return 0
        # Combine counts
        keeper.helpful += absorbed.helpful
        keeper.harmful += absorbed.harmful
        keeper.weighted_helpful += absorbed.weighted_helpful
        keeper.weighted_harmful += absorbed.weighted_harmful
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
