"""ACE Playbook Tools — all ace_* MCP tool functions + helpers."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from datetime import datetime, timezone

from mcp.types import ToolAnnotations
from mcp.server.fastmcp.exceptions import ToolError

from ccr.ace.playbook import DeltaOperation, Playbook, parse_delta_operations
from ccr.core.types import PlaybookSchema, SchemaMetrics
from ccr.mcp.server import mcp
import ccr.mcp.server as _srv

# All server functions/globals accessed via _srv to support test patching.

# ---------------------------------------------------------------------------
# Module-level idempotency key store (B3)
# ---------------------------------------------------------------------------

_applied_idempotency_keys: set[str] = set()


# ---------------------------------------------------------------------------
# Helper: resolve .ccr/ directory for a given scope (B2, B6)
# ---------------------------------------------------------------------------


def _get_playbook_dir(scope: str) -> str:
    """Return the .ccr/ directory path for the given scope."""
    if scope == "global":
        return os.path.expanduser("~/.ccr")
    # Project scope: derive from _playbook_path
    pp = _srv._playbook_path
    if pp:
        return os.path.dirname(pp)
    # Fallback: use project root
    return os.path.join(_srv._project_root, ".ccr")


from ccr.mcp_types import (
    AceApplyDeltaResult,
    AceEvolveFromFailuresResult,
    AceEvolveSchemaResult,
    AceFindSimilarResult,
    AceGenerateBulletsResult,
    AcePlaybookResult,
    AcePruneResult,
    AceUpdateCountersResult,
)
from ccr.utils.parsing import extract_json_string


# ===========================================================================
# ACE Playbook Tools
# ===========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def ace_get_playbook(
    task_context: str = "",
    include_stats: bool = False,
    section: str = "",
    keyword: str = "",
) -> AcePlaybookResult:
    """Get the current ACE playbook — both global and project-specific strategies.

    Returns two tiers:
    - GLOBAL: Universal heuristics that transfer across all projects (~/.ccr/)
    - PROJECT: Project-specific strategies (.ccr/)

    Review this after loading context to learn from past successes and failures.

    Args:
        task_context: Optional task description. When provided, prepends a
            policy-ranked top-5 section weighted by relevance to your task.
        include_stats: If True, append per-bullet stats (counts, decay,
            section breakdown, evolution triggers). Replaces ace_get_stats.
        section: Optional section name filter (e.g., "STRATEGIES & INSIGHTS").
            Only bullets from this section are returned. Case-insensitive.
        keyword: Optional keyword filter on bullet content. Only bullets
            containing this keyword (case-insensitive) are returned.
            Combine with section= for targeted retrieval.
    """
    gpb = _srv._ensure_global_playbook()
    ppb = _srv._ensure_playbook()

    parts = []

    # Policy-ranked section when task_context is provided
    if task_context.strip():
        p_ranked = ppb.get_policy_ranked(task_context, top_k=5)
        g_ranked = gpb.get_policy_ranked(task_context, top_k=5)
        seen: set[str] = set()
        merged: list = []
        for b in p_ranked + g_ranked:
            if b.id not in seen:
                seen.add(b.id)
                merged.append(b)
        ranked = sorted(merged, key=lambda b: b.grpo_advantage, reverse=True)[:5]
        if ranked:
            ranked_lines = ["# Policy-ranked skills for this task:"]
            for b in ranked:
                adv = f"{b.grpo_advantage:+.3f}"
                line = f"[{b.id}] (advantage={adv}) :: {b.content}"
                if b.trigger:
                    line += f"\n  TRIGGER: {b.trigger}"
                if b.action:
                    line += f"\n  ACTION: {b.action}"
                ranked_lines.append(line)
            parts.append("\n".join(ranked_lines))

    # Apply section/keyword filters to bullet lists if requested
    _sec_lower = section.strip().lower()
    _kw_lower = keyword.strip().lower()

    def _filter_bullets(pb_obj) -> list:
        """Return filtered bullet list if section/keyword filters active."""
        bullets = pb_obj.bullets
        if _sec_lower:
            bullets = [b for b in bullets if _sec_lower in b.section.lower()]
        if _kw_lower:
            bullets = [b for b in bullets if _kw_lower in b.content.lower()]
        return bullets

    if _sec_lower or _kw_lower:
        # Filtered mode: render only matching bullets
        g_bullets = _filter_bullets(gpb)
        g_lines = "\n".join(b.format_line() for b in g_bullets)
        filter_desc = " | ".join(f for f in [f"section={section!r}" if section else "", f"keyword={keyword!r}" if keyword else ""] if f)
        p_bullets = _filter_bullets(ppb)
        p_lines = "\n".join(b.format_line() for b in p_bullets)
        parts.append(f"# GLOBAL PLAYBOOK [filtered: {filter_desc}] ({len(g_bullets)} bullets)\n{g_lines or '(no matches)'}")
        parts.append(f"# PROJECT PLAYBOOK [filtered: {filter_desc}] ({len(p_bullets)} bullets)\n{p_lines or '(no matches)'}")
    else:
        g_text = _srv._serialize_playbook(gpb)
        if g_text.strip():
            parts.append(f"# GLOBAL PLAYBOOK (applies to all projects)\n{g_text}")
        else:
            parts.append("# GLOBAL PLAYBOOK (applies to all projects)\n(empty)")

        p_text = _srv._serialize_playbook(ppb)
        if p_text.strip():
            parts.append(f"# PROJECT PLAYBOOK (this project only)\n{p_text}")
        else:
            parts.append("# PROJECT PLAYBOOK (this project only)\n(empty)")

    # Append stats if requested (replaces standalone ace_get_stats)
    if include_stats:
        stats_data = json.dumps({
            "global": asdict(gpb.get_stats()),
            "project": asdict(ppb.get_stats()),
        }, indent=2)
        parts.append(f"# PLAYBOOK STATS\n{stats_data}")

    text = "\n\n".join(parts)
    return AcePlaybookResult(
        global_bullet_count=len(gpb.bullets),
        project_bullet_count=len(ppb.bullets),
        message=text,
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def ace_apply_delta(
    operations: list[dict],
    scope: str = "project",
    dry_run: bool = False,
    atomic: bool = False,
    author: str = "",
) -> AceApplyDeltaResult:
    """Apply delta operations to the playbook.

    Supported operations:
        ADD: {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "..."}
        UPDATE: {"type": "UPDATE", "bullet_id": "str-00001", "content": "new content"}
        MERGE: {"type": "MERGE", "bullet_id": "str-00001", "merge_target": "str-00002", "content": "merged"}
        REMOVE: {"type": "REMOVE", "bullet_id": "str-00001"}

    Use ADD when you discover a new insight or strategy.
    Use UPDATE to refine existing bullets.
    Use MERGE to combine similar bullets.
    Use REMOVE to delete unhelpful bullets.

    For ADD operations, optional ERL-inspired trigger/action fields enable
    context-aware retrieval (trigger matches get 1.5x weight boost):
        - trigger: When condition (e.g., "when adding API endpoints")
        - action: What to do (e.g., "add validation middleware first")

    Args:
        operations: List of delta operation dicts.
        scope: "project" (default) or "global". Global bullets apply across all projects.
        dry_run: If True, preview what would happen without modifying the playbook.
            Reports which bullet_ids would be found/missing, counts ADDs/UPDATEs/MERGEs/REMOVEs,
            and shows new playbook size — without actually saving any changes.
        atomic: If True, roll back all changes on any exception during apply.
        author: Optional author identifier recorded in the history log.
    """
    try:
        with _srv._state_lock:
            pb, save_fn = _srv._resolve_playbook(scope)
            ops = parse_delta_operations({"operations": operations})
            # Track bullet IDs that should exist for UPDATE/MERGE/REMOVE ops
            bullet_ids_before = {b.id for b in pb.bullets}
            failed_ids = [
                op.bullet_id
                for op in ops
                if op.op_type in ("UPDATE", "MERGE", "REMOVE")
                and op.bullet_id
                and op.bullet_id not in bullet_ids_before
            ]

            if dry_run:
                # Preview mode: show what would happen without modifying
                by_type: dict[str, int] = {}
                for op in ops:
                    by_type[op.op_type] = by_type.get(op.op_type, 0) + 1
                n_adds = by_type.get("ADD", 0)
                n_removes = by_type.get("REMOVE", 0)
                estimated_new_count = len(pb.bullets) + n_adds - n_removes
                lines = [f"[dry_run] Would apply {len(ops)} operation(s) to {scope} playbook."]
                for op_type, count in sorted(by_type.items()):
                    lines.append(f"  {op_type}: {count}")
                if failed_ids:
                    lines.append(f"  WARN: {len(failed_ids)} operation(s) would fail (missing bullet_id): {failed_ids}")
                lines.append(f"  Estimated bullet count after: {estimated_new_count} (currently {len(pb.bullets)})")
                lines.append("No changes made. Remove dry_run=True to apply.")
                result: AceApplyDeltaResult = {"applied": 0, "scope": scope, "message": "\n".join(lines)}
                if failed_ids:
                    result["failed_ids"] = failed_ids
                return result

            # Atomic snapshot before apply (B2)
            snapshot: str | None = None
            if atomic:
                snapshot = pb.serialize()

            try:
                applied = pb.apply_delta(ops)
                save_fn()
            except Exception:
                if atomic and snapshot is not None:
                    # Rollback: restore playbook from snapshot
                    try:
                        restored = Playbook(snapshot)
                        pb._bullets = restored._bullets  # type: ignore[attr-defined]
                        pb._next_id = restored._next_id  # type: ignore[attr-defined]
                        pb._id_index = restored._id_index  # type: ignore[attr-defined]
                        save_fn()
                    except Exception:
                        pass
                raise

        # Mark matching patterns as promoted (CER buffer close-the-loop)
        promoted_count = 0
        with _srv._state_lock:
            mem = _srv._ensure_memory()
        for op in ops:
            if op.op_type == "ADD" and op.content:
                try:
                    promoted_count += mem.mark_pattern_promoted_by_content(op.content)
                except Exception:
                    pass  # Never fail the delta apply

        text = f"Applied {applied} operation(s) to {scope} playbook. Now has {len(pb.bullets)} bullets."
        if promoted_count:
            text += f" Marked {promoted_count} pattern(s) as promoted in CER buffer."
        if failed_ids:
            text += f"\nWarning: {len(failed_ids)} operation(s) referenced missing bullet ID(s): {failed_ids}"

        # Write history log (B2)
        history_path: str | None = None
        try:
            ccr_dir = _get_playbook_dir(scope)
            history_path = os.path.join(ccr_dir, "playbook_history.json")
            existing: list[dict] = []
            if os.path.isfile(history_path):
                try:
                    with open(history_path, "r", encoding="utf-8") as fh:
                        existing = json.load(fh)
                    if not isinstance(existing, list):
                        existing = []
                except Exception:
                    existing = []
            entry: dict = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "author": author,
                "ops_count": len(ops),
                "scope": scope,
                "applied": applied,
                "failed_ids": failed_ids,
            }
            existing.append(entry)
            tmp_path = history_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(existing, fh, indent=2)
            os.replace(tmp_path, history_path)
        except Exception:
            pass  # Never fail the apply operation

        result: AceApplyDeltaResult = {"applied": applied, "scope": scope, "message": text}
        if failed_ids:
            result["failed_ids"] = failed_ids
        if history_path:
            result["delta_history_path"] = history_path
        if author:
            result["author"] = author
        return result
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def ace_update_counters(
    bullet_tags: list[dict],
    scope: str = "project",
    idempotency_key: str = "",
) -> AceUpdateCountersResult:
    """Update helpful/harmful counters for playbook bullets.

    After completing a task, reflect on which strategies helped or hurt,
    then update their counters. This drives evolutionary selection — high-scoring
    bullets persist, low-scoring ones get pruned.

    When tagging a bullet as "harmful", you SHOULD include a structured failure
    lesson explaining WHY it failed. This makes the harmful tag actionable:

    Args:
        bullet_tags: List of dicts. Each dict has:
            - "id": bullet ID (e.g., "str-00001")
            - "tag": "helpful" or "harmful"
            - "weight" (optional, 0.0-1.0, default 1.0): Contribution weight for
              proportional credit/blame when multiple strategies were active
              (AgentEvolver-inspired). Integer counters always increment by 1;
              weight accumulates in weighted_helpful/weighted_harmful.
            - "failure_lesson" (optional, for harmful tags): {
                "failure_point": "Where the strategy broke down",
                "flawed_reasoning": "What incorrect assumption was made",
                "counterfactual": "What should have been done instead",
                "prevention_principle": "General rule to avoid this failure"
              }
        scope: "project" (default) or "global".
        idempotency_key: If non-empty, prevents double-applying the same update.
            A second call with the same key returns early without modifying the playbook.
    """
    # Idempotency check (B3)
    if idempotency_key and idempotency_key in _applied_idempotency_keys:
        return AceUpdateCountersResult(
            updated=0,
            scope=scope,
            message=f"Already applied (idempotency_key: {idempotency_key!r}). Skipped.",
        )

    # Validate failure_lesson dicts — collect warnings, don't block (B3)
    _required_lesson_keys = {
        "failure_point", "flawed_reasoning", "counterfactual", "prevention_principle"
    }
    validation_warnings: list[str] = []
    for i, tag_entry in enumerate(bullet_tags):
        lesson = tag_entry.get("failure_lesson")
        if isinstance(lesson, dict) and lesson:
            missing_keys = _required_lesson_keys - set(lesson.keys())
            if missing_keys:
                validation_warnings.append(
                    f"failure_lesson[{i}] missing keys: {sorted(missing_keys)}"
                )

    try:
        with _srv._state_lock:
            pb, save_fn = _srv._resolve_playbook(scope)
            updated = pb.update_bullet_counts(bullet_tags)
            # Recompute GRPO advantages after counter update (SkillRL Eq.3)
            pb.recompute_grpo_advantages()
            save_fn()

        # Mark idempotency key as used after successful update (B3)
        if idempotency_key:
            _applied_idempotency_keys.add(idempotency_key)

        # Propagate quality feedback to source patterns (EvolveR-inspired)
        quality_propagated = 0
        try:
            with _srv._state_lock:
                mem = _srv._ensure_memory()
            for tag_entry in bullet_tags:
                bullet_id = tag_entry.get("id", "")
                tag = tag_entry.get("tag", "")
                if tag in ("helpful", "harmful"):
                    for b in pb.bullets:
                        if b.id == bullet_id:
                            if mem.update_pattern_quality(
                                b.content, success=(tag == "helpful")
                            ):
                                quality_propagated += 1
                            break
        except Exception:
            pass  # Quality propagation is best-effort

        # Count how many had structured lessons
        lessons_added = sum(
            1 for t in bullet_tags
            if t.get("tag") == "harmful" and isinstance(t.get("failure_lesson"), dict) and t["failure_lesson"]
        )
        # Compute which requested IDs were not found in the playbook
        found_ids = {b.id for b in pb.bullets}
        requested_ids = [t.get("id", "") for t in bullet_tags if t.get("id")]
        missing = [bid for bid in requested_ids if bid not in found_ids]

        parts = [f"Updated {updated} bullet(s) in {scope} playbook."]
        if missing:
            parts.append(f"IDs not found: {', '.join(missing)}.")
        if lessons_added:
            parts.append(f"Recorded {lessons_added} structured failure lesson(s).")
        if quality_propagated:
            parts.append(f"Propagated quality to {quality_propagated} source pattern(s).")
        if validation_warnings:
            parts.append("Validation warnings: " + "; ".join(validation_warnings))
        text = " ".join(parts)
        result: AceUpdateCountersResult = {"updated": updated, "scope": scope, "message": text}
        if missing:
            result["missing_ids"] = missing
        return result
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e



@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def ace_find_similar(
    threshold: float = 0.6,
    scope: str = "project",
    section: str = "",
    auto_merge_above: float | None = None,
) -> AceFindSimilarResult:
    """Find similar bullet pairs that may be candidates for merging.

    Uses Jaccard + trigram similarity. Returns pairs above the threshold
    so you can decide whether to MERGE them.

    Args:
        threshold: Similarity threshold (0.0-1.0, default 0.6).
        scope: "project" (default), "global", or "cross" (find duplicates between tiers).
        section: Optional section filter. Only pairs where at least one bullet's
            section contains this string (case-insensitive) are returned.
        auto_merge_above: If set, pairs with similarity >= this value are
            automatically merged (MERGE delta op). Count returned in message.
    """
    threshold = max(0.0, min(threshold, 1.0))
    if scope == "cross":
        gpb = _srv._ensure_global_playbook()
        ppb = _srv._ensure_playbook()
        # Cross-tier: compare every global bullet against every project bullet
        # Primary: ONNX cosine. Fallback: word+trigram Jaccard.
        from ccr.context.embeddings import get_embedding_model
        model = get_embedding_model()
        g_texts = [gb.content for gb in gpb.bullets]
        p_texts = [pb.content for pb in ppb.bullets]
        g_embeddings = None
        p_embeddings = None
        if model is not None and g_texts and p_texts:
            try:
                g_embeddings = model.embed_batch(g_texts)
                p_embeddings = model.embed_batch(p_texts)
            except Exception:
                g_embeddings = None
                p_embeddings = None

        pairs = []
        for gi, gb in enumerate(gpb.bullets):
            if len(gb.content.split()) < 2:
                continue
            for pi, pb_bullet in enumerate(ppb.bullets):
                if len(pb_bullet.content.split()) < 2:
                    continue
                if g_embeddings is not None and p_embeddings is not None:
                    combined = float(g_embeddings[gi] @ p_embeddings[pi])
                else:
                    words_a = set(gb.content.lower().split())
                    words_b = set(pb_bullet.content.lower().split())
                    trigrams_a = Playbook._char_trigrams(gb.content.lower())
                    trigrams_b = Playbook._char_trigrams(pb_bullet.content.lower())
                    word_inter = words_a & words_b
                    word_union = words_a | words_b
                    word_jaccard = len(word_inter) / len(word_union) if word_union else 0.0
                    tri_inter = trigrams_a & trigrams_b
                    tri_union = trigrams_a | trigrams_b
                    tri_jaccard = len(tri_inter) / len(tri_union) if tri_union else 0.0
                    combined = 0.4 * word_jaccard + 0.6 * tri_jaccard
                if combined >= threshold:
                    pairs.append((gb, pb_bullet, combined))
        pairs.sort(key=lambda x: x[2], reverse=True)

        # Section filter (B5)
        if section:
            sec_lower = section.lower()
            pairs = [
                (a, b, sim) for a, b, sim in pairs
                if sec_lower in a.section.lower() or sec_lower in b.section.lower()
            ]

        if not pairs:
            return AceFindSimilarResult(pairs_found=0, scope=scope, message="No cross-tier similar bullet pairs found.")
        lines = ["Cross-tier similarities (global vs project):"]
        for a, b, sim in pairs[:10]:
            lines.append(f"[{a.id}] (global) vs [{b.id}] (project) (similarity={sim:.2f})")
            lines.append(f"  G: {a.content[:100]}")
            lines.append(f"  P: {b.content[:100]}")
        text = "\n".join(lines)
        return AceFindSimilarResult(pairs_found=len(pairs), scope=scope, message=text)

    pb, save_fn = _srv._resolve_playbook(scope)
    pairs = pb.find_similar_pairs(threshold)

    # Section filter (B5)
    if section:
        sec_lower = section.lower()
        pairs = [
            (a, b, sim) for a, b, sim in pairs
            if sec_lower in a.section.lower() or sec_lower in b.section.lower()
        ]

    # Auto-merge pairs above threshold (B5)
    # Re-resolve playbook under lock before writing to avoid TOCTOU race
    auto_merged_count = 0
    if auto_merge_above is not None:
        with _srv._state_lock:
            pb, save_fn = _srv._resolve_playbook(scope)
            for a, b, sim in pairs:
                if sim >= auto_merge_above:
                    try:
                        merged_content = (
                            a.content if a.helpful >= b.helpful else b.content
                        )
                        keeper_id = a.id if a.helpful >= b.helpful else b.id
                        absorbed_id = b.id if keeper_id == a.id else a.id
                        merge_op = DeltaOperation(
                            op_type="MERGE",
                            section="",
                            content=merged_content,
                            bullet_id=keeper_id,
                            merge_target=absorbed_id,
                        )
                        pb.apply_delta([merge_op])
                        auto_merged_count += 1
                    except Exception:
                        pass  # Never fail the find operation
            if auto_merged_count:
                try:
                    save_fn()
                except Exception:
                    pass

    if not pairs:
        text = f"No similar bullet pairs found in {scope} playbook."
        return AceFindSimilarResult(pairs_found=0, scope=scope, message=text)
    lines = []
    for a, b, sim in pairs[:10]:
        lines.append(f"[{a.id}] vs [{b.id}] (similarity={sim:.2f})")
        lines.append(f"  A: {a.content[:100]}")
        lines.append(f"  B: {b.content[:100]}")
    if auto_merged_count:
        lines.append(f"\nAuto-merged {auto_merged_count} pair(s) above threshold {auto_merge_above}.")
    text = "\n".join(lines)
    pairs_list = []
    for a, b, sim in pairs[:25]:
        score_a = a.effective_score() * (1.0 + a.grpo_advantage)
        score_b = b.effective_score() * (1.0 + b.grpo_advantage)
        pairs_list.append({
            "a_id": a.id, "b_id": b.id,
            "similarity": round(sim, 3),
            "a_content": a.content[:120],
            "b_content": b.content[:120],
            "recommended_keep": a.id if score_a >= score_b else b.id,
        })
    return AceFindSimilarResult(pairs_found=len(pairs), scope=scope, pairs=pairs_list, message=text)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False))
def ace_prune(scope: str = "project", archive: bool = True) -> AcePruneResult:
    """Prune problematic bullets and enforce token budget.

    First evolves failure lessons into new skills (threshold=1, aggressive),
    then removes bullets where harmful >= helpful and harmful >= 3,
    then trims lowest-scoring bullets if playbook exceeds 80K chars.

    This ordering ensures failure lessons are distilled into new heuristic
    bullets before the harmful source bullets are removed.

    Args:
        scope: "project" (default) or "global".
        archive: If True (default), write pruned bullets to .ccr/archived_bullets.json
            before removing them.
    """
    try:
        with _srv._state_lock:
            pb, save_fn = _srv._resolve_playbook(scope)
            # Load schema for parameterized thresholds (MCE)
            sp = _srv._schema_path if scope == "project" else _srv._global_schema_path
            _schema_warn = ""
            if sp and not os.path.isfile(sp):
                _schema_warn = f" (schema file not found at {sp}; using defaults)"
            schema = _srv._load_schema(sp) if sp else PlaybookSchema.default()

            # Snapshot bullets that will be pruned BEFORE pruning (B6)
            prune_candidates = {
                b.id: {
                    "content": b.content,
                    "section": b.section,
                    "score": b.effective_score(),
                }
                for b in pb.bullets
                if b.harmful >= schema.prune_min_harmful and b.harmful >= b.helpful
            }

            # Evolve failure lessons BEFORE pruning to prevent permanent loss
            evolved = pb.evolve_from_failures(threshold=max(1, schema.evolution_threshold - 2))
            pruned = pb.prune_problematic(min_harmful=schema.prune_min_harmful)
            budget_pruned = pb.enforce_token_budget(
                max_chars=schema.token_budget, decay_rate=schema.decay_rate,
            )

            # Capture budget-pruned bullets for archive too (B6)
            for b in budget_pruned:
                if b.id not in prune_candidates:
                    prune_candidates[b.id] = {
                        "content": b.content,
                        "section": b.section,
                        "score": b.effective_score(),
                    }

            save_fn()

        # Archive pruned bullets (B6)
        if archive and prune_candidates:
            try:
                ccr_dir = _get_playbook_dir(scope)
                archive_path = os.path.join(ccr_dir, "archived_bullets.json")
                entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": "pruned",
                    "bullets": [
                        {"id": bid, **data}
                        for bid, data in prune_candidates.items()
                    ],
                }
                existing_archive: list[dict] = []
                if os.path.isfile(archive_path):
                    try:
                        with open(archive_path, "r", encoding="utf-8") as fh:
                            existing_archive = json.load(fh)
                        if not isinstance(existing_archive, list):
                            existing_archive = []
                    except Exception:
                        existing_archive = []
                existing_archive.append(entry)
                tmp_archive_path = archive_path + ".tmp"
                with open(tmp_archive_path, "w", encoding="utf-8") as fh:
                    json.dump(existing_archive, fh, indent=2)
                os.replace(tmp_archive_path, archive_path)
            except Exception:
                pass  # Never fail the prune operation

        total = len(pruned) + len(budget_pruned)
        removed_ids = [b.id for b in pruned] + [b.id for b in budget_pruned]
        parts = []
        if evolved:
            parts.append(f"Evolved {len(evolved)} new skill(s) from failure lessons.")
        parts.append(
            f"Pruned {total} bullet(s) ({', '.join(removed_ids) if removed_ids else 'none'}) "
            f"from {scope} playbook. Now has {len(pb.bullets)} bullets."
        )
        text = " ".join(parts)
        if _schema_warn:
            text += _schema_warn

        result: AcePruneResult = {
            "removed": total,
            "evolved": len(evolved),
            "scope": scope,
            "message": text,
        }
        if removed_ids:
            result["removed_ids"] = removed_ids
        return result
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e


# ---------------------------------------------------------------------------
# Re-exports from split submodules (test patchability + backward compat)
# ---------------------------------------------------------------------------

from ccr.mcp.ace_llm_tools import (  # noqa: E402,F401
    _word_jaccard,
    _semantic_or_jaccard,
    _auto_synthesize_skills,
    _ace_generator,
    _ace_reflector,
    _ace_curator,
    _run_ace_pipeline,
    ace_generate_bullets,
    ace_evolve_from_failures,
)
from ccr.mcp.ace_schema_tools import (  # noqa: E402,F401
    _apply_rollback,
    _build_retrieval_proposals,
    ace_evolve_schema,
)

