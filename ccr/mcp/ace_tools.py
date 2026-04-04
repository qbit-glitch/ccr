"""ACE Playbook Tools — all ace_* MCP tool functions + helpers."""

from __future__ import annotations

import json
from dataclasses import asdict

from mcp.types import ToolAnnotations
from mcp.server.fastmcp.exceptions import ToolError

from ccr.ace.playbook import DeltaOperation, Playbook, parse_delta_operations
from ccr.core.types import PlaybookSchema, SchemaMetrics
from ccr.mcp.server import mcp
import ccr.mcp.server as _srv

# All server functions/globals accessed via _srv to support test patching.
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
def ace_get_playbook(task_context: str = "", include_stats: bool = False) -> AcePlaybookResult:
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
def ace_apply_delta(operations: list[dict], scope: str = "project") -> AceApplyDeltaResult:
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
            applied = pb.apply_delta(ops)
            save_fn()

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
        result: AceApplyDeltaResult = {"applied": applied, "scope": scope, "message": text}
        if failed_ids:
            result["failed_ids"] = failed_ids
        return result
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def ace_update_counters(bullet_tags: list[dict], scope: str = "project") -> AceUpdateCountersResult:
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
    """
    try:
        with _srv._state_lock:
            pb, save_fn = _srv._resolve_playbook(scope)
            updated = pb.update_bullet_counts(bullet_tags)
            # Recompute GRPO advantages after counter update (SkillRL Eq.3)
            pb.recompute_grpo_advantages()
            save_fn()

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
def ace_find_similar(threshold: float = 0.6, scope: str = "project") -> AceFindSimilarResult:
    """Find similar bullet pairs that may be candidates for merging.

    Uses Jaccard + trigram similarity. Returns pairs above the threshold
    so you can decide whether to MERGE them.

    Args:
        threshold: Similarity threshold (0.0-1.0, default 0.6).
        scope: "project" (default), "global", or "cross" (find duplicates between tiers).
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
        if not pairs:
            return AceFindSimilarResult(pairs_found=0, scope=scope, message="No cross-tier similar bullet pairs found.")
        lines = ["Cross-tier similarities (global vs project):"]
        for a, b, sim in pairs[:10]:
            lines.append(f"[{a.id}] (global) vs [{b.id}] (project) (similarity={sim:.2f})")
            lines.append(f"  G: {a.content[:100]}")
            lines.append(f"  P: {b.content[:100]}")
        text = "\n".join(lines)
        return AceFindSimilarResult(pairs_found=len(pairs), scope=scope, message=text)

    pb, _ = _srv._resolve_playbook(scope)
    pairs = pb.find_similar_pairs(threshold)
    if not pairs:
        text = f"No similar bullet pairs found in {scope} playbook."
        return AceFindSimilarResult(pairs_found=0, scope=scope, message=text)
    lines = []
    for a, b, sim in pairs[:10]:
        lines.append(f"[{a.id}] vs [{b.id}] (similarity={sim:.2f})")
        lines.append(f"  A: {a.content[:100]}")
        lines.append(f"  B: {b.content[:100]}")
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
def ace_prune(scope: str = "project") -> AcePruneResult:
    """Prune problematic bullets and enforce token budget.

    First evolves failure lessons into new skills (threshold=1, aggressive),
    then removes bullets where harmful >= helpful and harmful >= 3,
    then trims lowest-scoring bullets if playbook exceeds 80K chars.

    This ordering ensures failure lessons are distilled into new heuristic
    bullets before the harmful source bullets are removed.

    Args:
        scope: "project" (default) or "global".
    """
    try:
        with _srv._state_lock:
            pb, save_fn = _srv._resolve_playbook(scope)
            # Load schema for parameterized thresholds (MCE)
            sp = _srv._schema_path if scope == "project" else _srv._global_schema_path
            schema = _srv._load_schema(sp) if sp else PlaybookSchema.default()
            # Evolve failure lessons BEFORE pruning to prevent permanent loss
            evolved = pb.evolve_from_failures(threshold=max(1, schema.evolution_threshold - 2))
            pruned = pb.prune_problematic(min_harmful=schema.prune_min_harmful)
            budget_pruned = pb.enforce_token_budget(
                max_chars=schema.token_budget, decay_rate=schema.decay_rate,
            )
            save_fn()
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


def _word_jaccard(a: str, b: str) -> float:
    """Compute word-level Jaccard similarity between two strings."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _semantic_or_jaccard(a: str, b: str) -> float:
    """ONNX cosine similarity with word Jaccard fallback."""
    from ccr.context.embeddings import quick_cosine
    result = quick_cosine(a, b)
    return result if result is not None else _word_jaccard(a, b)


def _auto_synthesize_skills(candidates: list[dict], sub_client) -> list[dict]:
    """Cross-synthesize generalized skills from related failure lessons (SkillRL §3.3).

    Groups candidates by task_context similarity (word Jaccard >= 0.5), then calls
    the sub-model once per group of 2+ candidates to generate 1-2 generalized
    "When X, do Y" skills that abstract across all prevention_principles in the group.

    Single-candidate groups are skipped — they were already handled by the mechanical
    path (prevention_principle verbatim copy).

    Args:
        candidates: List of dicts with keys: bullet_slug, failure_lesson (dict with
            prevention_principle, task_context, failure_point).
        sub_client: A ClaudeClient instance (or compatible) with .completion() method.

    Returns:
        List of dicts with keys: content (str), when_to_apply (str), scope (str).
        Returns [] on any error (graceful no-op).
    """
    try:
        if not candidates:
            return []

        # Greedy grouping by task_context similarity (threshold 0.5)
        groups: list[list[dict]] = []
        for cand in candidates:
            tc = cand.get("failure_lesson", {}).get("task_context", "")
            placed = False
            for group in groups:
                anchor_tc = group[0].get("failure_lesson", {}).get("task_context", "")
                if _semantic_or_jaccard(tc, anchor_tc) >= 0.5:
                    group.append(cand)
                    placed = True
                    break
            if not placed:
                groups.append([cand])

        synthesized: list[dict] = []
        for group in groups:
            if len(group) < 2:
                # Single-candidate groups handled by mechanical path — skip
                continue
            principles = [
                c.get("failure_lesson", {}).get("prevention_principle", "")
                for c in group
            ]
            # Filter empty principles
            principles = [p for p in principles if p.strip()]
            if not principles:
                continue

            principles_text = "\n".join(f"- {p}" for p in principles)
            prompt = (
                "You are synthesizing generalized skills from related failure lessons.\n\n"
                f"Failure lessons (all from similar task contexts):\n{principles_text}\n\n"
                'Write 1-2 generalized "When X, do Y" skills that abstract across all of these.\n'
                'Respond with a JSON array: [{"content": "...", "when_to_apply": "..."}]\n'
                "Be concise. Each skill should be 1-2 sentences."
            )

            try:
                raw = sub_client.completion([{"role": "user", "content": prompt}])
                json_str = extract_json_string(raw)
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and item.get("content", "").strip():
                            synthesized.append({
                                "content": item["content"].strip(),
                                "when_to_apply": item.get("when_to_apply", ""),
                                "scope": "project",
                            })
            except Exception:
                # Silent failure per spec — one bad group doesn't block others
                continue

        return synthesized
    except Exception:
        return []


# ===========================================================================
# ACE 3-Agent Pipeline (Generator -> Reflector -> Curator) — ACE §3.1-3.3
# ===========================================================================


def _ace_generator(trajectory: str, sub_client) -> list[str]:
    """Generate candidate strategy bullets from task trajectory (ACE §3.1)."""
    try:
        prompt = (
            "You are an AI strategy generator. Given this task trajectory:\n"
            f"{trajectory}\n\n"
            "Generate 2-3 actionable strategy bullets for a coding assistant playbook.\n"
            'Each bullet should be: specific, actionable, abstract (not tied to this exact task).\n'
            'Format as "When X, do Y" or "Always Y when Z".\n'
            'Respond with a JSON array: ["bullet 1", "bullet 2"]'
        )
        response = sub_client.completion([{"role": "user", "content": prompt}])
        raw = extract_json_string(response)
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(p) for p in parsed if p]
        return []
    except Exception:
        return []


def _ace_reflector(candidates: list[str], trajectory: str, sub_client) -> list[str]:
    """Filter/score candidates by quality and relevance (ACE §3.2)."""
    try:
        if not candidates:
            return []
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
        prompt = (
            "Rate each strategy bullet for quality (1-5):\n"
            "- 5: Highly actionable, specific, generalizable\n"
            "- 3: Decent but could be improved\n"
            "- 1: Too vague or too specific\n\n"
            f"Task context: {trajectory}\n"
            f"Candidates:\n{numbered}\n\n"
            'Respond with JSON: [{"bullet": "...", "score": N}, ...]\n'
            "Only include bullets with score >= 3."
        )
        response = sub_client.completion([{"role": "user", "content": prompt}])
        raw = extract_json_string(response)
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [
                item["bullet"]
                for item in parsed
                if isinstance(item, dict) and item.get("score", 0) >= 3 and item.get("bullet", "").strip()
            ]
        return candidates  # fallback: return original unfiltered
    except Exception:
        return candidates  # fallback: return original unfiltered on error


def _ace_curator(existing_bullets: list[str], candidates: list[str], sub_client) -> list[dict]:
    """Decide ADD/MERGE/SKIP for each candidate vs existing playbook (ACE §3.3)."""
    try:
        if not candidates:
            return []
        existing_sample = "\n".join(f"- {b}" for b in existing_bullets)
        n_shown = len(existing_bullets)
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
        prompt = (
            "For each candidate strategy bullet, decide: ADD (novel), MERGE (similar to existing), SKIP (duplicate).\n\n"
            f"Existing bullets (top {n_shown}):\n{existing_sample}\n\n"
            f"Candidates:\n{numbered}\n\n"
            'Respond with JSON: [{"bullet": "...", "action": "ADD"|"MERGE"|"SKIP", "merge_with": "existing bullet text if MERGE else null"}]'
        )
        response = sub_client.completion([{"role": "user", "content": prompt}])
        raw = extract_json_string(response)
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            result = []
            for item in parsed:
                if isinstance(item, dict) and item.get("bullet", "").strip():
                    result.append({
                        "bullet": item["bullet"],
                        "action": item.get("action", "ADD"),
                        "merge_with": item.get("merge_with"),
                    })
            return result
        return [{"bullet": c, "action": "ADD", "merge_with": None} for c in candidates]
    except Exception:
        return [{"bullet": c, "action": "ADD", "merge_with": None} for c in candidates]


def _run_ace_pipeline(title: str, what: str, why: str, sub_client) -> None:
    """Run the ACE 3-agent Generator -> Reflector -> Curator pipeline after a commit.

    Failures are silently swallowed — this must never affect the commit result.
    """
    try:
        # Build enriched trajectory: commit + rolling context + top bullets (ACE §3.1)
        trajectory = f"Title: {title}\nWhat: {what}\nWhy: {why}"
        try:
            mem = _srv._ensure_memory()
            rolling = mem._get_rolling_summary("main") or ""
            if rolling:
                trajectory += f"\n\nProject context (rolling summary):\n{rolling[:600]}"
        except Exception:
            pass
        try:
            pb = _srv._ensure_playbook()
            top_bullets = [b.content for b in pb.bullets[:5]]
            if top_bullets:
                trajectory += "\n\nExisting playbook (top 5):\n" + "\n".join(f"- {b}" for b in top_bullets)
        except Exception:
            pass

        # Step 1: Generator
        candidates = _ace_generator(trajectory, sub_client)
        if not candidates:
            return

        # Step 2: Reflector
        filtered = _ace_reflector(candidates, trajectory, sub_client)
        if not filtered:
            return

        # Step 3: Curator — pass up to 50 bullets sorted by grpo_advantage (matches ace_generate_bullets logic)
        with _srv._state_lock:
            pb = _srv._ensure_playbook()
        if len(pb.bullets) <= 50:
            existing_sample = [b.content for b in pb.bullets]
        else:
            existing_sample = [b.content for b in sorted(pb.bullets, key=lambda b: b.grpo_advantage, reverse=True)[:50]]
        decisions = _ace_curator(existing_sample, filtered, sub_client)

        # Step 4: Apply decisions
        with _srv._state_lock:
            pb = _srv._ensure_playbook()
            for decision in decisions:
                action = decision.get("action", "ADD")
                bullet_text = decision.get("bullet", "").strip()
                if not bullet_text:
                    continue
                if action == "ADD":
                    op = DeltaOperation(
                        op_type="ADD",
                        section="STRATEGIES & INSIGHTS",
                        content=bullet_text,
                    )
                    pb.apply_delta([op])
                elif action == "MERGE":
                    merge_with = decision.get("merge_with", "")
                    if merge_with:
                        # Find closest existing bullet (ONNX primary, Jaccard fallback)
                        best_id = None
                        best_sim = 0.0
                        for b in pb.bullets:
                            sim = _semantic_or_jaccard(b.content, merge_with)
                            if sim > best_sim:
                                best_sim = sim
                                best_id = b.id
                        if best_id and best_sim > 0.0:
                            # ADD new bullet, then MERGE it into keeper to combine counters
                            add_op = DeltaOperation(
                                op_type="ADD",
                                section="STRATEGIES & INSIGHTS",
                                content=bullet_text,
                            )
                            pb.apply_delta([add_op])
                            new_id = pb.bullets[-1].id  # newly added bullet
                            merge_op = DeltaOperation(
                                op_type="MERGE",
                                section="",
                                content=bullet_text,  # merged content
                                bullet_id=best_id,    # keeper (existing)
                                merge_target=new_id,  # absorbed (new)
                            )
                            pb.apply_delta([merge_op])
                # SKIP: do nothing
            _srv._save_playbook()
    except Exception:
        pass  # Never raise — commit result must not be affected


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def ace_generate_bullets(context: str, auto_apply: bool = False) -> AceGenerateBulletsResult:
    """Generate and optionally apply strategy bullets via the ACE 3-agent pipeline.

    Runs Generator -> Reflector -> Curator to produce candidate playbook bullets
    from a task context or trajectory. Returns a preview of all decisions.

    Args:
        context: Task context or trajectory to generate bullets from.
        auto_apply: If True, automatically apply ADD decisions. Default False (preview only).
    """
    try:
        sub = _srv._get_sub_client()
        if sub is None:
            text = "No sub-model configured. Set CCR_OLLAMA_MODEL or ANTHROPIC_API_KEY to enable ACE pipeline."
            return AceGenerateBulletsResult(decisions=0, applied=0, message=text)

        candidates = _ace_generator(context, sub)
        if not candidates:
            return AceGenerateBulletsResult(decisions=0, applied=0, message="Generator produced no candidates.")

        filtered = _ace_reflector(candidates, context, sub)
        if not filtered:
            text = f"Reflector filtered all {len(candidates)} candidate(s). None passed quality threshold."
            return AceGenerateBulletsResult(decisions=0, applied=0, message=text)

        with _srv._state_lock:
            pb = _srv._ensure_playbook()
        bullets = pb.bullets
        if len(bullets) <= 50:
            existing_sample = [b.content for b in bullets]
        else:
            # For large playbooks, show top-50 by grpo_advantage so Curator sees the best bullets
            sorted_bullets = sorted(bullets, key=lambda b: b.grpo_advantage, reverse=True)
            existing_sample = [b.content for b in sorted_bullets[:50]]
        decisions = _ace_curator(existing_sample, filtered, sub)

        if not decisions:
            return AceGenerateBulletsResult(decisions=0, applied=0, message="Curator produced no decisions.")

        lines = ["# ACE Pipeline Results"]
        applied_count = 0
        if auto_apply:
            with _srv._state_lock:
                pb = _srv._ensure_playbook()
                for decision in decisions:
                    action = decision.get("action", "ADD")
                    bullet_text = decision.get("bullet", "").strip()
                    if not bullet_text:
                        continue
                    if action == "ADD":
                        op = DeltaOperation(
                            op_type="ADD",
                            section="STRATEGIES & INSIGHTS",
                            content=bullet_text,
                        )
                        pb.apply_delta([op])
                        applied_count += 1
                    elif action == "MERGE":
                        merge_with = decision.get("merge_with", "")
                        if merge_with:
                            best_id = None
                            best_sim = 0.0
                            for b in pb.bullets:
                                sim = _semantic_or_jaccard(b.content, merge_with)
                                if sim > best_sim:
                                    best_sim = sim
                                    best_id = b.id
                            if best_id and best_sim > 0.0:
                                # Two-step: ADD candidate, then MERGE it into keeper
                                # (bullet_id=keeper, merge_target=newly added)
                                add_op = DeltaOperation(
                                    op_type="ADD",
                                    section="STRATEGIES & INSIGHTS",
                                    content=bullet_text,
                                )
                                pb.apply_delta([add_op])
                                new_id = pb.bullets[-1].id if pb.bullets else None
                                if new_id and new_id != best_id:
                                    merge_op = DeltaOperation(
                                        op_type="MERGE",
                                        section="",
                                        content=bullet_text,
                                        bullet_id=best_id,
                                        merge_target=new_id,
                                    )
                                    pb.apply_delta([merge_op])
                                applied_count += 1
                _srv._save_playbook()
            lines.append(f"Applied {applied_count} decision(s) to project playbook.")

        lines.append(f"\n## Decisions ({len(decisions)} total):")
        for d in decisions:
            action = d.get("action", "?")
            bullet = d.get("bullet", "")[:120]
            merge_with = d.get("merge_with", "")
            if merge_with:
                lines.append(f"- [{action}] {bullet}\n  (merge with: {str(merge_with)[:80]})")
            else:
                lines.append(f"- [{action}] {bullet}")

        text = "\n".join(lines)
        return AceGenerateBulletsResult(decisions=len(decisions), applied=applied_count, message=text)
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e


def ace_evolve_from_failures(
    threshold: int = 3,
    scope: str = "project",
    synthesized_skills: list[dict] | None = None,
) -> AceEvolveFromFailuresResult:
    """Evolve new skills from accumulated failure lessons.

    Two-step workflow:
      Step 1: Call with no synthesized_skills.
        -> Returns failure lessons and a prompt for you to synthesize skills.
      Step 2: Call again with your synthesized skills.
        -> Saves them as new playbook bullets.

    Example:
        # Step 1: Get lessons
        result = ace_evolve_from_failures()
        # Step 2: After synthesizing, save
        ace_evolve_from_failures(synthesized_skills=[
            {"content": "When X happens, do Y", "when_to_apply": "during debugging"}
        ])

    Falls back to mechanical extraction (copies prevention_principle verbatim)
    if synthesized_skills is not provided.

    Args:
        threshold: Minimum harmful-with-lessons bullets to trigger (default 3).
        scope: "project" (default) or "global".
        synthesized_skills: List of dicts with "content" and "when_to_apply" keys.
    """
    try:
        with _srv._state_lock:
            pb, save_fn = _srv._resolve_playbook(scope)

            # Path 2: Save synthesized skills from Claude Code
            if synthesized_skills:
                ops = []
                for skill in synthesized_skills:
                    content = skill.get("content", "").strip()
                    when = skill.get("when_to_apply", "")
                    if content:
                        ops.append(DeltaOperation(
                            op_type="ADD",
                            section="PROBLEM-SOLVING HEURISTICS",
                            content=content,
                        ))
                applied = pb.apply_delta(ops)
                # Set when_to_apply on newly added bullets
                for skill, bullet in zip(synthesized_skills, pb.bullets[-applied:]):
                    if skill.get("when_to_apply"):
                        bullet.when_to_apply = skill["when_to_apply"]
                    bullet.scope = skill.get("scope", "general")
                save_fn()
                text = f"Saved {applied} synthesized skill(s) to {scope} playbook."
                return AceEvolveFromFailuresResult(evolved=applied, synthesized=0, message=text)

            # Path 1: Check if evolution is needed
            check = pb.check_evolution_needed(threshold)
            if not check.get("needed"):
                text = (
                    f"Evolution not triggered in {scope} playbook. Need {threshold} harmful bullets with failure "
                    f"lessons, currently have {check['candidate_count']}."
                )
                return AceEvolveFromFailuresResult(evolved=0, synthesized=0, message=text)

            # Run mechanical evolution (backward compat + fallback)
            new_bullets = pb.evolve_from_failures(threshold)
            if new_bullets:
                save_fn()

            # Build synthesis prompt for Claude Code (teacher-model per SkillRL §4.2)
            # Include the candidates so Claude can optionally synthesize better skills
            candidate_ids = set(check.get("candidate_ids", []))
            candidates = [b for b in pb.bullets if b.id in candidate_ids]

            # Phase 2: LLM cross-synthesis of related failure lessons (SkillRL §3.3)
            # Runs when sub-client is available; graceful no-op otherwise.
            sub = _srv._get_sub_client()
            synthesized_count = 0
            if sub is not None and candidates:
                # Build candidate dicts expected by _auto_synthesize_skills
                cand_dicts = []
                for b in candidates:
                    for lesson in b.failure_lessons:
                        fl_dict = lesson if isinstance(lesson, dict) else {
                            "prevention_principle": getattr(lesson, "prevention_principle", ""),
                            "task_context": getattr(lesson, "task_context", ""),
                            "failure_point": getattr(lesson, "failure_point", ""),
                        }
                        cand_dicts.append({
                            "bullet_slug": b.id,
                            "failure_lesson": fl_dict,
                        })
                if cand_dicts:
                    synthesized = _auto_synthesize_skills(cand_dicts, sub)
                    for skill in synthesized:
                        content = skill.get("content", "").strip()
                        if content:
                            op = DeltaOperation(
                                op_type="ADD",
                                section="PROBLEM-SOLVING HEURISTICS",
                                content=content,
                            )
                            applied_count = pb.apply_delta([op])
                            # Set when_to_apply and scope on the newly added bullet
                            if applied_count > 0:
                                new_b = pb.bullets[-1]
                                if skill.get("when_to_apply"):
                                    new_b.when_to_apply = skill["when_to_apply"]
                                new_b.scope = skill.get("scope", "project")
                                synthesized_count += 1
                    if synthesized_count > 0:
                        save_fn()

            lessons_text = []
            for b in candidates:
                lessons_text.append(f"## Bullet [{b.id}]: {b.content[:100]}")
                for lesson in b.failure_lessons:
                    fp = getattr(lesson, 'failure_point', '?') if not isinstance(lesson, dict) else lesson.get('failure_point', '?')
                    fr = getattr(lesson, 'flawed_reasoning', '?') if not isinstance(lesson, dict) else lesson.get('flawed_reasoning', '?')
                    cf = getattr(lesson, 'counterfactual', '?') if not isinstance(lesson, dict) else lesson.get('counterfactual', '?')
                    pp = getattr(lesson, 'prevention_principle', '?') if not isinstance(lesson, dict) else lesson.get('prevention_principle', '?')
                    lessons_text.append(
                        f"  - Failure: {fp}\n"
                        f"    Flawed reasoning: {fr}\n"
                        f"    Counterfactual: {cf}\n"
                        f"    Prevention: {pp}"
                    )

            result_parts = []
            if new_bullets:
                result_parts.append(f"Evolved {len(new_bullets)} new skill(s) from {scope} failure lessons:")
                for b in new_bullets:
                    result_parts.append(f"  [{b.id}] {b.content[:100]}")
                    if b.when_to_apply:
                        result_parts.append(f"    When: {b.when_to_apply[:100]}")
            if synthesized_count > 0:
                result_parts.append(f"Synthesized {synthesized_count} cross-lesson skill(s) via sub-model (SkillRL §3.3).")

            if lessons_text:
                result_parts.append(
                    f"\n---\n# Optional: Caller-driven synthesis (SkillRL §4.2)\n"
                    f"For higher-quality skill synthesis, review these failure lessons and call\n"
                    f"ace_evolve_from_failures again with synthesized_skills=[\n"
                    f'  {{"content": "skill text", "when_to_apply": "condition", "scope": "general"}}\n'
                    f"]\n\n" + "\n".join(lessons_text)
                )

            text = "\n".join(result_parts) if result_parts else "No skills evolved."
            return AceEvolveFromFailuresResult(
                evolved=len(new_bullets),
                synthesized=synthesized_count,
                message=text,
            )
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e


def _apply_rollback(pb, schema, history, save_fn, sp, max_hist: int = 20) -> AceEvolveSchemaResult:
    """Revert playbook to the parent schema version.

    Shared by the rollback=True branch and the ROLLBACK proposal change type.

    Args:
        pb: Playbook instance (already locked by caller).
        schema: Current PlaybookSchema.
        history: List of schema history dicts.
        save_fn: Callable that persists the playbook.
        sp: Schema file path (may be None).
        max_hist: Maximum history entries to retain.

    Returns:
        AceEvolveSchemaResult with the rollback outcome.
    """
    if schema.parent_version is None:
        return AceEvolveSchemaResult(
            version=schema.version,
            message="Cannot rollback: no parent schema version exists.",
        )
    parent = None
    for h in history:
        if h.get("version") == schema.parent_version:
            parent = PlaybookSchema.from_dict(h)
            break
    if parent is None:
        return AceEvolveSchemaResult(
            version=schema.version,
            message=f"Cannot rollback: parent version {schema.parent_version} not found in history.",
        )
    # Apply parent schema to playbook
    moved = pb.apply_schema(parent)
    # Compute new metrics
    new_metrics = pb.compute_metrics(parent)
    parent.baseline_metrics = new_metrics
    # Push current schema to history, restore parent
    history.append(schema.to_dict())
    if len(history) > max_hist:
        history = history[-max_hist:]
    _srv._save_schema(parent, history, sp)
    save_fn()
    text = (
        f"Rolled back to schema v{parent.version}.\n"
        f"Moved {moved} bullet(s) between sections.\n"
        f"Current health: {new_metrics.overall_health:.3f}"
    )
    return AceEvolveSchemaResult(
        version=parent.version,
        health=new_metrics.overall_health,
        message=text,
    )


def _build_retrieval_proposals(metrics, schema, sc: int) -> list:
    """Build schema proposals for retrieval tuning (ADJUST_SEARCH_THRESHOLD, ADJUST_SCAN_WINDOW).

    Deduplicates the near-identical ALMA-inspired retrieval proposal blocks that appear
    in both the apply_proposal path and the evaluation path of ace_evolve_schema.

    Args:
        metrics: SchemaMetrics with search_zero_rate and link_density populated.
        schema: Current PlaybookSchema with link_semantic_threshold and link_scan_window.
        sc: Number of search_calls from memory metrics (gates ADJUST_SCAN_WINDOW).

    Returns:
        List of SchemaProposal objects (0-1 entries).
    """
    from ccr.core.types import SchemaProposal
    proposals = []
    if metrics.search_zero_rate > 0.5 and schema.link_semantic_threshold > 0.05:
        new_thresh = round(schema.link_semantic_threshold - 0.05, 2)
        proposals.append(SchemaProposal(
            change_type="ADJUST_SEARCH_THRESHOLD",
            description=(
                f"{metrics.search_zero_rate:.0%} of searches returned zero results. "
                f"Propose lowering link_semantic_threshold from "
                f"{schema.link_semantic_threshold} to {new_thresh} "
                f"(more permissive matching)."
            ),
            details={
                "old_threshold": schema.link_semantic_threshold,
                "new_threshold": new_thresh,
            },
            confidence=min(1.0, (metrics.search_zero_rate - 0.5) / 0.3),
        ))
    elif metrics.link_density < 0.5 and sc > 0:
        new_window = schema.link_scan_window + 5
        proposals.append(SchemaProposal(
            change_type="ADJUST_SCAN_WINDOW",
            description=(
                f"Link density is {metrics.link_density:.2f} links/commit (below 0.5). "
                f"Propose increasing link_scan_window from "
                f"{schema.link_scan_window} to {new_window} "
                f"(scan more history for cross-links)."
            ),
            details={
                "old_window": schema.link_scan_window,
                "new_window": new_window,
            },
            confidence=min(1.0, (0.5 - metrics.link_density) / 0.3),
        ))
    return proposals


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def ace_evolve_schema(
    scope: str = "project",
    apply_proposal: int | None = None,
    rollback: bool = False,
) -> AceEvolveSchemaResult:
    """Evolve playbook structure — sections, thresholds, decay parameters.

    Three modes:
      Evaluate (no args): Show health metrics and improvement proposals.
        -> ace_evolve_schema()
      Apply (apply_proposal=N): Apply the Nth proposed change.
        -> ace_evolve_schema(apply_proposal=1)
      Rollback (rollback=True): Revert to previous schema version.
        -> ace_evolve_schema(rollback=True)

    Stops proposing changes when overall health >= 0.8.

    Args:
        scope: "project" or "global".
        apply_proposal: 1-indexed proposal to apply (None = evaluate only).
        rollback: Revert to parent schema version.
    """
    from datetime import datetime, timezone

    try:
        with _srv._state_lock:
            pb, save_fn = _srv._resolve_playbook(scope)
            sp = _srv._schema_path if scope == "project" else _srv._global_schema_path
            schema = _srv._load_schema(sp) if sp else PlaybookSchema.default()
            history = _srv._load_schema_history(sp) if sp else []

            if rollback:
                return _apply_rollback(pb, schema, history, save_fn, sp)

            # Compute current metrics
            metrics = pb.compute_metrics(schema)

            # Enrich metrics with memory retrieval stats (ALMA-inspired)
            mem = _srv._ensure_memory()
            mem_metrics = mem.get_memory_metrics()
            sc = mem_metrics.get("search_calls", 0)
            szr = mem_metrics.get("search_zero_results", 0)
            lc = mem_metrics.get("link_creations", 0)
            tc = mem_metrics.get("total_commits", 0)
            metrics.search_zero_rate = szr / sc if sc > 0 else 0.0
            metrics.link_density = lc / tc if tc > 0 else 0.0
            # embedding_coverage: requires counting commits with cached embeddings
            # — left at 0.0 for now (would need scanning embed cache)

            if apply_proposal is not None:
                # Re-compute proposals to validate index
                proposals = pb.propose_schema_changes(
                    schema, metrics,
                    overflow_threshold=0.5,
                    min_cluster_size=3,
                    stop_health_threshold=0.8,
                    rollback_health_delta=-0.05,
                )
                # Append ALMA-inspired retrieval parameter proposals
                if not proposals:
                    proposals.extend(_build_retrieval_proposals(metrics, schema, sc))
                idx = apply_proposal - 1
                if idx < 0 or idx >= len(proposals):
                    return AceEvolveSchemaResult(
                        version=schema.version,
                        message=f"Invalid proposal index {apply_proposal}. Available: {len(proposals)}.",
                    )
                proposal = proposals[idx]

                # Create new schema version
                new_schema = PlaybookSchema(
                    version=schema.version + 1,
                    sections=list(schema.sections),
                    slug_map=dict(schema.slug_map),
                    decay_rate=schema.decay_rate,
                    prune_min_harmful=schema.prune_min_harmful,
                    evolution_threshold=schema.evolution_threshold,
                    token_budget=schema.token_budget,
                    parent_version=schema.version,
                    change_description=proposal.description,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    baseline_metrics=metrics,
                    link_scan_window=schema.link_scan_window,
                    link_semantic_threshold=schema.link_semantic_threshold,
                    context_level_default=schema.context_level_default,
                    search_result_limit=schema.search_result_limit,
                )

                # Apply the change
                ct = proposal.change_type
                details = proposal.details

                if ct == "ADD_SECTION":
                    new_name = details["name"]
                    slug_prefix = details.get("slug_prefix", new_name[:3].lower())
                    new_schema.sections.append(new_name)
                    from ccr.ace.playbook import _normalize_section
                    new_schema.slug_map[_normalize_section(new_name)] = slug_prefix
                    # Move specified bullets
                    bullet_ids = set(details.get("bullet_ids", []))
                    for bullet in pb.bullets:
                        if bullet.id in bullet_ids:
                            bullet.section = new_name
                    pb.apply_schema(new_schema)

                elif ct == "REMOVE_SECTION":
                    rm_name = details["name"]
                    if rm_name in new_schema.sections:
                        new_schema.sections.remove(rm_name)
                    pb.apply_schema(new_schema)

                elif ct == "ADJUST_DECAY":
                    new_schema.decay_rate = details["new_rate"]

                elif ct == "ADJUST_PRUNING":
                    new_schema.prune_min_harmful = details["new_min_harmful"]

                elif ct == "ADJUST_EVOLUTION":
                    new_schema.evolution_threshold = details["new_threshold"]

                elif ct == "ADJUST_BUDGET":
                    new_schema.token_budget = details["new_budget"]

                elif ct == "REBALANCE":
                    moves = details.get("moves", [])
                    for move in moves:
                        b = pb.get_bullet(move["bullet_id"])
                        if b:
                            b.section = move["to"]

                elif ct == "ADJUST_SEARCH_THRESHOLD":
                    new_schema.link_semantic_threshold = details["new_threshold"]

                elif ct == "ADJUST_SCAN_WINDOW":
                    new_schema.link_scan_window = details["new_window"]

                elif ct == "ROLLBACK":
                    return _apply_rollback(pb, schema, history, save_fn, sp)

                # Sync schema retrieval params to memory manager
                mem = _srv._ensure_memory()
                mem.set_schema_overrides({
                    "link_scan_window": new_schema.link_scan_window,
                    "link_semantic_threshold": new_schema.link_semantic_threshold,
                })

                # Push old schema to history
                history.append(schema.to_dict())
                max_hist = 20
                if len(history) > max_hist:
                    history = history[-max_hist:]

                _srv._save_schema(new_schema, history, sp)
                save_fn()

                new_metrics = pb.compute_metrics(new_schema)
                delta = new_metrics.overall_health - metrics.overall_health
                text = (
                    f"Applied {ct} -> schema v{new_schema.version}.\n"
                    f"{proposal.description}\n"
                    f"Health: {metrics.overall_health:.3f} -> {new_metrics.overall_health:.3f} "
                    f"({chr(916)}{delta:+.3f})"
                )
                return AceEvolveSchemaResult(
                    version=new_schema.version,
                    health=new_metrics.overall_health,
                    message=text,
                )

            # Evaluation mode: compute proposals
            proposals = pb.propose_schema_changes(
                schema, metrics,
                overflow_threshold=0.5,
                min_cluster_size=3,
                stop_health_threshold=0.8,
                rollback_health_delta=-0.05,
            )

            # ALMA-inspired: propose retrieval parameter adjustments from memory metrics
            # Only propose retrieval adjustments when no structural proposals pending
            if not proposals:
                proposals.extend(_build_retrieval_proposals(metrics, schema, sc))

            parts = [
                f"# Schema Health Report (v{schema.version})",
                f"",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Section Balance | {metrics.section_balance:.3f} |",
                f"| Utilization Rate | {metrics.utilization_rate:.3f} |",
                f"| Harmful Ratio | {metrics.harmful_ratio:.3f} |",
                f"| Unused Ratio | {metrics.unused_ratio:.3f} |",
                f"| Decay Impact | {metrics.decay_impact:.3f} |",
                f"| **Overall Health** | **{metrics.overall_health:.3f}** |",
                f"| Total Bullets | {metrics.total_bullets} |",
                f"| Total Sections | {metrics.total_sections} |",
            ]

            if metrics.empty_sections:
                parts.append(f"| Empty Sections | {', '.join(metrics.empty_sections)} |")
            if metrics.overflow_sections:
                parts.append(f"| Overflow Sections | {', '.join(metrics.overflow_sections)} |")

            # Baseline comparison
            if schema.baseline_metrics:
                bl = schema.baseline_metrics
                delta = metrics.overall_health - bl.overall_health
                parts.extend([
                    f"",
                    f"## Baseline Comparison (v{schema.version} adopted at {bl.overall_health:.3f})",
                    f"Health delta: {delta:+.3f}",
                ])

            # Memory retrieval metrics (ALMA-inspired)
            parts.extend([
                f"| Search Zero Rate | {metrics.search_zero_rate:.3f} |",
                f"| Link Density | {metrics.link_density:.3f} |",
                f"| Embedding Coverage | {metrics.embedding_coverage:.3f} |",
            ])

            # Schema parameters
            parts.extend([
                f"",
                f"## Schema Parameters",
                f"- Decay rate: {schema.decay_rate}",
                f"- Prune min harmful: {schema.prune_min_harmful}",
                f"- Evolution threshold: {schema.evolution_threshold}",
                f"- Token budget: {schema.token_budget}",
                f"- Sections: {len(schema.sections)}",
                f"- Link scan window: {schema.link_scan_window}",
                f"- Link semantic threshold: {schema.link_semantic_threshold}",
                f"- Context level default: {schema.context_level_default}",
                f"- Search result limit: {schema.search_result_limit}",
            ])

            # Proposals
            if proposals:
                parts.extend([f"", f"## Proposals"])
                for i, p in enumerate(proposals, 1):
                    parts.append(
                        f"  {i}. **{p.change_type}** (confidence: {p.confidence:.2f})\n"
                        f"     {p.description}"
                    )
                parts.append(
                    f"\nCall `ace_evolve_schema(apply_proposal=N)` to apply, "
                    f"or `ace_evolve_schema(rollback=True)` to revert."
                )
            else:
                parts.append(f"\n## No proposals — playbook schema is healthy.")

            text = "\n".join(parts)
            return AceEvolveSchemaResult(
                version=schema.version,
                health=metrics.overall_health,
                message=text,
            )
    except ValueError:
        raise
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e
