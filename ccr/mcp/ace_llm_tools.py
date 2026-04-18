"""ACE LLM Pipeline Tools — Generator/Reflector/Curator + failure evolution.

Extracted from ace_tools.py to satisfy 800-line coding-style limit.
"""

from __future__ import annotations

import json

from mcp.types import ToolAnnotations
from mcp.server.fastmcp.exceptions import ToolError

from ccr.ace.playbook import DeltaOperation
from ccr.mcp.server import mcp
import ccr.mcp.server as _srv

from ccr.mcp_types import (
    AceEvolveFromFailuresResult,
    AceGenerateBulletsResult,
)
from ccr.utils.parsing import extract_json_string


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------


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

        # Step 3: Curator — pass up to 50 bullets sorted by grpo_advantage
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
def ace_generate_bullets(
    context: str,
    auto_apply: bool = False,
    confirm_indices: list[int] | None = None,
) -> AceGenerateBulletsResult:
    """Generate and optionally apply strategy bullets via the ACE 3-agent pipeline.

    Runs Generator -> Reflector -> Curator to produce candidate playbook bullets
    from a task context or trajectory. Returns a preview of all decisions.

    Args:
        context: Task context or trajectory to generate bullets from.
        auto_apply: If True, automatically apply ADD decisions. Default False (preview only).
        confirm_indices: Optional list of decision indices (0-based) to apply.
            When provided with auto_apply=True, only the specified decisions are applied.
            When auto_apply=False, populates pending_decisions in the result.
    """
    try:
        sub = _srv._get_sub_client()
        if sub is None:
            text = "No sub-model configured. Set CCR_OLLAMA_MODEL or ANTHROPIC_API_KEY to enable ACE pipeline."
            return AceGenerateBulletsResult(decisions=0, applied=0, pending_decisions=[], message=text)

        candidates = _ace_generator(context, sub)
        if not candidates:
            return AceGenerateBulletsResult(decisions=0, applied=0, pending_decisions=[], message="Generator produced no candidates.")

        filtered = _ace_reflector(candidates, context, sub)
        if not filtered:
            text = f"Reflector filtered all {len(candidates)} candidate(s). None passed quality threshold."
            return AceGenerateBulletsResult(decisions=0, applied=0, pending_decisions=[], message=text)

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
            return AceGenerateBulletsResult(decisions=0, applied=0, pending_decisions=[], message="Curator produced no decisions.")

        # Build pending_decisions list for preview / confirm_indices (B4)
        pending_decisions: list[dict] = []
        for i, d in enumerate(decisions):
            action = d.get("action", "ADD")
            bullet_text = d.get("bullet", "")
            pending_decisions.append({
                "index": i,
                "op_type": action,
                "content": bullet_text,
            })

        lines = ["# ACE Pipeline Results"]
        applied_count = 0
        if auto_apply:
            # When confirm_indices given, restrict to those indices only (B4)
            indices_to_apply: set[int] | None = None
            if confirm_indices is not None:
                indices_to_apply = set(confirm_indices)

            with _srv._state_lock:
                pb = _srv._ensure_playbook()
                for i, decision in enumerate(decisions):
                    if indices_to_apply is not None and i not in indices_to_apply:
                        continue
                    action = decision.get("action", "ADD")
                    bullet_text = decision.get("bullet", "").strip()
                    if not bullet_text:
                        continue
                    if action == "ADD":
                        # Append audit tag to bullet content (B4)
                        tagged_content = bullet_text + " [_generated_by: ace_generate]"
                        op = DeltaOperation(
                            op_type="ADD",
                            section="STRATEGIES & INSIGHTS",
                            content=tagged_content,
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
        return AceGenerateBulletsResult(
            decisions=len(decisions),
            applied=applied_count,
            pending_decisions=pending_decisions if (not auto_apply or confirm_indices is not None) else [],
            message=text,
        )
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
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
