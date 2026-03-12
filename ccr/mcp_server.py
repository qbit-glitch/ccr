"""CCR MCP Server — exposes GCC memory, ACE playbook, RLM sandbox, and repo index as tools.

Claude Code connects to this server and gains persistent memory, self-evolving
playbooks, a sandboxed REPL, and repo search — all without a sub-model or API keys.

Usage:
    python -m ccr.mcp_server              # stdio transport (for Claude Code)
    python -m ccr.mcp_server --project .  # explicit project root
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from dataclasses import asdict

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ccr.ace.playbook import DeltaOperation, FailureLesson, Playbook, parse_delta_operations
from ccr.context.indexer import RepoIndex
from ccr.core.memory import MemoryManager
from ccr.core.types import CCRConfig
from ccr.rlm.repl import CCRRepl

# ---------------------------------------------------------------------------
# Globals — initialized once at startup
# ---------------------------------------------------------------------------

# M3: Module-level lock for thread safety on global state mutations
_state_lock = threading.Lock()

_project_root: str = ""
_memory: MemoryManager | None = None
_playbook: Playbook | None = None
_playbook_path: str = ""
_failure_lessons_path: str = ""
_global_playbook: Playbook | None = None
_global_playbook_path: str = ""
_global_failure_lessons_path: str = ""
_repo_index: RepoIndex | None = None
_repl: CCRRepl | None = None

mcp = FastMCP(
    "ccr",
    instructions=(
        "CCR gives you persistent project memory (GCC), self-evolving strategy "
        "playbooks (ACE), a sandboxed Python REPL (RLM), and repo indexing. "
        "Use gcc_* tools for memory, ace_* for playbook management, rlm_* for "
        "sandboxed code execution, and index_* for repo search."
    ),
)


def _init(project_root: str | None = None) -> None:
    """Initialize all subsystems for the given project root."""
    global _project_root, _memory, _playbook, _playbook_path, _failure_lessons_path
    global _global_playbook, _global_playbook_path, _global_failure_lessons_path, _repo_index

    _project_root = os.path.abspath(project_root or os.getcwd())
    _memory = MemoryManager(_project_root, CCRConfig())
    _memory.ensure_structure()

    # Clear session marker so hooks detect new session on first prompt
    session_marker = os.path.join(_project_root, ".ccr", ".session_active")
    if os.path.isfile(session_marker):
        try:
            os.remove(session_marker)
        except OSError:
            pass

    _playbook_path = os.path.join(_project_root, ".ccr", "playbook.txt")
    _failure_lessons_path = os.path.join(_project_root, ".ccr", "failure_lessons.json")
    _playbook = _load_playbook()

    # Initialize global playbook (~/.ccr/)
    global_ccr = os.path.expanduser("~/.ccr")
    os.makedirs(global_ccr, exist_ok=True)
    _global_playbook_path = os.path.join(global_ccr, "global_playbook.txt")
    _global_failure_lessons_path = os.path.join(global_ccr, "global_failure_lessons.json")
    _global_playbook = _load_global_playbook()

    # Build repo index
    _repo_index = RepoIndex.build(_project_root)

    # Cache index
    try:
        _memory.save_index(_repo_index.to_json())
    except Exception:
        pass


def _atomic_write(path: str, content: str) -> None:
    """Write content to path atomically via tmp + fsync + os.replace (H5)."""
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_playbook() -> Playbook:
    """Load playbook from disk or create empty. Also loads failure lessons."""
    if os.path.isfile(_playbook_path):
        with open(_playbook_path, "r", encoding="utf-8") as f:
            pb = Playbook(f.read())
    else:
        pb = Playbook()
    # Load structured failure lessons from companion JSON
    if _failure_lessons_path:
        pb.load_failure_lessons(_failure_lessons_path)
    return pb


def _save_playbook() -> None:
    """Persist playbook and failure lessons to disk (H5: atomic writes)."""
    if _playbook is not None:
        _atomic_write(_playbook_path, _playbook.serialize())
        # Save failure lessons to companion JSON
        if _failure_lessons_path:
            _playbook.save_failure_lessons(_failure_lessons_path)


def _load_global_playbook() -> Playbook:
    """Load global playbook from ~/.ccr/ or create empty."""
    if os.path.isfile(_global_playbook_path):
        with open(_global_playbook_path, "r", encoding="utf-8") as f:
            pb = Playbook(f.read())
    else:
        pb = Playbook()
    if _global_failure_lessons_path:
        pb.load_failure_lessons(_global_failure_lessons_path)
    return pb


def _save_global_playbook() -> None:
    """Persist global playbook to ~/.ccr/ (H5: atomic writes)."""
    if _global_playbook is not None:
        _atomic_write(_global_playbook_path, _global_playbook.serialize())
        if _global_failure_lessons_path:
            _global_playbook.save_failure_lessons(_global_failure_lessons_path)


def _ensure_global_playbook() -> Playbook:
    if _global_playbook is None:
        _init()
    assert _global_playbook is not None
    return _global_playbook


def _ensure_memory() -> MemoryManager:
    if _memory is None:
        _init()
    assert _memory is not None
    return _memory


def _ensure_playbook() -> Playbook:
    if _playbook is None:
        _init()
    assert _playbook is not None
    return _playbook


def _ensure_index() -> RepoIndex:
    if _repo_index is None:
        _init()
    assert _repo_index is not None
    return _repo_index


# ===========================================================================
# GCC Memory Tools
# ===========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def gcc_commit(
    title: str,
    what: str,
    why: str,
    files_changed: list[str],
    next_step: str,
    admission_threshold: float = 0.85,
    rejection_threshold: float = 0.0,
) -> str:
    """Commit progress to project memory.

    Creates a structured commit record in .ccr/ with what was done, why,
    which files changed, and what to do next. Builds a rolling summary
    that progressively captures the full project history.

    Use after meaningful progress — completing a feature, fixing a bug,
    reaching a milestone, or before context gets large.

    Admission control (A-MAC Algorithm 1, correct polarity):
    1. Computes S(m) and S(m_conflict) — admission scores (higher = more valuable)
    2. Novelty N(m) = 1 - max similarity across k=5 recent commits (Eq. 3)
    3. Similarity is recency-modulated: old conflicts dampened via R = exp(-0.01·hours) (Eq. 4)
    4. If S(m) < rejection_threshold → reject (low value, correct polarity)
    5. If similarity >= admission_threshold → FindConflict found
       - If S(m) > S(m_conflict) → create new (new outranks existing)
       - If S(m) <= S(m_conflict) → merge into existing
    6. Otherwise → create new commit

    Args:
        admission_threshold: Similarity score (0-1) above which FindConflict
            detects a conflict. Default 0.85 per paper §3.3. Set to 1.0 to disable.
        rejection_threshold: Admission score below which commits are rejected.
            Default 0.0 (disabled). Low score = low value = reject.
    """
    try:
        with _state_lock:
            mem = _ensure_memory()
            return mem.commit(title, what, why, files_changed, next_step,
                              admission_threshold, rejection_threshold)
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def gcc_branch(name: str, purpose: str, hypothesis: str) -> str:
    """Create an exploration branch for experimental work.

    Branches isolate experimental changes from the main line. Must be on
    main to create a branch. Use kebab-case names (e.g., 'try-new-parser').

    Args:
        name: Branch name in kebab-case.
        purpose: What this branch explores.
        hypothesis: What you expect to learn or achieve.
    """
    try:
        with _state_lock:
            mem = _ensure_memory()
            return mem.create_branch(name, purpose, hypothesis)
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False))
def gcc_merge(branch: str, outcome: str, conclusion: str) -> str:
    """Merge an exploration branch back into main.

    Must be on the branch being merged. Integrates the branch's rolling
    summary and OTA log into main.

    Args:
        branch: Branch name to merge.
        outcome: One of 'success', 'failure', or 'partial'.
        conclusion: Summary of what was learned.
    """
    try:
        with _state_lock:
            mem = _ensure_memory()
            return mem.merge(branch, outcome, conclusion)
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_context(
    level: int = 2,
    search_term: str | None = None,
    commit_id: str | None = None,
    log_window: int = 0,
) -> str:
    """Retrieve project memory at the specified depth.

    Levels:
        1: Project overview only (~200 tokens)
        2: + rolling summary + last 3 commits
        3: + branch summary (purpose/hypothesis/conclusion)
        4: + last 10 commits
        5: + specific commit by ID or keyword search

    Use level 2 at session start for grounding. Use level 5 with
    search_term to find specific past work.

    Args:
        level: Depth of context retrieval (1-5).
        search_term: Keyword to search commits (level 5 only).
        commit_id: Specific commit ID to retrieve (level 5 only).
        log_window: Number of recent OTA log entries to include.
    """
    with _state_lock:
        mem = _ensure_memory()
    return mem.get_context(
        level=level,
        search_term=search_term,
        commit_id=commit_id,
        log_window=log_window,
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def gcc_log_ota(observation: str, thought: str, action: str) -> str:
    """Log an Observation-Thought-Action triple to the project log.

    OTA triples track your reasoning process. They're attached to
    commits and preserved across sessions. Each call appends a new entry.

    Args:
        observation: What you observed (e.g., "Test failure in router.py").
        thought: Your reasoning (e.g., "The regex pattern is too greedy").
        action: What you did (e.g., "Fixed pattern to use non-greedy match").
    """
    try:
        with _state_lock:  # H3: protect memory state mutation
            mem = _ensure_memory()
            mem.log_ota(
                tool_name="claude-code",
                observation=observation,
                thought=thought,
                action=action,
            )
        return "OTA logged."
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_status() -> str:
    """Show current project memory status.

    Returns the active branch, recent milestones, open branches,
    and metadata summary.
    """
    with _state_lock:
        mem = _ensure_memory()

    parts = []
    branch = mem.get_active_branch()
    parts.append(f"Active branch: {branch}")

    # Level-1 context (overview)
    overview = mem.get_context(level=1)
    parts.append(overview)

    return "\n\n".join(parts)


# ===========================================================================
# GCC Hierarchical Summary Tools (TiMem-inspired)
# ===========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def gcc_consolidate(tier: str = "session", content: str | None = None) -> str:
    """Generate or save a hierarchical memory summary (TiMem-inspired).

    Three tiers of consolidation, inspired by TiMem's Temporal Memory Tree
    (note: TMT tree structure and §3.3 recall pipeline are NOT implemented —
    this uses flat aggregation with mechanical consolidation):

    Tiers:
        "session": Generate a session summary from recent commits (mechanical, immediate).
                   Consolidates the last N commits into a structured paragraph.
        "phase": Generate a phase summary from recent sessions (mechanical, immediate).
                 Aggregates session summaries + branch metadata into a strategic view.
        "project": Returns a prompt for Claude Code to generate a project overview.
                   Call again with content= to save the generated overview.

    The session and phase tiers produce summaries immediately using mechanical
    consolidation (template-based extraction from commits). The project tier
    requires two calls: first to get the prompt, then to save the result.

    Args:
        tier: "session", "phase", or "project".
        content: For tier="project" only — the generated overview text to save.
    """
    try:
        with _state_lock:
            mem = _ensure_memory()
            if tier == "project" and content is not None:
                mem.save_overview(content)
                return "Project overview saved."
            return mem.get_consolidation_prompt(tier=tier)
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_summaries(tier: str = "all", count: int = 5) -> str:
    """Retrieve hierarchical memory summaries.

    Returns consolidated summaries at the specified tier:
        "session": Recent session summaries from the active branch.
        "phase": Recent phase summaries (cross-branch).
        "project": The current project overview.
        "all": All tiers combined.

    Args:
        tier: Which tier(s) to retrieve ("session", "phase", "project", "all").
        count: Maximum number of summaries per tier (default 5).
    """
    with _state_lock:
        mem = _ensure_memory()
    count = max(1, min(count, 50))
    return mem.get_summaries(tier=tier, count=count)


# ===========================================================================
# ACE Playbook Tools
# ===========================================================================


def _resolve_playbook(scope: str) -> tuple[Playbook, callable]:
    """Resolve playbook and save function based on scope."""
    if scope == "global":
        return _ensure_global_playbook(), _save_global_playbook
    return _ensure_playbook(), _save_playbook


def _serialize_playbook(pb: Playbook) -> str:
    """Serialize a playbook, using failure-aware format if lessons exist."""
    has_lessons = any(b.has_failure_lessons for b in pb.bullets)
    return pb.serialize_with_failures() if has_lessons else pb.serialize()


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def ace_get_playbook() -> str:
    """Get the current ACE playbook — both global and project-specific strategies.

    Returns two tiers:
    - GLOBAL: Universal heuristics that transfer across all projects (~/.ccr/)
    - PROJECT: Project-specific strategies (.ccr/)

    Review this after loading context to learn from past successes and failures.
    """
    gpb = _ensure_global_playbook()
    ppb = _ensure_playbook()

    parts = []
    g_text = _serialize_playbook(gpb)
    if g_text.strip():
        parts.append(f"# GLOBAL PLAYBOOK (applies to all projects)\n{g_text}")
    else:
        parts.append("# GLOBAL PLAYBOOK (applies to all projects)\n(empty)")

    p_text = _serialize_playbook(ppb)
    if p_text.strip():
        parts.append(f"# PROJECT PLAYBOOK (this project only)\n{p_text}")
    else:
        parts.append("# PROJECT PLAYBOOK (this project only)\n(empty)")

    return "\n\n".join(parts)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def ace_apply_delta(operations: list[dict], scope: str = "project") -> str:
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

    Args:
        operations: List of delta operation dicts.
        scope: "project" (default) or "global". Global bullets apply across all projects.
    """
    try:
        with _state_lock:
            pb, save_fn = _resolve_playbook(scope)
            ops = parse_delta_operations({"operations": operations})
            applied = pb.apply_delta(ops)
            save_fn()
            return f"Applied {applied} operation(s) to {scope} playbook. Now has {len(pb.bullets)} bullets."
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def ace_update_counters(bullet_tags: list[dict], scope: str = "project") -> str:
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
            - "failure_lesson" (optional, for harmful tags): {
                "failure_point": "Where the strategy broke down",
                "flawed_reasoning": "What incorrect assumption was made",
                "counterfactual": "What should have been done instead",
                "prevention_principle": "General rule to avoid this failure"
              }
        scope: "project" (default) or "global".
    """
    try:
        with _state_lock:
            pb, save_fn = _resolve_playbook(scope)
            updated = pb.update_bullet_counts(bullet_tags)
            save_fn()

        # Count how many had structured lessons
        lessons_added = sum(
            1 for t in bullet_tags
            if t.get("tag") == "harmful" and isinstance(t.get("failure_lesson"), dict) and t["failure_lesson"]
        )
        parts = [f"Updated {updated} bullet(s) in {scope} playbook."]
        if lessons_added:
            parts.append(f"Recorded {lessons_added} structured failure lesson(s).")
        return " ".join(parts)
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def ace_get_stats() -> str:
    """Get playbook statistics — bullet counts, section breakdown, health metrics.

    Returns stats for both global and project playbooks. Includes evolution
    trigger info (SkillRL §3.3) and temporal decay stats.
    """
    gpb = _ensure_global_playbook()
    ppb = _ensure_playbook()
    return json.dumps({
        "global": asdict(gpb.get_stats()),
        "project": asdict(ppb.get_stats()),
    }, indent=2)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def ace_find_similar(threshold: float = 0.6, scope: str = "project") -> str:
    """Find similar bullet pairs that may be candidates for merging.

    Uses Jaccard + trigram similarity. Returns pairs above the threshold
    so you can decide whether to MERGE them.

    Args:
        threshold: Similarity threshold (0.0-1.0, default 0.6).
        scope: "project" (default), "global", or "cross" (find duplicates between tiers).
    """
    if scope == "cross":
        gpb = _ensure_global_playbook()
        ppb = _ensure_playbook()
        # Cross-tier: compare every global bullet against every project bullet
        pairs = []
        for gb in gpb.bullets:
            words_a = set(gb.content.lower().split())
            trigrams_a = Playbook._char_trigrams(gb.content.lower())
            if len(words_a) < 2:
                continue
            for pb_bullet in ppb.bullets:
                words_b = set(pb_bullet.content.lower().split())
                trigrams_b = Playbook._char_trigrams(pb_bullet.content.lower())
                if len(words_b) < 2:
                    continue
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
            return "No cross-tier similar bullet pairs found."
        lines = ["Cross-tier similarities (global vs project):"]
        for a, b, sim in pairs[:10]:
            lines.append(f"[{a.id}] (global) vs [{b.id}] (project) (similarity={sim:.2f})")
            lines.append(f"  G: {a.content[:100]}")
            lines.append(f"  P: {b.content[:100]}")
        return "\n".join(lines)

    pb, _ = _resolve_playbook(scope)
    pairs = pb.find_similar_pairs(threshold)
    if not pairs:
        return f"No similar bullet pairs found in {scope} playbook."
    lines = []
    for a, b, sim in pairs[:10]:
        lines.append(f"[{a.id}] vs [{b.id}] (similarity={sim:.2f})")
        lines.append(f"  A: {a.content[:100]}")
        lines.append(f"  B: {b.content[:100]}")
    return "\n".join(lines)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False))
def ace_prune(scope: str = "project") -> str:
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
        with _state_lock:
            pb, save_fn = _resolve_playbook(scope)
            # Evolve failure lessons BEFORE pruning to prevent permanent loss
            evolved = pb.evolve_from_failures(threshold=1)
            pruned = pb.prune_problematic(min_harmful=3)
            budget_pruned = pb.enforce_token_budget(max_chars=80000)
            save_fn()
        total = len(pruned) + len(budget_pruned)
        parts = []
        if evolved:
            parts.append(f"Evolved {len(evolved)} new skill(s) from failure lessons.")
        parts.append(f"Pruned {total} bullet(s) from {scope} playbook. Now has {len(pb.bullets)} bullets.")
        return " ".join(parts)
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True))
def ace_evolve_from_failures(
    threshold: int = 3,
    scope: str = "project",
    synthesized_skills: list[dict] | None = None,
) -> str:
    """Evolve new skills from accumulated failure lessons (SkillRL §3.3 / Prompt B.1).

    Two-call pattern (caller-driven synthesis):
      1. Call with no synthesized_skills → returns failure lessons + prompt for
         Claude Code to synthesize new skills (teacher-model role per SkillRL §4.2).
      2. Call again with synthesized_skills=[{"content": "...", "when_to_apply": "..."}]
         → saves the synthesized skills as new playbook bullets.

    If synthesized_skills is not provided and the threshold is met, the tool
    falls back to mechanical evolution (prevention_principle extraction) for
    backward compatibility. The caller-driven path is preferred for higher
    quality synthesis per SkillRL's teacher-model pattern.

    Args:
        threshold: Minimum harmful-with-lessons bullets to trigger (default 3).
        scope: "project" (default) or "global".
        synthesized_skills: Optional list of dicts with "content" and "when_to_apply"
            keys, provided by Claude Code after reviewing the failure lessons prompt.
    """
    try:
        with _state_lock:
            pb, save_fn = _resolve_playbook(scope)

            # Path 2: Save synthesized skills from Claude Code
            if synthesized_skills:
                from ccr.ace.playbook import DeltaOperation
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
                return f"Saved {applied} synthesized skill(s) to {scope} playbook."

            # Path 1: Check if evolution is needed
            check = pb.check_evolution_needed(threshold)
            if not check.get("needed"):
                return (
                    f"Evolution not triggered in {scope} playbook. Need {threshold} harmful bullets with failure "
                    f"lessons, currently have {check['candidate_count']}."
                )

            # Run mechanical evolution (backward compat + fallback)
            new_bullets = pb.evolve_from_failures(threshold)
            if new_bullets:
                save_fn()

            # Build synthesis prompt for Claude Code (teacher-model per SkillRL §4.2)
            # Include the candidates so Claude can optionally synthesize better skills
            candidate_ids = set(check.get("candidate_ids", []))
            candidates = [b for b in pb.bullets if b.id in candidate_ids]
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

            if lessons_text:
                result_parts.append(
                    f"\n---\n# Optional: Caller-driven synthesis (SkillRL §4.2)\n"
                    f"For higher-quality skill synthesis, review these failure lessons and call\n"
                    f"ace_evolve_from_failures again with synthesized_skills=[\n"
                    f'  {{"content": "skill text", "when_to_apply": "condition", "scope": "general"}}\n'
                    f"]\n\n" + "\n".join(lessons_text)
                )

            return "\n".join(result_parts) if result_parts else "No skills evolved."
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ===========================================================================
# RLM Sandbox Tools
# ===========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def rlm_init(task_prompt: str) -> str:
    """Initialize a sandboxed Python REPL for structured problem-solving.

    Sets up a REPL with these variables and tools pre-loaded:
        - task_prompt: Your problem statement (string)
        - context: Repo index metadata (dict with file paths, symbols, imports)
        - get_file(path): Fetch full content of any indexed file
        - search_repo(query): Search files by content/symbol/path
        - estimate_tokens(text): Estimate token count
        - FINAL_VAR(name): Signal completion and return a variable's value
        - SHOW_VARS(): List all user-created variables

    Use this for complex analysis that benefits from iterative exploration.

    Args:
        task_prompt: The problem or question to solve.
    """
    try:
        global _repl
        with _state_lock:
            idx = _ensure_index()

            # Clean up previous REPL if any
            if _repl is not None:
                _repl.cleanup()

            # Kernel sandbox runs code in a subprocess, which means in-process tools
            # (search_repo, get_file, etc.) are NOT available there. Use Python-level
            # sandboxing (AST validation + restricted builtins) for the RLM REPL,
            # which needs these tools. The kernel sandbox is available for standalone
            # code execution via KernelSandbox directly.
            _repl = CCRRepl(repo_index=idx, project_root=_project_root, use_kernel_sandbox=False)
            _repl.locals["task_prompt"] = task_prompt

        # Load playbook as variable if available
        pb = _ensure_playbook()
        pb_text = pb.serialize()
        if pb_text.strip():
            _repl.locals["playbook"] = pb_text

        prompt_preview = task_prompt[:120] + "..." if len(task_prompt) > 120 else task_prompt
        file_count = len(idx.files) if idx.files else 0

        return (
            f"REPL initialized.\n"
            f"- task_prompt: {len(task_prompt)} chars — \"{prompt_preview}\"\n"
            f"- context: repo metadata ({file_count} files indexed)\n"
            f"- Tools: get_file(), search_repo(), estimate_tokens(), FINAL_VAR(), SHOW_VARS()\n"
            f"\nUse rlm_execute to run code. Use rlm_finalize when done."
        )
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def rlm_execute(code: str) -> str:
    """Execute Python code in the sandboxed REPL.

    The REPL persists variables across calls. Use it to:
        - Explore the repo: search_repo("pattern"), get_file("path")
        - Process data: parse, filter, transform results
        - Build answers incrementally across multiple execute calls

    Stdout is captured and returned. Variables persist between calls.

    Args:
        code: Python code to execute.
    """
    try:
        with _state_lock:
            if _repl is None:
                return "Error: REPL not initialized. Call rlm_init first."
            repl_ref = _repl

        result = repl_ref.execute_code(code)

        parts = []
        if result.stdout.strip():
            # Metadata-only: show length + preview (RLM paper pattern)
            stdout = result.stdout.strip()
            if len(stdout) > 500:
                parts.append(f"stdout ({len(stdout)} chars): {stdout[:500]}...")
            else:
                parts.append(f"stdout: {stdout}")

        if result.error:
            parts.append(f"error: {result.error}")

        if result.final_answer is not None:
            parts.append(f"FINAL_ANSWER: {result.final_answer}")

        if result.locals_snapshot:
            var_summary = ", ".join(
                f"{k}: {v[:60]}" for k, v in list(result.locals_snapshot.items())[:10]
            )
            parts.append(f"vars: {var_summary}")

        parts.append(f"time: {result.execution_time:.3f}s")

        return "\n".join(parts)
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False))
def rlm_finalize(variable_name: str) -> str:
    """Finalize the REPL session and return a variable's value as the result.

    Calls FINAL_VAR internally to extract and serialize the named variable.
    Cleans up the REPL after extraction.

    Args:
        variable_name: Name of the variable to return.
    """
    try:
        global _repl
        with _state_lock:
            if _repl is None:
                return "Error: REPL not initialized. Call rlm_init first."
            repl_ref = _repl

        # H1: Validate variable_name to prevent code injection
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', variable_name):
            return f"Error: Invalid variable name '{variable_name}'. Must be a valid Python identifier."

        # Call _final_var directly instead of constructing code string (H1)
        answer = repl_ref._final_var(variable_name)
        with _state_lock:
            repl_ref.cleanup()
            _repl = None  # M4: Reset _repl after cleanup

        if answer is not None and not answer.startswith("Error:"):
            return answer
        return f"Error: Variable '{variable_name}' not found."
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ===========================================================================
# Repo Index Tools
# ===========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True))
def index_build() -> str:
    """Build or rebuild the repo index.

    Scans the project directory for source files, extracts symbols (classes,
    functions) and imports per file. The index enables search_repo and
    get_file in the RLM sandbox.
    """
    try:
        global _repo_index
        with _state_lock:  # H2: protect global state mutation
            _repo_index = RepoIndex.build(_project_root)

            # Cache
            mem = _ensure_memory()
            try:
                mem.save_index(_repo_index.to_json())
                mem.update_metadata_file_tree([f for f in _repo_index.files.keys()])
            except Exception:
                pass

        return _repo_index.get_summary()
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def index_search(query: str, top_k: int = 10) -> str:
    """Search the repo index for files matching a query.

    Searches file paths, symbol names, and content. Returns ranked results
    with path, language, symbols, and relevance score.

    Args:
        query: Search term (file name, symbol, or content keyword).
        top_k: Maximum results to return (default 10).
    """
    top_k = max(1, min(top_k, 100))  # M3: bound top_k to prevent excessive results
    idx = _ensure_index()
    results = idx.search(query)[:top_k]
    if not results:
        return f"No files matching '{query}'."
    lines = []
    for r in results:
        syms = ", ".join(r["symbols"][:5]) if r["symbols"] else ""
        lines.append(
            f"[{r['score']}] {r['path']} ({r['language']}, {r['lines']} lines)"
            + (f" — {syms}" if syms else "")
        )
    return "\n".join(lines)


# ===========================================================================
# Entry point
# ===========================================================================


def main():
    """Run the MCP server with stdio transport."""
    import argparse

    parser = argparse.ArgumentParser(description="CCR MCP Server")
    parser.add_argument(
        "--project", "-p",
        default=os.getcwd(),
        help="Project root directory (default: cwd)",
    )
    args = parser.parse_args()

    _init(args.project)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
