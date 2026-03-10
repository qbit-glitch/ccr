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
import sys
from dataclasses import asdict

from mcp.server.fastmcp import FastMCP

from ccr.ace.playbook import DeltaOperation, FailureLesson, Playbook, parse_delta_operations
from ccr.context.indexer import RepoIndex
from ccr.core.memory import MemoryManager
from ccr.core.types import CCRConfig
from ccr.rlm.repl import CCRRepl

# ---------------------------------------------------------------------------
# Globals — initialized once at startup
# ---------------------------------------------------------------------------

_project_root: str = ""
_memory: MemoryManager | None = None
_playbook: Playbook | None = None
_playbook_path: str = ""
_failure_lessons_path: str = ""
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
    global _project_root, _memory, _playbook, _playbook_path, _failure_lessons_path, _repo_index

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

    # Build repo index
    _repo_index = RepoIndex.build(_project_root)

    # Cache index
    try:
        _memory.save_index(_repo_index.to_json())
    except Exception:
        pass


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
    """Persist playbook and failure lessons to disk."""
    if _playbook is not None:
        os.makedirs(os.path.dirname(_playbook_path), exist_ok=True)
        with open(_playbook_path, "w", encoding="utf-8") as f:
            f.write(_playbook.serialize())
        # Save failure lessons to companion JSON
        if _failure_lessons_path:
            _playbook.save_failure_lessons(_failure_lessons_path)


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


@mcp.tool()
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
    mem = _ensure_memory()
    return mem.commit(title, what, why, files_changed, next_step,
                      admission_threshold, rejection_threshold)


@mcp.tool()
def gcc_branch(name: str, purpose: str, hypothesis: str) -> str:
    """Create an exploration branch for experimental work.

    Branches isolate experimental changes from the main line. Must be on
    main to create a branch. Use kebab-case names (e.g., 'try-new-parser').

    Args:
        name: Branch name in kebab-case.
        purpose: What this branch explores.
        hypothesis: What you expect to learn or achieve.
    """
    mem = _ensure_memory()
    return mem.create_branch(name, purpose, hypothesis)


@mcp.tool()
def gcc_merge(branch: str, outcome: str, conclusion: str) -> str:
    """Merge an exploration branch back into main.

    Must be on the branch being merged. Integrates the branch's rolling
    summary and OTA log into main.

    Args:
        branch: Branch name to merge.
        outcome: One of 'success', 'failure', or 'partial'.
        conclusion: Summary of what was learned.
    """
    mem = _ensure_memory()
    return mem.merge(branch, outcome, conclusion)


@mcp.tool()
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
    mem = _ensure_memory()
    return mem.get_context(
        level=level,
        search_term=search_term,
        commit_id=commit_id,
        log_window=log_window,
    )


@mcp.tool()
def gcc_log_ota(observation: str, thought: str, action: str) -> str:
    """Log an Observation-Thought-Action triple to the project log.

    OTA triples track your reasoning process. They're attached to
    commits and preserved across sessions.

    Args:
        observation: What you observed (e.g., "Test failure in router.py").
        thought: Your reasoning (e.g., "The regex pattern is too greedy").
        action: What you did (e.g., "Fixed pattern to use non-greedy match").
    """
    mem = _ensure_memory()
    mem.log_ota(
        tool_name="claude-code",
        observation=observation,
        thought=thought,
        action=action,
    )
    return "OTA logged."


@mcp.tool()
def gcc_status() -> str:
    """Show current project memory status.

    Returns the active branch, recent milestones, open branches,
    and metadata summary.
    """
    mem = _ensure_memory()
    parts = []

    branch = mem.get_active_branch()
    parts.append(f"Active branch: {branch}")

    # Level-1 context (overview)
    overview = mem.get_context(level=1)
    parts.append(overview)

    return "\n\n".join(parts)


# ===========================================================================
# ACE Playbook Tools
# ===========================================================================


@mcp.tool()
def ace_get_playbook() -> str:
    """Get the current ACE playbook — your evolved strategies and insights.

    The playbook contains structured bullets organized by section, each with
    helpful/harmful counters tracking how useful they've been. Bullets tagged
    harmful may include structured failure lessons explaining WHY they failed
    and WHAT to do instead. Review this after loading context to learn from
    past successes and failures.
    """
    pb = _ensure_playbook()
    # Use failure-aware serialization if any lessons exist
    has_lessons = any(b.has_failure_lessons for b in pb.bullets)
    if has_lessons:
        text = pb.serialize_with_failures()
    else:
        text = pb.serialize()
    return text if text.strip() else "(Playbook is empty — no strategies recorded yet.)"


@mcp.tool()
def ace_apply_delta(operations: list[dict]) -> str:
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
    """
    pb = _ensure_playbook()
    ops = parse_delta_operations({"operations": operations})
    applied = pb.apply_delta(ops)
    _save_playbook()
    return f"Applied {applied} operation(s). Playbook now has {len(pb.bullets)} bullets."


@mcp.tool()
def ace_update_counters(bullet_tags: list[dict]) -> str:
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

    Example:
        [{"id": "str-00001", "tag": "harmful", "failure_lesson": {
            "failure_point": "Strategy assumed all inputs are UTF-8",
            "flawed_reasoning": "Did not consider binary file inputs",
            "counterfactual": "Should check encoding before processing",
            "prevention_principle": "Always validate input encoding at boundaries"
        }}]
    """
    pb = _ensure_playbook()
    updated = pb.update_bullet_counts(bullet_tags)
    _save_playbook()

    # Count how many had structured lessons
    lessons_added = sum(
        1 for t in bullet_tags
        if t.get("tag") == "harmful" and isinstance(t.get("failure_lesson"), dict) and t["failure_lesson"]
    )
    parts = [f"Updated {updated} bullet(s)."]
    if lessons_added:
        parts.append(f"Recorded {lessons_added} structured failure lesson(s).")
    return " ".join(parts)


@mcp.tool()
def ace_get_stats() -> str:
    """Get playbook statistics — bullet counts, section breakdown, health metrics.

    Includes evolution trigger info (SkillRL §3.3): when enough harmful bullets
    have accumulated failure lessons, evolution_needed will be True. Call
    ace_evolve_from_failures to generate new skills from those failures.
    """
    pb = _ensure_playbook()
    stats = pb.get_stats()
    return json.dumps(asdict(stats), indent=2)


@mcp.tool()
def ace_find_similar(threshold: float = 0.6) -> str:
    """Find similar bullet pairs that may be candidates for merging.

    Uses Jaccard + trigram similarity. Returns pairs above the threshold
    so you can decide whether to MERGE them.

    Args:
        threshold: Similarity threshold (0.0-1.0, default 0.6).
    """
    pb = _ensure_playbook()
    pairs = pb.find_similar_pairs(threshold)
    if not pairs:
        return "No similar bullet pairs found."
    lines = []
    for a, b, score in pairs[:10]:
        lines.append(f"[{a.id}] vs [{b.id}] (similarity={score:.2f})")
        lines.append(f"  A: {a.content[:100]}")
        lines.append(f"  B: {b.content[:100]}")
    return "\n".join(lines)


@mcp.tool()
def ace_prune() -> str:
    """Prune problematic bullets and enforce token budget.

    Removes bullets where harmful >= helpful and harmful >= 3.
    Also trims lowest-scoring bullets if playbook exceeds 80K chars.
    """
    pb = _ensure_playbook()
    pruned = pb.prune_problematic(min_harmful=3)
    budget_pruned = pb.enforce_token_budget(max_chars=80000)
    _save_playbook()
    total = len(pruned) + len(budget_pruned)
    return f"Pruned {total} bullet(s). Playbook now has {len(pb.bullets)} bullets."


@mcp.tool()
def ace_evolve_from_failures(threshold: int = 3) -> str:
    """Evolve new skills from accumulated failure lessons (SkillRL §3.3 / Prompt B.1).

    When harmful bullets accumulate enough structured failure lessons, this tool
    generates NEW standalone skill bullets from their prevention principles.

    Per SkillRL Prompt B.1: "Generate 1-3 NEW actionable skills... each must have:
    skill_id, title, principle, when_to_apply." Failed strategies produce NEW skills,
    not mere annotations on existing ones.

    New skills are added to PROBLEM-SOLVING HEURISTICS with scope="general" and
    when_to_apply derived from the failure's task_context.

    Args:
        threshold: Minimum number of harmful-with-lessons bullets needed to trigger
                   evolution (default 3). Lower for aggressive learning.
    """
    pb = _ensure_playbook()
    new_bullets = pb.evolve_from_failures(threshold)
    if not new_bullets:
        check = pb.check_evolution_needed(threshold)
        return (
            f"Evolution not triggered. Need {threshold} harmful bullets with failure "
            f"lessons, currently have {check['candidate_count']}."
        )
    _save_playbook()
    lines = [f"Evolved {len(new_bullets)} new skill(s) from failure lessons:"]
    for b in new_bullets:
        lines.append(f"  [{b.id}] {b.content[:100]}")
        if b.when_to_apply:
            lines.append(f"    When: {b.when_to_apply[:100]}")
    return "\n".join(lines)


# ===========================================================================
# RLM Sandbox Tools
# ===========================================================================


@mcp.tool()
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
    global _repl
    idx = _ensure_index()

    # Clean up previous REPL if any
    if _repl is not None:
        _repl.cleanup()

    _repl = CCRRepl(repo_index=idx)
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


@mcp.tool()
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
    if _repl is None:
        return "Error: REPL not initialized. Call rlm_init first."

    result = _repl.execute_code(code)

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


@mcp.tool()
def rlm_finalize(variable_name: str) -> str:
    """Finalize the REPL session and return a variable's value as the result.

    Calls FINAL_VAR internally to extract and serialize the named variable.
    Cleans up the REPL after extraction.

    Args:
        variable_name: Name of the variable to return.
    """
    if _repl is None:
        return "Error: REPL not initialized. Call rlm_init first."

    # Execute FINAL_VAR
    result = _repl.execute_code(f'FINAL_VAR("{variable_name}")')

    answer = result.final_answer
    _repl.cleanup()

    if answer is not None:
        return answer
    return f"Error: Variable '{variable_name}' not found. {result.error or ''}"


# ===========================================================================
# Repo Index Tools
# ===========================================================================


@mcp.tool()
def index_build() -> str:
    """Build or rebuild the repo index.

    Scans the project directory for source files, extracts symbols (classes,
    functions) and imports per file. The index enables search_repo and
    get_file in the RLM sandbox.
    """
    global _repo_index
    _repo_index = RepoIndex.build(_project_root)

    # Cache
    mem = _ensure_memory()
    try:
        mem.save_index(_repo_index.to_json())
        mem.update_metadata_file_tree([f for f in _repo_index.files.keys()])
    except Exception:
        pass

    return _repo_index.get_summary()


@mcp.tool()
def index_search(query: str, top_k: int = 10) -> str:
    """Search the repo index for files matching a query.

    Searches file paths, symbol names, and content. Returns ranked results
    with path, language, symbols, and relevance score.

    Args:
        query: Search term (file name, symbol, or content keyword).
        top_k: Maximum results to return (default 10).
    """
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
