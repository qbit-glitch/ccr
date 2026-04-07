"""GCC Memory Tools — core memory ops and re-exports.

Core tools: gcc_commit, gcc_context, gcc_status, gcc_consolidate, gcc_scratchpad.
Utility functions (no @mcp.tool decorator): gcc_clusters, gcc_triples, gcc_log_ota.

Re-exports branch and search tools so that all gcc_* names remain importable
from ccr.mcp.gcc_tools (backward compat for tests and ccr.mcp_server shim).
"""

from __future__ import annotations

import os
import re
import time

from mcp.types import ToolAnnotations
from mcp.server.fastmcp.exceptions import ToolError

from ccr.mcp.server import mcp
import ccr.mcp.server as _srv

# All server functions/globals accessed via _srv to support test patching.
# Only `mcp` (the FastMCP singleton) is imported directly — it never changes.
from ccr.mcp_types import (
    GccClustersResult,
    GccCommitResult,
    GccConsolidateResult,
    GccContextResult,
    GccLogOtaResult,
    GccPatternsResult,
    GccScratchpadResult,
    GccStatusResult,
    GccTriplesResult,
)


# ---------------------------------------------------------------------------
# F7: Auto-Trigger Extraction — ERL (arXiv:2603.24639)
# ---------------------------------------------------------------------------

_TRIGGER_PATTERNS = [
    re.compile(r'(?:when|if)\s+(.+?),\s*(.+)', re.I),
    re.compile(r'(?:before|after)\s+(.+?),\s*(.+)', re.I),
    re.compile(r'(.+?)\s*→\s*(.+)'),  # "X → Y" arrow notation
]


def _extract_trigger_suggestions(patterns: list[str]) -> list[dict]:
    """Extract ERL-style trigger/action pairs from pattern strings."""
    suggestions: list[dict] = []
    for p in patterns:
        for rx in _TRIGGER_PATTERNS:
            m = rx.search(p)
            if m:
                suggestions.append({
                    "trigger": m.group(1).strip(),
                    "action": m.group(2).strip(),
                })
                break
    return suggestions


# ===========================================================================
# GCC Core Memory Tools
# ===========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def gcc_commit(
    title: str,
    what: str,
    why: str,
    files_changed: list[str],
    next_step: str,
    patterns_learned: list[str] | None = None,
    admission_threshold: float = 0.85,
    rejection_threshold: float = 0.0,
    compressed_summary: str | None = None,
    author: str = "",
    ci_context: dict | None = None,
    experiment: dict | None = None,
) -> GccCommitResult:
    """Commit progress to project memory.

    Creates a structured commit record tracking what was done, why, which
    files changed, and what to do next. Similar commits are auto-deduplicated.

    Use after meaningful progress — completing a feature, fixing a bug,
    reaching a milestone, or before context gets large.

    When the rolling summary gets too long, the return message includes a
    compression prompt. Compress and pass it back via compressed_summary.

    Args:
        patterns_learned: Transferable lessons from this task, e.g.,
            "When adding {feature_type}, update tests + docs together".
            Deduped automatically. Recurring patterns (3+ commits) are
            suggested for ACE playbook promotion.
        admission_threshold: Similarity threshold for dedup (0-1, default 0.85).
            Set to 1.0 to disable dedup entirely.
        rejection_threshold: Score below which commits are rejected
            (default 0.0 = disabled).
        compressed_summary: Compressed rolling summary. Provide this when
            prompted — compress into a concise paragraph of project context,
            milestones, and direction. Max 1500 chars.
        author: Optional author name/identifier stored in the commit block.
        ci_context: Optional CI metadata dict (e.g., {"run_id": "123"}).
            Stored as JSON in the commit block.
        experiment: Optional research experiment data. Expected keys: id (str),
            hypothesis (str), metrics (dict[str, float|int|str]), conclusion (str).
            Stored as **Experiment** section in the commit. Use for ML runs,
            ablations, and any quantitative results. Searchable later via
            gcc_context(level=5, search_term="<metric_name>"). Example:
            {"id": "exp-042", "hypothesis": "LoRA r=16 matches full FT",
             "metrics": {"val_loss": 0.23, "accuracy": 0.87},
             "conclusion": "Confirmed — 98% perf at 12% params"}
    """
    # Import ace_tools module (not function) to allow test patching via setattr
    import ccr.mcp.ace_tools as _ace_mod

    try:
        # --- Input validation ---
        if not title.strip():
            raise ValueError("gcc_commit: 'title' must not be empty or whitespace-only.")
        if '\n' in title or '\r' in title:
            raise ValueError("gcc_commit: 'title' cannot contain newlines (breaks commit header parsing).")
        _TITLE_MAX = 200
        if len(title) > _TITLE_MAX:
            title = title[:_TITLE_MAX]

        _FIELD_MAX = 2000
        _warnings: list[str] = []
        if len(what) > _FIELD_MAX:
            what = what[:_FIELD_MAX]
            _warnings.append(f"'what' truncated to {_FIELD_MAX} chars.")
        if len(why) > _FIELD_MAX:
            why = why[:_FIELD_MAX]
            _warnings.append(f"'why' truncated to {_FIELD_MAX} chars.")
        if next_step and len(next_step) > _FIELD_MAX:
            next_step = next_step[:_FIELD_MAX]
            _warnings.append(f"'next_step' truncated to {_FIELD_MAX} chars.")

        if patterns_learned:
            _seen: set[str] = set()
            _deduped: list[str] = []
            for p in patterns_learned:
                _k = p.strip().lower()
                if _k and _k not in _seen:
                    _seen.add(_k)
                    _deduped.append(p.strip())
            patterns_learned = _deduped if _deduped else None

        # Warn on files that don't exist on disk (don't block — may have been deleted intentionally)
        _proj_root = getattr(_srv, "_project_root", "") or ""
        _missing_files: list[str] = []
        if _proj_root and files_changed:
            for _f in files_changed:
                _abs = _f if os.path.isabs(_f) else os.path.join(_proj_root, _f)
                if not os.path.exists(_abs):
                    _missing_files.append(_f)
        if _missing_files:
            _warnings.append(
                f"{len(_missing_files)} file(s) not found on disk: "
                f"{', '.join(_missing_files[:3])}"
                f"{'...' if len(_missing_files) > 3 else ''} "
                f"(check paths are relative to project root)"
            )
        # --- End validation ---

        with _srv._state_lock:
            mem = _srv._ensure_memory()
            # Phase 2: Auto-extract patterns via sub-model when not provided (CER §3.2)
            # Retry up to 2 times on transient failure (A7 sub-model retry)
            if patterns_learned is None and mem.config.auto_extract_patterns and len(what) > 100:
                sub = _srv._get_sub_client()
                if sub is not None:
                    _extracted = None
                    for _attempt in range(2):
                        try:
                            _extracted = _srv._extract_patterns_from_commit(
                                title, what, why, files_changed or [], sub
                            )
                            break
                        except Exception:
                            if _attempt == 0:
                                time.sleep(1)
                    patterns_learned = _extracted or None
            result = mem.commit(title, what, why, files_changed, next_step,
                                patterns_learned,
                                admission_threshold, rejection_threshold,
                                compressed_summary,
                                author=author,
                                ci_context=ci_context,
                                experiment=experiment)

        # Write explicit-commit marker so on_stop.py skips auto-baseline for this session
        if result and not result.startswith("Error") and not result.startswith("[REJECTED]"):
            try:
                if _srv._project_root:
                    _marker = os.path.join(_srv._project_root, ".ccr", ".session_explicit_commit")
                    with open(_marker, "w", encoding="utf-8") as _mf:
                        _mf.write("1")
            except Exception:
                pass  # Non-fatal — worst case: on_stop creates a harmless duplicate baseline

        # ACE §3.1-3.3: Generator → Reflector → Curator pipeline (fire-and-forget)
        sub = _srv._get_sub_client()
        if sub is not None and result and not result.startswith("Error"):
            _ace_mod._run_ace_pipeline(title, what, why, sub)

        # Memori-inspired: extract semantic triples from commit text
        triple_store = _srv._triple_store
        if triple_store and result and not result.startswith("Error") and not result.startswith("[REJECTED]"):
            try:
                # Extract commit ID from result string (e.g. "[C021] ..." or "[C021+] ...")
                cid_match = re.match(r"\[(C\d{3,})\+?\]", result)
                if cid_match:
                    triple_store.extract_from_commit(
                        commit_id=cid_match.group(1),
                        title=title,
                        what=what,
                        why=why,
                        files=files_changed,
                    )
            except Exception:
                pass  # Triple extraction is non-critical

        # Determine admission decision and commit_id from the result string
        if result.startswith("[REJECTED]"):
            admission_decision = "rejected"
            cid = ""
        elif "merged" in result.lower() and "+]" in result:
            admission_decision = "merged"
            m = re.search(r"\[(\w+)\+?\]", result)
            cid = m.group(1) if m else ""
        else:
            admission_decision = "created"
            m = re.search(r"\[(\w+)\]", result)
            cid = m.group(1) if m else ""

        # Append any field-truncation warnings to the result message
        if _warnings:
            result += "\n\n[Warnings: " + " ".join(_warnings) + "]"

        branch = mem.get_active_branch()
        commit_result = GccCommitResult(
            commit_id=cid,
            branch=branch,
            title=title,
            admission_decision=admission_decision,
            message=result,
        )
        # F7: extract ERL trigger/action suggestions from patterns
        if patterns_learned:
            trigger_suggestions = _extract_trigger_suggestions(patterns_learned)
            if trigger_suggestions:
                commit_result["trigger_suggestions"] = trigger_suggestions
        return commit_result
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_context(
    level: int = 2,
    search_term: str | None = None,
    commit_id: str | None = None,
    log_window: int = 0,
    follow_links: bool = False,
    include_summaries: bool = False,
    summaries_tier: str = "all",
    summaries_count: int = 5,
    max_tokens: int | None = None,
    result_limit: int = 20,
    time_range_hours: int | None = None,
) -> GccContextResult:
    """Retrieve project memory at the specified depth.

    Levels:
        1: Project overview only (~200 tokens).
        2: + rolling summary + last 3 commits. Use at session start.
        3: + branch summary + thematic clusters + triples.
        4: + last 10 commits. For finding related work.
        5: + specific commit search + cross-links. For tracing history.

    Args:
        level: Depth of context retrieval (1-5).
        search_term: Keyword to search commits (level 5 only).
        commit_id: Specific commit ID to retrieve (level 5 only).
        log_window: Number of recent OTA log entries to include.
        follow_links: If True and level >= 5, include linked commit summaries (1-hop BFS).
        include_summaries: If True, append hierarchical summaries (replaces gcc_summaries).
        summaries_tier: Which summary tier — "session", "phase", "project", or "all".
        summaries_count: Max summaries per tier (default 5).
        max_tokens: Optional upper bound on output size (rough 4 chars/token estimate).
            Use to prevent overly large context at higher levels.
        result_limit: Maximum number of commit blocks to include (default 20).
        time_range_hours: If set, only include commit blocks from the last N hours.
    """
    from datetime import datetime, timezone, timedelta

    level = max(1, min(level, 5))
    log_window = max(0, min(log_window, 50))
    result_limit = max(1, result_limit)
    _level_warnings: list[str] = []
    if search_term and level < 5:
        _level_warnings.append(f"search_term is only active at level=5 (current level={level}); use level=5 to search commits.")
    if commit_id and level < 5:
        _level_warnings.append(f"commit_id is only active at level=5 (current level={level}); use level=5 for specific commit lookup.")
    with _srv._state_lock:
        mem = _srv._ensure_memory()
    result = mem.get_context(
        level=level,
        search_term=search_term,
        commit_id=commit_id,
        log_window=log_window,
        follow_links=follow_links,
    )

    # Apply result_limit and time_range_hours filtering on commit blocks
    # Commit blocks start with "\n## [C" or "## [C" at the beginning
    if "\n## [C" in result or result.startswith("## [C"):
        # Split on commit block boundaries, preserving delimiter
        _parts = result.split("\n## [C")
        _pre = _parts[0]  # Content before the first commit block
        _blocks = ["\n## [C" + p for p in _parts[1:]]

        # Apply time_range_hours filter
        if time_range_hours is not None and time_range_hours > 0:
            _cutoff = datetime.now(timezone.utc) - timedelta(hours=time_range_hours)
            _filtered = []
            for _blk in _blocks:
                # Header: ## [C001] 2026-03-10 22:25 | ...
                _ts_match = re.search(r"## \[C\d+\]\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})", _blk)
                if _ts_match:
                    try:
                        _ts = datetime.strptime(_ts_match.group(1), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                        if _ts >= _cutoff:
                            _filtered.append(_blk)
                    except ValueError:
                        _filtered.append(_blk)  # Keep if unparseable
                else:
                    _filtered.append(_blk)
            _blocks = _filtered

        # Apply result_limit
        _blocks = _blocks[:result_limit]

        result = _pre + "".join(_blocks)

    # Append thematic clusters (level 3+)
    if level >= 3:
        clusters_text = mem.format_clusters_for_context()
        if clusters_text:
            result += "\n\n" + clusters_text

    # Append knowledge graph triples (level 3+, Memori-inspired)
    triple_store = _srv._triple_store
    if level >= 3 and triple_store and triple_store.size > 0:
        triples_text = triple_store.format_for_context(top_k=8)
        if triples_text:
            result += "\n\n" + triples_text

    # Append working memory if scratchpad has entries (level 2+)
    scratchpad = _srv._scratchpad
    if level >= 2 and scratchpad and scratchpad.size > 0:
        scratchpad_text = scratchpad.format_for_context()
        if scratchpad_text:
            result += "\n\n" + scratchpad_text

    # Append hierarchical summaries if requested (replaces standalone gcc_summaries)
    if include_summaries:
        sc = max(1, min(summaries_count, 50))
        summaries_text = mem.get_summaries(tier=summaries_tier, count=sc)
        if summaries_text.strip():
            result += "\n\n" + summaries_text

    # Apply token budget AFTER all sections appended (prevents silent overflow from clusters/summaries)
    if max_tokens is not None and max_tokens > 0:
        budget_chars = max_tokens * 4  # ~4 chars/token heuristic (matches estimate_tokens)
        if len(result) > budget_chars:
            # Markdown-aware cut: prefer cutting at a section boundary
            cut = result.rfind("\n## ", 0, budget_chars)
            if cut == -1:
                cut = budget_chars
            result = result[:cut] + f"\n\n[Context truncated to ~{max_tokens} tokens. Use level=2 or sections filter for targeted retrieval.]"

    if _level_warnings:
        result = "\n".join(f"[Warning: {w}]" for w in _level_warnings) + "\n\n" + result

    branch = mem.get_active_branch()

    # B2: Context level heuristic — suggest richer retrieval for large projects
    if level == 2:
        try:
            total = len(mem._build_commit_index(branch))
            if total >= 30:
                result += (
                    f"\n\n[Hint: This project has {total} commits. "
                    f"Try gcc_context(level=3) for richer history or "
                    f"gcc_context(level=5, search_term='<topic>') for targeted search.]"
                )
        except Exception:
            pass

    return GccContextResult(level=level, branch=branch, message=result)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_status() -> GccStatusResult:
    """Show current project memory status.

    Returns the active branch, recent milestones, open branches,
    and metadata summary. Warns if any active branch is more than 30 days old.
    """
    from datetime import datetime, timezone

    with _srv._state_lock:
        mem = _srv._ensure_memory()

    parts = []
    branch = mem.get_active_branch()
    parts.append(f"Active branch: {branch}")

    # Level-1 context (overview)
    overview = mem.get_context(level=1)
    parts.append(overview)

    # Count total commits via cached index (O(1) after first build)
    total_commits = len(mem._build_commit_index(branch))

    text = "\n\n".join(parts)

    # Stale branch detection — warn if any active branch is > 30 days old
    try:
        meta = mem._load_metadata()
        now = datetime.now(timezone.utc)
        _stale: list[str] = []
        for b in meta.get("branches", []):
            if b.get("status") != "active":
                continue
            created_str = b.get("created", "")
            if not created_str:
                continue
            try:
                created_dt = datetime.strptime(str(created_str), "%Y-%m-%d").replace(tzinfo=timezone.utc)
                age_days = (now - created_dt).days
                if age_days > 30:
                    _stale.append(f"  - {b['name']} ({age_days} days old, created {created_str})")
            except ValueError:
                pass
        if _stale:
            text += "\n\n## ⚠ Stale Branches (> 30 days)\n" + "\n".join(_stale)
            text += "\n\nConsider merging or closing these branches."
    except Exception:
        pass  # Stale branch detection is supplementary — never fail status

    return GccStatusResult(branch=branch, total_commits=total_commits, message=text)


# ===========================================================================
# GCC Hierarchical Summary Tools (TiMem-inspired)
# ===========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def gcc_consolidate(tier: str = "session", content: str | None = None) -> GccConsolidateResult:
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
    _VALID_TIERS = {"session", "phase", "project"}
    if tier not in _VALID_TIERS:
        raise ToolError(f"tier must be one of {sorted(_VALID_TIERS)}, got: {tier!r}")

    try:
        with _srv._state_lock:
            mem = _srv._ensure_memory()
            if tier == "project" and content is not None:
                _OVERVIEW_MAX = 5000
                if len(content) > _OVERVIEW_MAX:
                    content = content[:_OVERVIEW_MAX] + "\n\n[Truncated to 5000 chars by system]"
                mem.save_overview(content)
                return GccConsolidateResult(tier=tier, message="Project overview saved.")
            text = mem.get_consolidation_prompt(tier=tier)
        return GccConsolidateResult(tier=tier, message=text)
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e


# ===========================================================================
# Working Memory (Scratchpad) Tools — AgeMem-inspired
# ===========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def gcc_scratchpad(
    mode: str = "get",
    key: str | None = None,
    value: str | None = None,
    ttl_seconds: int | None = None,
) -> GccScratchpadResult:
    """Working memory for temporary reasoning state (ephemeral, within session).

    Modes:
        "get": Retrieve a key (or list all if key is None).
        "set": Store a key-value pair. Requires both key and value.
        "clear": Delete a key (or clear all if key is None).

    Working memory entries appear in gcc_context at level 2+.
    Use for hypotheses, debug focus, or intermediate results that
    don't warrant a permanent gcc_commit.

    Args:
        mode: Operation — "get", "set", or "clear".
        key: The key to operate on. None means all entries (for get/clear).
        value: The value to store (required for "set" mode).
        ttl_seconds: Optional time-to-live in seconds for "set" mode. After
            expiry, get() returns None. Useful for session-scoped scratch data.
    """
    _srv._ensure_memory()  # ensures _init() has run
    scratchpad = _srv._scratchpad
    if not scratchpad:
        raise ToolError("Scratchpad not initialized")

    if mode == "set":
        if not key:
            raise ToolError("key is required for mode='set'")
        if value is None:
            raise ToolError("value is required for mode='set'")
        entry = scratchpad.set(key, value, ttl_seconds=ttl_seconds)
        return GccScratchpadResult(mode="set", key=key, message=f"Set '{key}' in working memory (updated: {entry.updated_at})")

    elif mode == "get":
        if key:
            entry = scratchpad.get(key)
            if entry is None:
                return GccScratchpadResult(mode="get", key=key, message=f"Key '{key}' not found in working memory")
            text = f"**{entry.key}**: {entry.value}\n(accessed {entry.access_count}x, updated: {entry.updated_at})"
            return GccScratchpadResult(mode="get", key=key, message=text)
        else:
            entries = scratchpad.list_entries()
            if not entries:
                return GccScratchpadResult(mode="get", message="Working memory is empty")
            lines = [f"Working memory ({len(entries)} entries):"]
            for e in entries:
                lines.append(f"- **{e.key}**: {e.value} (accessed {e.access_count}x)")
            return GccScratchpadResult(mode="get", message="\n".join(lines))

    elif mode == "clear":
        if key:
            deleted = scratchpad.delete(key)
            if deleted:
                return GccScratchpadResult(mode="clear", key=key, cleared=1, message=f"Deleted '{key}' from working memory")
            return GccScratchpadResult(mode="clear", key=key, cleared=0, message=f"Key '{key}' not found in working memory")
        else:
            count = scratchpad.clear()
            return GccScratchpadResult(mode="clear", cleared=count, message=f"Cleared {count} entries from working memory")

    else:
        raise ToolError(f"Invalid mode '{mode}'. Use 'get', 'set', or 'clear'.")


# ===========================================================================
# Utility functions (no @mcp.tool decorator — called internally / via tests)
# ===========================================================================


def gcc_clusters(min_size: int = 2, recompute: bool = True) -> GccClustersResult:
    """Compute or retrieve thematic commit clusters.

    Clusters related commits using connected components over the
    cross-link graph (entity + semantic links). Inspired by
    EverMemOS (arXiv:2601.02163) MemScene clustering.

    Args:
        min_size: Minimum commits per cluster.
        recompute: If True, recompute clusters from current links.
    """
    with _srv._state_lock:
        mem = _srv._ensure_memory()

    if recompute:
        clusters = mem.compute_clusters(min_cluster_size=min_size)
    else:
        data = mem._load_clusters()
        clusters = data.get("clusters", [])

    if not clusters:
        return GccClustersResult(cluster_count=0, message="No clusters found (need commits with cross-links).")

    lines = [f"Found {len(clusters)} cluster(s):"]
    for cl in clusters:
        lines.append(f"\n## {cl['name']} [{cl['id']}]")
        lines.append(f"Commits: {', '.join(cl['commit_ids'])}")
        if cl.get("top_keywords"):
            lines.append(f"Keywords: {', '.join(cl['top_keywords'])}")
    text = "\n".join(lines)
    return GccClustersResult(cluster_count=len(clusters), message=text)


def gcc_triples(
    query: str = "",
    commit_id: str = "",
    entity: str = "",
    top_k: int = 10,
) -> GccTriplesResult:
    """Search the semantic knowledge graph for entity relationships.

    Triples are automatically extracted from commits using regex patterns
    (Memori-inspired, arXiv:2603.19935). Zero LLM calls.
    Format: subject --predicate--> object (source_commit)

    Args:
        query: Free-text search (word Jaccard similarity).
        commit_id: Get all triples from a specific commit.
        entity: Get all triples involving an entity (subject or object).
        top_k: Maximum results to return.
    """
    top_k = max(1, min(top_k, 200))
    _srv._ensure_memory()  # ensures _init() has run, initializing _triple_store
    triple_store = _srv._triple_store
    if not triple_store:
        raise ToolError("Triple store not initialized")

    if commit_id:
        triples = triple_store.get_by_commit(commit_id)
    elif entity:
        triples = triple_store.get_by_entity(entity)
    elif query:
        triples = triple_store.search(query, top_k=top_k)
    else:
        # No filter — return recent
        with triple_store._lock:
            all_triples = sorted(
                triple_store._triples, key=lambda t: t.timestamp, reverse=True
            )
        triples = all_triples[:top_k]

    if not triples:
        return GccTriplesResult(count=0, message="No matching triples found.")

    lines = [f"Found {len(triples)} triple(s):"]
    for t in triples[:top_k]:
        lines.append(f"  {t.format_compact()}")
    text = "\n".join(lines)
    return GccTriplesResult(count=len(triples), message=text)


def gcc_log_ota(observation: str, thought: str, action: str) -> GccLogOtaResult:
    """Log an Observation-Thought-Action triple to the project log.

    OTA triples track your reasoning process. They're attached to
    commits and preserved across sessions. Each call appends a new entry.

    Args:
        observation: What you observed (e.g., "Test failure in router.py").
        thought: Your reasoning (e.g., "The regex pattern is too greedy").
        action: What you did (e.g., "Fixed pattern to use non-greedy match").
    """
    try:
        with _srv._state_lock:  # H3: protect memory state mutation
            mem = _srv._ensure_memory()
            mem.log_ota(
                tool_name="claude-code",
                observation=observation,
                thought=thought,
                action=action,
            )
        return GccLogOtaResult(message="OTA logged.")
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e


# ===========================================================================
# Re-exports from split submodules (test patchability + backward compat)
# ===========================================================================

from ccr.mcp.gcc_branch_tools import (  # noqa: E402,F401
    gcc_branch,
    gcc_merge,
    gcc_evolve_memory,
)
from ccr.mcp.gcc_search_tools import (  # noqa: E402,F401
    gcc_links,
    gcc_patterns,
    gcc_experiments,
    gcc_discuss,
    gcc_discussions,
    gcc_search,
)
