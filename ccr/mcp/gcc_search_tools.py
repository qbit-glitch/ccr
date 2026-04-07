"""GCC Search Tools — search and query MCP tool functions.

Covers: gcc_search, gcc_experiments, gcc_patterns, gcc_links,
        gcc_discussions, gcc_discuss.
"""

from __future__ import annotations

import re

from mcp.types import ToolAnnotations
from mcp.server.fastmcp.exceptions import ToolError

from ccr.mcp.server import mcp
import ccr.mcp.server as _srv

from ccr.mcp_types import (
    GccDiscussResult,
    GccDiscussionsResult,
    GccExperimentsResult,
    GccLinksResult,
    GccPatternsResult,
    GccScratchpadSearchResult,
    GccSearchResult,
)


def _log_search_error(source: str, query: str, exc: Exception) -> None:
    """Write non-fatal gcc_search error to .ccr/.hook_errors.log (never raises)."""
    import traceback as _tb
    try:
        import ccr.mcp.server as _srv2
        proj = getattr(_srv2, "_project_root", None) or ""
        if proj:
            import os as _os2
            import datetime as _dt2
            log_path = _os2.path.join(proj, ".ccr", ".hook_errors.log")
            if _os2.path.isdir(_os2.path.dirname(log_path)):
                ts = _dt2.datetime.now(_dt2.timezone.utc).isoformat()
                with open(log_path, "a", encoding="utf-8") as _f:
                    _f.write(
                        f"\n--- {ts} [gcc_search:{source}] query={query!r} ---\n"
                        f"{_tb.format_exc()}\n"
                    )
    except Exception:
        pass


# ===========================================================================
# GCC Search & Query Tools
# ===========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_links(
    commit_id: str,
    link_types: str | None = None,
    max_hops: int = 1,
    query: str | None = None,
    adaptive: bool = True,
) -> GccLinksResult:
    """Retrieve cross-links for a commit.

    Shows which other commits are related via shared files (entity),
    explicit C### references (causal), replacement language (supersession),
    or keyword overlap (semantic). Useful for tracing the evolution of
    a feature or understanding why a decision was made.

    Link taxonomy inspired by A-MEM (bidirectional links) and MAGMA (typed
    edges), but uses mechanical heuristics — not the papers' LLM inference
    or dense vector embeddings. See CLAUDE.md for limitations.

    Args:
        commit_id: The commit to look up (e.g., "C012").
        link_types: Comma-separated link types to filter (default: all).
                    Options: entity, causal, supersession, semantic.
        max_hops: How many link hops to traverse (1 = direct, 2 = friends-of-friends).
        query: Optional natural language search intent. When provided, BFS traversal
               is weighted by query relevance (MAGMA intent-aware traversal, Alg. 1).
               Example: query="why was the auth system changed"
        adaptive: If True (default), prune low-relevance BFS candidates at each
                  hop using MAGMA Algorithm 1 beam pruning. Set False for exhaustive
                  traversal when you want all linked commits regardless of score.
    """
    _VALID_LINK_TYPES = {"entity", "causal", "supersession", "semantic"}
    max_hops = max(1, min(max_hops, 5))
    with _srv._state_lock:
        mem = _srv._ensure_memory()
    types = None
    if link_types:
        parsed = [t.strip().lower() for t in link_types.split(",") if t.strip()]
        invalid = [t for t in parsed if t not in _VALID_LINK_TYPES]
        if invalid:
            raise ToolError(f"Invalid link_type(s): {invalid!r}. Valid: {sorted(_VALID_LINK_TYPES)}")
        types = parsed if parsed else None
    linked = mem.get_linked_commits(commit_id, link_types=types, max_hops=max_hops, query=query, adaptive=adaptive)
    if not linked:
        # Still show direct link summary even if BFS returned nothing
        direct = mem.get_commit_links(commit_id)
        if any(v for v in direct.values()):
            text = mem._format_links_for_context(commit_id, direct)
            return GccLinksResult(commit_id=commit_id, links_found=0, message=text)
        text = f"No links found for {commit_id}."
        return GccLinksResult(commit_id=commit_id, links_found=0, message=text)
    # Group by link type for readability
    lines = [f"# Links for {commit_id} (max_hops={max_hops})"]
    by_type: dict[str, list[dict]] = {}
    for entry in linked:
        by_type.setdefault(entry["link_type"], []).append(entry)
    for lt in ("entity", "causal", "supersession", "semantic"):
        entries = by_type.get(lt, [])
        if not entries:
            continue
        lines.append(f"\n## {lt.capitalize()} Links")
        for e in entries:
            hop_tag = f" (hop {e['hop']})" if e.get("hop", 1) > 1 else ""
            title = e.get("title", "")
            what = e.get("what", "")[:120]
            detail = ""
            if lt == "entity" and e.get("shared_files"):
                detail = f" | shared: {', '.join(e['shared_files'])}"
            elif lt in ("causal", "supersession") and e.get("snippet"):
                detail = f' | "{e["snippet"]}"'
            score_tag = ""
            if "query_score" in e:
                score_tag = f" [q: {e['query_score']:.3f}]"
            elif "embedding_score" in e:
                score_tag = f" [emb: {e['embedding_score']:.3f}]"
            lines.append(f"- **[{e['id']}]**{hop_tag}{score_tag} {title}{detail}")
            if what:
                lines.append(f"  {what}")
    text = "\n".join(lines)
    return GccLinksResult(commit_id=commit_id, links_found=len(linked), message=text)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_patterns(
    min_occurrences: int = 1,
    include_promoted: bool = True,
    search_term: str | None = None,
    max_age_hours: int | None = None,
    auto_promote: bool = False,
) -> GccPatternsResult:
    """Query the CER-inspired pattern buffer.

    Patterns are transferable decision-making skills observed across commits
    (CER arXiv:2506.06698). They are deduped by word similarity and tracked
    by occurrence count. Patterns appearing in 3+ commits are suggested for
    ACE playbook promotion.

    Args:
        min_occurrences: Minimum occurrence count to include (default 1).
        include_promoted: Whether to include already-promoted patterns (default True).
        search_term: Optional keyword filter on pattern text.
        max_age_hours: If set, only return patterns last seen within this many hours.
            Useful for surfacing recent patterns: max_age_hours=24 = last 24h only.
        auto_promote: If True, automatically promote all promotion candidates to the
            ACE playbook via ace_apply_delta ADD ops. Errors are silently ignored.
    """
    with _srv._state_lock:
        mem = _srv._ensure_memory()
    result = mem.get_patterns(
        min_occurrences=min_occurrences,
        include_promoted=include_promoted,
        search_term=search_term,
        max_age_hours=max_age_hours,
    )

    total = result["total"]
    matching = result["matching"]

    if not result["patterns"]:
        text = f"No patterns found (total buffer: {total}, filter: min_occurrences={min_occurrences})."
        return GccPatternsResult(total=total, matching=0, message=text)

    lines = [f"# Pattern Buffer ({matching}/{total} shown)"]
    for p in result["patterns"][:25]:
        promoted_tag = " [PROMOTED]" if p.get("promoted") else ""
        q_score = p.get("quality_score", 0.5)
        q_tag = f", q={q_score:.2f}" if (p.get("success_count", 0) + p.get("failure_count", 0)) > 0 else ""
        last_seen_tag = f", last: {p['last_seen']}" if p.get("last_seen") else ""
        lines.append(
            f"- **[{p['id']}]** ({p['occurrence_count']}x{q_tag}, first: {p['first_seen']}{last_seen_tag}){promoted_tag}\n"
            f"  {p['text']}\n"
            f"  Commits: {', '.join(p['commit_ids'])}"
        )

    # Promotion candidates
    threshold = mem.config.pattern_promotion_count
    candidates = [p for p in result["patterns"] if p["occurrence_count"] >= threshold and not p.get("promoted")]
    if candidates:
        lines.append(f"\n## Promotion Candidates (>= {threshold} occurrences)")
        for c in candidates:
            lines.append(f"- [{c['id']}] \"{c['text'][:100]}\" ({c['occurrence_count']}x)")

    # A1: auto_promote — lazily promote candidates to ACE playbook
    if auto_promote and candidates:
        try:
            import ccr.mcp.ace_tools as _ace_tools_mod
            ops = [
                {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": c["text"]}
                for c in candidates
            ]
            _ace_tools_mod.ace_apply_delta(ops)
            lines.append(f"\n[Auto-promoted {len(candidates)} candidate(s) to ACE playbook.]")
        except Exception:
            pass  # Auto-promotion is supplementary — never fail the query

    text = "\n".join(lines)
    return GccPatternsResult(total=total, matching=matching, message=text)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_experiments(
    experiment_id: str | None = None,
    hypothesis_contains: str | None = None,
    metric_filter: dict | None = None,
    date_range: list[str] | None = None,
    top_n: str | None = None,
    compare: list[str] | None = None,
) -> GccExperimentsResult:
    """Query and filter experiment records stored in commits via gcc_commit(experiment={...}).

    Parses **Experiment**: blocks from commit history and returns matching records.
    Supports metric filtering, date ranges, comparison, and metric-based sorting.

    Args:
        experiment_id: Filter to this exact experiment ID (e.g. "exp-042").
        hypothesis_contains: Substring match on hypothesis text (case-insensitive).
        metric_filter: Dict of {metric_name: {op: value}} conditions.
            Supported ops: lt, lte, gt, gte, eq.
            Example: {"val_loss": {"lt": 0.3}, "accuracy": {"gte": 0.8}}
        date_range: [start_date, end_date] as "YYYY-MM-DD" strings.
            Example: ["2026-01-01", "2026-04-01"]
        top_n: "metric_name:asc|desc" — sort all results by metric.
            Example: "val_loss:asc" (best validation loss first)
        compare: Two commit IDs for side-by-side comparison table.
            Example: ["C041", "C053"]

    Returns:
        count: Number of matching experiments.
        records: Raw list of experiment record dicts.
        message: Markdown table of results (or comparison table if compare= given).

    Usage examples:
        # Find all runs with val_loss < 0.3
        gcc_experiments(metric_filter={"val_loss": {"lt": 0.3}})

        # Compare two specific runs
        gcc_experiments(compare=["C041", "C053"])

        # Best runs sorted by accuracy (highest first)
        gcc_experiments(top_n="accuracy:desc")

        # LoRA experiments from last month
        gcc_experiments(hypothesis_contains="LoRA", date_range=["2026-03-01", "2026-04-05"])
    """
    mem = _srv._ensure_memory()

    result = mem.get_experiments(
        experiment_id=experiment_id,
        hypothesis_contains=hypothesis_contains,
        metric_filter=metric_filter,
        date_range=date_range,
        top_n=top_n,
        compare=compare,
    )

    return GccExperimentsResult(
        count=result["count"],
        records=result["records"],
        message=result["message"],
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def gcc_discuss(
    topic: str,
    hypothesis: str,
    alternatives_considered: str,
    decision: str,
    rationale: str,
    uncertainty: str = "",
    linked_commit: str | None = None,
) -> GccDiscussResult:
    """Log a decision or hypothesis to the persistent discussion log.

    Stores a structured record in .ccr/branches/{branch}/discussions.md.
    Use to preserve the reasoning behind design choices, experiment directions,
    and trade-off decisions that would otherwise be lost between sessions.

    Args:
        topic: Short title for the decision (e.g., "dataset preprocessing approach").
        hypothesis: The hypothesis or assumption being tested/decided.
        alternatives_considered: Comma-separated alternatives that were rejected.
        decision: The choice made.
        rationale: Why this decision was made (evidence, benchmarks, constraints).
        uncertainty: Open questions or risks that remain (optional).
        linked_commit: Commit ID (e.g., "C045") this decision relates to (optional).

    Returns:
        id: Discussion ID (D001, D002, ...).
        date: Timestamp of the record.
        topic: The topic as stored.
        message: The formatted discussion block.

    Example:
        gcc_discuss(
            topic="optimizer choice for LoRA fine-tuning",
            hypothesis="AdamW with warmup outperforms SGD",
            alternatives_considered="SGD, Adam (no weight decay), LAMB",
            decision="AdamW with linear warmup",
            rationale="15% lower val_loss vs SGD in 3-epoch ablation (C041)",
            uncertainty="Not tested beyond 10K steps",
            linked_commit="C041",
        )
    """
    mem = _srv._ensure_memory()

    result = mem.add_discussion(
        topic=topic,
        hypothesis=hypothesis,
        alternatives_considered=alternatives_considered,
        decision=decision,
        rationale=rationale,
        uncertainty=uncertainty,
        linked_commit=linked_commit,
    )

    return GccDiscussResult(
        id=result["id"],
        date=result["date"],
        topic=result["topic"],
        message=result["message"],
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_discussions(
    search: str | None = None,
    topic: str | None = None,
    date_range: list[str] | None = None,
) -> GccDiscussionsResult:
    """Query the persistent discussion log.

    Retrieves decision records logged via gcc_discuss. Useful for reviewing
    past reasoning, finding decisions related to a topic, or recalling why
    a certain approach was chosen.

    Args:
        search: Full-text substring match across all fields (case-insensitive).
        topic: Exact topic match (case-insensitive).
        date_range: [start_date, end_date] as "YYYY-MM-DD" strings.

    Returns:
        count: Number of matching discussions.
        records: Raw discussion record dicts.
        message: Markdown table of results.

    Example:
        gcc_discussions(search="optimizer")
        gcc_discussions(date_range=["2026-03-01", "2026-04-05"])
    """
    mem = _srv._ensure_memory()

    result = mem.get_discussions(
        search=search,
        topic=topic,
        date_range=date_range,
    )

    return GccDiscussionsResult(
        count=result["count"],
        records=result["records"],
        message=result["message"],
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_search(
    query: str,
    sources: list[str] | None = None,
    limit: int = 10,
    date_range: list[str] | None = None,
) -> GccSearchResult:
    """Unified search across all CCR memory sources.

    Searches commits, discussions, experiments, and session history in one call.
    Returns aggregated results grouped by source. Use this instead of calling
    gcc_context(level=5), gcc_discussions, gcc_experiments, and session_search
    separately.

    Args:
        query: Text to search for. Case-insensitive substring match.
        sources: Which sources to search. Default: all available.
            Options: "commits", "discussions", "experiments", "sessions".
        limit: Max results per source (default 10).
        date_range: [start_date, end_date] as "YYYY-MM-DD" strings.
            Applied to all sources that support date filtering.

    Returns:
        total: Total number of matches across all sources.
        sources_searched: List of sources that were queried.
        message: Aggregated markdown results grouped by source.

    Example:
        gcc_search("LoRA")
        gcc_search("val_loss", sources=["commits", "experiments"])
        gcc_search("dataset", date_range=["2026-03-01", "2026-04-05"])
    """
    mem = _srv._ensure_memory()

    all_sources = ["commits", "discussions", "experiments", "sessions"]
    if sources:
        active_sources = [s.lower() for s in sources if s.lower() in all_sources]
    else:
        active_sources = all_sources

    sections = []
    total = 0
    searched = []

    # --- Commits ---
    if "commits" in active_sources:
        searched.append("commits")
        try:
            branch = mem.get_active_branch()
            commits_text = mem._search_commits(branch, query)
            if commits_text and commits_text.strip():
                # Count results (## [C###] headers)
                count = len(re.findall(r"## \[C\d{3,}\]", commits_text))
                if count:
                    total += count
                    sections.append(f"## Commits ({count} match{'es' if count != 1 else ''})\n\n{commits_text.strip()}")
        except Exception as _exc:
            _log_search_error("commits", query, _exc)

    # --- Discussions ---
    if "discussions" in active_sources:
        searched.append("discussions")
        try:
            result = mem.get_discussions(search=query, date_range=date_range)
            if result["count"]:
                total += result["count"]
                sections.append(f"## Discussions ({result['count']} match{'es' if result['count'] != 1 else ''})\n\n{result['message']}")
        except Exception as _exc:
            _log_search_error("discussions", query, _exc)

    # --- Experiments ---
    if "experiments" in active_sources:
        searched.append("experiments")
        try:
            result = mem.get_experiments(
                hypothesis_contains=query,
                date_range=date_range,
            )
            # Also try metric key match
            result2 = mem.get_experiments()
            exp_matches = [
                r for r in result2["records"]
                if query.lower() in str(r).lower()
                and r not in result["records"]
            ]
            all_exp_records = result["records"] + exp_matches
            if all_exp_records:
                from ccr.core.memory_pkg.memory_experiments import _format_experiment_table
                count = len(all_exp_records)
                total += count
                sections.append(f"## Experiments ({count} match{'es' if count != 1 else ''})\n\n{_format_experiment_table(all_exp_records[:limit])}")
        except Exception as _exc:
            _log_search_error("experiments", query, _exc)

    # --- Sessions ---
    if "sessions" in active_sources:
        searched.append("sessions")
        try:
            from ccr.core.session_store import SessionStore
            import os as _os
            db_path = _os.path.join(mem.ccr_root, "sessions.db")
            if _os.path.isfile(db_path):
                store = SessionStore(db_path)
                turns = store.search_turns(query, limit=limit)
                if turns:
                    total += len(turns)
                    session_lines = ["| Date | Session | Snippet |", "|------|---------|---------|"]
                    for t in turns[:limit]:
                        date = (t.get("timestamp") or "")[:10]
                        sid = t.get("session_id", "?")[:8]
                        snippet = (t.get("assistant_message") or t.get("user_message") or "")[:80].replace("\n", " ")
                        session_lines.append(f"| {date} | {sid} | {snippet}... |")
                    sections.append(f"## Sessions ({len(turns)} match{'es' if len(turns) != 1 else ''})\n\n" + "\n".join(session_lines))
        except Exception as _exc:
            _log_search_error("sessions", query, _exc)

    if not sections:
        message = f"No results for '{query}' in {', '.join(searched)}."
    else:
        message = "\n\n".join(sections)

    return GccSearchResult(
        total=total,
        sources_searched=searched,
        message=message,
    )


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_scratchpad_search(
    query: str,
    top_k: int = 5,
) -> GccScratchpadSearchResult:
    """Search working memory scratchpad by semantic similarity to query.

    Inspired by AgeMem (arXiv:2601.01885) unified LTM/STM retrieval.
    Uses ONNX semantic embeddings when available, falls back to BM25-style
    word overlap. Useful for locating previously stored hypotheses, debug
    focus notes, or intermediate reasoning state in a multi-turn session.

    Args:
        query: Search text.
        top_k: Maximum number of results to return (default 5).

    Returns:
        total: Number of matching entries returned.
        results: List of {key, value, score, created_at, updated_at} dicts.
        message: Formatted markdown table of results.
    """
    scratchpad = _srv._scratchpad
    if scratchpad is None:
        return GccScratchpadSearchResult(total=0, results=[], message="Scratchpad not initialized.")

    results = scratchpad.search(query, top_k=top_k)
    if not results:
        return GccScratchpadSearchResult(
            total=0, results=[],
            message=f"No scratchpad entries matching '{query}'."
        )

    lines = [f"# Scratchpad Search ({len(results)} result{'s' if len(results) != 1 else ''})",
             "", "| Key | Value | Score |", "|-----|-------|-------|"]
    for r in results:
        key = r["key"]
        val = str(r["value"])[:60].replace("\n", " ")
        score = f"{r['score']:.3f}"
        lines.append(f"| {key} | {val} | {score} |")
    message = "\n".join(lines)

    return GccScratchpadSearchResult(total=len(results), results=results, message=message)
