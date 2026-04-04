"""GCC Memory Tools — all gcc_* MCP tool functions."""

from __future__ import annotations

import re

from mcp.types import ToolAnnotations
from mcp.server.fastmcp.exceptions import ToolError

from ccr.mcp.server import mcp
import ccr.mcp.server as _srv

# All server functions/globals accessed via _srv to support test patching.
# Only `mcp` (the FastMCP singleton) is imported directly — it never changes.
from ccr.mcp_types import (
    GccBranchResult,
    GccClustersResult,
    GccCommitResult,
    GccConsolidateResult,
    GccContextResult,
    GccEvolveMemoryResult,
    GccLinksResult,
    GccLogOtaResult,
    GccMergeResult,
    GccPatternsResult,
    GccScratchpadResult,
    GccStatusResult,
    GccTriplesResult,
)


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
    patterns_learned: list[str] | None = None,
    admission_threshold: float = 0.85,
    rejection_threshold: float = 0.0,
    compressed_summary: str | None = None,
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
        # --- End validation ---

        with _srv._state_lock:
            mem = _srv._ensure_memory()
            # Phase 2: Auto-extract patterns via sub-model when not provided (CER §3.2)
            if patterns_learned is None and mem.config.auto_extract_patterns and len(what) > 100:
                sub = _srv._get_sub_client()
                if sub is not None:
                    patterns_learned = _srv._extract_patterns_from_commit(
                        title, what, why, files_changed or [], sub
                    ) or None
            result = mem.commit(title, what, why, files_changed, next_step,
                                patterns_learned,
                                admission_threshold, rejection_threshold,
                                compressed_summary)

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
        return GccCommitResult(
            commit_id=cid,
            branch=branch,
            title=title,
            admission_decision=admission_decision,
            message=result,
        )
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def gcc_branch(name: str, purpose: str, hypothesis: str) -> GccBranchResult:
    """Create an exploration branch for experimental work.

    Branches isolate experimental changes from the main line. Must be on
    main to create a branch. Use kebab-case names (e.g., 'try-new-parser').

    Args:
        name: Branch name in kebab-case.
        purpose: What this branch explores.
        hypothesis: What you expect to learn or achieve.
    """
    try:
        with _srv._state_lock:
            mem = _srv._ensure_memory()
            result = mem.create_branch(name, purpose, hypothesis)
        return GccBranchResult(branch=name, message=result)
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False))
def gcc_merge(branch: str, outcome: str, conclusion: str) -> GccMergeResult:
    """Merge an exploration branch back into main.

    Must be on the branch being merged. Integrates the branch's rolling
    summary and OTA log into main.

    Args:
        branch: Branch name to merge.
        outcome: One of 'success', 'failure', or 'partial'.
        conclusion: Summary of what was learned.
    """
    try:
        with _srv._state_lock:
            mem = _srv._ensure_memory()
            result = mem.merge(branch, outcome, conclusion)
        return GccMergeResult(source=branch, target="main", message=result)
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
    """
    level = max(1, min(level, 5))
    log_window = max(0, min(log_window, 50))
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
            result = result[:budget_chars]
            result += f"\n\n[Context truncated to ~{max_tokens} tokens. Use level=2 or sections filter for targeted retrieval.]"

    if _level_warnings:
        result = "\n".join(f"[Warning: {w}]" for w in _level_warnings) + "\n\n" + result

    branch = mem.get_active_branch()
    return GccContextResult(level=level, branch=branch, message=result)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_links(
    commit_id: str,
    link_types: str | None = None,
    max_hops: int = 1,
    query: str | None = None,
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
    linked = mem.get_linked_commits(commit_id, link_types=types, max_hops=max_hops, query=query)
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


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def gcc_evolve_memory(commit_id: str | None = None) -> GccEvolveMemoryResult:
    """Manually trigger A-MEM evolution for a commit or all recent commits.

    When a sub-model is available, rewrites commit summaries to incorporate
    context from related commits (A-MEM §3.3 Eq.7 memory evolution).

    Args:
        commit_id: Specific commit to evolve (e.g. "C001"). If None, evolves
                   all recent commits that have semantic/supersession links.
    """
    sub = _srv._get_sub_client()
    if sub is None:
        text = (
            "Sub-model not available. Set CCR_OLLAMA_MODEL or ANTHROPIC_API_KEY "
            "to enable A-MEM memory evolution."
        )
        return GccEvolveMemoryResult(evolutions=0, message=text)

    with _srv._state_lock:
        mem = _srv._ensure_memory()
    mem.sub_client = sub

    branch = mem.get_active_branch()
    evolutions_performed: list[str] = []

    try:
        from ccr.core.types import CommitLink  # noqa: PLC0415
        links_data = mem._load_links()

        def _evolve_commit_with_links(cid: str) -> int:
            """Evolve one commit's linked peers. Returns count of newly evolved entries."""
            node = links_data.get("links", {}).get(cid, {})
            candidate_links: list[CommitLink] = []
            for lt in ("semantic", "supersession"):
                for entry in node.get(lt, []):
                    score = entry.get("score", 0.0)
                    if score > 0.5:
                        candidate_links.append(CommitLink(
                            target=entry["target"],
                            link_type=lt,
                            score=score,
                        ))
            if not candidate_links:
                return 0
            before = set(mem._evolved_summaries.keys())
            mem._trigger_memory_evolution(cid, candidate_links)
            after = set(mem._evolved_summaries.keys())
            return len(after - before)

        if commit_id:
            n = _evolve_commit_with_links(commit_id)
            if n > 0:
                evolutions_performed.append(f"{commit_id}: {n} new evolution(s)")
            else:
                evolutions_performed.append(
                    f"{commit_id}: no new evolutions "
                    "(no eligible links or sub-model produced no changes)"
                )
        else:
            # Scan last 10 commits
            recent = mem._parse_recent_commit_data(branch, k=10)
            for commit in recent:
                cid = commit.get("id", "")
                if not cid:
                    continue
                n = _evolve_commit_with_links(cid)
                if n > 0:
                    evolutions_performed.append(f"{cid}: {n} new evolution(s)")

    except Exception as e:
        raise ToolError(f"Error during memory evolution: {type(e).__name__}: {e}") from e

    if not evolutions_performed or all("no new evolutions" in e for e in evolutions_performed):
        text = "No evolutions performed. Ensure commits have semantic/supersession links with score > 0.5."
        return GccEvolveMemoryResult(evolutions=0, message=text)
    count = sum(1 for e in evolutions_performed if "no new evolutions" not in e)
    text = "A-MEM memory evolution complete:\n" + "\n".join(f"  - {e}" for e in evolutions_performed)
    return GccEvolveMemoryResult(evolutions=count, message=text)


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


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_status() -> GccStatusResult:
    """Show current project memory status.

    Returns the active branch, recent milestones, open branches,
    and metadata summary.
    """
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



@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def gcc_patterns(
    min_occurrences: int = 1,
    include_promoted: bool = True,
    search_term: str | None = None,
    max_age_hours: int | None = None,
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
        lines.append(
            f"- **[{p['id']}]** ({p['occurrence_count']}x{q_tag}, first: {p['first_seen']}){promoted_tag}\n"
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

    text = "\n".join(lines)
    return GccPatternsResult(total=total, matching=matching, message=text)


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
