"""GCC Memory Tools — all gcc_* MCP tool functions."""

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
    GccBranchResult,
    GccClustersResult,
    GccCommitResult,
    GccConsolidateResult,
    GccContextResult,
    GccDiscussResult,
    GccDiscussionsResult,
    GccEvolveMemoryResult,
    GccExperimentsResult,
    GccLinksResult,
    GccLogOtaResult,
    GccMergeResult,
    GccPatternsResult,
    GccScratchpadResult,
    GccSearchResult,
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
def gcc_branch(
    name: str,
    purpose: str,
    hypothesis: str,
    linked_issue: str = "",
    team_owner: str = "",
    priority: str = "",
) -> GccBranchResult:
    """Create an exploration branch for experimental work.

    Branches isolate experimental changes from the main line. Must be on
    main to create a branch. Use kebab-case names (e.g., 'try-new-parser').

    Args:
        name: Branch name in kebab-case.
        purpose: What this branch explores.
        hypothesis: What you expect to learn or achieve.
        linked_issue: Optional issue/ticket reference (e.g., "GH-123").
        team_owner: Optional team or owner identifier.
        priority: Optional priority level (e.g., "high", "p1").
    """
    try:
        with _srv._state_lock:
            mem = _srv._ensure_memory()
            result = mem.create_branch(
                name, purpose, hypothesis,
                linked_issue=linked_issue,
                team_owner=team_owner,
                priority=priority,
            )
        return GccBranchResult(branch=name, message=result)
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False))
def gcc_merge(
    branch: str,
    outcome: str,
    conclusion: str,
    custom_outcome: str = "",
) -> GccMergeResult:
    """Merge an exploration branch back into main.

    Must be on the branch being merged. Integrates the branch's rolling
    summary and OTA log into main.

    Args:
        branch: Branch name to merge.
        outcome: One of 'success', 'failure', or 'partial'.
        conclusion: Summary of what was learned.
        custom_outcome: If non-empty, overrides outcome validation and uses
            this string as the outcome (e.g. "in-progress", "abandoned").
            Bypasses the success/failure/partial constraint.
    """
    try:
        with _srv._state_lock:
            mem = _srv._ensure_memory()
            if custom_outcome:
                result = mem.merge(branch, custom_outcome, conclusion, allow_custom_outcome=True)
            else:
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
def gcc_evolve_memory(
    commit_id: str | None = None,
    rollback: bool = False,
) -> GccEvolveMemoryResult:
    """Manually trigger A-MEM evolution for a commit or all recent commits.

    When a sub-model is available, rewrites commit summaries to incorporate
    context from related commits (A-MEM §3.3 Eq.7 memory evolution).
    When no sub-model is available, applies a text fallback that deduplicates
    sentences in the 'what' field of commit summaries.

    Args:
        commit_id: Specific commit to evolve (e.g. "C001"). If None, evolves
                   all recent commits that have semantic/supersession links.
        rollback: If True, snapshot evolved_summaries before starting. On any
                  exception, restore the snapshot automatically.
    """
    sub = _srv._get_sub_client()

    with _srv._state_lock:
        mem = _srv._ensure_memory()
    mem.sub_client = sub

    # A6a: Snapshot for rollback support
    snapshot = dict(mem._evolved_summaries) if rollback else None

    branch = mem.get_active_branch()
    evolutions_performed: list[str] = []
    diff_lines: list[str] = []

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
            # A6c: Record before state for diff
            before_states: dict[str, str] = {}
            for link in candidate_links:
                tid = link.target if hasattr(link, "target") else link.get("target", "")
                existing = mem.get_evolved_what(tid)
                before_states[tid] = existing or ""
            mem._trigger_memory_evolution(cid, candidate_links)
            after = set(mem._evolved_summaries.keys())
            new_keys = after - before
            # Build diff entries
            for tid in new_keys:
                after_what = mem.get_evolved_what(tid) or ""
                before_what = before_states.get(tid, "")
                diff_lines.append(f"[{tid}]: '{before_what[:60]}' → '{after_what[:60]}'")
            return len(new_keys)

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
        if rollback and snapshot is not None:
            mem._evolved_summaries = snapshot
            try:
                mem._save_evolved_summaries()
            except Exception:
                pass
        raise ToolError(f"Error during memory evolution: {type(e).__name__}: {e}") from e

    if not evolutions_performed or all("no new evolutions" in e for e in evolutions_performed):
        text = "No evolutions performed. Ensure commits have semantic/supersession links with score > 0.5."
        return GccEvolveMemoryResult(evolutions=0, message=text)
    count = sum(1 for e in evolutions_performed if "no new evolutions" not in e)
    text = "A-MEM memory evolution complete:\n" + "\n".join(f"  - {e}" for e in evolutions_performed)
    if diff_lines:
        text += "\n\n## Diffs\n" + "\n".join(f"  {d}" for d in diff_lines)
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
    _srv._ensure_memory()
    mem = _srv._mem

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
    _srv._ensure_memory()
    mem = _srv._mem

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
    _srv._ensure_memory()
    mem = _srv._mem

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
    _srv._ensure_memory()
    mem = _srv._mem

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
        except Exception:
            pass

    # --- Discussions ---
    if "discussions" in active_sources:
        searched.append("discussions")
        try:
            result = mem.get_discussions(search=query, date_range=date_range)
            if result["count"]:
                total += result["count"]
                sections.append(f"## Discussions ({result['count']} match{'es' if result['count'] != 1 else ''})\n\n{result['message']}")
        except Exception:
            pass

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
        except Exception:
            pass

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
        except Exception:
            pass

    if not sections:
        message = f"No results for '{query}' in {', '.join(searched)}."
    else:
        message = "\n\n".join(sections)

    return GccSearchResult(
        total=total,
        sources_searched=searched,
        message=message,
    )
