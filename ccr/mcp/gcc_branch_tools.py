"""GCC Branch Tools — branch management MCP tool functions.

Covers: gcc_branch, gcc_merge, gcc_evolve_memory.
"""

from __future__ import annotations

from mcp.types import ToolAnnotations
from mcp.server.fastmcp.exceptions import ToolError

from ccr.mcp.server import mcp
import ccr.mcp.server as _srv

from ccr.mcp_types import (
    GccBranchResult,
    GccEvolveMemoryResult,
    GccMergeResult,
)


# ===========================================================================
# GCC Branch Management Tools
# ===========================================================================


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
