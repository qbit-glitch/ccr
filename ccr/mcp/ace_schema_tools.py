"""ACE Schema Evolution Tools — ace_evolve_schema + helpers.

Extracted from ace_tools.py to satisfy 800-line coding-style limit.
"""

from __future__ import annotations

from datetime import datetime, timezone

from mcp.types import ToolAnnotations
from mcp.server.fastmcp.exceptions import ToolError

from ccr.core.types import PlaybookSchema, SchemaMetrics  # noqa: F401
from ccr.mcp.server import mcp
import ccr.mcp.server as _srv

from ccr.mcp_types import AceEvolveSchemaResult


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


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
