"""Playbook schema mixin — MCE-inspired schema metrics, evolution, and application.

Contains methods for computing playbook health metrics, proposing schema changes
via rule-based heuristics, clustering OTHERS bullets, and applying schemas.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ccr.ace.playbook_types import _normalize_section, _SLUG_MAP

if TYPE_CHECKING:
    from ccr.ace.playbook_types import Bullet


class SchemaMixin:
    """Mixin providing MCE-inspired schema evolution methods for Playbook.

    Requires the host class to have:
        self._bullets: list[Bullet]
        self._sections: list[str]
        self.serialize() -> str
        self.get_section_bullets(section) -> list[Bullet]
    """

    # Stop words for clustering (minimal set for bullet content analysis)
    _META_STOP_WORDS: frozenset[str] = frozenset({
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "has", "have", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "that", "this", "it", "not", "no",
        "if", "when", "then", "than", "so", "as", "up", "out", "about",
    })

    def compute_metrics(self, schema: "PlaybookSchema | None" = None) -> "SchemaMetrics":
        """Compute mechanical health metrics for the playbook.

        Inspired by MCE §3.2 evaluation, but these are structural metrics
        (section balance, utilization, harmful ratio, decay impact) — not
        task-performance metrics as in the paper. MCE uses LLM-evaluated
        context quality; CCR uses mechanical heuristics only.

        Args:
            schema: Optional schema for decay_rate. Uses 0.95 default if None.

        Returns:
            SchemaMetrics with all fields populated.
        """
        from ccr.core.types import SchemaMetrics

        decay_rate = schema.decay_rate if schema else 0.95
        total = len(self._bullets)
        now_iso = datetime.now(timezone.utc).isoformat()

        if total == 0:
            return SchemaMetrics(
                empty_sections=list(self._sections),
                total_sections=len(self._sections),
                timestamp=now_iso,
            )

        # Section bullet counts
        section_counts: dict[str, int] = defaultdict(int)
        for b in self._bullets:
            section_counts[b.section] += 1

        # Section balance: normalized Shannon entropy
        non_empty = [c for c in section_counts.values() if c > 0]
        n_non_empty = len(non_empty)
        if n_non_empty <= 1:
            section_balance = 0.0
        else:
            entropy = 0.0
            for count in non_empty:
                p = count / total
                if p > 0:
                    entropy -= p * math.log2(p)
            max_entropy = math.log2(n_non_empty)
            section_balance = entropy / max_entropy if max_entropy > 0 else 0.0

        # Utilization, harmful, unused
        utilized = sum(1 for b in self._bullets if b.helpful + b.harmful > 0)
        utilization_rate = utilized / total

        harmful_count = sum(
            1 for b in self._bullets
            if b.helpful + b.harmful > 0 and b.harmful >= b.helpful and b.harmful > 0
        )
        harmful_ratio = harmful_count / utilized if utilized > 0 else 0.0

        unused_ratio = (total - utilized) / total

        # Decay impact: fraction where effective_score < 50% of raw score
        decay_affected = 0
        for b in self._bullets:
            raw = abs(b.score)
            if raw == 0:
                continue
            eff = abs(b.effective_score(decay_rate))
            if eff < 0.5 * raw:
                decay_affected += 1
        scored_bullets = sum(1 for b in self._bullets if b.score != 0)
        decay_impact = decay_affected / scored_bullets if scored_bullets > 0 else 0.0

        # Empty and overflow sections
        overflow_threshold = 0.5
        empty_sections = [s for s in self._sections if section_counts.get(s, 0) == 0]
        overflow_sections = [
            s for s in self._sections
            if section_counts.get(s, 0) > overflow_threshold * total
        ]

        # Overall health: weighted composite
        overall_health = (
            0.25 * section_balance
            + 0.25 * utilization_rate
            + 0.25 * (1.0 - harmful_ratio)
            + 0.15 * (1.0 - unused_ratio)
            + 0.10 * (1.0 - decay_impact)
        )

        return SchemaMetrics(
            section_balance=round(section_balance, 4),
            utilization_rate=round(utilization_rate, 4),
            harmful_ratio=round(harmful_ratio, 4),
            unused_ratio=round(unused_ratio, 4),
            decay_impact=round(decay_impact, 4),
            empty_sections=empty_sections,
            overflow_sections=overflow_sections,
            overall_health=round(overall_health, 4),
            total_bullets=total,
            total_sections=len(self._sections),
            timestamp=now_iso,
        )

    def propose_schema_changes(
        self,
        schema: "PlaybookSchema",
        metrics: "SchemaMetrics | None" = None,
        *,
        overflow_threshold: float = 0.5,
        min_cluster_size: int = 3,
        stop_health_threshold: float = 0.8,
        rollback_health_delta: float = -0.05,
    ) -> list["SchemaProposal"]:
        """Propose at most one schema change via rule-based heuristic proposals.

        Inspired by MCE's (1+1)-ES (single offspring per call), but MCE uses
        LLM-generated offspring via agentic crossover (§3.1). CCR uses
        deterministic condition checks — no LLM calls.

        Checks conditions in priority order, returns the first applicable
        proposal. Returns empty list if playbook is healthy enough.

        Args:
            schema: Current active schema.
            metrics: Pre-computed metrics (computed if None).
            overflow_threshold: Section fraction triggering overflow.
            min_cluster_size: Min bullets to propose new section from OTHERS.
            stop_health_threshold: Health above this → no proposals.
            rollback_health_delta: Health drop threshold for rollback.

        Returns:
            List with 0 or 1 SchemaProposal.
        """
        from ccr.core.types import SchemaProposal

        if metrics is None:
            metrics = self.compute_metrics(schema)

        # 1. ROLLBACK check: if health degraded from baseline
        if schema.baseline_metrics is not None:
            delta = metrics.overall_health - schema.baseline_metrics.overall_health
            if delta < rollback_health_delta:
                return [SchemaProposal(
                    change_type="ROLLBACK",
                    description=(
                        f"Playbook health degraded by {delta:+.3f} since schema v{schema.version} "
                        f"was adopted (from {schema.baseline_metrics.overall_health:.3f} to "
                        f"{metrics.overall_health:.3f}). Consider reverting to v{schema.parent_version}."
                    ),
                    details={
                        "current_health": metrics.overall_health,
                        "baseline_health": schema.baseline_metrics.overall_health,
                        "delta": round(delta, 4),
                        "parent_version": schema.parent_version,
                    },
                    confidence=min(1.0, abs(delta) / 0.1),
                )]

        # 2. Stop criteria (MCE Appendix D.3.2): healthy enough
        if metrics.overall_health >= stop_health_threshold:
            return []

        # 3. ADD_SECTION: clustered bullets in OTHERS
        clusters = self._cluster_others_bullets(min_cluster_size)
        if clusters:
            name, bullet_ids = clusters[0]  # Best cluster
            slug_prefix = _normalize_section(name)[:3]
            return [SchemaProposal(
                change_type="ADD_SECTION",
                description=(
                    f"OTHERS has {len(self.get_section_bullets('OTHERS'))} bullets. "
                    f"Found cluster of {len(bullet_ids)} related bullets. "
                    f"Propose new section '{name}'."
                ),
                details={
                    "name": name,
                    "slug_prefix": slug_prefix,
                    "from_section": "OTHERS",
                    "bullet_ids": bullet_ids,
                },
                confidence=min(1.0, len(bullet_ids) / (2 * min_cluster_size)),
            )]

        # 4. REMOVE_SECTION: persistently empty section
        if metrics.empty_sections:
            baseline_empty = (
                set(schema.baseline_metrics.empty_sections)
                if schema.baseline_metrics else set()
            )
            for sec in metrics.empty_sections:
                if sec == "OTHERS":
                    continue  # Never remove OTHERS
                if sec in baseline_empty or schema.baseline_metrics is None:
                    return [SchemaProposal(
                        change_type="REMOVE_SECTION",
                        description=f"Section '{sec}' has 0 bullets (persistently empty). Propose removal.",
                        details={"name": sec},
                        confidence=0.7,
                    )]

        # 5. ADJUST_DECAY
        if metrics.decay_impact > 0.5:
            new_rate = max(0.90, schema.decay_rate - 0.02)
            if new_rate != schema.decay_rate:
                return [SchemaProposal(
                    change_type="ADJUST_DECAY",
                    description=(
                        f"{metrics.decay_impact:.0%} of bullets heavily decayed. "
                        f"Propose lowering decay rate from {schema.decay_rate} to {new_rate} "
                        f"(slower decay, longer retention)."
                    ),
                    details={"old_rate": schema.decay_rate, "new_rate": new_rate},
                    confidence=min(1.0, (metrics.decay_impact - 0.5) / 0.3),
                )]
        elif metrics.decay_impact < 0.1 and metrics.unused_ratio > 0.5:
            new_rate = min(0.99, schema.decay_rate + 0.02)
            if new_rate != schema.decay_rate:
                return [SchemaProposal(
                    change_type="ADJUST_DECAY",
                    description=(
                        f"Only {metrics.decay_impact:.0%} decayed but {metrics.unused_ratio:.0%} unused. "
                        f"Propose raising decay rate from {schema.decay_rate} to {new_rate} "
                        f"(faster decay to clear stale bullets)."
                    ),
                    details={"old_rate": schema.decay_rate, "new_rate": new_rate},
                    confidence=0.5,
                )]

        # 6. ADJUST_PRUNING
        if metrics.harmful_ratio > 0.3:
            new_min = max(1, schema.prune_min_harmful - 1)
            if new_min != schema.prune_min_harmful:
                return [SchemaProposal(
                    change_type="ADJUST_PRUNING",
                    description=(
                        f"{metrics.harmful_ratio:.0%} of utilized bullets are net-harmful. "
                        f"Propose lowering prune threshold from {schema.prune_min_harmful} to {new_min}."
                    ),
                    details={
                        "old_min_harmful": schema.prune_min_harmful,
                        "new_min_harmful": new_min,
                    },
                    confidence=min(1.0, (metrics.harmful_ratio - 0.3) / 0.2),
                )]
        elif metrics.harmful_ratio < 0.05 and schema.prune_min_harmful < 5:
            new_min = schema.prune_min_harmful + 1
            return [SchemaProposal(
                change_type="ADJUST_PRUNING",
                description=(
                    f"Only {metrics.harmful_ratio:.0%} harmful. "
                    f"Propose raising prune threshold from {schema.prune_min_harmful} to {new_min} "
                    f"(less aggressive pruning)."
                ),
                details={
                    "old_min_harmful": schema.prune_min_harmful,
                    "new_min_harmful": new_min,
                },
                confidence=0.4,
            )]

        # 7. REBALANCE: overflow sections
        if metrics.overflow_sections:
            overflow_sec = metrics.overflow_sections[0]
            overflow_bullets = self.get_section_bullets(overflow_sec)
            if len(overflow_bullets) > 1:
                # Find most dissimilar bullet in overflow section
                avg_words: set[str] = set()
                for b in overflow_bullets:
                    avg_words |= set(b.content.lower().split())
                # Find bullet with lowest overlap to section average
                best_move = None
                lowest_sim = 1.0
                for b in overflow_bullets:
                    b_words = set(b.content.lower().split())
                    union = avg_words | b_words
                    inter = avg_words & b_words
                    sim = len(inter) / len(union) if union else 0.0
                    if sim < lowest_sim:
                        lowest_sim = sim
                        best_move = b
                if best_move:
                    # Find best target section
                    target = "OTHERS" if overflow_sec != "OTHERS" else (
                        self._sections[0] if self._sections else "OTHERS"
                    )
                    return [SchemaProposal(
                        change_type="REBALANCE",
                        description=(
                            f"Section '{overflow_sec}' has {len(overflow_bullets)} bullets "
                            f"(>{overflow_threshold:.0%} of total). "
                            f"Propose moving [{best_move.id}] to '{target}'."
                        ),
                        details={
                            "moves": [{
                                "bullet_id": best_move.id,
                                "from": overflow_sec,
                                "to": target,
                            }],
                        },
                        confidence=0.5,
                    )]

        # 8. ADJUST_BUDGET
        total_chars = len(self.serialize())
        if total_chars > 0.9 * schema.token_budget:
            new_budget = schema.token_budget + 20000
            return [SchemaProposal(
                change_type="ADJUST_BUDGET",
                description=(
                    f"Playbook using {total_chars}/{schema.token_budget} chars "
                    f"({total_chars/schema.token_budget:.0%}). "
                    f"Propose increasing budget to {new_budget}."
                ),
                details={"old_budget": schema.token_budget, "new_budget": new_budget},
                confidence=min(1.0, (total_chars / schema.token_budget - 0.9) / 0.1),
            )]
        elif total_chars < 0.3 * schema.token_budget and schema.token_budget > 40000:
            new_budget = max(40000, schema.token_budget - 20000)
            if new_budget != schema.token_budget:
                return [SchemaProposal(
                    change_type="ADJUST_BUDGET",
                    description=(
                        f"Playbook using only {total_chars}/{schema.token_budget} chars "
                        f"({total_chars/schema.token_budget:.0%}). "
                        f"Propose decreasing budget to {new_budget}."
                    ),
                    details={"old_budget": schema.token_budget, "new_budget": new_budget},
                    confidence=0.3,
                )]

        return []

    def _cluster_others_bullets(self, min_cluster: int = 3) -> list[tuple[str, list[str]]]:
        """Cluster OTHERS bullets by content similarity for section proposals.

        Uses single-linkage clustering via BFS on a word Jaccard similarity
        graph (threshold >= 0.3).

        Returns:
            List of (proposed_section_name, [bullet_ids]) sorted by cluster size desc.
        """
        others = self.get_section_bullets("OTHERS")
        if len(others) < min_cluster:
            return []

        # Build word sets (excluding stop words)
        bullet_words: dict[str, set[str]] = {}
        for b in others:
            words = {
                w for w in b.content.lower().split()
                if w not in self._META_STOP_WORDS and len(w) > 2
            }
            if words:
                bullet_words[b.id] = words

        if len(bullet_words) < min_cluster:
            return []

        # Build adjacency graph with word Jaccard >= 0.3
        adjacency: dict[str, set[str]] = defaultdict(set)
        ids = list(bullet_words.keys())
        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1:]:
                inter = bullet_words[id_a] & bullet_words[id_b]
                union = bullet_words[id_a] | bullet_words[id_b]
                jaccard = len(inter) / len(union) if union else 0.0
                if jaccard >= 0.3:
                    adjacency[id_a].add(id_b)
                    adjacency[id_b].add(id_a)

        # BFS connected components
        visited: set[str] = set()
        clusters: list[list[str]] = []
        for start in ids:
            if start in visited:
                continue
            component: list[str] = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                queue.extend(adjacency.get(node, set()) - visited)
            if len(component) >= min_cluster:
                clusters.append(component)

        # Generate proposed names from top-3 frequent words
        results: list[tuple[str, list[str]]] = []
        for cluster in sorted(clusters, key=len, reverse=True):
            word_freq: dict[str, int] = defaultdict(int)
            for bid in cluster:
                for w in bullet_words.get(bid, set()):
                    word_freq[w] += 1
            top_words = sorted(word_freq, key=word_freq.get, reverse=True)[:3]  # type: ignore[arg-type]
            name = " ".join(w.upper() for w in top_words) if top_words else "UNNAMED"
            results.append((name, cluster))

        return results

    def apply_schema(self, schema: "PlaybookSchema") -> int:
        """Apply a schema to the playbook, updating sections.

        Bullets in sections removed by the schema get moved to OTHERS.
        New sections are added. Existing bullets stay in place.

        Args:
            schema: The schema to apply.

        Returns:
            Number of bullets moved.
        """
        moved = 0
        schema_sections = set(schema.sections)

        # Move bullets from removed sections to OTHERS
        for bullet in self._bullets:
            if bullet.section and bullet.section not in schema_sections:
                bullet.section = "OTHERS"
                moved += 1

        # Update sections list
        self._sections = list(schema.sections)

        # Ensure OTHERS exists
        if "OTHERS" not in self._sections:
            self._sections.append("OTHERS")

        # Register new slug prefixes at runtime
        for section_name in schema.sections:
            normalized = _normalize_section(section_name)
            if normalized not in _SLUG_MAP and normalized in schema.slug_map:
                _SLUG_MAP[normalized] = schema.slug_map[normalized]

        return moved
