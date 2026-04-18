"""Playbook analytics mixin — similarity, pruning, stats, ranking.

Contains methods for finding similar bullet pairs, pruning problematic bullets,
enforcing token budgets, computing statistics, and GRPO-based ranking.
"""

from __future__ import annotations

import math
from typing import Any

from ccr.ace.playbook_types import Bullet, PlaybookStats


class AnalyticsMixin:
    """Mixin providing analytics methods for Playbook.

    Requires the host class to have:
        self._bullets: list[Bullet]
        self._sections: list[str]
        self.serialize() -> str
        self.get_bullet(bullet_id) -> Bullet | None
        self.check_evolution_needed(threshold) -> dict
    """

    def find_similar_pairs(self, threshold: float = 0.6) -> list[tuple[Bullet, Bullet, float]]:
        """Find bullet pairs with high text similarity.

        Primary: ONNX cosine similarity (all-MiniLM-L6-v2).
        Fallback: combined word-overlap Jaccard + character trigram similarity.

        Returns:
            List of (bullet_a, bullet_b, similarity_score) tuples.
        """
        from ccr.context.embeddings import get_embedding_model

        texts = [(a.content + " " + a.trigger).strip() for a in self._bullets]
        model = get_embedding_model()
        embeddings = None

        # Try ONNX batch embedding for O(n) embed + O(n^2) dot product
        if model is not None and len(texts) >= 2:
            try:
                non_empty = [t if len(t.split()) >= 2 else "empty" for t in texts]
                embeddings = model.embed_batch(non_empty)
            except Exception:
                embeddings = None

        pairs = []
        for i, a in enumerate(self._bullets):
            text_a = texts[i]
            if len(text_a.split()) < 2:
                continue
            for j in range(i + 1, len(self._bullets)):
                b = self._bullets[j]
                text_b = texts[j]
                if len(text_b.split()) < 2:
                    continue

                if embeddings is not None:
                    # ONNX cosine (primary)
                    combined = float(embeddings[i] @ embeddings[j])
                else:
                    # Jaccard + trigram fallback
                    words_a = set(text_a.lower().split())
                    words_b = set(text_b.lower().split())
                    trigrams_a = self._char_trigrams(text_a.lower())
                    trigrams_b = self._char_trigrams(text_b.lower())
                    word_inter = words_a & words_b
                    word_union = words_a | words_b
                    word_jaccard = len(word_inter) / len(word_union) if word_union else 0.0
                    tri_inter = trigrams_a & trigrams_b
                    tri_union = trigrams_a | trigrams_b
                    tri_jaccard = len(tri_inter) / len(tri_union) if tri_union else 0.0
                    combined = 0.4 * word_jaccard + 0.6 * tri_jaccard

                if combined >= threshold:
                    pairs.append((a, b, combined))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    @staticmethod
    def _char_trigrams(text: str) -> set[str]:
        """Extract character trigrams from text."""
        text = text.strip()
        if len(text) < 3:
            return set()
        return {text[i:i+3] for i in range(len(text) - 2)}

    def prune_problematic(self, min_harmful: int = 3) -> list[Bullet]:
        """Remove bullets where harmful >= helpful and harmful >= min_harmful.

        Per ACE paper (§3.2): Grow-and-refine prunes redundancy.
        Per P1-3: Returned bullets retain their failure_lessons so callers
        can feed them into evolve_from_failures() before discarding.

        Returns:
            List of removed bullets (with failure_lessons intact).
        """
        removed = []
        kept = []
        for b in self._bullets:
            if b.harmful >= b.helpful and b.harmful >= min_harmful:
                removed.append(b)
            else:
                kept.append(b)
        self._bullets = kept
        for b in removed:
            self._id_index.pop(b.id, None)
        return removed

    def enforce_token_budget(self, max_chars: int, decay_rate: float = 0.95) -> list[Bullet]:
        """Remove lowest-scoring bullets until under budget (M7: O(n log n) not O(n^2)).

        Pre-sorts bullets by score once, then removes cheapest until under budget,
        estimating size reduction from format_line() length instead of re-serializing.

        Args:
            max_chars: Maximum character count for the serialized playbook.
            decay_rate: Temporal decay rate for effective_score (MCE schema param).

        Returns:
            List of removed bullets.
        """
        removed = []
        current_size = len(self.serialize())
        if current_size <= max_chars:
            return removed
        # Sort ascending: (worst_score, highest_harmful, oldest_timestamp) pruned first.
        # ISO string comparison: "" < "2026-01-..." < "2026-04-..." so newly-evolved
        # bullets (recent last_updated) sort LATER and are spared when over budget.
        ranked = sorted(
            self._bullets,
            key=lambda b: (b.effective_score(decay_rate), -b.harmful, b.last_updated or ""),
        )
        keep = set(id(b) for b in self._bullets)
        for bullet in ranked:
            if current_size <= max_chars:
                break
            keep.discard(id(bullet))
            removed.append(bullet)
            # Estimate size reduction: format_line + newline + section header overhead
            current_size -= len(bullet.format_line()) + 1
        self._bullets = [b for b in self._bullets if id(b) in keep]
        for b in removed:
            self._id_index.pop(b.id, None)
        return removed

    def get_stats(self) -> PlaybookStats:
        """Get playbook statistics."""
        stats = PlaybookStats(
            total_chars=len(self.serialize()),
            total_sections=len(self._sections),
        )

        for bullet in self._bullets:
            stats.total_bullets += 1
            if bullet.helpful > 5 and bullet.harmful < 2:
                stats.high_performing += 1
            elif bullet.is_problematic:
                stats.problematic += 1
            elif bullet.is_unused:
                stats.unused += 1

            section = bullet.section or "general"
            if section not in stats.by_section:
                stats.by_section[section] = {"count": 0, "helpful": 0, "harmful": 0}
            stats.by_section[section]["count"] += 1
            stats.by_section[section]["helpful"] += bullet.helpful
            stats.by_section[section]["harmful"] += bullet.harmful

            # Failure lesson stats
            stats.total_failure_lessons += len(bullet.failure_lessons)
            if bullet.harmful > 0:
                if bullet.has_failure_lessons:
                    stats.harmful_with_lessons += 1
                else:
                    stats.harmful_without_lessons += 1

        # Temporal decay stats
        for bullet in self._bullets:
            raw = float(bullet.score)
            if raw != 0 and bullet.last_updated:
                eff = bullet.effective_score()
                if abs(eff) < abs(raw) * 0.5:
                    stats.decayed_bullets += 1

        # Evolution trigger check (SkillRL §3.3)
        evo = self.check_evolution_needed()
        stats.evolution_needed = evo["needed"]
        stats.evolution_candidates = evo["candidate_count"]

        return stats

    def check_evolution_needed(self, threshold: int = 3) -> dict[str, Any]:
        """Check if skill evolution should be triggered (SkillRL §3.3).

        Per SkillRL §3.3: Evolution is triggered when accuracy for a task category
        falls below threshold δ. In our adaptation, we trigger when the number of
        harmful bullets with failure lessons meets or exceeds the threshold.

        Args:
            threshold: Minimum number of harmful-with-lessons bullets to trigger evolution.

        Returns:
            Dict with 'needed' (bool), 'candidate_count' (int), 'candidate_ids' (list[str]).
        """
        candidates = [
            b for b in self._bullets
            if b.harmful > 0 and b.has_failure_lessons
        ]
        return {
            "needed": len(candidates) >= threshold,
            "candidate_count": len(candidates),
            "candidate_ids": [b.id for b in candidates],
        }

    # ------------------------------------------------------------------
    # GRPO-inspired group-relative advantage scoring (SkillRL GRPO Eq.3)
    # ------------------------------------------------------------------

    @staticmethod
    def _word_jaccard_sim(a: str, b: str) -> float:
        """Word-level Jaccard similarity between two strings."""
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def recompute_grpo_advantages(self) -> int:
        """Recompute GRPO group-relative advantages for all bullets (SkillRL Eq.3).

        Groups bullets within each section by word-Jaccard similarity (>= 0.4).
        Reward r_i = effective_score (helpful * decay). Advantage A_i = (r_i - mean) / (std + epsilon).
        Single-bullet groups get advantage 0.0.
        Returns number of bullets updated.
        """
        updated = 0
        for section in self._sections:
            section_bullets = [b for b in self._bullets if b.section == section]
            if not section_bullets:
                continue

            # Greedy word-Jaccard clustering with threshold 0.4
            groups: list[list[Bullet]] = []
            for bullet in section_bullets:
                placed = False
                for group in groups:
                    anchor = group[0]
                    if self._word_jaccard_sim(bullet.content, anchor.content) >= 0.4:
                        group.append(bullet)
                        placed = True
                        break
                if not placed:
                    groups.append([bullet])

            for group in groups:
                rewards = [b.effective_score() for b in group]
                n = len(rewards)
                if n == 1:
                    group[0].grpo_advantage = 0.0
                    updated += 1
                    continue
                mean_r = sum(rewards) / n
                variance = sum((r - mean_r) ** 2 for r in rewards) / n
                std_r = math.sqrt(variance)
                if std_r <= 1e-10:
                    # Degenerate group: all rewards identical — advantage is 0 for all
                    for bullet in group:
                        bullet.grpo_advantage = 0.0
                        updated += 1
                    continue
                for bullet, r_i in zip(group, rewards):
                    bullet.grpo_advantage = (r_i - mean_r) / (std_r + 1e-8)
                    updated += 1

        # Also handle bullets with no section or section not in self._sections
        orphans = [b for b in self._bullets if b.section not in self._sections]
        for bullet in orphans:
            bullet.grpo_advantage = 0.0
            updated += 1

        return updated

    def get_policy_ranked(self, task_context: str = "", top_k: int = 10) -> list[Bullet]:
        """Return bullets ranked by effective_score * (1 + grpo_advantage) — GRPO policy-weighted.

        When task_context is provided, filters to bullets with similarity >= 0.1
        against bullet content, when_to_apply, or trigger. Primary: ONNX cosine.
        Fallback: word-Jaccard. Trigger matches get a 1.5x weight boost
        (ERL-inspired: structured trigger/action enables context-aware retrieval).
        Returns top_k results.
        """
        candidates = list(self._bullets)
        retrieval_scores: dict[str, float] = {}

        if task_context.strip():
            from ccr.context.embeddings import quick_cosine

            filtered = []
            for b in candidates:
                # Try ONNX cosine first, fall back to word Jaccard
                trigger_text = b.trigger if b.trigger else ""
                content_text = b.content + " " + b.when_to_apply

                trigger_sim_onnx = quick_cosine(task_context, trigger_text) if trigger_text else None
                content_sim_onnx = quick_cosine(task_context, content_text)

                if trigger_sim_onnx is not None:
                    trigger_sim = trigger_sim_onnx
                else:
                    trigger_sim = self._word_jaccard_sim(task_context, trigger_text) if trigger_text else 0.0

                if content_sim_onnx is not None:
                    content_sim = content_sim_onnx
                else:
                    content_sim = self._word_jaccard_sim(task_context, content_text)

                combined = max(trigger_sim * 1.5, content_sim)
                if combined >= 0.1:
                    retrieval_scores[b.id] = combined
                    filtered.append(b)
            candidates = filtered if filtered else candidates

        def policy_score(b: Bullet) -> float:
            base = b.effective_score() * (1.0 + b.grpo_advantage)
            return base * (1.0 + retrieval_scores.get(b.id, 0.0))

        candidates.sort(key=policy_score, reverse=True)
        return candidates[:top_k]
