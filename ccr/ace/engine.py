"""ACE Engine — orchestrates the online adaptation loop.

Per ACE paper (§3): The workflow begins with the Generator producing reasoning
trajectories, the Reflector critiques these to extract lessons, and the Curator
synthesizes these into compact delta entries merged deterministically.

For CCR, we primarily use **online mode**: the playbook improves as Claude Code
works on tasks, using execution feedback (not ground truth labels).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from ccr.ace.agents import ACECurator, ACEDeduplicator, ACEGenerator, ACEReflector
from ccr.core.exceptions import ModelError, PlaybookError
from ccr.ace.playbook import (
    DeltaOperation,
    Playbook,
    create_empty_playbook,
    parse_delta_operations,
)

logger = logging.getLogger(__name__)


@dataclass
class ACEConfig:
    """Configuration for the ACE engine."""

    enabled: bool = True
    playbook_token_budget: int = 80000  # max chars for playbook
    curator_frequency: int = 1  # curate after every N tasks
    max_reflection_rounds: int = 3
    prune_min_harmful: int = 3  # min harmful count before pruning
    max_tokens: int = 4096
    playbook_path: str = ""  # auto-set to .ccr/playbook.txt
    refinement_frequency: int = 10  # run dedup every N steps
    dedup_similarity_threshold: float = 0.6  # Jaccard threshold for candidate pairs


@dataclass
class AdaptationResult:
    """Result of a single online adaptation step."""

    task_summary: str = ""
    was_successful: bool = False
    reflection_ran: bool = False
    reflection_rounds: int = 0
    curator_ran: bool = False
    refinement_ran: bool = False
    bullets_added: int = 0
    bullets_updated: int = 0
    bullets_pruned: int = 0
    bullets_merged: int = 0
    playbook_size: int = 0


class ACEEngine:
    """Manages the ACE adaptation loop for CCR.

    The primary mode is **online adaptation**: after each task execution,
    the engine reflects on the result and updates the playbook.

    The playbook is stored at `.ccr/playbook.txt` and persists across sessions
    via the GCC memory system.
    """

    def __init__(
        self,
        sub_client: Any,
        config: ACEConfig | None = None,
        playbook_path: str | None = None,
    ):
        self.config = config or ACEConfig()
        self._step_count = 0

        # Initialize agents (all use same sub-model)
        self.generator = ACEGenerator(sub_client, self.config.max_tokens)
        self.reflector = ACEReflector(sub_client, self.config.max_tokens)
        self.curator = ACECurator(sub_client, self.config.max_tokens)
        self.deduplicator = ACEDeduplicator(sub_client, self.config.max_tokens)

        # Load or create playbook
        self._playbook_path = playbook_path or self.config.playbook_path
        self._playbook = self._load_or_create_playbook()

    def _load_or_create_playbook(self) -> Playbook:
        """Load playbook from disk or create a new one."""
        if self._playbook_path and os.path.isfile(self._playbook_path):
            try:
                with open(self._playbook_path) as f:
                    text = f.read()
                if text.strip():
                    logger.info(f"Loaded playbook from {self._playbook_path}")
                    return Playbook(text)
            except OSError as e:
                logger.warning(f"Failed to load playbook: {e}")

        return create_empty_playbook()

    def _save_playbook(self) -> None:
        """Persist playbook to disk."""
        if not self._playbook_path:
            return
        try:
            os.makedirs(os.path.dirname(self._playbook_path), exist_ok=True)
            with open(self._playbook_path, "w") as f:
                f.write(self._playbook.serialize())
        except OSError as e:
            logger.warning(f"Failed to save playbook: {e}")

    @property
    def playbook(self) -> Playbook:
        """Current playbook."""
        return self._playbook

    @property
    def playbook_text(self) -> str:
        """Current playbook as text, for injection into prompts."""
        return self._playbook.serialize()

    def adapt_online(
        self,
        task: str,
        execution_trace: str,
        execution_result: str,
        was_successful: bool,
        context: str = "",
        ground_truth: str | None = None,
    ) -> AdaptationResult:
        """Run one online adaptation step.

        Called after each task execution. Uses execution feedback to
        reflect and update the playbook.

        Args:
            task: The original task/question.
            execution_trace: The full reasoning/execution trace.
            execution_result: The final answer/result.
            was_successful: Whether execution succeeded (from env feedback).
            context: Additional context (files, code).
            ground_truth: Optional ground truth for supervised adaptation.

        Returns:
            AdaptationResult with details of what was updated.
        """
        self._step_count += 1
        result = AdaptationResult(task_summary=task[:100])

        if not self.config.enabled:
            return result

        # Step 1: Get bullets the Generator would have used
        playbook_text = self._playbook.serialize()
        bullets_used_text = "(no bullets referenced)"

        # Step 2: Iterative reflection (per ACE paper — multiple rounds improve quality)
        env_feedback = "Execution succeeded" if was_successful else "Execution failed"
        if execution_result:
            env_feedback += f"\nResult: {execution_result[:500]}"

        reflection_response = ""
        bullet_tags = []
        key_insight = ""
        for round_idx in range(self.config.max_reflection_rounds):
            try:
                reflection_response, new_tags, key_insight = self.reflector.reflect(
                    question=task,
                    reasoning_trace=execution_trace[:3000],
                    predicted_answer=execution_result[:500],
                    bullets_used=bullets_used_text,
                    environment_feedback=env_feedback,
                    ground_truth=ground_truth,
                )
                result.reflection_ran = True
                result.reflection_rounds = round_idx + 1
                bullet_tags.extend(new_tags)

                # If reflector found no issues, stop early
                if not key_insight or key_insight.lower() in ("n/a", "none", ""):
                    break

                # Update env_feedback with insight for next round
                env_feedback = f"{env_feedback}\nPrevious reflection insight: {key_insight}"
            except ModelError as e:
                logger.error(f"ACE reflection round {round_idx} failed: {e}")
                if round_idx == 0:
                    return result  # First round failure is fatal
                break  # Later round failures are OK
            except Exception as e:
                logger.error(f"ACE reflection round {round_idx} unexpected error: {e}")
                if round_idx == 0:
                    return result
                break

        # Step 3: Update bullet counts from tags
        if bullet_tags:
            result.bullets_updated = self._playbook.update_bullet_counts(bullet_tags)

        # Step 4: Run Curator (respecting frequency)
        if self._step_count % self.config.curator_frequency == 0:
            try:
                stats = self._playbook.get_stats()
                stats_str = (
                    f"Total bullets: {stats.total_bullets}, "
                    f"High-performing: {stats.high_performing}, "
                    f"Problematic: {stats.problematic}, "
                    f"Unused: {stats.unused}"
                )

                curator_response, raw_ops = self.curator.curate(
                    current_playbook=playbook_text,
                    recent_reflection=reflection_response,
                    question_context=context[:1000] if context else task[:500],
                    playbook_stats=stats_str,
                    current_step=self._step_count,
                    total_samples=0,  # unknown in online mode
                    token_budget=self.config.playbook_token_budget,
                    available_sections=", ".join(self._playbook.sections),
                )
                result.curator_ran = True

                # Parse and apply delta operations
                ops = parse_delta_operations({"operations": raw_ops})
                if ops:
                    result.bullets_added = self._playbook.apply_delta(ops)
            except (ModelError, PlaybookError) as e:
                logger.error(f"ACE curation failed: {e}")
            except Exception as e:
                logger.error(f"ACE curation failed (unexpected): {e}")

        # Step 5: Periodic refinement (deduplication)
        if (
            self.config.refinement_frequency > 0
            and self._step_count % self.config.refinement_frequency == 0
            and len(self._playbook.bullets) >= 3
        ):
            try:
                merged = self._run_refinement()
                result.bullets_merged = merged
                result.refinement_ran = True
            except (ModelError, PlaybookError) as e:
                logger.error(f"ACE refinement failed: {e}")
            except Exception as e:
                logger.error(f"ACE refinement failed (unexpected): {e}")

        # Step 6: Prune problematic bullets
        pruned = self._playbook.prune_problematic(
            min_harmful=self.config.prune_min_harmful
        )
        result.bullets_pruned = len(pruned)

        # Step 7: Enforce token budget
        if len(self._playbook.serialize()) > self.config.playbook_token_budget:
            removed = self._playbook.enforce_token_budget(
                self.config.playbook_token_budget
            )
            result.bullets_pruned += len(removed)

        # Step 8: Save
        result.was_successful = was_successful
        result.playbook_size = len(self._playbook.serialize())
        self._save_playbook()

        logger.info(
            f"ACE step {self._step_count}: "
            f"+{result.bullets_added} added, "
            f"{result.bullets_updated} updated, "
            f"-{result.bullets_pruned} pruned, "
            f"~{result.bullets_merged} merged, "
            f"size={result.playbook_size}"
        )

        return result

    def adapt_offline(
        self,
        samples: list[dict[str, Any]],
        epochs: int = 1,
    ) -> list[AdaptationResult]:
        """Run offline adaptation over a batch of samples.

        Per ACE paper (§4): Offline mode optimizes the playbook on a training set
        over multiple epochs.

        Args:
            samples: List of dicts with keys: task, execution_trace, execution_result,
                     was_successful, ground_truth (optional).
            epochs: Number of passes over the samples.

        Returns:
            List of AdaptationResult for each sample.
        """
        all_results = []
        for epoch in range(epochs):
            logger.info(f"ACE offline epoch {epoch + 1}/{epochs}, {len(samples)} samples")
            for i, sample in enumerate(samples):
                result = self.adapt_online(
                    task=sample.get("task", ""),
                    execution_trace=sample.get("execution_trace", ""),
                    execution_result=sample.get("execution_result", ""),
                    was_successful=sample.get("was_successful", True),
                    context=sample.get("context", ""),
                    ground_truth=sample.get("ground_truth"),
                )
                all_results.append(result)
        return all_results

    def warmup(self, seed_bullets: list[dict[str, str]]) -> int:
        """Initialize playbook with seed bullets (offline warmup).

        Per ACE paper ablation: +17.1% from offline warmup.

        Args:
            seed_bullets: List of dicts with 'section' and 'content' keys.

        Returns:
            Number of bullets added.
        """
        ops = [
            DeltaOperation(op_type="ADD", section=b.get("section", "OTHERS"), content=b["content"])
            for b in seed_bullets if b.get("content")
        ]
        added = self._playbook.apply_delta(ops)
        self._save_playbook()
        logger.info(f"ACE warmup: added {added} seed bullets")
        return added

    def _run_refinement(self) -> int:
        """Run grow-and-refine: find similar bullets and deduplicate.

        Per ACE paper (§3.2): Periodic deduplication prevents redundancy
        from accumulating as the playbook grows.

        Returns:
            Number of bullets merged/removed.
        """
        # Stage 1: Cheap heuristic — find candidate pairs by text similarity
        similar_pairs = self._playbook.find_similar_pairs(
            threshold=self.config.dedup_similarity_threshold
        )
        if not similar_pairs:
            return 0

        # Format for the deduplicator
        candidate_pairs = [
            (a.id, a.content, b.id, b.content, sim)
            for a, b, sim in similar_pairs[:10]  # cap to 10 pairs
        ]

        stats = self._playbook.get_stats()
        stats_str = (
            f"Total bullets: {stats.total_bullets}, "
            f"High-performing: {stats.high_performing}, "
            f"Problematic: {stats.problematic}"
        )

        # Stage 2: LLM-based judgment
        _, raw_ops = self.deduplicator.deduplicate(candidate_pairs, stats_str)

        # Apply operations
        ops = parse_delta_operations({"operations": raw_ops})
        if ops:
            return self._playbook.apply_delta(ops)
        return 0

    def get_playbook_for_prompt(self, max_chars: int = 0) -> str:
        """Get playbook text suitable for injection into a prompt.

        Args:
            max_chars: Maximum characters (0 = no limit).

        Returns:
            Playbook text, possibly truncated.
        """
        text = self._playbook.serialize()
        if not text.strip():
            return ""

        if max_chars > 0 and len(text) > max_chars:
            # Truncate but keep complete bullets
            lines = text.split("\n")
            result_lines = []
            total = 0
            for line in lines:
                if total + len(line) + 1 > max_chars:
                    break
                result_lines.append(line)
                total += len(line) + 1
            text = "\n".join(result_lines)

        return f"# PLAYBOOK_BEGIN\n\n{text}\n\n# PLAYBOOK_END"

    def reset(self) -> None:
        """Reset the playbook to empty."""
        self._playbook = create_empty_playbook()
        self._step_count = 0
        self._save_playbook()
