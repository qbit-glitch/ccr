"""ACE agents — Generator, Reflector, and Curator.

Per ACE paper (§3): Three specialized roles that work together to
continuously improve contexts through generation, reflection, and curation.
"""

from __future__ import annotations

import logging
from typing import Any

from ccr.core.exceptions import ModelError
from ccr.utils.parsing import extract_json_from_llm
from ccr.ace.prompts import (
    CURATOR_SYSTEM,
    CURATOR_USER,
    DEDUPLICATOR_SYSTEM,
    DEDUPLICATOR_USER,
    GENERATOR_SYSTEM,
    GENERATOR_USER,
    REFLECTOR_SYSTEM_NO_GT,
    REFLECTOR_SYSTEM_WITH_GT,
    REFLECTOR_USER_NO_GT,
    REFLECTOR_USER_WITH_GT,
)

logger = logging.getLogger(__name__)


class ACEGenerator:
    """Produces answers using playbook knowledge.

    The Generator takes a task + current playbook and produces a reasoning
    trace with the answer. It reports which bullet IDs it used, enabling
    the Reflector to tag them.
    """

    def __init__(self, sub_client: Any, max_tokens: int = 4096):
        self.sub_client = sub_client
        self.max_tokens = max_tokens

    def generate(
        self,
        question: str,
        playbook: str,
        context: str = "",
        reflection: str = "(empty)",
    ) -> tuple[str, list[str], str]:
        """Generate an answer using the playbook.

        Args:
            question: The task/question to answer.
            playbook: Current playbook text.
            context: Additional context (e.g., code, files).
            reflection: Reflection from previous attempt, or "(empty)".

        Returns:
            Tuple of (raw_response, bullet_ids_used, final_answer).
        """
        system = GENERATOR_SYSTEM.format(playbook=playbook, reflection=reflection)
        user = GENERATOR_USER.format(question=question, context=context)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            response = self.sub_client.completion(messages)
        except ModelError as e:
            logger.error(f"Generator LLM call failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Generator unexpected error: {e}")
            return str(e), [], ""

        # Parse response
        parsed = extract_json_from_llm(response)
        if parsed:
            bullet_ids = parsed.get("bullet_ids", [])
            final_answer = parsed.get("final_answer", "")
            return response, bullet_ids, final_answer

        # Fallback: return raw response as answer
        return response, [], response.strip()


class ACEReflector:
    """Analyzes outputs and extracts insights.

    The Reflector critiques Generator traces to extract lessons. It tags
    bullets as helpful/harmful/neutral, providing feedback that guides
    the Curator in proposing corrective updates.
    """

    def __init__(self, sub_client: Any, max_tokens: int = 4096):
        self.sub_client = sub_client
        self.max_tokens = max_tokens

    def reflect(
        self,
        question: str,
        reasoning_trace: str,
        predicted_answer: str,
        bullets_used: str,
        environment_feedback: str = "",
        ground_truth: str | None = None,
    ) -> tuple[str, list[dict[str, str]], str]:
        """Reflect on a generation attempt.

        Args:
            question: The original task.
            reasoning_trace: The Generator's full response.
            predicted_answer: The extracted answer.
            bullets_used: Formatted bullets the Generator referenced.
            environment_feedback: Execution feedback (success/error).
            ground_truth: Ground truth answer, if available.

        Returns:
            Tuple of (reflection_text, bullet_tags, key_insight).
        """
        has_gt = ground_truth is not None

        if has_gt:
            system = REFLECTOR_SYSTEM_WITH_GT
            user = REFLECTOR_USER_WITH_GT.format(
                question=question,
                reasoning_trace=reasoning_trace,
                predicted_answer=predicted_answer,
                ground_truth=ground_truth,
                environment_feedback=environment_feedback or "N/A",
                bullets_used=bullets_used,
            )
        else:
            system = REFLECTOR_SYSTEM_NO_GT
            user = REFLECTOR_USER_NO_GT.format(
                question=question,
                reasoning_trace=reasoning_trace,
                predicted_answer=predicted_answer,
                environment_feedback=environment_feedback or "N/A",
                bullets_used=bullets_used,
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            response = self.sub_client.completion(messages)
        except ModelError as e:
            logger.error(f"Reflector LLM call failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Reflector unexpected error: {e}")
            return str(e), [], ""

        # Parse response
        parsed = extract_json_from_llm(response)
        if parsed:
            bullet_tags = parsed.get("bullet_tags", [])
            key_insight = parsed.get("key_insight", "")
            return response, bullet_tags, key_insight

        return response, [], ""


class ACECurator:
    """Proposes delta operations to update the playbook.

    The Curator converts Reflector lessons into structured delta entries,
    which are merged deterministically into the playbook. Because updates
    are itemized and localized, context collapse is prevented.
    """

    def __init__(self, sub_client: Any, max_tokens: int = 4096):
        self.sub_client = sub_client
        self.max_tokens = max_tokens

    def curate(
        self,
        current_playbook: str,
        recent_reflection: str,
        question_context: str,
        playbook_stats: str,
        current_step: int = 0,
        total_samples: int = 0,
        token_budget: int = 80000,
        available_sections: str = "",
    ) -> tuple[str, list[dict[str, Any]]]:
        """Propose delta operations for the playbook.

        Args:
            current_playbook: Current playbook text.
            recent_reflection: Reflection output to learn from.
            question_context: The task context.
            playbook_stats: Stats string about current playbook.
            current_step: Current training step.
            total_samples: Total samples in dataset.
            token_budget: Max token budget for playbook.
            available_sections: Comma-separated section names.

        Returns:
            Tuple of (raw_response, operations_list).
        """
        system = CURATOR_SYSTEM.format(
            available_sections=available_sections or "STRATEGIES & INSIGHTS, CODE SNIPPETS & TEMPLATES, COMMON MISTAKES TO AVOID, PROBLEM-SOLVING HEURISTICS, CONTEXT CLUES & INDICATORS, OTHERS",
        )
        user = CURATOR_USER.format(
            token_budget=token_budget,
            current_step=current_step,
            total_samples=total_samples,
            playbook_stats=playbook_stats,
            recent_reflection=recent_reflection,
            current_playbook=current_playbook,
            question_context=question_context,
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            response = self.sub_client.completion(messages)
        except ModelError as e:
            logger.error(f"Curator LLM call failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Curator unexpected error: {e}")
            return str(e), []

        # Parse response
        parsed = extract_json_from_llm(response)
        if parsed:
            operations = parsed.get("operations", [])
            return response, operations

        return response, []


class ACEDeduplicator:
    """Identifies and merges duplicate/near-duplicate bullets.

    Per ACE paper (§3.2): Grow-and-refine periodically deduplicates the
    playbook to prevent redundancy from accumulating as bullets grow.
    Uses a two-stage approach:
    1. Cheap text-similarity heuristic to find candidate pairs
    2. LLM-based judgment to decide MERGE/REMOVE/KEEP
    """

    def __init__(self, sub_client: Any, max_tokens: int = 4096):
        self.sub_client = sub_client
        self.max_tokens = max_tokens

    def deduplicate(
        self,
        candidate_pairs: list[tuple[str, str, str, str, float]],
        playbook_stats: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Propose merge/remove operations for candidate duplicate pairs.

        Args:
            candidate_pairs: List of (id_a, content_a, id_b, content_b, similarity).
            playbook_stats: Stats string about current playbook.

        Returns:
            Tuple of (raw_response, operations_list).
        """
        if not candidate_pairs:
            return "", []

        # Format pairs for the prompt
        pairs_text = []
        for i, (id_a, content_a, id_b, content_b, sim) in enumerate(candidate_pairs[:10]):
            pairs_text.append(
                f"Pair {i+1} (similarity={sim:.2f}):\n"
                f"  [{id_a}] :: {content_a}\n"
                f"  [{id_b}] :: {content_b}"
            )
        pairs_formatted = "\n\n".join(pairs_text)

        system = DEDUPLICATOR_SYSTEM
        user = DEDUPLICATOR_USER.format(
            candidate_pairs=pairs_formatted,
            playbook_stats=playbook_stats,
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            response = self.sub_client.completion(messages)
        except ModelError as e:
            logger.error(f"Deduplicator LLM call failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Deduplicator unexpected error: {e}")
            return str(e), []

        parsed = extract_json_from_llm(response)
        if parsed:
            operations = parsed.get("operations", [])
            return response, operations

        return response, []
