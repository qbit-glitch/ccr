"""Task classifier and model router.

Classifies incoming requests by complexity and decides routing:
- TRIVIAL → cheap model direct
- SIMPLE → cheap model with memory context
- MODERATE → Claude with context pack
- COMPLEX → Claude with full context pack + deep memory
"""

from __future__ import annotations

import json
import logging
import re

from ccr.context.packer import ContextPacker
from ccr.context.prompts import TASK_CLASSIFICATION_SYSTEM
from ccr.core.exceptions import ModelError, PackingError
from ccr.core.memory import MemoryManager
from ccr.utils.parsing import extract_json_string
from ccr.core.types import (
    CCRRequest,
    ComplexityTier,
    ContextPack,
    RouteDecision,
    RouterConfig,
    TaskClassification,
)
from ccr.models.base import BaseLMClient
from ccr.utils.tokens import count_tokens

logger = logging.getLogger(__name__)


class TaskRouter:
    """Classifies tasks and makes routing decisions.

    Heuristic-first: catches 60-70% without a model call.
    Falls back to cheap LLM classification for ambiguous cases.
    """

    def __init__(
        self,
        classifier_client: BaseLMClient,
        context_packer: ContextPacker,
        memory: MemoryManager,
        config: RouterConfig | None = None,
    ):
        self.classifier = classifier_client
        self.packer = context_packer
        self.memory = memory
        self.config = config or RouterConfig()

    def classify(self, request: CCRRequest) -> TaskClassification:
        """Classify request complexity. Heuristic first, LLM fallback."""
        user_msg = request.last_user_message
        token_count = count_tokens(request.original_messages)

        # Fast path: heuristic classification
        tier = self._heuristic_classify(user_msg, token_count)
        if tier is not None:
            return TaskClassification(
                tier=tier,
                confidence=0.8,
                reasoning="heuristic",
                estimated_tokens=token_count,
            )

        # Slow path: LLM classification
        return self._llm_classify(user_msg, token_count)

    def route(self, request: CCRRequest, classification: TaskClassification) -> RouteDecision:
        """Decide where to route the request based on classification."""
        tier = classification.tier

        if tier == ComplexityTier.TRIVIAL:
            return RouteDecision(
                target="qwen_direct",
                context_pack=None,
                memory_context_level=0,
                should_autocommit=False,
            )

        if tier == ComplexityTier.SIMPLE:
            return RouteDecision(
                target="qwen_with_context",
                context_pack=None,
                memory_context_level=1,
                should_autocommit=False,
            )

        # MODERATE or COMPLEX: build context pack
        memory_level = 2 if tier == ComplexityTier.MODERATE else 3
        memory_ctx = self.memory.get_context(level=memory_level)

        try:
            pack = self.packer.pack(
                task=request.last_user_message,
                memory_context=memory_ctx,
            )
        except (PackingError, ModelError) as e:
            logger.warning(f"Context packing failed: {e}")
            pack = None
        except Exception as e:
            logger.warning(f"Context packing failed (unexpected): {e}")
            pack = None

        target = "claude_with_pack" if pack else "claude_direct"
        should_commit = tier.value >= self.config.auto_commit_threshold.value

        # Use RLM for COMPLEX tasks (iterative code-based reasoning)
        use_rlm = tier == ComplexityTier.COMPLEX

        return RouteDecision(
            target=target,
            context_pack=pack,
            memory_context_level=memory_level,
            should_autocommit=should_commit,
            use_rlm=use_rlm,
        )

    def _heuristic_classify(self, msg: str, token_count: int) -> ComplexityTier | None:
        """Fast heuristic classification. Returns None if ambiguous.

        Order matters:
        1. COMPLEX keywords (architecture, refactor, ...) — highest priority
        2. COMPLEX patterns (understand the codebase, ...) — exploration prompts
        3. TRIVIAL (very short, no code/file refs) — before SIMPLE to avoid
           greedy pattern matches on short questions like "what is X?"
        4. SIMPLE (action verbs, question words) — only for non-trivial length
        5. MODERATE (action verbs + longer context suggesting multi-step work)
        6. Ambiguous → LLM fallback
        """
        msg_lower = msg.lower()

        # COMPLEX: check keywords first (takes priority over length)
        for keyword in self.config.complex_keywords:
            if keyword.lower() in msg_lower:
                return ComplexityTier.COMPLEX

        # COMPLEX: check exploration patterns (short prompts that need deep work)
        for pat in self.config.complex_patterns:
            if re.search(pat, msg_lower):
                return ComplexityTier.COMPLEX

        # TRIVIAL: very short, no code, no file references — check BEFORE simple
        # patterns to avoid "what is X?" being classified as SIMPLE
        if token_count < self.config.trivial_token_threshold:
            has_code = "```" in msg
            has_file_ref = bool(re.search(r"\.\w{2,4}\b", msg))
            if not has_code and not has_file_ref:
                return ComplexityTier.TRIVIAL

        # Action verb patterns — SIMPLE for short, MODERATE for longer prompts
        # that likely involve multi-step or multi-file work
        action_patterns = [
            r"^\s*(fix|add|remove|rename|update|change)\s",
        ]
        question_patterns = [
            r"^(what|how|why|where|when)\s",
            r"^\s*explain\s",
        ]

        is_action = any(re.match(p, msg_lower) for p in action_patterns)
        is_question = any(re.match(p, msg_lower) for p in question_patterns)

        if is_question:
            return ComplexityTier.SIMPLE

        if is_action:
            # Short action prompts are SIMPLE ("fix the typo on line 12")
            # Longer ones are MODERATE ("add a new tool that... follow patterns... update schema...")
            if token_count < self.config.simple_token_threshold:
                return ComplexityTier.SIMPLE
            return ComplexityTier.MODERATE

        # Ambiguous — let LLM decide
        return None

    def _llm_classify(self, msg: str, token_count: int) -> TaskClassification:
        """Use cheap LLM to classify when heuristics are ambiguous."""
        try:
            response = self.classifier.completion(
                messages=[
                    {"role": "system", "content": TASK_CLASSIFICATION_SYSTEM},
                    {"role": "user", "content": msg[:2000]},  # cap input
                ],
                max_tokens=200,
            )
            data = json.loads(extract_json_string(response))
            tier_str = data.get("tier", "moderate")
            return TaskClassification(
                tier=ComplexityTier(tier_str),
                confidence=float(data.get("confidence", 0.7)),
                reasoning=data.get("reasoning", "llm_classified"),
                estimated_tokens=token_count,
            )
        except (json.JSONDecodeError, ValueError, Exception) as e:
            logger.warning(f"LLM classification failed: {e}")
            return TaskClassification(
                tier=ComplexityTier.MODERATE,
                confidence=0.5,
                reasoning=f"fallback (classification error: {e})",
                estimated_tokens=token_count,
            )

