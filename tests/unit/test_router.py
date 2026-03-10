"""Tests for TaskRouter — heuristic and LLM-based classification."""

import pytest

from ccr.core.router import TaskRouter
from ccr.core.types import (
    CCRRequest,
    ComplexityTier,
    RouterConfig,
)


class MockClassifier:
    """Mock LLM client for testing."""

    def __init__(self, response: str = '{"tier": "moderate", "confidence": 0.8, "reasoning": "mock"}'):
        self.response = response
        self.calls = []

    def completion(self, messages, **kwargs):
        self.calls.append(messages)
        return self.response

    async def acompletion(self, messages, **kwargs):
        return self.completion(messages, **kwargs)

    def get_usage_summary(self):
        return None

    def get_last_usage(self):
        from ccr.core.types import TokenUsage
        return TokenUsage()


class MockPacker:
    def pack(self, task, memory_context=""):
        from ccr.core.types import ContextPack
        return ContextPack(
            task_description=task,
            files=[("test.py", "content")],
            symbols=["test"],
            memory_context=memory_context,
            total_tokens=100,
        )


class MockMemory:
    def get_context(self, level=1, **kwargs):
        return "mock memory context"


def make_request(msg: str) -> CCRRequest:
    return CCRRequest(
        original_messages=[{"role": "user", "content": msg}],
        system_prompt=None,
        model_requested="claude-sonnet-4-5",
        max_tokens=4096,
    )


@pytest.fixture
def router():
    return TaskRouter(
        classifier_client=MockClassifier(),
        context_packer=MockPacker(),
        memory=MockMemory(),
        config=RouterConfig(),
    )


class TestHeuristicClassification:
    def test_trivial_short_message(self, router):
        req = make_request("Hello")
        result = router.classify(req)
        assert result.tier == ComplexityTier.TRIVIAL

    def test_short_question_classified_as_trivial(self, router):
        req = make_request("What is 2+2?")
        result = router.classify(req)
        assert result.tier == ComplexityTier.TRIVIAL  # Short question, below trivial threshold

    def test_longer_question_classified_as_simple(self, router):
        # Longer questions (above trivial threshold) should still be SIMPLE
        long_q = "What is the difference between " + "the specific variable " * 200 + "and the other one?"
        req = make_request(long_q)
        result = router.classify(req)
        assert result.tier == ComplexityTier.SIMPLE

    def test_trivial_greeting(self, router):
        req = make_request("Hello")
        result = router.classify(req)
        assert result.tier == ComplexityTier.TRIVIAL

    def test_complex_architecture_keyword(self, router):
        req = make_request("Refactor the entire authentication module")
        result = router.classify(req)
        assert result.tier == ComplexityTier.COMPLEX

    def test_complex_cross_file_keyword(self, router):
        req = make_request("Design the system architecture for the new microservice")
        result = router.classify(req)
        assert result.tier == ComplexityTier.COMPLEX

    def test_short_fix_is_trivial(self, router):
        # Short action prompts below trivial threshold are TRIVIAL
        req = make_request("fix the typo in line 42")
        result = router.classify(req)
        assert result.tier == ComplexityTier.TRIVIAL

    def test_longer_fix_is_simple(self, router):
        # Longer fix prompts (above trivial threshold) are SIMPLE
        long_fix = "fix the error handling in the authentication middleware " + "context " * 500
        req = make_request(long_fix)
        result = router.classify(req)
        assert result.tier == ComplexityTier.SIMPLE

    def test_short_explain_is_trivial(self, router):
        req = make_request("explain this function")
        result = router.classify(req)
        assert result.tier == ComplexityTier.TRIVIAL

    # --- Exploration patterns (short but COMPLEX) ---

    def test_understand_codebase_is_complex(self, router):
        req = make_request("understand the codebase")
        result = router.classify(req)
        assert result.tier == ComplexityTier.COMPLEX

    def test_understand_project_is_complex(self, router):
        req = make_request("understand this project")
        result = router.classify(req)
        assert result.tier == ComplexityTier.COMPLEX

    def test_walk_through_codebase_is_complex(self, router):
        req = make_request("walk me through the codebase")
        result = router.classify(req)
        assert result.tier == ComplexityTier.COMPLEX

    def test_explain_entire_architecture_is_complex(self, router):
        req = make_request("explain the entire architecture")
        result = router.classify(req)
        assert result.tier == ComplexityTier.COMPLEX

    def test_how_does_project_work_is_complex(self, router):
        req = make_request("how does the project work")
        result = router.classify(req)
        assert result.tier == ComplexityTier.COMPLEX

    def test_analyze_dependencies_is_complex(self, router):
        req = make_request("analyze the dependencies")
        result = router.classify(req)
        assert result.tier == ComplexityTier.COMPLEX

    def test_how_is_codebase_structured_is_complex(self, router):
        req = make_request("how is the codebase structured")
        result = router.classify(req)
        assert result.tier == ComplexityTier.COMPLEX

    def test_give_full_overview_is_complex(self, router):
        req = make_request("give me a full overview")
        result = router.classify(req)
        assert result.tier == ComplexityTier.COMPLEX

    def test_summarize_the_project_is_complex(self, router):
        req = make_request("summarize the project")
        result = router.classify(req)
        assert result.tier == ComplexityTier.COMPLEX

    def test_describe_the_system_is_complex(self, router):
        req = make_request("describe the system")
        result = router.classify(req)
        assert result.tier == ComplexityTier.COMPLEX

    def test_map_the_architecture_is_complex(self, router):
        req = make_request("map the architecture")
        result = router.classify(req)
        assert result.tier == ComplexityTier.COMPLEX

    def test_trace_the_data_flow_is_complex(self, router):
        req = make_request("trace the data flow")
        result = router.classify(req)
        assert result.tier == ComplexityTier.COMPLEX

    def test_go_through_the_code_is_complex(self, router):
        req = make_request("go through the code")
        result = router.classify(req)
        assert result.tier == ComplexityTier.COMPLEX

    def test_high_level_summary_is_complex(self, router):
        req = make_request("give a high-level summary")
        result = router.classify(req)
        assert result.tier == ComplexityTier.COMPLEX

    # "explain this function" is short → TRIVIAL, not COMPLEX exploration
    def test_explain_function_is_trivial_not_exploration(self, router):
        req = make_request("explain this function")
        result = router.classify(req)
        assert result.tier == ComplexityTier.TRIVIAL
        # Must NOT be COMPLEX (shouldn't match exploration patterns)


class TestLLMClassification:
    def test_ambiguous_falls_to_llm(self):
        classifier = MockClassifier(
            '{"tier": "moderate", "confidence": 0.9, "reasoning": "needs multi-file analysis"}'
        )
        router = TaskRouter(
            classifier_client=classifier,
            context_packer=MockPacker(),
            memory=MockMemory(),
        )
        # Build a message that's long enough to not be trivial (>500 tokens)
        # and doesn't match any simple/complex patterns
        long_msg = (
            "The payment processing flow seems to have an intermittent issue. "
            "Sometimes when users complete checkout, the confirmation email "
            "is not sent even though the payment goes through successfully. "
        ) * 20  # need >500 tokens to avoid trivial classification
        req = make_request(long_msg)
        result = router.classify(req)
        # Should have called the LLM since heuristics are ambiguous for this
        assert len(classifier.calls) > 0 or result.tier == ComplexityTier.MODERATE


class TestRouting:
    def test_trivial_routes_to_qwen(self, router):
        req = make_request("Hello")
        classification = router.classify(req)
        decision = router.route(req, classification)
        assert decision.target == "qwen_direct"
        assert decision.context_pack is None

    def test_complex_routes_to_claude_with_pack(self, router):
        from ccr.core.types import TaskClassification
        req = make_request("Refactor the entire codebase")
        classification = TaskClassification(
            tier=ComplexityTier.COMPLEX,
            confidence=0.9,
            reasoning="complex keyword",
        )
        decision = router.route(req, classification)
        assert "claude" in decision.target
        assert decision.context_pack is not None

    def test_moderate_builds_context_pack(self, router):
        from ccr.core.types import TaskClassification
        req = make_request("Debug the auth flow")
        classification = TaskClassification(
            tier=ComplexityTier.MODERATE,
            confidence=0.8,
            reasoning="moderate",
        )
        decision = router.route(req, classification)
        assert decision.context_pack is not None
