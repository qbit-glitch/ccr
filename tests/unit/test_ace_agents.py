"""Tests for ACE agents (Generator, Reflector, Curator)."""

import json
import pytest

from ccr.ace.agents import ACEGenerator, ACEReflector, ACECurator
from ccr.utils.parsing import extract_json_from_llm as _extract_json


class MockSubClient:
    """Mock sub-model that returns canned responses."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self._call_idx = 0
        self._captured_messages = []

    def completion(self, messages, **kwargs):
        self._captured_messages.append(messages)
        if self._call_idx < len(self.responses):
            resp = self.responses[self._call_idx]
            self._call_idx += 1
            return resp
        return "{}"


class TestExtractJson:
    def test_direct_json(self):
        result = _extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_json_in_text(self):
        text = 'Here is the result: {"key": "value"} and more text'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_nested_json(self):
        text = '{"outer": {"inner": "value"}, "list": [1, 2]}'
        result = _extract_json(text)
        assert result["outer"]["inner"] == "value"

    def test_no_json(self):
        result = _extract_json("no json here")
        assert result is None

    def test_empty_string(self):
        result = _extract_json("")
        assert result is None


class TestACEGenerator:
    def test_generate_with_json_response(self):
        response = json.dumps({
            "reasoning": "I checked the playbook",
            "bullet_ids": ["str-00001", "code-00003"],
            "final_answer": "42",
        })
        client = MockSubClient([response])
        gen = ACEGenerator(client)

        raw, bullet_ids, answer = gen.generate(
            question="What is the answer?",
            playbook="## STRATEGIES\n[str-00001] helpful=1 harmful=0 :: test",
        )
        assert bullet_ids == ["str-00001", "code-00003"]
        assert answer == "42"

    def test_generate_with_plain_text_fallback(self):
        client = MockSubClient(["The answer is 42"])
        gen = ACEGenerator(client)

        raw, bullet_ids, answer = gen.generate(
            question="What is the answer?",
            playbook="",
        )
        assert bullet_ids == []
        assert "42" in answer

    def test_generate_passes_playbook_in_system(self):
        client = MockSubClient(['{"reasoning":"x","bullet_ids":[],"final_answer":"y"}'])
        gen = ACEGenerator(client)

        gen.generate(
            question="test",
            playbook="## MY PLAYBOOK\n[str-00001] helpful=1 harmful=0 :: important rule",
        )
        messages = client._captured_messages[0]
        system_msg = messages[0]["content"]
        assert "MY PLAYBOOK" in system_msg
        assert "important rule" in system_msg

    def test_generate_includes_reflection(self):
        client = MockSubClient(['{"reasoning":"x","bullet_ids":[],"final_answer":"y"}'])
        gen = ACEGenerator(client)

        gen.generate(
            question="test",
            playbook="",
            reflection="You made an error in step 3",
        )
        messages = client._captured_messages[0]
        system_msg = messages[0]["content"]
        assert "error in step 3" in system_msg

    def test_generate_handles_error(self):
        class FailClient:
            def completion(self, messages, **kw):
                raise RuntimeError("API down")

        gen = ACEGenerator(FailClient())
        raw, bullet_ids, answer = gen.generate("test", "")
        assert "API down" in raw
        assert bullet_ids == []


class TestACEReflector:
    def test_reflect_with_ground_truth(self):
        response = json.dumps({
            "reasoning": "The model misidentified the data type",
            "error_identification": "Wrong type conversion",
            "root_cause_analysis": "Misunderstood the API return type",
            "correct_approach": "Check API docs first",
            "key_insight": "Always verify return types",
            "bullet_tags": [
                {"id": "str-00001", "tag": "helpful"},
                {"id": "str-00002", "tag": "harmful"},
            ],
        })
        client = MockSubClient([response])
        ref = ACEReflector(client)

        raw, bullet_tags, insight = ref.reflect(
            question="Parse this data",
            reasoning_trace="I tried to parse...",
            predicted_answer="wrong",
            bullets_used="[str-00001] helpful=1 harmful=0 :: test",
            ground_truth="correct answer",
        )
        assert len(bullet_tags) == 2
        assert bullet_tags[0]["tag"] == "helpful"
        assert "verify return types" in insight

    def test_reflect_without_ground_truth(self):
        response = json.dumps({
            "reasoning": "Execution failed with error",
            "error_identification": "Runtime error",
            "root_cause_analysis": "Missing import",
            "correct_approach": "Add the import",
            "key_insight": "Check imports first",
            "bullet_tags": [],
        })
        client = MockSubClient([response])
        ref = ACEReflector(client)

        raw, bullet_tags, insight = ref.reflect(
            question="Run this code",
            reasoning_trace="import missing_module...",
            predicted_answer="error",
            bullets_used="(no bullets)",
            environment_feedback="ImportError: No module named 'missing_module'",
        )
        assert "imports" in insight.lower()

    def test_reflect_uses_correct_prompt(self):
        client = MockSubClient(['{"reasoning":"x","bullet_tags":[],"key_insight":"y"}'])
        ref = ACEReflector(client)

        # With ground truth
        ref.reflect("q", "trace", "pred", "bullets", ground_truth="answer")
        messages_gt = client._captured_messages[0]
        assert "Ground Truth" in messages_gt[1]["content"]

        # Without ground truth
        ref.reflect("q", "trace", "pred", "bullets")
        messages_no_gt = client._captured_messages[1]
        assert "Ground Truth" not in messages_no_gt[1]["content"]

    def test_reflect_handles_error(self):
        class FailClient:
            def completion(self, messages, **kw):
                raise RuntimeError("timeout")

        ref = ACEReflector(FailClient())
        raw, tags, insight = ref.reflect("q", "trace", "pred", "bullets")
        assert "timeout" in raw
        assert tags == []


class TestACECurator:
    def test_curate_returns_operations(self):
        response = json.dumps({
            "reasoning": "Need to add a new strategy",
            "operations": [
                {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Always check types"},
            ],
        })
        client = MockSubClient([response])
        cur = ACECurator(client)

        raw, ops = cur.curate(
            current_playbook="## STRATEGIES & INSIGHTS\n",
            recent_reflection="The model failed due to type errors",
            question_context="Parse financial data",
            playbook_stats="Total: 5 bullets",
        )
        assert len(ops) == 1
        assert ops[0]["type"] == "ADD"
        assert "check types" in ops[0]["content"]

    def test_curate_empty_operations(self):
        response = json.dumps({
            "reasoning": "Playbook already covers this",
            "operations": [],
        })
        client = MockSubClient([response])
        cur = ACECurator(client)

        raw, ops = cur.curate(
            current_playbook="## STRATEGIES\n[str-00001] helpful=5 harmful=0 :: Already good",
            recent_reflection="Everything worked fine",
            question_context="Simple task",
            playbook_stats="Total: 1 bullet",
        )
        assert len(ops) == 0

    def test_curate_passes_context(self):
        client = MockSubClient(['{"reasoning":"x","operations":[]}'])
        cur = ACECurator(client)

        cur.curate(
            current_playbook="## TEST\n",
            recent_reflection="reflection content",
            question_context="the task context",
            playbook_stats="stats here",
            current_step=5,
            total_samples=100,
            token_budget=50000,
        )
        messages = client._captured_messages[0]
        user_msg = messages[1]["content"]
        assert "50000" in user_msg
        assert "5" in user_msg
        assert "100" in user_msg
        assert "reflection content" in user_msg

    def test_curate_handles_error(self):
        class FailClient:
            def completion(self, messages, **kw):
                raise RuntimeError("rate limited")

        cur = ACECurator(FailClient())
        raw, ops = cur.curate("", "", "", "")
        assert "rate limited" in raw
        assert ops == []
