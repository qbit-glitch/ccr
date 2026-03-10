"""Tests for ACE Engine (online adaptation orchestrator)."""

import json
import os
import tempfile

import pytest

from ccr.ace.engine import ACEConfig, ACEEngine, AdaptationResult
from ccr.ace.playbook import Playbook


class MockSubClient:
    """Mock sub-model for ACE engine tests."""

    def __init__(self, reflector_response=None, curator_response=None):
        self._call_idx = 0
        self._reflector_response = reflector_response or json.dumps({
            "reasoning": "Analysis of the trace",
            "error_identification": "No major errors",
            "root_cause_analysis": "N/A",
            "correct_approach": "Current approach works",
            "key_insight": "N/A",
            "bullet_tags": [],
        })
        self._curator_response = curator_response or json.dumps({
            "reasoning": "Adding new insight",
            "operations": [
                {"type": "ADD", "section": "STRATEGIES & INSIGHTS", "content": "Always validate inputs before processing"},
            ],
        })

    def completion(self, messages, **kwargs):
        self._call_idx += 1
        # First call is reflector, second is curator
        if self._call_idx % 2 == 1:
            return self._reflector_response
        return self._curator_response


class TestACEEngineInit:
    def test_init_creates_empty_playbook(self):
        client = MockSubClient()
        engine = ACEEngine(client)
        assert len(engine.playbook.bullets) == 0
        assert len(engine.playbook.sections) >= 6

    def test_init_loads_existing_playbook(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("## STRATEGIES & INSIGHTS\n[str-00001] helpful=3 harmful=0 :: Existing strategy\n")
            path = f.name

        try:
            client = MockSubClient()
            engine = ACEEngine(client, playbook_path=path)
            assert len(engine.playbook.bullets) == 1
            assert engine.playbook.get_bullet("str-00001") is not None
        finally:
            os.unlink(path)

    def test_init_with_config(self):
        config = ACEConfig(
            curator_frequency=5,
            max_reflection_rounds=2,
            playbook_token_budget=50000,
        )
        client = MockSubClient()
        engine = ACEEngine(client, config=config)
        assert engine.config.curator_frequency == 5
        assert engine.config.max_reflection_rounds == 2


class TestACEEngineAdaptation:
    def test_adapt_online_basic(self):
        client = MockSubClient()
        engine = ACEEngine(client)

        result = engine.adapt_online(
            task="Fix the authentication bug",
            execution_trace="I looked at auth.py and found the issue...",
            execution_result="Fixed the bug",
            was_successful=True,
        )
        assert isinstance(result, AdaptationResult)
        assert result.reflection_ran
        assert result.was_successful

    def test_adapt_online_adds_bullets(self):
        client = MockSubClient()
        engine = ACEEngine(client)

        result = engine.adapt_online(
            task="Parse the data",
            execution_trace="trace...",
            execution_result="done",
            was_successful=True,
        )
        # Curator should have added a bullet
        assert result.bullets_added >= 1
        assert len(engine.playbook.bullets) >= 1

    def test_adapt_online_disabled(self):
        config = ACEConfig(enabled=False)
        client = MockSubClient()
        engine = ACEEngine(client, config=config)

        result = engine.adapt_online(
            task="test",
            execution_trace="trace",
            execution_result="result",
            was_successful=True,
        )
        assert not result.reflection_ran
        assert not result.curator_ran
        assert len(engine.playbook.bullets) == 0

    def test_adapt_online_respects_curator_frequency(self):
        config = ACEConfig(curator_frequency=3)
        client = MockSubClient()
        engine = ACEEngine(client, config=config)

        # Steps 1 and 2: curator should NOT run
        for _ in range(2):
            result = engine.adapt_online("task", "trace", "result", True)
            assert not result.curator_ran

        # Step 3: curator SHOULD run
        result = engine.adapt_online("task", "trace", "result", True)
        assert result.curator_ran

    def test_adapt_online_saves_playbook(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name

        try:
            client = MockSubClient()
            engine = ACEEngine(client, playbook_path=path)

            engine.adapt_online("task", "trace", "result", True)

            # Verify file was written
            with open(path) as f:
                content = f.read()
            assert "STRATEGIES" in content
        finally:
            os.unlink(path)

    def test_adapt_online_accumulates_bullets(self):
        client = MockSubClient()
        engine = ACEEngine(client)

        for i in range(5):
            engine.adapt_online(f"Task {i}", f"trace {i}", f"result {i}", True)

        assert len(engine.playbook.bullets) >= 5

    def test_adapt_online_with_ground_truth(self):
        client = MockSubClient()
        engine = ACEEngine(client)

        result = engine.adapt_online(
            task="What is 2+2?",
            execution_trace="I computed 2+2...",
            execution_result="5",
            was_successful=False,
            ground_truth="4",
        )
        assert result.reflection_ran

    def test_adapt_online_handles_reflector_failure(self):
        """When reflector fails, engine should not crash."""
        class FailReflectorClient:
            _call_idx = 0
            def completion(self, messages, **kw):
                self._call_idx += 1
                raise RuntimeError("reflector failed")

        engine = ACEEngine(FailReflectorClient())
        result = engine.adapt_online("task", "trace", "result", True)
        # Should not crash — reflector error is caught internally
        assert isinstance(result, AdaptationResult)


class TestACEEnginePlaybook:
    def test_playbook_text_property(self):
        client = MockSubClient()
        engine = ACEEngine(client)
        text = engine.playbook_text
        assert "## STRATEGIES & INSIGHTS" in text

    def test_get_playbook_for_prompt_empty(self):
        client = MockSubClient()
        engine = ACEEngine(client)
        # Empty playbook should return empty string
        prompt = engine.get_playbook_for_prompt()
        # Even empty playbook has section headers
        assert "PLAYBOOK_BEGIN" in prompt

    def test_get_playbook_for_prompt_with_content(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("## STRATEGIES & INSIGHTS\n[str-00001] helpful=3 harmful=0 :: Test strategy\n")
            path = f.name

        try:
            client = MockSubClient()
            engine = ACEEngine(client, playbook_path=path)
            prompt = engine.get_playbook_for_prompt()
            assert "PLAYBOOK_BEGIN" in prompt
            assert "PLAYBOOK_END" in prompt
            assert "Test strategy" in prompt
        finally:
            os.unlink(path)

    def test_get_playbook_for_prompt_max_chars(self):
        client = MockSubClient()
        engine = ACEEngine(client)

        # Add many bullets
        for i in range(20):
            engine.adapt_online(f"Task {i}", "trace", "result", True)

        full = engine.get_playbook_for_prompt()
        truncated = engine.get_playbook_for_prompt(max_chars=200)
        assert len(truncated) <= len(full)

    def test_reset(self):
        client = MockSubClient()
        engine = ACEEngine(client)

        # Add some bullets
        engine.adapt_online("task", "trace", "result", True)
        assert len(engine.playbook.bullets) > 0

        # Reset
        engine.reset()
        assert len(engine.playbook.bullets) == 0


class TestACEEnginePruning:
    def test_prune_problematic_during_adaptation(self):
        """Bullets with high harmful counts get pruned."""
        # Start with a playbook that has a problematic bullet
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(
                "## STRATEGIES & INSIGHTS\n"
                "[str-00001] helpful=0 harmful=5 :: Bad advice that keeps failing\n"
                "[str-00002] helpful=10 harmful=0 :: Great advice\n"
            )
            path = f.name

        try:
            config = ACEConfig(prune_min_harmful=3)
            client = MockSubClient()
            engine = ACEEngine(client, config=config, playbook_path=path)

            result = engine.adapt_online("task", "trace", "result", True)
            assert result.bullets_pruned >= 1
            assert engine.playbook.get_bullet("str-00001") is None
            assert engine.playbook.get_bullet("str-00002") is not None
        finally:
            os.unlink(path)

    def test_enforce_budget_during_adaptation(self):
        """Playbook is trimmed when it exceeds token budget."""
        config = ACEConfig(playbook_token_budget=500)
        client = MockSubClient()
        engine = ACEEngine(client, config=config)

        # Add many bullets to exceed budget
        for i in range(20):
            engine.adapt_online(f"Task {i}", "trace", "result", True)

        assert len(engine.playbook.serialize()) <= 500
