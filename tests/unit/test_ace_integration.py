"""Tests for ACE integration with the CCR pipeline (Phase 4 wiring)."""

import json
import os
import tempfile

import pytest

from ccr.core.types import (
    ACEConfig,
    CCREngineConfig,
    CCRRequest,
    HookEvent,
    TokenUsage,
)
from ccr.core.hooks import HookManager, create_default_hooks
from ccr.utils.parsing import build_messages_with_context


# --- build_messages_with_context with playbook ---


class TestPlaybookInjection:
    def test_playbook_injected_into_messages(self):
        messages = [{"role": "user", "content": "Fix the bug"}]
        result = build_messages_with_context(
            messages, playbook_text="## STRATEGIES\n[str-00001] helpful=3 harmful=0 :: Test"
        )
        assert "<ace_playbook>" in result[0]["content"]
        assert "STRATEGIES" in result[0]["content"]

    def test_playbook_comes_before_memory_and_context(self):
        messages = [{"role": "user", "content": "task"}]
        result = build_messages_with_context(
            messages,
            playbook_text="PLAYBOOK_DATA",
            memory_context="MEMORY_DATA",
            context_pack_text="CONTEXT_DATA",
        )
        content = result[0]["content"]
        pb_pos = content.index("PLAYBOOK_DATA")
        mem_pos = content.index("MEMORY_DATA")
        ctx_pos = content.index("CONTEXT_DATA")
        assert pb_pos < mem_pos < ctx_pos

    def test_no_playbook_returns_original(self):
        messages = [{"role": "user", "content": "hello"}]
        result = build_messages_with_context(messages)
        assert result == messages

    def test_playbook_only_no_other_context(self):
        messages = [{"role": "user", "content": "task"}]
        result = build_messages_with_context(messages, playbook_text="my playbook")
        assert "ace_playbook" in result[0]["content"]
        assert "session_memory" not in result[0]["content"]

    def test_playbook_with_list_content(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "the task"}]}
        ]
        result = build_messages_with_context(messages, playbook_text="PB")
        content = result[0]["content"]
        assert isinstance(content, list)
        assert "ace_playbook" in content[0]["text"]


# --- HookEvent.POST_TASK_COMPLETE ---


class TestPostTaskCompleteHook:
    def test_hook_event_exists(self):
        assert HookEvent.POST_TASK_COMPLETE == "PostTaskComplete"

    def test_hook_fires(self):
        hooks = HookManager()
        fired = []
        hooks.register(
            HookEvent.POST_TASK_COMPLETE,
            lambda ctx: fired.append(ctx),
        )
        hooks.fire(HookEvent.POST_TASK_COMPLETE, {
            "task": "Fix bug",
            "bullets_added": 1,
            "bullets_pruned": 0,
            "playbook_size": 500,
        })
        assert len(fired) == 1
        assert fired[0]["bullets_added"] == 1

    def test_default_hooks_include_post_task(self):
        """Default hooks should register a POST_TASK_COMPLETE handler."""
        from unittest.mock import MagicMock
        mock_memory = MagicMock()
        hooks = create_default_hooks(mock_memory)
        assert hooks.has_handlers(HookEvent.POST_TASK_COMPLETE)


# --- ACEConfig in CCREngineConfig ---


class TestACEConfigWiring:
    def test_default_ace_config(self):
        config = CCREngineConfig()
        assert config.ace.enabled is True
        assert config.ace.playbook_token_budget == 80000
        assert config.ace.curator_frequency == 1

    def test_ace_config_override(self):
        config = CCREngineConfig(
            ace=ACEConfig(enabled=False, curator_frequency=5)
        )
        assert config.ace.enabled is False
        assert config.ace.curator_frequency == 5


# --- RLM orchestrator with playbook ---


class TestRLMPlaybookIntegration:
    def test_rlm_accepts_playbook_text(self):
        from ccr.rlm.orchestrator import CCRRlm

        class FakeClient:
            def completion(self, messages, **kw):
                return "FINAL_VAR('result')"
            def get_last_usage(self):
                return TokenUsage()

        rlm = CCRRlm(
            sub_client=FakeClient(),
            playbook_text="## STRATEGIES\n[str-00001] helpful=1 harmful=0 :: Test tip",
        )
        assert rlm.playbook_text is not None
        assert "STRATEGIES" in rlm.playbook_text

    def test_rlm_metadata_mentions_playbook(self):
        from ccr.rlm.orchestrator import CCRRlm

        class FakeClient:
            def completion(self, messages, **kw):
                return "FINAL_VAR('x')"
            def get_last_usage(self):
                return TokenUsage()

        rlm = CCRRlm(
            sub_client=FakeClient(),
            playbook_text="my playbook content",
        )
        metadata = rlm._build_prompt_metadata("test prompt")
        assert "playbook" in metadata.lower()

    def test_rlm_without_playbook(self):
        from ccr.rlm.orchestrator import CCRRlm

        class FakeClient:
            def completion(self, messages, **kw):
                return "FINAL_VAR('x')"
            def get_last_usage(self):
                return TokenUsage()

        rlm = CCRRlm(sub_client=FakeClient())
        metadata = rlm._build_prompt_metadata("test prompt")
        assert "playbook" not in metadata.lower()


# --- ACE engine in CCR engine ---


class TestACEEngineInCCREngine:
    def test_ace_engine_attribute_exists(self):
        """CCREngine should have an _ace_engine attribute after init."""
        from ccr.core.engine import CCREngine

        config = CCREngineConfig(
            ace=ACEConfig(enabled=False),
            anthropic_api_key="test-key",
        )
        engine = CCREngine("/tmp/test-project", config)
        assert hasattr(engine, "_ace_engine")
        assert engine._ace_engine is None  # not initialized yet

    def test_ace_disabled_no_engine(self):
        """When ACE is disabled, _ace_engine should remain None after init."""
        from ccr.core.engine import CCREngine
        from unittest.mock import patch, MagicMock

        config = CCREngineConfig(
            ace=ACEConfig(enabled=False),
            anthropic_api_key="test-key",
        )
        engine = CCREngine("/tmp/test-project", config)

        # Mock all the dependencies to avoid real API calls
        with patch.object(engine, '_build_index'), \
             patch('ccr.core.engine.ClaudeClient'), \
             patch('ccr.core.engine.OpenAICompatClient'):
            engine.memory = MagicMock()
            engine.memory.ensure_structure.return_value = False
            engine.memory.get_session_context.return_value = ""
            engine.initialize()
            assert engine._ace_engine is None
