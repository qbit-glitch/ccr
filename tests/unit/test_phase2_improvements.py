"""Tests for Phase 2 improvements: shared JSON extraction, token cache, config constants."""

import json
import pytest

from ccr.utils.parsing import extract_json_from_llm, extract_json_string
from ccr.utils.tokens import _count_tokens_cached, count_tokens


# --- Shared JSON extraction ---


class TestExtractJsonFromLlm:
    """Tests for the unified extract_json_from_llm function."""

    def test_direct_json(self):
        result = extract_json_from_llm('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_code_block(self):
        text = 'Here is the result:\n```json\n{"tier": "simple"}\n```\nDone.'
        result = extract_json_from_llm(text)
        assert result == {"tier": "simple"}

    def test_json_in_bare_code_block(self):
        text = 'Result:\n```\n{"answer": 42}\n```'
        result = extract_json_from_llm(text)
        assert result == {"answer": 42}

    def test_json_with_surrounding_text(self):
        text = 'The analysis shows:\n\n{"reasoning": "test", "operations": []}\n\nEnd.'
        result = extract_json_from_llm(text)
        assert result == {"reasoning": "test", "operations": []}

    def test_nested_json(self):
        text = 'Output: {"data": {"nested": {"deep": true}}}'
        result = extract_json_from_llm(text)
        assert result == {"data": {"nested": {"deep": True}}}

    def test_json_with_array(self):
        result = extract_json_from_llm('{"items": [1, 2, 3]}')
        assert result == {"items": [1, 2, 3]}

    def test_no_json_returns_none(self):
        assert extract_json_from_llm("just plain text") is None

    def test_empty_string_returns_none(self):
        assert extract_json_from_llm("") is None
        assert extract_json_from_llm("   ") is None

    def test_none_input(self):
        # Should handle None-like empty input gracefully
        assert extract_json_from_llm("") is None

    def test_malformed_json_tries_brace_scan(self):
        # First brace block is invalid, second is valid
        text = 'bad: {invalid} ok: {"valid": true}'
        result = extract_json_from_llm(text)
        assert result == {"valid": True}

    def test_json_with_whitespace(self):
        text = '\n\n  {"key": "value"}  \n\n'
        result = extract_json_from_llm(text)
        assert result == {"key": "value"}

    def test_case_insensitive_code_block(self):
        text = '```JSON\n{"result": "ok"}\n```'
        result = extract_json_from_llm(text)
        assert result == {"result": "ok"}


class TestExtractJsonString:
    """Tests for extract_json_string (raw string extraction)."""

    def test_code_block(self):
        text = '```json\n[{"path": "a.py"}]\n```'
        result = extract_json_string(text)
        assert result == '[{"path": "a.py"}]'

    def test_raw_json(self):
        text = 'result: {"tier": "complex"} end'
        result = extract_json_string(text)
        assert '{"tier": "complex"}' in result

    def test_fallback_to_strip(self):
        result = extract_json_string("just text")
        assert result == "just text"

    def test_empty(self):
        assert extract_json_string("") == ""


class TestJsonExtractionBackwardsCompat:
    """Verify the shared functions work for all previous callers."""

    def test_ace_agents_pattern(self):
        """ACE agents expect dict | None return."""
        response = '{"reasoning": "test", "bullet_tags": [], "key_insight": "N/A"}'
        parsed = extract_json_from_llm(response)
        assert parsed is not None
        assert parsed["key_insight"] == "N/A"
        assert isinstance(parsed["bullet_tags"], list)

    def test_router_pattern(self):
        """Router does json.loads(extract_json_string(...))."""
        response = '```json\n{"tier": "simple", "confidence": 0.9}\n```'
        data = json.loads(extract_json_string(response))
        assert data["tier"] == "simple"

    def test_packer_pattern(self):
        """Packer does json.loads(extract_json_string(...))."""
        response = '```json\n{"symbols": ["foo"], "keywords": ["bar"]}\n```'
        data = json.loads(extract_json_string(response))
        assert data["symbols"] == ["foo"]


# --- Token counting LRU cache ---


class TestTokenCountingCache:
    """Tests for LRU-cached token counting."""

    def test_count_tokens_returns_int(self):
        result = count_tokens("hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_same_input_returns_same_result(self):
        text = "The quick brown fox jumps over the lazy dog."
        a = count_tokens(text)
        b = count_tokens(text)
        assert a == b

    def test_cache_is_used(self):
        """LRU cache should have hits after repeated calls."""
        _count_tokens_cached.cache_clear()
        text = "cache test string 12345"
        count_tokens(text)
        count_tokens(text)
        count_tokens(text)
        info = _count_tokens_cached.cache_info()
        assert info.hits >= 2

    def test_different_inputs_different_results(self):
        a = count_tokens("short")
        b = count_tokens("a much longer string with many more tokens in it")
        assert a != b

    def test_list_input_not_cached_directly(self):
        """List input is flattened to string, then cached."""
        msgs = [{"role": "user", "content": "hello"}]
        result = count_tokens(msgs)
        assert isinstance(result, int)
        assert result > 0

    def test_cache_clear(self):
        _count_tokens_cached.cache_clear()
        info = _count_tokens_cached.cache_info()
        assert info.hits == 0
        assert info.misses == 0


# --- Magic numbers extracted to config ---


class TestConfigConstants:
    """Verify magic numbers are now configurable."""

    def test_pre_compact_threshold_default(self):
        from ccr.core.types import CCREngineConfig
        config = CCREngineConfig()
        assert config.pre_compact_char_threshold == 50000

    def test_pre_compact_threshold_custom(self):
        from ccr.core.types import CCREngineConfig
        config = CCREngineConfig(pre_compact_char_threshold=100000)
        assert config.pre_compact_char_threshold == 100000

    def test_max_search_candidates_default(self):
        from ccr.core.types import CCREngineConfig
        config = CCREngineConfig()
        assert config.max_search_candidates == 50

    def test_min_relevance_score_default(self):
        from ccr.core.types import CCREngineConfig
        config = CCREngineConfig()
        assert config.min_relevance_score == 0.3

    def test_packer_receives_config_values(self):
        """Packer constructor accepts the new parameters."""
        from unittest.mock import MagicMock
        from ccr.context.packer import ContextPacker

        packer = ContextPacker(
            repo_index=MagicMock(),
            sub_client=MagicMock(),
            token_budget=8000,
            max_search_candidates=100,
            min_relevance_score=0.5,
        )
        assert packer.max_search_candidates == 100
        assert packer.min_relevance_score == 0.5
