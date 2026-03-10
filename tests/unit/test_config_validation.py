"""Tests for configuration validation at startup."""

import pytest

from ccr.core.exceptions import ConfigError
from ccr.core.types import (
    ACEConfig,
    CCRConfig,
    CCREngineConfig,
    RLMConfig,
    RouterConfig,
)


class TestValidConfig:
    """Valid configurations should pass without errors."""

    def test_default_config_needs_api_key(self):
        """Default config is valid except for missing API key."""
        config = CCREngineConfig()
        with pytest.raises(ConfigError, match="anthropic_api_key"):
            config.validate()

    def test_minimal_valid_config(self):
        config = CCREngineConfig(anthropic_api_key="sk-ant-test123")
        warnings = config.validate()
        assert isinstance(warnings, list)

    def test_full_valid_config(self):
        config = CCREngineConfig(
            anthropic_api_key="sk-ant-api03-abc123",
            sub_model="openai/gpt-oss-20b",
            sub_model_base_url="http://localhost:8000/v1",
            gateway_port=7447,
            pack_token_budget=8000,
        )
        warnings = config.validate()
        assert len(warnings) == 0


class TestRequiredFields:
    """Required fields must be present."""

    def test_missing_api_key(self):
        config = CCREngineConfig(anthropic_api_key="")
        with pytest.raises(ConfigError, match="anthropic_api_key"):
            config.validate()

    def test_missing_sub_model(self):
        config = CCREngineConfig(anthropic_api_key="sk-ant-test", sub_model="")
        with pytest.raises(ConfigError, match="sub_model"):
            config.validate()

    def test_missing_sub_model_url(self):
        config = CCREngineConfig(anthropic_api_key="sk-ant-test", sub_model_base_url="")
        with pytest.raises(ConfigError, match="sub_model_base_url"):
            config.validate()

    def test_multiple_errors_reported(self):
        """All errors collected, not just the first."""
        config = CCREngineConfig(
            anthropic_api_key="",
            sub_model="",
            sub_model_base_url="",
        )
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        msg = str(exc_info.value)
        assert "anthropic_api_key" in msg
        assert "sub_model:" in msg
        assert "sub_model_base_url:" in msg
        assert "3 error(s)" in msg


class TestFormatChecks:
    """Format validation for strings."""

    def test_api_key_format_warning(self):
        """Non-standard API key format produces warning, not error."""
        config = CCREngineConfig(anthropic_api_key="some-random-key")
        warnings = config.validate()
        assert any("sk-ant-" in w for w in warnings)

    def test_valid_api_key_no_warning(self):
        config = CCREngineConfig(anthropic_api_key="sk-ant-api03-abc123")
        warnings = config.validate()
        assert not any("sk-ant-" in w for w in warnings)

    def test_sk_prefix_also_valid(self):
        """sk- prefix (without ant-) is also accepted."""
        config = CCREngineConfig(anthropic_api_key="sk-abc123")
        warnings = config.validate()
        assert not any("sk-ant-" in w for w in warnings)

    def test_invalid_sub_model_url(self):
        config = CCREngineConfig(
            anthropic_api_key="sk-ant-test",
            sub_model_base_url="not-a-url",
        )
        with pytest.raises(ConfigError, match="sub_model_base_url.*http"):
            config.validate()

    def test_invalid_anthropic_base_url(self):
        config = CCREngineConfig(
            anthropic_api_key="sk-ant-test",
            anthropic_real_base_url="ftp://example.com",
        )
        with pytest.raises(ConfigError, match="anthropic_real_base_url.*http"):
            config.validate()


class TestRangeChecks:
    """Numeric range validation."""

    def test_port_too_low(self):
        config = CCREngineConfig(anthropic_api_key="sk-ant-test", gateway_port=0)
        with pytest.raises(ConfigError, match="gateway_port"):
            config.validate()

    def test_port_too_high(self):
        config = CCREngineConfig(anthropic_api_key="sk-ant-test", gateway_port=70000)
        with pytest.raises(ConfigError, match="gateway_port"):
            config.validate()

    def test_valid_port_range(self):
        for port in [1, 80, 443, 7447, 8080, 65535]:
            config = CCREngineConfig(anthropic_api_key="sk-ant-test", gateway_port=port)
            config.validate()  # should not raise

    def test_pack_budget_too_small(self):
        config = CCREngineConfig(anthropic_api_key="sk-ant-test", pack_token_budget=50)
        with pytest.raises(ConfigError, match="pack_token_budget"):
            config.validate()

    def test_pack_budget_warning_very_large(self):
        config = CCREngineConfig(anthropic_api_key="sk-ant-test", pack_token_budget=300000)
        warnings = config.validate()
        assert any("pack_token_budget" in w for w in warnings)


class TestRouterConfigValidation:
    """Router-specific validation."""

    def test_negative_trivial_threshold(self):
        config = CCREngineConfig(
            anthropic_api_key="sk-ant-test",
            router=RouterConfig(trivial_token_threshold=-1),
        )
        with pytest.raises(ConfigError, match="trivial_token_threshold"):
            config.validate()

    def test_simple_less_than_trivial(self):
        config = CCREngineConfig(
            anthropic_api_key="sk-ant-test",
            router=RouterConfig(trivial_token_threshold=2000, simple_token_threshold=500),
        )
        with pytest.raises(ConfigError, match="simple_token_threshold.*trivial"):
            config.validate()

    def test_valid_threshold_ordering(self):
        config = CCREngineConfig(
            anthropic_api_key="sk-ant-test",
            router=RouterConfig(trivial_token_threshold=500, simple_token_threshold=2000),
        )
        config.validate()  # should not raise


class TestRLMConfigValidation:
    """RLM-specific validation."""

    def test_negative_depth(self):
        config = CCREngineConfig(
            anthropic_api_key="sk-ant-test",
            rlm=RLMConfig(max_depth=-1),
        )
        with pytest.raises(ConfigError, match="max_depth"):
            config.validate()

    def test_zero_iterations(self):
        config = CCREngineConfig(
            anthropic_api_key="sk-ant-test",
            rlm=RLMConfig(max_iterations=0),
        )
        with pytest.raises(ConfigError, match="max_iterations"):
            config.validate()

    def test_zero_timeout(self):
        config = CCREngineConfig(
            anthropic_api_key="sk-ant-test",
            rlm=RLMConfig(max_timeout_seconds=0),
        )
        with pytest.raises(ConfigError, match="max_timeout_seconds"):
            config.validate()

    def test_zero_consecutive_errors(self):
        config = CCREngineConfig(
            anthropic_api_key="sk-ant-test",
            rlm=RLMConfig(max_consecutive_errors=0),
        )
        with pytest.raises(ConfigError, match="max_consecutive_errors"):
            config.validate()

    def test_valid_rlm_config(self):
        config = CCREngineConfig(
            anthropic_api_key="sk-ant-test",
            rlm=RLMConfig(max_depth=0, max_iterations=1, max_timeout_seconds=0.1),
        )
        config.validate()  # should not raise


class TestACEConfigValidation:
    """ACE-specific validation."""

    def test_disabled_ace_skips_validation(self):
        """When ACE is disabled, its fields aren't checked."""
        config = CCREngineConfig(
            anthropic_api_key="sk-ant-test",
            ace=ACEConfig(enabled=False, playbook_token_budget=0, curator_frequency=0),
        )
        config.validate()  # should not raise

    def test_tiny_playbook_budget(self):
        config = CCREngineConfig(
            anthropic_api_key="sk-ant-test",
            ace=ACEConfig(playbook_token_budget=10),
        )
        with pytest.raises(ConfigError, match="playbook_token_budget"):
            config.validate()

    def test_zero_curator_frequency(self):
        config = CCREngineConfig(
            anthropic_api_key="sk-ant-test",
            ace=ACEConfig(curator_frequency=0),
        )
        with pytest.raises(ConfigError, match="curator_frequency"):
            config.validate()

    def test_zero_reflection_rounds(self):
        config = CCREngineConfig(
            anthropic_api_key="sk-ant-test",
            ace=ACEConfig(max_reflection_rounds=0),
        )
        with pytest.raises(ConfigError, match="max_reflection_rounds"):
            config.validate()

    def test_dedup_threshold_out_of_range(self):
        config = CCREngineConfig(
            anthropic_api_key="sk-ant-test",
            ace=ACEConfig(dedup_similarity_threshold=0.0),
        )
        with pytest.raises(ConfigError, match="dedup_similarity_threshold"):
            config.validate()

        config = CCREngineConfig(
            anthropic_api_key="sk-ant-test",
            ace=ACEConfig(dedup_similarity_threshold=1.5),
        )
        with pytest.raises(ConfigError, match="dedup_similarity_threshold"):
            config.validate()


class TestMemoryConfigValidation:
    """Memory-specific validation."""

    def test_negative_commit_count(self):
        config = CCREngineConfig(
            anthropic_api_key="sk-ant-test",
            memory=CCRConfig(recent_commit_count=-1),
        )
        with pytest.raises(ConfigError, match="recent_commit_count"):
            config.validate()

    def test_zero_file_size(self):
        config = CCREngineConfig(
            anthropic_api_key="sk-ant-test",
            memory=CCRConfig(index_max_file_size_kb=0),
        )
        with pytest.raises(ConfigError, match="index_max_file_size_kb"):
            config.validate()


class TestConfigErrorAttributes:
    """Verify ConfigError has correct attributes."""

    def test_config_error_not_recoverable(self):
        config = CCREngineConfig(anthropic_api_key="")
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert exc_info.value.recoverable is False

    def test_config_error_has_field(self):
        config = CCREngineConfig(anthropic_api_key="")
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert exc_info.value.field != ""

    def test_single_error_field_name(self):
        """Single error → field name is the specific field."""
        config = CCREngineConfig(
            anthropic_api_key="sk-ant-test",
            gateway_port=0,
        )
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert exc_info.value.field == "gateway_port"

    def test_multiple_errors_field_is_multiple(self):
        config = CCREngineConfig(anthropic_api_key="", sub_model="")
        with pytest.raises(ConfigError) as exc_info:
            config.validate()
        assert exc_info.value.field == "multiple"


class TestEngineValidationIntegration:
    """Verify engine calls validate() during initialize()."""

    def test_engine_rejects_bad_config(self, tmp_path):
        from ccr.core.engine import CCREngine

        config = CCREngineConfig(anthropic_api_key="")
        engine = CCREngine(str(tmp_path), config)

        with pytest.raises(ConfigError, match="anthropic_api_key"):
            engine.initialize()

    def test_engine_rejects_bad_port(self, tmp_path):
        from ccr.core.engine import CCREngine

        config = CCREngineConfig(
            anthropic_api_key="sk-ant-test",
            gateway_port=99999,
        )
        engine = CCREngine(str(tmp_path), config)

        with pytest.raises(ConfigError, match="gateway_port"):
            engine.initialize()
