"""Real API integration tests — only run when API keys are set.

These test against actual model APIs to verify the full pipeline works
with real models. Skip if no API keys are available.

Run with: pytest tests/integration/test_real_api.py -v --run-real-api
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

# Custom marker for real API tests
real_api = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping real API tests",
)

# For sub-model tests (vLLM or OpenRouter)
has_sub_model = pytest.mark.skipif(
    not (os.environ.get("CCR_SUB_MODEL_URL") or os.environ.get("OPENROUTER_API_KEY")),
    reason="No sub-model endpoint configured",
)


@real_api
class TestClaudeClientReal:
    """Test the Claude client against the real API."""

    def test_basic_completion(self):
        from ccr.models.anthropic_client import ClaudeClient

        client = ClaudeClient(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model_name="claude-haiku-3-5-20241022",  # Use cheapest model
            max_tokens=100,
        )
        response = client.completion("Say 'test passed' and nothing else.")
        assert "test" in response.lower() or "passed" in response.lower()

        usage = client.get_last_usage()
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert usage.cost_usd is not None and usage.cost_usd > 0

    def test_message_list_format(self):
        from ccr.models.anthropic_client import ClaudeClient

        client = ClaudeClient(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model_name="claude-haiku-3-5-20241022",
            max_tokens=100,
        )
        response = client.completion([
            {"role": "user", "content": "Say 'hello' and nothing else."},
        ])
        assert len(response) > 0


@has_sub_model
class TestSubModelReal:
    """Test the OpenAI-compatible client against a real sub-model."""

    def _get_client(self):
        from ccr.models.openai_compat import OpenAICompatClient

        url = os.environ.get("CCR_SUB_MODEL_URL", "http://localhost:8000/v1")
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("CCR_SUB_MODEL_API_KEY")
        model = os.environ.get("CCR_SUB_MODEL", "openai/gpt-oss-20b")

        return OpenAICompatClient(
            model_name=model,
            base_url=url,
            api_key=key,
            max_tokens=200,
        )

    def test_basic_completion(self):
        client = self._get_client()
        response = client.completion("Say 'test passed' and nothing else.")
        assert len(response) > 0

    def test_json_output(self):
        client = self._get_client()
        response = client.completion([
            {"role": "system", "content": "Output only valid JSON."},
            {"role": "user", "content": 'Return: {"status": "ok"}'},
        ])
        # Should contain JSON
        assert "{" in response


@real_api
class TestFullPipelineReal:
    """Test the full CCR pipeline with real API."""

    def test_engine_processes_trivial_request(self):
        """Process a trivial request through the full engine."""
        from ccr.core.engine import CCREngine
        from ccr.core.types import CCREngineConfig
        from ccr.utils.parsing import parse_anthropic_request

        with tempfile.TemporaryDirectory() as project_dir:
            # Create a minimal project
            with open(os.path.join(project_dir, "hello.py"), "w") as f:
                f.write("print('hello world')\n")

            config = CCREngineConfig(
                anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
                claude_model="claude-haiku-3-5-20241022",
                sub_model="claude-haiku-3-5-20241022",  # Use Claude as sub too for testing
                sub_model_base_url="https://api.anthropic.com/v1",  # Won't be used for trivial
            )

            engine = CCREngine(project_dir, config)
            engine.initialize()

            try:
                request = parse_anthropic_request(json.dumps({
                    "model": "claude-haiku-3-5",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Say hello"}],
                }).encode())

                response = engine.process(request)
                assert response.content
                assert response.classification.tier.value in ("trivial", "simple")

                report = engine.get_usage_report()
                print(f"\nUsage report: {json.dumps(report, indent=2)}")
            finally:
                engine.shutdown()
