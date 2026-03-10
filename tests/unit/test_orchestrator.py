"""Tests for the CCR RLM Orchestrator (completion loop)."""

import pytest

from ccr.core.types import RLMConfig, RLMResult, TokenUsage
from ccr.rlm.orchestrator import (
    CCRRlm,
    format_execution_result,
    format_iteration,
    _make_metadata_preview,
)
from ccr.core.types import REPLResult


class MockSubClient:
    """Mock sub-model that returns code blocks or plain text."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self._call_idx = 0
        self._last_usage = TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30)

    def completion(self, messages, **kwargs):
        if self._call_idx < len(self.responses):
            resp = self.responses[self._call_idx]
            self._call_idx += 1
            return resp
        return "No more responses."

    def get_last_usage(self):
        return self._last_usage


class TestFormatHelpers:
    def test_format_execution_result_with_stdout(self):
        result = REPLResult(stdout="hello world\n")
        text = format_execution_result(result)
        assert "hello world" in text

    def test_format_execution_result_with_error(self):
        result = REPLResult(stderr="ZeroDivisionError: division by zero")
        text = format_execution_result(result)
        assert "ZeroDivisionError" in text

    def test_format_execution_result_with_final(self):
        result = REPLResult(final_answer="the answer is 42")
        text = format_execution_result(result)
        assert "42" in text

    def test_format_execution_result_empty(self):
        result = REPLResult()
        text = format_execution_result(result)
        assert "no output" in text.lower()

    def test_format_iteration_no_code(self):
        msgs = format_iteration("Just text response", [], [])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"

    def test_format_iteration_with_code(self):
        result = REPLResult(stdout="42\n")
        msgs = format_iteration("Here's code", ["x = 42\nprint(x)"], [result])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "assistant"
        assert msgs[1]["role"] == "user"
        assert "42" in msgs[1]["content"]


class TestOrchestratorBasics:
    def test_fallback_at_max_depth(self):
        client = MockSubClient(["plain answer"])
        rlm = CCRRlm(sub_client=client, depth=5, config=RLMConfig(max_depth=2))
        result = rlm.completion("test prompt")
        assert result.final_answer_source == "fallback"
        assert result.response == "plain answer"

    def test_direct_final_in_code(self):
        """If the model writes code with FINAL_VAR, it should terminate."""
        client = MockSubClient([
            "Let me compute:\n```repl\nanswer = 42\nFINAL_VAR('answer')\n```"
        ])
        rlm = CCRRlm(sub_client=client, config=RLMConfig(max_iterations=5))
        result = rlm.completion("What is the answer?")
        assert result.response == "42"
        assert result.final_answer_source == "FINAL_VAR"
        assert result.iterations_used == 1

    def test_multi_iteration(self):
        """Model needs two iterations to reach FINAL_VAR."""
        client = MockSubClient([
            "First, let me explore:\n```repl\nx = 10\nprint(x)\n```",
            "Now compute:\n```repl\ny = x * 2\nFINAL_VAR('y')\n```",
        ])
        rlm = CCRRlm(sub_client=client, config=RLMConfig(max_iterations=5))
        result = rlm.completion("Compute something")
        assert result.response == "20"
        assert result.iterations_used == 2

    def test_max_iterations_exhausted(self):
        """When iterations run out, should ask for default answer."""
        client = MockSubClient([
            "```repl\nx = 1\n```",
            "```repl\ny = 2\n```",
            "The best answer based on my work is: x=1, y=2",  # default answer
        ])
        rlm = CCRRlm(sub_client=client, config=RLMConfig(max_iterations=2))
        result = rlm.completion("test")
        assert result.final_answer_source == "default"
        assert result.iterations_used == 2

    def test_timeout_respected(self):
        """Should stop when timeout is exceeded."""
        import time

        class SlowClient:
            def completion(self, messages, **kw):
                time.sleep(0.1)
                return "```repl\nx = 1\n```"

            def get_last_usage(self):
                return TokenUsage()

        rlm = CCRRlm(
            sub_client=SlowClient(),
            config=RLMConfig(max_iterations=100, max_timeout_seconds=0.2),
        )
        result = rlm.completion("test")
        assert result.iterations_used < 100

    def test_error_threshold(self):
        """Should stop after too many consecutive errors."""
        client = MockSubClient([
            "```repl\n1/0\n```",
            "```repl\n1/0\n```",
            "```repl\n1/0\n```",
            "default answer",
        ])
        rlm = CCRRlm(
            sub_client=client,
            config=RLMConfig(max_iterations=10, max_consecutive_errors=3),
        )
        result = rlm.completion("test")
        assert result.iterations_used <= 4


class TestOrchestratorWithRepoIndex:
    def test_context_available_in_repl(self):
        """REPL should have context variable from repo index."""
        mock_index = {"files": {"main.py": {"symbols": ["App"], "lines": 10}}}
        client = MockSubClient([
            "```repl\nfile_count = len(context['files'])\nFINAL_VAR('file_count')\n```"
        ])
        rlm = CCRRlm(sub_client=client, repo_index=mock_index)
        result = rlm.completion("How many files?")
        assert result.response == "1"

    def test_llm_query_in_repl(self):
        """llm_query should work inside REPL code."""
        class SmartClient:
            def __init__(self):
                self._calls = 0
            def completion(self, messages, **kw):
                self._calls += 1
                if self._calls == 1:
                    return '```repl\nanalysis = llm_query("summarize")\nFINAL_VAR("analysis")\n```'
                return "summary of the code"
            def get_last_usage(self):
                return TokenUsage()

        rlm = CCRRlm(sub_client=SmartClient())
        result = rlm.completion("Analyze the code")
        assert "summary" in result.response.lower()


class TestOrchestratorRecursion:
    def test_subcall_spawns_child(self):
        """rlm_query in REPL should spawn a child CCRRlm."""
        class RecursiveClient:
            def __init__(self):
                self._calls = 0
            def completion(self, messages, **kw):
                self._calls += 1
                if self._calls == 1:
                    # Parent: use rlm_query
                    return '```repl\nsub_result = rlm_query("child task")\nFINAL_VAR("sub_result")\n```'
                # Child (and any subsequent): direct answer
                return "```repl\nresult = 'child answer'\nFINAL_VAR('result')\n```"
            def get_last_usage(self):
                return TokenUsage()

        rlm = CCRRlm(
            sub_client=RecursiveClient(),
            config=RLMConfig(max_depth=2, max_iterations=5),
        )
        result = rlm.completion("parent task")
        assert "child answer" in result.response

    def test_max_depth_prevents_infinite_recursion(self):
        """At max depth, rlm_query should fall back to llm_query."""
        client = MockSubClient([
            "```repl\nr = rlm_query('deep task')\nFINAL_VAR('r')\n```",
            "plain llm response",
        ])
        rlm = CCRRlm(
            sub_client=client,
            config=RLMConfig(max_depth=1),  # depth=0, so children at depth=1 = fallback
            depth=0,
        )
        result = rlm.completion("test")
        # Should still complete without infinite recursion
        assert result.response is not None


class TestMetadataOnlyStdout:
    """Gap 1: RLM paper requires metadata-only stdout in message history."""

    def test_metadata_preview_short(self):
        preview = _make_metadata_preview("hello")
        assert "5 chars" in preview
        assert "hello" in preview

    def test_metadata_preview_long(self):
        long_text = "x" * 200
        preview = _make_metadata_preview(long_text)
        assert "200 chars" in preview
        assert "..." in preview

    def test_stdout_is_metadata_only(self):
        """Stdout should show only length + preview, not full content."""
        big_output = "line\n" * 100  # 500 chars
        result = REPLResult(stdout=big_output)
        text = format_execution_result(result)
        # Should NOT contain the full output
        assert text.count("line") <= 30  # preview only
        # Should contain metadata
        assert "chars" in text
        assert "preview" in text

    def test_stderr_preserved_for_debugging(self):
        """Errors get more detail to help the model debug."""
        result = REPLResult(stderr="TypeError: 'int' object is not callable")
        text = format_execution_result(result)
        assert "TypeError" in text

    def test_final_var_preserved(self):
        result = REPLResult(final_answer="42")
        text = format_execution_result(result)
        assert "42" in text

    def test_variables_shown(self):
        result = REPLResult(locals_snapshot={"x": "42", "data": "{'key': 'val'}"})
        text = format_execution_result(result)
        assert "x" in text
        assert "data" in text


class TestPromptAsREPLVariable:
    """Gap 2: RLM paper requires prompt loaded as REPL variable, metadata-first history."""

    def test_task_prompt_loaded_in_repl(self):
        """The prompt should be accessible as task_prompt in the REPL."""
        client = MockSubClient([
            "```repl\n# Read the task\ntask_text = task_prompt[:50]\nFINAL_VAR('task_text')\n```"
        ])
        rlm = CCRRlm(sub_client=client, config=RLMConfig(max_iterations=5))
        result = rlm.completion("Fix the authentication bug in auth.py")
        assert "Fix the authentication" in result.response

    def test_metadata_first_history(self):
        """Initial message should contain metadata, NOT the full prompt."""
        captured_messages = []

        class CapturingClient:
            def completion(self, messages, **kw):
                captured_messages.append(messages)
                return "```repl\nresult = 'done'\nFINAL_VAR('result')\n```"
            def get_last_usage(self):
                return TokenUsage()

        rlm = CCRRlm(sub_client=CapturingClient(), config=RLMConfig(max_iterations=5))
        long_prompt = "Refactor the entire authentication system. " * 50  # ~2000 chars
        rlm.completion(long_prompt)

        # The first message set should NOT contain the full prompt
        first_msgs = captured_messages[0]
        user_msg = next(m for m in first_msgs if m["role"] == "user")
        # Should have metadata, not the full prompt text
        assert "task_prompt" in user_msg["content"]
        assert "chars" in user_msg["content"]
        # The full 2000-char prompt should NOT be in the message
        assert long_prompt not in user_msg["content"]

    def test_prompt_metadata_contains_length(self):
        """Metadata message should include prompt length and token estimate."""
        captured = []

        class CapturingClient:
            def completion(self, messages, **kw):
                captured.append(messages)
                return "```repl\nFINAL_VAR(42)\n```"
            def get_last_usage(self):
                return TokenUsage()

        rlm = CCRRlm(sub_client=CapturingClient())
        rlm.completion("Short task")

        user_msg = next(m for m in captured[0] if m["role"] == "user")
        assert "10 chars" in user_msg["content"]
        assert "tokens" in user_msg["content"]

    def test_prompt_accessible_via_repl_code(self):
        """Model should be able to read the full prompt via REPL code."""
        client = MockSubClient([
            "```repl\nprompt_len = len(task_prompt)\nFINAL_VAR('prompt_len')\n```"
        ])
        rlm = CCRRlm(sub_client=client)
        result = rlm.completion("Hello world task")
        assert result.response == "16"  # len("Hello world task")
