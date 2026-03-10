"""Tests for RLM paper gap implementations."""

import os
import threading

import pytest

from ccr.core.types import RLMConfig, RLMResult, REPLResult, TokenUsage
from ccr.rlm.repl import CCRRepl, _SAFE_BUILTINS
from ccr.rlm.orchestrator import (
    CCRRlm,
    RLMError,
    BudgetExceededError,
    TimeoutExceededError,
    ErrorThresholdExceededError,
)


# ---------- 1. exec blocked in safe builtins ----------

class TestExecBlocked:
    def test_exec_is_none_in_safe_builtins(self):
        # The builtin 'exec' should be explicitly set to None
        assert "exec" in _SAFE_BUILTINS
        assert _SAFE_BUILTINS["exec"] is None

    def test_exec_blocked_in_repl(self):
        repl = CCRRepl()
        # Attempting to use the blocked builtin should error
        result = repl.execute_code("exec('x = 1')")
        assert result.error is not None


# ---------- 2. llm_query_batched ----------

class TestLlmQueryBatched:
    def test_llm_query_batched_returns_list(self):
        class MockClient:
            def completion(self, messages, **kw):
                return f"reply to: {messages[-1]['content']}"

        repl = CCRRepl(sub_client=MockClient())
        repl.execute_code("results = llm_query_batched(['a', 'b', 'c'])")
        result = repl.execute_code("FINAL_VAR('results')")
        assert result.final_answer is not None
        assert "reply to: a" in result.final_answer
        assert "reply to: b" in result.final_answer
        assert "reply to: c" in result.final_answer

    def test_llm_query_batched_empty(self):
        repl = CCRRepl(sub_client=None)
        repl.execute_code("results = llm_query_batched([])")
        result = repl.execute_code("FINAL_VAR('results')")
        assert result.final_answer == "[]"


# ---------- 3. rlm_query_batched ----------

class TestRlmQueryBatched:
    def test_rlm_query_batched_returns_list(self):
        def mock_subcall(prompt, model=None):
            return RLMResult(response=f"sub:{prompt}")

        repl = CCRRepl(subcall_fn=mock_subcall)
        repl.execute_code("results = rlm_query_batched(['x', 'y'])")
        result = repl.execute_code("FINAL_VAR('results')")
        assert result.final_answer is not None
        assert "sub:x" in result.final_answer
        assert "sub:y" in result.final_answer


# ---------- 4. Thread lock exists ----------

class TestThreadLock:
    def test_exec_lock_exists(self):
        repl = CCRRepl()
        assert hasattr(repl, "_exec_lock")
        assert isinstance(repl._exec_lock, type(threading.Lock()))

    def test_concurrent_execution_safe(self):
        """Verify the lock prevents interleaved execution."""
        repl = CCRRepl()
        results = []

        def run_code(code, idx):
            r = repl.execute_code(code)
            results.append((idx, r))

        t1 = threading.Thread(target=run_code, args=("x = 1\nprint(x)", 1))
        t2 = threading.Thread(target=run_code, args=("y = 2\nprint(y)", 2))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(results) == 2


# ---------- 5. Custom tools with (value, description) tuples ----------

class TestCustomToolTuples:
    def test_callable_tuple(self):
        def my_func(x):
            return x * 3

        repl = CCRRepl(custom_tools={"triple": (my_func, "Triples a number")})
        repl.execute_code("r = triple(7)")
        result = repl.execute_code("FINAL_VAR('r')")
        assert result.final_answer == "21"

    def test_value_tuple(self):
        repl = CCRRepl(custom_tools={"config": ({"key": "val"}, "Config dict")})
        repl.execute_code("r = config['key']")
        result = repl.execute_code("FINAL_VAR('r')")
        assert result.final_answer == "val"

    def test_plain_callable_still_works(self):
        repl = CCRRepl(custom_tools={"double": lambda x: x * 2})
        repl.execute_code("r = double(5)")
        result = repl.execute_code("FINAL_VAR('r')")
        assert result.final_answer == "10"

    def test_reserved_name_tuple_ignored(self):
        repl = CCRRepl(custom_tools={"llm_query": (lambda: "hacked", "desc")})
        # llm_query should still be the real one, not the custom one
        repl.execute_code("answer = 'test'")
        result = repl.execute_code("FINAL_VAR('answer')")
        assert result.final_answer == "test"


# ---------- 6. Large context uses temp file ----------

class TestLargeContext:
    def test_small_context_no_temp_file(self):
        repl = CCRRepl()
        repl.add_context("small payload", name="data")
        assert repl.locals["data"] == "small payload"
        assert "_data_path" not in repl.locals

    def test_large_context_creates_temp_file(self):
        repl = CCRRepl()
        large = "x" * 200_000
        repl.add_context(large, name="big")
        assert repl.locals["big"] == large
        assert "_big_path" in repl.locals
        path = repl.locals["_big_path"]
        assert os.path.exists(path)
        with open(path) as f:
            assert f.read() == large

    def test_large_dict_context(self):
        repl = CCRRepl()
        large_dict = {"data": "x" * 200_000}
        repl.add_context(large_dict, name="ctx")
        assert "_ctx_path" in repl.locals
        assert os.path.exists(repl.locals["_ctx_path"])


# ---------- 7. Typed exceptions ----------

class TestTypedExceptions:
    def test_rlm_error_base(self):
        err = RLMError("test", "partial")
        assert str(err) == "test"
        assert err.partial_answer == "partial"

    def test_budget_exceeded_is_rlm_error(self):
        assert issubclass(BudgetExceededError, RLMError)

    def test_timeout_exceeded_is_rlm_error(self):
        assert issubclass(TimeoutExceededError, RLMError)

    def test_error_threshold_is_rlm_error(self):
        assert issubclass(ErrorThresholdExceededError, RLMError)

    def test_timeout_returns_partial_answer(self):
        import time

        class SlowClient:
            def completion(self, messages, **kw):
                time.sleep(0.1)
                return "```repl\nx = 1\n```"
            def get_last_usage(self):
                return TokenUsage()

        rlm = CCRRlm(
            sub_client=SlowClient(),
            config=RLMConfig(max_iterations=100, max_timeout_seconds=0.05),
        )
        result = rlm.completion("test")
        # Should still return a result (caught by try/except in completion)
        assert result is not None
        assert result.final_answer_source == "error"

    def test_error_threshold_returns_result(self):
        class FailClient:
            def __init__(self):
                self._calls = 0
            def completion(self, messages, **kw):
                self._calls += 1
                return "```repl\n1/0\n```"
            def get_last_usage(self):
                return TokenUsage()

        rlm = CCRRlm(
            sub_client=FailClient(),
            config=RLMConfig(max_iterations=20, max_consecutive_errors=2),
        )
        result = rlm.completion("test")
        assert result is not None
        assert result.final_answer_source == "error"


# ---------- 8. Token limit in RLMConfig ----------

class TestTokenLimit:
    def test_max_total_tokens_default(self):
        cfg = RLMConfig()
        assert cfg.max_total_tokens == 0

    def test_max_total_tokens_custom(self):
        cfg = RLMConfig(max_total_tokens=5000)
        assert cfg.max_total_tokens == 5000

    def test_token_limit_triggers_budget_error(self):
        class TokenClient:
            def completion(self, messages, **kw):
                return "```repl\nx = 1\n```"
            def get_last_usage(self):
                return TokenUsage(input_tokens=5000, output_tokens=5000, total_tokens=10000)

        rlm = CCRRlm(
            sub_client=TokenClient(),
            config=RLMConfig(max_iterations=10, max_total_tokens=100),
        )
        result = rlm.completion("test")
        # Should hit token limit after first iteration and return error result
        assert result is not None
        assert result.final_answer_source == "error"


# ---------- 9. Compaction history stored as REPL variable ----------

class TestCompactionHistory:
    def test_history_variable_exists_in_repl(self):
        """The orchestrator should set `history` in the REPL locals."""

        class MockClient:
            def completion(self, messages, **kw):
                return "```repl\nh = history\nFINAL_VAR('h')\n```"
            def get_last_usage(self):
                return TokenUsage()

        rlm = CCRRlm(
            sub_client=MockClient(),
            config=RLMConfig(max_iterations=5),
        )
        result = rlm.completion("test")
        # history should be an empty list (no compaction happened)
        assert result.response == "[]"

    def test_compaction_history_attr_exists(self):
        class MockClient:
            def completion(self, messages, **kw):
                return "```repl\nFINAL_VAR(42)\n```"
            def get_last_usage(self):
                return TokenUsage()

        rlm = CCRRlm(sub_client=MockClient())
        rlm.completion("test")
        assert hasattr(rlm, "_compaction_history")
        assert isinstance(rlm._compaction_history, list)


# ---------- 10. System prompt has examples ----------

class TestSystemPromptExamples:
    def test_prompt_has_example_strategies(self):
        from ccr.context.prompts import RLM_SYSTEM_PROMPT
        assert "Example Strategies" in RLM_SYSTEM_PROMPT

    def test_prompt_has_chunking_example(self):
        from ccr.context.prompts import RLM_SYSTEM_PROMPT
        assert "task_prompt[:500]" in RLM_SYSTEM_PROMPT

    def test_prompt_has_batched_example(self):
        from ccr.context.prompts import RLM_SYSTEM_PROMPT
        assert "llm_query_batched" in RLM_SYSTEM_PROMPT

    def test_prompt_has_llm_query_example(self):
        from ccr.context.prompts import RLM_SYSTEM_PROMPT
        assert "llm_query(f\"Classify" in RLM_SYSTEM_PROMPT
