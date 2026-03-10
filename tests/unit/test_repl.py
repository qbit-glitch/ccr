"""Tests for the CCR REPL (sandboxed Python execution environment)."""

import pytest

from ccr.core.types import REPLResult
from ccr.rlm.repl import CCRRepl


class TestREPLBasics:
    def test_simple_print(self):
        repl = CCRRepl()
        result = repl.execute_code("print('hello')")
        assert result.stdout.strip() == "hello"
        assert result.error is None

    def test_variable_persistence(self):
        repl = CCRRepl()
        repl.execute_code("x = 42")
        result = repl.execute_code("print(x + 8)")
        assert "50" in result.stdout

    def test_final_var(self):
        repl = CCRRepl()
        repl.execute_code("answer = 'the result'")
        result = repl.execute_code("FINAL_VAR('answer')")
        assert result.final_answer == "the result"

    def test_final_var_dict(self):
        repl = CCRRepl()
        repl.execute_code("data = {'key': 'value'}")
        result = repl.execute_code("FINAL_VAR('data')")
        assert '"key"' in result.final_answer
        assert '"value"' in result.final_answer

    def test_final_var_direct_value(self):
        repl = CCRRepl()
        result = repl.execute_code("FINAL_VAR(123)")
        assert result.final_answer == "123"

    def test_final_var_not_found(self):
        repl = CCRRepl()
        result = repl.execute_code("FINAL_VAR('nonexistent')")
        assert result.final_answer is None
        assert "not found" in result.stdout.lower()

    def test_show_vars_empty(self):
        repl = CCRRepl()
        result = repl.execute_code("print(SHOW_VARS())")
        assert "no variables" in result.stdout.lower()

    def test_show_vars_with_data(self):
        repl = CCRRepl()
        repl.execute_code("x = 42\ny = 'hello'")
        result = repl.execute_code("print(SHOW_VARS())")
        assert "x" in result.stdout
        assert "y" in result.stdout

    def test_error_handling(self):
        repl = CCRRepl()
        result = repl.execute_code("1 / 0")
        assert result.error is not None
        assert "ZeroDivisionError" in result.error

    def test_safe_builtins_block_input(self):
        repl = CCRRepl()
        result = repl.execute_code("input('test')")
        assert result.error is not None

    def test_imports_work(self):
        repl = CCRRepl()
        result = repl.execute_code("import json\nprint(json.dumps({'a': 1}))")
        assert '{"a": 1}' in result.stdout

    def test_locals_snapshot(self):
        repl = CCRRepl()
        repl.execute_code("x = [1, 2, 3]\ny = 'hello'")
        result = repl.execute_code("z = x + [4]")
        assert "x" in result.locals_snapshot
        assert "y" in result.locals_snapshot
        assert "z" in result.locals_snapshot


class TestREPLWithContext:
    def test_context_dict_loaded(self):
        repl = CCRRepl(repo_index={"files": {"a.py": {"symbols": ["foo"]}}})
        result = repl.execute_code("print(type(context))")
        assert "dict" in result.stdout

    def test_context_accessible(self):
        repl = CCRRepl(repo_index={"files": {"a.py": {"symbols": ["foo"]}}})
        result = repl.execute_code("print(context['files']['a.py']['symbols'])")
        assert "foo" in result.stdout

    def test_custom_tools(self):
        def my_tool(x):
            return x * 2

        repl = CCRRepl(custom_tools={"double": my_tool})
        result = repl.execute_code("result = double(21)\nprint(result)")
        assert "42" in result.stdout

    def test_reserved_names_not_overwritten(self):
        repl = CCRRepl(custom_tools={"FINAL_VAR": lambda: "hacked"})
        repl.execute_code("answer = 'real'")
        result = repl.execute_code("FINAL_VAR('answer')")
        assert result.final_answer == "real"


class TestREPLWithMockClient:
    def test_llm_query(self):
        class MockClient:
            def completion(self, messages, **kw):
                return "mock response"

        repl = CCRRepl(sub_client=MockClient())
        result = repl.execute_code("r = llm_query('hello')\nprint(r)")
        assert "mock response" in result.stdout

    def test_llm_query_no_client(self):
        repl = CCRRepl(sub_client=None)
        result = repl.execute_code("r = llm_query('hello')\nprint(r)")
        assert "Error" in result.stdout

    def test_rlm_query_falls_back_to_llm(self):
        class MockClient:
            def completion(self, messages, **kw):
                return "fallback response"

        repl = CCRRepl(sub_client=MockClient(), subcall_fn=None)
        result = repl.execute_code("r = rlm_query('hello')\nprint(r)")
        assert "fallback response" in result.stdout

    def test_rlm_query_with_subcall(self):
        from ccr.core.types import RLMResult

        def mock_subcall(prompt, model=None):
            return RLMResult(response="recursive result")

        repl = CCRRepl(subcall_fn=mock_subcall)
        result = repl.execute_code("r = rlm_query('sub-task')\nprint(r)")
        assert "recursive result" in result.stdout


class TestREPLScaffoldRestore:
    def test_overwrite_llm_query_restored(self):
        class MockClient:
            def completion(self, messages, **kw):
                return "works"

        repl = CCRRepl(sub_client=MockClient())
        # Overwrite llm_query in user code
        repl.execute_code("llm_query = 'broken'")
        # Should still work after restore
        result = repl.execute_code("r = llm_query('test')\nprint(r)")
        assert "works" in result.stdout

    def test_overwrite_final_var_restored(self):
        repl = CCRRepl()
        repl.execute_code("FINAL_VAR = 'broken'")
        repl.execute_code("answer = 42")
        result = repl.execute_code("FINAL_VAR('answer')")
        assert result.final_answer == "42"


class TestREPLCleanup:
    def test_context_manager(self):
        with CCRRepl() as repl:
            result = repl.execute_code("x = 1")
            assert result.error is None
        assert len(repl.globals) == 0

    def test_execution_time_tracked(self):
        repl = CCRRepl()
        result = repl.execute_code("x = sum(range(1000))")
        assert result.execution_time >= 0
