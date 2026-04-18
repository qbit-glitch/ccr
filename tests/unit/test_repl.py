"""Tests for the CCR REPL (sandboxed Python execution environment)."""

import builtins as _builtins_module
import os
import tempfile

import pytest

from ccr.context.indexer import RepoIndex
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
        # Error routed to logger, not stdout (prevents MCP transport corruption)
        assert "not found" not in result.stdout.lower()

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


class TestSandboxSecurity:
    """Tests for REPL sandbox security — verifying dangerous modules and ops are blocked."""

    def test_os_module_blocked(self):
        repl = CCRRepl()
        result = repl.execute_code("import os")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_sys_module_blocked(self):
        repl = CCRRepl()
        result = repl.execute_code("import sys")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_socket_module_blocked(self):
        repl = CCRRepl()
        result = repl.execute_code("import socket")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_importlib_blocked(self):
        repl = CCRRepl()
        result = repl.execute_code("import importlib")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_http_blocked(self):
        repl = CCRRepl()
        result = repl.execute_code("import http")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_http_client_blocked(self):
        repl = CCRRepl()
        result = repl.execute_code("import http.client")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_urllib_blocked(self):
        repl = CCRRepl()
        result = repl.execute_code("import urllib")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_pathlib_blocked(self):
        repl = CCRRepl()
        result = repl.execute_code("import pathlib")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_open_restricted(self):
        repl = CCRRepl()
        result = repl.execute_code("open('/etc/passwd')")
        assert result.error is not None
        assert "PermissionError" in result.error

    def test_open_allowed_in_repo(self):
        """Reading a file within the project root should work."""
        with tempfile.TemporaryDirectory() as proj_root:
            # Create a test file in the project root
            test_file = os.path.join(proj_root, "test.txt")
            with open(test_file, "w") as f:
                f.write("hello from project")
            repl = CCRRepl(project_root=proj_root)
            result = repl.execute_code(
                f"content = open({test_file!r}).read()\nprint(content)"
            )
            assert result.error is None
            assert "hello from project" in result.stdout

    def test_open_allowed_in_temp_dir(self):
        """Writing/reading within the REPL temp dir should work."""
        repl = CCRRepl()
        temp_dir = repl.temp_dir
        result = repl.execute_code(
            f"f = open({temp_dir!r} + '/test.txt', 'w')\nf.write('hi')\nf.close()\n"
            f"print(open({temp_dir!r} + '/test.txt').read())"
        )
        assert result.error is None
        assert "hi" in result.stdout

    def test_subprocess_still_blocked(self):
        """Verify original blocks still work."""
        repl = CCRRepl()
        result = repl.execute_code("import subprocess")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_nested_import_blocked(self):
        """exec('import os') within REPL should be blocked (exec is None in sandbox)."""
        repl = CCRRepl()
        # exec is set to None in _SAFE_BUILTINS, so this should error
        result = repl.execute_code("exec('import os')")
        assert result.error is not None

    def test_importlib_import_module_blocked(self):
        """Even if importlib somehow loaded, it should be blocked at import level."""
        repl = CCRRepl()
        result = repl.execute_code("import importlib\nimportlib.import_module('subprocess')")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_asyncio_blocked(self):
        repl = CCRRepl()
        result = repl.execute_code("import asyncio")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_io_blocked(self):
        repl = CCRRepl()
        result = repl.execute_code("import io")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_glob_blocked(self):
        repl = CCRRepl()
        result = repl.execute_code("import glob")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_safe_modules_still_work(self):
        """Ensure safe modules (json, re, math, collections, datetime, itertools) are not blocked."""
        repl = CCRRepl()
        result = repl.execute_code(
            "import json\nimport re\nimport math\n"
            "import collections\nimport datetime\nimport itertools\n"
            "print('all safe imports ok')"
        )
        assert result.error is None
        assert "all safe imports ok" in result.stdout

    def test_from_os_import_blocked(self):
        """'from os import path' should also be blocked."""
        repl = CCRRepl()
        result = repl.execute_code("from os import path")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_dunder_import_in_globals(self):
        """__import__ at globals level should be the safe version."""
        repl = CCRRepl()
        result = repl.execute_code("__import__('os')")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_vars_removed_from_builtins(self):
        """vars() should not be available (introspection hardening)."""
        repl = CCRRepl()
        result = repl.execute_code("vars()")
        assert result.error is not None

    def test_setattr_removed_from_builtins(self):
        """setattr() should not be available (introspection hardening)."""
        repl = CCRRepl()
        result = repl.execute_code("setattr(object, 'x', 1)")
        assert result.error is not None

    def test_delattr_removed_from_builtins(self):
        """delattr() should not be available (introspection hardening)."""
        repl = CCRRepl()
        result = repl.execute_code("delattr(object, '__doc__')")
        assert result.error is not None


class TestSandboxEscapePrevention:
    """Tests for sandbox escape vectors — C1/C2/C3/C4 from security audit."""

    def test_module_builtins_import_blocked(self):
        """C1: json.__builtins__['__import__']('os') must be blocked.

        Now blocked at AST level — __builtins__ is a dangerous dunder.
        """
        repl = CCRRepl()
        result = repl.execute_code(
            "import json\n"
            "real_import = json.__builtins__['__import__']\n"
            "os_mod = real_import('os')\n"
            "os_mod.system('whoami')"
        )
        assert result.error is not None
        assert "blocked" in (result.error or "").lower()

    def test_gc_module_blocked(self):
        """C2: gc module must be blocked — gc.get_objects() exposes all live objects."""
        repl = CCRRepl()
        result = repl.execute_code("import gc")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_pickle_blocked(self):
        """C3: pickle module must be blocked — arbitrary code execution via __reduce__."""
        repl = CCRRepl()
        result = repl.execute_code("import pickle")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_marshal_blocked(self):
        """C3: marshal module must be blocked — bytecode manipulation."""
        repl = CCRRepl()
        result = repl.execute_code("import marshal")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_ctypes_blocked(self):
        """C3: ctypes module must be blocked — FFI escape."""
        repl = CCRRepl()
        result = repl.execute_code("import ctypes")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_subclasses_no_os_access(self):
        """C4: __subclasses__() chain via __class__.__bases__ must be blocked at AST level.

        Direct attribute access to __class__, __bases__, __subclasses__, __globals__
        is now blocked by AST validation before exec — the code never runs.
        """
        repl = CCRRepl()
        result = repl.execute_code(
            "found_os = None\n"
            "for c in ().__class__.__bases__[0].__subclasses__():\n"
            "    pass\n"
        )
        # AST validator blocks __class__ access
        assert result.error is not None
        assert "blocked" in result.error.lower() or "PermissionError" in result.error

    def test_builtins_import_restored_after_exec(self):
        """After execution, builtins.__import__ must be the original (not safe version)."""
        original = _builtins_module.__import__
        repl = CCRRepl()
        repl.execute_code("x = 1")
        # builtins.__import__ should be restored to original after exec
        assert _builtins_module.__import__ is original

    def test_safe_modules_still_work_after_hardening(self):
        """json, re, math, collections, datetime, itertools must still be importable."""
        repl = CCRRepl()
        result = repl.execute_code(
            "import json\nimport re\nimport math\n"
            "import collections\nimport datetime\nimport itertools\n"
            "import functools\nimport string\nimport hashlib\n"
            "print('all safe imports ok')"
        )
        assert result.error is None
        assert "all safe imports ok" in result.stdout

    def test_nested_import_via_module_builtins(self):
        """Even reaching the real import via module attributes, blocked at AST level now."""
        repl = CCRRepl()
        result = repl.execute_code(
            "import json\n"
            "bi = json.__builtins__\n"
        )
        assert result.error is not None
        assert "blocked" in (result.error or "").lower()

    def test_module_builtins_scrubbed_between_calls(self):
        """After exec, module.__builtins__ access is blocked at AST level."""
        repl = CCRRepl()
        repl.execute_code("import json")
        result = repl.execute_code(
            "bi = json.__builtins__\n"
        )
        assert result.error is not None
        assert "blocked" in (result.error or "").lower()

    def test_posixsubprocess_blocked(self):
        """_posixsubprocess must be blocked."""
        repl = CCRRepl()
        result = repl.execute_code("import _posixsubprocess")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_struct_blocked(self):
        """struct module must be blocked — raw memory packing."""
        repl = CCRRepl()
        result = repl.execute_code("import struct")
        assert result.error is not None
        assert "ImportError" in result.error


class TestCriticalSandboxFixes:
    """Tests for C1-C5+H2 critical security fixes."""

    def test_c1_type_3arg_blocked(self):
        """C1: type('X', (object,), {}) metaclass form must be blocked."""
        repl = CCRRepl()
        result = repl.execute_code("type('X', (object,), {})")
        assert result.error is not None
        assert "blocked" in result.error.lower() or "TypeError" in result.error

    def test_c1_type_1arg_works(self):
        """C1: type(42) single-arg form must still work."""
        repl = CCRRepl()
        result = repl.execute_code("print(type(42))")
        assert result.error is None
        assert "int" in result.stdout

    def test_c2_object_subclasses_blocked(self):
        """C2: object.__subclasses__() is blocked at AST level."""
        repl = CCRRepl()
        result = repl.execute_code(
            "subs = object.__subclasses__()\n"
            "print(len(subs))"
        )
        # __subclasses__ is a dangerous dunder, blocked by AST validation
        assert result.error is not None
        assert "blocked" in (result.error or "").lower()

    def test_c2_getattr_subclasses_blocked(self):
        """C2: getattr(object, '__subclasses__') is blocked by C3 dunder protection."""
        repl = CCRRepl()
        result = repl.execute_code("getattr(object, '__subclasses__')()")
        assert result.error is not None

    def test_c3_getattr_dunder_blocked(self):
        """C3: getattr(obj, '__globals__') must be blocked."""
        repl = CCRRepl()
        result = repl.execute_code("getattr(42, '__class__')")
        assert result.error is not None
        assert "blocked" in result.error.lower() or "AttributeError" in result.error

    def test_c3_hasattr_dunder_returns_false(self):
        """C3: hasattr(obj, '__globals__') must return False."""
        repl = CCRRepl()
        result = repl.execute_code("print(hasattr(42, '__globals__'))")
        assert result.error is None
        assert "False" in result.stdout

    def test_c5_h2_threading_blocked(self):
        """C5+H2: import threading must be blocked."""
        repl = CCRRepl()
        result = repl.execute_code("import threading")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_class_creation_still_works(self):
        """Class creation via 'class' statement (without type()) must still work."""
        repl = CCRRepl()
        result = repl.execute_code("class Foo:\n    x = 1\nprint(Foo.x)")
        assert result.error is None
        assert "1" in result.stdout

    def test_isinstance_still_works(self):
        """isinstance(x, int) must still work."""
        repl = CCRRepl()
        result = repl.execute_code("print(isinstance(42, int))")
        assert result.error is None
        assert "True" in result.stdout

    def test_safe_builtins_still_work(self):
        """sorted, enumerate, range must still work."""
        repl = CCRRepl()
        result = repl.execute_code(
            "print(sorted([3, 1, 2]))\n"
            "print(list(enumerate(['a', 'b'])))\n"
            "print(list(range(3)))"
        )
        assert result.error is None
        assert "[1, 2, 3]" in result.stdout
        assert "[(0, 'a'), (1, 'b')]" in result.stdout
        assert "[0, 1, 2]" in result.stdout


class TestASTSandboxHardening:
    """Tests for C1-C4 AST-level sandbox hardening — blocks direct dunder attribute access."""

    def test_class_bases_blocked(self):
        """C1: ().__class__.__bases__[0] is blocked at AST level."""
        repl = CCRRepl()
        result = repl.execute_code("().__class__.__bases__[0]")
        assert result.error is not None
        assert "blocked" in result.error.lower()

    def test_traceback_tb_frame_blocked(self):
        """C2: Exception __traceback__.tb_frame is blocked at AST level."""
        repl = CCRRepl()
        result = repl.execute_code(
            "try:\n"
            "    1/0\n"
            "except Exception as e:\n"
            "    e.__traceback__.tb_frame\n"
        )
        assert result.error is not None
        assert "blocked" in result.error.lower()

    def test_generator_gi_frame_blocked(self):
        """C3: Generator gi_frame is blocked at AST level."""
        repl = CCRRepl()
        result = repl.execute_code(
            "def gen():\n"
            "    yield 1\n"
            "g = gen()\n"
            "g.gi_frame\n"
        )
        assert result.error is not None
        assert "blocked" in result.error.lower()

    def test_init_subclass_def_blocked(self):
        """C4: __init_subclass__ definition is blocked at AST level."""
        repl = CCRRepl()
        result = repl.execute_code(
            "class Evil:\n"
            "    def __init_subclass__(cls, **kw):\n"
            "        pass\n"
        )
        assert result.error is not None
        assert "blocked" in result.error.lower()

    def test_set_name_def_blocked(self):
        """C4: __set_name__ definition is blocked at AST level."""
        repl = CCRRepl()
        result = repl.execute_code(
            "class Descriptor:\n"
            "    def __set_name__(self, owner, name):\n"
            "        pass\n"
        )
        assert result.error is not None
        assert "blocked" in result.error.lower()

    def test_globals_blocked(self):
        """obj.__globals__ is blocked at AST level."""
        repl = CCRRepl()
        result = repl.execute_code(
            "def f(): pass\n"
            "f.__globals__\n"
        )
        assert result.error is not None
        assert "blocked" in result.error.lower()

    def test_import_builtins_blocked(self):
        """M4: import builtins must be blocked."""
        repl = CCRRepl()
        result = repl.execute_code("import builtins")
        assert result.error is not None
        assert "ImportError" in result.error

    def test_allowed_dunder_len(self):
        """Allowed dunders still work: __len__."""
        repl = CCRRepl()
        result = repl.execute_code(
            "class Foo:\n"
            "    def __len__(self):\n"
            "        return 42\n"
            "print(len(Foo()))"
        )
        assert result.error is None
        assert "42" in result.stdout

    def test_allowed_dunder_str(self):
        """Allowed dunders still work: __str__."""
        repl = CCRRepl()
        result = repl.execute_code(
            "class Foo:\n"
            "    def __str__(self):\n"
            "        return 'hello'\n"
            "print(str(Foo()))"
        )
        assert result.error is None
        assert "hello" in result.stdout

    def test_allowed_dunder_getitem(self):
        """Allowed dunders still work: __getitem__."""
        repl = CCRRepl()
        result = repl.execute_code(
            "class Foo:\n"
            "    def __getitem__(self, key):\n"
            "        return key * 2\n"
            "print(Foo()[3])"
        )
        assert result.error is None
        assert "6" in result.stdout

    def test_allowed_dunder_iter_next(self):
        """Allowed dunders still work: __iter__ and __next__."""
        repl = CCRRepl()
        result = repl.execute_code(
            "class Counter:\n"
            "    def __init__(self, n):\n"
            "        self.n = n\n"
            "        self.i = 0\n"
            "    def __iter__(self):\n"
            "        return self\n"
            "    def __next__(self):\n"
            "        if self.i >= self.n:\n"
            "            raise StopIteration\n"
            "        self.i += 1\n"
            "        return self.i\n"
            "print(list(Counter(3)))"
        )
        assert result.error is None
        assert "[1, 2, 3]" in result.stdout

    def test_allowed_dunder_enter_exit(self):
        """Allowed dunders still work: __enter__ and __exit__."""
        repl = CCRRepl()
        result = repl.execute_code(
            "class CM:\n"
            "    def __enter__(self):\n"
            "        return 'inside'\n"
            "    def __exit__(self, *a):\n"
            "        return False\n"
            "with CM() as val:\n"
            "    print(val)"
        )
        assert result.error is None
        assert "inside" in result.stdout

    def test_simple_class_still_works(self):
        """Simple class creation without dangerous hooks still works."""
        repl = CCRRepl()
        result = repl.execute_code(
            "class Foo:\n"
            "    pass\n"
            "print('ok')"
        )
        assert result.error is None
        assert "ok" in result.stdout

    def test_class_with_init_still_works(self):
        """Class with __init__ still works."""
        repl = CCRRepl()
        result = repl.execute_code(
            "class Foo:\n"
            "    def __init__(self):\n"
            "        self.x = 42\n"
            "print(Foo().x)"
        )
        assert result.error is None
        assert "42" in result.stdout

    def test_class_with_len_still_works(self):
        """Class with __len__ still works."""
        repl = CCRRepl()
        result = repl.execute_code(
            "class Bag:\n"
            "    def __len__(self):\n"
            "        return 0\n"
            "print(len(Bag()))"
        )
        assert result.error is None
        assert "0" in result.stdout

    def test_dict_attr_blocked(self):
        """obj.__dict__ is blocked at AST level."""
        repl = CCRRepl()
        result = repl.execute_code("x = {}\nx.__dict__")
        assert result.error is not None
        assert "blocked" in result.error.lower()

    def test_code_attr_blocked(self):
        """obj.__code__ is blocked at AST level."""
        repl = CCRRepl()
        result = repl.execute_code(
            "def f(): pass\n"
            "f.__code__\n"
        )
        assert result.error is not None
        assert "blocked" in result.error.lower()

    def test_f_globals_blocked(self):
        """frame.f_globals is blocked at AST level."""
        repl = CCRRepl()
        result = repl.execute_code("x.f_globals")
        assert result.error is not None
        assert "blocked" in result.error.lower()

    def test_co_consts_blocked(self):
        """code.co_consts is blocked at AST level."""
        repl = CCRRepl()
        result = repl.execute_code("x.co_consts")
        assert result.error is not None
        assert "blocked" in result.error.lower()


# ---------------------------------------------------------------------------
# Bug A1: rlm_finalize must NOT destroy the session on a bad variable name
# ---------------------------------------------------------------------------


class TestREPLFinalizePreservation:
    """Verify that _final_var with a bad name returns an error string but leaves
    the REPL intact so subsequent calls can still access computed state."""

    def test_final_var_nonexistent_returns_error_string(self):
        """_final_var('nonexistent_variable') must return a string starting with 'Error:'."""
        repl = CCRRepl()
        repl.execute_code("real_answer = 42")
        result = repl._final_var("nonexistent_variable")
        assert result is not None
        assert str(result).startswith("Error:")

    def test_repl_still_usable_after_failed_final_var(self):
        """After a failed _final_var call the REPL session is unaffected.

        Previously set variables must still be accessible, and new code must
        execute without errors.
        """
        repl = CCRRepl()
        # Set a variable in the session
        repl.execute_code("computed = 99")
        # Attempt to retrieve a name that does not exist
        bad_result = repl._final_var("nonexistent_variable")
        assert str(bad_result).startswith("Error:")
        # The session must still be alive — existing variable reachable
        result = repl.execute_code("print(computed)")
        assert result.error is None
        assert "99" in result.stdout
        # And new code runs fine too
        result2 = repl.execute_code("new_var = computed + 1\nprint(new_var)")
        assert result2.error is None
        assert "100" in result2.stdout

    def test_final_var_succeeds_after_failed_attempt(self):
        """After a failed _final_var, the correct variable is still retrievable."""
        repl = CCRRepl()
        repl.execute_code("answer = 'success'")
        # First call with wrong name — should not clobber state
        bad = repl._final_var("wrong_name")
        assert str(bad).startswith("Error:")
        # Now retrieve with the correct name
        good = repl._final_var("answer")
        assert good == "success"


# ---------------------------------------------------------------------------
# Bug A2: search_repo must forward file_glob to RepoIndex.search
# ---------------------------------------------------------------------------


class TestSearchRepoFileGlob:
    """Verify that _search_repo forwards the file_glob parameter to the underlying
    RepoIndex.search call so that callers can restrict results by extension."""

    def test_search_repo_respects_file_glob(self):
        """Only .py files should be returned when file_glob='*.py'.

        Both a .py file and a .yaml file contain the keyword 'myuniquekeyword'.
        After filtering with '*.py', only the Python file must appear.
        Note: the indexer uses fnmatch for glob matching, which does not support
        '**' recursive patterns — use single-level globs like '*.py' instead.
        """
        keyword = "myuniquekeyword"
        with tempfile.TemporaryDirectory() as tmpdir:
            # Python file containing the keyword
            py_path = os.path.join(tmpdir, "code.py")
            with open(py_path, "w") as f:
                f.write(f"# {keyword}\ndef foo():\n    pass\n")

            # YAML file also containing the keyword — must be excluded
            yaml_path = os.path.join(tmpdir, "config.yaml")
            with open(yaml_path, "w") as f:
                f.write(f"key: {keyword}\n")

            idx = RepoIndex.build(tmpdir)
            repl = CCRRepl(repo_index=idx)

            results = repl._search_repo(keyword, file_glob="*.py")
            result_paths = [r["path"] for r in results]

            # The .yaml file must NOT appear
            assert not any(p.endswith(".yaml") for p in result_paths), (
                f"YAML file leaked through glob filter: {result_paths}"
            )
            # The .py file MUST appear
            assert any(p.endswith(".py") for p in result_paths), (
                f"Expected a .py file in results but got: {result_paths}"
            )

    def test_search_repo_no_glob_returns_all_types(self):
        """With default file_glob='**/*' results include all file types."""
        keyword = "sharedterm"
        with tempfile.TemporaryDirectory() as tmpdir:
            py_path = os.path.join(tmpdir, "module.py")
            with open(py_path, "w") as f:
                f.write(f"# {keyword}\ndef bar():\n    pass\n")

            yaml_path = os.path.join(tmpdir, "settings.yaml")
            with open(yaml_path, "w") as f:
                f.write(f"setting: {keyword}\n")

            idx = RepoIndex.build(tmpdir)
            repl = CCRRepl(repo_index=idx)

            results = repl._search_repo(keyword)  # default glob
            result_paths = [r["path"] for r in results]

            # Both file types should be present (no filtering)
            assert any(p.endswith(".py") for p in result_paths)
            assert any(p.endswith(".yaml") for p in result_paths)


# ---------------------------------------------------------------------------
# Edge cases for repl_security.py primitives
# ---------------------------------------------------------------------------

class TestBoundedStringIO:
    """Tests for _BoundedStringIO — output DoS prevention."""

    def test_write_within_limit(self):
        from ccr.rlm.repl_security import _BoundedStringIO
        buf = _BoundedStringIO(max_chars=100)
        n = buf.write("hello")
        assert n == 5
        assert buf.getvalue() == "hello"

    def test_write_truncates_at_limit(self):
        from ccr.rlm.repl_security import _BoundedStringIO
        buf = _BoundedStringIO(max_chars=10)
        buf.write("12345")
        n = buf.write("6789012345")  # 10 chars, but only 5 remaining
        assert n == 5
        assert buf.getvalue() == "1234567890"

    def test_write_returns_zero_at_capacity(self):
        from ccr.rlm.repl_security import _BoundedStringIO
        buf = _BoundedStringIO(max_chars=5)
        buf.write("12345")
        n = buf.write("more")
        assert n == 0
        assert buf.getvalue() == "12345"

    def test_multiple_writes_accumulate(self):
        from ccr.rlm.repl_security import _BoundedStringIO
        buf = _BoundedStringIO(max_chars=20)
        buf.write("aaa")
        buf.write("bbb")
        buf.write("ccc")
        assert buf.getvalue() == "aaabbbccc"
        assert buf._current_chars == 9

    def test_max_chars_one(self):
        from ccr.rlm.repl_security import _BoundedStringIO
        buf = _BoundedStringIO(max_chars=1)
        buf.write("ab")
        assert buf.getvalue() == "a"

    def test_exact_boundary(self):
        from ccr.rlm.repl_security import _BoundedStringIO
        buf = _BoundedStringIO(max_chars=5)
        n = buf.write("12345")
        assert n == 5
        assert buf.getvalue() == "12345"
        n = buf.write("6")
        assert n == 0

    def test_empty_write(self):
        from ccr.rlm.repl_security import _BoundedStringIO
        buf = _BoundedStringIO(max_chars=10)
        n = buf.write("")
        assert n == 0
        assert buf.getvalue() == ""

    def test_large_default_limit(self):
        from ccr.rlm.repl_security import _BoundedStringIO
        buf = _BoundedStringIO()
        assert buf._max_chars == 10_000_000


class TestRestrictedObject:
    """Tests for _RestrictedObject — blocks __subclasses__ traversal."""

    def test_is_a_class(self):
        from ccr.rlm.repl_security import _RestrictedObject
        assert isinstance(_RestrictedObject, type)

    def test_can_instantiate(self):
        from ccr.rlm.repl_security import _RestrictedObject
        obj = _RestrictedObject()
        assert obj is not None

    def test_used_as_object_replacement_in_builtins(self):
        from ccr.rlm.repl_security import _SAFE_BUILTINS, _RestrictedObject
        assert _SAFE_BUILTINS["object"] is _RestrictedObject

    def test_does_not_expose_real_object_subclasses(self):
        """_RestrictedObject should not have all of object's subclasses."""
        from ccr.rlm.repl_security import _RestrictedObject
        # _RestrictedObject.__subclasses__() should be empty or minimal,
        # unlike object.__subclasses__() which returns hundreds of classes
        subs = _RestrictedObject.__subclasses__()
        assert len(subs) < 5  # real object has hundreds


class TestValidateAstEdgeCases:
    """Edge cases for _validate_ast — AST-level security."""

    def test_empty_code(self):
        from ccr.rlm.repl_security import _validate_ast
        # Should not raise
        _validate_ast("")

    def test_comment_only(self):
        from ccr.rlm.repl_security import _validate_ast
        _validate_ast("# just a comment")

    def test_whitespace_only(self):
        from ccr.rlm.repl_security import _validate_ast
        _validate_ast("   \n\n   ")

    def test_syntax_error_returns_none(self):
        from ccr.rlm.repl_security import _validate_ast
        # Syntax errors should not raise (let exec handle them)
        _validate_ast("def foo(:")  # invalid syntax

    def test_nested_class_dangerous_func(self):
        from ccr.rlm.repl_security import _validate_ast
        code = """
class Outer:
    class Inner:
        def __init_subclass__(cls):
            pass
"""
        with pytest.raises(PermissionError, match="__init_subclass__"):
            _validate_ast(code)

    def test_allowed_dunder_passes(self):
        from ccr.rlm.repl_security import _validate_ast
        # __len__ is in the allowlist
        _validate_ast("x = obj.__len__()")

    def test_unknown_dunder_blocked(self):
        from ccr.rlm.repl_security import _validate_ast
        # __secret_method__ is not in allowlist or dangerous list
        with pytest.raises(PermissionError, match="__secret_method__"):
            _validate_ast("x = obj.__secret_method__()")


class TestSafeImportEdgeCases:
    """Edge cases for _safe_import."""

    def test_empty_module_name(self):
        from ccr.rlm.repl_security import _safe_import
        with pytest.raises(ImportError, match="blocked"):
            _safe_import("")

    def test_submodule_of_allowed(self):
        from ccr.rlm.repl_security import _safe_import
        # collections.abc should be allowed (top_level = "collections")
        mod = _safe_import("collections.abc")
        assert mod is not None

    def test_blocked_module_with_dots(self):
        from ccr.rlm.repl_security import _safe_import
        with pytest.raises(ImportError, match="blocked"):
            _safe_import("os.path")


class TestRestrictedOpenEdgeCases:
    """Edge cases for _make_restricted_open."""

    def test_empty_allowed_dirs(self):
        from ccr.rlm.repl_security import _make_restricted_open
        restricted = _make_restricted_open([])
        with pytest.raises(PermissionError, match="access denied"):
            restricted("/tmp/anything.txt")

    def test_symlink_resolved(self):
        from ccr.rlm.repl_security import _make_restricted_open
        with tempfile.TemporaryDirectory() as d:
            # Create a file and a symlink to it
            target = os.path.join(d, "real.txt")
            with open(target, "w") as f:
                f.write("content")
            link = os.path.join(d, "link.txt")
            os.symlink(target, link)
            restricted = _make_restricted_open([d])
            # Symlink within allowed dir should work
            with restricted(link) as f:
                assert f.read() == "content"

    def test_path_traversal_blocked(self):
        from ccr.rlm.repl_security import _make_restricted_open
        with tempfile.TemporaryDirectory() as d:
            restricted = _make_restricted_open([d])
            # Attempt to traverse out of allowed dir
            with pytest.raises(PermissionError):
                restricted(os.path.join(d, "..", "..", "etc", "passwd"))


class TestNoGlobalImportMutation:
    """Verify builtins.__import__ is NOT mutated during REPL execution."""

    def test_builtins_import_unchanged_after_execution(self):
        original = _builtins_module.__import__
        repl = CCRRepl()
        repl.execute_code("x = 1 + 2")
        assert _builtins_module.__import__ is original

    def test_blocked_import_does_not_leak_global_patch(self):
        original = _builtins_module.__import__
        repl = CCRRepl()
        repl.execute_code("try:\n    import subprocess\nexcept ImportError:\n    pass")
        assert _builtins_module.__import__ is original


# ── Round-5: _final_var stdout leak prevention ────────────────────────


class TestFinalVarStdoutLeak:
    """Verify _final_var does not write to stdout (MCP transport safety)."""

    def test_direct_call_no_stdout(self):
        """Direct _final_var call produces no stdout output."""
        import io
        import sys
        repl = CCRRepl()
        old_stdout = sys.stdout
        capture = io.StringIO()
        sys.stdout = capture
        try:
            repl._final_var("nonexistent")
        finally:
            sys.stdout = old_stdout
        assert capture.getvalue() == "", "stdout should be empty — error goes to logger"

    def test_error_in_stderr_not_stdout_via_execute(self):
        """Via execute_code, error does NOT appear in stdout."""
        repl = CCRRepl()
        result = repl.execute_code("FINAL_VAR('nonexistent')")
        assert "not found" not in result.stdout.lower()


class TestFinalVarObjectEdgeCases:
    """Edge cases for _final_var with non-standard objects."""

    def test_non_serializable_object(self):
        """_final_var handles non-dict/list objects via str() fallback."""
        repl = CCRRepl()
        # set() is available in builtins but not JSON-serializable
        repl.execute_code("obj = set([1, 2, 3])")
        result = repl.execute_code("FINAL_VAR('obj')")
        # Should return str(obj) since set is not dict/list
        assert result.final_answer is not None
        # str representation of a set
        assert "1" in result.final_answer

    def test_tuple_uses_str_not_json(self):
        """_final_var with tuple — not dict/list so uses str()."""
        repl = CCRRepl()
        repl.execute_code("t = (1, 'hello', 3.14)")
        result = repl.execute_code("FINAL_VAR('t')")
        assert result.final_answer is not None
        assert "hello" in result.final_answer

    def test_very_large_variable(self):
        """_final_var handles very large strings without crash."""
        repl = CCRRepl()
        repl.execute_code("big = 'x' * 1_000_000")
        result = repl.execute_code("FINAL_VAR('big')")
        assert result.final_answer is not None
        assert len(result.final_answer) == 1_000_000
