"""Tests for the CCR REPL (sandboxed Python execution environment)."""

import builtins as _builtins_module
import os
import tempfile

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
