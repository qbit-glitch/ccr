"""Security primitives for the CCR REPL sandbox.

Contains restricted builtins, AST validation, safe imports, and all
module-level security infrastructure used by CCRRepl. Extracted from
repl.py to keep the main REPL class file under the 800-line limit.
"""

from __future__ import annotations

import ast
import builtins
import io
import os
import re
from typing import Any


# Modules ALLOWED for import inside the REPL sandbox.
# Allowlist is safer than denylist: new stdlib modules are blocked by default,
# and no new dangerous module can slip through by being added to Python.
_ALLOWED_MODULES = frozenset({
    # Math/numeric
    "math", "decimal", "fractions", "statistics", "random",
    # String/text
    "string", "re", "textwrap", "unicodedata", "difflib",
    # Data structures
    "collections", "functools", "itertools", "operator",
    # Date/time
    "datetime", "time", "calendar", "zoneinfo",
    # Serialization (safe subset)
    "json", "csv", "base64", "hashlib", "hmac",
    # Type system
    "dataclasses", "enum", "typing", "types", "abc",
    # Utilities
    "copy", "pprint", "numbers",
    "heapq", "bisect", "array",
    "contextlib", "warnings",
})


# Store the real __import__ once at module load time, before any patching.
_REAL_IMPORT = builtins.__import__


def _safe_import(name: str, *args, **kwargs):
    """Restricted __import__ that only allows safe modules (allowlist)."""
    top_level = name.split(".")[0]
    if top_level not in _ALLOWED_MODULES:
        raise ImportError(
            f"Module '{name}' is blocked in the CCR sandbox (not in allowlist). "
            f"Allowed modules: {', '.join(sorted(_ALLOWED_MODULES))}"
        )
    return _REAL_IMPORT(name, *args, **kwargs)


def _make_restricted_open(allowed_dirs: list[str]):
    """Create a restricted open() that only allows access to specific directories.

    Args:
        allowed_dirs: List of directory paths that the sandbox is allowed to access.
                      Paths are resolved to their real (canonical) form for comparison.
    """
    # Use os.path directly (already imported at module level) to avoid
    # going through builtins.__import__ which may be patched during sandbox exec.
    osp = os.path
    _real_open = open

    def restricted_open(file, mode='r', *args, **kwargs):
        resolved = osp.realpath(str(file))
        for d in allowed_dirs:
            if resolved.startswith(osp.realpath(d) + osp.sep) or resolved == osp.realpath(d):
                return _real_open(resolved, mode, *args, **kwargs)  # H1: use resolved, not file
        raise PermissionError(f"REPL sandbox: access denied to {file}")

    restricted_open.__name__ = "restricted_open"
    restricted_open.__doc__ = "Sandbox-restricted open(). Only allows access to project root and temp dir."
    return restricted_open


# --- C1: Safe type() wrapper — blocks 3-arg metaclass form ---
def _safe_type(*args):
    """type(obj) is allowed; type(name, bases, dict) is blocked in the sandbox."""
    if len(args) != 1:
        raise TypeError("type() with multiple arguments is blocked in the sandbox")
    return type(args[0])

_safe_type.__name__ = "type"


# --- C2: Restricted object proxy — blocks __subclasses__ traversal ---
class _RestrictedObject:
    """Proxy for object that blocks __subclasses__ and other dangerous methods."""
    pass


# --- C3: Safe getattr/hasattr — blocks dunder attribute access ---
_DUNDER_RE = re.compile(r'^__.*__$')


def _safe_getattr(obj, name, *default):
    """getattr() that blocks access to dunder attributes in the sandbox."""
    if isinstance(name, str) and _DUNDER_RE.match(name):
        raise AttributeError(f"Access to dunder attribute '{name}' is blocked in the sandbox")
    if default:
        return getattr(obj, name, default[0])
    return getattr(obj, name)


def _safe_hasattr(obj, name):
    """hasattr() that returns False for dunder attributes in the sandbox."""
    if isinstance(name, str) and _DUNDER_RE.match(name):
        return False
    return hasattr(obj, name)


# --- C1-C4: AST-level sandbox hardening ---
# Direct attribute syntax (obj.__class__) bypasses _safe_getattr,
# so we must inspect the AST before execution.

_ALLOWED_DUNDERS = frozenset({
    '__name__', '__doc__', '__str__', '__repr__', '__len__',
    '__init__', '__enter__', '__exit__', '__iter__', '__next__',
    '__getitem__', '__setitem__', '__delitem__', '__contains__',
    '__eq__', '__ne__', '__lt__', '__gt__', '__le__', '__ge__',
    '__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__',
    '__mod__', '__pow__', '__neg__', '__pos__', '__abs__',
    '__and__', '__or__', '__xor__', '__invert__',
    '__radd__', '__rsub__', '__rmul__', '__rtruediv__',
    '__iadd__', '__isub__', '__imul__', '__itruediv__',
    '__hash__', '__bool__', '__int__', '__float__', '__complex__',
    '__index__', '__call__',
})

_DANGEROUS_DUNDERS = frozenset({
    '__class__', '__bases__', '__mro__', '__subclasses__',
    '__globals__', '__code__', '__closure__', '__func__',
    '__self__', '__dict__', '__slots__',
    '__traceback__', '__context__', '__cause__', '__suppress_context__',
    '__builtins__', '__import__', '__loader__', '__spec__',
    '__file__', '__path__', '__package__', '__qualname__',
    '__module__', '__annotations__', '__wrapped__',
    '__init_subclass__', '__set_name__', '__class_getitem__',
    '__getattribute__', '__getattr__', '__setattr__', '__delattr__',
    'gi_frame', 'gi_code', 'gi_yieldfrom',
    'cr_frame', 'cr_code', 'cr_origin',
    'ag_frame', 'ag_code',
    'tb_frame', 'tb_next', 'tb_lineno', 'tb_lasti',
    'f_globals', 'f_locals', 'f_builtins', 'f_code', 'f_back',
    'co_consts', 'co_names', 'co_code',
})

_DANGEROUS_FUNC_DEFS = frozenset({
    '__init_subclass__', '__set_name__', '__del__',
    '__getattr__', '__getattribute__',
})


def _validate_ast(code: str) -> None:
    """Reject code that accesses dangerous dunder attributes.

    Direct attribute syntax (obj.__class__) bypasses _safe_getattr,
    so we must inspect the AST before execution.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return  # Let exec handle syntax errors naturally

    for node in ast.walk(tree):
        # Block dangerous attribute access: obj.__class__, obj.__globals__, etc.
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr in _DANGEROUS_DUNDERS:
                raise PermissionError(
                    f"Access to '{attr}' is blocked in the CCR sandbox"
                )
            # Block any remaining dunder not in allowlist
            if attr.startswith('__') and attr.endswith('__') and attr not in _ALLOWED_DUNDERS:
                raise PermissionError(
                    f"Access to dunder attribute '{attr}' is blocked in the CCR sandbox"
                )

        # Block dangerous function definitions (__init_subclass__, __set_name__, etc.)
        if isinstance(node, ast.FunctionDef) and node.name in _DANGEROUS_FUNC_DEFS:
            raise PermissionError(
                f"Defining '{node.name}' is blocked in the CCR sandbox"
            )


# Safe builtins — blocks eval/exec/compile/input, allows everything else.
# C1: type replaced with _safe_type (blocks 3-arg form)
# C2: object replaced with _RestrictedObject (blocks __subclasses__)
# C3: getattr/hasattr replaced with safe versions (blocks dunder access)
# C4: super, property, staticmethod, classmethod removed (descriptor abuse)
_SAFE_BUILTINS: dict[str, Any] = {
    "print": print, "len": len, "str": str, "int": int, "float": float,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "bool": bool,
    "type": _safe_type, "isinstance": isinstance, "issubclass": issubclass,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "sorted": sorted, "reversed": reversed, "range": range,
    "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
    "any": any, "all": all, "pow": pow, "divmod": divmod,
    "chr": chr, "ord": ord, "hex": hex, "bin": bin, "oct": oct,
    "repr": repr, "ascii": ascii, "format": format, "hash": hash, "id": id,
    "iter": iter, "next": next, "slice": slice, "callable": callable,
    "hasattr": _safe_hasattr, "getattr": _safe_getattr,
    "bytes": bytes, "bytearray": bytearray,
    "complex": complex, "object": _RestrictedObject,
    "__build_class__": __build_class__,  # Required for 'class' statement
    "__import__": _safe_import,
    # Exceptions
    "Exception": Exception, "BaseException": BaseException,
    "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
    "IndexError": IndexError, "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError, "OSError": OSError, "IOError": IOError,
    "RuntimeError": RuntimeError, "NameError": NameError, "ImportError": ImportError,
    "StopIteration": StopIteration, "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError, "ArithmeticError": ArithmeticError,
    "LookupError": LookupError, "Warning": Warning,
    # Blocked (set to None to raise clear errors)
    "input": None, "eval": None, "compile": None, "exec": None,
    "globals": None, "locals": None,
}

class _BoundedStringIO(io.StringIO):
    """StringIO with a maximum size limit to prevent unbounded output DoS (M1).

    Tracks by character count (not bytes) since StringIO operates on str.
    This avoids the char/byte mismatch where s[:remaining] could truncate
    differently than the byte count suggests for multi-byte characters.
    """

    def __init__(self, *args, max_chars: int = 10_000_000, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_chars = max_chars
        self._current_chars = 0

    def write(self, s: str) -> int:
        if self._current_chars + len(s) > self._max_chars:
            remaining = self._max_chars - self._current_chars
            if remaining <= 0:
                return 0
            s = s[:remaining]
        self._current_chars += len(s)
        return super().write(s)


# Default execution timeout in seconds
_DEFAULT_TIMEOUT_SECONDS = 30

# Names that must not be overwritten by user code
_RESERVED_NAMES = {
    "llm_query", "rlm_query", "llm_query_batched", "rlm_query_batched",
    "FINAL_VAR", "SHOW_VARS",
    "get_file", "search_repo", "estimate_tokens", "context",
}


def _run_in_namespace(code: str, namespace: dict) -> None:
    """Execute code string in the given namespace.

    This is the core of the RLM paper's REPL mechanism — the model writes
    Python code that runs against the repo context programmatically,
    rather than loading everything into the LLM's context window.

    The namespace is restricted via _SAFE_BUILTINS (no eval/compile/input).
    """
    _validate_ast(code)  # AST-level security check (C1-C4)
    # Python's built-in code execution in a restricted namespace
    builtins.__dict__  # ensure builtins loaded
    co = builtins.compile(code, "<ccr-repl>", "exec")
    builtins.eval(co, namespace, namespace)  # noqa: S307
