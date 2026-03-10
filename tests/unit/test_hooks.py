"""Tests for the CCR hook system."""

import pytest

from ccr.core.hooks import HookManager, create_default_hooks
from ccr.core.types import HookEvent


class TestHookManager:
    def test_register_and_fire(self):
        hooks = HookManager()
        results = []
        hooks.register(HookEvent.SESSION_START, lambda ctx: results.append("fired"))
        hooks.fire(HookEvent.SESSION_START)
        assert results == ["fired"]

    def test_multiple_handlers(self):
        hooks = HookManager()
        results = []
        hooks.register(HookEvent.STOP, lambda ctx: results.append("a"))
        hooks.register(HookEvent.STOP, lambda ctx: results.append("b"))
        hooks.fire(HookEvent.STOP)
        assert results == ["a", "b"]

    def test_context_passed_to_handler(self):
        hooks = HookManager()
        received = {}

        def handler(ctx):
            received.update(ctx)

        hooks.register(HookEvent.POST_TOOL_USE, handler)
        hooks.fire(HookEvent.POST_TOOL_USE, {"tool_name": "edit", "file_path": "a.py"})
        assert received["tool_name"] == "edit"

    def test_handler_error_doesnt_crash(self):
        hooks = HookManager()
        hooks.register(HookEvent.STOP, lambda ctx: 1 / 0)
        results = hooks.fire(HookEvent.STOP)
        assert results == [None]

    def test_has_handlers(self):
        hooks = HookManager()
        assert not hooks.has_handlers(HookEvent.SESSION_START)
        hooks.register(HookEvent.SESSION_START, lambda ctx: None)
        assert hooks.has_handlers(HookEvent.SESSION_START)

    def test_fire_returns_results(self):
        hooks = HookManager()
        hooks.register(HookEvent.SESSION_START, lambda ctx: "result1")
        hooks.register(HookEvent.SESSION_START, lambda ctx: "result2")
        results = hooks.fire(HookEvent.SESSION_START)
        assert results == ["result1", "result2"]

    def test_no_handlers_returns_empty(self):
        hooks = HookManager()
        results = hooks.fire(HookEvent.STOP)
        assert results == []


class TestDefaultHooks:
    def test_create_default_hooks(self):
        from unittest.mock import MagicMock

        mem = MagicMock()
        mem.get_session_context.return_value = "context"
        mem.get_active_branch.return_value = "main"
        mem.log_ota = MagicMock()
        mem._read_file = MagicMock(return_value="")
        mem._get_log_path = MagicMock(return_value="/tmp/log.md")
        mem._locks = {}

        hooks = create_default_hooks(mem)
        assert hooks.has_handlers(HookEvent.SESSION_START)
        assert hooks.has_handlers(HookEvent.POST_TOOL_USE)
        assert hooks.has_handlers(HookEvent.PRE_COMPACT)
        assert hooks.has_handlers(HookEvent.STOP)

    def test_session_start_returns_context(self):
        from unittest.mock import MagicMock

        mem = MagicMock()
        mem.get_session_context.return_value = "project context"
        mem.log_ota = MagicMock()

        hooks = create_default_hooks(mem)
        results = hooks.fire(HookEvent.SESSION_START, {})
        assert results[0] == "project context"
