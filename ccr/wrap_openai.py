"""CCR OpenAI SDK wrapper — injects project memory into API calls.

Usage:
    import openai
    from ccr.wrap_openai import wrap_openai

    client = openai.OpenAI()
    wrapped = wrap_openai(client)

    # All calls now automatically prepend project context
    response = wrapped.chat.completions.create(...)
"""

from __future__ import annotations

import os
from typing import Any


def wrap_openai(client: Any, project_root: str | None = None) -> Any:
    """Wrap an OpenAI client to inject CCR context into every request.

    Args:
        client: An openai.OpenAI() instance.
        project_root: Project directory with .ccr/ memory. Defaults to cwd.

    Returns:
        A proxy object that intercepts chat.completions.create().
    """
    if project_root is None:
        project_root = os.getcwd()

    # Load context from .ccr/
    context = _load_context(project_root)

    class _WrappedClient:
        def __init__(self, inner: Any):
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    class _WrappedChat:
        def __init__(self, inner: Any):
            self._inner = inner
            self.completions = _WrappedCompletions(inner.completions, context)

    class _WrappedCompletions:
        def __init__(self, inner: Any, context: str):
            self._inner = inner
            self._context = context

        def create(self, *args: Any, **kwargs: Any) -> Any:
            messages = kwargs.get("messages", [])
            if messages and self._context:
                # Prepend context as system message or prefix to first user message
                system_msg = next((m for m in messages if m.get("role") == "system"), None)
                if system_msg:
                    system_msg["content"] = f"{self._context}\n\n{system_msg['content']}"
                else:
                    messages.insert(0, {"role": "system", "content": self._context})
                kwargs["messages"] = messages
            return self._inner.create(*args, **kwargs)

    wrapped = _WrappedClient(client)
    wrapped.chat = _WrappedChat(client.chat)
    return wrapped


def _load_context(project_root: str) -> str:
    """Load CCR context for the project."""
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ccr.core.memory import MemoryManager
        from ccr.core.types import CCRConfig
        mem = MemoryManager(project_root, CCRConfig())
        if os.path.isdir(mem.ccr_root):
            return mem.get_context(level=2)
    except Exception:
        pass
    return ""
