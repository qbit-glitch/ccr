"""OpenAI adapter — SDK wrapper + CLI prefix for OpenAI API users.

Provides:
1. Python SDK wrapper: ccr.wrap_openai(client) → wrapped client with auto-context
2. CLI prefix: ccr-openai <prompt> — prepends context and calls OpenAI API
"""

from __future__ import annotations

import os

from ccr.adapters import BaseAgentAdapter, InstallResult, UninstallResult, IntegrationLevel


class OpenAIAdapter(BaseAgentAdapter):
    """OpenAI API / ChatGPT integration via SDK wrapper.

    No native hooks or MCP. Context is injected at the Python/SDK level
    or via a CLI wrapper script.
    """

    name = "openai"
    display_name = "OpenAI API"

    _BIN_DIR = os.path.expanduser("~/.ccr/bin")

    def is_installed(self) -> bool:
        """Detect via openai Python package or OPENAI_API_KEY env var."""
        try:
            import openai  # noqa: F401
            return True
        except ImportError:
            return "OPENAI_API_KEY" in os.environ

    def supports_mcp(self) -> bool:
        return False

    def supports_hooks(self) -> bool:
        return False

    def integration_level(self) -> int:
        return IntegrationLevel.SDK_WRAPPER

    def install(self, python_exe: str, hooks_dir: str, ccr_pkg: str) -> InstallResult:
        os.makedirs(self._BIN_DIR, exist_ok=True)

        # Write CLI wrapper script
        wrapper_path = os.path.join(self._BIN_DIR, "ccr-openai")
        with open(wrapper_path, "w", encoding="utf-8") as f:
            f.write(self._wrapper_script(python_exe))
        os.chmod(wrapper_path, 0o755)

        # Write SDK wrapper module stub (full implementation in ccr/wrap_openai.py)
        wrap_mod_path = os.path.join(ccr_pkg, "wrap_openai.py")
        if not os.path.isfile(wrap_mod_path):
            with open(wrap_mod_path, "w", encoding="utf-8") as f:
                f.write(self._sdk_wrapper_source())

        return InstallResult(
            success=True,
            message=f"OpenAI wrapper: {wrapper_path}\nSDK module: {wrap_mod_path}",
            files_created=[wrapper_path, wrap_mod_path],
        )

    def uninstall(self) -> UninstallResult:
        removed = []
        wrapper_path = os.path.join(self._BIN_DIR, "ccr-openai")
        if os.path.isfile(wrapper_path):
            os.remove(wrapper_path)
            removed.append(wrapper_path)

        wrap_mod_path = os.path.join(os.path.dirname(os.path.expanduser("~/.ccr")), "wrap_openai.py")
        # Note: wrap_openai.py lives in the package dir, don't remove it on uninstall

        return UninstallResult(
            success=True,
            message=f"Removed OpenAI CLI wrapper",
            files_removed=removed,
        )

    def context_format(self) -> str:
        return "markdown"

    def _wrapper_script(self, python_exe: str) -> str:
        import shlex

        py = shlex.quote(python_exe)
        return f'''#!/bin/bash
# CCR-OpenAI wrapper — injects project memory before calling OpenAI API
# Usage: ccr-openai "Your prompt here"
# Requires: OPENAI_API_KEY environment variable

set -e

PROJECT_ROOT="$(pwd)"

# Auto-init .ccr/ if missing
if [ ! -d "$PROJECT_ROOT/.ccr" ]; then
    echo "[CCR] Initializing memory in $PROJECT_ROOT..."
    CCR_PROJECT_ROOT="$PROJECT_ROOT" {py} -m ccr.cli init "$PROJECT_ROOT" 2>/dev/null || true
fi

# Build context
CONTEXT=""
CONTEXT_FILE="$PROJECT_ROOT/.ccr/context_inject.md"
if [ -f "$CONTEXT_FILE" ]; then
    CONTEXT=$(cat "$CONTEXT_FILE")
fi

PROMPT="$*"
if [ -z "$PROMPT" ]; then
    echo "Usage: ccr-openai <prompt>"
    exit 1
fi

# Prepend context
if [ -n "$CONTEXT" ]; then
    PROMPT="$CONTEXT

$PROMPT"
fi

# Call OpenAI API via Python one-liner
{py} -c "
import os, sys
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'),
    messages=[
        {{'role': 'system', 'content': 'You are a helpful coding assistant with project memory.'}},
        {{'role': 'user', 'content': sys.argv[1]}},
    ],
)
print(response.choices[0].message.content)
" "$PROMPT"
'''

    def _sdk_wrapper_source(self) -> str:
        return '''"""CCR OpenAI SDK wrapper — injects project memory into API calls.

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
                    system_msg["content"] = f"{self._context}\\n\\n{system_msg['content']}"
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
'''
