"""Ollama adapter — file-watcher + CLI wrapper for local LLMs.

Ollama has no native MCP or hook support. CCR integrates via:
1. A wrapper script `ollama-ccr` that injects context before calling `ollama run`
2. A context file `.ccr/context_inject.md` that gets prepended to prompts
"""

from __future__ import annotations

import os
import shutil

from ccr.adapters import BaseAgentAdapter, InstallResult, UninstallResult, IntegrationLevel


class OllamaAdapter(BaseAgentAdapter):
    name = "ollama"
    display_name = "Ollama"

    _BIN_DIR = os.path.expanduser("~/.ccr/bin")

    def is_installed(self) -> bool:
        return shutil.which("ollama") is not None

    def supports_mcp(self) -> bool:
        return False

    def supports_hooks(self) -> bool:
        return False

    def integration_level(self) -> int:
        return IntegrationLevel.FILE_WATCHER

    def install(self, python_exe: str, hooks_dir: str, ccr_pkg: str) -> InstallResult:
        os.makedirs(self._BIN_DIR, exist_ok=True)

        wrapper_path = os.path.join(self._BIN_DIR, "ollama-ccr")
        with open(wrapper_path, "w", encoding="utf-8") as f:
            f.write(self._wrapper_script(python_exe))
        os.chmod(wrapper_path, 0o755)

        return InstallResult(
            success=True,
            message=f"Ollama wrapper created: {wrapper_path}",
            files_created=[wrapper_path],
        )

    def uninstall(self) -> UninstallResult:
        wrapper_path = os.path.join(self._BIN_DIR, "ollama-ccr")
        if os.path.isfile(wrapper_path):
            os.remove(wrapper_path)
            return UninstallResult(
                success=True,
                message=f"Removed {wrapper_path}",
                files_removed=[wrapper_path],
            )
        return UninstallResult(success=True, message="No Ollama wrapper found")

    def context_format(self) -> str:
        return "plain"

    def _wrapper_script(self, python_exe: str) -> str:
        import shlex

        py = shlex.quote(python_exe)
        return f'''#!/bin/bash
# CCR-Ollama wrapper — injects project memory before each prompt
# Usage: ollama-ccr <model> [ollama-args...]
# Example: ollama-ccr llama3.2 "Explain this codebase"

set -e

MODEL="${{1:-llama3.2}}"
shift || true

PROJECT_ROOT="$(pwd)"

# Auto-init .ccr/ if missing
if [ ! -d "$PROJECT_ROOT/.ccr" ]; then
    echo "[CCR] Initializing memory in $PROJECT_ROOT..."
    CCR_PROJECT_ROOT="$PROJECT_ROOT" {py} -m ccr.cli init "$PROJECT_ROOT" 2>/dev/null || true
fi

# Build context injection
CONTEXT_FILE="$PROJECT_ROOT/.ccr/context_inject.md"
if [ -f "$CONTEXT_FILE" ]; then
    CONTEXT=$(cat "$CONTEXT_FILE")
fi

# If no prompt provided, enter interactive mode with context as system message
if [ $# -eq 0 ]; then
    # Interactive mode — inject context as system message
    if [ -n "$CONTEXT" ]; then
        echo "$CONTEXT" | ollama run "$MODEL" -
    else
        ollama run "$MODEL"
    fi
else
    # Single prompt mode — prepend context
    PROMPT="$*"
    if [ -n "$CONTEXT" ]; then
        PROMPT="$CONTEXT

$PROMPT"
    fi
    echo "$PROMPT" | ollama run "$MODEL" -
fi
'''
