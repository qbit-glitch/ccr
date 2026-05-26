"""CCR — Claude Context Reducer.

Persistent memory (GCC), self-evolving playbooks (ACE), and sandboxed
REPL (RLM) for Claude Code via MCP.  No API keys required.
"""

__version__ = "0.4.0"

# Core symbols used by hooks and MCP server (no heavy deps)
from ccr.core.memory import MemoryManager
from ccr.core.types import CCRConfig

__all__ = [
    "__version__",
    "CCRConfig",
    "MemoryManager",
]
