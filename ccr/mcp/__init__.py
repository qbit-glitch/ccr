"""CCR MCP package — split from monolithic mcp_server.py.

Importing this package triggers @mcp.tool registration for all tool modules.
"""

from ccr.mcp.server import main, mcp  # noqa: F401

# Import tool modules to trigger @mcp.tool decorator registration
import ccr.mcp.gcc_branch_tools  # noqa: F401
import ccr.mcp.gcc_search_tools  # noqa: F401
import ccr.mcp.gcc_todo_tools  # noqa: F401
import ccr.mcp.gcc_tools  # noqa: F401
import ccr.mcp.ace_tools  # noqa: F401
import ccr.mcp.ace_llm_tools  # noqa: F401  (ace_generate_bullets, ace_evolve_schema)
import ccr.mcp.ace_schema_tools  # noqa: F401  (ace_evolve_schema MCE)
import ccr.mcp.rlm_tools  # noqa: F401
import ccr.mcp.index_tools  # noqa: F401
import ccr.mcp.session_tools  # noqa: F401  (session_log_turn, session_get_history, session_search, session_export)

__all__ = ["main", "mcp"]
