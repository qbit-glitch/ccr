"""CCR MCP package — split from monolithic mcp_server.py.

Importing this package triggers @mcp.tool registration for all tool modules.
"""

from ccr.mcp.server import main, mcp  # noqa: F401

# Import tool modules to trigger @mcp.tool decorator registration
import ccr.mcp.gcc_tools  # noqa: F401
import ccr.mcp.ace_tools  # noqa: F401
import ccr.mcp.rlm_tools  # noqa: F401
import ccr.mcp.index_tools  # noqa: F401

__all__ = ["main", "mcp"]
