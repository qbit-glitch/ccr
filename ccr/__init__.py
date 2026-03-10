"""CCR — Claude Context Reducer.

Token-efficient middleware for Claude Code.
Combines RLM (Recursive Language Models) REPL-based context packing
with GCC (Git Context Controller) version-controlled memory.
"""

__version__ = "0.1.0"

from ccr.core.engine import CCREngine
from ccr.core.exceptions import (
    CCRError,
    ConfigError,
    ModelAuthError,
    ModelConnectionError,
    ModelError,
    ModelRateLimitError,
    ModelTimeoutError,
    PackingError,
    PlaybookError,
    RoutingError,
)
from ccr.core.hooks import HookManager
from ccr.core.memory import MemoryManager
from ccr.core.types import CCREngineConfig, CCRConfig, RouterConfig, RLMConfig
from ccr.gateway import CCRGateway
from ccr.rlm import CCRRlm, CCRRepl

__all__ = [
    "CCREngine",
    "CCRError",
    "CCRGateway",
    "ConfigError",
    "MemoryManager",
    "ModelAuthError",
    "ModelConnectionError",
    "ModelError",
    "ModelRateLimitError",
    "ModelTimeoutError",
    "CCREngineConfig",
    "CCRConfig",
    "PackingError",
    "PlaybookError",
    "RouterConfig",
    "RoutingError",
]
