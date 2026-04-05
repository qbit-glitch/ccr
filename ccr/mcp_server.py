"""Backward-compatible entry point. Delegates to ccr.mcp package.

All state, helpers, and tool functions now live in ccr/mcp/ submodules.
This shim re-exports everything so existing imports continue to work:
    from ccr.mcp_server import gcc_commit, _init, mcp
    import ccr.mcp_server as mcp_mod; mcp_mod._memory = ...
"""

# Re-export the FastMCP instance and main entry point
from ccr.mcp import main  # noqa: F401
from ccr.mcp.server import mcp  # noqa: F401

# Re-export all globals (tests mutate these via `mcp_mod._memory = None` etc.)
from ccr.mcp.server import (  # noqa: F401
    _state_lock,
    _get_sub_client,
    _init,
    _atomic_write,
    _load_playbook,
    _save_playbook,
    _load_global_playbook,
    _save_global_playbook,
    _load_schema,
    _load_schema_history,
    _save_schema,
    _ensure_global_playbook,
    _ensure_memory,
    _ensure_playbook,
    _ensure_index,
    _ensure_session_store,
    _extract_patterns_from_commit,
    _resolve_playbook,
    _serialize_playbook,
)

# Re-export all GCC tool functions
from ccr.mcp.gcc_tools import (  # noqa: F401
    gcc_commit,
    gcc_branch,
    gcc_merge,
    gcc_context,
    gcc_links,
    gcc_clusters,
    gcc_triples,
    gcc_evolve_memory,
    gcc_log_ota,
    gcc_status,
    gcc_consolidate,
    gcc_patterns,
    gcc_scratchpad,
)

# Re-export all ACE tool functions and helpers
from ccr.mcp.ace_tools import (  # noqa: F401
    ace_get_playbook,
    ace_apply_delta,
    ace_update_counters,
    ace_find_similar,
    ace_prune,
    ace_generate_bullets,
    ace_evolve_from_failures,
    ace_evolve_schema,
    _word_jaccard,
    _semantic_or_jaccard,
    _auto_synthesize_skills,
    _ace_generator,
    _ace_reflector,
    _ace_curator,
    _run_ace_pipeline,
)

# Re-export all RLM tool functions
from ccr.mcp.rlm_tools import (  # noqa: F401
    rlm_init,
    rlm_execute,
    rlm_finalize,
    _summarize_stdout,
    _get_repl_for_session,
    _get_session_age_warning,
    _cleanup_session,
)

# Re-export all Index tool functions
from ccr.mcp.index_tools import (  # noqa: F401
    index_build,
    index_search,
    index_status,
)

# Re-export all Session Logger tool functions
from ccr.mcp.session_tools import (  # noqa: F401
    session_log_turn,
    session_get_history,
    session_search,
    session_export,
)

# ---------------------------------------------------------------------------
# Mutable global proxies — tests do `mcp_mod._memory = None` etc.
# We need attribute access on THIS module to read/write the REAL globals
# in ccr.mcp.server. Python module attribute access doesn't auto-proxy,
# so we use __getattr__ and a custom setter mechanism.
# ---------------------------------------------------------------------------
import sys as _sys
import ccr.mcp.server as _server_mod
import ccr.mcp.gcc_tools as _gcc_mod
import ccr.mcp.ace_tools as _ace_mod
import ccr.mcp.rlm_tools as _rlm_mod
import ccr.mcp.index_tools as _index_mod

# Mutable globals that tests read/write via `import ccr.mcp_server as mcp_mod`
# Maps attribute name -> canonical module where it lives.
_PROXIED_ATTRS: dict[str, object] = {}

# State globals + helper functions → ccr.mcp.server
for _name in (
    "_project_root", "_memory", "_playbook", "_playbook_path",
    "_failure_lessons_path", "_global_playbook", "_global_playbook_path",
    "_global_failure_lessons_path", "_repo_index", "_repl",
    "_repl_sessions", "_repl_sessions_lock", "_repl_session_ttl",
    "_schema_path", "_global_schema_path", "_embedding_model",
    "_embeddings_path", "_chunk_embeddings_path", "_scratchpad",
    "_triple_store", "_state_lock",
    "_session_store", "_session_db_path", "_current_session_id",
    "_get_sub_client", "_extract_patterns_from_commit",
    "_init", "_atomic_write", "_load_playbook", "_save_playbook",
    "_load_global_playbook", "_save_global_playbook",
    "_load_schema", "_load_schema_history", "_save_schema",
    "_ensure_global_playbook", "_ensure_memory", "_ensure_playbook",
    "_ensure_index", "_ensure_session_store", "_resolve_playbook", "_serialize_playbook",
):
    _PROXIED_ATTRS[_name] = _server_mod

# ACE helpers → ccr.mcp.ace_tools (tests patch these via mcp_mod)
for _name in (
    "_run_ace_pipeline", "_ace_generator", "_ace_reflector",
    "_ace_curator", "_auto_synthesize_skills", "_word_jaccard",
    "_semantic_or_jaccard",
):
    _PROXIED_ATTRS[_name] = _ace_mod

# RLM helpers → ccr.mcp.rlm_tools
for _name in (
    "_summarize_stdout",
    "_get_repl_for_session",
    "_get_session_age_warning",
    "_cleanup_session",
):
    _PROXIED_ATTRS[_name] = _rlm_mod


class _BackwardCompatModule(_sys.modules[__name__].__class__):
    """Module subclass that proxies attribute access to canonical submodules."""

    def __getattr__(self, name: str):
        target = _PROXIED_ATTRS.get(name)
        if target is not None:
            return getattr(target, name)
        raise AttributeError(f"module 'ccr.mcp_server' has no attribute {name!r}")

    def __setattr__(self, name: str, value):
        target = _PROXIED_ATTRS.get(name)
        if target is not None:
            setattr(target, name, value)
        else:
            super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _BackwardCompatModule


if __name__ == "__main__":
    main()
