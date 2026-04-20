"""CCR MCP Server — core state, helpers, and FastMCP instance.

All global state and helper functions live here. Tool modules import from
this module to access the shared ``mcp`` instance and state.

Usage:
    python -m ccr.mcp_server              # stdio transport (for Claude Code)
    python -m ccr.mcp_server --project .  # explicit project root
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading

logger = logging.getLogger(__name__)

from mcp.server.fastmcp import FastMCP

from ccr.ace.playbook import Playbook
from ccr.context.indexer import RepoIndex
from ccr.core.memory import MemoryManager
from ccr.core.scratchpad import Scratchpad
from ccr.core.triples import TripleStore
from ccr.core.types import CCRConfig, PlaybookSchema
from ccr.mcp.audit import configure_audit_log
from ccr.utils.parsing import extract_json_string

# Default sub-model for optional Anthropic-backed features (non-critical path)
_DEFAULT_SUB_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Globals — initialized once at startup
# ---------------------------------------------------------------------------

# M3: Module-level lock for thread safety on global state mutations
_state_lock = threading.Lock()

_project_root: str = ""
_memory: MemoryManager | None = None
_playbook: Playbook | None = None
_playbook_path: str = ""
_failure_lessons_path: str = ""
_global_playbook: Playbook | None = None
_global_playbook_path: str = ""
_global_failure_lessons_path: str = ""
_repo_index: RepoIndex | None = None
_repl: object | None = None  # CCRRepl when initialized (backward-compat single session)
_repl_sessions: dict[str, object] = {}
_repl_session_ttl: dict[str, float] = {}
_schema_path: str = ""
_global_schema_path: str = ""
_embedding_model: object | None = None  # EmbeddingModel when available
_embeddings_path: str = ""
_chunk_embeddings_path: str = ""
_scratchpad: Scratchpad | None = None
_triple_store: TripleStore | None = None
_index_db: object | None = None  # IndexDB when SQLite index is active

# Session logger (chat turn persistence)
_session_store: object | None = None   # SessionStore when initialized
_session_db_path: str = ""
_current_session_id: str = ""          # cached for current session

mcp = FastMCP(
    "ccr",
    instructions=(
        "CCR gives you persistent project memory (GCC), self-evolving strategy "
        "playbooks (ACE), a sandboxed Python REPL (RLM), and repo indexing. "
        "Use gcc_* tools for memory, ace_* for playbook management, rlm_* for "
        "sandboxed code execution, and index_* for repo search."
    ),
)


def _get_sub_client() -> object | None:
    """Return a sub-model client, or None when no backend is configured.

    Priority order (first available wins):
      1. Ollama (CCR_OLLAMA_MODEL env var, e.g. "qwen2.5:7b") — free, local
      2. Anthropic Haiku (ANTHROPIC_API_KEY_SUB or ANTHROPIC_API_KEY)

    Graceful no-op: when nothing is configured, all Phase 2 features are silently skipped.
    """
    ollama_model = os.environ.get("CCR_OLLAMA_MODEL")
    if ollama_model:
        try:
            from ccr.models.openai_compat import OpenAICompatClient  # noqa: PLC0415
            return OpenAICompatClient(
                model_name=ollama_model,
                base_url=os.environ.get("CCR_OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                api_key="ollama",
                max_tokens=1024,
            )
        except Exception:
            pass

    api_key = os.environ.get("ANTHROPIC_API_KEY_SUB") or os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            from ccr.models.anthropic_client import ClaudeClient  # noqa: PLC0415
            return ClaudeClient(api_key=api_key, model_name=_DEFAULT_SUB_MODEL, max_tokens=1024)
        except Exception:
            pass

    return None


def _init(project_root: str | None = None) -> None:
    """Initialize all subsystems for the given project root."""
    global _project_root, _memory, _playbook, _playbook_path, _failure_lessons_path
    global _global_playbook, _global_playbook_path, _global_failure_lessons_path, _repo_index
    global _schema_path, _global_schema_path
    global _embedding_model, _embeddings_path, _chunk_embeddings_path
    global _scratchpad
    global _triple_store
    global _index_db
    global _session_store, _session_db_path, _current_session_id

    _project_root = os.path.abspath(project_root or os.getcwd())
    _default_backend = CCRConfig().storage_backend
    storage_backend = os.environ.get("CCR_STORAGE_BACKEND", _default_backend)
    _memory = MemoryManager(_project_root, CCRConfig(storage_backend=storage_backend))
    _memory.ensure_structure()

    # Auto-migrate flat files → SQLite on first access (Phase 7)
    if storage_backend == "sqlite":
        ccr_root = os.path.join(_project_root, ".ccr")
        db_path = os.path.join(ccr_root, "memory.db")
        try:
            from ccr.core.storage.migration import needs_migration, auto_migrate
            if needs_migration(ccr_root):
                from ccr.core.storage.sqlite_backend import SqliteStorageBackend
                _backend = SqliteStorageBackend(ccr_root)
                _backend.close()
                mig = auto_migrate(ccr_root, db_path)
                if mig["errors"]:
                    logger.warning("Auto-migration errors: %s", mig["errors"])
                else:
                    logger.info(
                        "Auto-migrated %d records across phases %s",
                        mig["total_migrated"], mig["phases_run"],
                    )
        except Exception as exc:
            logger.warning("Auto-migration failed (will use flat files): %s", exc)

    def _cleanup_storage() -> None:
        if _memory and hasattr(_memory, "_storage"):
            try:
                _memory._storage.close()
            except Exception:
                pass

    atexit.register(_cleanup_storage)

    # Wire optional sub-model (Phase 2): activates GCC LLM rolling summary + ACE synthesis
    sub = _get_sub_client()
    if sub is not None:
        _memory.set_sub_client(sub)

    # Clear session marker so hooks detect new session on first prompt
    session_marker = os.path.join(_project_root, ".ccr", ".session_active")
    if os.path.isfile(session_marker):
        try:
            os.remove(session_marker)
        except OSError:
            pass

    _playbook_path = os.path.join(_project_root, ".ccr", "playbook.txt")
    _failure_lessons_path = os.path.join(_project_root, ".ccr", "failure_lessons.json")
    _playbook = _load_playbook()

    # Initialize global playbook (~/.ccr/)
    global_ccr = os.path.expanduser("~/.ccr")
    os.makedirs(global_ccr, exist_ok=True)
    _global_playbook_path = os.path.join(global_ccr, "global_playbook.txt")
    _global_failure_lessons_path = os.path.join(global_ccr, "global_failure_lessons.json")
    _global_playbook = _load_global_playbook()

    # Schema paths (MCE-inspired schema evolution)
    _schema_path = os.path.join(_project_root, ".ccr", "playbook_schema.json")
    _global_schema_path = os.path.join(global_ccr, "global_playbook_schema.json")

    # Embeddings paths (A-RAG semantic search)
    _embeddings_path = os.path.join(_project_root, ".ccr", "index_embeddings.json.gz")
    _chunk_embeddings_path = os.path.join(_project_root, ".ccr", "index_chunk_embeddings.json.gz")

    # Working memory scratchpad (AgeMem-inspired ephemeral KV store)
    _scratchpad = Scratchpad(os.path.join(_project_root, ".ccr", "scratchpad.json"))

    # Triple store (Memori-inspired semantic triple extraction)
    _triple_store = TripleStore(
        os.path.join(_project_root, ".ccr", "triples.json"),
        max_buffer_size=_memory.config.triple_max_buffer_size,
    )

    # Session logger — path set here; store initialized lazily on first use
    _session_db_path = os.path.join(_project_root, ".ccr", "sessions.db")
    _session_store = None
    _current_session_id = ""

    # Initialize IndexDB (SQLite-backed index persistence)
    _index_db_path = os.path.join(_project_root, ".ccr", "index.db")
    try:
        from ccr.context.index_db import IndexDB
        _index_db = IndexDB(_index_db_path)
    except Exception:
        _index_db = None

    # Build repo index — try SQLite load first, then JSON cache, then full build
    _repo_index = None
    if _index_db is not None:
        try:
            cached = RepoIndex.from_db(_project_root, _index_db)
            if cached is not None:
                live_sig = cached._compute_mtime_sig()
                if live_sig == cached._mtime_sig and cached._mtime_sig:
                    _repo_index = cached
        except Exception:
            pass
    if _repo_index is None:
        try:
            cache_json = _memory.load_index()
            if cache_json:
                cached = RepoIndex.from_cache(_project_root, cache_json)
                if cached is not None:
                    live_sig = cached._compute_mtime_sig()
                    if live_sig == cached._mtime_sig and cached._mtime_sig:
                        _repo_index = cached
        except Exception:
            pass
    if _repo_index is None:
        _repo_index = RepoIndex.build(_project_root)

    # Load cached embeddings if available
    if os.path.isfile(_embeddings_path):
        _repo_index.load_embeddings(_embeddings_path)

    # Load cached chunk embeddings if available (A-RAG §3.1 sentence-level chunks)
    if os.path.isfile(_chunk_embeddings_path):
        _repo_index.load_chunk_embeddings(_chunk_embeddings_path)

    # Save index to IndexDB (always, so FTS5 is populated for search)
    if _index_db is not None:
        try:
            _repo_index.save_to_db(_index_db)
        except Exception:
            pass
    # Save JSON cache (only when freshly built — loaded-from-cache already has valid JSON)
    if not (hasattr(_repo_index, "_mtime_sig") and _repo_index._mtime_sig):
        try:
            _memory.save_index(_repo_index.to_json())
        except Exception:
            pass

    # Configure audit logging
    ccr_root = os.path.join(_project_root, ".ccr")
    configure_audit_log(ccr_root)

    # Register project in global registry
    try:
        from ccr.cli import _register_project
        _register_project(_project_root)
    except Exception:
        pass  # Non-critical


@mcp.resource("ccr://health")
def health_check() -> str:
    """Health check resource — returns server status, commit count, and capability flags."""
    import json as _json

    status: dict = {"status": "ok", "project_root": _project_root}
    if _memory is not None:
        try:
            branch = _memory.get_active_branch()
            index = _memory._build_commit_index(branch)
            status["commit_count"] = len(index)
        except Exception:
            status["commit_count"] = "unknown"
    else:
        status["commit_count"] = 0

    # Capability flags
    status["onnx_available"] = False
    try:
        from ccr.context.embeddings import get_embedding_model
        status["onnx_available"] = get_embedding_model() is not None
    except Exception:
        pass

    status["sqlite_vec_available"] = False
    try:
        import sqlite_vec  # noqa: F401
        status["sqlite_vec_available"] = True
    except ImportError:
        pass

    status["sub_model_available"] = _get_sub_client() is not None
    status["playbook_bullets"] = len(_playbook.bullets) if _playbook else 0

    return _json.dumps(status, indent=2)


def _atomic_write(path: str, content: str) -> None:
    """Write content to path atomically via tmp + fsync + os.replace (H5)."""
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_playbook() -> Playbook:
    """Load playbook. Prefer SQLite backend when available; else flat file."""
    # SQLite path: hydrate from memory.db playbook_bullets/sections/failure_lessons
    if _memory is not None:
        storage = getattr(_memory, "_storage", None)
        if storage is not None:
            from ccr.core.storage.sqlite_backend import SqliteStorageBackend
            if isinstance(storage, SqliteStorageBackend):
                try:
                    return Playbook.from_backend(storage, scope="project")
                except Exception as exc:
                    logger.warning(
                        "SQLite playbook load failed, falling back to flat file: %s", exc,
                    )

    # Flat-file path (default backend or SQLite error fallback)
    if os.path.isfile(_playbook_path):
        with open(_playbook_path, "r", encoding="utf-8") as f:
            pb = Playbook(f.read())
    else:
        pb = Playbook()
    if _failure_lessons_path:
        pb.load_failure_lessons(_failure_lessons_path)
    return pb


def _save_playbook() -> None:
    """Persist playbook. Prefer SQLite backend when available; else flat file."""
    if _playbook is None:
        return

    # SQLite path
    if _memory is not None:
        storage = getattr(_memory, "_storage", None)
        if storage is not None:
            from ccr.core.storage.sqlite_backend import SqliteStorageBackend
            if isinstance(storage, SqliteStorageBackend):
                try:
                    _playbook.save_to_backend(storage, scope="project")
                    return
                except Exception as exc:
                    logger.warning(
                        "SQLite playbook save failed, falling back to flat file: %s", exc,
                    )

    # Flat-file path (default backend or SQLite error fallback)
    _atomic_write(_playbook_path, _playbook.serialize())
    if _failure_lessons_path:
        _playbook.save_failure_lessons(_failure_lessons_path)


def _load_global_playbook() -> Playbook:
    """Load global playbook from ~/.ccr/ or create empty."""
    if os.path.isfile(_global_playbook_path):
        with open(_global_playbook_path, "r", encoding="utf-8") as f:
            pb = Playbook(f.read())
    else:
        pb = Playbook()
    if _global_failure_lessons_path:
        pb.load_failure_lessons(_global_failure_lessons_path)
    return pb


def _save_global_playbook() -> None:
    """Persist global playbook to ~/.ccr/ (H5: atomic writes)."""
    if _global_playbook is not None:
        _atomic_write(_global_playbook_path, _global_playbook.serialize())
        if _global_failure_lessons_path:
            _global_playbook.save_failure_lessons(_global_failure_lessons_path)


def _load_schema(path: str) -> PlaybookSchema:
    """Load schema from JSON or return default (MCE schema persistence)."""
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.loads(f.read())
            return PlaybookSchema.from_dict(data.get("current", {}))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return PlaybookSchema.default()


def _load_schema_history(path: str) -> list[dict]:
    """Load schema version history from JSON."""
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.loads(f.read())
            return data.get("history", [])
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return []


def _save_schema(schema: PlaybookSchema, history: list[dict], path: str) -> None:
    """Save schema + version history to JSON via atomic write."""
    data = {
        "current": schema.to_dict(),
        "history": history,
        "next_version": schema.version + 1,
    }
    _atomic_write(path, json.dumps(data, indent=2, ensure_ascii=False))


def _ensure_global_playbook() -> Playbook:
    if _global_playbook is None:
        _init()
    if _global_playbook is None:
        raise RuntimeError("Global playbook failed to initialize")
    return _global_playbook


def _ensure_memory() -> MemoryManager:
    if _memory is None:
        _init()
    if _memory is None:
        raise RuntimeError("MemoryManager failed to initialize")
    return _memory


def _ensure_playbook() -> Playbook:
    if _playbook is None:
        _init()
    if _playbook is None:
        raise RuntimeError("Playbook failed to initialize")
    return _playbook


def _ensure_index() -> RepoIndex:
    if _repo_index is None:
        _init()
    if _repo_index is None:
        raise RuntimeError("RepoIndex failed to initialize")
    return _repo_index


def _extract_patterns_from_commit(
    title: str, what: str, why: str, files_changed: list[str], sub_client
) -> list[str]:
    """Auto-extract transferable patterns from a commit via sub-model (CER §3.2).

    Returns "When X, do Y" pattern strings, or [] on any error.
    Failures are transparent and never block the commit path.
    """
    try:
        files_str = ", ".join(files_changed[:5])
        prompt = (
            "Extract 1-2 transferable patterns from this commit for future reference.\n\n"
            f"Commit: {title}\n"
            f"What: {what}\n"
            f"Why: {why}\n"
            f"Files: {files_str}\n\n"
            'Write patterns as abstract "When X, do Y" rules that would help in similar future situations.\n'
            'Respond with a JSON array of strings: ["When X, do Y", "When A, do B"]\n'
            "Be concise. Each pattern should be 1 sentence."
        )
        response = sub_client.completion(prompt)
        raw = extract_json_string(response)
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(p) for p in parsed if p]
        return []
    except Exception:
        return []


def _resolve_playbook(scope: str) -> tuple[Playbook, callable]:
    """Resolve playbook and save function based on scope."""
    if scope == "global":
        return _ensure_global_playbook(), _save_global_playbook
    return _ensure_playbook(), _save_playbook


def _serialize_playbook(pb: Playbook) -> str:
    """Serialize a playbook, using failure-aware format if lessons exist."""
    has_lessons = any(b.has_failure_lessons for b in pb.bullets)
    return pb.serialize_with_failures() if has_lessons else pb.serialize()


def _ensure_session_store():
    """Return the SessionStore, initializing lazily on first call.

    Also refreshes _current_session_id from .ccr/.current_session_id if not cached.
    """
    global _session_store, _current_session_id
    if _session_store is None:
        from ccr.core.session_store import SessionStore  # noqa: PLC0415
        db = _session_db_path or os.path.join(_project_root, ".ccr", "sessions.db")
        _session_store = SessionStore(db)
    if not _current_session_id:
        id_file = os.path.join(_project_root, ".ccr", ".current_session_id")
        try:
            if os.path.isfile(id_file):
                with open(id_file, "r", encoding="utf-8") as f:
                    _current_session_id = f.read().strip()
        except OSError:
            pass
    return _session_store


def main():
    """Run the MCP server with stdio transport."""
    import argparse

    parser = argparse.ArgumentParser(description="CCR MCP Server")
    parser.add_argument(
        "--project", "-p",
        default=os.getcwd(),
        help="Project root directory (default: cwd)",
    )
    args = parser.parse_args()

    _init(args.project)

    # Import tool modules to trigger @mcp.tool registration
    import sys as _sys
    _TOOL_MODULES = [
        ("ccr.mcp.gcc_branch_tools", "GCC branch management tools"),
        ("ccr.mcp.gcc_search_tools", "GCC search and query tools"),
        ("ccr.mcp.gcc_tools", "GCC core memory tools"),
        ("ccr.mcp.ace_tools", "ACE core playbook tools"),
        ("ccr.mcp.ace_llm_tools", "ACE LLM pipeline tools"),
        ("ccr.mcp.ace_schema_tools", "ACE schema evolution tools"),
        ("ccr.mcp.rlm_tools", "RLM sandboxed REPL"),
        ("ccr.mcp.index_tools", "Index search tools"),
        ("ccr.mcp.session_tools", "Session logger tools"),
    ]
    for _mod, _label in _TOOL_MODULES:
        try:
            __import__(_mod)
        except ImportError as _exc:
            print(f"[CCR] Failed to load {_label} ({_mod}): {_exc}", file=_sys.stderr)
            raise

    mcp.run(transport="stdio")
