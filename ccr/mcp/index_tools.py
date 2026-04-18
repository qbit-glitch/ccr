"""Repo Index Tools — index_build, index_search."""

from __future__ import annotations

from mcp.types import ToolAnnotations
from mcp.server.fastmcp.exceptions import ToolError

from ccr.context.indexer import RepoIndex, _detect_mode
from ccr.mcp.server import mcp
from ccr.mcp_types import (
    IndexBuildResult,
    IndexSearchResult,
    IndexStatusResult,
)

# Import server module to access mutable globals via attribute
import ccr.mcp.server as _srv


# ===========================================================================
# Repo Index Tools
# ===========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def index_build(
    incremental: bool = False,
    filter_extensions: str = "",
    force: bool = False,
    max_file_size_kb: int = 500,
    progress_interval: int = 100,
) -> IndexBuildResult:
    """Build or rebuild the repo index.

    Scans the project directory for source files, extracts symbols (classes,
    functions) and imports per file. The index enables search_repo and
    get_file in the RLM sandbox.

    If onnxruntime + tokenizers are installed, also computes dense embeddings
    for semantic search (A-RAG §3.1). Otherwise, BM25 fallback is available.

    Args:
        incremental: If True, skip rebuild when file mtimes are unchanged
            since the last index build. Useful for large repos on repeated
            sessions. Falls through to full rebuild on any error.
        filter_extensions: Comma-separated list of file extensions to index
            (e.g. "py,ts,go"). Leading dots optional. Empty means all files.
        force: If True, bypass incremental check and always do a full rebuild.
            Overrides incremental=True. Use when index may be corrupt or stale
            despite unchanged mtimes (e.g., after manual .ccr/ edits).
        max_file_size_kb: Files larger than this are indexed for metadata only
            (path, language, size) but content and symbols are skipped (default
            500 KB). Increase for repos with large generated files you want
            to search; decrease to speed up indexing.
        progress_interval: Log a progress message every N files indexed (default 100).
    """
    # Parse filter_extensions
    exts: set[str] | None = None
    if filter_extensions.strip():
        exts = {
            ("." + e.strip().lstrip("."))
            for e in filter_extensions.split(",")
            if e.strip()
        } or None

    try:
        # Incremental mode: skip rebuild if mtime signature unchanged
        if incremental and not force:
            # Try SQLite-backed incremental build first
            if _srv._index_db is not None:
                try:
                    idx, changed = RepoIndex.incremental_build(
                        _srv._project_root, _srv._index_db,
                        extensions=exts,
                        max_file_size_kb=max(1, max_file_size_kb),
                    )
                    if changed == 0:
                        with _srv._state_lock:
                            _srv._repo_index = idx
                        return IndexBuildResult(
                            files_indexed=len(idx.files),
                            message=(
                                f"Index up to date ({len(idx.files)} files,"
                                " no rebuild needed). [SQLite-backed]"
                            ),
                        )
                    with _srv._state_lock:
                        _srv._repo_index = idx
                    # Save JSON cache too for backward compat
                    try:
                        mem = _srv._ensure_memory()
                        mem.save_index(_srv._repo_index.to_json())
                    except Exception:
                        pass
                except Exception:
                    pass  # Fall through to legacy incremental

            # Legacy JSON cache incremental check
            cache_json = None
            try:
                mem = _srv._ensure_memory()
                cache_json = mem.load_index()
            except Exception:
                pass
            if cache_json:
                try:
                    cached = RepoIndex.from_cache(_srv._project_root, cache_json)
                    if cached is not None:
                        live_sig = cached._compute_mtime_sig()
                        if live_sig == cached._mtime_sig and cached._mtime_sig:
                            with _srv._state_lock:
                                _srv._repo_index = cached
                            return IndexBuildResult(
                                files_indexed=len(cached.files),
                                message=(
                                    f"Index up to date ({len(cached.files)} files,"
                                    " no rebuild needed)."
                                ),
                            )
                except Exception:
                    pass  # Fall through to full rebuild

        with _srv._state_lock:  # H2: protect global state mutation
            _srv._repo_index = RepoIndex.build(
                _srv._project_root,
                extensions=exts,
                max_file_size_kb=max(1, max_file_size_kb),
                progress_interval=progress_interval,
            )

            # Save to IndexDB + JSON cache
            if _srv._index_db is not None:
                try:
                    _srv._repo_index.save_to_db(_srv._index_db)
                except Exception:
                    pass
            mem = _srv._ensure_memory()
            try:
                mem.save_index(_srv._repo_index.to_json())
                mem.update_metadata_file_tree([f for f in _srv._repo_index.files.keys()])
            except Exception:
                pass

        # Semantic: compute embeddings if available (A-RAG §3.1)
        emb_status = ""
        try:
            from ccr.context.embeddings import SEMANTIC_AVAILABLE, get_embedding_model

            if SEMANTIC_AVAILABLE:
                model = get_embedding_model()
                if model is not None:
                    count = _srv._repo_index.build_embeddings(model)
                    _srv._repo_index.save_embeddings(_srv._embeddings_path)
                    _srv._embedding_model = model
                    emb_status = f"\nEmbeddings: {count} files ({model.MODEL_NAME})"
                    # A-RAG §3.1: build chunk-level embeddings for snippet extraction
                    try:
                        chunk_count, chunk_files = _srv._repo_index.build_chunk_embeddings(model)
                        _srv._repo_index.save_chunk_embeddings(_srv._chunk_embeddings_path)
                        emb_status += f"\nChunk embeddings: {chunk_count} chunks across {chunk_files} files"
                    except Exception as ce:
                        emb_status += f"\nChunk embeddings: skipped ({ce})"
            else:
                emb_status = (
                    "\nEmbeddings: unavailable (install onnxruntime + tokenizers for semantic search)"
                )
        except Exception as e:
            emb_status = f"\nEmbeddings: skipped ({e})"

        files_indexed = len(_srv._repo_index.files) if _srv._repo_index.files else 0
        text = _srv._repo_index.get_summary() + emb_status
        return IndexBuildResult(files_indexed=files_indexed, message=text)
    except ValueError:
        raise  # User input validation — let MCP propagate
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def index_search(
    query: str,
    mode: str = "hybrid",
    top_k: int = 10,
    return_snippets: bool = False,
    file_glob: str = "**/*",
    min_score: float = 0.0,
    symbols_only: bool = False,
    mode_hint: bool = False,
    exclude_paths: list[str] | None = None,
) -> IndexSearchResult:
    """Search the repo index for files matching a query.

    Four search modes:
      - "keyword": Fast substring matching on paths, symbols, and content.
        Best for specific file names or symbol names.
      - "semantic": Meaning-based search using embeddings (or BM25 fallback).
        Best for conceptual queries like "authentication logic".
      - "hybrid" (default): Combines keyword + semantic scores.
        Best general-purpose mode for natural language queries.
      - "auto": Automatically selects keyword/semantic/hybrid based on query
        shape (A-RAG §3.2). Short/symbol queries → keyword; question-form
        queries → semantic; everything else → hybrid. resolved_mode in result
        shows which mode was chosen.

    Args:
        query: Search term or natural language description.
        mode: "keyword", "semantic", "hybrid" (default), or "auto".
        top_k: Maximum results to return (1-100, default 10).
        return_snippets: If True, include a brief code snippet in each result
            (hybrid mode only; requires chunk embeddings for semantic snippets,
            falls back to BM25-derived snippets otherwise).
        file_glob: Glob pattern to restrict results (e.g. "**/*.py"). Defaults
            to "**/*" (all files).
        min_score: Minimum score threshold to include in results (default 0.0 =
            no filtering). Useful to suppress low-confidence matches.
        symbols_only: If True, only return files where the query matches a symbol
            name (function, class). Filters out pure content/path matches.
            Best for "find all implementations of AuthService" queries.
        mode_hint: If True, appends a recommended mode suggestion based on the query
            shape (long query → semantic, path/extension → keyword, else → hybrid).
        exclude_paths: Optional list of file paths to exclude from results
            (A-RAG C^read filter). Useful to skip already-read files in
            multi-turn exploration sessions.
    """
    if mode not in ("keyword", "semantic", "hybrid", "auto"):
        raise ToolError(f"Invalid mode '{mode}'. Use 'keyword', 'semantic', 'hybrid', or 'auto'.")

    # A-RAG §3.2: auto-detect mode from query shape
    resolved_mode = _detect_mode(query) if mode == "auto" else mode

    top_k = max(1, min(top_k, 100))  # M3: bound top_k to prevent excessive results
    idx = _srv._ensure_index()

    # Compute mode_hint suggestion upfront (independent of search results)
    hint_suffix = ""
    if mode_hint:
        suggestion = _detect_mode(query)
        hint_suffix = f"\n[mode_hint: recommended mode is '{suggestion}']"

    suffix = ""
    if return_snippets and resolved_mode != "hybrid":
        suffix = f" (note: return_snippets only supported in hybrid mode; ignored for {resolved_mode})"
    if resolved_mode == "keyword":
        results = idx.search(query, file_glob=file_glob)
    elif resolved_mode == "semantic":
        if _srv._embedding_model is not None and idx._embeddings:
            results = idx.semantic_search(query, _srv._embedding_model, top_k=top_k, file_glob=file_glob)
        elif _srv._index_db is not None and _srv._index_db.fts_available:
            results = idx.fts5_search(query, _srv._index_db, top_k=top_k)
            suffix = " (FTS5 fallback)"
        else:
            results = idx.bm25_search(query, top_k=top_k, file_glob=file_glob)
            suffix = " (BM25 fallback)"
    else:  # hybrid
        results = idx.hybrid_search(
            query,
            model=_srv._embedding_model,
            top_k=top_k,
            file_glob=file_glob,
            return_snippets=return_snippets,
        )
        if _srv._embedding_model is None or not idx._embeddings:
            suffix = " (BM25 fallback)"

    # A-RAG C^read: exclude already-seen paths before top-k truncation
    if exclude_paths:
        results = [r for r in results if r.get("path") not in exclude_paths]

    # Apply min_score filter
    if min_score > 0.0:
        results = [r for r in results if r.get("score", 0.0) >= min_score]

    # Apply symbols_only filter: keep results where query matches a symbol name
    if symbols_only:
        q_lower = query.lower()
        results = [
            r for r in results
            if any(q_lower in s.lower() for s in r.get("symbols", []))
        ]

    # Final top-k truncation — applied once after all filters
    results = results[:top_k]

    auto_suffix = f" [auto→{resolved_mode}]" if mode == "auto" else ""
    if not results:
        text = f"No files matching '{query}' ({resolved_mode} mode{auto_suffix})."
        if min_score > 0.0:
            text += f" (min_score={min_score} may be filtering results)"
        if symbols_only:
            text += " (symbols_only=True: no symbol matched query)"
        if hint_suffix:
            text += hint_suffix
        return IndexSearchResult(result_count=0, mode=resolved_mode, resolved_mode=resolved_mode, message=text)

    # Normalize scores to [0.0, 1.0] relative to top result
    max_score = max((r.get("score", 0) for r in results), default=1)
    if max_score == 0:
        max_score = 1
    for r in results:
        r["score"] = round(r.get("score", 0) / max_score, 4)

    lines = [f"# {resolved_mode} search{suffix}{auto_suffix}"]
    for r in results:
        syms = ", ".join(r["symbols"][:5]) if r["symbols"] else ""
        line = (
            f"[{r['score']}] {r['path']} ({r['language']}, {r['lines']} lines)"
            + (f" — {syms}" if syms else "")
        )
        lines.append(line)
        if return_snippets and "snippet" in r:
            lines.append(f"  snippet: {r['snippet']}")
    if hint_suffix:
        lines.append(hint_suffix)
    text = "\n".join(lines)
    return IndexSearchResult(result_count=len(results), mode=resolved_mode, resolved_mode=resolved_mode, message=text)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True))
def index_status() -> IndexStatusResult:
    """Return current state of the repo index — staleness, embeddings, file count.

    Use to check whether index_build is needed before running index_search.
    Reports file count, build timestamp, embedding availability, and whether
    the index is stale (file mtimes have changed since last build).
    """
    try:
        idx = _srv._repo_index
        if idx is None:
            return IndexStatusResult(
                built_at="never",
                file_count=0,
                embeddings_available=False,
                bm25_cache_built=False,
                chunk_embeddings_available=False,
                chunk_count=0,
                is_stale=True,
                message="Index not built. Call index_build first.",
            )

        built_at = ""
        if getattr(idx, "_built_at", None):
            from datetime import datetime, timezone
            built_at = datetime.fromtimestamp(idx._built_at, tz=timezone.utc).isoformat()

        file_count = len(idx.files) if hasattr(idx, "files") else 0
        embeddings_available = bool(getattr(idx, "_embeddings", {}))
        bm25_cache_built = getattr(idx, "_bm25_cache", None) is not None
        chunk_embeddings_available = bool(getattr(idx, "_chunk_embeddings", {}))
        chunk_count = len(getattr(idx, "_chunk_embeddings", {}))

        # Staleness: compare live mtime_sig to stored one
        is_stale = True
        if hasattr(idx, "_compute_mtime_sig") and hasattr(idx, "_mtime_sig"):
            live_sig = idx._compute_mtime_sig()
            is_stale = (live_sig != idx._mtime_sig) or not idx._mtime_sig

        # IndexDB stats
        db_info = ""
        if _srv._index_db is not None:
            db_files = _srv._index_db.file_count()
            db_chunks = _srv._index_db.chunk_count()
            fts = "yes" if _srv._index_db.fts_available else "no"
            db_info = f" SQLite index: {db_files} files, {db_chunks} chunks, FTS5={fts}."

        status = "stale" if is_stale else "up to date"
        msg = (
            f"Index {status}. {file_count} files indexed"
            + (f", built at {built_at}" if built_at else "")
            + (f". {chunk_count} chunk embeddings." if chunk_embeddings_available else ".")
            + db_info
            + ("" if not is_stale else " Run index_build to refresh.")
        )
        return IndexStatusResult(
            built_at=built_at,
            file_count=file_count,
            embeddings_available=embeddings_available,
            bm25_cache_built=bm25_cache_built,
            chunk_embeddings_available=chunk_embeddings_available,
            chunk_count=chunk_count,
            is_stale=is_stale,
            message=msg,
        )
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e
