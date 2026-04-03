"""Repo Index Tools — index_build, index_search."""

from __future__ import annotations

from mcp.types import ToolAnnotations
from mcp.server.fastmcp.exceptions import ToolError

from ccr.context.indexer import RepoIndex
from ccr.mcp.server import mcp
from ccr.mcp_types import (
    IndexBuildResult,
    IndexSearchResult,
)

# Import server module to access mutable globals via attribute
import ccr.mcp.server as _srv


# ===========================================================================
# Repo Index Tools
# ===========================================================================


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
def index_build() -> IndexBuildResult:
    """Build or rebuild the repo index.

    Scans the project directory for source files, extracts symbols (classes,
    functions) and imports per file. The index enables search_repo and
    get_file in the RLM sandbox.

    If onnxruntime + tokenizers are installed, also computes dense embeddings
    for semantic search (A-RAG §3.1). Otherwise, BM25 fallback is available.
    """
    try:
        with _srv._state_lock:  # H2: protect global state mutation
            _srv._repo_index = RepoIndex.build(_srv._project_root)

            # Cache
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
) -> IndexSearchResult:
    """Search the repo index for files matching a query.

    Three search modes:
      - "keyword": Fast substring matching on paths, symbols, and content.
        Best for specific file names or symbol names.
      - "semantic": Meaning-based search using embeddings (or BM25 fallback).
        Best for conceptual queries like "authentication logic".
      - "hybrid" (default): Combines keyword + semantic scores.
        Best general-purpose mode for natural language queries.

    Args:
        query: Search term or natural language description.
        mode: "keyword", "semantic", or "hybrid" (default).
        top_k: Maximum results to return (1-100, default 10).
        return_snippets: If True, include a brief code snippet in each result
            (hybrid mode only; requires chunk embeddings for semantic snippets,
            falls back to BM25-derived snippets otherwise).
        file_glob: Glob pattern to restrict results (e.g. "**/*.py"). Defaults
            to "**/*" (all files).
    """
    if mode not in ("keyword", "semantic", "hybrid"):
        raise ToolError(f"Invalid mode '{mode}'. Use 'keyword', 'semantic', or 'hybrid'.")

    top_k = max(1, min(top_k, 100))  # M3: bound top_k to prevent excessive results
    idx = _srv._ensure_index()

    suffix = ""
    if mode == "keyword":
        results = idx.search(query, file_glob=file_glob)[:top_k]
    elif mode == "semantic":
        if _srv._embedding_model is not None and idx._embeddings:
            results = idx.semantic_search(query, _srv._embedding_model, top_k=top_k, file_glob=file_glob)
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

    if not results:
        text = f"No files matching '{query}' ({mode} mode)."
        return IndexSearchResult(result_count=0, mode=mode, message=text)

    lines = [f"# {mode} search{suffix}"]
    for r in results:
        syms = ", ".join(r["symbols"][:5]) if r["symbols"] else ""
        line = (
            f"[{r['score']}] {r['path']} ({r['language']}, {r['lines']} lines)"
            + (f" — {syms}" if syms else "")
        )
        lines.append(line)
        if return_snippets and "snippet" in r:
            lines.append(f"  snippet: {r['snippet']}")
    text = "\n".join(lines)
    return IndexSearchResult(result_count=len(results), mode=mode, message=text)
