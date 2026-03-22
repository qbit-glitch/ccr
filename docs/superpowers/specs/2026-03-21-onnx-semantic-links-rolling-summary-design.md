# Design: ONNX Semantic Links + Rolling Summary Prompt Improvement

**Date**: 2026-03-21
**Status**: Approved (revised after spec review)
**Scope**: Two independent improvements to `ccr/core/memory.py` and `ccr/mcp_server.py`

---

## Sub-project 1: ONNX Embeddings for MAGMA Semantic Links

### Problem

`_compute_links()` (line 675 of `memory.py`) uses word Jaccard similarity for semantic links — low discriminative power compared to MAGMA's dense vector cosine similarity. The ONNX embedding model (`all-MiniLM-L6-v2`, 384-dim) already exists in `ccr/context/embeddings.py` but is unused by the memory system.

### Architecture

**New file**: `.ccr/commit_embeddings.json.gz` — gzip-compressed JSON mapping `commit_id → [float, ...]` (384-dim). Same format as `ccr/context/indexer.py` uses (`save_embeddings` / `load_embeddings` from `embeddings.py`). Cache is capped to `config.link_scan_window * 2` entries — oldest commit IDs evicted when saving, so the file stays bounded (~3KB compressed for 40 entries).

**Changes to `memory.py`**:
1. Import `get_embedding_model`, `save_embeddings`, `load_embeddings` from `ccr.context.embeddings` at the top of `memory.py`. Do NOT add `import numpy as np` at module level — numpy is an optional dep. Instead use local `import numpy as np` inside `_embed_commit` and inside `_load_commit_embeddings` (only reachable when ONNX is available, so safe).
2. New method `_get_commit_embeddings_path() -> str` → `.ccr/commit_embeddings.json.gz`
3. New method `_embed_commit(commit_id: str, text: str) -> np.ndarray | None` — embeds text, appends to cache with cap enforcement, saves atomically. Returns the computed vector so the caller can reuse it without a second inference pass. Returns `None` if `get_embedding_model()` returns `None`. Not called on the A-MAC merge path (where `_merge_into_last_commit` returns early and no new `commit_id` is assigned).
4. New method `_load_commit_embeddings(commit_ids: list[str]) -> dict[str, np.ndarray]` — loads cache, returns only requested IDs as numpy arrays (converts `list[float]` → `np.ndarray` for matrix ops)
5. `_compute_links()` gains an optional `new_vec = None` parameter (type `Any` to avoid unconditional numpy import). Step 4 (semantic links), per-commit fallback:
   - If `new_vec` is not None and cached embedding exists for that commit: `float(cache[cid] @ new_vec)` — valid because both vectors are L2-normalized (dot product of L2-normalized vectors = cosine similarity). No `model` instance needed in `_compute_links`.
   - Else: `self._jaccard(new_keywords, old_keywords)` (unchanged)
   - Same threshold `config.link_semantic_threshold` applies to both (both are in [0,1] range since cosine similarity of L2-normalized vectors is bounded to [-1,1] but in practice near-zero for unrelated text, and Jaccard is in [0,1])
6. `commit()` — after `commit_id` is assigned (normal path only, not merge path): call `new_vec = _embed_commit(commit_id, combined_text)` and pass `new_vec` to `_compute_links(…, new_vec=new_vec)`. This avoids a second inference pass — the vector computed in `_embed_commit` is reused directly.

### Data Flow

```
gcc_commit called (normal path, not A-MAC merge)
  → commit_id assigned
  → combined_text = f"{title} {what} {why} {next_step}"
  → _embed_commit(commit_id, combined_text)
       → model = get_embedding_model()  # None if ONNX deps missing
       → if model is None: return (no-op)
       → new_vec = model.embed_query(combined_text)  # (384,) float32, L2-normed
       → cache = load_embeddings(path)  # dict[str, list[float]]
       → cache[commit_id] = new_vec.tolist()
       → if len(cache) > link_scan_window * 2:
             # evict oldest: sort by commit_id (C### lexicographic = chronological)
             sorted_ids = sorted(cache.keys())
             for old_id in sorted_ids[:len(cache) - link_scan_window * 2]:
                 del cache[old_id]
       → save_embeddings(cache, path)  # atomic write
  → _compute_links(commit_id, branch, title, what, why, next_step, files_changed, new_vec=new_vec)
       → ... entity, causal, supersession links unchanged ...
       → step 4 (semantic, per-commit):
           → cache = _load_commit_embeddings(window_commit_ids)
             # returns dict[str, np.ndarray], only IDs present in cache
             # new_vec already computed by _embed_commit — no second inference pass
           → for each commit in window:
               if new_vec is not None and cid in cache:
                   score = float(cache[cid] @ new_vec)  # dot product of L2-normed vecs = cosine sim
               else:
                   score = _jaccard(new_keywords, old_keywords)
               if score > link_semantic_threshold: emit semantic link
```

### Error Handling

- ONNX unavailable → `get_embedding_model()` returns `None` → `_embed_commit` is a no-op → `_compute_links` uses Jaccard for all semantic links (unchanged behavior)
- Cache file corrupt/missing → `load_embeddings` returns `{}` → all commits missing from cache → Jaccard fallback for all semantic links
- Any embedding error → catch, log, fall back to Jaccard for that commit
- Merge path (A-MAC) → `_embed_commit` not called (no new commit_id created)

### Testing

- `test_memory.py`: mock `get_embedding_model()` with a mock that returns controlled cosine scores; verify semantic link uses cosine path when model available
- Verify Jaccard fallback when `get_embedding_model()` returns `None`
- Verify per-commit fallback: commit with no cached embedding uses Jaccard, commit with cached embedding uses cosine — in the same `_compute_links` call
- Verify cache grows, caps at `link_scan_window * 2`, oldest evicted
- Verify A-MAC merge path does NOT add an embedding entry

---

## Sub-project 2: Rolling Summary Prompt — Include Current Summary Content

### What Already Exists

The rolling summary compression prompt is **already implemented** in `memory.commit()` at line 449 (threshold: 1200 chars). It fires when `len(current_summary) > 1200 and compressed_summary is None` and appends a warning to the return value.

**The gap**: the prompt tells Claude Code to provide a compressed summary but does NOT include the current summary text. Claude Code has to call `gcc_context` to retrieve it, adding an extra round-trip before it can write the compressed version. This breaks the two-call pattern in practice.

**Secondary gap**: the `mcp_server.py` docstring at line 273 says "exceeds 2000 chars" — wrong, the actual threshold is 1200.

### Change

Modify the compression prompt in `memory.commit()` (lines 452–459) to include the current summary inline:

**Before** (current):
```
⚠️ Rolling summary is getting long ({N} chars). To preserve summary quality
(GCC paper S_t = f(S_{t-1}, D_t)), call gcc_commit with compressed_summary=
containing a concise compression of the current rolling summary, or call
gcc_consolidate to compress project memory. Without compression, the summary
will degrade to structured truncation.
```

**After** (new):
```
⚠️ Rolling summary is getting long ({N} chars). To preserve summary quality,
call gcc_commit again with compressed_summary='<your synthesis>'. Current
summary to compress:

---
{current_summary}
---

Write a 2-3 sentence synthesis capturing the key decisions and current
direction, then pass it as compressed_summary= in your next gcc_commit call.
```

Also fix `mcp_server.py` docstring line 273: `"2000 chars"` → `"1200 chars"`.

### Data Flow (unchanged)

The two-call pattern already works:
1. `gcc_commit(what=..., why=...)` → summary > 1200 → returns prompt with current summary inline
2. Claude Code synthesizes → `gcc_commit(what=..., why=..., compressed_summary="<synthesis>")` → Strategy 1 used, clean summary written

### Testing

- `test_memory.py` / `test_mcp_server.py`: verify compression prompt includes the current summary text when > 1200 chars
- Verify prompt absent when ≤ 1200 chars or when `compressed_summary` provided
- Verify docstring in `mcp_server.py` updated to say 1200 (manual verification — docstrings not unit-tested)

---

## Sub-project 3: NeurIPS Audit Re-run

No implementation. Spawn reviewer agents (same as C018) against current codebase to verify improved scores after C020 fixes. Expected: A-MAC ~3/10 → ~6/10 (inversion fixed), docs honesty improvements across all papers.

---

## Files Changed

| File | Change |
|---|---|
| `ccr/core/memory.py` | ONNX embed at commit time, cosine similarity in `_compute_links`, include current summary in compression prompt |
| `ccr/context/embeddings.py` | No change (reused as-is) |
| `ccr/mcp_server.py` | Fix docstring threshold 2000 → 1200 |
| `tests/unit/test_memory.py` | Embedding path + fallback tests, updated compression prompt tests |
| `tests/unit/test_mcp_server.py` | Updated docstring check if tested |

## Out of Scope

- Persisting embeddings for commits beyond `link_scan_window * 2`
- Changing the 1200-char threshold for the compression prompt
- Any LLM calls from within CCR itself
- Landlock sandbox (separate feature)
