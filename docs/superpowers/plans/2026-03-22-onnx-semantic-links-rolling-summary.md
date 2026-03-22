# ONNX Semantic Links + Rolling Summary Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire ONNX embeddings into commit cross-linking (replacing word Jaccard for semantic links) and include current summary inline in the rolling summary compression prompt.

**Architecture:** `memory.py` gains `_get_commit_embeddings_path`, `_embed_commit` (returns vector, saves to `.ccr/commit_embeddings.json.gz`), and `_load_commit_embeddings`. `_compute_links` gains a `new_vec` param; semantic step uses dot product when vector available, Jaccard fallback otherwise. Rolling summary prompt gains the current summary inline. All ONNX code is gated — zero behavior change when deps unavailable.

**Tech Stack:** Python 3.11+, `ccr/context/embeddings.py` (existing ONNX backend, all-MiniLM-L6-v2), `numpy` (soft dep, local import only), `pytest`

**Spec:** `docs/superpowers/specs/2026-03-21-onnx-semantic-links-rolling-summary-design.md`

---

## File Map

| File | Change |
|---|---|
| `ccr/core/memory.py` | Add 3 methods, modify `_compute_links` + `commit` + rolling summary prompt |
| `ccr/mcp_server.py` | Fix docstring threshold `2000` → `1200` |
| `tests/unit/test_memory.py` | Add `TestCommitEmbeddings` class + update `TestRollingSummaryCompressionPrompt` |

---

## Task 1: Add commit embeddings cache infrastructure

**Files:**
- Modify: `ccr/core/memory.py` (after `_get_links_path` at line ~579)
- Test: `tests/unit/test_memory.py` (new `TestCommitEmbeddings` class)

- [ ] **Step 1: Write the failing tests**

First, add this import near the top of `tests/unit/test_memory.py` (after existing imports):

```python
from unittest.mock import MagicMock, patch
```

Then add this class at the end of `tests/unit/test_memory.py`:

```python
class TestCommitEmbeddings:
    """Tests for ONNX commit embedding cache."""

    def test_get_commit_embeddings_path(self, memory):
        path = memory._get_commit_embeddings_path()
        assert path.endswith("commit_embeddings.json.gz")
        assert memory.ccr_root in path

    def test_embed_commit_no_op_when_model_unavailable(self, memory):
        """When no ONNX model, _embed_commit returns None without error."""
        with patch("ccr.core.memory.get_embedding_model", return_value=None):
            result = memory._embed_commit("C001", "some text")
        assert result is None

    def test_embed_commit_stores_vector_in_cache(self, memory):
        """When model available, vector is persisted to cache file."""
        import numpy as np

        fake_vec = np.ones(384, dtype=np.float32)
        fake_vec /= np.linalg.norm(fake_vec)

        mock_model = MagicMock()
        mock_model.embed_query.return_value = fake_vec

        with patch("ccr.core.memory.get_embedding_model", return_value=mock_model):
            result = memory._embed_commit("C001", "test text")

        assert result is not None
        # Cache file must exist
        assert os.path.isfile(memory._get_commit_embeddings_path())
        # Returned vector matches
        np.testing.assert_allclose(result, fake_vec, rtol=1e-5)

    def test_embed_commit_cache_grows(self, memory):
        """Each call appends one entry to cache."""
        import numpy as np

        def make_vec():
            v = np.random.rand(384).astype(np.float32)
            return v / np.linalg.norm(v)

        mock_model = MagicMock()
        mock_model.embed_query.side_effect = [make_vec(), make_vec(), make_vec()]

        with patch("ccr.core.memory.get_embedding_model", return_value=mock_model):
            memory._embed_commit("C001", "a")
            memory._embed_commit("C002", "b")
            memory._embed_commit("C003", "c")

        from ccr.context.embeddings import load_embeddings
        cache = load_embeddings(memory._get_commit_embeddings_path())
        assert set(cache.keys()) == {"C001", "C002", "C003"}

    def test_embed_commit_caps_cache(self, memory):
        """Cache is capped at link_scan_window * 2; oldest entries evicted."""
        import numpy as np

        cap = memory.config.link_scan_window * 2  # default: 40

        def make_vec():
            v = np.random.rand(384).astype(np.float32)
            return v / np.linalg.norm(v)

        mock_model = MagicMock()
        mock_model.embed_query.side_effect = [make_vec() for _ in range(cap + 5)]

        with patch("ccr.core.memory.get_embedding_model", return_value=mock_model):
            for i in range(cap + 5):
                memory._embed_commit(f"C{i:03d}", f"text {i}")

        from ccr.context.embeddings import load_embeddings
        cache = load_embeddings(memory._get_commit_embeddings_path())
        assert len(cache) == cap
        # Oldest evicted: C000..C004 should be gone
        assert "C000" not in cache
        assert "C004" not in cache

    def test_load_commit_embeddings_returns_ndarrays(self, memory):
        """_load_commit_embeddings converts list[float] → np.ndarray."""
        import numpy as np

        fake_vec = np.ones(384, dtype=np.float32)
        fake_vec /= np.linalg.norm(fake_vec)
        mock_model = MagicMock()
        mock_model.embed_query.return_value = fake_vec

        with patch("ccr.core.memory.get_embedding_model", return_value=mock_model):
            memory._embed_commit("C001", "text")

        result = memory._load_commit_embeddings(["C001"])
        assert "C001" in result
        assert isinstance(result["C001"], np.ndarray)
        assert result["C001"].shape == (384,)

    def test_load_commit_embeddings_missing_ids_ignored(self, memory):
        """IDs not in cache are silently omitted."""
        result = memory._load_commit_embeddings(["C999", "C998"])
        assert result == {}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/qbit-glitch/Desktop/coding-projects/powering_claude_with_less_tokens
source .venv/bin/activate
pytest tests/unit/test_memory.py::TestCommitEmbeddings -x -q 2>&1 | head -30
```

Expected: `AttributeError: 'MemoryManager' object has no attribute '_get_commit_embeddings_path'`

- [ ] **Step 3: Add `from ccr.context.embeddings import get_embedding_model, save_embeddings, load_embeddings` import to `memory.py`**

In `ccr/core/memory.py`, find the existing import block near the top. After the `from ccr.core.types import CCRConfig, CommitLink` line, add:

```python
from ccr.context.embeddings import get_embedding_model, save_embeddings, load_embeddings
```

- [ ] **Step 4: Add the three new methods to `memory.py`**

Insert these three methods after `_get_links_path` (line ~579) and before `_load_links`:

```python
def _get_commit_embeddings_path(self) -> str:
    return os.path.join(self.ccr_root, "commit_embeddings.json.gz")

def _embed_commit(self, commit_id: str, text: str):
    """Embed commit text and persist to cache. Returns vector or None.

    Appends to .ccr/commit_embeddings.json.gz (capped at
    link_scan_window * 2 entries, oldest evicted). Returns the computed
    (384,) float32 L2-normalized vector so the caller can reuse it
    without a second inference pass. Returns None if ONNX unavailable.
    """
    model = get_embedding_model()
    if model is None:
        return None
    import numpy as np  # soft dep — only reachable when ONNX available
    try:
        vec = model.embed_query(text)
        cache = load_embeddings(self._get_commit_embeddings_path())
        cache[commit_id] = vec.tolist()
        cap = self.config.link_scan_window * 2
        if len(cache) > cap:
            for old_id in sorted(cache.keys())[: len(cache) - cap]:
                del cache[old_id]
        save_embeddings(cache, self._get_commit_embeddings_path())
        return vec
    except Exception:
        return None

def _load_commit_embeddings(self, commit_ids: list) -> dict:
    """Load cached embeddings for given commit IDs as numpy arrays.

    Returns dict[str, np.ndarray] with only IDs present in cache.
    Silently omits missing IDs. Returns empty dict on any error.
    """
    try:
        import numpy as np  # soft dep
        raw = load_embeddings(self._get_commit_embeddings_path())
        return {
            cid: np.array(raw[cid], dtype=np.float32)
            for cid in commit_ids
            if cid in raw
        }
    except Exception:
        return {}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/test_memory.py::TestCommitEmbeddings -x -q 2>&1 | tail -10
```

Expected: `7 passed` (some may be skipped if ONNX unavailable — that's fine)

- [ ] **Step 6: Run full test suite to verify no regressions**

```bash
pytest tests/unit/ -x -q 2>&1 | tail -5
```

Expected: all passing (same count as before)

- [ ] **Step 7: Commit**

```bash
git add ccr/core/memory.py tests/unit/test_memory.py
git commit -m "feat: add commit embedding cache infrastructure (_embed_commit, _load_commit_embeddings)"
```

---

## Task 2: Wire ONNX embeddings into `_compute_links` and `commit()`

**Files:**
- Modify: `ccr/core/memory.py` (lines ~408-416 in `commit()`, lines ~675-770 in `_compute_links`)
- Test: `tests/unit/test_memory.py` (new tests in `TestCommitEmbeddings`)

- [ ] **Step 1: Write the failing tests**

Add these tests inside `TestCommitEmbeddings` in `tests/unit/test_memory.py`:

```python
def test_compute_links_uses_cosine_when_vec_available(self, memory):
    """When new_vec provided and old embedding cached, cosine used for semantic link."""
    import numpy as np

    # Commit C001 — store its embedding in cache
    vec_old = np.zeros(384, dtype=np.float32)
    vec_old[0] = 1.0  # unit vector in dim 0

    from ccr.context.embeddings import save_embeddings
    cache = {"C001": vec_old.tolist()}
    save_embeddings(cache, memory._get_commit_embeddings_path())

    # Parse a fake C001 in recent commits
    with patch.object(memory, "_parse_recent_commit_data") as mock_recent:
        mock_recent.return_value = [{
            "id": "C001", "title": "old", "what": "old work", "why": "old reason",
            "files": [], "next_step": "old next",
        }]

        # new_vec similar to vec_old — dot product ~1.0 (above default threshold)
        vec_new = np.zeros(384, dtype=np.float32)
        vec_new[0] = 1.0

        links = memory._compute_links(
            "main", "C002", "similar title", "similar work", "similar reason",
            [], "next", new_vec=vec_new,
        )

    semantic_links = [l for l in links if l.link_type == "semantic"]
    assert len(semantic_links) == 1
    assert semantic_links[0].score > 0.9  # cosine of near-identical vectors

def test_compute_links_falls_back_to_jaccard_when_no_vec(self, memory):
    """When new_vec=None, semantic link uses Jaccard (existing behavior)."""
    with patch.object(memory, "_parse_recent_commit_data") as mock_recent:
        mock_recent.return_value = [{
            "id": "C001", "title": "unique rare keyword zebra",
            "what": "zebra keyword work", "why": "zebra reason",
            "files": [], "next_step": "next",
        }]
        links = memory._compute_links(
            "main", "C002", "unique rare keyword zebra",
            "zebra keyword work", "zebra reason", [], "next", new_vec=None,
        )
    semantic_links = [l for l in links if l.link_type == "semantic"]
    assert len(semantic_links) == 1  # Jaccard high for identical rare keywords

def test_commit_calls_embed_commit_and_passes_vec_to_compute_links(self, memory):
    """commit() embeds and passes new_vec to _compute_links (no double embed)."""
    import numpy as np

    fake_vec = np.ones(384, dtype=np.float32)
    fake_vec /= np.linalg.norm(fake_vec)

    mock_model = MagicMock()
    mock_model.embed_query.return_value = fake_vec

    with patch("ccr.core.memory.get_embedding_model", return_value=mock_model):
        with patch.object(memory, "_compute_links", wraps=memory._compute_links) as mock_cl:
            memory.commit("T1", "did A", "reason A", ["a.py"], "next A")

    # embed_query called exactly once (in _embed_commit; NOT again in _compute_links)
    assert mock_model.embed_query.call_count == 1
    # _compute_links received new_vec as keyword argument
    _, kwargs = mock_cl.call_args
    assert kwargs.get("new_vec") is not None
```

Also add these two tests (M1: A-MAC merge path, M2: mixed-mode per-commit fallback):

```python
def test_embed_commit_not_called_on_amac_merge_path(self, memory):
    """When A-MAC takes the merge path (new_score > conflict_score → early return),
    _embed_commit is not called — it lives after the merge return point in commit().

    Key: raw_sim = 0.50*file_sim + 0.50*kw_sim. Same files → file_sim=1 but
    novelty suppressed (raw_sim≥0.5 → new_score ≈ conflict_score → tie → fall-through).
    Fix: drive conflict via keyword overlap only, use DIFFERENT files so file_sim=0.

    stored_score=0.05 short-circuits to conflict_novelty=0.5 (memory.py line ~1265):
      conflict_score = 0.50*0.5 + 0.35*0.5 + 0.15*1.0 = 0.575
      Greek-letter keywords: overlap=9/union=11 → kw_sim≈0.818
      raw_sim(new) = 0.50*0 + 0.50*0.818 = 0.409  (file_sim=0, different files)
      novelty(new) = 1 - 0.409 = 0.591
      new_score = 0.50*0.5 + 0.35*0.591 + 0.15*1.0 = 0.607 > 0.575 → merge ✓
    """
    import numpy as np

    fake_vec = np.ones(384, dtype=np.float32)
    fake_vec /= np.linalg.norm(fake_vec)
    mock_model = MagicMock()
    mock_model.embed_query.return_value = fake_vec

    with patch("ccr.core.memory.get_embedding_model", return_value=mock_model):
        # First commit — keyword-rich with rare Greek letters, file "f.py"
        memory.commit(
            "alpha beta gamma delta epsilon sigma",
            "alpha beta gamma work", "delta epsilon sigma reason",
            ["f.py"], "alpha next",
            admission_threshold=0.3,
        )
        count_after_first = mock_model.embed_query.call_count
        assert count_after_first == 1

        # Lower C001's stored score to 0.05 → conflict_score=0.575 via short-circuit.
        # Score is written as "**Score**: 1.00" (:.2f format, memory.py line ~379),
        # so use regex replace to match any stored score value.
        import re as _re
        path = memory._get_commits_path("main")
        content = memory._read_file(path)
        content = _re.sub(r'\*\*Score\*\*: [\d.]+', '**Score**: 0.05', content)
        memory._write_file(path, content)

        # Second commit: overlapping keywords (kw_sim≈0.818) but different file "g.py"
        # → file_sim=0, novelty=0.591, new_score=0.607 > conflict_score=0.575 → merge
        memory.commit(
            "alpha beta gamma delta epsilon sigma zeta eta",
            "alpha beta gamma zeta work", "delta epsilon sigma eta reason",
            ["g.py"], "zeta eta next",
            admission_threshold=0.3,
        )

    # Merge path early return: _embed_commit was not reached
    assert mock_model.embed_query.call_count == count_after_first

def test_compute_links_per_commit_mixed_fallback(self, memory):
    """In one _compute_links call: commit with cached embedding uses cosine,
    commit without cached embedding uses Jaccard — both in the same scan."""
    import numpy as np
    from ccr.context.embeddings import save_embeddings

    # C001 has a cached embedding (unit vector in dim 0)
    vec_old = np.zeros(384, dtype=np.float32)
    vec_old[0] = 1.0
    save_embeddings({"C001": vec_old.tolist()}, memory._get_commit_embeddings_path())

    # C002 has NO cached embedding

    with patch.object(memory, "_parse_recent_commit_data") as mock_recent:
        mock_recent.return_value = [
            # C001: has embedding, will use cosine
            {"id": "C001", "title": "alpha beta gamma", "what": "alpha work",
             "why": "alpha reason", "files": []},
            # C002: no embedding, will fall back to Jaccard
            # Use rare keywords that match new commit text → Jaccard fires
            {"id": "C002", "title": "unique zebra quux xyzzy",
             "what": "zebra quux xyzzy", "why": "xyzzy reason", "files": []},
        ]

        # new_vec is a unit vector in dim 0 — cosine with C001 ≈ 1.0
        vec_new = np.zeros(384, dtype=np.float32)
        vec_new[0] = 1.0

        links = memory._compute_links(
            "main", "C003",
            "unique zebra quux xyzzy alpha beta gamma",
            "zebra quux xyzzy alpha work", "alpha reason",
            [], "next", new_vec=vec_new,
        )

    # C001: cosine similarity ~1.0 → semantic link
    # C002: Jaccard on rare matching keywords → semantic link
    semantic_links = {l.target: l for l in links if l.link_type == "semantic"}
    assert "C001" in semantic_links, "C001 should link via cosine"
    assert "C002" in semantic_links, "C002 should link via Jaccard fallback"
    # C001 cosine score should be very high
    assert semantic_links["C001"].score > 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_memory.py::TestCommitEmbeddings::test_compute_links_uses_cosine_when_vec_available tests/unit/test_memory.py::TestCommitEmbeddings::test_compute_links_falls_back_to_jaccard_when_no_vec tests/unit/test_memory.py::TestCommitEmbeddings::test_commit_calls_embed_commit_and_passes_vec_to_compute_links tests/unit/test_memory.py::TestCommitEmbeddings::test_embed_commit_not_called_on_amac_merge_path tests/unit/test_memory.py::TestCommitEmbeddings::test_compute_links_per_commit_mixed_fallback -x -q 2>&1 | head -20
```

Expected: `TypeError: _compute_links() got an unexpected keyword argument 'new_vec'`

- [ ] **Step 3: Modify `_compute_links` to accept and use `new_vec`**

In `ccr/core/memory.py`, update `_compute_links` signature (line ~675):

```python
def _compute_links(
    self,
    branch: str,
    commit_id: str,
    title: str,
    what: str,
    why: str,
    files_changed: list[str],
    next_step: str,
    new_vec=None,  # (384,) float32 L2-normalized ndarray, or None
) -> list[CommitLink]:
```

Update the docstring's item 4 to:

```
4. Semantic links: dense cosine similarity when ONNX embedding available
   for both commits; word Jaccard fallback otherwise (per-commit fallback).
   Cf. MAGMA semantic graph which uses dense vector cosine — we now use
   dense cosine when embeddings are cached, Jaccard otherwise.
```

Replace the semantic link block (lines ~760-768):

```python
            # 4. Semantic links (only if no other link type to this target)
            if not has_typed_link:
                cached = self._load_commit_embeddings([cid])
                if new_vec is not None and cid in cached:
                    # Dense cosine: dot product of L2-normalized vectors
                    score = float(cached[cid] @ new_vec)
                else:
                    old_text = f"{commit.get('title', '')} {commit.get('what', '')} {commit.get('why', '')}".lower()
                    old_keywords = self._extract_keywords(old_text)
                    score = self._jaccard(new_keywords, old_keywords)
                if score > self.config.link_semantic_threshold:
                    links.append(CommitLink(
                        target=cid, link_type="semantic", score=round(score, 3),
                    ))
```

- [ ] **Step 4: Modify `commit()` to embed and pass vector**

In `ccr/core/memory.py`, find the cross-linking block in `commit()` (lines ~408-416):

```python
        # Heuristic commit cross-linking (A-MEM/MAGMA inspired taxonomy)
        try:
            commit_links = self._compute_links(
                branch, commit_id, title, what, why, files_changed, next_step,
            )
```

Replace with:

```python
        # Heuristic commit cross-linking (A-MEM/MAGMA inspired taxonomy)
        # _embed_commit is called here (normal path only) and its vector is
        # passed to _compute_links to avoid a second inference pass.
        try:
            new_vec = self._embed_commit(
                commit_id, f"{title} {what} {why} {next_step}"
            )
            commit_links = self._compute_links(
                branch, commit_id, title, what, why, files_changed, next_step,
                new_vec=new_vec,
            )
```

- [ ] **Step 5: Run new tests**

```bash
pytest tests/unit/test_memory.py::TestCommitEmbeddings -x -q 2>&1 | tail -10
```

Expected: all passing

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/unit/ -x -q 2>&1 | tail -5
```

Expected: all passing

- [ ] **Step 7: Commit**

```bash
git add ccr/core/memory.py tests/unit/test_memory.py
git commit -m "feat: wire ONNX embeddings into commit cross-linking (cosine replaces Jaccard for semantic links)"
```

---

## Task 3: Include current summary inline in compression prompt

**Files:**
- Modify: `ccr/core/memory.py` (lines 452-459)
- Modify: `ccr/mcp_server.py` (docstring line ~273)
- Test: `tests/unit/test_memory.py` (`TestRollingSummaryCompressionPrompt`)

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_memory.py`, inside `TestRollingSummaryCompressionPrompt` (after the existing tests around line 1573), add:

```python
def test_warning_includes_current_summary_inline(self, memory):
    """Compression prompt includes the actual summary so Claude Code
    doesn't need an extra round-trip to retrieve it."""
    summary = "x" * 700 + " important context here"
    memory._write_rolling_summary("main", summary)
    result = memory.commit("Next", "added more", "reason", ["f.py"], "done")
    assert "important context here" in result
    # Should show the summary between delimiters
    assert "---" in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest "tests/unit/test_memory.py::TestRollingSummaryCompressionPrompt::test_warning_includes_current_summary_inline" -x -q
```

Expected: FAIL — summary content not currently in prompt

- [ ] **Step 3: Update the compression prompt in `memory.py`**

In `ccr/core/memory.py`, replace lines 452-459:

```python
        if len(current_summary) > summary_compression_threshold and compressed_summary is None:
            result += (
                f"\n\n\u26a0\ufe0f Rolling summary is getting long ({len(current_summary)} chars). "
                f"To preserve summary quality (GCC paper S_t = f(S_{{t-1}}, D_t)), "
                f"call gcc_commit with compressed_summary= containing a concise "
                f"compression of the current rolling summary, or call gcc_consolidate "
                f"to compress project memory. Without compression, the summary will "
                f"degrade to structured truncation."
            )
```

With:

```python
        if len(current_summary) > summary_compression_threshold and compressed_summary is None:
            result += (
                f"\n\n\u26a0\ufe0f Rolling summary is getting long ({len(current_summary)} chars). "
                f"Call gcc_commit again with compressed_summary='<your 2-3 sentence synthesis>'. "
                f"Current summary to compress:\n\n---\n{current_summary}\n---\n\n"
                f"Write a concise synthesis capturing key decisions and current direction, "
                f"then pass it as compressed_summary= in your next gcc_commit call. "
                f"Alternatively, call gcc_consolidate to compress project memory."
            )
```

- [ ] **Step 4: Fix docstring in `mcp_server.py`**

In `ccr/mcp_server.py`, line ~273, change:

```
    When the rolling summary exceeds 2000 chars, the return value includes a
```

To:

```
    When the rolling summary exceeds 1200 chars, the return value includes a
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/unit/test_memory.py::TestRollingSummaryCompressionPrompt -x -q 2>&1 | tail -10
```

Expected: all passing (including the new test and all existing ones)

Note: existing test `test_warning_suggests_two_call_pattern` checks for `"gcc_consolidate"` — the new prompt still contains this, so it passes. Test `test_long_summary_triggers_warning` checks for `"Rolling summary is getting long"` — still present.

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/unit/ -x -q 2>&1 | tail -5
```

Expected: all passing

- [ ] **Step 7: Commit**

```bash
git add ccr/core/memory.py ccr/mcp_server.py tests/unit/test_memory.py
git commit -m "feat: include current summary inline in rolling summary compression prompt"
```

---

## Task 4: NeurIPS Audit Re-run

**No code changes.** Spawn reviewer agents to verify improved scores after C020 fixes.

- [ ] **Step 1: Run the audit**

Spawn a NeurIPS reviewer agent with the following prompt (same as C018 approach):

> "You are a brutal NeurIPS reviewer. Your job is to evaluate how faithfully CCR implements the claims in each of these papers. Read the actual paper PDF and compare line-by-line against the CCR implementation. Score each paper 1-10 for implementation fidelity. Be specific about what matches and what doesn't. Papers: GCC (arXiv:2508.00031), A-MAC (arXiv:2603.04549), ACE (arXiv:2510.04618), SkillRL (arXiv:2602.08234), MCE (arXiv:2601.21557), RLM (arXiv:2512.24601), CER (arXiv:2506.06698), A-RAG (arXiv:2602.03442), A-MEM (arXiv:2502.12110), MAGMA (arXiv:2601.03236). Focus on: (1) algorithm correctness vs paper pseudocode, (2) honest documentation vs overclaiming, (3) key gaps. Code at: ccr/core/memory.py, ccr/ace/playbook.py, ccr/context/indexer.py, ccr/context/embeddings.py, ccr/mcp_server.py, CLAUDE.md."

- [ ] **Step 2: Document results**

Save audit findings to `.ccr/neurips_audit_2026-03-22.md` and commit with `gcc_commit`.

---

## Completion

After all tasks pass:

```bash
pytest tests/unit/ tests/integration/ -q 2>&1 | tail -5
```

Expected: all tests passing (≥1215).

Then call `gcc_commit` with what was done, and `ace_update_counters` on relevant playbook bullets.
