# Hybrid commit recall: design

Branch `feat/hybrid-commit-recall` (from `feat/bench-suite`).

## Why

The full benchmark ranks `ccr_memory` (`gcc_recall` over commits) last on every
dataset, far behind plain BM25 (LoCoMo nDCG@10 0.063 against 0.406). Users who
ask natural-language questions of long memory get little back. The research papers
CCR is built on (A-RAG, ExpRAG, A-MEM, EverMemOS, Memori) describe better
retrieval than CCR currently implements (`docs/papers.md`: ExpRAG about 25%, A-RAG about 45%).

## Root causes (read from the code)

1. `commit_search_text` runs an SQLite FTS `MATCH` on the whole question, which needs
   every word to match, or falls back to one exact substring. Questions with `?` can
   break the FTS syntax.
2. The semantic fallback `_search_commits` returns at most 5 commits, and its BM25
   fallback rescans every commit per query and is skipped whenever any exact or
   semantic hit exists.
3. Without `sqlite-vec` (not in the default install) commit embeddings live in a
   gzip cache capped at `link_scan_window * 2` entries, so dense search only sees
   recent commits.
4. `RecallEngine._commit_evidence` then drops every hit that shares no word with the
   query, which removes exactly the semantic hits dense search adds.
5. The default lexical reranker adds up to +0.25 by word overlap and can reorder a
   good fused ranking.

## Goals

- `gcc_recall` returns useful commits for natural-language questions: recall@10 and
  nDCG@10 near `ccr_index_*` (BM25 / hybrid) on LoCoMo and LongMemEval.
- Short project notes behave as today: the exact-phrase path stays first.
- Each paper technique is added only if it measurably helps; the write-up reports
  the ones that did nothing.

## Non-goals

- Changing the benchmark harness, datasets or metrics to flatter any system.
- Routing recall through `RepoIndex` (loses facts, graph and discussion sources).
- LLM calls inside recall (CCR tools stay zero-LLM).

## Approaches considered

- A. Fix the plumbing: BM25 plus dense with reciprocal-rank fusion, no 5-cap, no
  word-overlap filter. Chosen as step one.
- B. A plus paper techniques, each tested alone. Chosen as step two.
- C. Route recall through `RepoIndex`. Rejected: drops facts, graph, discussions.

## Design

New module `ccr/core/commit_search.py` (target under 300 lines):

- `CommitSearchIndex`: built lazily from `commit_list`, updated when the commit
  count changes. Holds a BM25 index (OR semantics, k1=1.5, b=0.75, stop words
  removed) and, when the ONNX model is present, a dense matrix embedded in
  batches. Vectors persist to `.ccr/commit_search_vecs.npz` (atomic write) so a
  restart does not re-embed. No size cap.
- `rrf(rankings, k=60)`: reciprocal-rank fusion.
- `RecallOptions` (frozen dataclass): `commit_search` (`"hybrid"`, `"bm25"`,
  `"dense"`, `"legacy"`), `rerank` (bool), and one flag per paper technique below.
  Defaults are the winners of the ablation. `RecallEngine(mem, options=...)`.
- `_commit_evidence` uses the index for ranked hits; exact phrase hits keep rank 1..n;
  hybrid hits fill the rest; scores are scaled so the lexical reranker cannot
  reorder across more than a small margin (and `rerank=False` skips it).

Paper techniques, each behind its own flag and measured alone:

| Flag | Paper | Idea |
|---|---|---|
| `link_expansion` | A-MEM / MAGMA | add top hits' linked commits (entity / causal / semantic links) at a discounted score |
| `scene_expansion` | EverMemOS | add siblings from the same connected-component cluster |
| `triple_boost` | Memori | boost commits whose stored triples match query entities |
| `chunk_dense` | A-RAG | embed 800-token chunks, score a commit by its best chunk |

Bench: `ccr_memory` uses the product default; `ccr_memory_legacy` uses
`commit_search="legacy"` (today's behavior); each ablation is a registered
variant `ccr_memory+<flag>`.

## Evaluation protocol (fixed before looking at results)

- Development sample: seed 7, 30% of worlds (LoCoMo 3 conversations, LongMemEval
  150 questions), via a new `--fraction` option. The earlier seed-42 pilot and the
  running full benchmark are not used for tuning.
- Metric: nDCG@10 and recall@10 with the paired bootstrap in `ccr bench compare`.
- Keep rule: a technique stays only if it improves nDCG@10 on LoCoMo-dev or
  LongMemEval-dev with paired p < 0.05 and is not significantly worse on the other.
  Defaults for hybrid search itself (RRF k, rerank on/off) are picked the same way.
- Final numbers: full runs of all datasets, reported with and without the dev
  worlds, so the held-out subset is visible. SciFact, NFCorpus and code are not
  used for tuning.

## Risks

- Existing tests (3142) pin current `gcc_recall` behavior; legacy mode and the
  exact-phrase-first rule keep them green. Anything that cannot is called out.
- Dense search needs ONNX (optional). Without it the index is BM25-only; still a
  large gain.
- Small dev samples cannot detect small gains: a "no effect" verdict means "not
  detectable here", and the write-up says so.

## Task order

1. `--fraction` option in `ccr bench run`.
2. `commit_search.py`: BM25 + RRF + dense, with tests.
3. `RecallOptions`, `_commit_evidence` rewrite, legacy mode, tests.
4. Bench retrievers `ccr_memory`, `ccr_memory_legacy`; dev run for A; decide defaults.
5. Techniques one by one (links, scenes, triples, chunks): test, implement, dev run, keep or drop.
6. Full run with the final defaults, ranking, docs (`benchmarks.md`, `papers.md`).
