# CCR Benchmark Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A free, judge-free, reproducible `ccr bench` harness that measures CCR memory recall and `RepoIndex` search on public datasets and compares against baselines run under identical conditions.

**Architecture:** New package `ccr/bench/`. Every dataset adapter yields `World`s (a corpus plus queries with gold ids). Every system implements one `Retriever` protocol. A runner ingests a world into an isolated temp project, retrieves, optionally lets a local Qwen reader answer from the retrieved text, scores with deterministic metrics, and caches per-query results so runs resume.

**Tech Stack:** Python 3.11+, click, pytest, stdlib `urllib`/`json`, `pyarrow` (new optional extra `bench`, only for MemoryAgentBench parquet), Ollama HTTP API for the reader (`qwen3.8:27b`).

**Spec:** `docs/superpowers/specs/2026-11-14-benchmark-design.md`

## Global Constraints

- No paid API keys, no LLM judge of any kind; the reader only answers, it never scores.
- Public datasets only, downloaded at run time into a cache dir (never committed).
- CCR itself makes no LLM calls; the harness never modifies CCR retrieval behaviour.
- Reader: `qwen3.8:27b` via Ollama, `temperature` 0, `think` false; exact tag recorded in every report.
- Pilot is 10% sampled by whole world, fixed seed, sample ids saved; pilot numbers are labeled "pilot (10%)" and never used in README claims.
- Repo rules: files 200-400 lines, type hints everywhere, `@dataclass(frozen=True)` config, `logging` not `print`, `__all__` in every `__init__.py`, no hardcoded hyperparameters (top-k values, seed, chunk size come from a config dataclass).
- Tests: `.venv/bin/python -m pytest tests/unit/bench -x -q`; no test touches the network or the user's real `~/.ccr`.

## Review Focus

- A near-duplicate turn must not be merged away by commit admission, or gold ids vanish. Disable merging; test N similar turns give N retrievable ids.
- Natural-language queries containing `"`, `*`, `-`, `:`, `?` (and dia ids like `D1:3`) must neither crash retrieval nor silently return nothing. Report an empty-retrieval rate per system.
- Evidence strings that pack several ids in one field, or ids absent from the conversation, must be normalized or dropped and counted, not scored as misses.
- Queries with no gold evidence (LoCoMo adversarial, LongMemEval abstention) are excluded from retrieval metrics and counted in the report.
- `RecallEngine` caps `limit` at 25: memory-system `k` values above 25 must fail loudly, not silently score at 25.
- Reader returning empty text, timing out, or the Ollama server being down gives a clear error or a scored-zero-and-counted answer, never a crash mid-run.
- Full-context baseline must be reported "not run: exceeds reader context", never silently truncated.
- Runs must not write outside the temp project and the cache dir (no `.ccr/` in cwd, no global DB writes).

---

### Task 1: Types and deterministic metrics

**Files:**
- Create: `ccr/bench/__init__.py`, `ccr/bench/types.py`, `ccr/bench/metrics.py`
- Modify: `pyproject.toml` (add optional extra `bench = ["pyarrow>=15"]`)
- Test: `tests/unit/bench/test_metrics.py`

**Interfaces:**
- Produces (`types.py`, all frozen): `Doc(id: str, text: str)`; `Query(id: str, text: str, gold_ids: frozenset[str], answer: str | None, category: str)`; `World(id: str, docs: tuple[Doc, ...], queries: tuple[Query, ...], group: str)`; `BenchConfig(ks: tuple[int, ...], seed: int, sample_fraction: float, chunk_tokens: int, cache_dir: str)`.
- Produces (`metrics.py`): `recall_at_k(ranked: list[str], gold: frozenset[str], k: int) -> float`; `ndcg_at_k(ranked: list[str], gains: dict[str, int], k: int) -> float`; `mrr(ranked: list[str], gold: frozenset[str]) -> float`; `token_f1(pred: str, gold: str) -> float`; `exact_match(pred: str, gold: str) -> float`; `sub_exact_match(pred: str, gold: str) -> float`; `bleu1(pred: str, gold: str) -> float`; `count_tokens(text: str) -> int` (wraps `ccr.utils.tokens.count_tokens`).

- [ ] **Step 1: Write failing tests** with hand-computed values: `recall_at_k(["a","b","c"], frozenset({"b","z"}), 2) == 0.5`; `mrr(["x","b"], frozenset({"b"})) == 0.5`; `ndcg_at_k(["a","b"], {"b": 1}, 2) == 1/log2(3)` (linear gain, trec_eval style); `token_f1("the cat sat", "cat sat") == 0.8`; `exact_match` is case/punctuation/article-insensitive (SQuAD normalization); `sub_exact_match("It was Paris.", "paris") == 1.0`; empty gold returns 0.0 and never raises.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/unit/bench/test_metrics.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** the signatures above. SQuAD-style normalization (lowercase, strip punctuation and articles `a|an|the`, collapse whitespace) shared by `token_f1`/`exact_match`/`sub_exact_match`/`bleu1`.
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(bench): add types and judge-free metrics`.

### Task 2: Retriever protocol, BM25 and full-context baselines

**Files:**
- Create: `ccr/bench/retrievers/__init__.py`, `ccr/bench/retrievers/base.py`, `ccr/bench/retrievers/bm25.py`, `ccr/bench/retrievers/fullcontext.py`
- Test: `tests/unit/bench/test_baselines.py`

**Interfaces:**
- Consumes: `Doc`, `Query` from Task 1.
- Produces (`base.py`): `class Retriever(Protocol)` with `name: str`, `max_k: int | None`, `index(self, docs: Sequence[Doc], workdir: str) -> None`, `retrieve(self, query: str, k: int) -> list[str]` (ranked doc ids, may be shorter than `k`), `close(self) -> None`; `REGISTRY: dict[str, Callable[[], Retriever]]` and `register(name: str)` decorator; `class RetrieverUnavailable(Exception)` (raised when an optional dependency such as `onnxruntime` is missing; the runner skips that system and records why).
- Produces: `Bm25Retriever` (own Okapi BM25, k1=1.5, b=0.75, regex word tokenizer; independent of CCR), `FullContextRetriever` (returns every doc id in order; `name = "full_context"`).

- [ ] **Step 1: Write failing tests:** BM25 ranks the doc containing the rare query term first; `retrieve` on an empty index returns `[]`; queries with `"*-:?` characters do not raise; `FullContextRetriever.retrieve(q, 3)` returns all ids regardless of `k`.
- [ ] **Step 2: Run** the file. Expected: FAIL.
- [ ] **Step 3: Implement** both retrievers and the registry; BM25 tokenizer is lowercase `\w+`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(bench): retriever protocol with BM25 and full-context baselines`.

### Task 3: CCR memory retriever (isolated project)

**Files:**
- Create: `ccr/bench/retrievers/ccr_memory.py`
- Test: `tests/unit/bench/test_ccr_memory_retriever.py`

**Interfaces:**
- Consumes: `Retriever`/`register`; `MemoryManager(project_root).commit(...)` (`ccr/core/memory_pkg/memory_commit.py:52`); `RecallEngine(mm).recall(query, limit=..., include_sessions=False)` returning `RecallResult.evidence[*].title` (`ccr/core/recall.py`).
- Produces: `CcrMemoryRetriever` registered as `"ccr_memory"`, `max_k = 25`.

- [ ] **Step 1: Write failing tests** (temp dir, no network): ingest 5 near-identical docs with ids `D1:1`..`D1:5`; `retrieve("alpha beta", 5)` returns all five ids; a doc whose text is unique is returned first for a query on its unique word; queries `'what "is" *x* -y: D1:3?'` do not raise; `retrieve(q, 26)` raises `ValueError` mentioning 25; after the test, nothing exists under the cwd's `.ccr/` and `os.path.expanduser("~/.ccr")` is untouched (patch HOME as in `tests/unit/test_memory_recall_eval.py:_patch_home`).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** `index()`: init a `MemoryManager` in `workdir`, one `commit(title=doc.id, what=doc.text, why="bench", files_changed=[], next_step="", admission_threshold=1.0, rejection_threshold=0.0)` per doc. `retrieve()` maps evidence back to doc ids via `evidence.title`, drops non-commit sources, keeps rank order, de-duplicates. If recall raises on special characters, sanitize the query to word characters and retry once; never swallow into an empty list without incrementing `self.empty_count`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(bench): CCR memory retriever with admission merging disabled`.

### Task 4: CCR index retriever and dense baseline

**Files:**
- Create: `ccr/bench/retrievers/ccr_index.py`, `ccr/bench/retrievers/dense.py`
- Test: `tests/unit/bench/test_index_retrievers.py`

**Interfaces:**
- Consumes: `RepoIndex.build(root, extensions=...)` (`ccr/context/indexer.py:67`); `idx.bm25_search(query, top_k, file_glob)`, `idx.hybrid_search(query, model, top_k, file_glob)`, `idx.search(pattern)` returning dicts with a `path` key; `EmbeddingModel` from `ccr/context/embeddings.py`.
- Produces: `CcrIndexRetriever(mode: str)` registered as `ccr_index_bm25`, `ccr_index_hybrid`, `ccr_index_keyword` (writes each doc to `<workdir>/docs/<n>.txt`, keeps a file-name-to-doc-id map, `max_k = 100`); `DenseMiniLmRetriever` registered as `dense_minilm`, which raises `RetrieverUnavailable` (defined in `base.py`) when `onnxruntime` is missing.

- [ ] **Step 1: Write failing tests:** index 3 docs (ids with `/` and spaces), `ccr_index_bm25` returns the right doc id for a unique term; `top_k > 100` raises `ValueError`; `.txt` files are actually indexed (assert `idx.files` is non-empty); `DenseMiniLmRetriever` raises `RetrieverUnavailable` via monkeypatched import failure.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Pass `extensions={".txt"}` explicitly. For hybrid, build chunk/doc embeddings only if the model loads, else report `RetrieverUnavailable`, not a silent BM25 fallback.
- [ ] **Step 4: Run.** Expected: PASS. Then `pip install -e ".[semantic]"` into `.venv` so the dense baseline can run later.
- [ ] **Step 5: Commit** `feat(bench): CCR index and dense baselines`.

### Task 5: Sampling and dataset download cache

**Files:**
- Create: `ccr/bench/sampling.py`, `ccr/bench/download.py`
- Test: `tests/unit/bench/test_sampling.py`, `tests/unit/bench/test_download.py`

**Interfaces:**
- Produces: `sample_worlds(worlds: Sequence[World], fraction: float, seed: int, stratify_by: Callable[[World], str] | None = None) -> list[World]` (whole worlds; at least one per stratum; same seed gives same ids; `fraction == 1.0` returns all); `fetch(url: str, cache_dir: str, expected_sha256: str | None = None) -> str` (stdlib `urllib`, streams to `<cache_dir>/<hash-of-url>/<basename>`, skips if present, returns local path, records sha256 in a sidecar `.sha256` file, raises `DownloadError` on HTTP failure).

- [ ] **Step 1: Write failing tests:** 10 worlds, fraction 0.1 gives exactly 1; two calls with seed 7 agree; different seeds differ; stratified sample includes every stratum; `fetch` with a `file://` URL copies once and a second call does not re-copy; wrong `expected_sha256` raises `DownloadError`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** both modules.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(bench): seeded world sampling and cached downloads`.

### Task 6: Ollama reader

**Files:**
- Create: `ccr/bench/reader.py`
- Test: `tests/unit/bench/test_reader.py`

**Interfaces:**
- Produces: `class OllamaReader` with `__init__(self, model: str = "qwen3.8:27b", host: str | None = None, num_predict: int = 64, timeout_s: float = 600.0, transport: Callable[[str, dict], dict] | None = None)`, `answer(self, question: str, context: str) -> str`, `tag: str` property; `ReaderUnavailable(Exception)`. `host` defaults to `OLLAMA_HOST` env, then `http://127.0.0.1:11434`. `transport` is the injectable POST function used by tests.

- [ ] **Step 1: Write failing tests** using a fake transport: request body has `think: false`, `options.temperature == 0`, `options.num_predict == 64`, `stream: false`; empty `response` returns `""`; transport raising `ConnectionError` surfaces `ReaderUnavailable` with a message naming the host; prompt contains the instruction "Answer with a short phrase. If the context does not contain the answer, say unknown." followed by the context and question.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** with `urllib.request` against `/api/generate`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(bench): local Ollama reader`.

### Task 7: Runner with caching and resume

**Files:**
- Create: `ccr/bench/runner.py`, `ccr/bench/results.py`
- Test: `tests/unit/bench/test_runner.py`

**Interfaces:**
- Consumes: Tasks 1-6.
- Produces (`results.py`, frozen): `QueryResult(query_id, world_id, system, category, ranked_ids: tuple[str, ...], retrieved_tokens: int, answer: str | None, scores: dict[str, float], empty: bool, ingest_s: float, retrieve_s: float)`. `ResultStore(path: str)` with `has(key: str) -> bool`, `add(key: str, result: QueryResult) -> None`, `all() -> list[QueryResult]` backed by append-only JSONL.
- Produces (`runner.py`): `run_world(world: World, retriever: Retriever, cfg: BenchConfig, reader: OllamaReader | None, store: ResultStore, context_limit_tokens: int | None) -> int` (returns number of new results). Key = `f"{world.id}|{retriever.name}|{query.id}|{reader.tag if reader else '-'}"`. Context handed to the reader is the full text of the retrieved docs (looked up by id in `world.docs`), identical for every system; `retrieved_tokens` counts that same text. Queries with empty `gold_ids` and no `answer` are skipped; retrieval scores need non-empty gold. `FullContextRetriever` whose total tokens exceed `context_limit_tokens` stores no answer and a score `{"not_run": 1.0}`.

- [ ] **Step 1: Write failing tests** with a tiny in-memory world, a fake retriever and fake reader: results contain `recall@k` for each configured k; rerunning performs zero new work (resume); queries without gold are excluded and counted; `k` above `retriever.max_k` raises `ValueError`; fake reader returning `""` scores `f1 == 0.0` and is not dropped; over-limit full-context yields `not_run`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Use a fresh `tempfile.TemporaryDirectory` per world as `workdir`; always call `retriever.close()` in `finally`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(bench): resumable runner`.

### Task 8: Report and published-numbers table

**Files:**
- Create: `ccr/bench/report.py`, `docs/bench/published.yaml`
- Test: `tests/unit/bench/test_report.py`

**Interfaces:**
- Produces: `build_report(results: list[QueryResult], meta: dict[str, str]) -> BenchReport` (frozen; per-system, per-category means of every metric, `empty_rate`, `n_queries`, `n_excluded`, mean `retrieved_tokens`, ingest and retrieve seconds; `meta` carries dataset, git sha, reader tag, `pilot: "true"|"false"`, seed, sample ids); `BenchReport.to_json() -> str`, `to_markdown() -> str` (Table 1; title says "pilot (10%)" when `meta["pilot"] == "true"`); `load_published(path: str) -> list[PublishedEntry]` where `PublishedEntry(system, dataset, metric, value, source, locator, protocol_note)`; loading raises `ValueError` if `source` or `locator` is empty. `to_markdown` renders Table 2 separately.

- [ ] **Step 1: Write failing tests:** known results give expected means; pilot title appears; `empty_rate` computed; published entry without `source` raises; Table 2 never appears inside Table 1.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Leave `docs/bench/published.yaml` as `entries: []` with a header comment saying every number must be copied from the dataset authors' own judge-free table with a locator.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(bench): reports with separate published-numbers table`.

### Task 9: CLI

**Files:**
- Create: `ccr/cli_bench.py`
- Modify: `ccr/cli.py` (register the group near the other `cli.add_command` calls)
- Test: `tests/unit/bench/test_cli_bench.py`

**Interfaces:**
- Produces: `bench` click group with `ccr bench list` (datasets and systems), `ccr bench run DATASET --systems a,b --pilot/--full --reader/--no-reader --output DIR --cache-dir DIR --seed INT`, `ccr bench report DIR`. `DATASET` is one of `locomo`, `longmemeval`, `beir-scifact`, `beir-nfcorpus`, `memoryagentbench`, `code`. Adapters are looked up in `ADAPTERS: dict[str, Callable[[BenchConfig], list[World]]]` in `ccr/bench/datasets/__init__.py` (created here, empty-registry tested with a fake adapter).

- [ ] **Step 1: Write failing tests** with `CliRunner` and a registered fake adapter: `list` prints the fake dataset; `run fake --pilot --no-reader` writes `results.jsonl`, `report.json`, `report.md` in `--output`; missing Ollama with `--reader` exits non-zero with the host in the message; unknown dataset exits non-zero.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** `--pilot` is default and sets `sample_fraction` 0.1; `--full` sets 1.0.
- [ ] **Step 4: Run.** Expected: PASS; then run `.venv/bin/python -m pytest tests/unit -x -q` to confirm nothing regressed.
- [ ] **Step 5: Commit** `feat(bench): ccr bench CLI`.

### Task 10: LoCoMo adapter

**Files:**
- Create: `ccr/bench/datasets/locomo.py`
- Test: `tests/unit/bench/test_locomo.py` (inline fixture JSON)

**Interfaces:**
- Produces: `load_locomo(cfg: BenchConfig) -> list[World]`, registered as `"locomo"`. Source `https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json`. One world per conversation; one `Doc` per turn with `id = dia_id` (e.g. `D1:3`) and text `"{speaker}: {text}"`; one `Query` per qa item with `gold_ids` from `evidence`, `answer` as a string, `category` as `cat1`..`cat5`.

- [ ] **Step 1: Write failing tests** from a fixture: an evidence entry packing two ids (`"D1:3; D2:4"`) yields both; an evidence id not in the conversation is dropped; category 5 (no `answer`, has `adversarial_answer`) gets empty `gold_ids` and `answer=None` and is reported as excluded; integer `answer` values become strings.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** and record the number of dropped ids on the world (`group` field carries the conversation id).
- [ ] **Step 4: Run.** Expected: PASS. Then verify against the real file once: `ccr bench run locomo --pilot --no-reader --systems bm25 --output <scratch>` and confirm the report lists 1 world.
- [ ] **Step 5: Commit** `feat(bench): LoCoMo adapter`.

### Task 11: LongMemEval adapter (retrieval protocol only)

**Files:**
- Create: `ccr/bench/datasets/longmemeval.py`
- Test: `tests/unit/bench/test_longmemeval.py`

**Interfaces:**
- Produces: `load_longmemeval(cfg: BenchConfig) -> list[World]`, registered as `"longmemeval"`. Source file `longmemeval_s_cleaned.json` from the Hugging Face dataset `xiaowu0162/longmemeval-cleaned` (`/resolve/main/` URL). One world per question instance; one `Doc` per haystack session (`id` = session id, text = `"role: content"` lines); `gold_ids = answer_session_ids`; `category = question_type`; abstention questions (`question_id` ends `_abs`) have empty gold and are excluded and counted. QA accuracy is not computed (judge-based).

- [ ] **Step 1: Write failing tests** from a two-question fixture: session docs and gold ids correct; abstention excluded; answer stored for optional reader runs.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Stream-parse is not required; load the JSON once and cache the parsed worlds to avoid reparsing.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(bench): LongMemEval retrieval adapter`.

### Task 12: BEIR adapter (index_search)

**Files:**
- Create: `ccr/bench/datasets/beir.py`
- Test: `tests/unit/bench/test_beir.py`

**Interfaces:**
- Produces: `load_beir(name: str, cfg: BenchConfig) -> list[World]`, registered as `beir-scifact` and `beir-nfcorpus`. Sources: Hugging Face `mteb/<name>` files `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`. One world holding the full corpus (`Doc` text = `title + "\n" + text`) and every test query that has qrels. Graded relevance is kept for nDCG: `Query` gets gains through `category = "beir"` and a module-level `GAINS: dict[str, dict[str, int]]` exposed as `world_gains(world_id) -> dict[str, dict[str, int]]`; the runner uses it when present.
- Sampling for BEIR: sample queries, never corpus (pilot keeps the full corpus). Provide `sample_queries(world: World, fraction: float, seed: int) -> World` here.

- [ ] **Step 1: Write failing tests** with 3-doc fixture files: qrels header row skipped; score-0 qrels not counted as gold; queries without qrels dropped; `sample_queries` keeps every doc.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** and wire `world_gains` into the runner's nDCG call (small edit in `runner.py`, covered by one new runner test).
- [ ] **Step 4: Run.** Expected: PASS. Then run once on real SciFact with `ccr_index_bm25` and `bm25`; both should land in a plausible nDCG@10 range for BM25 (sanity check against BEIR's published BM25 for SciFact, read from the BEIR paper table, not from memory).
- [ ] **Step 5: Commit** `feat(bench): BEIR adapter`.

### Task 13: MemoryAgentBench adapter (judge-free sub-tasks only)

**Files:**
- Create: `ccr/bench/datasets/mab.py`
- Test: `tests/unit/bench/test_mab.py`

**Interfaces:**
- Produces: `load_mab(cfg: BenchConfig) -> list[World]`, registered as `"memoryagentbench"`. Source: Hugging Face `ai-hyz/MemoryAgentBench` parquet files under `data/` (read with `pyarrow`; skip with a clear message if `pyarrow` is missing). One world per example; docs are fixed-size chunks of `context` (`cfg.chunk_tokens`); each question in `questions[i]` with `answers[i]` becomes a `Query` with no gold ids (these examples have no evidence labels), so only answer metrics apply (`sub_exact_match`, `token_f1`) and `retrieved_tokens`. `JUDGE_FREE_SOURCES: frozenset[str]` lists the `metadata.source` values whose official metric is exact match, substring match or ROUGE; examples from any other source are dropped and counted.

- [ ] **Step 1: Read** the dataset README metric table and fill `JUDGE_FREE_SOURCES` from it; add the README line used as a comment per source.
- [ ] **Step 2: Write failing tests** from a tiny parquet written with `pyarrow` in the test: chunking splits at `chunk_tokens`; a judge-based source is dropped and counted; examples with several questions yield several queries.
- [ ] **Step 3: Run.** Expected: FAIL.
- [ ] **Step 4: Implement**, then **Run** tests. Expected: PASS. Contexts over the reader limit make the full-context baseline report `not_run` (runner behavior from Task 7).
- [ ] **Step 5: Commit** `feat(bench): MemoryAgentBench judge-free adapter`.

### Task 14: Code-retrieval adapter

**Files:**
- Create: `ccr/bench/datasets/code.py`
- Test: `tests/unit/bench/test_code.py`

**Interfaces:**
- Produces: `load_code(cfg: BenchConfig) -> list[World]`, registered as `"code"`. One world: a corpus of code snippets as `Doc`s and natural-language queries with one gold snippet each. Metrics: nDCG@10, MRR, recall@10.

- [ ] **Step 1: Resolve the source.** The CoIR CodeSearchNet repo on Hugging Face did not return a usable file listing during planning. Try `CoIR-Retrieval/codesearchnet` (and its python/java splits) first; if unreachable or gated, use the official `code-search-net/code_search_net` data. Record the chosen source, its license, and the subset (one language, capped corpus size from `cfg`) in the module docstring. If neither works, stop and report to the user instead of substituting an unverified dataset.
- [ ] **Step 2: Write failing tests** from a 3-snippet fixture in the chosen format: one gold per query; docstring text never appears inside the snippet doc when the source puts it in the query.
- [ ] **Step 3: Run.** Expected: FAIL.
- [ ] **Step 4: Implement**, then **Run** tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat(bench): code retrieval adapter`.

### Task 15: 10% pilot run and documentation

**Files:**
- Create: `docs/benchmarks.md`
- Modify: `README.md` (short "Benchmarks" section linking `docs/benchmarks.md`, no numbers yet)

- [ ] **Step 1: Run the pilot with retrieval only** for each dataset, all systems that apply, `--pilot --no-reader`, outputs under `bench_out/pilot/<dataset>/` (gitignored). Expected: each ends with a report whose title says "pilot (10%)" and whose `n_excluded` and `empty_rate` are present.
- [ ] **Step 2: Measure reader speed** with `--reader` on LoCoMo pilot only; record tokens/s and projected full-run time in `docs/benchmarks.md`.
- [ ] **Step 3: Review the pilot** for silent failures: any system with `empty_rate` above 5% is investigated before any full run.
- [ ] **Step 4: Write `docs/benchmarks.md`** describing datasets, systems, metrics, how to reproduce, the pilot-versus-full rule, and the known limits from the spec.
- [ ] **Step 5: Run** `.venv/bin/python -m pytest tests/unit -x -q`. Expected: all pass.
- [ ] **Step 6: Commit** `docs(bench): benchmark guide and pilot procedure`. Full runs and README numbers happen only after the user reviews the pilot.
