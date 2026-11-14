# CCR Benchmark Suite: Design

Status: approved by the user on 2026-11-14, who asked for all five datasets
below (LoCoMo, LongMemEval retrieval, MemoryAgentBench judge-free sub-tasks,
a BEIR subset, a code-retrieval subset). Delivery is phased, 10% pilot first.

## Intent and why

CCR is about to be distributed to users. Users need numbers that compare CCR
with published retrievers and memory systems on public benchmarks. Every
number must be deterministic and reproducible by users on their own machines
at no cost, so no LLM judge and no paid API anywhere.

## Goals

- A free, reproducible harness (`ccr bench <dataset>`; name is the builder's
  call) that downloads a public dataset, ingests it into an isolated CCR
  project, retrieves, scores, and writes a report.
- A README results table of CCR against baselines we run under identical
  conditions, plus published numbers in a separate, labeled column.
- Metrics that need no judge: evidence recall@k, nDCG@10, MRR, tokens returned
  per question, and, where a dataset defines it, token-F1, exact match, or
  BLEU-1 on a local reader's answers.

## Non-goals

- Any LLM-as-judge score (no J-score, no LongMemEval QA accuracy, no judge
  of any kind, Claude included). Those published numbers cannot be
  reproduced and are not compared against.
- Paid API keys, closed-source models, or private data.
- Agent-level SWE-bench runs.
- Changing CCR's retrieval behaviour. This project measures; it does not tune.

## Anti-goals

- Putting a judge-based published number beside a deterministic one of ours.
- A harness whose results users cannot rerun.

## Constraints

- Public datasets only, downloaded from their official public sources.
- CCR stays zero-LLM-call. An LLM appears only as the optional local reader.
- Reader, where used: `qwen3.8:27b` (open-source Qwen, 27B parameters) served
  by Ollama from the local SSD store (`OLLAMA_MODELS=$HOME/.ollama/models`).
  The exact tag and quantization are recorded in every report. It answers
  questions only. It never scores anything.
- Dev machine: M4 Pro 48GB, GPU often busy. Runs must be resumable,
  cacheable and subsample-able.
- Repo rules: files of 200-400 lines, type hints, frozen config dataclasses,
  logger not print, no hardcoded hyperparameters.

## Approaches considered

1. Retrieval-only deterministic metrics. Free, exact, comparable to published
   retriever tables. Chosen as the core.
2. Replicate the papers' LLM-judge protocol. Needs a paid or judge model.
   Rejected by the user twice (paid keys, then any LLM judge).
3. Agent-level SWE-bench. Deferred.
4. Chosen: approach 1, plus an optional local-reader answer layer scored by
   F1 / EM / BLEU-1 only on datasets whose official metric is judge-free.

## Design

Datasets, all public, in this order:

| Dataset | Source | CCR part | Judge-free metrics |
|---|---|---|---|
| LoCoMo | public GitHub release | memory recall | evidence recall@k; answer F1 and BLEU-1 (the original LoCoMo metrics) |
| LongMemEval | public Hugging Face release | memory recall | session- and turn-level recall@k, nDCG@10 (its retrieval protocol). QA accuracy is judge-based, so it is not run. |
| MemoryAgentBench | public Hugging Face release | memory recall | only the sub-tasks with exact-match or ROUGE metrics |
| BEIR subset (e.g. SciFact, NFCorpus) | public BEIR | `index_search` | nDCG@10, recall@100 |
| Code retrieval (CodeSearchNet or CoIR subset) | public | `index_search` | nDCG@10, MRR |

Pipeline per dataset:

1. Adapter converts the dataset into `(documents/sessions, queries, gold ids)`.
2. Ingest (my call, correctable): each conversation turn becomes one commit
   in a temp-isolated CCR project. Evidence ids map turn id to commit id. No
   user memory is touched.
3. Retrieve with `RecallEngine` (memory) or `index_search` (files), at the
   same top-k as every baseline.
4. Optional reader: Qwen answers from the retrieved context; score with F1 /
   EM / BLEU-1 against gold.
5. Report JSON plus markdown, reusing the `EvalReport` shape in
   `ccr/core/memory_eval.py`, which already scores evidence-id recall.

Systems compared, same top-k, same reader where a reader is used:

- CCR (keyword, BM25, ONNX, hybrid modes, reported separately)
- BM25 baseline
- Dense MiniLM baseline (the ONNX model CCR already ships)
- Full-context (accuracy ceiling, token worst case) for the reader layer

Published comparisons (Table 2, clearly labeled): only numbers from the
dataset authors' own judge-free tables, for example the retriever baselines
in the LongMemEval paper's retrieval results and BEIR's published BM25
nDCG@10. Every number is copied from the paper's table, never from memory,
with a citation. If a published number uses a different evaluation setup, it
is marked as such or left out.

Reporting rule: two tables. Table 1 is what we ran under identical
conditions. Table 2 is published context.

## Pilot run (first milestone)

Before any full run, the harness runs a 10% pilot to validate the pipeline,
measure real speed, and size the full run.

- Sample at the unit of a dataset's natural grouping, not at random across the
  whole corpus. For LoCoMo that is whole conversations (it has 10, so the
  pilot is 1 conversation, stratified so every question category appears).
  Sampling questions from the full corpus would be wrong for retrieval, since
  CCR would then search a corpus with its distractors removed, which makes
  recall look better than it is. For LongMemEval, sample whole question
  instances with their haystacks. For BEIR and code retrieval, sample queries
  but keep the full corpus.
- Retrieval metrics (recall@k, nDCG, MRR, tokens) are cheap and need no LLM,
  so they run on 100% of the data even in the pilot where possible.
- Only the Qwen reader step is limited to the 10% sample, because Qwen 3.8 27B
  generated at about 3 tokens/s in a first test.
- Fixed random seed, sample ids saved in the report, so the pilot is
  repeatable and the full run is a superset.
- Pilot numbers are labeled "pilot (10%)" and are never used in README claims.

## Decided by the user

- No paid API keys, no LLM judge (including Claude), public datasets only.
- Reader is Qwen 3.8 27B (`qwen3.8:27b`), local and open-source.
- Goal is user-facing numbers against papers and similar memory kits.

## My calls (please check)

- Turn-per-commit ingestion adapter.
- LoCoMo first, then LongMemEval retrieval, then BEIR/code retrieval for
  `index_search`.
- Drop LongMemEval QA accuracy and LoCoMo J-score entirely, since both need
  a judge. This means no direct comparison with Mem0/Zep headline numbers,
  which are J-scores. The comparison with them is limited to retrieval and
  to F1/BLEU-1 if their papers report those.
- Skip the subagent builder check; I re-read this doc as the builder instead.

## Left to the builder

- CLI shape and file layout under `ccr/bench/`.
- Caching format and resumability; top-k values; quick-run subsample sizes.
- Which exact BEIR and code-retrieval subsets; which published tables to cite.

## Known risks

- The Ollama model store was moved from the unmounted `/Volumes/code_files_2`
  to `$HOME/.ollama/models` on the SSD. The disk is about 93% full, so a
  27B download (roughly 17 GB at the default quantization) needs checking
  against free space.
- Without a judge, CCR cannot be compared with Mem0 or Zep on their headline
  metric. The story is "retrieval quality and token cost under identical
  conditions", which is narrower but defensible.
- Local reader quality varies by model; only relative results across systems
  are meaningful.
- Dataset licenses and download URLs must be checked per dataset before the
  adapter ships.
