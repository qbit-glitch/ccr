# Benchmarks

`ccr bench` measures CCR's memory recall (`gcc_recall`) and repo search
(`RepoIndex`, behind `index_search`) on public datasets, with no LLM judge and
no paid API. Every number is a deterministic metric, so you can rerun it.

## What is measured

| Dataset | Source (public) | CCR part | Metrics |
|---|---|---|---|
| `locomo` | github.com/snap-research/locomo (`locomo10.json`) | memory | evidence recall@k, nDCG, MRR, tokens; with a reader: F1, EM, BLEU-1 |
| `longmemeval` | Hugging Face `xiaowu0162/longmemeval-cleaned` (S) | memory | session-level recall@k, nDCG, MRR, tokens |
| `memoryagentbench` | Hugging Face `ai-hyz/MemoryAgentBench` | memory | exact match, substring match, F1, tokens (reader required) |
| `beir-scifact`, `beir-nfcorpus` | Hugging Face `mteb/scifact`, `mteb/nfcorpus` | `index_search` | nDCG@k, recall@k, MRR |
| `code` | Hugging Face `mteb/CodeSearchNetRetrieval` (python) | `index_search` | nDCG@k, recall@k, MRR |

Not run, because they need an LLM judge: LoCoMo J-score, LongMemEval QA
accuracy, MemoryAgentBench's LongMemEval, summarization and recsys sources.

## Systems

All systems see the same documents, the same top-k, and (with a reader) the same
retrieved text.

- `ccr_memory`: every document becomes one commit titled with its id (commit
  merging disabled) and is recalled with `RecallEngine` using commit evidence only.
  Limited to k <= 25 by the recall engine.
- `ccr_index_bm25`, `ccr_index_hybrid`, `ccr_index_keyword`: `RepoIndex` over one
  `.txt` file per document. Hybrid needs the ONNX embeddings and is skipped
  without them. Keyword mode matches the query as one substring, which suits
  symbol names, not natural-language questions, so pilots leave it out.
- `bm25`: independent Okapi BM25 (k1=1.5, b=0.75), not CCR code.
- `dense_minilm`: all-MiniLM-L6-v2 with cosine ranking. Its ONNX model reads
  only the first 256 tokens of each document, which hurts long documents.
- `full_context`: gives the reader everything (an upper bound on accuracy and
  the worst case for tokens). When the text does not fit the reader's context
  window it is reported `not_run`, never truncated.

## Reproduce

```bash
uv venv .venv-bench && uv pip install -p .venv-bench -e ".[semantic,bench]"
.venv-bench/bin/python -m ccr.cli bench list
.venv-bench/bin/python -m ccr.cli bench run locomo --pilot --no-reader
.venv-bench/bin/python -m ccr.cli bench report bench_out/locomo
```

Use a separate environment: installing `onnxruntime` into the main dev venv
turns on CCR's semantic paths and changes some existing unit tests. Each run
writes `results.jsonl` (resumable), `report.json`, `report.md`, `meta.json`
and `sample.json` into `bench_out/<dataset>/`. The meta records the git sha,
seed, reader tag, whether ONNX embeddings were active, and any `k` values
dropped for a system's cap.

### Reader runs

The reader is a local Ollama model, `qwen3.8:27b` by default (`--model`),
called at temperature 0 with thinking off and a 16384-token context
(`--num-ctx`). It answers only; it never scores. Long runs belong in tmux:

```bash
tmux new -d -s ccr-bench 'bash scripts/bench_pilot.sh all 2>&1 | tee bench_out/pilot/run.log'
```

`scripts/bench_pilot.sh` runs the retrieval pilots (phase `a`), the LoCoMo
reader pilot (`b`), and, only when asked, the MemoryAgentBench reader pilot
(`mab`).

## Pilot versus full

`--pilot` (the default) samples 10% of whole worlds with a fixed seed: one
LoCoMo conversation, question instances stratified by type for LongMemEval,
task families for MemoryAgentBench. BEIR and code samples queries but keeps the
whole corpus, because a smaller corpus would make retrieval look easier. A pilot
is contained in the full run for the same seed. Pilot reports are titled
"pilot (10%)" and are not used for README claims; `--full` runs everything.

## Things that change the numbers

- LoCoMo: each turn is prefixed with its session date and image captions are
  appended; adversarial (category 5) questions are excluded; evidence strings
  that pack several ids are split and ids missing from the conversation are dropped.
- LongMemEval: sessions carry their date; abstention questions are excluded.
- Code: queries are docstrings and the gold functions still contain theirs, as
  in MTEB's published set, so lexical overlap is high by design.
- MemoryAgentBench has no evidence labels. Chunks are about 512 tokens (4
  characters per token), and only substring/exact-match sources are loaded.
- Reader answers are short phrases; scores depend on the reader, so compare
  systems against each other, not against papers that used other readers.
- Published numbers go in `docs/bench/published.yaml` and appear in a separate
  table. Each needs its paper and table locator, copied from the authors.
