#!/usr/bin/env bash
# Full (100%) run of every benchmark dataset (see docs/benchmarks.md).
#   Phase A: retrieval-only, no LLM, about an hour.
#   Phase B: local Qwen reader on all of LoCoMo, hours.
#   Phase C ('mab'): MemoryAgentBench with the reader, many hours.
# Usage: scripts/bench_full.sh [a|b|mab|all]   (run it inside tmux, under caffeinate)
# Needs the bench venv: uv venv .venv-bench && uv pip install -e ".[semantic,bench]"
# Reruns resume: finished queries are skipped, reader timeouts are retried.
set -u
PY="${PY:-.venv-bench/bin/python}"
OUT="${OUT:-bench_out/full}"
PHASE="${1:-all}"
FULLCTX_NUM_CTX="${FULLCTX_NUM_CTX:-49152}"
# System lists can be overridden per run (the v2 run compares the new ccr_memory with legacy).
MEM_SYSTEMS="${MEM_SYSTEMS:-bm25,dense_minilm,ccr_memory,ccr_index_bm25}"
FILE_SYSTEMS="${FILE_SYSTEMS:-bm25,dense_minilm,ccr_index_bm25,ccr_index_hybrid,ccr_memory}"
READER_SYSTEMS="${READER_SYSTEMS:-bm25,dense_minilm,ccr_memory,ccr_index_bm25}"
RUN_FULLCTX="${RUN_FULLCTX:-1}"

run() { echo "=== $(date '+%F %T') ccr bench run $*"; "$PY" -m ccr.cli bench run "$@" --full; }

phase_a() {
  run locomo --systems "$MEM_SYSTEMS" --no-reader -o "$OUT/locomo"
  run longmemeval --systems "$MEM_SYSTEMS" --no-reader -o "$OUT/longmemeval"
  run beir-scifact --systems "$FILE_SYSTEMS" --no-reader -o "$OUT/beir-scifact"
  run beir-nfcorpus --systems "$FILE_SYSTEMS" --no-reader -o "$OUT/beir-nfcorpus"
  run code --systems "$FILE_SYSTEMS" --no-reader -o "$OUT/code"
}
phase_b() {
  run locomo --systems "$READER_SYSTEMS" --reader -o "$OUT/locomo-reader"
  # A whole LoCoMo conversation is 19k-36.5k Qwen tokens (measured, about 1.1x tiktoken),
  # so full_context needs a window above the 16384 default.
  [ "$RUN_FULLCTX" = 1 ] || return 0
  run locomo --systems full_context --reader --num-ctx "$FULLCTX_NUM_CTX" -o "$OUT/locomo-reader-fullctx"
}
phase_mab() {
  run memoryagentbench --systems "$READER_SYSTEMS" --reader -o "$OUT/memoryagentbench-reader"
}

mkdir -p "$OUT"
case "$PHASE" in
  a) phase_a ;;
  b) phase_b ;;
  mab) phase_mab ;;
  all) phase_a; phase_b; phase_mab ;;
  *) echo "usage: $0 [a|b|mab|all]"; exit 2 ;;
esac
echo "=== $(date '+%F %T') done: $PHASE"
