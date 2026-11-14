#!/usr/bin/env bash
# 10% pilot for every benchmark dataset (see docs/benchmarks.md).
#   Phase A: retrieval-only, no LLM, minutes.
#   Phase B: local Qwen reader via Ollama on LoCoMo, hours.
#   Phase C (opt-in, 'mab'): MemoryAgentBench with the reader, many hours.
# Usage: scripts/bench_pilot.sh [a|b|mab|all]   (run it inside tmux)
# Needs the bench venv: uv venv .venv-bench && uv pip install -e ".[semantic,bench]"
set -u
PY="${PY:-.venv-bench/bin/python}"
OUT="${OUT:-bench_out/pilot}"
PHASE="${1:-all}"
MEM_SYSTEMS="bm25,dense_minilm,ccr_memory,ccr_index_bm25"
FILE_SYSTEMS="bm25,dense_minilm,ccr_index_bm25,ccr_index_hybrid,ccr_memory"
READER_SYSTEMS="bm25,dense_minilm,ccr_memory"

run() { echo "=== $(date '+%F %T') ccr bench run $*"; "$PY" -m ccr.cli bench run "$@" --pilot; }

phase_a() {
  run locomo --systems "$MEM_SYSTEMS" --no-reader -o "$OUT/locomo"
  run longmemeval --systems "$MEM_SYSTEMS" --no-reader -o "$OUT/longmemeval"
  run beir-scifact --systems "$FILE_SYSTEMS" --no-reader -o "$OUT/beir-scifact"
  run beir-nfcorpus --systems "$FILE_SYSTEMS" --no-reader -o "$OUT/beir-nfcorpus"
  run code --systems "$FILE_SYSTEMS" --no-reader -o "$OUT/code"
}
phase_b() {
  run locomo --systems "$READER_SYSTEMS" --reader -o "$OUT/locomo-reader"
  # A whole LoCoMo conversation is ~28k tokens, so full_context needs a bigger window.
  run locomo --systems full_context --reader --num-ctx 32768 -o "$OUT/locomo-reader-fullctx"
}
phase_mab() {
  run memoryagentbench --systems "bm25,ccr_memory,ccr_index_bm25,dense_minilm,full_context" \
    --reader -o "$OUT/memoryagentbench-reader"
}

mkdir -p "$OUT"
case "$PHASE" in
  a) phase_a ;;
  b) phase_b ;;
  mab) phase_mab ;;
  all) phase_a; phase_b ;;
  *) echo "usage: $0 [a|b|mab|all]"; exit 2 ;;
esac
echo "=== $(date '+%F %T') done: $PHASE"
