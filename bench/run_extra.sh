#!/usr/bin/env bash
# The remaining experiments: kernel microbenchmarks, mutable-segment size, graph
# reordering, Python vs C++ QPS, and the server-level runs.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${BUILD:-build/release}
PY=${PYTHON:-python3}
DATA=${DATA_DIR:-data}
OUT=${OUT:-bench/results}
B="$BUILD/vdb_bench"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "kernel microbenchmarks"
"$BUILD/vdb_micro_bench" --benchmark_out="$OUT/micro.json" --benchmark_out_format=json --benchmark_min_time=0.2s \
  > "$OUT/micro.txt"
log "mutable segment size"
"$B" mutable --data-dir "$DATA" --dataset sift --max-base 300000 --max-query 1000 --out "$OUT/mutable.json"
log "graph reordering"
"$B" reorder --data-dir "$DATA" --dataset sift --max-query 10000 --runs 3 --out "$OUT/reorder.json"
log "segments for the python comparison"
"$B" segments --data-dir "$DATA" --dataset sift --max-base 200000 --max-query 1000 --segments 1 \
  --ef 16,32,64,128,256 --out "$OUT/segments_one.json"
log "python batched search"
"$PY" bench/python_qps.py --data-dir "$DATA" --dataset sift --max-base 200000 --max-query 1000 --out "$OUT/python_qps.json"
log "server"
BUILD="$BUILD" PYTHON="$PY" DATA_DIR="$DATA" OUT="$OUT" bash bench/server_bench.sh
log "done"
