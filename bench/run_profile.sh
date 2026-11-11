#!/usr/bin/env bash
# perf profile of single-threaded search (HNSW, HNSW-SQ8, IVF-PQ on SIFT1M), plus the
# mutable-segment-vs-incremental-HNSW comparison.
#   BUILD=build/release DATA_DIR=data bash bench/run_profile.sh
# Writes bench/results/perf_<index>.txt (counters + hottest symbols) and mutable.json.
set -euo pipefail
BUILD=${BUILD:-build/release}
DATA=${DATA_DIR:-data}
WORK=${WORK_DIR:-$HOME/vdb_profile_work}
OUT=${OUT:-bench/results}
B="$BUILD/vdb_bench"
log() { echo "[$(date +%H:%M:%S)] $*"; }

if [ -z "${SKIP_MUTABLE:-}" ]; then
  log "mutable segment vs incremental HNSW"
  "$B" mutable --data-dir "$DATA" --dataset sift --max-base 300000 --max-query 1000 \
    --work-dir "$WORK" --out "$OUT/mutable.json"
fi

EVENTS=cycles:u,instructions:u,cache-references:u,cache-misses:u,L1-dcache-loads:u,L1-dcache-load-misses:u
for spec in "hnsw --ef 80" "hnsw_sq8 --ef 80" "ivf_pq --nprobe 64"; do
  set -- $spec
  index=$1
  shift
  log "build (once) $index"
  "$B" search-loop --data-dir "$DATA" --dataset sift --work-dir "$WORK" --index "$index" "$@" --seconds 1
  log "perf $index"
  perf stat -e "$EVENTS" -x, -o "$WORK/stat_$index.csv" \
    "$B" search-loop --data-dir "$DATA" --dataset sift --work-dir "$WORK" --index "$index" "$@" --seconds 20 \
    > "$WORK/run_$index.txt"
  perf record -F 1999 -o "$WORK/perf_$index.data" \
    "$B" search-loop --data-dir "$DATA" --dataset sift --work-dir "$WORK" --index "$index" "$@" --seconds 20 \
    > /dev/null 2>&1
  {
    echo "# vdb_bench search-loop --index $index $* (SIFT1M, 1 thread, k=10, 20 s)"
    grep "queries in" "$WORK/run_$index.txt"
    python3 - "$WORK/stat_$index.csv" "$WORK/run_$index.txt" <<'EOF'
import re, sys
c = {}
for line in open(sys.argv[1]):
    f = line.strip().split(",")
    if len(f) > 2 and f[0].replace(".", "").isdigit():
        c[f[2].split(":")[0]] = float(f[0])
q = int(re.search(r"(\d+) queries in", open(sys.argv[2]).read()).group(1))
print(f"per query: {c['cycles'] / q:,.0f} cycles, {c['instructions'] / q:,.0f} instructions "
      f"(IPC {c['instructions'] / c['cycles']:.2f}), {c['cache-misses'] / q:,.0f} cache misses "
      f"({100 * c['cache-misses'] / c['cache-references']:.0f}% of cache references), "
      f"{c['L1-dcache-load-misses'] / q:,.0f} L1d load misses "
      f"({100 * c['L1-dcache-load-misses'] / c['L1-dcache-loads']:.1f}% of loads)")
EOF
    echo "# hottest symbols (self time)"
    perf report -i "$WORK/perf_$index.data" --stdio --no-children --sort symbol 2>/dev/null \
      | grep -v -E "^#|^$" | sed -E 's/ +$//; s/\(.*\) const$/(...) const/; s/\(.*\)$/(...)/' | head -15 || true
  } > "$OUT/perf_$index.txt" 2>&1
  cat "$OUT/perf_$index.txt"
done
log "done"
