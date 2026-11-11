#!/usr/bin/env bash
# Runs every system and sweep (three runs each by default), writes raw JSON to
# bench/results/, then turns it into charts and the README table.
#
#   bench/run_all.sh                    # full suite (hours on a laptop)
#   QUICK=1 bench/run_all.sh            # small subsets, one run: minutes, for smoke testing
#
# Hygiene: release build, plugged in, nothing else running. Results record the CPU,
# cores, RAM, OS, compiler and git commit.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${BUILD:-build/release}
PY=${PYTHON:-python3}
OUT=bench/results
RUNS=${RUNS:-3}
THREADS=${THREADS:-0}
DATASETS=${DATASETS:-"sift glove-100"}
mkdir -p "$OUT"
EXTRA=()
if [ "${QUICK:-0}" = 1 ]; then RUNS=1; EXTRA=(--max-base 100000 --max-query 1000); fi

B="$BUILD/vdb_bench"
[ -x "$B" ] || { echo "build first: cmake --preset release && cmake --build --preset release"; exit 1; }

for ds in $DATASETS; do
  [ -f "data/$ds/base.fbin" ] || { echo "missing data/$ds (run bench/prepare_data.sh $ds)"; continue; }
  PQM=16; [ "$ds" = glove-100 ] && PQM=20
  # Library level: our indexes...
  "$B" index --dataset "$ds" --index flat --runs "$RUNS" --threads "$THREADS" --no-multi "${EXTRA[@]}" --out "$OUT/index_flat_$ds.json"
  "$B" index --dataset "$ds" --index hnsw --M 8,16,32,48 --efc 100,200,400 --ef 10,20,40,80,160,320,640 \
       --runs "$RUNS" --threads "$THREADS" "${EXTRA[@]}" --out "$OUT/index_hnsw_$ds.json"
  "$B" index --dataset "$ds" --index hnsw_sq8 --M 16 --efc 200 --ef 10,20,40,80,160,320 --rerank -1,0,100 \
       --runs "$RUNS" --threads "$THREADS" "${EXTRA[@]}" --out "$OUT/index_sq8_$ds.json"
  "$B" index --dataset "$ds" --index ivf_pq --nlist 1024,4096 --pq-m "8,$PQM,32" --max-train 64 --nprobe 1,2,4,8,16,32,64,128,256 \
       --rerank 0,100,1000 --runs "$RUNS" --threads "$THREADS" --no-multi "${EXTRA[@]}" --out "$OUT/index_ivfpq_$ds.json"
  # ...and the baselines with the same parameters and thread count.
  "$PY" bench/baselines.py --dataset "$ds" --systems hnswlib,faiss_hnsw --M 8,16,32,48 --efc 200 \
       --runs "$RUNS" --threads 1 "${EXTRA[@]}" --out "$OUT/baselines_hnsw_$ds.json"
  "$PY" bench/baselines.py --dataset "$ds" --systems faiss_flat,faiss_ivfpq --nlist 4096 --pq-m "$PQM" \
       --runs "$RUNS" --threads 1 "${EXTRA[@]}" --out "$OUT/baselines_ivfpq_$ds.json"
done

# Build time vs threads (HNSW, M=16, efC=200).
for t in 1 2 4 8; do
  "$B" index --dataset sift --index hnsw --M 16 --efc 200 --ef 64 --build-threads "$t" --no-multi \
       "${EXTRA[@]}" --out "$OUT/build_threads_$t.json"
done

# Engine-level experiments.
"$B" filter --dataset sift "${EXTRA[@]}" --out "$OUT/filter_uncorrelated.json"
"$B" filter --dataset sift --correlated "${EXTRA[@]}" --out "$OUT/filter_correlated.json"
"$B" segments --dataset sift "${EXTRA[@]}" --out "$OUT/segments.json"
"$B" deletes --dataset sift --max-base "${DELETE_BASE:-200000}" --max-query 1000 --out "$OUT/deletes.json"
"$B" ingest --dim 128 --n "${INGEST_N:-20000}" --out "$OUT/ingest.json"
"$B" recovery --dim 128 --log-mb "${LOG_MB:-10,100,1000}" --out "$OUT/recovery.json"

# Server level: offered load against one bulk-loaded collection.
if [ "${SKIP_SERVER:-0}" != 1 ]; then
  rm -rf /tmp/vdb_server_bench && "$BUILD/vdb-server" --data-dir /tmp/vdb_server_bench --port 8091 --threads 16 --log-level warn &
  SERVER=$!
  trap 'kill $SERVER 2>/dev/null || true' EXIT
  sleep 1
  "$PY" - <<'EOF'
import sys, numpy as np, struct
sys.path.insert(0, "python")
from vdb.client import Client
def fbin(p, n=0):
    with open(p, "rb") as f:
        rows, d = struct.unpack("<II", f.read(8)); rows = min(rows, n) if n else rows
        return np.fromfile(f, dtype=np.float32, count=rows * d).reshape(rows, d)
base = fbin("data/sift/base.fbin", 200000)
c = Client("http://127.0.0.1:8091", timeout=600)
col = c.create_collection("sift", dim=128, sync="none")
col.bulk_load(np.arange(len(base), dtype=np.uint64), base)
print("server collection:", col.stats()["rows"], "rows")
EOF
  "$PY" bench/server_load.py --url http://127.0.0.1:8091 --collection sift --rates "${RATES:-200,500,1000,2000,4000,8000}" \
       --seconds "${LOAD_SECONDS:-10}" --out "$OUT/server_load.json"
fi

"$PY" bench/plot.py --results "$OUT" --out "$OUT/charts"
