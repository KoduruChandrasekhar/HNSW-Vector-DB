#!/usr/bin/env bash
# The benchmark set used for the README on an 8-thread laptop with ~8 GB RAM: the full
# SIFT1M and GloVe-100 index comparisons (median of 3 runs for the headline HNSW
# configuration), and the engine-level experiments on a 200k subset of SIFT.
# bench/run_all.sh is the exhaustive version (every sweep, three runs each).
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${BUILD:-build/release}
PY=${PYTHON:-python3}
DATA=${DATA_DIR:-data}
OUT=${OUT:-bench/results}
B="$BUILD/vdb_bench"
mkdir -p "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*"; }

for ds in sift glove-100; do
  [ -f "$DATA/$ds/base.fbin" ] || { log "skip $ds (missing)"; continue; }
  PQM=16; [ "$ds" = glove-100 ] && PQM=20
  log "$ds: exact search"
  "$B" index --data-dir "$DATA" --dataset "$ds" --index flat --k 100 --max-query 1000 --out "$OUT/index_flat_$ds.json"
  log "$ds: vdb HNSW M=16 efC=200 (3 runs)"
  "$B" index --data-dir "$DATA" --dataset "$ds" --index hnsw --M 16 --efc 200 --ef 10,20,40,80,160,320,640 --runs 3 \
       --out "$OUT/index_hnsw_m16_$ds.json"
  log "$ds: hnswlib and FAISS HNSW M=16 efC=200 (3 runs)"
  "$PY" bench/baselines.py --data-dir "$DATA" --dataset "$ds" --systems hnswlib,faiss_hnsw --M 16 --efc 200 \
       --runs 3 --threads 1 --out "$OUT/baselines_hnsw_m16_$ds.json"
  if [ "$ds" = sift ]; then
    log "sift: M sweep"
    "$B" index --data-dir "$DATA" --dataset sift --index hnsw --M 8,32 --efc 200 --ef 10,20,40,80,160,320,640 \
         --no-multi --out "$OUT/index_hnsw_msweep_sift.json"
    "$B" index --data-dir "$DATA" --dataset sift --index hnsw --M 16 --efc 100,400 --ef 10,20,40,80,160,320,640 \
         --no-multi --out "$OUT/index_hnsw_efcsweep_sift.json"
    "$PY" bench/baselines.py --data-dir "$DATA" --dataset sift --systems hnswlib --M 8,32 --efc 200 \
         --threads 1 --out "$OUT/baselines_hnsw_msweep_sift.json"
    log "sift: SQ8, IVF-PQ, FAISS flat and IVFPQ"
    "$B" index --data-dir "$DATA" --dataset sift --index hnsw_sq8 --M 16 --efc 200 --ef 10,20,40,80,160,320 --no-multi \
         --out "$OUT/index_sq8_sift.json"
    "$B" index --data-dir "$DATA" --dataset sift --index ivf_pq --nlist 4096 --pq-m 8,16,32 --max-train 64 \
         --nprobe 1,2,4,8,16,32,64,128,256 --rerank 0,100,1000 --max-query 2000 --no-multi \
         --out "$OUT/index_ivfpq_sift.json"
    "$PY" bench/baselines.py --data-dir "$DATA" --dataset sift --systems faiss_flat,faiss_ivfpq --nlist 4096 --pq-m 16 \
         --max-query 1000 --threads 1 --out "$OUT/baselines_ivfpq_sift.json"
    log "sift-200k: build time vs threads"
    for t in 1 2 4 8; do
      "$B" index --data-dir "$DATA" --dataset sift --max-base 200000 --max-query 1000 --index hnsw --M 16 --efc 200 \
           --ef 64 --build-threads "$t" --no-multi --out "$OUT/build_threads_$t.json"
    done
  fi
done

SUB=(--data-dir "$DATA" --dataset sift --max-base 200000 --max-query 1000)
log "filters (uncorrelated, correlated)"
"$B" filter "${SUB[@]}" --out "$OUT/filter_uncorrelated.json"
"$B" filter "${SUB[@]}" --correlated --out "$OUT/filter_correlated.json"
log "segments"
"$B" segments "${SUB[@]}" --out "$OUT/segments.json"
log "deletes"
"$B" deletes "${SUB[@]}" --out "$OUT/deletes.json"
log "ingest"
"$B" ingest --dim 128 --n 20000 --writers 1,8,64 --batch 1,100 --out "$OUT/ingest.json"
log "recovery"
"$B" recovery --dim 128 --log-mb 10,100,500 --checksum-rows 200000 --out "$OUT/recovery.json"
log "done"
