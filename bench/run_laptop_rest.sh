#!/usr/bin/env bash
# Resumes bench/run_laptop.sh after the SQ8 step (IVF-PQ onward), then re-measures the
# headline HNSW comparison (vdb, hnswlib, FAISS; M=16, efC=200, 3 runs) on a quiet machine.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${BUILD:-build/release}
PY=${PYTHON:-python3}
DATA=${DATA_DIR:-data}
OUT=${OUT:-bench/results}
B="$BUILD/vdb_bench"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "sift: IVF-PQ sweep (64 training points per centroid, as for the FAISS baseline)"
"$B" index --data-dir "$DATA" --dataset sift --index ivf_pq --nlist 4096 --pq-m 8,16,32 --max-train 64 \
     --nprobe 1,2,4,8,16,32,64,128,256 --rerank 0,100,1000 --max-query 2000 --no-multi --out "$OUT/index_ivfpq_sift.json"
log "sift: FAISS flat and IVFPQ"
"$PY" bench/baselines.py --data-dir "$DATA" --dataset sift --systems faiss_flat,faiss_ivfpq --nlist 4096 --pq-m 16 \
     --rerank 0,100,1000 --max-query 2000 --threads 1 --out "$OUT/baselines_ivfpq_sift.json"
log "sift-200k: build time vs threads"
for t in 1 2 4 8; do
  "$B" index --data-dir "$DATA" --dataset sift --max-base 200000 --max-query 1000 --index hnsw --M 16 --efc 200 \
       --ef 64 --build-threads "$t" --no-multi --out "$OUT/build_threads_$t.json"
done

log "glove-100: exact search"
"$B" index --data-dir "$DATA" --dataset glove-100 --index flat --k 100 --max-query 1000 --out "$OUT/index_flat_glove-100.json"
log "glove-100: vdb HNSW M=16 efC=200 (3 runs)"
"$B" index --data-dir "$DATA" --dataset glove-100 --index hnsw --M 16 --efc 200 --ef 10,20,40,80,160,320,640 --runs 3 \
     --out "$OUT/index_hnsw_m16_glove-100.json"
log "glove-100: hnswlib and FAISS (3 runs)"
"$PY" bench/baselines.py --data-dir "$DATA" --dataset glove-100 --systems hnswlib,faiss_hnsw --M 16 --efc 200 \
     --runs 3 --threads 1 --out "$OUT/baselines_hnsw_m16_glove-100.json"

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

log "sift: headline HNSW comparison again, quiet machine (3 runs each)"
"$B" index --data-dir "$DATA" --dataset sift --index hnsw --M 16 --efc 200 --ef 10,20,40,80,160,320,640 --runs 3 \
     --out "$OUT/index_hnsw_m16_sift.json"
"$PY" bench/baselines.py --data-dir "$DATA" --dataset sift --systems hnswlib,faiss_hnsw --M 16 --efc 200 \
     --runs 3 --threads 1 --out "$OUT/baselines_hnsw_m16_sift.json"
log "done"
