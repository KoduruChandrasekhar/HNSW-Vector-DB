#!/usr/bin/env bash
# Downloads the ANN-Benchmarks datasets and converts them to data/<name>/{base,query}.fbin, gt.ibin.
#   bench/prepare_data.sh                 # fashion-mnist, sift, glove-100 (~1.2 GB download)
#   bench/prepare_data.sh sift            # just one
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
BUILD=${BUILD:-build/release}
DATASETS=("$@")
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(fashion-mnist sift glove-100)
for d in "${DATASETS[@]}"; do
  if [ -f "data/$d/base.fbin" ]; then echo "data/$d ready"; continue; fi
  "$PY" tools/convert_hdf5.py --download "$d" --data-dir data
done
# The cached 100k subset used by the recall-regression test.
if [ -f data/sift/base.fbin ] && [ ! -f data/sift-100k/base.fbin ]; then
  "$BUILD/vdb_bench" subset --data-dir data --dataset sift --n 100000 --name sift-100k --max-query 1000
fi
