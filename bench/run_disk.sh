#!/usr/bin/env bash
# Durability-cost and recovery experiments on a real disk. Point WORK at a directory on
# the filesystem you want to measure (not tmpfs: check `df -T`); fsync cost is the point.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${BUILD:-build/release}
OUT=${OUT:-bench/results}
WORK=${WORK:-$HOME/vdb_bench_disk}
B="$BUILD/vdb_bench"
mkdir -p "$WORK"
echo "work dir: $WORK ($(df -T "$WORK" | awk 'NR==2 {print $2}'))"
"$B" ingest --work-dir "$WORK" --dim 128 --n "${INGEST_N:-4000}" --writers 1,8,64 --batch 1,100 --out "$OUT/ingest.json"
"$B" recovery --work-dir "$WORK" --dim 128 --log-mb "${LOG_MB:-10,100,500}" --checksum-rows 200000 --out "$OUT/recovery.json"
rm -rf "$WORK"
