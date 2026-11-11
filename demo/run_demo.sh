#!/usr/bin/env bash
# Record the demo live and render it to docs/demo/vdb-demo.mp4 and .gif (about 5 minutes).
#   BUILD=build/release PYTHON=python3 RAG_WORK=rag/work bash demo/run_demo.sh
# Needs: a release build, `pip install .`, the RAG work dir from rag/ingest.py
# (embeddings.npy + passages.sqlite), sentence-transformers, Pillow and ffmpeg.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${BUILD:-build/release}
PYTHON=${PYTHON:-python3}
"$PYTHON" demo/record.py --server "$BUILD/vdb-server" --build "$BUILD" --python "$PYTHON" --out demo/out/cast.json
"$PYTHON" demo/render.py --cast demo/out/cast.json --out demo/out/vdb-demo
mkdir -p docs/demo
cp demo/out/vdb-demo.mp4 demo/out/vdb-demo.gif docs/demo/   # the copies the README shows
