#!/usr/bin/env bash
# Server-level benchmarks:
#   1. p50/p99 latency vs offered load against one bulk-loaded collection (open loop);
#   2. 100 concurrent HTTP clients searching while another client bulk-loads a second
#      collection: no errors allowed, report p50 and p99.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${BUILD:-build/release}
PY=${PYTHON:-python3}
DATA=${DATA_DIR:-data}
OUT=${OUT:-bench/results}
PORT=${PORT:-8091}
ROWS=${ROWS:-200000}
WORK=$(mktemp -d /tmp/vdb_server_bench.XXXX)
# cpp-httplib keeps one worker thread per keep-alive connection, so the server needs more
# threads than the load generator opens connections (64, then 100 + the loader).
"$BUILD/vdb-server" --data-dir "$WORK" --port "$PORT" --threads "${SERVER_THREADS:-128}" --log-level warn \
  --max-inflight-searches 256 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; rm -rf "$WORK"' EXIT
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do sleep 0.2; done

"$PY" - "$DATA" "$PORT" "$ROWS" <<'EOF'
import sys, struct, numpy as np
sys.path.insert(0, "python")
from vdb.client import Client
data, port, rows = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(f"{data}/sift/base.fbin", "rb") as f:
    n, d = struct.unpack("<II", f.read(8))
    base = np.fromfile(f, dtype=np.float32, count=rows * d).reshape(rows, d)
c = Client(f"http://127.0.0.1:{port}", timeout=900)
col = c.create_collection("sift", dim=d, sync="none")
col.bulk_load(np.arange(rows, dtype=np.uint64), base)
print("server collection sift:", col.stats()["rows"], "rows,", col.stats()["segments"], "segments")
EOF

"$PY" bench/server_load.py --url "http://127.0.0.1:$PORT" --collection sift --data-dir "$DATA" --dataset sift \
  --rates "${RATES:-250,500,1000,2000,4000,8000}" --seconds "${LOAD_SECONDS:-10}" --connections 64 \
  --out "$OUT/server_load.json"

# 100 concurrent clients while another client bulk-loads.
"$PY" - "$DATA" "$PORT" "$ROWS" <<'EOF' &
import sys, struct, numpy as np, time
sys.path.insert(0, "python")
from vdb.client import Client
data, port, rows = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(f"{data}/sift/base.fbin", "rb") as f:
    n, d = struct.unpack("<II", f.read(8))
    f.seek(8 + rows * d * 4)
    more = np.fromfile(f, dtype=np.float32, count=rows * d).reshape(rows, d)
t = time.time()
col = Client(f"http://127.0.0.1:{port}", timeout=900).create_collection("bulk", dim=d, sync="none")
col.bulk_load(np.arange(rows, dtype=np.uint64), more)
print(f"concurrent bulk load of {rows} rows finished in {time.time() - t:.1f}s")
EOF
LOADER=$!
"$PY" bench/server_load.py --url "http://127.0.0.1:$PORT" --collection sift --data-dir "$DATA" --dataset sift \
  --rates "${CONCURRENT_RATE:-1000}" --seconds "${CONCURRENT_SECONDS:-20}" --connections 100 \
  --out "$OUT/server_concurrent_bulk.json"
wait $LOADER
curl -s "http://127.0.0.1:$PORT/metrics" | grep -E '^vdb_(requests_total|collection_rows|segments)' | head -20
