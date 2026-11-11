#!/usr/bin/env python3
"""kill -9 crash harness for vdb-server.

    python tools/crash_test.py --server build/release/vdb-server --iterations 1000

Each iteration: start vdb-server (sync mode "always"), run writer threads that upsert
and delete random ids and record every acknowledged request, SIGKILL the server after a
random delay (10 ms to 5 s), restart it, and check that:
  * every acknowledged upsert is present with exactly the right vector,
  * every acknowledged delete is gone,
  * no unacknowledged write is half-applied (a row is either fully there or absent).
Failures are logged with their seed so they can be replayed (--seed S --iterations 1).

Note: kill -9 does not simulate power loss; the page cache survives the process. The
FaultInjectionEnv tests (tests/recovery_test.cc) cover that.
"""

import argparse
import http.client
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

DIM = 8
IDS_PER_WRITER = 200


def vec(id_, version):
    # Exactly representable in float32 and in JSON, so comparisons can be exact.
    return [((id_ * 31 + version * 7 + d * 3) % 97) / 8.0 for d in range(DIM)]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Conn:
    def __init__(self, port):
        self.port = port
        self.c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)

    def request(self, method, path, body=None):
        data = json.dumps(body) if body is not None else None
        self.c.request(method, path, body=data, headers={"Content-Type": "application/json"})
        r = self.c.getresponse()
        payload = r.read()
        return r.status, (json.loads(payload) if payload and payload[:1] in (b"{", b"[") else payload)


def start(server, data_dir, port, log):
    p = subprocess.Popen([server, "--data-dir", data_dir, "--port", str(port), "--host", "127.0.0.1",
                          "--threads", "16", "--log-level", "warn"], stdout=log, stderr=log)
    deadline = time.time() + 60
    while time.time() < deadline:
        if p.poll() is not None:
            raise RuntimeError(f"server exited with {p.returncode} during startup")
        try:
            st, _ = Conn(port).request("GET", "/health")
            if st == 200:
                return p
        except OSError:
            time.sleep(0.02)
    raise RuntimeError("server did not become ready")


def writer(port, wid, rng, acked, inflight, stop):
    """Owns ids [wid*IDS_PER_WRITER, ...): operations on one id are serial, so the model is exact."""
    conn = Conn(port)
    version = 0
    while not stop.is_set():
        id_ = wid * IDS_PER_WRITER + rng.randrange(IDS_PER_WRITER)
        version += 1
        op = ("del", None) if rng.random() < 0.25 else ("put", version)
        inflight[id_] = op
        try:
            if op[0] == "put":
                st, _ = conn.request("POST", "/v1/collections/crash/upsert",
                                     {"points": [{"id": id_, "vector": vec(id_, version)}]})
            else:
                st, _ = conn.request("POST", "/v1/collections/crash/delete", {"ids": [id_]})
        except (OSError, http.client.HTTPException):
            return  # the server died: this op stays in flight (maybe applied, maybe not)
        if st != 200:
            return
        acked[id_] = op
        del inflight[id_]


def verify(port, acked, inflight):
    conn = Conn(port)
    errors = []
    for id_ in set(acked) | set(inflight):
        st, body = conn.request("GET", f"/v1/collections/crash/points/{id_}")
        got = body["vector"] if st == 200 else None
        allowed = []
        for state in ([acked[id_]] if id_ in acked else [("del", None)]) + ([inflight[id_]] if id_ in inflight else []):
            allowed.append(None if state[0] == "del" else vec(id_, state[1]))
        if got not in allowed:
            errors.append(f"id {id_}: got {got}, acknowledged {acked.get(id_)}, in flight {inflight.get(id_)}")
    return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="build/release/vdb-server")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--writers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--min-delay", type=float, default=0.01)
    ap.add_argument("--max-delay", type=float, default=5.0)
    ap.add_argument("--keep", action="store_true", help="keep the data directory")
    args = ap.parse_args()

    data_dir = tempfile.mkdtemp(prefix="vdb_crash_")
    log = open(os.path.join(data_dir, "server.log"), "ab")
    acked, inflight = {}, {}  # survive across iterations: the collection persists
    port = free_port()
    proc = start(args.server, data_dir, port, log)
    st, body = Conn(port).request("POST", "/v1/collections", {"name": "crash", "dim": DIM, "sync": "always",
                                                              "flush_bytes": 16384, "tier_base_rows": 200,
                                                              "index": {"type": "hnsw", "M": 8, "ef_construction": 40}})
    assert st == 201, body
    failures = 0
    base_seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    t0 = time.time()
    for it in range(args.iterations):
        seed = base_seed + it
        rng = random.Random(seed)
        stop = threading.Event()
        threads = [threading.Thread(target=writer, args=(port, w, random.Random(seed * 100 + w), acked, inflight, stop))
                   for w in range(args.writers)]
        for t in threads:
            t.start()
        time.sleep(rng.uniform(args.min_delay, args.max_delay))
        proc.send_signal(signal.SIGKILL)
        proc.wait()
        stop.set()
        for t in threads:
            t.join()
        port = free_port()
        try:
            proc = start(args.server, data_dir, port, log)
        except RuntimeError as e:
            print(f"iteration {it} seed {seed}: restart FAILED: {e}", flush=True)
            failures += 1
            break
        errors = verify(port, acked, inflight)
        # In-flight ops are now resolved: whatever the server holds is the truth.
        for id_, op in list(inflight.items()):
            st, body = Conn(port).request("GET", f"/v1/collections/crash/points/{id_}")
            if st != 200:
                acked[id_] = ("del", None)
            elif op[0] == "put" and body["vector"] == vec(id_, op[1]):
                acked[id_] = op  # the in-flight upsert was applied
            # otherwise the previous acknowledged upsert is still the state
        inflight.clear()
        if errors:
            failures += 1
            print(f"iteration {it} seed {seed}: {len(errors)} errors, e.g. {errors[:3]}", flush=True)
        if it % 10 == 0 or errors:
            live = sum(1 for v in acked.values() if v[0] == "put")
            print(f"[{time.time() - t0:7.1f}s] iteration {it + 1}/{args.iterations}: {len(acked)} ids tracked, "
                  f"{live} live, failures {failures}", flush=True)
    proc.send_signal(signal.SIGTERM)
    proc.wait()
    print(json.dumps({"iterations": args.iterations, "failures": failures, "seed": base_seed,
                      "acknowledged_ids": len(acked), "seconds": time.time() - t0}))
    if not args.keep and failures == 0:
        shutil.rmtree(data_dir, ignore_errors=True)
    else:
        print(f"data kept in {data_dir}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
