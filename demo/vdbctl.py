#!/usr/bin/env python3
"""Small commands used by the recorded demo (demo/run_demo.sh); each one talks to a running
vdb-server over HTTP through vdb.Client.

    python demo/vdbctl.py load-wiki                 # the RAG passages, with metadata, as JSON upserts
    python demo/vdbctl.py stats wiki
    python demo/vdbctl.py search "Who discovered penicillin?" [--filter JSON] [--explain]
    python demo/vdbctl.py burst --clients 8 --seconds 8
    python demo/vdbctl.py write 900000 500          # acknowledged writes, for the kill -9 scene
    python demo/vdbctl.py check 900000 500
"""

import argparse
import http.client
import json
import multiprocessing
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))
from common import WORK, db, embed, passages  # noqa: E402

from vdb.client import Client, VdbError  # noqa: E402

URL = os.environ.get("VDB_URL", "http://localhost:8090")


def vectors():
    return np.load(os.path.join(WORK, "embeddings.npy"))


def load_wiki(a):
    c = Client(URL)
    vecs = vectors()
    con = db()
    rows = con.execute("SELECT id, title, year FROM passages ORDER BY id").fetchall()
    n = min(len(rows), len(vecs), a.max)
    col = c.create_collection("wiki", dim=vecs.shape[1], metric="cosine",
                              index={"type": "hnsw", "M": 16, "ef_construction": 200},
                              schema={"year": "int64", "title": "keyword"}, sync="always")
    print(f"created collection wiki: dim {vecs.shape[1]}, cosine, hnsw M=16, sync=always")
    t = time.time()
    batch = 1000
    for s in range(0, n, batch):
        e = min(n, s + batch)
        meta = [{"title": r[1], **({"year": r[2]} if r[2] is not None else {})} for r in rows[s:e]]
        lsn = col.upsert(np.arange(s, e), vecs[s:e], metadata=meta)
        if e % 5000 == 0 or e == n:
            print(f"  upserted {e:6d} passages  ({e / (time.time() - t):6.0f}/s, last LSN {lsn})", flush=True)
    print(f"done: {n} passages with metadata in {time.time() - t:.1f} s")


def stats(a):
    st = Client(URL).collection(a.name).stats()
    print(f"{a.name}: {st['rows']} rows, {st['segments']} segments, {st['disk_bytes'] / 1e6:.1f} MB on disk")
    for s in st["segment_list"]:
        print(f"  segment {s['id']:3d}  {s['kind']:9s}  {s['index']:5s}  {s['rows']:6d} rows")
    m = st["memory"]
    print(f"memory: vectors {m['vectors'] / 1e6:.1f} MB, graph {m['graph'] / 1e6:.1f} MB, "
          f"metadata {m['metadata'] / 1e6:.1f} MB")
    print(f"log: last LSN {st['last_lsn']}, flushed LSN {st['flushed_lsn']}, "
          f"flushes {st['flushes']}, compactions {st['compactions']}")


def search(a):
    t = time.time()
    embed(["warm-up"])  # loads the MiniLM model
    t1 = time.time()
    q = embed([a.question])[0]
    print(f"loaded MiniLM in {t1 - t:.1f} s; embedded the question in {(time.time() - t1) * 1e3:.0f} ms")
    flt = json.loads(a.filter) if a.filter else None
    r = Client(URL).collection("wiki").query(q, k=a.k, ef=64, filter=flt, include=["metadata"], explain=a.explain)
    print(f"search: {r['took_ms']:.2f} ms" + (f"  filter {a.filter}" if flt else ""))
    rows = {p[0]: p for p in passages(db(), [h["id"] for h in r["results"]])}
    for i, h in enumerate(r["results"]):
        p = rows.get(h["id"])
        snippet = p[5][:44].replace("\n", " ") if p else ""
        print(f"  {i + 1}. {h['score']:.3f}  {h['metadata'].get('title', '')[:24]:24s}  {snippet}")
    if a.explain:
        for s in r["explain"]["segments"]:
            print(f"  explain: segment {s['segment']:>2} {s['kind']:9s} {s['strategy']:15s} "
                  f"matches {s['matches']:5d}/{s['live_rows']:<5d}  distances {s['distance_computations']}")


def _burst_worker(args):
    """One client process: a keep-alive connection sending pre-encoded search bodies."""
    seed, seconds, start = args
    from urllib.parse import urlparse

    u = urlparse(URL)
    vecs = vectors()
    rng = np.random.default_rng(seed)
    bodies = [json.dumps({"vector": vecs[i].tolist(), "k": 10, "ef": 64}).encode()
              for i in rng.integers(len(vecs), size=200)]
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
    while time.time() < start:
        time.sleep(0.001)
    lat, errors, i = [], 0, 0
    stop = start + seconds
    while time.time() < stop:
        t = time.perf_counter()
        conn.request("POST", "/v1/collections/wiki/search", body=bodies[i % len(bodies)],
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        r.read()
        if r.status == 200:
            lat.append((time.perf_counter() - t) * 1e3)
        else:
            errors += 1
        i += 1
    return lat, errors


def burst(a):
    print(f"{a.clients} client processes (keep-alive connections) searching for {a.seconds:.0f} s ...", flush=True)
    start = time.time() + 2.0  # let every client load its queries first
    with multiprocessing.Pool(a.clients) as pool:
        out = pool.map(_burst_worker, [(i, a.seconds, start) for i in range(a.clients)])
    lat = sorted(x for part, _ in out for x in part)
    errors = sum(e for _, e in out)
    print(f"{len(lat)} searches, {len(lat) / a.seconds:.0f} QPS, p50 {lat[len(lat) // 2]:.2f} ms, "
          f"p99 {lat[int(len(lat) * 0.99)]:.2f} ms, {errors} errors")


def write(a):
    vecs = vectors()
    col = Client(URL).collection("wiki")
    ids = np.arange(a.first, a.first + a.n)
    lsn = col.upsert(ids, vecs[: a.n], metadata=[{"title": "probe"}] * a.n, batch=50)
    print(f"acknowledged {a.n} new points (ids {a.first}..{a.first + a.n - 1}), last LSN {lsn}; "
          f"each response came after fsync")


def check(a):
    vecs = vectors()
    col = Client(URL).collection("wiki")
    ok = 0
    for i in range(a.n):
        try:
            v, md = col.get(a.first + i)
            ok += bool(np.allclose(v, vecs[i], atol=1e-5) and md.get("title") == "probe")
        except VdbError:
            pass
    print(f"{ok}/{a.n} acknowledged points present and intact after kill -9")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("load-wiki")
    p.add_argument("--max", type=int, default=10 ** 9)
    p.set_defaults(fn=load_wiki)
    p = sub.add_parser("stats")
    p.add_argument("name")
    p.set_defaults(fn=stats)
    p = sub.add_parser("search")
    p.add_argument("question")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--filter", help='filter JSON, e.g. {"field": "title", "in": ["NASA"]}')
    p.add_argument("--explain", action="store_true")
    p.set_defaults(fn=search)
    p = sub.add_parser("burst")
    p.add_argument("--clients", type=int, default=16)
    p.add_argument("--seconds", type=float, default=8)
    p.set_defaults(fn=burst)
    for name, fn in (("write", write), ("check", check)):
        p = sub.add_parser(name)
        p.add_argument("first", type=int)
        p.add_argument("n", type=int)
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
