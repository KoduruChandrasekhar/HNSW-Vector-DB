#!/usr/bin/env python3
"""Batched search from Python vs the C++ runner, same collection and parameters.

    python bench/python_qps.py --dataset sift --max-base 200000 --out bench/results/python_qps.json

Bulk-loads one segment through the bindings, then times col.search(queries) with one
thread (the C++ thread pool is sized by `threads`), so the number is directly
comparable with `vdb_bench segments --segments 1` at the same ef.
"""

import argparse
import json
import os
import shutil
import struct
import sys
import time

import numpy as np

import vdb  # the installed package (pip install .), which has the compiled module


def fbin(path, n=0):
    with open(path, "rb") as f:
        rows, d = struct.unpack("<II", f.read(8))
        rows = min(rows, n) if n else rows
        return np.fromfile(f, dtype=np.float32, count=rows * d).reshape(rows, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--dataset", default="sift")
    ap.add_argument("--max-base", type=int, default=200000)
    ap.add_argument("--max-query", type=int, default=1000)
    ap.add_argument("--ef", default="16,32,64,128,256")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--work", default="/tmp/vdb_python_qps")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    base = fbin(os.path.join(a.data_dir, a.dataset, "base.fbin"), a.max_base)
    q = fbin(os.path.join(a.data_dir, a.dataset, "query.fbin"), a.max_query)
    shutil.rmtree(a.work, ignore_errors=True)
    t = time.time()
    col = vdb.Collection.create(a.work, dim=base.shape[1], sync="none")
    col.bulk_load(np.arange(len(base), dtype=np.uint64), base)
    col.close()
    print(f"bulk-loaded {len(base)} rows in {time.time() - t:.1f}s")
    results = []
    for threads in (1, 0):
        col = vdb.Collection.open(a.work, threads=threads)
        for ef in [int(x) for x in a.ef.split(",")]:
            col.search(q[:200], k=a.k, ef=ef)  # warm-up
            t = time.perf_counter()
            col.search(q, k=a.k, ef=ef)
            qps = len(q) / (time.perf_counter() - t)
            results.append({"system": "vdb-python", "threads": threads or os.cpu_count(), "ef": ef, "qps": qps})
            print(f"  threads={threads or 'all'} ef={ef:4d}: {qps:9.1f} qps")
        col.close()
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"command": "python_qps", "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
