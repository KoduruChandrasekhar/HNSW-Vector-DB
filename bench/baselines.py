#!/usr/bin/env python3
"""hnswlib and FAISS baselines with parameters matched to vdb_bench.

    python bench/baselines.py --dataset sift --systems hnswlib,faiss_hnsw,faiss_flat,faiss_ivfpq \
        --threads 1 --out bench/results/baselines_sift.json

Fairness rules (bench/README in the plan): the same k, metric and thread count; QPS comes
from each library's *batch* search with the thread count fixed, so Python loop overhead
is not counted; per-query latency percentiles come from one query per call (they do
include the Python call overhead, a few microseconds, and are labeled as such).
"""

import argparse
import json
import os
import platform
import struct
import subprocess
import time

import numpy as np


def read_bin(path, dtype, max_rows=0):
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        if max_rows:
            n = min(n, max_rows)
        return np.fromfile(f, dtype=dtype, count=n * d).reshape(n, d)


def load(data_dir, name, max_base, max_query):
    d = os.path.join(data_dir, name)
    base = read_bin(os.path.join(d, "base.fbin"), np.float32, max_base)
    query = read_bin(os.path.join(d, "query.fbin"), np.float32, max_query)
    angular = any(s in name for s in ("glove", "angular", "rag"))
    if angular:
        base = base / np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-30)
        query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-30)
    full_n = struct.unpack("<II", open(os.path.join(d, "base.fbin"), "rb").read(8))[0]
    if max_base and max_base < full_n:
        gt = exact_gt(base, query, angular)
    else:
        gt = read_bin(os.path.join(d, "gt.ibin"), np.int32, max_query)
    return base.astype(np.float32), query.astype(np.float32), gt, angular


def exact_gt(base, query, angular, k=100):
    import faiss

    idx = faiss.IndexFlatIP(base.shape[1]) if angular else faiss.IndexFlatL2(base.shape[1])
    idx.add(base)
    return idx.search(query, k)[1].astype(np.int32)


def recall(found, gt, k):
    return float(np.mean([len(set(found[i, :k]) & set(gt[i, :k])) / k for i in range(len(found))]))


def machine():
    cpu = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"cpu": cpu, "cores": os.cpu_count(), "os": platform.platform(), "python": platform.python_version(),
            "git_commit": commit}


def measure(name, params, search_batch, search_one, query, gt, k, build_s, index_bytes, n, extra):
    search_batch(query[: min(1000, len(query))])  # warm-up
    t = time.perf_counter()
    found = search_batch(query)
    qps = len(query) / (time.perf_counter() - t)
    lat = []
    for i in range(min(len(query), 2000)):
        t1 = time.perf_counter()
        search_one(query[i: i + 1])
        lat.append((time.perf_counter() - t1) * 1e3)
    r = {"system": name, "params": params, "k": k, "threads": extra["threads"],
         "recall@1": recall(found, gt, 1), "recall@10": recall(found, gt, min(10, k)), "qps": qps,
         "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
         "p99_ms": float(np.percentile(lat, 99)), "latency_includes_python_call": True,
         "build_s": build_s, "index_bytes": index_bytes, "bytes_per_vector": index_bytes / n,
         "label": f"{name} {json.dumps(params)}", "dataset": extra["dataset"], "run": extra["run"]}
    if k >= 100:
        r["recall@100"] = recall(found, gt, 100)
    print(f"  {r['label']:<48} recall@10 {r['recall@10']:.4f}  qps {qps:9.1f}  p99 {r['p99_ms']:.3f} ms", flush=True)
    return r


def run_hnswlib(base, query, gt, angular, a, extra):
    import hnswlib

    out = []
    for M in a.M:
        for efc in a.efc:
            idx = hnswlib.Index(space="cosine" if angular else "l2", dim=base.shape[1])
            idx.init_index(max_elements=len(base), M=M, ef_construction=efc, random_seed=100)
            idx.set_num_threads(a.build_threads or os.cpu_count())
            t = time.perf_counter()
            idx.add_items(base, np.arange(len(base)))
            build_s = time.perf_counter() - t
            idx.set_num_threads(a.threads)
            # hnswlib's level-0 block: links + vector + label per element.
            nbytes = len(base) * (4 * (2 * M + 1) + 4 * base.shape[1] + 8)
            for ef in a.ef:
                idx.set_ef(max(ef, a.k))
                for run in range(a.runs):
                    extra["run"] = run
                    out.append(measure("hnswlib", {"M": M, "efc": efc, "ef": ef},
                                       lambda q: idx.knn_query(q, k=a.k)[0], lambda q: idx.knn_query(q, k=a.k),
                                       query, gt, a.k, build_s, nbytes, len(base), extra))
    return out


def run_faiss(kind, base, query, gt, angular, a, extra):
    import faiss

    d = base.shape[1]
    metric = faiss.METRIC_INNER_PRODUCT if angular else faiss.METRIC_L2
    out = []
    faiss.omp_set_num_threads(a.build_threads or os.cpu_count())
    if kind == "faiss_flat":
        idx = faiss.IndexFlat(d, metric)
        t = time.perf_counter()
        idx.add(base)
        build_s = time.perf_counter() - t
        faiss.omp_set_num_threads(a.threads)
        for run in range(a.runs):
            extra["run"] = run
            out.append(measure("faiss_flat", {}, lambda q: idx.search(q, a.k)[1], lambda q: idx.search(q, a.k),
                               query, gt, a.k, build_s, base.nbytes, len(base), extra))
    elif kind == "faiss_hnsw":
        for M in a.M:
            for efc in a.efc:
                idx = faiss.IndexHNSWFlat(d, M, metric)
                idx.hnsw.efConstruction = efc
                t = time.perf_counter()
                idx.add(base)
                build_s = time.perf_counter() - t
                faiss.omp_set_num_threads(a.threads)
                nbytes = base.nbytes + len(base) * 4 * 2 * M
                for ef in a.ef:
                    idx.hnsw.efSearch = max(ef, a.k)
                    for run in range(a.runs):
                        extra["run"] = run
                        out.append(measure("faiss_hnsw", {"M": M, "efc": efc, "ef": ef},
                                           lambda q: idx.search(q, a.k)[1], lambda q: idx.search(q, a.k),
                                           query, gt, a.k, build_s, nbytes, len(base), extra))
                faiss.omp_set_num_threads(a.build_threads or os.cpu_count())
    elif kind == "faiss_ivfpq":
        for nlist in a.nlist:
            for m in a.pq_m:
                quant = faiss.IndexFlat(d, metric)
                idx = faiss.IndexIVFPQ(quant, d, nlist, m, 8, metric)
                t = time.perf_counter()
                train = base[np.random.default_rng(0).choice(len(base), min(len(base), nlist * 64), replace=False)]
                idx.train(train)
                idx.add(base)
                build_s = time.perf_counter() - t
                faiss.omp_set_num_threads(a.threads)
                nbytes = len(base) * (m + 8) + nlist * d * 4
                for nprobe in a.nprobe:
                    for rr in a.rerank:
                        idx.nprobe = nprobe
                        if rr > 0:
                            ref = faiss.IndexRefineFlat(idx, faiss.swig_ptr(base))
                            ref.k_factor = rr / a.k
                            searcher = ref
                        else:
                            searcher = idx
                        for run in range(a.runs):
                            extra["run"] = run
                            out.append(measure("faiss_ivfpq", {"nlist": nlist, "pq_m": m, "nprobe": nprobe, "rerank": rr},
                                               lambda q, s=searcher: s.search(q, a.k)[1],
                                               lambda q, s=searcher: s.search(q, a.k), query, gt, a.k, build_s,
                                               nbytes, len(base), extra))
                faiss.omp_set_num_threads(a.build_threads or os.cpu_count())
    return out


def ints(s):
    return [int(x) for x in s.split(",") if x]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--dataset", default="sift")
    ap.add_argument("--systems", default="hnswlib,faiss_hnsw")
    ap.add_argument("--max-base", type=int, default=0)
    ap.add_argument("--max-query", type=int, default=10000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--build-threads", type=int, default=0)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--M", type=ints, default=[16])
    ap.add_argument("--efc", type=ints, default=[200])
    ap.add_argument("--ef", type=ints, default=[10, 20, 40, 80, 160, 320, 640])
    ap.add_argument("--nlist", type=ints, default=[4096])
    ap.add_argument("--pq-m", type=ints, default=[16])
    ap.add_argument("--nprobe", type=ints, default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--rerank", type=ints, default=[0, 100])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    base, query, gt, angular = load(a.data_dir, a.dataset, a.max_base, a.max_query)
    print(f"{a.dataset}: {base.shape} base, {query.shape[0]} queries, {'angular' if angular else 'l2'}")
    extra = {"dataset": a.dataset, "threads": a.threads, "run": 0}
    results = []
    for s in a.systems.split(","):
        if s == "hnswlib":
            results += run_hnswlib(base, query, gt, angular, a, extra)
        elif s.startswith("faiss"):
            results += run_faiss(s, base, query, gt, angular, a, extra)
        else:
            raise SystemExit(f"unknown system {s}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"machine": machine(), "command": "baselines", "results": results}, f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
