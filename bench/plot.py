#!/usr/bin/env python3
"""Turn the raw JSON in bench/results/ into every benchmark chart and the README table.

    python bench/plot.py --results bench/results --out bench/results/charts

Each chart is written as <name>.png plus <name>.csv (the table view of the same data).
Missing inputs are skipped with a note, so partial runs still plot what they have.

Charts:
  1 recall_vs_qps_<dataset>       Recall@10 vs QPS (log), one line per system (ANN-Benchmarks style)
  2 recall_vs_ef_<dataset>        Recall@10 vs ef_search, one line per M
  3 build_time_vs_threads         HNSW build time vs threads
  4 bytes_vs_recall_<dataset>     Bytes per vector vs recall@10: float, SQ8, PQ, IVF-PQ
  5 ivfpq_nprobe_<dataset>        IVF-PQ recall vs nprobe per m, with and without re-ranking
  6 ingest                        Ingest rate per sync mode and writer count
  7 recovery                      Recovery time vs log size
  8 filter_<kind>                 Filtered recall and QPS vs selectivity, per strategy
  9 deletes                       Recall and latency vs percent deleted, before/after compaction
  10 server_latency               Server p99 latency vs offered QPS
  11 segments                     QPS at recall@10 >= 0.95 vs number of segments
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical palette (validated: adjacent CVD dE >= 9.1 light), fixed order.
# Color follows the entity: a system keeps its color in every chart.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SYSTEM_COLORS = {
    "vdb hnsw": PALETTE[0], "hnswlib": PALETTE[1], "faiss_hnsw": PALETTE[2], "vdb hnsw_sq8": PALETTE[3],
    "vdb ivf_pq": PALETTE[4], "faiss_ivfpq": PALETTE[5], "vdb pq": PALETTE[6], "vdb flat": PALETTE[7],
    "faiss_flat": PALETTE[7],
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.labelsize": 10, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linestyle": "-", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 6,
    "legend.frameon": False, "legend.fontsize": 9, "font.family": "sans-serif", "text.color": INK,
})


def load_results(results_dir):
    by_cmd = defaultdict(list)
    machine = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            doc = json.load(open(path))
        except (OSError, ValueError):
            print(f"skip unreadable {path}")
            continue
        cmd = doc.get("command", "index")
        for r in doc.get("results", []):
            r["_file"] = os.path.basename(path)
            by_cmd[cmd].append(r)
        machine = doc.get("machine", machine) or machine
    return by_cmd, machine


def system_name(r):
    return f"vdb {r['index']}" if r.get("system") == "vdb" and "index" in r else r.get("system", "?")


def med(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return median(vals) if vals else None


def group_median(rows, key_fn, fields):
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    out = {}
    for k, g in groups.items():
        out[k] = {f: med(g, f) for f in fields}
        out[k]["_rows"] = g
    return out


def save(fig, out_dir, name, header, rows):
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name + ".png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(out_dir, name + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {name}.png / .csv")


def color_for(name, i):
    return SYSTEM_COLORS.get(name, PALETTE[i % len(PALETTE)])


def pareto(points):
    """Upper-right frontier of (recall, qps) points, as ANN-Benchmarks plots."""
    pts = sorted(points, key=lambda p: (-p[0], -p[1]))
    out, best = [], -1.0
    for rec, qps, *rest in pts:
        if qps > best:
            out.append((rec, qps, *rest))
            best = qps
    return sorted(out)


# ---------------------------------------------------------------------- charts

def chart_recall_qps(res, out):
    rows = [r for r in res["index"] + res["baselines"] if r.get("threads", 1) == 1 and r.get("recall@10") is not None]
    for ds in sorted({r.get("dataset") for r in rows}):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        table = []
        systems = sorted({system_name(r) for r in rows if r.get("dataset") == ds},
                         key=lambda s: list(SYSTEM_COLORS).index(s) if s in SYSTEM_COLORS else 99)
        for i, s in enumerate(systems):
            g = group_median([r for r in rows if r.get("dataset") == ds and system_name(r) == s],
                             lambda r: json.dumps(r.get("params", {}), sort_keys=True), ["recall@10", "qps"])
            pts = pareto([(v["recall@10"], v["qps"], k) for k, v in g.items()])
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=MARKERS[i % 8], color=color_for(s, i), label=s)
            table += [[s, p[2], f"{p[0]:.4f}", f"{p[1]:.1f}"] for p in pts]
        ax.set_yscale("log")
        ax.set_xlabel("Recall@10")
        ax.set_ylabel("Queries per second (1 thread, log)")
        ax.set_title(f"Recall vs QPS - {ds}")
        ax.legend(loc="lower left")
        save(fig, out, f"recall_vs_qps_{ds}", ["system", "params", "recall@10", "qps"], table)


def chart_recall_ef(res, out):
    rows = [r for r in res["index"] if r.get("index") == "hnsw" and "ef" in r.get("params", {})]
    for ds in sorted({r.get("dataset") for r in rows}):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        table = []
        for i, M in enumerate(sorted({r["params"]["M"] for r in rows if r.get("dataset") == ds})):
            for efc in sorted({r["params"]["efc"] for r in rows if r.get("dataset") == ds and r["params"]["M"] == M}):
                g = group_median([r for r in rows if r.get("dataset") == ds and r["params"]["M"] == M and r["params"]["efc"] == efc],
                                 lambda r: r["params"]["ef"], ["recall@10"])
                xs = sorted(g)
                ax.plot(xs, [g[x]["recall@10"] for x in xs], marker=MARKERS[i % 8], color=PALETTE[i % 8],
                        label=f"M={M}, efC={efc}", alpha=1.0 if efc == 200 else 0.55)
                table += [[M, efc, x, f"{g[x]['recall@10']:.4f}"] for x in xs]
        ax.set_xscale("log", base=2)
        ax.set_xlabel("ef_search")
        ax.set_ylabel("Recall@10")
        ax.set_title(f"Recall vs ef_search - {ds}")
        ax.legend(loc="lower right")
        save(fig, out, f"recall_vs_ef_{ds}", ["M", "efc", "ef", "recall@10"], table)


def chart_build_threads(res, out):
    rows = [r for r in res["index"] + res["baselines"] if r.get("build_s") is not None and r.get("build_threads")]
    rows = [r for r in rows if system_name(r) in ("vdb hnsw", "hnswlib")]
    if len({r["build_threads"] for r in rows}) < 2:
        print("  skip build_time_vs_threads (needs builds at 2+ thread counts)")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    table = []
    for i, s in enumerate(sorted({system_name(r) for r in rows})):
        g = group_median([r for r in rows if system_name(r) == s], lambda r: r["build_threads"], ["build_s"])
        xs = sorted(g)
        ax.plot(xs, [g[x]["build_s"] for x in xs], marker=MARKERS[i], color=color_for(s, i), label=s)
        table += [[s, x, f"{g[x]['build_s']:.2f}"] for x in xs]
    ax.set_xlabel("Build threads")
    ax.set_ylabel("Build time (s)")
    ax.set_title("HNSW build time vs threads")
    ax.legend()
    save(fig, out, "build_time_vs_threads", ["system", "threads", "build_s"], table)


def chart_bytes(res, out):
    rows = [r for r in res["index"] if r.get("bytes_per_vector") and r.get("index") in ("hnsw", "hnsw_sq8", "ivf_pq", "pq")]
    for ds in sorted({r.get("dataset") for r in rows}):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        table = []
        for i, idx in enumerate(["hnsw", "hnsw_sq8", "pq", "ivf_pq"]):
            g = group_median([r for r in rows if r.get("dataset") == ds and r["index"] == idx],
                             lambda r: json.dumps(r["params"], sort_keys=True), ["recall@10", "bytes_per_vector", "qps"])
            if not g:
                continue
            # Best recall per configuration family, at its memory cost.
            best = max(g.values(), key=lambda v: v["recall@10"])
            ax.scatter([best["bytes_per_vector"]], [best["recall@10"]], s=80, marker=MARKERS[i],
                       color=color_for("vdb " + idx, i), label=f"vdb {idx}", zorder=3, edgecolors=SURFACE, linewidths=2)
            ax.annotate(f"{best['bytes_per_vector']:.0f} B", (best["bytes_per_vector"], best["recall@10"]),
                        textcoords="offset points", xytext=(6, -12), fontsize=8, color=INK2)
            table += [[idx, k, f"{v['bytes_per_vector']:.1f}", f"{v['recall@10']:.4f}", f"{v['qps']:.1f}"] for k, v in g.items()]
        ax.set_xscale("log")
        ax.set_xlabel("Bytes per vector in RAM (log)")
        ax.set_ylabel("Best recall@10")
        ax.set_title(f"Memory vs recall - {ds}")
        ax.legend(loc="lower right")
        save(fig, out, f"bytes_vs_recall_{ds}", ["index", "params", "bytes_per_vector", "recall@10", "qps"], table)


def chart_ivfpq(res, out):
    rows = [r for r in res["index"] if r.get("index") == "ivf_pq"]
    for ds in sorted({r.get("dataset") for r in rows}):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        table = []
        keys = sorted({(r["params"]["pq_m"], r["params"].get("rerank", -1)) for r in rows if r.get("dataset") == ds})
        for i, (m, rr) in enumerate(keys):
            g = group_median([r for r in rows if r.get("dataset") == ds and r["params"]["pq_m"] == m and r["params"].get("rerank", -1) == rr],
                             lambda r: r["params"]["nprobe"], ["recall@10"])
            xs = sorted(g)
            label = f"m={m}, " + ("no re-rank" if rr == 0 else f"re-rank R={'auto' if rr < 0 else rr}")
            ax.plot(xs, [g[x]["recall@10"] for x in xs], marker=MARKERS[i % 8], color=PALETTE[i % 8], label=label)
            table += [[m, rr, x, f"{g[x]['recall@10']:.4f}"] for x in xs]
        ax.set_xscale("log", base=2)
        ax.set_xlabel("nprobe")
        ax.set_ylabel("Recall@10")
        ax.set_title(f"IVF-PQ recall vs nprobe - {ds}")
        ax.legend(loc="lower right")
        save(fig, out, f"ivfpq_nprobe_{ds}", ["pq_m", "rerank", "nprobe", "recall@10"], table)


def chart_ingest(res, out):
    rows = res["ingest"]
    if not rows:
        return print("  skip ingest (no data)")
    batches = sorted({r["batch"] for r in rows})
    fig, axes = plt.subplots(1, len(batches), figsize=(5 * len(batches), 4.2), squeeze=False)
    table = []
    modes = ["always", "always+group_commit", "interval", "none"]
    for ax, b in zip(axes[0], batches):
        for i, m in enumerate(modes):
            g = group_median([r for r in rows if r["sync"] == m and r["batch"] == b], lambda r: r["writers"], ["vectors_per_s", "p99_ms"])
            xs = sorted(g)
            if not xs:
                continue
            ax.plot(xs, [g[x]["vectors_per_s"] for x in xs], marker=MARKERS[i], color=PALETTE[i], label=m)
            table += [[m, b, x, f"{g[x]['vectors_per_s']:.0f}", f"{g[x]['p99_ms']:.3f}"] for x in xs]
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Concurrent writers")
        ax.set_ylabel("Vectors per second (log)")
        ax.set_title(f"Ingest, batch of {b}")
        ax.legend()
    save(fig, out, "ingest", ["sync", "batch", "writers", "vectors_per_s", "p99_ms"], table)


def chart_recovery(res, out):
    rows = [r for r in res["recovery"] if r.get("kind") == "log_replay"]
    if not rows:
        return print("  skip recovery (no data)")
    g = group_median(rows, lambda r: r["log_mb"], ["seconds"])
    xs = sorted(g)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, [g[x]["seconds"] for x in xs], marker="o", color=PALETTE[0])
    for x in xs[-1:]:
        ax.annotate(f"{g[x]['seconds']:.2f} s", (x, g[x]["seconds"]), textcoords="offset points", xytext=(-40, 6), color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Log size (MB, log)")
    ax.set_ylabel("Seconds to ready (log)")
    ax.set_title("Recovery time vs log size")
    save(fig, out, "recovery", ["log_mb", "seconds"], [[x, f"{g[x]['seconds']:.3f}"] for x in xs])


def chart_filter(res, out):
    rows = res["filter"]
    for corr in sorted({bool(r.get("correlated")) for r in rows}):
        sub = [r for r in rows if bool(r.get("correlated")) == corr]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        table = []
        for i, s in enumerate(["planner", "prefilter", "graph", "postfilter"]):
            g = group_median([r for r in sub if r["strategy"] == s], lambda r: r["selectivity"], ["recall@10", "qps"])
            xs = sorted(g)
            if not xs:
                continue
            kw = dict(marker=MARKERS[i], color=PALETTE[i], label=s, linewidth=3 if s == "planner" else 2)
            a1.plot(xs, [g[x]["recall@10"] for x in xs], **kw)
            a2.plot(xs, [g[x]["qps"] for x in xs], **kw)
            table += [[s, f"{x:.4f}", f"{g[x]['recall@10']:.4f}", f"{g[x]['qps']:.1f}"] for x in xs]
        for ax in (a1, a2):
            ax.set_xscale("log")
            ax.set_xlabel("Selectivity (fraction of rows matching, log)")
        a1.set_ylabel("Recall@10")
        a2.set_ylabel("QPS (1 thread, log)")
        a2.set_yscale("log")
        kind = "correlated" if corr else "uncorrelated"
        a1.set_title(f"Filtered recall ({kind})")
        a2.set_title(f"Filtered QPS ({kind})")
        a1.legend(loc="lower right")
        save(fig, out, f"filter_{kind}", ["strategy", "selectivity", "recall@10", "qps"], table)


def chart_deletes(res, out):
    rows = res["deletes"]
    if not rows:
        return print("  skip deletes (no data)")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    table = []
    for i, comp in enumerate([False, True]):
        g = group_median([r for r in rows if bool(r["compacted"]) == comp], lambda r: r["deleted_pct"], ["recall@10", "p50_ms"])
        xs = sorted(g)
        label = "after compaction" if comp else "before compaction"
        a1.plot(xs, [g[x]["recall@10"] for x in xs], marker=MARKERS[i], color=PALETTE[i], label=label)
        a2.plot(xs, [g[x]["p50_ms"] for x in xs], marker=MARKERS[i], color=PALETTE[i], label=label)
        table += [[label, x, f"{g[x]['recall@10']:.4f}", f"{g[x]['p50_ms']:.3f}"] for x in xs]
    a1.set_xlabel("Rows deleted (%)")
    a2.set_xlabel("Rows deleted (%)")
    a1.set_ylabel("Recall@10")
    a2.set_ylabel("p50 latency (ms)")
    a1.set_title("Recall vs deleted rows")
    a2.set_title("Latency vs deleted rows")
    a1.legend()
    save(fig, out, "deletes", ["phase", "deleted_pct", "recall@10", "p50_ms"], table)


def chart_server(res, out):
    # The concurrent-bulk-load run is reported separately; keep only the load sweep.
    rows = [r for r in res["server_load"] if r.get("_file") == "server_load.json"]
    if not rows:
        return print("  skip server_latency (no data)")
    g = group_median(rows, lambda r: r["offered_qps"], ["p99_ms", "p50_ms", "achieved_qps"])
    xs = sorted(g)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(xs, [g[x]["p99_ms"] for x in xs], marker="o", color=PALETTE[0], label="p99")
    ax.plot(xs, [g[x]["p50_ms"] for x in xs], marker="s", color=PALETTE[1], label="p50")
    ax.set_yscale("log")
    ax.set_xlabel("Offered load (queries per second)")
    ax.set_ylabel("Latency (ms, log)")
    ax.set_title("vdb-server latency vs offered load")
    ax.legend()
    save(fig, out, "server_latency", ["offered_qps", "achieved_qps", "p50_ms", "p99_ms"],
         [[x, f"{g[x]['achieved_qps']:.1f}", f"{g[x]['p50_ms']:.2f}", f"{g[x]['p99_ms']:.2f}"] for x in xs])


def chart_segments(res, out, target=0.95):
    rows = res["segments"]
    if not rows:
        return print("  skip segments (no data)")
    best = {}
    for segs in sorted({r["segments"] for r in rows}):
        g = group_median([r for r in rows if r["segments"] == segs], lambda r: r["ef"], ["recall@10", "qps"])
        ok = [v for v in g.values() if v["recall@10"] >= target]
        if ok:
            best[segs] = max(ok, key=lambda v: v["qps"])
    xs = sorted(best)
    if not xs:
        return print("  skip segments (no configuration reached the recall target)")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar([str(x) for x in xs], [best[x]["qps"] for x in xs], color=PALETTE[0], width=0.6)
    for i, x in enumerate(xs):
        ax.annotate(f"{best[x]['qps']:.0f}", (i, best[x]["qps"]), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("Segments holding the same vectors")
    ax.set_ylabel(f"QPS at recall@10 >= {target}")
    ax.set_title("Why compaction matters: QPS vs segment count")
    ax.grid(axis="x", visible=False)
    save(fig, out, "segments", ["segments", "qps", "recall@10"],
         [[x, f"{best[x]['qps']:.1f}", f"{best[x]['recall@10']:.4f}"] for x in xs])


def chart_mutable(res, out):
    rows = res["mutable"]
    if not rows:
        return print("  skip mutable_segment (no data)")
    g = group_median(rows, lambda r: r["mutable_rows"], ["p50_ms", "p99_ms", "flush_s"])
    xs = sorted(g)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = [f"{x // 1000}k" if x else "0" for x in xs]
    ax.bar(labels, [g[x]["p50_ms"] for x in xs], color=PALETTE[0], width=0.6)
    for i, x in enumerate(xs):
        ax.annotate(f"{g[x]['p50_ms']:.2f} ms", (i, g[x]["p50_ms"]), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("Rows waiting in the mutable segment (brute-force scanned by every query)")
    ax.set_ylabel("p50 query latency (ms)")
    ax.set_title("The mutable segment is a tax on every search")
    ax.grid(axis="x", visible=False)
    save(fig, out, "mutable_segment", ["mutable_rows", "p50_ms", "p99_ms", "flush_s"],
         [[x, f"{g[x]['p50_ms']:.3f}", f"{g[x]['p99_ms']:.3f}", f"{g[x]['flush_s']:.2f}"] for x in xs])


def chart_reorder(res, out):
    rows = res["reorder"]
    if not rows:
        return print("  skip reorder (no data)")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    table = []
    for i, re in enumerate([False, True]):
        g = group_median([r for r in rows if bool(r["reordered"]) == re], lambda r: r["ef"], ["qps", "recall@10"])
        xs = sorted(g)
        label = "BFS-reordered" if re else "insertion order"
        ax.plot(xs, [g[x]["qps"] for x in xs], marker=MARKERS[i], color=PALETTE[i], label=label)
        table += [[label, x, f"{g[x]['recall@10']:.4f}", f"{g[x]['qps']:.1f}"] for x in xs]
    ax.set_xscale("log", base=2)
    ax.set_xlabel("ef_search")
    ax.set_ylabel("QPS (1 thread)")
    ax.set_title("Graph reordering for cache locality")
    ax.legend()
    save(fig, out, "reorder", ["layout", "ef", "recall@10", "qps"], table)


# ---------------------------------------------------------------------- table

def results_table(res, out_dir, target=0.95):
    """The README results table: for each system, the fastest configuration reaching
    recall@10 >= target (single thread)."""
    def same_params(r):
        # Compare every system at the same build parameters: M=16, efC=200 for graphs;
        # nlist=4096, m=16 for IVF-PQ.
        p = r.get("params", {})
        if "M" in p and (p.get("M") != 16 or p.get("efc") != 200):
            return False
        if "nlist" in p and (p.get("nlist") != 4096 or p.get("pq_m") != 16):
            return False
        return True

    # The build-thread sweep indexes a 200k subset, so it is not comparable with full-dataset runs.
    rows = [r for r in res["index"] + res["baselines"]
            if r.get("threads", 1) == 1 and r.get("recall@10") is not None and same_params(r)
            and not r.get("_file", "").startswith("build_threads")]
    lines = ["| Dataset | System | Parameters | Recall@10 | QPS, 1 thread | p99 ms | Build s | Bytes per vector |",
             "|---|---|---|---|---|---|---|---|"]
    for ds in sorted({r.get("dataset") for r in rows}):
        for s in ["vdb hnsw", "hnswlib", "faiss_hnsw", "vdb hnsw_sq8", "vdb ivf_pq", "faiss_ivfpq", "vdb pq", "vdb flat", "faiss_flat"]:
            g = group_median([r for r in rows if r.get("dataset") == ds and system_name(r) == s],
                             lambda r: json.dumps(r.get("params", {}), sort_keys=True),
                             ["recall@10", "qps", "p99_ms", "build_s", "bytes_per_vector"])
            if not g:
                continue
            ok = {k: v for k, v in g.items() if v["recall@10"] >= target}
            pick_from = ok or g
            key, v = max(pick_from.items(), key=lambda kv: kv[1]["qps"] if ok else kv[1]["recall@10"])
            params = ", ".join(f"{a}={b}" for a, b in json.loads(key).items())
            bpv = f"{v['bytes_per_vector']:.0f}" if v.get("bytes_per_vector") else ""
            build = f"{v['build_s']:.1f}" if v.get("build_s") is not None else ""
            lines.append(f"| {ds} | {s} | {params} | {v['recall@10']:.4f} | {v['qps']:.0f} | {v['p99_ms']:.3f} | {build} | {bpv} |")
    path = os.path.join(out_dir, "results_table.md")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote results_table.md ({len(lines) - 2} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="bench/results")
    ap.add_argument("--out", default="bench/results/charts")
    a = ap.parse_args()
    res, machine = load_results(a.results)
    res = defaultdict(list, res)
    print(f"loaded {sum(len(v) for v in res.values())} results; machine: {machine.get('cpu', '?')}")
    for fn in (chart_recall_qps, chart_recall_ef, chart_build_threads, chart_bytes, chart_ivfpq, chart_ingest,
               chart_recovery, chart_filter, chart_deletes, chart_server, chart_segments, chart_mutable,
               chart_reorder):
        fn(res, a.out)
    results_table(res, a.out)
    with open(os.path.join(a.out, "machine.json"), "w") as f:
        json.dump(machine, f, indent=1)


if __name__ == "__main__":
    main()
