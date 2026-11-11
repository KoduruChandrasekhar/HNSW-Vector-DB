#!/usr/bin/env python3
"""Hit rate@5 for the four retrieval methods: does the passage that answers each
question appear in the top 5? This measures how much approximate-search recall loss
matters for the end task.

    python rag/eval.py --questions rag/questions.jsonl
    python rag/eval.py --sample 50          # print 50 random passages to write questions from

questions.jsonl, one per line:
    {"question": "When did the Berlin Wall fall?", "passage_id": 18234}
    {"question": "...", "title": "Penicillin"}     # any passage of that article counts
"""

import argparse
import json
import os
import random
import time

import numpy as np

from common import METHODS, collection_path, db, embed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="rag/questions.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ef", type=int, default=64)
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench",
                                                  "results", "rag_hit_rate.json"))
    args = ap.parse_args()
    con = db()
    if args.sample:
        n = con.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
        for pid in random.Random(0).sample(range(n), args.sample):
            title, text = con.execute("SELECT title, text FROM passages WHERE id=?", (pid,)).fetchone()
            print(json.dumps({"passage_id": pid, "title": title, "text": text[:400]}))
        return
    qs = [json.loads(line) for line in open(args.questions) if line.strip()]
    # Only questions whose answer is in the ingested corpus count.
    answerable = []
    for q in qs:
        if "title" in q and con.execute("SELECT 1 FROM passages WHERE title=? LIMIT 1", (q["title"],)).fetchone() is None:
            print(f"  skipped (article not in corpus): {q['title']}")
            continue
        answerable.append(q)
    qs = answerable
    qvecs = embed([q["question"] for q in qs])
    titles = {}
    import vdb

    results = []
    print(f"{len(qs)} questions, k={args.k}, ef={args.ef}, nprobe={args.nprobe}")
    exact_ids = None
    for method in METHODS:  # "exact" first: the others' recall is measured against it
        col = vdb.Collection.open(collection_path(method))
        t = time.time()
        ids, _ = col.search(qvecs, k=args.k, ef=args.ef, nprobe=args.nprobe)
        ms = (time.time() - t) * 1e3 / len(qs)
        col.close()
        if exact_ids is None:
            exact_ids = ids
        recall = float(np.mean([len(set(a.tolist()) & set(b.tolist())) / args.k for a, b in zip(ids, exact_ids)]))
        hits = 0
        for q, row in zip(qs, ids):
            found = [int(i) for i in row if i != np.iinfo(np.uint64).max]
            if "passage_id" in q:
                hits += int(q["passage_id"]) in found
            else:
                for pid in found:
                    if pid not in titles:
                        titles[pid] = con.execute("SELECT title FROM passages WHERE id=?", (pid,)).fetchone()[0]
                hits += any(titles[p] == q["title"] for p in found)
        rate = hits / len(qs)
        results.append({"method": method, "hit_rate@k": rate, "recall@k_vs_exact": recall, "k": args.k,
                        "ef": args.ef, "nprobe": args.nprobe, "ms_per_query": ms})
        print(f"  {method:6s}  hit rate@{args.k} {rate:.3f}  recall@{args.k} vs exact {recall:.3f}  ({ms:.2f} ms/query)")
    doc = {"command": "rag_eval", "results": []}
    if os.path.exists(args.out):  # keep the runs at other ef / nprobe settings
        doc = json.load(open(args.out))
    doc["questions"] = len(qs)
    doc["results"] = [r for r in doc["results"] if (r.get("ef"), r.get("nprobe")) != (args.ef, args.nprobe)] + results
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
