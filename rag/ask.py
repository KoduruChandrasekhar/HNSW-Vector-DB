#!/usr/bin/env python3
"""Answer a question end to end: embed it, retrieve k passages from vdb, and stream an
answer with numbered citations from your LLM inference server.

    LLM_URL=http://localhost:8000 python rag/ask.py "Who discovered penicillin?"
    python rag/ask.py "..." --year-gte 2020 --method sq8 --k 5
    python rag/ask.py "..." --vdb-url http://localhost:8090 --collection wiki   # via vdb-server

The inference server must expose an OpenAI-compatible POST /v1/chat/completions with
"stream": true (server-sent events: "data: {...}" lines, then "data: [DONE]").
"""

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np

from common import METHODS, collection_path, db, embed, passages

SYSTEM = ("Answer the question using only the numbered passages. Cite passages like [1] or [2][3]. "
          "If the passages do not contain the answer, say so.")


def retrieve(args, qvec):
    flt = {"field": "year", "gte": args.year_gte} if args.year_gte else None
    if args.vdb_url:
        from vdb import Client

        ids, dists = Client(args.vdb_url).collection(args.collection).search(qvec, k=args.k, ef=args.ef, filter=flt)
    else:
        import vdb

        col = vdb.Collection.open(collection_path(args.method))
        t = time.time()
        ids, dists = col.search(qvec, k=args.k, ef=args.ef, filter=flt)
        args.search_ms = (time.time() - t) * 1e3
        col.close()
    keep = ids != np.iinfo(np.uint64).max
    return ids[keep], dists[keep]


def stream_chat(url, model, messages, max_tokens):
    body = json.dumps({"model": model, "messages": messages, "stream": True, "max_tokens": max_tokens,
                       "temperature": 0.2}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                delta = json.loads(data)["choices"][0].get("delta", {})
            except (ValueError, KeyError, IndexError):
                continue
            if delta.get("content"):
                yield delta["content"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ef", type=int, default=64)
    ap.add_argument("--method", default="hnsw", choices=sorted(METHODS))
    ap.add_argument("--year-gte", type=int)
    ap.add_argument("--vdb-url", help="search through vdb-server instead of the embedded collection")
    ap.add_argument("--collection", default="wiki")
    ap.add_argument("--llm-url", default=os.environ.get("LLM_URL", "http://localhost:8000"))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "default"))
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--no-llm", action="store_true", help="only show the retrieved passages")
    args = ap.parse_args()

    t = time.time()
    qvec = embed([args.question])[0]  # the first call also loads the embedding model
    t_embed = time.time() - t
    args.search_ms = None
    t = time.time()
    ids, dists = retrieve(args, qvec)
    t_retrieve = time.time() - t
    rows = passages(db(), ids)
    context = "\n\n".join(f"[{i + 1}] {r[1]}{' - ' + r[3] if r[3] else ''}\n{r[5]}" for i, r in enumerate(rows))
    search = f", search {args.search_ms:.2f} ms" if args.search_ms is not None else ""
    print(f"retrieved {len(rows)} passages ({args.method}): embed {t_embed * 1e3:.0f} ms, "
          f"open + search {t_retrieve * 1e3:.0f} ms{search}\n", file=sys.stderr)
    if args.no_llm:
        print(context)
        return
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {args.question}"}]
    t = time.time()
    first = None
    for tok in stream_chat(args.llm_url, args.model, messages, args.max_tokens):
        first = first or time.time()
        print(tok, end="", flush=True)
    print("\n\nSources:")
    for i, r in enumerate(rows):
        print(f"  [{i + 1}] {r[1]}{' / ' + r[3] if r[3] else ''} {r[2]}")
    if first:
        print(f"\n(time to first token {(first - t) * 1e3:.0f} ms)", file=sys.stderr)


if __name__ == "__main__":
    main()
