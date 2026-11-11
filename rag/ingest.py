#!/usr/bin/env python3
"""Chunk a corpus, embed it, and load it into vdb (one collection per retrieval method).

    python rag/ingest.py --max-passages 200000                    # Simple English Wikipedia
    python rag/ingest.py --corpus my_docs.jsonl                    # {"title","url","text","date"} per line

Chunks are about 250 words with 50 words of overlap. Title, URL, section and year are
kept as metadata so filters have something to work with; the chunk text goes into a
small SQLite table keyed by id. Embeddings are saved as work/embeddings.npy, so
re-ingesting with other index settings skips the slow embedding step.
"""

import argparse
import json
import os
import re
import time

import numpy as np

from common import METHODS, WORK, collection_path, db, embed


def chunk_words(text, size=250, overlap=50):
    words = text.split()
    step = max(1, size - overlap)
    for start in range(0, max(1, len(words) - overlap), step):
        piece = words[start:start + size]
        if len(piece) >= 30:
            yield " ".join(piece)


def sections(text):
    """Split a Wikipedia article on its section headings (lines that look like titles)."""
    current, buf = "", []
    for line in text.split("\n"):
        s = line.strip()
        if s and len(s) < 80 and not s.endswith(".") and len(s.split()) <= 8 and buf:
            yield current, "\n".join(buf)
            current, buf = s, []
        elif s:
            buf.append(s)
    if buf:
        yield current, "\n".join(buf)


def documents(args):
    if args.corpus:
        with open(args.corpus) as f:
            for line in f:
                yield json.loads(line)
        return
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")  # cached after the first run
    first = set()
    if args.priority_questions and os.path.exists(args.priority_questions):
        # Articles that answer the evaluation questions go in first; the rest of the
        # corpus (in dataset order) fills up to --max-passages as distractors.
        wanted = {json.loads(line).get("title") for line in open(args.priority_questions) if line.strip()}
        titles = ds["title"]
        first = {i for i, t in enumerate(titles) if t in wanted}
        for i in sorted(first):
            row = ds[i]
            yield {"title": row["title"], "url": row["url"], "text": row["text"], "date": "2023-11-01"}
    for i, row in enumerate(ds):
        if i not in first:
            yield {"title": row["title"], "url": row["url"], "text": row["text"], "date": "2023-11-01"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", help="JSONL file; default: Simple English Wikipedia from Hugging Face")
    ap.add_argument("--max-passages", type=int, default=200000)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--reembed", action="store_true")
    ap.add_argument("--priority-questions", default="questions.jsonl",
                    help="ingest the articles these questions name first (if the file exists)")
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    con = db()
    emb_path = os.path.join(WORK, "embeddings.npy")
    if args.reembed or not os.path.exists(emb_path):
        con.execute("DELETE FROM passages")
        texts, n, t = [], 0, time.time()
        for doc in documents(args):
            year = int(m.group(0)) if (m := re.search(r"\d{4}", doc.get("date", ""))) else None
            for section, body in sections(doc["text"]):
                for chunk in chunk_words(body):
                    con.execute("INSERT INTO passages VALUES (?,?,?,?,?,?)",
                                (n, doc["title"], doc.get("url", ""), section, year, chunk))
                    texts.append(f"{doc['title']}. {chunk}")
                    n += 1
                    if n >= args.max_passages:
                        break
                if n >= args.max_passages:
                    break
            if n >= args.max_passages:
                break
        con.commit()
        print(f"{n} passages chunked in {time.time() - t:.1f}s; embedding with the MiniLM model...")
        t = time.time()
        vecs = embed(texts)
        np.save(emb_path, vecs)
        print(f"embedded {len(vecs)} x {vecs.shape[1]} in {time.time() - t:.1f}s -> {emb_path}")
    vecs = np.load(emb_path)
    ids = np.arange(len(vecs), dtype=np.uint64)
    years = [r[0] for r in con.execute("SELECT year FROM passages ORDER BY id")]
    meta = [{"year": y} if y is not None else {} for y in years]

    import vdb

    for method in args.methods.split(","):
        path = collection_path(method)
        if os.path.exists(path):
            import shutil

            shutil.rmtree(path)
        t = time.time()
        col = vdb.Collection.create(path, dim=vecs.shape[1], metric="cosine", index=METHODS[method],
                                    schema={"year": "int64"}, sync="none")
        col.bulk_load(ids, vecs, metadata=meta)
        st = col.stats()
        col.close()
        print(f"{method:6s}: {st['rows']} rows in {len(st['segments']) - 1} segment(s), "
              f"{(st['memory']['graph'] + st['memory']['codes']) / 1e6:.1f} MB index, loaded in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
