"""Shared pieces of the RAG demo: paths, the embedding model, the passage store."""

import os
import sqlite3

# Requires the installed package (pip install .): the embedded engine is a compiled module.
HERE = os.path.dirname(os.path.abspath(__file__))

WORK = os.environ.get("RAG_WORK", os.path.join(HERE, "work"))
MODEL = os.environ.get("RAG_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # 384 dims, cosine
METHODS = {
    # name -> index options; every collection holds the same embeddings
    "exact": {"type": "flat"},
    "hnsw": {"type": "hnsw", "M": 16, "ef_construction": 200},
    "sq8": {"type": "hnsw_sq8", "M": 16, "ef_construction": 200},
    "ivfpq": {"type": "ivf_pq", "pq_m": 48, "nlist": 1024},
}


def collection_path(method):
    return os.path.join(WORK, f"col_{method}")


def db():
    os.makedirs(WORK, exist_ok=True)
    con = sqlite3.connect(os.path.join(WORK, "passages.sqlite"))
    con.execute("CREATE TABLE IF NOT EXISTS passages (id INTEGER PRIMARY KEY, title TEXT, url TEXT, "
                "section TEXT, year INTEGER, text TEXT)")
    return con


def passages(con, ids):
    ids = [int(i) for i in ids]
    q = f"SELECT id, title, url, section, year, text FROM passages WHERE id IN ({','.join('?' * len(ids))})"
    rows = {r[0]: r for r in con.execute(q, ids)}
    return [rows[i] for i in ids if i in rows]


_model = None


def embed(texts, batch_size=128):
    """Unit-length float32 embeddings."""
    global _model
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if _model is None:
        _model = SentenceTransformer(MODEL)
    v = _model.encode(list(texts), batch_size=batch_size, show_progress_bar=len(texts) > 1000,
                      convert_to_numpy=True, normalize_embeddings=True)
    return v.astype(np.float32)
