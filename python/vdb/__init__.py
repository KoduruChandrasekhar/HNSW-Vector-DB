"""vdb: a vector database with HNSW, product quantization, a WAL and filtered search.

    import numpy as np, vdb
    col = vdb.Collection.create("./data/sift", dim=128, metric="l2",
                                index={"type": "hnsw", "M": 16, "ef_construction": 200})
    col.bulk_load(ids=np.arange(len(base), dtype=np.uint64), vectors=base)  # one segment
    col.upsert(ids=new_ids, vectors=new_rows)                                # normal write path
    ids, dists = col.search(queries, k=10, ef=64, filter={"field": "year", "gte": 2020})
    col.optimize()                                                           # merge to one segment
    col.close()
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from ._vdb import _Collection
except ImportError:  # a source checkout without the compiled module: only the HTTP client works
    _Collection = None
from .client import Client, RemoteCollection

__all__ = ["Collection", "Client", "RemoteCollection"]
__version__ = "0.1.0"

MISSING_ID = np.iinfo(np.uint64).max  # fills result slots when fewer than k rows exist


def _ids(ids: Any) -> np.ndarray:
    return np.ascontiguousarray(ids, dtype=np.uint64).reshape(-1)


def _rows(vectors: Any) -> np.ndarray:
    # A C-contiguous float32 array passes through with no copy.
    a = np.ascontiguousarray(vectors, dtype=np.float32)
    return a.reshape(1, -1) if a.ndim == 1 else a


def _metadata(md: Optional[Sequence[Mapping[str, Any]]], n: int) -> Optional[list]:
    if md is None:
        return None
    if len(md) != n:
        raise ValueError("metadata must have one entry per vector")
    return [json.dumps(m) for m in md]


class Collection:
    """An embedded collection. Thread-safe: searches release the GIL."""

    def __init__(self, impl: "_Collection"):
        if _Collection is None:
            raise ImportError("vdb was imported without its compiled module; run `pip install .`")
        self._c = impl

    @classmethod
    def create(cls, path: str, dim: int, metric: str = "l2", index: Optional[Mapping[str, Any]] = None,
               schema: Optional[Mapping[str, str]] = None, sync: str = "always", threads: int = 0,
               **options: Any) -> "Collection":
        """Create a collection. index: {"type": "hnsw"|"hnsw_sq8"|"pq"|"ivf_pq"|"flat", "M", "ef_construction",
        "pq_m", "nlist"}. schema: {"year": "int64", "tags": "keyword_list", ...}. Other options (flush_bytes,
        tier_base_rows, ...) pass through."""
        opts = {"dim": int(dim), "metric": metric, "sync": sync}
        if index is not None:
            opts["index"] = dict(index)
        if schema is not None:
            opts["schema"] = dict(schema)
        opts.update(options)
        cls._require()
        return cls(_Collection.create(str(path), json.dumps(opts), threads))

    @classmethod
    def open(cls, path: str, threads: int = 0) -> "Collection":
        cls._require()
        return cls(_Collection.open(str(path), threads))

    @staticmethod
    def _require() -> None:
        if _Collection is None:
            raise ImportError("vdb was imported without its compiled module; run `pip install .`")

    @property
    def dim(self) -> int:
        return self._c.dim()

    def upsert(self, ids: Any, vectors: Any, metadata: Optional[Sequence[Mapping[str, Any]]] = None) -> int:
        """Insert or replace rows through the write-ahead log. Returns the last LSN."""
        i, v = _ids(ids), _rows(vectors)
        return self._c.upsert(i, v, _metadata(metadata, len(i)))

    def bulk_load(self, ids: Any, vectors: Any, metadata: Optional[Sequence[Mapping[str, Any]]] = None) -> None:
        """Load ids not yet in the collection into one new segment with one index (all or nothing)."""
        i, v = _ids(ids), _rows(vectors)
        self._c.bulk_load(i, v, _metadata(metadata, len(i)))

    def delete(self, ids: Any) -> int:
        return self._c.delete(_ids(ids))

    def search(self, queries: Any, k: int = 10, ef: int = 64, nprobe: int = 16, rerank: int = -1,
               filter: Optional[Mapping[str, Any]] = None, strategy: str = "auto") -> Tuple[np.ndarray, np.ndarray]:
        """Batched k-NN search. Returns (ids, distances), each of shape (nq, k); a single query vector gives
        shape (k,). Smaller distance is closer for every metric (cosine/ip: minus the dot product)."""
        q = np.asarray(queries)
        single = q.ndim == 1
        f = json.dumps(filter) if filter is not None else None
        ids, dists = self._c.search(_rows(q), int(k), int(ef), int(nprobe), int(rerank), f, strategy)
        return (ids[0], dists[0]) if single else (ids, dists)

    def get(self, id: int) -> Tuple[np.ndarray, dict]:
        vec, md = self._c.get(int(id))
        return np.asarray(vec, dtype=np.float32), json.loads(md)

    def flush(self) -> None:
        self._c.flush()

    def compact(self, target_segments: int = 0) -> None:
        self._c.compact(int(target_segments))

    def optimize(self) -> None:
        """Merge everything into one segment (one graph)."""
        self._c.compact(1)

    def stats(self) -> dict:
        return json.loads(self._c.stats())

    def close(self) -> None:
        self._c.close()

    def __enter__(self) -> "Collection":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
