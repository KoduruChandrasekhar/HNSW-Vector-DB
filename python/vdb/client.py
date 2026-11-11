"""A thin HTTP client for vdb-server with the same methods as vdb.Collection.

    client = vdb.Client("http://localhost:8090")
    col = client.create_collection("docs", dim=384, metric="cosine", schema={"year": "int64"})
    col.upsert(ids, vectors, metadata=[{"year": 2021}, ...])
    ids, dists = col.search(query, k=5, filter={"field": "year", "gte": 2020})

Only the standard library is used (urllib), so it works anywhere NumPy does.
"""

from __future__ import annotations

import json
import struct
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


class VdbError(RuntimeError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.message = status, code, message


def encode_binary(ids: np.ndarray, vectors: np.ndarray) -> bytes:
    """The /upsert-binary and /bulk-load body: "VDBB", version, count, dim, ids, rows."""
    ids = np.ascontiguousarray(ids, dtype="<u8").reshape(-1)
    vectors = np.ascontiguousarray(vectors, dtype="<f4")
    if vectors.ndim != 2 or vectors.shape[0] != ids.shape[0]:
        raise ValueError("vectors must be (n, dim) with one id per row")
    return b"VDBB" + struct.pack("<III", 1, ids.shape[0], vectors.shape[1]) + ids.tobytes() + vectors.tobytes()


class Client:
    def __init__(self, url: str = "http://localhost:8090", timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: Any = None, raw: Optional[bytes] = None) -> Any:
        data, headers = None, {}
        if raw is not None:
            data, headers["Content-Type"] = raw, "application/octet-stream"
        elif body is not None:
            data, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = r.read()
                ctype = r.headers.get("Content-Type", "")
        except urllib.error.HTTPError as e:
            payload = e.read()
            try:
                err = json.loads(payload)["error"]
                raise VdbError(e.code, err.get("code", ""), err.get("message", "")) from None
            except (ValueError, KeyError):
                raise VdbError(e.code, "error", payload.decode(errors="replace")) from None
        return json.loads(payload) if ctype.startswith("application/json") else payload.decode()

    def health(self) -> bool:
        try:
            return self._request("GET", "/health").strip() == "OK"
        except (OSError, VdbError):
            return False

    def metrics(self) -> str:
        return self._request("GET", "/metrics")

    def list_collections(self) -> list:
        return self._request("GET", "/v1/collections")["collections"]

    def create_collection(self, name: str, dim: int, metric: str = "l2", index: Optional[Mapping[str, Any]] = None,
                          schema: Optional[Mapping[str, str]] = None, sync: str = "always",
                          **options: Any) -> "RemoteCollection":
        body = {"name": name, "dim": dim, "metric": metric, "sync": sync, **options}
        if index is not None:
            body["index"] = dict(index)
        if schema is not None:
            body["schema"] = dict(schema)
        self._request("POST", "/v1/collections", body)
        return RemoteCollection(self, name)

    def collection(self, name: str) -> "RemoteCollection":
        return RemoteCollection(self, name)

    def drop_collection(self, name: str) -> None:
        self._request("DELETE", f"/v1/collections/{name}")


class RemoteCollection:
    def __init__(self, client: Client, name: str):
        self.client, self.name = client, name
        self._base = f"/v1/collections/{name}"

    def stats(self) -> dict:
        return self.client._request("GET", self._base)

    def upsert(self, ids: Any, vectors: Any, metadata: Optional[Sequence[Mapping[str, Any]]] = None,
               batch: int = 1000) -> int:
        ids = np.asarray(ids, dtype=np.uint64).reshape(-1)
        vectors = np.asarray(vectors, dtype=np.float32)
        vectors = vectors.reshape(1, -1) if vectors.ndim == 1 else vectors
        last = 0
        for s in range(0, len(ids), batch):
            e = min(len(ids), s + batch)
            if metadata is None:
                r = self.client._request("POST", self._base + "/upsert-binary", raw=encode_binary(ids[s:e], vectors[s:e]))
            else:
                pts = [{"id": int(ids[i]), "vector": vectors[i].tolist(), "metadata": metadata[i]} for i in range(s, e)]
                r = self.client._request("POST", self._base + "/upsert", {"points": pts})
            last = r["last_lsn"]
        return last

    def bulk_load(self, ids: Any, vectors: Any) -> dict:
        return self.client._request("POST", self._base + "/bulk-load", raw=encode_binary(np.asarray(ids), np.asarray(vectors)))

    def delete(self, ids: Any) -> int:
        return self.client._request("POST", self._base + "/delete", {"ids": [int(i) for i in np.asarray(ids).reshape(-1)]})["deleted"]

    def get(self, id: int) -> Tuple[np.ndarray, dict]:
        r = self.client._request("GET", f"{self._base}/points/{int(id)}")
        return np.asarray(r["vector"], dtype=np.float32), r["metadata"]

    def _body(self, q: np.ndarray, k: int, ef: int, nprobe: int, rerank: int, filter: Any, strategy: str,
              include: Sequence[str], explain: bool) -> dict:
        body = {"vector": q.tolist(), "k": k, "ef": ef, "nprobe": nprobe, "rerank": rerank, "strategy": strategy}
        if filter is not None:
            body["filter"] = filter
        if include:
            body["include"] = list(include)
        if explain:
            body["explain"] = True
        return body

    def query(self, vector: Any, k: int = 10, ef: int = 64, nprobe: int = 16, rerank: int = -1,
              filter: Optional[Mapping[str, Any]] = None, strategy: str = "auto",
              include: Sequence[str] = ("metadata",), explain: bool = False) -> dict:
        """One search returning the raw response (ids, scores, metadata, optional explain)."""
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        return self.client._request("POST", self._base + "/search",
                                    self._body(q, k, ef, nprobe, rerank, filter, strategy, include, explain))

    def search(self, queries: Any, k: int = 10, ef: int = 64, nprobe: int = 16, rerank: int = -1,
               filter: Optional[Mapping[str, Any]] = None, strategy: str = "auto") -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(queries, dtype=np.float32)
        single = q.ndim == 1
        q = q.reshape(1, -1) if single else q
        ids = np.full((len(q), k), np.iinfo(np.uint64).max, dtype=np.uint64)
        dists = np.full((len(q), k), np.inf, dtype=np.float32)
        for s in range(0, len(q), 256):
            body = {"queries": [self._body(row, k, ef, nprobe, rerank, filter, strategy, (), False) for row in q[s:s + 256]]}
            res = self.client._request("POST", self._base + "/search-batch", body)["results"]
            for i, hits in enumerate(res):
                for j, h in enumerate(hits):
                    ids[s + i, j] = h["id"]
                    dists[s + i, j] = h["distance"]
        return (ids[0], dists[0]) if single else (ids, dists)

    def flush(self) -> None:
        self.client._request("POST", self._base + "/flush")

    def compact(self, target_segments: int = 0) -> None:
        self.client._request("POST", f"{self._base}/compact?target_segments={int(target_segments)}")

    def optimize(self) -> None:
        self.compact(1)
