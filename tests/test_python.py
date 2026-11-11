"""Python bindings: create, bulk load, upsert, filtered batched search, reopen, threads.

    pip install . && pytest tests/test_python.py
"""

import threading

import numpy as np
import pytest

vdb = pytest.importorskip("vdb")


def clustered(n, dim, seed):
    rng = np.random.default_rng(seed)
    centers = np.random.default_rng(1).standard_normal((20, dim)).astype(np.float32)
    return (centers[rng.integers(0, 20, n)] + 0.3 * rng.standard_normal((n, dim))).astype(np.float32)


def exact(base, q, k):
    d = ((q[:, None, :] - base[None, :, :]) ** 2).sum(-1)
    return np.argsort(d, axis=1)[:, :k]


def test_round_trip(tmp_path):
    base = clustered(5000, 32, 2)
    queries = clustered(50, 32, 3)
    col = vdb.Collection.create(tmp_path / "c", dim=32, metric="l2",
                                index={"type": "hnsw", "M": 16, "ef_construction": 200},
                                schema={"year": "int64", "tag": "keyword"}, sync="none")
    md = [{"year": 2000 + i % 30, "tag": "a" if i % 2 else "b"} for i in range(len(base))]
    col.bulk_load(np.arange(len(base)), base[:], metadata=md)
    ids, dists = col.search(queries, k=10, ef=128)
    assert ids.shape == (50, 10) and dists.dtype == np.float32
    gt = exact(base, queries, 10)
    recall = np.mean([len(set(ids[i]) & set(gt[i])) / 10 for i in range(len(queries))])
    assert recall >= 0.95
    # One vector in, one row out.
    one_ids, _ = col.search(queries[0], k=3)
    assert one_ids.shape == (3,)
    # Filters.
    ids, _ = col.search(queries, k=5, filter={"and": [{"field": "year", "gte": 2020}, {"field": "tag", "eq": "a"}]})
    for i in ids.reshape(-1):
        if i != vdb.MISSING_ID:
            assert md[int(i)]["year"] >= 2020 and md[int(i)]["tag"] == "a"
    # Upserts through the write path, deletes, get.
    lsn = col.upsert(np.array([100000]), base[:1] + 1, metadata=[{"year": 1999}])
    assert lsn >= 1
    vec, meta = col.get(100000)
    assert np.allclose(vec, base[0] + 1) and meta == {"year": 1999}
    assert col.delete([100000, 5]) == 2
    with pytest.raises(KeyError):
        col.get(5)
    with pytest.raises(ValueError):
        col.search(np.zeros((1, 31), dtype=np.float32))
    col.optimize()
    stats = col.stats()
    assert stats["rows"] == len(base) - 1
    col.close()
    col = vdb.Collection.open(tmp_path / "c")
    assert col.stats()["rows"] == len(base) - 1
    col.close()


def test_threads_search_in_parallel(tmp_path):
    base = clustered(3000, 16, 4)
    with vdb.Collection.create(tmp_path / "t", dim=16, sync="none") as col:
        col.bulk_load(np.arange(len(base)), base)
        errors = []

        def worker():
            try:
                for _ in range(20):
                    ids, _ = col.search(base[:64], k=1, ef=32)
                    assert (ids[:, 0] == np.arange(64)).mean() > 0.95
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


def test_zero_copy_inputs_accept_views(tmp_path):
    base = clustered(1000, 8, 5)
    with vdb.Collection.create(tmp_path / "z", dim=8, index={"type": "flat"}, sync="none") as col:
        col.upsert(np.arange(500, dtype=np.int64), base[::2])  # strided view: copied once, then read in place
        ids, d = col.search(base[0], k=1)
        assert ids[0] == 0 and d[0] == pytest.approx(0, abs=1e-5)
