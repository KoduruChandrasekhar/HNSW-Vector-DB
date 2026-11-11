"""HTTP API contract and error codes, with pytest + httpx against a running vdb-server.

    VDB_SERVER=build/release/vdb-server pytest tests/api_test.py
"""

import os
import socket
import struct
import subprocess
import tempfile
import time

import pytest

httpx = pytest.importorskip("httpx")
SERVER = os.environ.get("VDB_SERVER", "build/release/vdb-server")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def client():
    if not os.path.exists(SERVER):
        pytest.skip(f"no server binary at {SERVER}")
    data = tempfile.mkdtemp(prefix="vdb_api_")
    port = free_port()
    proc = subprocess.Popen([SERVER, "--data-dir", data, "--port", str(port), "--host", "127.0.0.1",
                             "--log-level", "warn", "--max-k", "100"])
    c = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30)
    for _ in range(200):
        try:
            if c.get("/health").status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.05)
    yield c
    proc.terminate()
    assert proc.wait(timeout=30) == 0  # graceful shutdown on SIGTERM


def test_create_upsert_search(client):
    r = client.post("/v1/collections", json={"name": "a", "dim": 3, "metric": "cosine",
                                             "schema": {"lang": "keyword"}, "sync": "always"})
    assert r.status_code == 201, r.text
    assert client.post("/v1/collections", json={"name": "a", "dim": 3}).status_code == 409
    pts = [{"id": i, "vector": [1, i, 0], "metadata": {"lang": "en" if i % 2 else "fr"}} for i in range(1, 21)]
    r = client.post("/v1/collections/a/upsert", json={"points": pts})
    assert r.json()["acknowledged"] == 20
    r = client.post("/v1/collections/a/search", json={"vector": [1, 5, 0], "k": 3, "include": ["metadata"]})
    body = r.json()
    assert body["results"][0]["id"] == 5
    assert body["results"][0]["score"] == pytest.approx(1.0, abs=1e-5)  # cosine similarity
    r = client.post("/v1/collections/a/search",
                    json={"vector": [1, 5, 0], "k": 3, "filter": {"field": "lang", "eq": "fr"}, "explain": True})
    assert all(h["id"] % 2 == 0 for h in r.json()["results"])
    assert r.json()["explain"]["segments"][0]["kind"] == "mutable"
    for strategy in ("prefilter", "graph", "postfilter", "auto"):
        r = client.post("/v1/collections/a/search",
                        json={"vector": [1, 5, 0], "k": 3, "filter": {"field": "lang", "eq": "fr"},
                              "strategy": strategy, "post_filter_alpha": 4, "ef": 32, "explain": True})
        assert r.status_code == 200, r.text
        assert [h["id"] for h in r.json()["results"]] == [6, 4, 8]
    r = client.post("/v1/collections/a/search-batch",
                    json={"queries": [{"vector": [1, 1, 0], "k": 1}, {"vector": [1, 9.5, 0], "k": 2}]})
    assert [[h["id"] for h in q] for q in r.json()["results"]] == [[1], [10, 9]]
    r = client.post("/v1/collections/a/search", json={"vector": [1, 2, 0], "k": 1, "include": ["vector"]})
    assert r.json()["results"][0]["vector"] == pytest.approx([5 ** -0.5, 2 * 5 ** -0.5, 0])  # cosine stores unit vectors
    full = client.get("/v1/collections/a").json()
    for key in ("config", "total_rows", "deleted_rows", "segment_list", "memory", "last_lsn", "flushed_lsn"):
        assert key in full
    assert full["config"]["schema"] == [{"name": "lang", "type": "keyword"}]


def test_errors(client):
    client.post("/v1/collections", json={"name": "e", "dim": 2})
    r = client.post("/v1/collections/e/search", json={"vector": [1, 2, 3]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_argument"
    assert client.post("/v1/collections/missing/search", json={"vector": [1, 2]}).status_code == 404
    assert client.get("/v1/collections/e/points/123").status_code == 404
    assert client.post("/v1/collections/e/search", json={"vector": [1, 2], "k": 101}).status_code == 400
    assert client.post("/v1/collections/e/search", json={"vector": [1, 2], "filter": {"x": 1}}).status_code == 400
    assert client.post("/v1/collections/e/upsert", content=b"nope").status_code == 400


def code_of(r):
    return r.status_code, r.json()["error"]["code"]


BAD_CREATE = [
    {"dim": 2},                                                  # no name
    {"name": "bad name!", "dim": 2},                             # name characters
    {"name": "x" * 65, "dim": 2},                                # name length
    {"name": "c1"},                                              # no dim
    {"name": "c1", "dim": 0},
    {"name": "c1", "dim": 16385},
    {"name": "c1", "dim": 2, "metric": "hamming"},
    {"name": "c1", "dim": 2, "index": {"type": "lsh"}},
    {"name": "c1", "dim": 2, "index": {"M": 1}},
    {"name": "c1", "dim": 2, "index": {"ef_construction": 0}},
    {"name": "c1", "dim": 10, "index": {"type": "pq", "pq_m": 3}},  # 3 does not divide 10
    {"name": "c1", "dim": 8, "index": {"type": "ivf_pq", "pq_m": 4, "nlist": 0}},
    {"name": "c1", "dim": 2, "sync": "sometimes"},
    {"name": "c1", "dim": 2, "flush_bytes": 100},
    {"name": "c1", "dim": 2, "tier_fanout": 1},
    {"name": "c1", "dim": 2, "schema": {"f": "complex"}},
    {"name": "c1", "dim": -2},
]


@pytest.mark.parametrize("body", BAD_CREATE)
def test_create_rejects(client, body):
    assert code_of(client.post("/v1/collections", json=body)) == (400, "invalid_argument")


def test_create_variants(client):
    for i, idx in enumerate(["flat", "hnsw", "hnsw_sq8", {"type": "pq", "m": 2}, {"type": "ivf_pq", "pq_m": 2, "nlist": 2}]):
        r = client.post("/v1/collections", json={"name": f"v{i}", "dim": 4, "metric": ["l2", "ip", "cosine"][i % 3],
                                                 "index": idx, "sync": ["always", "interval", "none"][i % 3],
                                                 "schema": [{"name": "n", "type": "int64"}, {"name": "t", "type": "tags"}]})
        assert r.status_code == 201, r.text
        pts = [{"id": j, "vector": [j, 1, 2, 3], "metadata": {"n": j, "t": ["x"] if j % 2 else []}} for j in range(40)]
        assert client.post(f"/v1/collections/v{i}/upsert", json={"points": pts}).status_code == 200
        assert client.post(f"/v1/collections/v{i}/flush").json() == {"status": "accepted"}
        r = client.post(f"/v1/collections/v{i}/search",
                        json={"vector": [5, 1, 2, 3], "k": 40, "filter": {"field": "t", "eq": "x"}})
        assert r.status_code == 200 and {h["id"] for h in r.json()["results"]} == set(range(1, 40, 2))
        assert client.delete(f"/v1/collections/v{i}").json() == {"dropped": f"v{i}"}


def test_every_error_branch(client):
    client.post("/v1/collections", json={"name": "eb", "dim": 2, "schema": {"lang": "keyword", "year": "int"}})
    s = "/v1/collections/eb/search"
    bad_searches = [
        {},                                               # no vector
        {"vector": "x"},
        {"vector": [1, "a"]},
        {"vector": [1, 2], "k": 0},
        {"vector": [1, 2], "k": -1},
        {"vector": [1, 2], "k": 1.5},
        {"vector": [1, 2], "ef": 10001},
        {"vector": [1, 2], "post_filter_alpha": 0},
        {"vector": [1, 2], "strategy": "magic"},
        {"vector": [1, 2], "strategy": 3},
        {"vector": [1, 2], "include": "metadata"},
        {"vector": [1, 2], "include": ["score"]},
        {"vector": [1, 2], "filter": {"field": "nope", "eq": 1}},
        {"vector": [1, 2], "filter": {"field": "lang", "gt": "a"}},     # range on a keyword
        {"vector": [1, 2], "filter": {"field": "year", "eq": "2020"}},  # string vs number
        {"vector": [1, 2], "filter": {"field": "lang", "in": []}},
        {"vector": [1, 2], "filter": {"field": "lang", "eq": "a", "ne": "b"}},
        {"vector": [1, 2], "filter": {"and": [{"field": "lang", "eq": "a"}], "or": []}},
        {"vector": [1, 2], "filter": {"field": "lang", "exists": 1}},
        {"vector": [1, 2], "filter": {"field": "lang", "regex": "a"}},
        {"vector": [1, 2], "filter": [1]},
    ]
    for body in bad_searches:
        assert code_of(client.post(s, json=body)) == (400, "invalid_argument"), body
    assert code_of(client.post(s, content=b"[1, 2]")) == (400, "invalid_argument")  # not an object
    assert code_of(client.post(s + "-batch", json={})) == (400, "invalid_argument")
    assert code_of(client.post(s + "-batch", json={"queries": [{"vector": [1, 2]}] * 1025})) == (400, "invalid_argument")
    assert code_of(client.post(s + "-batch", json={"queries": [{"vector": [1]}]})) == (400, "invalid_argument")
    assert code_of(client.post(s + "-batch", json={"queries": [3]})) == (400, "invalid_argument")

    u = "/v1/collections/eb/upsert"
    bad_upserts = [
        {},
        {"points": {}},
        {"points": [{"id": 1}]},
        {"points": [{"vector": [1, 2]}]},
        {"points": [{"id": -1, "vector": [1, 2]}]},
        {"points": [{"id": "1", "vector": [1, 2]}]},
        {"points": [{"id": 1, "vector": [1, 2, 3]}]},
        {"points": [{"id": 1, "vector": [1, 2], "metadata": {"unknown": 1}}]},
        {"points": [{"id": 1, "vector": [1, 2], "metadata": {"year": "x"}}]},
        {"points": [{"id": i, "vector": [1, 2]} for i in range(1025)]},
    ]
    for body in bad_upserts:
        assert code_of(client.post(u, json=body)) == (400, "invalid_argument"), body
    assert client.get("/v1/collections/eb").json()["rows"] == 0  # a rejected batch writes nothing

    d = "/v1/collections/eb/delete"
    for body in ({}, {"ids": 3}, {"ids": [-1]}, {"ids": ["a"]}, {"ids": list(range(1024 * 16 + 1))}):
        assert code_of(client.post(d, json=body)) == (400, "invalid_argument"), body
    assert client.post(d, json={"ids": [42]}).json() == {"deleted": 0}

    good = b"VDBB" + struct.pack("<III", 1, 1, 2) + struct.pack("<Q", 7) + struct.pack("<2f", 1, 2)
    bad_binary = [
        b"",
        b"XXXX" + good[4:],                                         # magic
        good[:4] + struct.pack("<I", 2) + good[8:],                 # version
        good[:12] + struct.pack("<I", 3) + good[16:],               # dim
        good[:-1],                                                  # truncated
        good + b"\0",                                               # trailing bytes
        good[:-4] + struct.pack("<f", float("nan")),                # NaN
        good[:-4] + struct.pack("<f", float("inf")),
    ]
    for route in ("upsert-binary", "bulk-load"):
        for body in bad_binary:
            assert code_of(client.post(f"/v1/collections/eb/{route}", content=body)) == (400, "invalid_argument")
    assert client.post("/v1/collections/eb/upsert-binary", content=good).json()["acknowledged"] == 1
    assert code_of(client.post("/v1/collections/eb/bulk-load", content=good)) == (409, "already_exists")
    twice = b"VDBB" + struct.pack("<III", 1, 2, 2) + struct.pack("<2Q", 8, 8) + struct.pack("<4f", 1, 2, 3, 4)
    assert code_of(client.post("/v1/collections/eb/bulk-load", content=twice)) == (400, "invalid_argument")
    assert client.get("/v1/collections/eb").json()["rows"] == 1  # neither load wrote anything
    assert client.get("/v1/collections/eb/points/7").json() == {"id": 7, "vector": [1.0, 2.0], "metadata": {}}
    assert client.get("/v1/collections/eb/points/abc").status_code == 404           # route needs digits
    assert code_of(client.get("/v1/collections/eb/points/99999999999999999999999")) == (400, "invalid_argument")
    assert code_of(client.post("/v1/collections/eb/compact?target_segments=x")) == (400, "invalid_argument")
    assert client.post("/v1/collections/eb/compact").json() == {"status": "accepted", "target_segments": 0}

    for method, path in [("GET", "/v1/collections/zz"), ("DELETE", "/v1/collections/zz"),
                         ("POST", "/v1/collections/zz/upsert"), ("POST", "/v1/collections/zz/upsert-binary"),
                         ("POST", "/v1/collections/zz/bulk-load"), ("POST", "/v1/collections/zz/delete"),
                         ("GET", "/v1/collections/zz/points/1"), ("POST", "/v1/collections/zz/search"),
                         ("POST", "/v1/collections/zz/search-batch"), ("POST", "/v1/collections/zz/flush"),
                         ("POST", "/v1/collections/zz/compact"), ("GET", "/v2/nothing")]:
        r = client.request(method, path, json={})
        assert code_of(r) == (404, "not_found"), (path, r.text)
    # A body on GET is never read by the HTTP library; the connection must not be reused with it.
    for _ in range(3):
        assert client.request("GET", "/v1/collections/eb", content=b'{"x": 1}').status_code == 200
        assert client.get("/v1/collections/eb").status_code == 200
    assert client.get("/health").text == "OK\n"
    assert client.get("/metrics").headers["content-type"].startswith("text/plain")


def test_python_http_client(client):
    """vdb.Client: every method against the running server."""
    np = pytest.importorskip("numpy")
    vdb_client = pytest.importorskip("vdb.client")
    c = vdb_client.Client(str(client.base_url))
    assert c.health()
    col = c.create_collection("pyc", dim=4, metric="l2", index={"type": "hnsw", "M": 8},
                              schema={"tag": "keyword"}, sync="always")
    with pytest.raises(vdb_client.VdbError) as e:
        c.create_collection("pyc", dim=4)
    assert e.value.status == 409 and e.value.code == "already_exists"
    vecs = np.eye(4, dtype=np.float32).repeat(5, axis=0) + np.arange(20, dtype=np.float32)[:, None] * 0.01
    # JSON path (with metadata) and binary path (without).
    col.upsert(np.arange(10), vecs[:10], metadata=[{"tag": "a" if i % 2 else "b"} for i in range(10)])
    lsn = col.upsert(np.arange(10, 15), vecs[10:15])
    assert lsn >= 15
    assert col.bulk_load(np.arange(100, 105), vecs[15:20])["loaded"] == 5
    ids, dists = col.search(vecs[:3], k=2)
    assert ids.shape == (3, 2) and list(ids[:, 0]) == [0, 1, 2]
    one_ids, _ = col.search(vecs[7], k=1, filter={"field": "tag", "eq": "a"})
    assert int(one_ids[0]) == 7
    r = col.query(vecs[1], k=1, include=["metadata"], explain=True)
    assert r["results"][0]["metadata"] == {"tag": "a"} and "explain" in r
    v, md = col.get(3)
    assert np.allclose(v, vecs[3]) and md == {"tag": "a"}
    assert col.delete([3, 999]) == 1
    with pytest.raises(vdb_client.VdbError) as e:
        col.get(3)
    assert e.value.status == 404
    col.flush()
    col.compact(1)
    col.optimize()
    assert col.stats()["rows"] == 19
    assert any(x["name"] == "pyc" for x in c.list_collections())
    assert "vdb_collection_rows" in c.metrics()
    c.drop_collection("pyc")
    assert all(x["name"] != "pyc" for x in c.list_collections())


def test_config_file(tmp_path):
    """Options can come from a TOML config file (--config)."""
    if not os.path.exists(SERVER):
        pytest.skip(f"no server binary at {SERVER}")
    port = free_port()
    cfg = tmp_path / "server.toml"
    cfg.write_text(f'data-dir = "{tmp_path / "data"}"\nport = {port}\nhost = "127.0.0.1"\n'
                   'threads = 4\nlog-level = "warn"\nmax-k = 7\n')
    proc = subprocess.Popen([SERVER, "--config", str(cfg)])
    try:
        c = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30)
        for _ in range(200):
            try:
                if c.get("/health").status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.05)
        assert c.post("/v1/collections", json={"name": "x", "dim": 2}).status_code == 201
        assert c.post("/v1/collections/x/search", json={"vector": [1, 0], "k": 8}).status_code == 400  # max-k = 7
        assert c.post("/v1/collections/x/search", json={"vector": [1, 0], "k": 7}).status_code == 200
        assert (tmp_path / "data" / "collections" / "x" / "CURRENT").exists()
    finally:
        proc.terminate()
        assert proc.wait(timeout=30) == 0


def test_binary_and_metrics(client):
    client.post("/v1/collections", json={"name": "b", "dim": 2, "sync": "none"})
    ids = list(range(100))
    body = b"VDBB" + struct.pack("<III", 1, len(ids), 2) + struct.pack(f"<{len(ids)}Q", *ids)
    body += struct.pack(f"<{2 * len(ids)}f", *[float(x) for i in ids for x in (i, 1)])
    assert client.post("/v1/collections/b/bulk-load", content=body).status_code == 200
    assert client.get("/v1/collections/b").json()["rows"] == 100
    assert client.post("/v1/collections/b/compact?target_segments=1").status_code == 202
    m = client.get("/metrics").text
    assert 'vdb_collection_rows{collection="b"} 100' in m
    assert "vdb_request_duration_seconds_bucket" in m
    assert client.delete("/v1/collections/b").status_code == 200
