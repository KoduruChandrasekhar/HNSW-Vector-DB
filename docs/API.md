# vdb HTTP API

`vdb-server` serves JSON over HTTP/1.1 (keep-alive, `TCP_NODELAY`). Every route below is covered by
`tests/api_test.py` (pytest + httpx against the real binary) and `tests/server_test.cc`.

```bash
vdb-server --data-dir ./data --port 8090
```

| Method | Path | Purpose | Success |
|---|---|---|---|
| GET | `/health` | liveness | 200 `OK\n` (text) |
| GET | `/metrics` | Prometheus text format | 200 |
| POST | `/v1/collections` | create a collection | 201 |
| GET | `/v1/collections` | list collections (short stats) | 200 |
| GET | `/v1/collections/{name}` | full stats and config | 200 |
| DELETE | `/v1/collections/{name}` | drop a collection and its files | 200 |
| POST | `/v1/collections/{name}/upsert` | insert/replace points (JSON, with metadata) | 200 |
| POST | `/v1/collections/{name}/upsert-binary` | insert/replace points (binary, no metadata) | 200 |
| POST | `/v1/collections/{name}/bulk-load` | build one indexed segment directly, skipping the log | 200 |
| POST | `/v1/collections/{name}/delete` | delete by id | 200 |
| GET | `/v1/collections/{name}/points/{id}` | fetch one point | 200 |
| POST | `/v1/collections/{name}/search` | k-NN search, optional filter | 200 |
| POST | `/v1/collections/{name}/search-batch` | several searches in one request | 200 |
| POST | `/v1/collections/{name}/flush` | ask for the mutable segment to be flushed | 202 |
| POST | `/v1/collections/{name}/compact[?target_segments=N]` | ask for compaction | 202 |

Collection names match `[A-Za-z0-9_-]{1,64}`. Ids are unsigned 64-bit integers. Request bodies must be
JSON objects unless noted.

## Errors

Every error is JSON:

```json
{"error": {"code": "invalid_argument", "message": "wrong dimension: got 3, expected 2"}}
```

| HTTP | `code` | When |
|---|---|---|
| 400 | `invalid_argument` | malformed JSON, wrong types, wrong dimension, unknown field/operator, limits exceeded, bad binary body, unparseable request |
| 404 | `not_found` | unknown collection, unknown point id, unknown route |
| 409 | `already_exists` | creating a collection that exists; bulk-loading an id that exists |
| 413 | `payload_too_large` | body larger than `--max-body-mb` |
| 503 | `unavailable` | search queue full (sent with `Retry-After: 1`) |
| 503 | `resource_exhausted`, `failed_precondition` | the engine refused the write (e.g. collection closing) |
| 500 | `internal`, `io_error`, `corruption`, … | everything else; the message says what failed |

A batch is validated completely before anything is written: a 400 on an upsert means no point in it was
stored.

## Create a collection

`POST /v1/collections`

```json
{
  "name": "docs",
  "dim": 384,
  "metric": "cosine",
  "index": {"type": "hnsw", "M": 16, "ef_construction": 200},
  "schema": {"lang": "keyword", "year": "int64", "score": "float64", "tags": "keyword_list", "draft": "bool"},
  "sync": "always"
}
```

| Field | Default | Notes |
|---|---|---|
| `name` | required | `[A-Za-z0-9_-]`, 1–64 chars |
| `dim` | required | 1..16384 |
| `metric` | `l2` | `l2` (`euclidean`), `ip` (`inner_product`, `dot`), `cosine` (`angular`) |
| `index` | `{"type": "hnsw"}` | object, or just the type as a string |
| `index.type` | `hnsw` | `flat`, `hnsw`, `hnsw_sq8` (`sq8`), `pq` (`pq_flat`), `ivf_pq` (`ivfpq`) |
| `index.M` | 16 | HNSW links per node (2..256); layer 0 uses 2M |
| `index.ef_construction` | 200 | > 0 |
| `index.pq_m` (alias `m`) | 16 | PQ sub-quantizers; must divide `dim` for `pq`/`ivf_pq` |
| `index.nlist` | 1024 | IVF lists, > 0 for `ivf_pq` |
| `index.max_train_points_per_centroid` | 256 | k-means sample size per centroid |
| `index.seed` | 42 | makes builds reproducible |
| `schema` | `{}` | `{"field": "type"}` or `[{"name": ..., "type": ...}]` to fix the order. Types: `int64` (`int`), `float64` (`float`, `double`), `keyword` (`string`), `keyword_list` (`tags`), `bool` |
| `sync` | `always` | `always`: fsync (group commit) before acknowledging; `interval`: background fsync every `sync_interval_ms`; `none`: leave it to the OS (benchmarks only) |
| `sync_interval_ms` | 100 | for `interval` |
| `flush_bytes` | 8388608 | flush the mutable segment at this size (≥ 4096) |
| `flush_age_ms` | 300000 | …or when its oldest row is this old (0 = no age limit) |
| `tier_base_rows` | 20000 | size-tiered compaction: smallest tier |
| `tier_fanout` | 4 | segments per tier before merging (≥ 2) |
| `merge_into_largest_ratio` | 0.2 | a tier this small relative to the largest segment is merged into it by extending its graph |
| `delete_compaction_ratio` | 0.2 | rewrite a segment once this fraction of its rows is deleted |

Response `201`:

```json
{"name": "docs", "config": {"dim": 384, "metric": "cosine", "index": {"type": "hnsw", "M": 16, ...}, ...}}
```

The `config` echo always contains every option with its effective value.

## List, inspect, drop

`GET /v1/collections` → `{"collections": [{"name": "docs", "rows": 1000, "segments": 2, "disk_bytes": 1638400}]}`

`GET /v1/collections/{name}` adds:

```json
{
  "name": "docs", "rows": 990, "segments": 2, "disk_bytes": 1638400,
  "config": {...},
  "total_rows": 1000, "deleted_rows": 10, "deleted_ratio": 0.01,
  "segment_list": [
    {"id": 7, "kind": "immutable", "index": "hnsw", "rows": 1000, "deleted": 10, "lsn": 1000,
     "memory_bytes": 1700000, "disk_bytes": 1638400},
    {"id": 8, "kind": "mutable", "index": "flat", "rows": 0, "deleted": 0, "lsn": 0, "memory_bytes": 0, "disk_bytes": 0}
  ],
  "memory": {"vectors": 1540096, "graph": 160000, "codes": 0, "id_map": 32000, "metadata": 12000, "total": 1744096},
  "last_lsn": 1010, "flushed_lsn": 1000, "flushes": 1, "compactions": 0, "last_recovery_seconds": 0.004
}
```

`rows` counts live points. `last_lsn - flushed_lsn` is the part of the log a crash would replay.

`DELETE /v1/collections/{name}` → `{"dropped": "docs"}`. 404 if it does not exist.

## Upsert (JSON)

`POST /v1/collections/{name}/upsert`

```json
{"points": [
  {"id": 1, "vector": [0.1, 0.2, 0.3], "metadata": {"lang": "en", "year": 2021, "tags": ["a", "b"]}},
  {"id": 2, "vector": [0.3, 0.2, 0.1]}
]}
```

→ `{"acknowledged": 2, "last_lsn": 1012}`

- At most 1024 points per request.
- Upserting an existing id replaces its vector **and** metadata; omitted `metadata` means none.
- Metadata keys must be in the schema and values must match its types.
- With `sync: always` the response is sent after the log records are on disk.
- Each point is its own log record: a crash *before* the response can keep a prefix of the batch. Once
  the response is sent, all of it survives.
- `cosine` collections store unit-length vectors; reads return the normalized vector.

## Binary bodies (`upsert-binary`, `bulk-load`)

JSON float parsing dominates ingest cost, so both routes take a little-endian binary body:

```
"VDBB" | u32 version = 1 | u32 count | u32 dim | count × u64 ids | count × dim × f32
```

The body size must match the header exactly; NaN and infinities are rejected. `vdb.client.encode_binary`
builds it from NumPy arrays.

- `upsert-binary` → `{"acknowledged": n, "last_lsn": lsn}`. Same semantics as JSON upsert, without metadata.
- `bulk-load` → `{"loaded": n, "segments": s}`. Builds an indexed immutable segment in one step and
  installs it with one manifest edit: all of the rows appear at once or none do. It accepts only new
  ids: an id already in the collection is `409 already_exists`, and an id repeated within the body is
  `400 invalid_argument`. Use it for the initial load; it is much faster than streaming upserts through
  the log. Replace existing points with `upsert-binary`.

## Delete

`POST /v1/collections/{name}/delete` with `{"ids": [1, 2, 999]}` → `{"deleted": 2}` (ids that were not
present are not counted). At most 16384 ids per request. Deletes are logged like upserts and take effect
for searches immediately; the space comes back at compaction.

## Get one point

`GET /v1/collections/{name}/points/{id}` → `{"id": 1, "vector": [...], "metadata": {"lang": "en"}}`.
404 with code `not_found` if the id is absent or deleted; 400 if the id does not fit in 64 bits.

## Search

`POST /v1/collections/{name}/search`

```json
{
  "vector": [0.1, 0.2, 0.3],
  "k": 10,
  "ef": 64,
  "nprobe": 16,
  "rerank": -1,
  "filter": {"and": [{"field": "lang", "eq": "en"}, {"field": "year", "gte": 2020}]},
  "strategy": "auto",
  "prefilter_threshold": 10000,
  "post_filter_alpha": 10,
  "include": ["metadata", "vector"],
  "explain": true
}
```

| Field | Default | Notes |
|---|---|---|
| `vector` | required | length must equal `dim` |
| `k` | 10 | 1..`--max-k` (default 1000) |
| `ef` | 64 | HNSW beam width, ≤ 10000; `max(ef, k)` is used |
| `nprobe` | 16 | IVF lists to scan |
| `rerank` | -1 | candidates re-scored with full vectors for SQ8/PQ indexes; -1 = automatic (`max(ef, k)` for `hnsw_sq8`, `max(10k, 100)` for `pq`/`ivf_pq`), 0 = off |
| `filter` | none | see below |
| `strategy` | `auto` | `auto`, `prefilter` (alias `exact`), `graph`, `postfilter` |
| `prefilter_threshold` | 10000 | `auto` scans the matches exactly when there are at most this many |
| `post_filter_alpha` | 10 | `postfilter` searches `k × alpha` candidates, then filters (1..1000) |
| `include` | `[]` | any of `"metadata"`, `"vector"` |
| `explain` | false | adds the per-segment plan |

Response:

```json
{
  "results": [{"id": 5, "score": 0.98, "distance": 0.02, "metadata": {"lang": "en"}}],
  "took_ms": 0.41,
  "explain": {
    "segments": [{"segment": 7, "kind": "immutable", "index": "hnsw", "strategy": "graph", "rows": 1000,
                  "live_rows": 990, "matches": 412, "selectivity": 0.416, "ef": 64,
                  "distance_computations": 820, "results": 10, "micros": 180}],
    "distance_computations": 1300,
    "total_micros": 230
  }
}
```

`distance` follows the "smaller is closer" convention for every metric: squared L2 for `l2`, `-dot`
for `ip`, `-cos` for `cosine` (vectors are stored normalized, so it is `-dot` of unit vectors). `score`
is the same number for `l2`, and the similarity (`dot`, `cos`) for `ip` and `cosine`.
Results are sorted by distance; each id appears once even when it has older versions in other segments.

Searches slower than `--slow-query-ms` are logged with their explain output.

### Strategies

Every segment is planned separately, using its own match count from the metadata columns:

- `prefilter`: score every matching row exactly. Best when few rows match.
- `graph`: walk the HNSW graph, only admitting matching rows to the results (non-matching nodes are
  still traversed). Best for broad filters.
- `postfilter`: unfiltered search for `k × post_filter_alpha` candidates, then drop non-matches. Good when
  the filter is uncorrelated with the vectors and matches most rows.
- `auto` picks exact when matches ≤ `prefilter_threshold` or when scanning them is cheaper than a graph
  walk at the current `ef`, otherwise `graph`. If a segment returns fewer than `k` matches, it falls back
  to an exact scan of that segment.

### Filters

A condition is `{"field": NAME, OP: VALUE}` with exactly one operator; combine with
`{"and": [...]}`, `{"or": [...]}`, `{"not": {...}}`.

| Operator | Field types | Meaning |
|---|---|---|
| `eq` | all | equal; on `keyword_list`, contains the value |
| `ne` | all | present and not equal; on `keyword_list`, does not contain it |
| `in` | all but `bool` | equal to one of a non-empty list (≤ 1024 items) |
| `lt`, `lte`, `gt`, `gte` | `int64`, `float64` | numeric range |
| `exists` | all | `true`: field is set; `false`: it is not |

Numbers must be finite. Filters are limited to depth 32 and 4096 nodes. Unknown fields, unknown
operators, type mismatches and extra keys are 400s.

## Search batch

`POST /v1/collections/{name}/search-batch` with `{"queries": [<search body>, ...]}` (≤ 1024) →
`{"results": [[hits of query 1], [hits of query 2], ...]}`. Each query takes every search field; the
whole batch fails with the first query's error.

## Flush and compact

Both return `202` immediately; the work runs on the collection's background threads.

- `POST /flush` → `{"status": "accepted"}`: turn the mutable segment into an indexed segment now.
- `POST /compact` → `{"status": "accepted", "target_segments": 0}`. With `?target_segments=N` it merges
  until at most N segments remain (`N = 1` is "optimize"); without it the normal policy runs (deleted
  ratio, size tiers, merge into largest). Flushing and compaction also happen on their own.

Watch `flushes`, `compactions` and `segments` in the collection stats to see them finish.

## Back-pressure and limits

- Searches in flight above `--max-inflight-searches` (default 64) are answered at once with
  `503 unavailable` and `Retry-After: 1`. Writes do not go through this queue.
- Bodies above `--max-body-mb` (default 1024) get `413 payload_too_large`.
- The HTTP library uses one thread per open keep-alive connection; `--threads` (default 64) caps how many
  clients are served at once. Set it above the number of concurrent clients.
- A GET or HEAD request that carries a body is answered, and then its connection is closed (the HTTP
  library never reads such bodies, so the bytes cannot stay on the connection).

## Metrics

`GET /metrics` (`text/plain; version=0.0.4`):

| Metric | Type | Labels |
|---|---|---|
| `vdb_requests_total` | counter | `route`, `code` |
| `vdb_request_duration_seconds` | histogram | `route` |
| `vdb_search_distance_computations` | histogram | |
| `vdb_wal_bytes_total`, `vdb_wal_records_total` | counter | |
| `vdb_wal_fsync_duration_seconds` | histogram | |
| `vdb_flush_duration_seconds`, `vdb_compaction_duration_seconds` | histogram | |
| `vdb_collection_rows`, `vdb_segments`, `vdb_deleted_ratio` | gauge | `collection` |
| `vdb_memory_bytes` | gauge | `collection`, `component` (`vectors`, `graph`, `codes`, `id_map`, `metadata`) |

Route labels: `health`, `metrics`, `create_collection`, `list_collections`, `get_collection`,
`drop_collection`, `upsert`, `upsert_binary`, `bulk_load`, `delete`, `get_point`, `search`,
`search_batch`, `flush`, `compact`.

## Server options

Every flag can also come from a TOML/INI file with `--config server.toml` (keys without the dashes).

| Flag | Default | |
|---|---|---|
| `--data-dir` | `./data` | collections live in `<data-dir>/collections/<name>/` |
| `--host` | `0.0.0.0` | |
| `--port` | 8090 | |
| `--threads` | 64 | HTTP worker threads (one per keep-alive connection) |
| `--index-threads` | 0 (all cores) | index builds and batch search |
| `--log-level` | `info` | `trace`, `debug`, `info`, `warn`, `error` |
| `--max-body-mb` | 1024 | |
| `--max-inflight-searches` | 64 | |
| `--max-k` | 1000 | |
| `--slow-query-ms` | 50 | |
| `--verify-checksums` | true | check segment CRCs when opening collections |

```toml
data-dir = "/var/lib/vdb"
port = 8090
threads = 128
log-level = "warn"
```

On start the server recovers every collection (replaying the log tail) before it binds the port, and logs
how long that took. `SIGTERM`/`SIGINT` stop accepting connections, finish in-flight requests, sync the
logs and exit 0.

## Python client

`vdb.Client` (standard library only, plus NumPy) wraps every route:

```python
from vdb import Client
c = Client("http://localhost:8090")
col = c.create_collection("docs", dim=384, metric="cosine", schema={"year": "int64"})
col.upsert(ids, vectors, metadata=[{"year": 2021}, ...])  # JSON with metadata, else binary
col.bulk_load(ids, vectors)
ids, dists = col.search(queries, k=10, filter={"field": "year", "gte": 2020})  # (n, k) arrays
r = col.query(vector, k=5, include=["metadata"], explain=True)                 # raw JSON response
vec, meta = col.get(42); col.delete([1, 2]); col.flush(); col.compact(); col.optimize(); col.stats()
c.list_collections(); c.metrics(); c.health(); c.drop_collection("docs")
```

Errors raise `vdb.client.VdbError` with `.status`, `.code` and `.message`.
