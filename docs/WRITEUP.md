# What I learned building a vector database in C++

Every number here comes from `bench/results/*.json`, measured on one laptop:
Intel i7-1165G7 (4 cores / 8 threads, AVX-512), 8 GB RAM given to WSL2 (Linux 6.6), GCC 15.2, `-O3
-march=native`. Single-threaded search unless noted. SIFT1M is 1M × 128-d (L2); GloVe-100 is 1.18M ×
100-d (cosine). Each QPS figure is the median of three runs.

## Resume lines

- Built a vector database from scratch in C++20 with HNSW, SQ8 and product-quantization indexes, reaching
  recall@10 of **0.974 at 3,391 queries/sec** on SIFT1M single-threaded: **30% faster than hnswlib**
  (2,601 QPS) at the same recall and graph parameters, and 48% faster than FAISS HNSW (2,285 QPS at
  0.978).
- Designed an LSM-style storage engine with a CRC-checksummed write-ahead log, group commit and crash
  recovery; a **1,000-run kill -9 test lost zero acknowledged writes**, and group commit raised
  fsync-per-write ingest from 129 to 4,608 vectors/sec (177k/sec in batches of 100).
- Implemented filtered vector search with a per-segment, selectivity-based planner over roaring bitmaps,
  keeping **recall@10 ≥ 0.98 from 0.1% to 100% selectivity**, and used it for retrieval-augmented
  question answering over 40,000 Wikipedia passages, where approximate search matched exact search's
  hit rate at 1/18 of the latency.

## The nineteen questions

### 1. Why approximate search at all?

Exact search on SIFT1M reads all 512 MB of vectors for every query: **22 QPS** on one core (41 ms p50),
77 QPS on all 8 threads. One core streams about 11 GB/s through the AVX-512 kernel, which is roughly the
machine's memory bandwidth, so more arithmetic speed would not help. HNSW at recall@10 0.993 runs at
1,806 QPS, 80× faster.

Exact is still the right choice when the candidate set is small. The planner scans exactly when a filter
leaves few matches: at 0.1% selectivity an exact scan of the ~1,000 matching rows runs at **115k QPS with
recall 1.0**, while walking the graph for those rows manages 20 QPS. The mutable segment is always
scanned exactly too; it is small by design.

### 2. How does HNSW work, and why the layers?

It is a skip list built from graphs. Every node gets a random level (geometric, `1/ln M`); level 0
holds every node with up to 2M links, and each level above holds an exponentially thinner sample with M
links. A search starts at the top-level entry point, walks greedily to the closest node on that layer,
drops a layer, and repeats. On layer 0 it runs a best-first beam search of width `ef`. The upper layers
give long jumps across the data set, so layer 0 starts close to the answer and only has to refine.
Inserting a node is that same search at `ef_construction`, followed by linking the node to the neighbours
the heuristic selects on each of its layers, and pruning those neighbours' lists back to size.

Levels are derived from `SplitMix64(seed ^ id)`, which makes builds reproducible regardless of thread
timing.

### 3. What do M, ef_construction and ef_search trade off?

SIFT1M, recall@10:

| M | build (8 threads) | bytes/vector | first ef with recall ≥ 0.95 | best recall seen |
|---|---|---|---|---|
| 8 | 115 s | 613 | ef 160: 0.967 at 2,494 QPS | 0.9957 |
| 16 | 210 s | 676 | ef 80: 0.974 at 3,391 QPS | 0.9990 |
| 32 | 289 s | 804 | ef 40: 0.959 at 3,758 QPS | 0.9992 |

- **M** buys recall per unit of `ef` and raises the ceiling, at the cost of memory (each link is 4 bytes,
  layer 0 has 2M of them) and build time. M=8 never quite reaches the recall of M=16 or M=32.
- **ef_construction** is paid once: 102 s / 210 s / 338 s for 100 / 200 / 400. Beyond 200 the graph
  barely improves on SIFT.
- **ef_search** is the per-query dial: on M=16, ef 80 → 0.974 at 3,391 QPS, ef 160 → 0.993 at 1,806 QPS.
  Doubling ef roughly halves QPS.

### 4. Why the neighbour-selection heuristic instead of the M closest?

Within a cluster the M closest neighbours of a node are all close to each other too, so the links would
all point the same way and the cluster would barely connect to the rest of the graph. The heuristic keeps
a candidate only if it is closer to the new node than to any neighbour already kept, which spreads the
links in different directions and keeps bridges between clusters. (Like hnswlib, it does not top the
list back up with pruned candidates; with fewer than M candidates it keeps them all.)

Even with the heuristic, pruning a neighbour's full list can drop the only link into some node. After a
build, `repair_connectivity` runs a BFS from the entry point on layer 0 and links every node it cannot
reach from its nearest reachable node with a free slot. This is counted as `repaired_links` in the results: 362 on SIFT1M, 0.04% of
nodes. Without the pass, those nodes could never be returned.

### 5. HNSW vs IVF vs PQ: when would you use each?

SIFT1M:

| Index | bytes/vector in RAM | build | recall@10 | QPS |
|---|---|---|---|---|
| Flat (exact) | 512 | 0 | 0.999 | 22 |
| HNSW M16, float vectors | 676 | 210 s | 0.974 (ef 80) | 3,391 |
| HNSW M16, SQ8 codes + float re-rank | 292 | 106 s | 0.975 (ef 80) | 6,153 |
| IVF-PQ nlist 4096, m 16, nprobe 64, R=100 | 89 | 142 s | 0.955 | 1,763 |
| IVF-PQ nlist 4096, m 16, nprobe 256, R=1000 | 89 | 142 s | 0.998 | 428 |

SQ8 and IVF-PQ keep the float vectors memory-mapped on disk for re-ranking, so only the touched rows come
into the page cache. IVF-PQ's 89 bytes are about 22 bytes per vector of codes and list entries, plus
the precomputed distance tables (question 6): 67 MB, or 67 bytes per vector at 1M rows.

- **HNSW (float)**: best latency and recall when the vectors fit in RAM.
- **HNSW + SQ8**: the default I would ship. It needs 43% of the memory of the float graph and is *faster*,
  because a 128-byte code is a quarter of the memory traffic per distance. Re-ranking the beam with float
  vectors recovers the recall.
- **IVF-PQ**: when memory is the constraint (7.6× smaller than HNSW here), or for very large collections.
  It is slower at equal recall and needs training, but recall is tunable after the build with `nprobe`
  and `R`.
- **Flat**: small collections, heavily filtered queries, the mutable segment, and ground truth.

### 6. How does PQ compute distances without decompressing?

The vector is split into m sub-vectors, each replaced by the index of its nearest of 256 centroids
(one byte). At query time, for each sub-space, I compute the distance from the query's sub-vector to all
256 centroids once: an m × 256 table. A coded vector's distance is then m table lookups and additions,
with no floating-point vector work per row.

IVF-PQ encodes the *residual* from the list centroid. For L2,
`‖q − c − r‖² = ‖q − c‖² + ‖r‖² + 2⟨c, r⟩ − 2⟨q, r⟩`. The middle terms do not depend on the query, so
I precompute them per list at build time (the FAISS trick). A query then needs only one `⟨q, codebook⟩`
table, shared by every list it probes. That took nprobe-64 search from 763 to 1,763 QPS.

### 7. Why re-rank, and how do you pick R?

PQ distances are approximate enough to reorder close neighbours, and recall@10 suffers. Re-ranking takes
the best R candidates by PQ distance and re-scores them with the exact float vectors. IVF-PQ, m 16,
nprobe 64:

| R | recall@10 | QPS |
|---|---|---|
| 0 (PQ only) | 0.561 | 1,908 |
| 100 | 0.955 | 1,763 |
| 1000 | 0.969 | 993 |

R = 100 costs 8% QPS and lifts recall from 0.56 to 0.96. After that, recall is limited by `nprobe`, not R:
nprobe 256 with R = 1000 reaches 0.998. Coarser codes need a larger R: with m = 8, R = 100 reaches only
0.84 and R = 1000 reaches 0.97. I pick R as the smallest value past which recall stops rising at the
`nprobe` in use. `rerank: -1` uses `max(10k, 100)` for IVF-PQ, and the whole beam (`max(ef, k)`) for
HNSW-SQ8, where the beam is small and already holds the right neighbours.

### 8. How do you guarantee an acknowledged write survives a crash?

- Every write is appended to the WAL as a record with a length and a CRC32C. With `sync: always`, the
  response is sent only after `fdatasync` returns.
- New files (log segments, index segments, manifests) are written and fsynced, then the *directory* is
  fsynced, so the file's name survives too.
- The manifest is replaced atomically: write `MANIFEST-n`, fsync it, write a temporary `CURRENT`, fsync,
  `rename` over `CURRENT`, then fsync the directory.
- On recovery the log is replayed from the last flushed LSN. A record with a bad CRC or a short length at
  the tail is a torn write from the crash: the log is truncated there. The same thing in the *middle* of
  the log is corruption, and opening fails loudly.
- A segment records the highest LSN it contains, so replay skips records that a flushed segment already
  holds, and a crash between "segment written" and "log truncated" cannot apply a write twice.

The crash harness (`tools/crash_test.py`) runs a writer, `kill -9`s it at a random moment, reopens, and
checks that every acknowledged id is present with the right vector: **1,000 runs, 0 failures**.

### 9. What does a kill -9 test not cover?

Power loss. After `kill -9` the kernel still holds the page cache, so data that was written but never
fsynced still reaches the disk. A missing fsync would pass every kill -9 test.

`FaultInjectionEnv` covers that. It wraps the file system API and records which bytes were synced and
which file creations and renames were followed by a directory sync. "Crashing" it drops every unsynced
byte, optionally tears the last write, and undoes un-synced creates and renames. It can also fail the
k-th operation to exercise error paths. The recovery tests crash at every point in flush and compaction
this way. Still not covered: drives that lie about flushes, and bit rot inside a segment between CRC
checks (`--verify-checksums` covers that at open time).

### 10. What is group commit and what did it buy you?

With fsync on every write, the writer waits for the disk, and with many writers each waits for the
others' fsyncs too. Group commit makes the first waiting writer the leader: it appends every queued
record and fsyncs once for all of them. Ingest on ext4, `sync: always`, 4,000 vectors:

| Writers × batch | no group commit | group commit |
|---|---|---|
| 1 × 1 | 141 /s | 129 /s |
| 64 × 1 | 129 /s (p50 500 ms) | **4,608 /s** (p50 14 ms) |
| 64 × 100 | 13,807 /s | **177,547 /s** |
| `sync: interval` / `none`, 64 × 100 | | 538k / 535k /s |

A single writer gains nothing; it still waits for each fsync. Concurrent writers gain 36× (single
points) to 13× (batches), and their tail latency drops from half a second to 20 ms.

### 11. How do deletes work in a graph index, and why compact?

Removing a node from an HNSW graph means repairing every link into it, which is expensive and racy.
Instead a delete sets a bit: an atomic bit array per segment in memory, saved as a roaring bitmap
(`deletes-<n>.roar`) when the segment's delete set is persisted. Search still walks through deleted
nodes, since they keep the graph connected, but never returns them.

The cost grows with the share of deleted rows. On 200k SIFT rows at ef 64:

| Deleted | QPS | after compaction |
|---|---|---|
| 0% | 8,415 | 9,299 |
| 10% | 7,358 | 8,313 |
| 30% | 6,463 | 9,427 |
| 50% | 5,126 | 10,160 |

Half the work is wasted on rows that cannot be returned. Recall against the live rows *rises* as deletes
pile up (0.980 → 0.995), because the beam effectively explores more. Compaction rewrites a segment once
20% of it is deleted (`delete_compaction_ratio`), and restores the speed.

### 12. How do queries keep running during flush and compaction?

The collection's state is an immutable `Snapshot` (the mutable segment, the list of immutable segments,
the id map) held by a `shared_ptr`. A query copies the pointer under a short lock, then runs without
locks. Flush and compaction build the new segments off to the side, then swap a new snapshot in. Old
segments are freed when the last query holding them drops its reference. Deletes that land while a
compaction is running are replayed onto the new segment in a short exclusive section before the swap, so
none are lost.

The test: 100 clients searching through the HTTP server while a 200k-row bulk load runs. 20,000 / 20,000
searches succeeded, p50 5.7 ms, p99 36.8 ms.

### 13. Why does filtered search fail at low selectivity, and what does your planner do?

A graph walk that may only *return* matching rows explores the same neighbourhood as an unfiltered search,
but finds few results. At 1% selectivity it has to visit about 100× more nodes to collect k matches. A
post-filter (search for k × α, then filter) simply finds none. On SIFT 200k, uniform random filter,
recall@10 / QPS:

| Selectivity | exact scan of matches | graph | post-filter | **planner** |
|---|---|---|---|---|
| 0.1% | 1.000 / 114,673 | 1.000 / 20 | 0.008 / 5,239 | **1.000 / 115,404** |
| 1% | 1.000 / 8,294 | 0.997 / 127 | 0.090 / 2,800 | **1.000 / 7,228** |
| 10% | 1.000 / 515 | 0.999 / 794 | 0.855 / 2,540 | **0.999 / 774** |
| 50% | 1.000 / 157 | 0.993 / 1,577 | 0.988 / 1,612 | **0.993 / 1,485** |
| 100% | 1.000 / 115 | 0.980 / 1,186 | 0.993 / 1,219 | **0.980 / 1,274** |

The planner counts each segment's matches from the roaring postings (cheap), then estimates both costs:
an exact scan costs `matches` distance computations, and a filtered graph walk costs about
`10 · max(ef, k) / selectivity`. It picks the cheaper one, scans exactly whenever matches ≤
`prefilter_threshold`, and falls back to exact if the graph returns fewer than k. It tracks the best
strategy at every selectivity, with or without correlation between the filter and the vectors. The first
version raised `ef` as selectivity fell instead. At 10% that gave 179 QPS; the cost model gives 774.

### 14. Why is cosine the same as inner product on normalized vectors, and why is inner product not a true distance?

For unit vectors, `‖a − b‖² = 2 − 2⟨a, b⟩` and `cos(a, b) = ⟨a, b⟩`, so ranking by cosine, inner product
and L2 gives the same order. A cosine collection normalizes vectors on insert and then runs the inner
product kernel (distance `-⟨q, x⟩`).

Inner product on unnormalized vectors is not a metric. `d(x, x) = −‖x‖²` is not zero and not even the
minimum (a longer vector in the same direction is "closer" to x than x itself), and the triangle
inequality fails. Graph construction assumes that neighbours of neighbours are close; with IP,
large-norm vectors become hubs that are everyone's nearest neighbour. vdb supports `ip` because
embeddings often call for it, but recommends normalizing.

### 15. Where does time go in one HNSW query?

At ef 80 on SIFT, a query computes **1,479 distances**. One 128-d L2 is 10.5 ns with AVX-512 when its
inputs are in cache (27 ns AVX2, 79 ns scalar), so the arithmetic is ~16 µs of a ~300 µs query. The rest
is waiting on memory: each visited node is a random 512-byte vector plus a neighbour list somewhere in a
676 MB structure.

`perf` (`bench/run_profile.sh`: 20 s of single-threaded searches on SIFT1M, user-space counters)
confirms it:

| | cycles/query | IPC | cache misses/query | L1d miss rate | hottest code |
|---|---|---|---|---|---|
| HNSW, ef 80 | 640k | **0.40** | 29,065 | 32% | graph walk 49%, `l2_sq_avx512` 39%, heap 4% |
| HNSW-SQ8, ef 80 | 538k | 0.57 | 14,107 | 14% | graph walk 42%, `sq8_l2_avx512` 38%, re-rank 6% |
| IVF-PQ, nprobe 64, R 100 | 1.4M | **2.29** | 19,365 | 7.5% | code scan 72%, re-rank `l2_sq_avx512` 19% |

- **HNSW retires 0.4 instructions per cycle**: the core sits idle most of the time. About 20 cache misses
  per distance is 8 cache lines of vector plus the neighbour list and visited-set lines, and 87% of the
  last-level cache references miss. The 39% "in the kernel" is mostly stalls on its first loads, not
  arithmetic.
- **SQ8 halves the misses** (a 128-byte code is 2 lines instead of 8) and is faster for exactly that
  reason, though it executes *more* instructions.
- **IVF-PQ is the opposite**: IPC 2.29, compute-bound in the table-lookup loop over contiguous codes.
  Its lever is the scan loop itself, and that is where FAISS's 1.4× lead comes from.

A second experiment backs this up: renumbering nodes in BFS order, so that neighbours sit near each other
in memory, raised QPS by **23–40% at identical recall** (ef 16: 7,696 → 10,870; ef 32: 6,043 → 7,000).
Faster kernels alone would not do that. Prefetching the next candidates' vectors while scoring the
current one is the next step for HNSW.

### 16. How would you scale to a billion vectors?

At 1B × 128-d, float vectors alone are 512 GB, and an HNSW graph adds 100+ GB.

- **Compress in RAM, keep full vectors on SSD.** PQ codes (16–32 bytes each) and a graph sit in RAM; the
  float vectors on NVMe are read only to re-rank the top R. DiskANN (Vamana graph on SSD with PQ in RAM)
  serves billion-scale on one machine with a few SSD reads per query.
- **Shard by id hash** across nodes. Each shard is an independent collection; a router fans out the query
  and merges the top k. Hash sharding keeps shards balanced. Partitioning by IVF cluster would let a
  query skip shards, but hot clusters cause skew.
- **Replicate** each shard for read throughput and availability: ship the WAL to followers, which apply
  it and serve reads.
- Keep the per-node LSM design. Bulk load and compaction into large segments matter even more at this
  scale (question 19).

### 17. Why C++, and how did you keep it memory-safe?

C++ gives control over memory layout (64-byte-aligned rows padded to 16 floats, mmap'd segments,
contiguous neighbour lists), SIMD intrinsics with runtime dispatch, and no GC pauses in the query path.
That is why vdb beats hnswlib at all. Keeping it safe:

- **RAII everywhere**: `AlignedBuffer`, `unique_ptr`, `shared_ptr` snapshots, file handles in classes.
  Raw `new`/`delete` appears in one place, the mutable segment's chunk array: chunks are published to
  concurrent readers through `std::atomic<Chunk*>`, so they cannot be `unique_ptr`s, and the destructor
  owns them.
- **ASan + UBSan** on the whole test suite, and **TSan** on the concurrency, compaction, server and
  recovery tests. All are clean, and CI runs both.
- **libFuzzer** on every parser that takes outside input: WAL records, manifests, filter JSON, the binary
  upsert format.
- `-Wall -Wextra -Wpedantic -Wshadow -Werror` and `clang-tidy` in CI. Status values instead of exceptions
  across the engine, so every error path is explicit.
- A model-based test that runs random operation sequences against a `std::map` reference, including
  reopen and compaction.

### 18. How much does approximate-search recall loss hurt RAG answers?

Not at all at normal settings, and less than recall suggests at aggressive ones. 40,000 Wikipedia
passages, MiniLM embeddings, 59 questions; hit = a passage from the right article is in the top 5:

| Setting | exact | HNSW | HNSW-SQ8 | IVF-PQ |
|---|---|---|---|---|
| ef 64 / nprobe 16 | 0.797 | 0.797 (recall 0.997) | 0.797 (0.997) | 0.797 (0.942) |
| ef 10 / nprobe 2 | 0.797 | 0.780 (0.973) | 0.797 (0.969) | 0.729 (0.793) |
| ef 5 / nprobe 1 | 0.797 | 0.780 (0.895) | 0.797 (0.908) | 0.661 (0.631) |

- The ceiling is the embedding model: exact search misses 12 of the 59 questions, the same ones every
  index misses. Better embeddings (or a re-ranker) would move the number. A better index would not.
- Answers survive recall loss because any of several passages from the right article counts, and the
  neighbours an approximate search drops are the borderline ones near the bottom of the top k, rarely all
  of the relevant ones.
- The cost shows up only when recall falls far: IVF-PQ at nprobe 1 (recall 0.63) loses 17% of the hits.

So for RAG I would spend the latency budget on the embedding model and on k, not on squeezing the last
0.01 of recall out of the index. The search is 0.24 ms; the LLM call that follows takes seconds.

### 19. Why do immutable segments and graph indexes pull against each other, and how did you resolve it?

LSM storage wants many small immutable segments: every flush makes one, and nothing is updated in place.
A graph index wants one big graph, because a query on S segments runs S graph searches and merges them,
and each one pays the fixed cost of descending from the top layer. SIFT 200k rows, each split into S
segments; best QPS at recall@10 ≥ 0.97:

| Segments | QPS | recall@10 |
|---|---|---|
| 1 | 8,278 | 0.981 (ef 64) |
| 2 | 8,628 | 0.969 (ef 32) |
| 5 | 4,426 | 0.986 (ef 32) |
| 10 | 4,022 | 0.972 (ef 16) |
| 50 | 1,435 | 0.991 (ef 16) |

Small segments are individually more accurate (a 4k-row graph is nearly exact at ef 16), so recall holds
up, but throughput falls about 6× from 1 to 50 segments. Three things keep the count low:

- **Bulk load** builds one segment for the initial data instead of hundreds of flushes.
- **Size-tiered compaction**: when `tier_fanout` (4) segments of a similar size exist, merge them.
- **Merge into the largest**: when the small tiers add up to less than 20% of the largest segment, insert
  their rows into the largest segment's graph instead of rebuilding a new one. Extending a graph costs
  only the new inserts, so the collection converges to one big segment plus a few small ones. An idle
  collection is merged down further after `idle_compact_ms`.

The mutable segment is the other half: it is scanned exactly, so the more it holds, the slower queries
get (p50 0.14 ms empty, 1.18 ms at 20k rows, 4.75 ms at 100k). That is why it flushes at 8 MB or 5 minutes,
whichever comes first. Indexing it with a small HNSW instead would make those searches 2–20× faster (2× at 5k rows, 7× at 20k),
but each write would cost 100–250 µs of graph insertion instead of a sub-microsecond append, and fresh
rows would no longer be found exactly. I measured both on the same rows and kept the flat scan.

## Three stories

### A bug I hunted: compaction freed a segment a query was still reading

TSan reported a use-after-free in `Compact`, only in some runs of the concurrency test. The stack
pointed at a loop over the current segments:

```cpp
for (const auto& seg : snapshot()->segments) { ... }
```

`snapshot()` returns a `shared_ptr` by value. In a range-for, that temporary dies once the range is
initialized, so the loop iterates a vector owned by nothing. If a flush swapped the snapshot in the
meantime, the last reference was gone and the segments were freed under the loop. (C++23 extends the
temporary's lifetime here; C++20 does not.) The fix holds the snapshot in a local first:
`auto snap = snapshot(); for (const auto& seg : snap->segments)`, with a comment saying why. CI runs
the concurrency test (flush, compaction, deletes and searches together) for 60 s under TSan.

A smaller one while writing the API tests: a GET carrying a body made the *next* request on that
keep-alive connection fail with 400. cpp-httplib never reads bodies on GET, so the leftover bytes were
parsed as the next request line. The server now closes such connections after answering, and the test
suite sends exactly that sequence.

### A tradeoff I measured: durability vs ingest rate

`sync: always` with fsync per write gives 129 single-point writes/sec, whatever the number of writers.
That is the disk's fsync latency. The table in question 10 drove two decisions. Group commit became the
default (36× for concurrent writers, same durability). And bulk load got its own path, which builds a
segment directly and commits it with one manifest edit: 200k rows in 25 s while 100 clients kept
searching. `sync: interval` is offered for workloads that can lose 100 ms of writes, at 538k/s.

### A surprise: every HTTP request took 40 ms

The first server benchmark showed a flat ~44 ms per request at any load, though the same search took
0.3 ms in-process. Responses were written as headers and then body in two `send` calls. With Nagle's
algorithm on, the second small write waited for the ACK of the first, and the client delayed that ACK by
40 ms. `TCP_NODELAY` dropped p50 to under 1 ms. The other surprise was the BFS reordering in question 15:
the same graph and the same distance computations, 23–40% faster just by putting neighbours next to each
other in memory.

## What I would do next

- Prefetch candidate vectors during graph search (question 15).
- Replace cpp-httplib's thread-per-connection model with an event loop; above ~4,000 QPS the server, not
  the index, is the bottleneck.
- A DiskANN-style SSD index for data sets larger than RAM.
- WAL shipping to a read replica.
