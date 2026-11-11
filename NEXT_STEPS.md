# Status (2026-11-01)

Every part of the project plan is implemented, tested, benchmarked and written up. What is left
needs you: version control, CI, and the parts that need hardware this laptop does not have.

## Done

- **Code**: library, server, Python bindings and HTTP client, tools, benchmarks, RAG scripts, fuzzers, CI
  workflow.
- **Tests on the final code** (2026-11-01):
  - Release: 15/15 C++ test binaries.
  - ASan + UBSan: 15/15.
  - TSan, 0 races: concurrency (60 s stress), compaction, server, recovery, filter, hnsw, model, pq.
  - Python bindings: 3/3. HTTP API (`tests/api_test.py`): 24/24, covering every route, every error branch
    in the handlers, the Python `vdb.Client`, and a `--config` file.
  - Earlier: fuzzers 60 s each with no crash; kill -9 harness 1,000/1,000.
- **Benchmarks**: all results in `bench/results/`, charts and `results_table.md` in
  `bench/results/charts/` (from `python bench/plot.py`).
- **RAG**: 40k passages; hit rate@5 for exact / HNSW / SQ8 / IVF-PQ at four search settings
  (`bench/results/rag_hit_rate.json`); `rag/ask.py` end to end against `rag/mock_llm_server.py`.
- **Docs**: README (headline numbers, results, design decisions, correctness, limitations),
  `docs/API.md` (every endpoint, field, limit, error code, metric and flag), `docs/WRITEUP.md` (resume
  lines, the plan's 19 interview questions answered from the measurements, three stories).

- **Profiling**: `bench/run_profile.sh` runs `perf` on HNSW, HNSW-SQ8 and IVF-PQ search
  (`bench/results/perf_*.txt`). HNSW is memory-bound (IPC 0.40); IVF-PQ is compute-bound (IPC 2.29).
  `vdb_bench mutable` now also compares the flat mutable segment with a small incremental HNSW.
- **Demo video**: `docs/demo/vdb-demo.mp4` (90 s, 1280×720) and `vdb-demo.gif`, recorded live by
  `demo/record.py` and drawn by `demo/render.py` (`bash demo/run_demo.sh` redoes both).

Fixed on the last day: the HTTP API ignored `post_filter_alpha`; httplib's unanswered 400s had the
code `"error"` (now `invalid_argument`); and a GET carrying a body poisoned its keep-alive connection
(httplib never reads it), which now gets closed after the response.

## Left (your call)

1. `git init`, a first commit, and a push to GitHub so `.github/workflows/ci.yml` runs.
2. Point `LLM_URL` at a real inference server for real RAG answers (only the mock was available), then
   re-record the demo with `bash demo/run_demo.sh` so the RAG scene shows a real model's answer.
3. Optional: a larger RAG corpus with a GPU (CPU embedding ran at ~23 passages/s); prefetching in graph
   search (perf shows HNSW at 0.4 IPC, memory-bound); a faster IVF-PQ scan loop (compute-bound, the gap
   to FAISS); an event-loop HTTP server (cpp-httplib's thread per connection limits throughput above
   ~4k QPS).

## Commands (inside WSL: `wsl -d Ubuntu`)

```bash
cd /mnt/c/Users/chand/OneDrive/Desktop/VectorDB
cmake --build ~/vdb-build/bench -j8 && (cd ~/vdb-build/bench && ctest -j3)
VDB_SERVER=~/vdb-build/bench/vdb-server ~/vdb-venv/bin/python -m pytest tests/api_test.py tests/test_python.py
export RAG_WORK=~/rag-work                      # keep it off the OneDrive folder (see below)
~/vdb-venv/bin/python rag/eval.py --questions rag/questions.jsonl
~/vdb-venv/bin/python bench/plot.py
```

Sanitizer trees are `~/vdb-build/asan2` and `~/vdb-build/tsan2` (each with its own
`FETCHCONTENT_BASE_DIR`).

## Environment notes

- Toolchain in WSL Ubuntu 26.04 (GCC 15, Clang 21, CMake 4.2); venv `~/vdb-venv`; datasets `~/vdb-data`.
  `apt` without a password: `wsl -d Ubuntu -u root -- apt-get install ...`.
- `/tmp` is RAM-backed (tmpfs): measure anything involving fsync under `$HOME` (ext4).
- The repo is in a OneDrive-synced folder. From WSL, renaming a freshly written directory there can
  fail with "Permission denied", so keep vdb data directories (`RAG_WORK`, `--data-dir`) on ext4.
- Keep the laptop awake during long runs: sleep pauses the VM and kills background jobs.
