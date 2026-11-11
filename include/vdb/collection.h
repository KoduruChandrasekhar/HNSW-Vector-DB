#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "vdb/metadata.h"
#include "vdb/options.h"
#include "vdb/status.h"
#include "vdb/types.h"

namespace vdb {

class Env;
class CollectionImpl;

// A parsed, schema-validated filter expression. Create one with Collection::ParseFilter.
class Filter {
 public:
  virtual ~Filter() = default;
  virtual std::string ToString() const = 0;
};

struct SearchHit {
  VectorId id = 0;
  float distance = 0;  // smaller = closer: squared L2, or minus the dot product
  float score = 0;     // l2: same as distance; inner product / cosine: the similarity
  Metadata metadata;          // filled when include_metadata is set
  std::vector<float> vector;  // filled when include_vector is set (real dims only)
};

struct SearchRequest {
  const float* vector = nullptr;  // dim floats
  size_t k = 10;
  SearchParams params;
  const Filter* filter = nullptr;
  bool include_metadata = false;
  bool include_vector = false;
};

// What the planner did in one segment ("explain": true).
struct SegmentExplain {
  uint32_t segment_id = 0;
  std::string kind;      // mutable, frozen, disk
  std::string index;     // flat, hnsw, ...
  std::string strategy;  // exact, graph, post_filter, prefilter_exact, ...
  size_t rows = 0;
  size_t live_rows = 0;
  size_t matches = 0;
  double selectivity = 1.0;
  size_t ef = 0;
  uint64_t distance_computations = 0;
  size_t results = 0;
  double micros = 0;
};

struct SearchExplain {
  std::vector<SegmentExplain> segments;
  double total_micros = 0;
  uint64_t distance_computations = 0;
};

struct UpsertPoint {
  VectorId id = 0;
  const float* vector = nullptr;      // dim floats
  const Metadata* metadata = nullptr; // optional
};

struct SegmentStats {
  uint32_t id = 0;
  std::string kind;
  std::string index;
  size_t rows = 0;
  size_t deleted = 0;
  uint64_t lsn = 0;
  size_t memory_bytes = 0;
  uint64_t disk_bytes = 0;
};

struct MemoryStats {
  size_t vectors = 0;
  size_t graph = 0;
  size_t codes = 0;
  size_t id_map = 0;
  size_t metadata = 0;
  size_t total() const { return vectors + graph + codes + id_map + metadata; }
};

struct CollectionStats {
  size_t live_rows = 0;
  size_t total_rows = 0;
  size_t deleted_rows = 0;
  uint64_t last_lsn = 0;
  uint64_t flushed_lsn = 0;
  std::vector<SegmentStats> segments;
  MemoryStats memory;
  uint64_t disk_bytes = 0;
  uint64_t flushes = 0;
  uint64_t compactions = 0;
  double last_recovery_seconds = 0;
  uint64_t last_recovery_records = 0;
};

// Settings that apply to one open handle and are not persisted.
struct OpenOptions {
  size_t num_threads = 0;  // 0 = hardware concurrency
  bool verify_checksums = true;
  bool auto_flush = true;
  bool auto_compact = true;
  // After this long without writes, merge every segment into the largest (0 = never).
  uint32_t idle_compact_ms = 60000;
  // Batch concurrent writers into one log append + fsync. Off only to measure what it buys.
  bool group_commit = true;
};

class Collection {
 public:
  // Creates a new collection directory (fails if one already exists there).
  static StatusOr<std::unique_ptr<Collection>> Create(const std::string& path,
                                                      const CollectionOptions& options,
                                                      const OpenOptions& open = {},
                                                      Env* env = nullptr);
  // Opens an existing collection, running crash recovery.
  static StatusOr<std::unique_ptr<Collection>> Open(const std::string& path,
                                                    const OpenOptions& open = {},
                                                    Env* env = nullptr);
  ~Collection();
  Collection(const Collection&) = delete;
  Collection& operator=(const Collection&) = delete;

  const CollectionOptions& options() const;

  // ---- writes: each point is its own log record; a batch is not atomic.
  Status Upsert(VectorId id, const float* vector, const Metadata* metadata = nullptr,
                uint64_t* lsn = nullptr);
  Status UpsertBatch(const std::vector<UpsertPoint>& points, uint64_t* last_lsn = nullptr);
  // Contiguous rows of dim floats.
  Status UpsertBatch(const VectorId* ids, const float* vectors, size_t n,
                     const std::vector<Metadata>* metadata = nullptr,
                     uint64_t* last_lsn = nullptr);
  Status Remove(VectorId id, bool* found = nullptr, uint64_t* lsn = nullptr);
  Status RemoveBatch(const VectorId* ids, size_t n, size_t* removed = nullptr);
  // All-or-nothing load of ids not yet in the collection into one new segment with one
  // index. Acknowledged after the manifest commit; writes no log records.
  Status BulkLoad(const VectorId* ids, const float* vectors, size_t n,
                  const std::vector<Metadata>* metadata = nullptr);

  // ---- reads
  StatusOr<std::unique_ptr<Filter>> ParseFilter(std::string_view json) const;
  Status Search(const SearchRequest& req, std::vector<SearchHit>* out,
                SearchExplain* explain = nullptr) const;
  // nq queries of dim floats each, searched in parallel on the collection's thread pool.
  Status SearchBatch(const float* queries, size_t nq, size_t k, const SearchParams& params,
                     const Filter* filter, std::vector<std::vector<SearchHit>>* out) const;
  Status Get(VectorId id, std::vector<float>* vector, Metadata* metadata) const;

  // ---- maintenance
  Status Flush();                               // synchronous
  void RequestFlush();                          // asynchronous
  Status Compact(size_t target_segments = 0);   // 0 = run the policy until it settles
  void RequestCompaction(size_t target_segments = 0);
  CollectionStats Stats() const;
  // Stops background work and syncs the log. Called by the destructor.
  Status Close();

 private:
  explicit Collection(std::unique_ptr<CollectionImpl> impl);
  std::unique_ptr<CollectionImpl> impl_;
};

}  // namespace vdb
