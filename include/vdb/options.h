#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "vdb/metadata.h"
#include "vdb/types.h"

namespace vdb {

enum class IndexType : uint8_t { Flat = 0, Hnsw = 1, HnswSq8 = 2, PqFlat = 3, IvfPq = 4 };

enum class SyncMode : uint8_t {
  Always = 0,    // fsync (group commit) before acknowledging a write
  Interval = 1,  // a background thread fsyncs every sync_interval_ms
  None = 2,      // let the OS flush; for benchmarks only
};

struct IndexOptions {
  IndexType type = IndexType::Hnsw;
  // HNSW (also used by HnswSq8)
  size_t M = 16;
  size_t ef_construction = 200;
  // Product quantization (PqFlat, IvfPq): number of sub-vectors; must divide dim.
  size_t pq_m = 16;
  // IVF: number of inverted lists (scaled down automatically for small segments).
  size_t nlist = 1024;
  // k-means training sample cap, in points per centroid (FAISS uses 256).
  size_t max_train_points_per_centroid = 256;
  uint64_t seed = 42;
};

struct CollectionOptions {
  size_t dim = 0;
  Metric metric = Metric::L2;
  IndexOptions index;
  Schema schema;
  SyncMode sync = SyncMode::Always;
  uint32_t sync_interval_ms = 100;

  // The mutable segment is flushed once it holds this many bytes of vectors, or once its
  // oldest row is this old (0 = no age limit), whichever comes first.
  size_t flush_bytes = 8u << 20;
  uint32_t flush_age_ms = 300000;

  // Size-tiered compaction: tiers of base, base*fanout, base*fanout^2 ... rows.
  size_t tier_base_rows = 20000;
  size_t tier_fanout = 4;
  // Merge small segments into the largest once they pass this fraction of it.
  double merge_into_largest_ratio = 0.2;
  // Rebuild a segment once this fraction of its rows is deleted.
  double delete_compaction_ratio = 0.2;
};

enum class FilterStrategy : uint8_t { Auto = 0, PreFilter = 1, Graph = 2, PostFilter = 3 };

struct SearchParams {
  size_t ef = 64;       // HNSW beam width; max(ef, k) is used
  size_t nprobe = 16;   // IVF lists to scan
  int rerank = -1;      // candidates re-ranked with float vectors; -1 = auto, 0 = none
  FilterStrategy strategy = FilterStrategy::Auto;
  size_t prefilter_threshold = 10000;  // planner: matches at or below -> always exact
  size_t post_filter_alpha = 10;       // post-filter searches k * alpha
  bool explain = false;
};

const char* IndexTypeName(IndexType t);
bool ParseIndexType(std::string_view s, IndexType* out);
const char* SyncModeName(SyncMode m);
bool ParseSyncMode(std::string_view s, SyncMode* out);
const char* FilterStrategyName(FilterStrategy s);
bool ParseFilterStrategy(std::string_view s, FilterStrategy* out);

}  // namespace vdb
