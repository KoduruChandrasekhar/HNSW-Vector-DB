#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "distance/distance.h"
#include "util/aligned_buffer.h"
#include "vdb/options.h"
#include "vdb/status.h"
#include "vdb/types.h"

namespace vdb {

class ThreadPool;
class WritableFile;

// Which rows a search may return. Rows that are not allowed are still walked through
// by graph searches (they keep the graph connected) but never enter the results.
class RowFilter {
 public:
  virtual ~RowFilter() = default;
  virtual bool allowed(InternalId id) const = 0;
};

struct IndexSearchStats {
  uint64_t distance_computations = 0;
  size_t ef = 0;
};

// One interface for every index: Flat, Hnsw, HnswSq8, PqFlat, IvfPq. An index never
// owns the float rows; it reads them through a VectorView owned by the segment.
class Index {
 public:
  virtual ~Index() = default;
  virtual IndexType type() const = 0;
  // Trains (when needed) and adds every row of the view.
  virtual Status build(ThreadPool* pool) = 0;
  // The k nearest allowed rows, closest first. filter == nullptr allows every row.
  virtual void search(const float* q, size_t k, const SearchParams& p, const RowFilter* filter,
                      std::vector<Neighbor>& out, IndexSearchStats* stats) const = 0;
  virtual size_t size() const = 0;
  // Bytes of index structures (graph, codes, centroids); excludes the float rows.
  virtual size_t memory_bytes() const = 0;
  virtual size_t graph_bytes() const { return 0; }
  virtual size_t code_bytes() const { return 0; }
  virtual Status save(WritableFile& f) const = 0;
  // A copy over `vectors`, whose first size() rows are this index's rows, with the
  // remaining rows inserted. Used to merge small segments into the largest one.
  virtual StatusOr<std::unique_ptr<Index>> extend(const VectorView& vectors,
                                                  ThreadPool* pool) const;
};

// An empty index of the requested type over `vectors`; call build() next.
StatusOr<std::unique_ptr<Index>> CreateIndex(const IndexOptions& opts, Metric metric,
                                             const VectorView& vectors);
// Parses a file written by Index::save (checks the magic and the CRC32C footer).
StatusOr<std::unique_ptr<Index>> LoadIndex(std::string_view bytes, const IndexOptions& opts,
                                           Metric metric, const VectorView& vectors);

// Index files: 8-byte magic, body, CRC32C of everything before the footer.
Status WriteIndexFile(WritableFile& f, std::string_view magic, const std::string& body);
Status CheckIndexFile(std::string_view bytes, std::string_view* magic, std::string_view* body);

// Re-ranks candidates with exact float distances and keeps the k best.
void Rerank(const VectorView& v, DistFn fn, const float* q, const std::vector<Neighbor>& cands,
            size_t k, std::vector<Neighbor>& out, uint64_t* distance_computations);

}  // namespace vdb
