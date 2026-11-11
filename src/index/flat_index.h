#pragma once

#include <vector>

#include "index/index.h"

namespace vdb {

// Exact top-k for one query. With a pool, rows are split across threads (each with its
// own heap) and the heaps are merged at the end.
void FlatSearch(const VectorView& v, DistFn fn, const float* q, size_t k,
                const RowFilter* filter, std::vector<Neighbor>& out, ThreadPool* pool = nullptr);

// Exact top-k for a batch of queries (padded rows of v.padded_dim floats). Queries are
// processed in blocks of 64 against blocks of base rows that fit in L2 cache, so each row
// fetched from memory serves the whole query block. Query blocks run in parallel.
void FlatSearchBatch(const VectorView& v, DistFn fn, const float* queries, size_t nq, size_t k,
                     std::vector<std::vector<Neighbor>>& out, ThreadPool* pool = nullptr);

class FlatIndex final : public Index {
 public:
  FlatIndex(Metric metric, const VectorView& v, bool force_scalar = false);
  IndexType type() const override { return IndexType::Flat; }
  Status build(ThreadPool*) override { return Status::OK(); }
  void search(const float* q, size_t k, const SearchParams& p, const RowFilter* filter,
              std::vector<Neighbor>& out, IndexSearchStats* stats) const override;
  size_t size() const override { return view_.rows; }
  size_t memory_bytes() const override { return 0; }
  Status save(WritableFile& f) const override;
  StatusOr<std::unique_ptr<Index>> extend(const VectorView& vectors,
                                          ThreadPool* pool) const override;

  static constexpr std::string_view kMagic = "VDBFLAT1";

 private:
  Metric metric_;
  VectorView view_;
  DistFn dist_;
};

}  // namespace vdb
