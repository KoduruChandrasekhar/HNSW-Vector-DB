#pragma once

#include <memory>

#include "index/hnsw/hnsw_graph.h"
#include "index/index.h"
#include "index/quant/sq8.h"

namespace vdb {

// HNSW over the segment's float rows.
class HnswIndex final : public Index {
 public:
  using Graph = HnswGraph<FloatSpace>;

  HnswIndex(const IndexOptions& opts, Metric metric, const VectorView& v);
  IndexType type() const override { return IndexType::Hnsw; }
  Status build(ThreadPool* pool) override;
  void search(const float* q, size_t k, const SearchParams& p, const RowFilter* filter,
              std::vector<Neighbor>& out, IndexSearchStats* stats) const override;
  size_t size() const override { return graph_ ? graph_->size() : 0; }
  size_t memory_bytes() const override { return graph_ ? graph_->memory_bytes() : 0; }
  size_t graph_bytes() const override { return memory_bytes(); }
  Status save(WritableFile& f) const override;
  StatusOr<std::unique_ptr<Index>> extend(const VectorView& vectors,
                                          ThreadPool* pool) const override;
  Status load(ByteReader& r);

  const Graph& graph() const { return *graph_; }

  static constexpr std::string_view kMagic = "VDBHNSW1";

 private:
  FloatSpace space() const { return FloatSpace{view_.data, view_.padded_dim, dist_}; }

  IndexOptions opts_;
  Metric metric_;
  VectorView view_;
  DistFn dist_;
  std::unique_ptr<Graph> graph_;
};

// Query (float) against 8-bit codes, decoded inside the SIMD loop.
struct Sq8Space {
  const uint8_t* codes = nullptr;
  size_t stride = 0;  // padded dim
  const Sq8Quantizer* sq = nullptr;
  Sq8DistFn fn = nullptr;

  float dist(const float* q, uint32_t id) const {
    return fn(q, codes + static_cast<size_t>(id) * stride, sq->vmin(), sq->scale(), stride);
  }
  const float* node_vec(uint32_t id, float* scratch) const {
    sq->decode(codes + static_cast<size_t>(id) * stride, scratch);
    return scratch;
  }
  void prefetch(uint32_t id) const { Prefetch(codes + static_cast<size_t>(id) * stride); }
  size_t padded_dim() const { return stride; }
};

// HNSW over SQ8 codes (4x smaller than float rows), then the top candidates are
// re-ranked with the float rows.
class HnswSq8Index final : public Index {
 public:
  using Graph = HnswGraph<Sq8Space>;

  HnswSq8Index(const IndexOptions& opts, Metric metric, const VectorView& v);
  IndexType type() const override { return IndexType::HnswSq8; }
  Status build(ThreadPool* pool) override;
  void search(const float* q, size_t k, const SearchParams& p, const RowFilter* filter,
              std::vector<Neighbor>& out, IndexSearchStats* stats) const override;
  size_t size() const override { return graph_ ? graph_->size() : 0; }
  size_t memory_bytes() const override;
  size_t graph_bytes() const override { return graph_ ? graph_->memory_bytes() : 0; }
  size_t code_bytes() const override { return codes_.size(); }
  Status save(WritableFile& f) const override;
  Status load(ByteReader& r);

  const Sq8Quantizer& quantizer() const { return sq_; }
  const Graph& graph() const { return *graph_; }

  static constexpr std::string_view kMagic = "VDBHSQ81";

 private:
  Sq8Space space() const {
    return Sq8Space{codes_.as<uint8_t>(), view_.padded_dim, &sq_, sq8_dist_};
  }

  IndexOptions opts_;
  VectorView view_;
  DistFn dist_;
  Sq8DistFn sq8_dist_;
  Sq8Quantizer sq_;
  AlignedBuffer codes_;  // rows * padded_dim bytes
  std::unique_ptr<Graph> graph_;
};

}  // namespace vdb
