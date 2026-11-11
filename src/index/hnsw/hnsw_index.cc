#include "index/hnsw/hnsw_index.h"

#include <algorithm>
#include <cstring>

#include "storage/env.h"
#include "util/thread_pool.h"

namespace vdb {

// ------------------------------------------------------------------ HnswIndex

HnswIndex::HnswIndex(const IndexOptions& opts, Metric metric, const VectorView& v)
    : opts_(opts), metric_(metric), view_(v), dist_(MetricDistance(metric)) {}

Status HnswIndex::build(ThreadPool* pool) {
  graph_ = std::make_unique<Graph>(space(), view_.rows, opts_.M, opts_.ef_construction, opts_.seed);
  graph_->build(0, view_.rows, pool);
  return Status::OK();
}

void HnswIndex::search(const float* q, size_t k, const SearchParams& p, const RowFilter* filter,
                       std::vector<Neighbor>& out, IndexSearchStats* stats) const {
  if (!graph_) {
    out.clear();
    return;
  }
  uint64_t dc = 0;
  graph_->search(q, k, std::max(p.ef, k), filter, out, &dc);
  if (stats != nullptr) {
    stats->distance_computations += dc;
    stats->ef = std::max(p.ef, k);
  }
}

Status HnswIndex::save(WritableFile& f) const {
  std::string body;
  PutU32(&body, static_cast<uint32_t>(view_.padded_dim));
  graph_->serialize(&body);
  return WriteIndexFile(f, kMagic, body);
}

Status HnswIndex::load(ByteReader& r) {
  uint32_t pd = 0;
  if (!r.GetU32(&pd) || pd != view_.padded_dim) return Status::Corruption("hnsw dim mismatch");
  VDB_ASSIGN_OR_RETURN(graph_, Graph::Deserialize(r, space(), view_.rows));
  return Status::OK();
}

StatusOr<std::unique_ptr<Index>> HnswIndex::extend(const VectorView& vectors,
                                                   ThreadPool* pool) const {
  if (vectors.rows < size() || vectors.padded_dim != view_.padded_dim) {
    return Status::InvalidArgument("extend needs a superset of the indexed rows");
  }
  auto idx = std::make_unique<HnswIndex>(opts_, metric_, vectors);
  idx->graph_ = std::make_unique<Graph>(*graph_, idx->space(), vectors.rows);
  idx->graph_->build(graph_->size(), vectors.rows, pool);
  return std::unique_ptr<Index>(std::move(idx));
}

// ------------------------------------------------------------------ HnswSq8Index

HnswSq8Index::HnswSq8Index(const IndexOptions& opts, Metric metric, const VectorView& v)
    : opts_(opts), view_(v), dist_(MetricDistance(metric)), sq8_dist_(MetricSq8Distance(metric)) {}

Status HnswSq8Index::build(ThreadPool* pool) {
  sq_.train(view_);
  const size_t pd = view_.padded_dim;
  codes_.reset(view_.rows * pd);
  uint8_t* codes = codes_.as<uint8_t>();
  ParallelForRange(
      pool, 0, view_.rows,
      [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) sq_.encode(view_.row(i), codes + i * pd);
      },
      1024);
  graph_ = std::make_unique<Graph>(space(), view_.rows, opts_.M, opts_.ef_construction, opts_.seed);
  graph_->build(0, view_.rows, pool);
  return Status::OK();
}

void HnswSq8Index::search(const float* q, size_t k, const SearchParams& p,
                          const RowFilter* filter, std::vector<Neighbor>& out,
                          IndexSearchStats* stats) const {
  out.clear();
  if (!graph_ || k == 0) return;
  // Re-rank the whole beam by default: the graph already paid for those candidates.
  size_t depth = p.rerank < 0 ? std::max(p.ef, k) : static_cast<size_t>(p.rerank);
  size_t ef = std::max({p.ef, k, depth});
  uint64_t dc = 0;
  if (depth == 0) {
    graph_->search(q, k, ef, filter, out, &dc);
  } else {
    std::vector<Neighbor> cands;
    graph_->search(q, std::max(depth, k), ef, filter, cands, &dc);
    Rerank(view_, dist_, q, cands, k, out, &dc);
  }
  if (stats != nullptr) {
    stats->distance_computations += dc;
    stats->ef = ef;
  }
}

size_t HnswSq8Index::memory_bytes() const {
  return codes_.size() + sq_.memory_bytes() + (graph_ ? graph_->memory_bytes() : 0);
}

Status HnswSq8Index::save(WritableFile& f) const {
  std::string body;
  PutU64(&body, view_.rows);
  sq_.serialize(&body);
  PutBytes(&body, codes_.data(), codes_.size());
  graph_->serialize(&body);
  return WriteIndexFile(f, kMagic, body);
}

Status HnswSq8Index::load(ByteReader& r) {
  uint64_t n = 0;
  if (!r.GetU64(&n) || n != view_.rows || !sq_.deserialize(r) ||
      sq_.padded_dim() != view_.padded_dim) {
    return Status::Corruption("hnsw-sq8 index does not match the segment");
  }
  std::string_view bytes;
  if (!r.GetBytes(n * view_.padded_dim, &bytes)) return Status::Corruption("truncated sq8 codes");
  codes_.reset(bytes.size());
  if (!bytes.empty()) std::memcpy(codes_.data(), bytes.data(), bytes.size());
  VDB_ASSIGN_OR_RETURN(graph_, Graph::Deserialize(r, space(), view_.rows));
  return Status::OK();
}

}  // namespace vdb
