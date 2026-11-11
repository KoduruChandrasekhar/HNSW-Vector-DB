#include "index/flat_index.h"

#include <algorithm>

#include "storage/env.h"
#include "util/thread_pool.h"
#include "util/topk.h"

namespace vdb {

void FlatSearch(const VectorView& v, DistFn fn, const float* q, size_t k,
                const RowFilter* filter, std::vector<Neighbor>& out, ThreadPool* pool) {
  out.clear();
  if (k == 0 || v.rows == 0) return;
  auto scan = [&](size_t b, size_t e, TopK& top) {
    for (size_t i = b; i < e; ++i) {
      auto id = static_cast<InternalId>(i);
      if (filter != nullptr && !filter->allowed(id)) continue;
      top.push(fn(q, v.row(i), v.padded_dim), id);
    }
  };
  if (pool == nullptr || pool->parallelism() == 1 || v.rows < 4096) {
    TopK top(k);
    scan(0, v.rows, top);
    top.sorted_into(&out);
    return;
  }
  size_t parts = pool->parallelism();
  size_t chunk = (v.rows + parts - 1) / parts;
  std::vector<TopK> tops(parts, TopK(k));
  pool->parallel_for(0, parts, [&](size_t t) {
    size_t b = t * chunk, e = std::min(v.rows, b + chunk);
    if (b < e) scan(b, e, tops[t]);
  });
  TopK merged(k);
  for (const auto& t : tops)
    for (const auto& n : t.raw()) merged.push(n.dist, n.id);
  merged.sorted_into(&out);
}

void FlatSearchBatch(const VectorView& v, DistFn fn, const float* queries, size_t nq, size_t k,
                     std::vector<std::vector<Neighbor>>& out, ThreadPool* pool) {
  out.assign(nq, {});
  if (nq == 0) return;
  constexpr size_t kQueryBlock = 64;
  // About 256 KB of base rows per block: stays in L2 while the query block reuses it.
  const size_t row_block = std::max<size_t>(64, (256u << 10) / (v.padded_dim * sizeof(float)));
  const size_t nblocks = (nq + kQueryBlock - 1) / kQueryBlock;
  auto run_block = [&](size_t qb) {
    size_t q0 = qb * kQueryBlock, q1 = std::min(nq, q0 + kQueryBlock);
    std::vector<TopK> tops(q1 - q0, TopK(k));
    for (size_t r0 = 0; r0 < v.rows; r0 += row_block) {
      size_t r1 = std::min(v.rows, r0 + row_block);
      for (size_t qi = q0; qi < q1; ++qi) {
        const float* q = queries + qi * v.padded_dim;
        TopK& top = tops[qi - q0];
        for (size_t r = r0; r < r1; ++r)
          top.push(fn(q, v.row(r), v.padded_dim), static_cast<InternalId>(r));
      }
    }
    for (size_t qi = q0; qi < q1; ++qi) tops[qi - q0].sorted_into(&out[qi]);
  };
  if (pool == nullptr) {
    for (size_t b = 0; b < nblocks; ++b) run_block(b);
  } else {
    pool->parallel_for(0, nblocks, run_block);
  }
}

FlatIndex::FlatIndex(Metric metric, const VectorView& v, bool force_scalar)
    : metric_(metric), view_(v), dist_(MetricDistance(metric, force_scalar)) {}

void FlatIndex::search(const float* q, size_t k, const SearchParams&, const RowFilter* filter,
                       std::vector<Neighbor>& out, IndexSearchStats* stats) const {
  FlatSearch(view_, dist_, q, k, filter, out, nullptr);
  if (stats != nullptr) stats->distance_computations += view_.rows;
}

Status FlatIndex::save(WritableFile& f) const { return WriteIndexFile(f, kMagic, std::string()); }

StatusOr<std::unique_ptr<Index>> FlatIndex::extend(const VectorView& vectors, ThreadPool*) const {
  return std::unique_ptr<Index>(std::make_unique<FlatIndex>(metric_, vectors));
}

}  // namespace vdb
