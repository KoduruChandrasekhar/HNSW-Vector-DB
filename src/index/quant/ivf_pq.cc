#include "index/quant/ivf_pq.h"

#include <algorithm>
#include <cstring>

#include "index/quant/kmeans.h"
#include "storage/env.h"
#include "util/thread_pool.h"
#include "util/topk.h"

namespace vdb {
namespace {

constexpr size_t kMaxPqTrainRows = 65536;  // 256 centroids x 256 points

size_t RerankDepth(const SearchParams& p, size_t k) {
  if (p.rerank < 0) return std::max<size_t>(10 * k, 100);
  return static_cast<size_t>(p.rerank);
}

KMeansOptions MakeKMeansOptions(const IndexOptions& o) {
  KMeansOptions ko;
  ko.seed = o.seed;
  ko.max_points_per_centroid = o.max_train_points_per_centroid;
  return ko;
}

void FinishSearch(const VectorView& v, DistFn fn, const float* q, const TopK& top, size_t k,
                  size_t depth, std::vector<Neighbor>& out, uint64_t* dc) {
  if (depth == 0) {
    top.sorted_into(&out);
    if (out.size() > k) out.resize(k);
    return;
  }
  Rerank(v, fn, q, top.sorted(), k, out, dc);
}

}  // namespace

// ------------------------------------------------------------------ PqFlatIndex

PqFlatIndex::PqFlatIndex(const IndexOptions& opts, Metric metric, const VectorView& v)
    : opts_(opts), metric_(metric), view_(v), dist_(MetricDistance(metric)), pq_(v.dim, opts.pq_m) {}

Status PqFlatIndex::build(ThreadPool* pool) {
  const size_t n = view_.rows;
  if (n == 0) return Status::OK();
  VDB_RETURN_IF_ERROR(pq_.train(view_.data, n, view_.padded_dim, MakeKMeansOptions(opts_), pool));
  codes_.assign(n * pq_.code_size(), 0);
  ParallelForRange(
      pool, 0, n,
      [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) pq_.encode(view_.row(i), codes_.data() + i * pq_.code_size());
      },
      512);
  return Status::OK();
}

void PqFlatIndex::search(const float* q, size_t k, const SearchParams& p, const RowFilter* filter,
                         std::vector<Neighbor>& out, IndexSearchStats* stats) const {
  out.clear();
  if (view_.rows == 0 || k == 0) return;
  size_t depth = RerankDepth(p, k);
  std::vector<float> table(pq_.table_size());
  pq_.compute_table(q, metric_, table.data());
  TopK top(std::max(depth, k));
  const size_t cs = pq_.code_size();
  for (size_t i = 0; i < view_.rows; ++i) {
    auto id = static_cast<InternalId>(i);
    if (filter != nullptr && !filter->allowed(id)) continue;
    top.push(pq_.adc(table.data(), codes_.data() + i * cs), id);
  }
  uint64_t dc = view_.rows;
  FinishSearch(view_, dist_, q, top, k, depth, out, &dc);
  if (stats != nullptr) stats->distance_computations += dc;
}

Status PqFlatIndex::save(WritableFile& f) const {
  std::string body;
  PutU64(&body, view_.rows);
  pq_.serialize(&body);
  PutBytes(&body, codes_.data(), codes_.size());
  return WriteIndexFile(f, kMagic, body);
}

Status PqFlatIndex::load(ByteReader& r) {
  uint64_t n = 0;
  if (!r.GetU64(&n) || n != view_.rows || !pq_.deserialize(r) || pq_.dim() != view_.dim) {
    return Status::Corruption("pq index does not match the segment");
  }
  std::string_view bytes;
  if (!r.GetBytes(n * pq_.code_size(), &bytes)) return Status::Corruption("truncated pq codes");
  codes_.assign(bytes.begin(), bytes.end());
  return Status::OK();
}

// ------------------------------------------------------------------ IvfPqIndex

IvfPqIndex::IvfPqIndex(const IndexOptions& opts, Metric metric, const VectorView& v)
    : opts_(opts), metric_(metric), view_(v), dist_(MetricDistance(metric)), pq_(v.dim, opts.pq_m) {}

Status IvfPqIndex::build(ThreadPool* pool) {
  const size_t n = view_.rows;
  const size_t pd = view_.padded_dim;
  if (n == 0) return Status::OK();
  // Small segments get fewer lists: FAISS warns below 39 training points per centroid.
  nlist_ = std::clamp<size_t>(opts_.nlist, 1, std::max<size_t>(1, n / 39));
  KMeansOptions ko = MakeKMeansOptions(opts_);
  VDB_RETURN_IF_ERROR(KMeans(view_.data, n, pd, pd, nlist_, ko, pool, &coarse_));

  std::vector<uint32_t> assign(n);
  AssignNearest(view_.data, n, pd, pd, coarse_.data(), nlist_, pool, assign.data(), nullptr);

  // Train PQ on residuals of a strided sample.
  const size_t d = view_.dim;
  size_t step = std::max<size_t>(1, n / kMaxPqTrainRows);
  std::vector<float> res;
  res.reserve(std::min(n, kMaxPqTrainRows) * d);
  for (size_t i = 0; i < n; i += step) {
    const float* x = view_.row(i);
    const float* c = coarse_.data() + static_cast<size_t>(assign[i]) * pd;
    for (size_t j = 0; j < d; ++j) res.push_back(x[j] - c[j]);
  }
  VDB_RETURN_IF_ERROR(pq_.train(res.data(), res.size() / d, d, ko, pool));

  const size_t cs = pq_.code_size();
  std::vector<uint8_t> all(n * cs);
  ParallelForRange(
      pool, 0, n,
      [&](size_t b, size_t e) {
        std::vector<float> r(d);
        for (size_t i = b; i < e; ++i) {
          const float* x = view_.row(i);
          const float* c = coarse_.data() + static_cast<size_t>(assign[i]) * pd;
          for (size_t j = 0; j < d; ++j) r[j] = x[j] - c[j];
          pq_.encode(r.data(), all.data() + i * cs);
        }
      },
      512);
  ids_.assign(nlist_, {});
  codes_.assign(nlist_, {});
  for (size_t i = 0; i < n; ++i) {
    ids_[assign[i]].push_back(static_cast<uint32_t>(i));
    auto& cl = codes_[assign[i]];
    cl.insert(cl.end(), all.begin() + static_cast<std::ptrdiff_t>(i * cs),
              all.begin() + static_cast<std::ptrdiff_t>((i + 1) * cs));
  }
  precompute_tables();
  return Status::OK();
}

void IvfPqIndex::precompute_tables() {
  term2_.clear();
  const size_t m = pq_.m(), ds = pq_.dsub(), ksub = ProductQuantizer::kCentroids;
  if (metric_ != Metric::L2 || nlist_ == 0 || nlist_ * m * ksub * sizeof(float) > kMaxPrecomputedBytes) return;
  term2_.resize(nlist_ * m * ksub);
  std::vector<float> norms(m * ksub);
  for (size_t j = 0; j < m; ++j)
    for (size_t c = 0; c < ksub; ++c) norms[j * ksub + c] = DotAny(pq_.centroid(j, c), pq_.centroid(j, c), ds);
  for (size_t l = 0; l < nlist_; ++l) {
    const float* cl = coarse_.data() + l * view_.padded_dim;
    float* t = term2_.data() + l * m * ksub;
    for (size_t j = 0; j < m; ++j)
      for (size_t c = 0; c < ksub; ++c)
        t[j * ksub + c] = norms[j * ksub + c] + 2.f * DotAny(cl + j * ds, pq_.centroid(j, c), ds);
  }
}

void IvfPqIndex::search(const float* q, size_t k, const SearchParams& p, const RowFilter* filter,
                        std::vector<Neighbor>& out, IndexSearchStats* stats) const {
  out.clear();
  if (nlist_ == 0 || k == 0) return;
  const size_t pd = view_.padded_dim;
  const size_t d = view_.dim;
  uint64_t dc = nlist_;
  std::vector<Neighbor> cd(nlist_);
  for (size_t c = 0; c < nlist_; ++c) {
    cd[c] = {dist_(q, coarse_.data() + c * pd, pd), static_cast<InternalId>(c)};
  }
  size_t nprobe = std::clamp<size_t>(p.nprobe, 1, nlist_);
  std::partial_sort(cd.begin(), cd.begin() + static_cast<std::ptrdiff_t>(nprobe), cd.end());

  const size_t depth = RerankDepth(p, k);
  TopK top(std::max(depth, k));
  const size_t ts = pq_.table_size();
  std::vector<float> table(ts);
  std::vector<float> qtab;  // -2 <q_j, C_j[c]>, for the precomputed-table path
  std::vector<float> resid(d);
  const bool l2 = metric_ == Metric::L2;
  const bool pre = l2 && !term2_.empty();
  if (!l2) pq_.compute_table(q, metric_, table.data());  // shared by every list
  if (pre) {
    qtab.resize(ts);
    pq_.compute_table(q, Metric::InnerProduct, qtab.data());  // -<q_j, C_j[c]>
    for (float& v : qtab) v *= 2.f;
  }
  const size_t cs = pq_.code_size();
  for (size_t pi = 0; pi < nprobe; ++pi) {
    const size_t l = cd[pi].id;
    const float* c = coarse_.data() + l * pd;
    float offset = 0;
    if (pre) {
      offset = cd[pi].dist;  // ||q - c||^2 from the coarse step
      const float* t2 = term2_.data() + l * ts;
      for (size_t i = 0; i < ts; ++i) table[i] = t2[i] + qtab[i];
    } else if (l2) {
      for (size_t j = 0; j < d; ++j) resid[j] = q[j] - c[j];
      pq_.compute_table(resid.data(), Metric::L2, table.data());
    } else {
      offset = -DotAny(q, c, d);  // -<q, c + r> = -<q, c> - <q, r>
    }
    const auto& ids = ids_[l];
    const uint8_t* codes = codes_[l].data();
    for (size_t i = 0; i < ids.size(); ++i) {
      if (filter != nullptr && !filter->allowed(ids[i])) continue;
      top.push(offset + pq_.adc(table.data(), codes + i * cs), ids[i]);
    }
    dc += ids.size();
  }
  FinishSearch(view_, dist_, q, top, k, depth, out, &dc);
  if (stats != nullptr) stats->distance_computations += dc;
}

size_t IvfPqIndex::code_bytes() const {
  size_t b = 0;
  for (const auto& c : codes_) b += c.size();
  return b;
}

size_t IvfPqIndex::memory_bytes() const {
  size_t b = code_bytes() + coarse_.size() * sizeof(float) + pq_.memory_bytes() + term2_.size() * sizeof(float);
  for (const auto& l : ids_) b += l.size() * sizeof(uint32_t);
  return b;
}

Status IvfPqIndex::save(WritableFile& f) const {
  std::string body;
  PutU64(&body, view_.rows);
  PutU32(&body, static_cast<uint32_t>(view_.padded_dim));
  PutU32(&body, static_cast<uint32_t>(nlist_));
  PutBytes(&body, coarse_.data(), coarse_.size() * sizeof(float));
  pq_.serialize(&body);
  for (size_t l = 0; l < nlist_; ++l) {
    PutU64(&body, ids_[l].size());
    PutBytes(&body, ids_[l].data(), ids_[l].size() * sizeof(uint32_t));
    PutBytes(&body, codes_[l].data(), codes_[l].size());
  }
  return WriteIndexFile(f, kMagic, body);
}

Status IvfPqIndex::load(ByteReader& r) {
  uint64_t n = 0;
  uint32_t pd = 0, nl = 0;
  if (!r.GetU64(&n) || !r.GetU32(&pd) || !r.GetU32(&nl) || n != view_.rows ||
      pd != view_.padded_dim) {
    return Status::Corruption("ivf-pq index does not match the segment");
  }
  nlist_ = nl;
  std::string_view bytes;
  if (!r.GetBytes(size_t(nl) * pd * sizeof(float), &bytes)) return Status::Corruption("truncated ivf centroids");
  coarse_.resize(size_t(nl) * pd);
  std::memcpy(coarse_.data(), bytes.data(), bytes.size());
  if (!pq_.deserialize(r) || pq_.dim() != view_.dim) return Status::Corruption("bad pq codebook");
  ids_.assign(nl, {});
  codes_.assign(nl, {});
  for (size_t l = 0; l < nl; ++l) {
    uint64_t sz = 0;
    if (!r.GetU64(&sz) || sz > n) return Status::Corruption("bad ivf list size");
    if (!r.GetBytes(sz * sizeof(uint32_t), &bytes)) return Status::Corruption("truncated ivf ids");
    ids_[l].resize(sz);
    std::memcpy(ids_[l].data(), bytes.data(), bytes.size());
    for (uint32_t id : ids_[l])
      if (id >= n) return Status::Corruption("ivf id out of range");
    if (!r.GetBytes(sz * pq_.code_size(), &bytes)) return Status::Corruption("truncated ivf codes");
    codes_[l].assign(bytes.begin(), bytes.end());
  }
  precompute_tables();  // derived data: rebuilt rather than stored
  return Status::OK();
}

}  // namespace vdb
