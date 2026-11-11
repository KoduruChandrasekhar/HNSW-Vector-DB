#include "index/quant/pq.h"

#include <cstring>
#include <limits>

#include "distance/distance.h"
#include "util/thread_pool.h"

namespace vdb {

ProductQuantizer::ProductQuantizer(size_t dim, size_t m)
    : dim_(dim), m_(m), dsub_(m == 0 ? 0 : dim / m) {}

Status ProductQuantizer::train(const float* data, size_t n, size_t stride,
                               const KMeansOptions& opts, ThreadPool* pool) {
  if (m_ == 0 || dim_ % m_ != 0) {
    return Status::InvalidArgument("pq_m must divide the dimension (dim " + std::to_string(dim_) +
                                   ", m " + std::to_string(m_) + ")");
  }
  centroids_.assign(m_ * kCentroids * dsub_, 0.f);
  // Each subspace is trained on its own contiguous slice of the (subsampled) rows.
  size_t cap = std::min(n, kCentroids * std::max<size_t>(1, opts.max_points_per_centroid));
  size_t step = std::max<size_t>(1, n / std::max<size_t>(1, cap));
  std::vector<float> sub;
  sub.reserve(cap * dsub_);
  std::vector<float> cen;
  for (size_t j = 0; j < m_; ++j) {
    sub.clear();
    for (size_t i = 0, taken = 0; i < n && taken < cap; i += step, ++taken) {
      const float* p = data + i * stride + j * dsub_;
      sub.insert(sub.end(), p, p + dsub_);
    }
    size_t rows = sub.size() / dsub_;
    KMeansOptions o = opts;
    o.seed = opts.seed + j * 7919;
    VDB_RETURN_IF_ERROR(KMeans(sub.data(), rows, dsub_, dsub_, kCentroids, o, pool, &cen));
    std::memcpy(centroids_.data() + j * kCentroids * dsub_, cen.data(),
                kCentroids * dsub_ * sizeof(float));
  }
  return Status::OK();
}

void ProductQuantizer::encode(const float* x, uint8_t* code) const {
  for (size_t j = 0; j < m_; ++j) {
    const float* xs = x + j * dsub_;
    float best = std::numeric_limits<float>::infinity();
    size_t arg = 0;
    for (size_t c = 0; c < kCentroids; ++c) {
      float d = L2SqAny(xs, centroid(j, c), dsub_);
      if (d < best) {
        best = d;
        arg = c;
      }
    }
    code[j] = static_cast<uint8_t>(arg);
  }
}

void ProductQuantizer::decode(const uint8_t* code, float* x) const {
  for (size_t j = 0; j < m_; ++j) std::memcpy(x + j * dsub_, centroid(j, code[j]), dsub_ * sizeof(float));
}

void ProductQuantizer::compute_table(const float* q, Metric metric, float* table) const {
  for (size_t j = 0; j < m_; ++j) {
    const float* qs = q + j * dsub_;
    float* t = table + j * kCentroids;
    if (metric == Metric::L2) {
      for (size_t c = 0; c < kCentroids; ++c) t[c] = L2SqAny(qs, centroid(j, c), dsub_);
    } else {
      for (size_t c = 0; c < kCentroids; ++c) t[c] = -DotAny(qs, centroid(j, c), dsub_);
    }
  }
}

void ProductQuantizer::serialize(std::string* out) const {
  PutU32(out, static_cast<uint32_t>(dim_));
  PutU32(out, static_cast<uint32_t>(m_));
  PutBytes(out, centroids_.data(), centroids_.size() * sizeof(float));
}

bool ProductQuantizer::deserialize(ByteReader& r) {
  uint32_t d = 0, m = 0;
  if (!r.GetU32(&d) || !r.GetU32(&m) || m == 0 || d % m != 0) return false;
  dim_ = d;
  m_ = m;
  dsub_ = d / m;
  std::string_view bytes;
  if (!r.GetBytes(m_ * kCentroids * dsub_ * sizeof(float), &bytes)) return false;
  centroids_.resize(m_ * kCentroids * dsub_);
  std::memcpy(centroids_.data(), bytes.data(), bytes.size());
  return true;
}

}  // namespace vdb
