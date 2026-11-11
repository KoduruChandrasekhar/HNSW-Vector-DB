#include "index/quant/kmeans.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>

#include "distance/distance.h"
#include "util/thread_pool.h"

namespace vdb {
namespace {

DistFn PickL2(size_t d, size_t stride) {
  // SIMD kernels need padded rows; sub-vectors of a larger row use the any-length kernel.
  return (d % kPadFloats == 0 && stride % kPadFloats == 0) ? Kernels().l2 : L2SqAny;
}

}  // namespace

void AssignNearest(const float* data, size_t n, size_t d, size_t stride, const float* centroids,
                   size_t k, ThreadPool* pool, uint32_t* assign, float* dist) {
  DistFn fn = PickL2(d, stride);
  // Blocked: a block of points stays in L1 while each centroid row is scored against all
  // of them, so every centroid fetched from L3 serves the whole block. With thousands of
  // centroids (IVF), the one-pair-at-a-time loop is bound by re-reading the centroids.
  constexpr size_t kBlock = 8;
  ParallelForRange(
      pool, 0, n,
      [&](size_t b, size_t e) {
        for (size_t p0 = b; p0 < e; p0 += kBlock) {
          const size_t pn = std::min(kBlock, e - p0);
          float best[kBlock];
          uint32_t arg[kBlock] = {};
          std::fill(best, best + kBlock, std::numeric_limits<float>::infinity());
          for (size_t c = 0; c < k; ++c) {
            const float* cr = centroids + c * d;
            for (size_t j = 0; j < pn; ++j) {
              float dd = fn(data + (p0 + j) * stride, cr, d);
              if (dd < best[j]) {
                best[j] = dd;
                arg[j] = static_cast<uint32_t>(c);
              }
            }
          }
          for (size_t j = 0; j < pn; ++j) {
            assign[p0 + j] = arg[j];
            if (dist != nullptr) dist[p0 + j] = best[j];
          }
        }
      },
      256);
}

Status KMeans(const float* data, size_t n, size_t d, size_t stride, size_t k,
              const KMeansOptions& opts, ThreadPool* pool, std::vector<float>* centroids,
              KMeansStats* stats) {
  if (k == 0 || d == 0) return Status::InvalidArgument("k-means needs k > 0 and d > 0");
  if (n == 0) return Status::InvalidArgument("k-means needs training points");
  centroids->assign(k * d, 0.f);
  std::mt19937_64 rng(opts.seed);

  // Fewer points than centroids: use the points themselves (with duplicates).
  if (n <= k) {
    for (size_t c = 0; c < k; ++c) std::memcpy(centroids->data() + c * d, data + (c % n) * stride, d * sizeof(float));
    if (stats != nullptr) *stats = KMeansStats{0, n, 0, 0};
    return Status::OK();
  }

  // Subsample to at most k * max_points_per_centroid points, into a contiguous buffer.
  size_t m = std::min(n, k * std::max<size_t>(1, opts.max_points_per_centroid));
  std::vector<float> sample;
  const float* x = data;
  size_t xs = stride;
  if (m < n) {
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    for (size_t i = 0; i < m; ++i) {
      std::uniform_int_distribution<size_t> u(i, n - 1);
      std::swap(idx[i], idx[u(rng)]);
    }
    std::sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(m));
    sample.resize(m * d);
    for (size_t i = 0; i < m; ++i) std::memcpy(sample.data() + i * d, data + idx[i] * stride, d * sizeof(float));
    x = sample.data();
    xs = d;
  }
  DistFn fn = PickL2(d, xs);

  // Initialization.
  float* cen = centroids->data();
  bool use_pp = opts.kmeanspp && static_cast<double>(k) * static_cast<double>(m) <= 2.0e7;
  if (use_pp) {
    std::vector<float> mind(m, std::numeric_limits<float>::infinity());
    std::uniform_int_distribution<size_t> u0(0, m - 1);
    size_t first = u0(rng);
    std::memcpy(cen, x + first * xs, d * sizeof(float));
    for (size_t c = 1; c < k; ++c) {
      const float* last = cen + (c - 1) * d;
      double total = 0;
      for (size_t i = 0; i < m; ++i) {
        mind[i] = std::min(mind[i], fn(x + i * xs, last, d));
        total += mind[i];
      }
      size_t pick = 0;
      if (total > 0) {
        std::uniform_real_distribution<double> ur(0.0, total);
        double t = ur(rng), acc = 0;
        for (pick = 0; pick + 1 < m; ++pick) {
          acc += mind[pick];
          if (acc >= t) break;
        }
      } else {
        pick = u0(rng);
      }
      std::memcpy(cen + c * d, x + pick * xs, d * sizeof(float));
    }
  } else {
    std::vector<size_t> idx(m);
    std::iota(idx.begin(), idx.end(), 0);
    for (size_t c = 0; c < k; ++c) {
      std::uniform_int_distribution<size_t> u(c, m - 1);
      std::swap(idx[c], idx[u(rng)]);
      std::memcpy(cen + c * d, x + idx[c] * xs, d * sizeof(float));
    }
  }

  std::vector<uint32_t> assign(m);
  std::vector<float> dist(m);
  std::vector<double> sums(k * d);
  std::vector<size_t> counts(k);
  KMeansStats st;
  st.train_points = m;
  constexpr float kEps = 1.0f / 1024.0f;

  for (size_t it = 0; it < opts.iters; ++it) {
    AssignNearest(x, m, d, xs, cen, k, pool, assign.data(), dist.data());
    st.inertia = 0;
    for (size_t i = 0; i < m; ++i) st.inertia += dist[i];
    std::fill(sums.begin(), sums.end(), 0.0);
    std::fill(counts.begin(), counts.end(), 0);
    for (size_t i = 0; i < m; ++i) {
      const float* p = x + i * xs;
      double* s = sums.data() + static_cast<size_t>(assign[i]) * d;
      for (size_t j = 0; j < d; ++j) s[j] += p[j];
      ++counts[assign[i]];
    }
    for (size_t c = 0; c < k; ++c) {
      if (counts[c] == 0) continue;
      double inv = 1.0 / static_cast<double>(counts[c]);
      for (size_t j = 0; j < d; ++j) cen[c * d + j] = static_cast<float>(sums[c * d + j] * inv);
    }
    // Empty clusters: split the largest cluster in two with a symmetric perturbation.
    for (size_t c = 0; c < k; ++c) {
      if (counts[c] != 0) continue;
      size_t big = static_cast<size_t>(std::max_element(counts.begin(), counts.end()) - counts.begin());
      std::uniform_real_distribution<float> noise(-1.f, 1.f);
      for (size_t j = 0; j < d; ++j) {
        float v = cen[big * d + j];
        float delta = kEps * (std::fabs(v) + 1e-3f) * noise(rng);
        cen[c * d + j] = v + delta;
        cen[big * d + j] = v - delta;
      }
      counts[c] = counts[big] / 2;
      counts[big] -= counts[c];
      ++st.empty_reseeds;
    }
    st.iterations = it + 1;
  }
  if (stats != nullptr) *stats = st;
  return Status::OK();
}

}  // namespace vdb
