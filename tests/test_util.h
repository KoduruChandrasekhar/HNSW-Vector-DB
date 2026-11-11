#pragma once

// Shared helpers: synthetic clustered data, brute-force ground truth, recall, temp dirs.

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "distance/distance.h"
#include "storage/env.h"
#include "util/aligned_buffer.h"
#include "util/dataset.h"
#include "util/topk.h"
#include "vdb/types.h"

namespace vdb::test {

// Gaussian blobs around `clusters` random centers: much closer to real embeddings than
// uniform noise, and hard enough that a broken graph shows up in recall. The centers
// depend only on (clusters, dim), so base vectors and queries made with different seeds
// come from the same distribution.
inline Dataset MakeClustered(size_t n, size_t dim, size_t clusters, uint64_t seed,
                             bool normalize = false, float spread = 0.35f) {
  std::mt19937_64 crng(clusters * 1000003ull + dim);
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> g(0.f, 1.f);
  std::vector<float> centers(clusters * dim);
  for (auto& c : centers) c = g(crng);
  Dataset ds;
  ds.rows = n;
  ds.dim = dim;
  ds.padded_dim = PaddedDim(dim);
  ds.buffer.reset(n * ds.padded_dim * sizeof(float));
  std::uniform_int_distribution<size_t> pick(0, clusters - 1);
  for (size_t i = 0; i < n; ++i) {
    size_t c = pick(rng);
    float* row = ds.buffer.floats() + i * ds.padded_dim;
    for (size_t d = 0; d < dim; ++d) row[d] = centers[c * dim + d] + spread * g(rng);
    if (normalize) NormalizeVector(row, dim);
  }
  return ds;
}

// Exact top-k ids for every query.
inline std::vector<std::vector<InternalId>> GroundTruth(const Dataset& base, const Dataset& queries,
                                                        size_t k, Metric metric) {
  DistFn fn = MetricDistance(metric, /*force_scalar=*/true);
  std::vector<std::vector<InternalId>> gt(queries.rows);
  for (size_t q = 0; q < queries.rows; ++q) {
    TopK top(k);
    for (size_t i = 0; i < base.rows; ++i)
      top.push(fn(queries.row(q), base.row(i), base.padded_dim), static_cast<InternalId>(i));
    for (const auto& n : top.sorted()) gt[q].push_back(n.id);
  }
  return gt;
}

inline double Recall(const std::vector<Neighbor>& found, const std::vector<InternalId>& truth,
                     size_t k) {
  std::set<InternalId> t(truth.begin(), truth.begin() + static_cast<std::ptrdiff_t>(std::min(k, truth.size())));
  size_t hit = 0;
  for (size_t i = 0; i < std::min(k, found.size()); ++i) hit += t.count(found[i].id);
  return static_cast<double>(hit) / static_cast<double>(k);
}

// A fresh directory under $TEST_TMPDIR (or /tmp), removed by the destructor.
class TempDir {
 public:
  TempDir() {
    const char* base = std::getenv("TEST_TMPDIR");
    std::string tmpl = std::string(base != nullptr ? base : "/tmp") + "/vdb_test_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    char* p = ::mkdtemp(buf.data());
    path_ = p != nullptr ? p : tmpl;
  }
  ~TempDir() { Env::Default()->remove_dir_all(path_); }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  const std::string& path() const { return path_; }
  std::string sub(const std::string& name) const { return path_ + "/" + name; }

 private:
  std::string path_;
};

inline std::vector<float> RandomVector(size_t dim, std::mt19937_64& rng) {
  std::normal_distribution<float> g(0.f, 1.f);
  std::vector<float> v(dim);
  for (auto& x : v) x = g(rng);
  return v;
}

}  // namespace vdb::test
