#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vdb/status.h"

namespace vdb {

class ThreadPool;

struct KMeansOptions {
  size_t iters = 25;  // FAISS's default
  uint64_t seed = 1234;
  // FAISS warns below 39 points per centroid and subsamples above 256.
  size_t max_points_per_centroid = 256;
  // k-means++ initialization when k * n_train is small enough; random sample otherwise.
  bool kmeanspp = true;
};

struct KMeansStats {
  double inertia = 0;  // sum of squared distances after the last assignment
  size_t train_points = 0;
  size_t empty_reseeds = 0;
  size_t iterations = 0;
};

// Lloyd's algorithm on n rows of d floats (row stride `stride` floats). Writes k * d
// centroid floats. Empty clusters are re-seeded from the largest cluster with a tiny
// perturbation, so no centroid is ever NaN.
Status KMeans(const float* data, size_t n, size_t d, size_t stride, size_t k,
              const KMeansOptions& opts, ThreadPool* pool, std::vector<float>* centroids,
              KMeansStats* stats = nullptr);

// Nearest centroid (k * d floats) of each row; dist may be null.
void AssignNearest(const float* data, size_t n, size_t d, size_t stride, const float* centroids,
                   size_t k, ThreadPool* pool, uint32_t* assign, float* dist);

}  // namespace vdb
