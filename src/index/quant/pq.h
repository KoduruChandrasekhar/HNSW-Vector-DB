#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "index/quant/kmeans.h"
#include "util/coding.h"
#include "vdb/status.h"
#include "vdb/types.h"

namespace vdb {

class ThreadPool;

// Product quantization (Jegou, Douze, Schmid): the real dimensions are split into m
// sub-vectors of dim / m floats, each quantized to one of 256 centroids (one byte).
// Searches use asymmetric distance computation: one m x 256 lookup table per query,
// then m table lookups and additions per encoded vector.
class ProductQuantizer {
 public:
  static constexpr size_t kCentroids = 256;

  ProductQuantizer() = default;
  ProductQuantizer(size_t dim, size_t m);

  // Trains one 256-centroid codebook per subspace on n rows (row stride `stride` floats).
  Status train(const float* data, size_t n, size_t stride, const KMeansOptions& opts,
               ThreadPool* pool);
  void encode(const float* x, uint8_t* code) const;
  void decode(const uint8_t* code, float* x) const;  // writes dim floats

  // L2: T[j][c] = ||q_j - C_j[c]||^2.  Inner product: T[j][c] = -q_j . C_j[c].
  void compute_table(const float* q, Metric metric, float* table) const;
  float adc(const float* table, const uint8_t* code) const {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    size_t j = 0;
    for (; j + 4 <= m_; j += 4) {
      s0 += table[(j + 0) * kCentroids + code[j + 0]];
      s1 += table[(j + 1) * kCentroids + code[j + 1]];
      s2 += table[(j + 2) * kCentroids + code[j + 2]];
      s3 += table[(j + 3) * kCentroids + code[j + 3]];
    }
    for (; j < m_; ++j) s0 += table[j * kCentroids + code[j]];
    return (s0 + s1) + (s2 + s3);
  }

  size_t dim() const { return dim_; }
  size_t m() const { return m_; }
  size_t dsub() const { return dsub_; }
  size_t code_size() const { return m_; }
  size_t table_size() const { return m_ * kCentroids; }
  bool trained() const { return !centroids_.empty(); }
  const float* centroid(size_t j, size_t c) const {
    return centroids_.data() + (j * kCentroids + c) * dsub_;
  }
  size_t memory_bytes() const { return centroids_.size() * sizeof(float); }

  void serialize(std::string* out) const;
  bool deserialize(ByteReader& r);

 private:
  size_t dim_ = 0;
  size_t m_ = 0;
  size_t dsub_ = 0;
  std::vector<float> centroids_;  // m * 256 * dsub
};

}  // namespace vdb
