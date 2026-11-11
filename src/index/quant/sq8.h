#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "util/aligned_buffer.h"
#include "util/coding.h"

namespace vdb {

// Scalar quantization to one byte per dimension: code = round((x - min) / (max - min) * 255).
// Reconstruction error is at most half a step, (max - min) / 510, for values inside the
// trained range. Padding dimensions get min = scale = 0 and decode to 0.
class Sq8Quantizer {
 public:
  void train(const VectorView& v);
  void encode(const float* x, uint8_t* code) const;  // padded_dim values
  void decode(const uint8_t* code, float* x) const;
  size_t dim() const { return dim_; }
  size_t padded_dim() const { return padded_; }
  const float* vmin() const { return vmin_.data(); }
  const float* scale() const { return scale_.data(); }
  // Largest reconstruction error for dimension i: half a quantization step.
  float half_step(size_t i) const { return scale_[i] * 0.5f; }
  size_t memory_bytes() const { return (vmin_.size() + scale_.size()) * sizeof(float); }

  void serialize(std::string* out) const;
  bool deserialize(ByteReader& r);

 private:
  size_t dim_ = 0;
  size_t padded_ = 0;
  std::vector<float> vmin_;
  std::vector<float> scale_;  // (max - min) / 255
};

}  // namespace vdb
