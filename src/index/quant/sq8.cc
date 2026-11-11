#include "index/quant/sq8.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vdb {

void Sq8Quantizer::train(const VectorView& v) {
  dim_ = v.dim;
  padded_ = v.padded_dim;
  vmin_.assign(padded_, 0.f);
  scale_.assign(padded_, 0.f);
  if (v.rows == 0) return;
  std::vector<float> vmax(dim_, -std::numeric_limits<float>::infinity());
  std::fill(vmin_.begin(), vmin_.begin() + static_cast<std::ptrdiff_t>(dim_),
            std::numeric_limits<float>::infinity());
  for (size_t r = 0; r < v.rows; ++r) {
    const float* x = v.row(r);
    for (size_t i = 0; i < dim_; ++i) {
      vmin_[i] = std::min(vmin_[i], x[i]);
      vmax[i] = std::max(vmax[i], x[i]);
    }
  }
  for (size_t i = 0; i < dim_; ++i) scale_[i] = (vmax[i] - vmin_[i]) / 255.f;
}

void Sq8Quantizer::encode(const float* x, uint8_t* code) const {
  for (size_t i = 0; i < padded_; ++i) {
    if (i >= dim_ || scale_[i] <= 0.f) {
      code[i] = 0;
      continue;
    }
    float c = std::nearbyint((x[i] - vmin_[i]) / scale_[i]);
    code[i] = static_cast<uint8_t>(std::clamp(c, 0.f, 255.f));
  }
}

void Sq8Quantizer::decode(const uint8_t* code, float* x) const {
  for (size_t i = 0; i < padded_; ++i) x[i] = vmin_[i] + static_cast<float>(code[i]) * scale_[i];
}

void Sq8Quantizer::serialize(std::string* out) const {
  PutU32(out, static_cast<uint32_t>(dim_));
  PutU32(out, static_cast<uint32_t>(padded_));
  PutBytes(out, vmin_.data(), padded_ * sizeof(float));
  PutBytes(out, scale_.data(), padded_ * sizeof(float));
}

bool Sq8Quantizer::deserialize(ByteReader& r) {
  uint32_t d = 0, p = 0;
  if (!r.GetU32(&d) || !r.GetU32(&p) || d > p || p % 16 != 0) return false;
  std::string_view a, b;
  if (!r.GetBytes(size_t(p) * sizeof(float), &a) || !r.GetBytes(size_t(p) * sizeof(float), &b)) {
    return false;
  }
  dim_ = d;
  padded_ = p;
  vmin_.resize(p);
  scale_.resize(p);
  std::memcpy(vmin_.data(), a.data(), a.size());
  std::memcpy(scale_.data(), b.data(), b.size());
  return true;
}

}  // namespace vdb
