// Scalar reference kernels. Without -ffast-math the compiler keeps the float additions
// in order and does not vectorize these loops, which is what makes them a reference.

#include "distance/distance.h"

namespace vdb {

float l2_sq_scalar(const float* a, const float* b, size_t d) {
  float s = 0.f;
  for (size_t i = 0; i < d; ++i) {
    float t = a[i] - b[i];
    s += t * t;
  }
  return s;
}

float dot_scalar(const float* a, const float* b, size_t d) {
  float s = 0.f;
  for (size_t i = 0; i < d; ++i) s += a[i] * b[i];
  return s;
}

float ip_dist_scalar(const float* a, const float* b, size_t d) { return -dot_scalar(a, b, d); }

float sq8_l2_scalar(const float* q, const uint8_t* c, const float* vmin, const float* scale,
                    size_t d) {
  float s = 0.f;
  for (size_t i = 0; i < d; ++i) {
    float x = vmin[i] + static_cast<float>(c[i]) * scale[i];
    float t = q[i] - x;
    s += t * t;
  }
  return s;
}

float sq8_ip_scalar(const float* q, const uint8_t* c, const float* vmin, const float* scale,
                    size_t d) {
  float s = 0.f;
  for (size_t i = 0; i < d; ++i) s += q[i] * (vmin[i] + static_cast<float>(c[i]) * scale[i]);
  return -s;
}

float L2SqAny(const float* a, const float* b, size_t d) {
  // Four independent accumulators: faster than the reference and still exact enough.
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  size_t i = 0;
  for (; i + 4 <= d; i += 4) {
    float t0 = a[i] - b[i], t1 = a[i + 1] - b[i + 1];
    float t2 = a[i + 2] - b[i + 2], t3 = a[i + 3] - b[i + 3];
    s0 += t0 * t0;
    s1 += t1 * t1;
    s2 += t2 * t2;
    s3 += t3 * t3;
  }
  for (; i < d; ++i) {
    float t = a[i] - b[i];
    s0 += t * t;
  }
  return (s0 + s1) + (s2 + s3);
}

float DotAny(const float* a, const float* b, size_t d) {
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  size_t i = 0;
  for (; i + 4 <= d; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  for (; i < d; ++i) s0 += a[i] * b[i];
  return (s0 + s1) + (s2 + s3);
}

}  // namespace vdb
