// NEON kernels for Apple Silicon and other ARM64 machines. Four accumulators of four
// floats cover one 16-float block per iteration. All kernels require d % 16 == 0.

#include <arm_neon.h>

#include "distance/distance.h"

namespace vdb {
namespace {

// Decodes 4 code values (already widened to u32) as vmin + code * scale.
inline float32x4_t Decode4(uint32x4_t c, const float* vmin, const float* scale) {
  return vfmaq_f32(vld1q_f32(vmin), vcvtq_f32_u32(c), vld1q_f32(scale));
}

}  // namespace

float l2_sq_neon(const float* a, const float* b, size_t d) {
  float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0), acc2 = vdupq_n_f32(0),
              acc3 = vdupq_n_f32(0);
  for (size_t i = 0; i < d; i += 16) {
    float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    acc0 = vfmaq_f32(acc0, d0, d0);
    acc1 = vfmaq_f32(acc1, d1, d1);
    acc2 = vfmaq_f32(acc2, d2, d2);
    acc3 = vfmaq_f32(acc3, d3, d3);
  }
  return vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
}

float dot_neon(const float* a, const float* b, size_t d) {
  float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0), acc2 = vdupq_n_f32(0),
              acc3 = vdupq_n_f32(0);
  for (size_t i = 0; i < d; i += 16) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  return vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
}

float ip_dist_neon(const float* a, const float* b, size_t d) { return -dot_neon(a, b, d); }

float sq8_l2_neon(const float* q, const uint8_t* c, const float* vmin, const float* scale,
                  size_t d) {
  float32x4_t acc = vdupq_n_f32(0);
  for (size_t i = 0; i < d; i += 16) {
    uint8x16_t bytes = vld1q_u8(c + i);
    uint16x8_t lo = vmovl_u8(vget_low_u8(bytes)), hi = vmovl_u8(vget_high_u8(bytes));
    uint32x4_t w[4] = {vmovl_u16(vget_low_u16(lo)), vmovl_u16(vget_high_u16(lo)),
                       vmovl_u16(vget_low_u16(hi)), vmovl_u16(vget_high_u16(hi))};
    for (int j = 0; j < 4; ++j) {
      float32x4_t x = Decode4(w[j], vmin + i + 4 * j, scale + i + 4 * j);
      float32x4_t diff = vsubq_f32(vld1q_f32(q + i + 4 * j), x);
      acc = vfmaq_f32(acc, diff, diff);
    }
  }
  return vaddvq_f32(acc);
}

float sq8_ip_neon(const float* q, const uint8_t* c, const float* vmin, const float* scale,
                  size_t d) {
  float32x4_t acc = vdupq_n_f32(0);
  for (size_t i = 0; i < d; i += 16) {
    uint8x16_t bytes = vld1q_u8(c + i);
    uint16x8_t lo = vmovl_u8(vget_low_u8(bytes)), hi = vmovl_u8(vget_high_u8(bytes));
    uint32x4_t w[4] = {vmovl_u16(vget_low_u16(lo)), vmovl_u16(vget_high_u16(lo)),
                       vmovl_u16(vget_low_u16(hi)), vmovl_u16(vget_high_u16(hi))};
    for (int j = 0; j < 4; ++j) {
      float32x4_t x = Decode4(w[j], vmin + i + 4 * j, scale + i + 4 * j);
      acc = vfmaq_f32(acc, vld1q_f32(q + i + 4 * j), x);
    }
  }
  return -vaddvq_f32(acc);
}

}  // namespace vdb
