// AVX2 + FMA kernels. This file alone is compiled with -mavx2 -mfma. Two accumulators
// hide the latency of each FMA. All kernels require d % 16 == 0.

#include <immintrin.h>

#include "distance/distance.h"

namespace vdb {
namespace {

inline float HorizontalSum(__m256 acc) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}

// Widens 8 code bytes to floats and decodes them: vmin + code * scale.
inline __m256 Decode8(__m128i bytes, const float* vmin, const float* scale) {
  __m256 c = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
  return _mm256_fmadd_ps(c, _mm256_loadu_ps(scale), _mm256_loadu_ps(vmin));
}

}  // namespace

float l2_sq_avx2(const float* a, const float* b, size_t d) {
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  for (size_t i = 0; i < d; i += 16) {
    __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    acc1 = _mm256_fmadd_ps(d1, d1, acc1);
  }
  return HorizontalSum(_mm256_add_ps(acc0, acc1));
}

float dot_avx2(const float* a, const float* b, size_t d) {
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  for (size_t i = 0; i < d; i += 16) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
  }
  return HorizontalSum(_mm256_add_ps(acc0, acc1));
}

float ip_dist_avx2(const float* a, const float* b, size_t d) { return -dot_avx2(a, b, d); }

float sq8_l2_avx2(const float* q, const uint8_t* c, const float* vmin, const float* scale,
                  size_t d) {
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  for (size_t i = 0; i < d; i += 16) {
    __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c + i));
    __m256 x0 = Decode8(bytes, vmin + i, scale + i);
    __m256 x1 = Decode8(_mm_srli_si128(bytes, 8), vmin + i + 8, scale + i + 8);
    __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(q + i), x0);
    __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(q + i + 8), x1);
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    acc1 = _mm256_fmadd_ps(d1, d1, acc1);
  }
  return HorizontalSum(_mm256_add_ps(acc0, acc1));
}

float sq8_ip_avx2(const float* q, const uint8_t* c, const float* vmin, const float* scale,
                  size_t d) {
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  for (size_t i = 0; i < d; i += 16) {
    __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c + i));
    __m256 x0 = Decode8(bytes, vmin + i, scale + i);
    __m256 x1 = Decode8(_mm_srli_si128(bytes, 8), vmin + i + 8, scale + i + 8);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), x0, acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + i + 8), x1, acc1);
  }
  return -HorizontalSum(_mm256_add_ps(acc0, acc1));
}

}  // namespace vdb
