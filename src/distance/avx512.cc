// AVX-512F kernels. This file alone is compiled with -mavx512f. One 512-bit register
// holds 16 floats, so a padded row needs no masking; two accumulators hide FMA latency.

#include <immintrin.h>

#include "distance/distance.h"

namespace vdb {
namespace {

inline __m512 Decode16(const uint8_t* c, const float* vmin, const float* scale) {
  __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c));
  __m512 x = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(bytes));
  return _mm512_fmadd_ps(x, _mm512_loadu_ps(scale), _mm512_loadu_ps(vmin));
}

}  // namespace

float l2_sq_avx512(const float* a, const float* b, size_t d) {
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  size_t i = 0;
  for (; i + 32 <= d; i += 32) {
    __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
    __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16));
    acc0 = _mm512_fmadd_ps(d0, d0, acc0);
    acc1 = _mm512_fmadd_ps(d1, d1, acc1);
  }
  if (i < d) {
    __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
    acc0 = _mm512_fmadd_ps(d0, d0, acc0);
  }
  return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

float dot_avx512(const float* a, const float* b, size_t d) {
  __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
  size_t i = 0;
  for (; i + 32 <= d; i += 32) {
    acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
    acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
  }
  if (i < d) acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
  return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

float ip_dist_avx512(const float* a, const float* b, size_t d) { return -dot_avx512(a, b, d); }

float sq8_l2_avx512(const float* q, const uint8_t* c, const float* vmin, const float* scale,
                    size_t d) {
  __m512 acc = _mm512_setzero_ps();
  for (size_t i = 0; i < d; i += 16) {
    __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(q + i), Decode16(c + i, vmin + i, scale + i));
    acc = _mm512_fmadd_ps(diff, diff, acc);
  }
  return _mm512_reduce_add_ps(acc);
}

float sq8_ip_avx512(const float* q, const uint8_t* c, const float* vmin, const float* scale,
                    size_t d) {
  __m512 acc = _mm512_setzero_ps();
  for (size_t i = 0; i < d; i += 16) {
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(q + i), Decode16(c + i, vmin + i, scale + i), acc);
  }
  return -_mm512_reduce_add_ps(acc);
}

}  // namespace vdb
