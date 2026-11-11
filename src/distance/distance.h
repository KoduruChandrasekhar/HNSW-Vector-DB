#pragma once

// Distance kernels and runtime dispatch.
//
// Every SIMD kernel requires d % 16 == 0: rows are zero-padded to a multiple of 16
// floats, and zeros change neither L2 nor inner product, so no tail loop is needed.
// Kernels are chosen once at startup from the CPU's features; set VDB_KERNELS=scalar
// (or avx2, avx512, neon) to override.

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "vdb/types.h"

namespace vdb {

using DistFn = float (*)(const float*, const float*, size_t);
// Float query against an SQ8 code: x_i = vmin_i + code_i * scale_i.
using Sq8DistFn = float (*)(const float* q, const uint8_t* code, const float* vmin,
                            const float* scale, size_t d);

struct KernelSet {
  const char* name;
  DistFn l2;      // squared L2
  DistFn dot;     // dot product
  DistFn ip;      // minus the dot product ("smaller is closer")
  Sq8DistFn sq8_l2;
  Sq8DistFn sq8_ip;
};

const KernelSet& Kernels();        // best kernels for this CPU (or VDB_KERNELS)
const KernelSet& ScalarKernels();  // the reference every SIMD kernel is tested against
// Every compiled-in kernel set the running CPU supports, scalar first.
std::vector<const KernelSet*> AvailableKernels();
const KernelSet* FindKernels(std::string_view name);

DistFn pick_l2(bool force_scalar);
DistFn pick_ip(bool force_scalar);
// L2 -> squared L2; InnerProduct and Cosine -> minus the dot product.
DistFn MetricDistance(Metric m, bool force_scalar = false);
Sq8DistFn MetricSq8Distance(Metric m, bool force_scalar = false);

// Scalar helpers that accept any length (PQ sub-vectors, k-means on subspaces).
float L2SqAny(const float* a, const float* b, size_t d);
float DotAny(const float* a, const float* b, size_t d);

// Individual kernels, exposed for tests and microbenchmarks.
float l2_sq_scalar(const float* a, const float* b, size_t d);
float dot_scalar(const float* a, const float* b, size_t d);
float ip_dist_scalar(const float* a, const float* b, size_t d);
float sq8_l2_scalar(const float* q, const uint8_t* c, const float* vmin, const float* scale,
                    size_t d);
float sq8_ip_scalar(const float* q, const uint8_t* c, const float* vmin, const float* scale,
                    size_t d);

#if defined(VDB_HAVE_X86_KERNELS)
float l2_sq_avx2(const float* a, const float* b, size_t d);
float dot_avx2(const float* a, const float* b, size_t d);
float ip_dist_avx2(const float* a, const float* b, size_t d);
float sq8_l2_avx2(const float* q, const uint8_t* c, const float* vmin, const float* scale,
                  size_t d);
float sq8_ip_avx2(const float* q, const uint8_t* c, const float* vmin, const float* scale,
                  size_t d);
float l2_sq_avx512(const float* a, const float* b, size_t d);
float dot_avx512(const float* a, const float* b, size_t d);
float ip_dist_avx512(const float* a, const float* b, size_t d);
float sq8_l2_avx512(const float* q, const uint8_t* c, const float* vmin, const float* scale,
                    size_t d);
float sq8_ip_avx512(const float* q, const uint8_t* c, const float* vmin, const float* scale,
                    size_t d);
#endif

#if defined(VDB_HAVE_NEON_KERNELS)
float l2_sq_neon(const float* a, const float* b, size_t d);
float dot_neon(const float* a, const float* b, size_t d);
float ip_dist_neon(const float* a, const float* b, size_t d);
float sq8_l2_neon(const float* q, const uint8_t* c, const float* vmin, const float* scale,
                  size_t d);
float sq8_ip_neon(const float* q, const uint8_t* c, const float* vmin, const float* scale,
                  size_t d);
#endif

inline void Prefetch(const void* p) { __builtin_prefetch(p, 0, 3); }

}  // namespace vdb
