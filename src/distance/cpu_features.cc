// Picks the distance kernels once, from the CPU's features at runtime. The binary is
// never built with -march=native, so it runs on CPUs without AVX2 too.

#include <cstdlib>
#include <string>

#include "distance/distance.h"

namespace vdb {
namespace {

const KernelSet kScalar{"scalar",      l2_sq_scalar,  dot_scalar,
                        ip_dist_scalar, sq8_l2_scalar, sq8_ip_scalar};

#if defined(VDB_HAVE_X86_KERNELS)
const KernelSet kAvx2{"avx2", l2_sq_avx2, dot_avx2, ip_dist_avx2, sq8_l2_avx2, sq8_ip_avx2};
const KernelSet kAvx512{"avx512",       l2_sq_avx512,  dot_avx512,
                        ip_dist_avx512, sq8_l2_avx512, sq8_ip_avx512};
bool CpuHasAvx2() { return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"); }
bool CpuHasAvx512() { return CpuHasAvx2() && __builtin_cpu_supports("avx512f"); }
#endif

#if defined(VDB_HAVE_NEON_KERNELS)
const KernelSet kNeon{"neon", l2_sq_neon, dot_neon, ip_dist_neon, sq8_l2_neon, sq8_ip_neon};
#endif

const KernelSet& SelectBest() {
  std::vector<const KernelSet*> avail = AvailableKernels();
  if (const char* env = std::getenv("VDB_KERNELS"); env != nullptr && *env != '\0') {
    if (const KernelSet* k = FindKernels(env)) return *k;
  }
  return *avail.back();  // AvailableKernels() lists the fastest last
}

}  // namespace

std::vector<const KernelSet*> AvailableKernels() {
  std::vector<const KernelSet*> v{&kScalar};
#if defined(VDB_HAVE_X86_KERNELS)
  if (CpuHasAvx2()) v.push_back(&kAvx2);
  if (CpuHasAvx512()) v.push_back(&kAvx512);
#endif
#if defined(VDB_HAVE_NEON_KERNELS)
  v.push_back(&kNeon);
#endif
  return v;
}

const KernelSet* FindKernels(std::string_view name) {
  for (const KernelSet* k : AvailableKernels())
    if (name == k->name) return k;
  return nullptr;
}

const KernelSet& Kernels() {
  static const KernelSet& best = SelectBest();
  return best;
}

const KernelSet& ScalarKernels() { return kScalar; }

DistFn pick_l2(bool force_scalar) { return force_scalar ? kScalar.l2 : Kernels().l2; }
DistFn pick_ip(bool force_scalar) { return force_scalar ? kScalar.ip : Kernels().ip; }

DistFn MetricDistance(Metric m, bool force_scalar) {
  return m == Metric::L2 ? pick_l2(force_scalar) : pick_ip(force_scalar);
}

Sq8DistFn MetricSq8Distance(Metric m, bool force_scalar) {
  const KernelSet& k = force_scalar ? kScalar : Kernels();
  return m == Metric::L2 ? k.sq8_l2 : k.sq8_ip;
}

}  // namespace vdb
