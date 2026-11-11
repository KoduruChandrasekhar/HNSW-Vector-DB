#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "distance/distance.h"
#include "index/quant/sq8.h"
#include "util/aligned_buffer.h"
#include "util/crc32c.h"

namespace vdb {
namespace {

// Relative error against a magnitude that cannot cancel (sum of |terms|).
void ExpectClose(float got, float want, double magnitude, const std::string& what) {
  double tol = 1e-4 * std::max(1.0, magnitude);
  EXPECT_NEAR(got, want, tol) << what;
}

TEST(DistanceTest, SimdMatchesScalarAtEveryDimension) {
  std::mt19937_64 rng(7);
  std::normal_distribution<float> g(0.f, 1.f);
  const auto kernels = AvailableKernels();
  ASSERT_FALSE(kernels.empty());
  for (size_t dim = 1; dim <= 1024; ++dim) {
    size_t pd = PaddedDim(dim);
    AlignedBuffer a(pd * sizeof(float)), b(pd * sizeof(float));
    double mag_l2 = 0, mag_dot = 0;
    for (size_t i = 0; i < dim; ++i) {
      a.floats()[i] = g(rng);
      b.floats()[i] = g(rng);
      double t = a.floats()[i] - b.floats()[i];
      mag_l2 += t * t;
      mag_dot += std::fabs(a.floats()[i] * b.floats()[i]);
    }
    const KernelSet& ref = ScalarKernels();
    float l2 = ref.l2(a.floats(), b.floats(), pd);
    float dot = ref.dot(a.floats(), b.floats(), pd);
    for (const KernelSet* k : kernels) {
      std::string tag = std::string(k->name) + " dim=" + std::to_string(dim);
      ExpectClose(k->l2(a.floats(), b.floats(), pd), l2, mag_l2, "l2 " + tag);
      ExpectClose(k->dot(a.floats(), b.floats(), pd), dot, mag_dot, "dot " + tag);
      ExpectClose(k->ip(a.floats(), b.floats(), pd), -dot, mag_dot, "ip " + tag);
    }
    ExpectClose(L2SqAny(a.floats(), b.floats(), dim), l2, mag_l2, "L2SqAny");
    ExpectClose(DotAny(a.floats(), b.floats(), dim), dot, mag_dot, "DotAny");
  }
}

TEST(DistanceTest, Sq8KernelsMatchScalar) {
  std::mt19937_64 rng(11);
  std::uniform_real_distribution<float> u(-2.f, 2.f);
  std::uniform_int_distribution<int> byte(0, 255);
  for (size_t dim : {1u, 7u, 16u, 100u, 128u, 384u, 960u}) {
    size_t pd = PaddedDim(dim);
    std::vector<float> q(pd, 0.f), vmin(pd, 0.f), scale(pd, 0.f);
    std::vector<uint8_t> code(pd, 0);
    double mag = 0;
    for (size_t i = 0; i < dim; ++i) {
      q[i] = u(rng);
      vmin[i] = u(rng);
      scale[i] = std::fabs(u(rng)) / 255.f;
      code[i] = static_cast<uint8_t>(byte(rng));
      float x = vmin[i] + code[i] * scale[i];
      mag += std::fabs(q[i] * x) + (q[i] - x) * (q[i] - x);
    }
    float l2 = sq8_l2_scalar(q.data(), code.data(), vmin.data(), scale.data(), pd);
    float ip = sq8_ip_scalar(q.data(), code.data(), vmin.data(), scale.data(), pd);
    for (const KernelSet* k : AvailableKernels()) {
      ExpectClose(k->sq8_l2(q.data(), code.data(), vmin.data(), scale.data(), pd), l2, mag, k->name);
      ExpectClose(k->sq8_ip(q.data(), code.data(), vmin.data(), scale.data(), pd), ip, mag, k->name);
    }
  }
}

TEST(DistanceTest, ZeroPaddingDoesNotChangeDistances) {
  std::vector<float> a(112, 0.f), b(112, 0.f);
  for (size_t i = 0; i < 100; ++i) {
    a[i] = static_cast<float>(i) * 0.01f;
    b[i] = 1.f - static_cast<float>(i) * 0.01f;
  }
  EXPECT_NEAR(Kernels().l2(a.data(), b.data(), 112), l2_sq_scalar(a.data(), b.data(), 100), 1e-3);
  EXPECT_NEAR(Kernels().dot(a.data(), b.data(), 112), dot_scalar(a.data(), b.data(), 100), 1e-3);
}

TEST(DistanceTest, MetricConvention) {
  float a[16] = {1, 0}, b[16] = {0.5f, 0.5f};
  EXPECT_FLOAT_EQ(MetricDistance(Metric::L2)(a, b, 16), 0.5f);
  EXPECT_FLOAT_EQ(MetricDistance(Metric::InnerProduct)(a, b, 16), -0.5f);  // smaller = closer
  EXPECT_FLOAT_EQ(MetricDistance(Metric::Cosine)(a, b, 16), -0.5f);
}

TEST(Crc32cTest, KnownValuesAndHardwareMatchesPortable) {
  EXPECT_EQ(crc32c::Value("123456789", 9), 0xE3069283u);
  EXPECT_EQ(crc32c::Value("", 0), 0u);
  std::string data(10000, '\0');
  std::mt19937 rng(3);
  for (auto& c : data) c = static_cast<char>(rng());
  for (size_t n : {0u, 1u, 7u, 8u, 9u, 63u, 64u, 1000u, 10000u}) {
    EXPECT_EQ(crc32c::Value(data.data(), n), crc32c::ExtendPortable(0, data.data(), n)) << n;
  }
  uint32_t split = crc32c::Extend(crc32c::Value(data.data(), 1234), data.data() + 1234, data.size() - 1234);
  EXPECT_EQ(split, crc32c::Value(data));
}

TEST(Sq8Test, ReconstructionWithinHalfStep) {
  std::mt19937_64 rng(5);
  std::normal_distribution<float> g(0.f, 3.f);
  const size_t n = 2000, dim = 100, pd = PaddedDim(dim);
  AlignedBuffer buf(n * pd * sizeof(float));
  for (size_t i = 0; i < n; ++i)
    for (size_t d = 0; d < dim; ++d) buf.floats()[i * pd + d] = g(rng);
  VectorView v{buf.floats(), n, dim, pd};
  Sq8Quantizer sq;
  sq.train(v);
  std::vector<uint8_t> code(pd);
  std::vector<float> back(pd);
  for (size_t i = 0; i < n; ++i) {
    sq.encode(v.row(i), code.data());
    sq.decode(code.data(), back.data());
    for (size_t d = 0; d < dim; ++d) {
      ASSERT_LE(std::fabs(back[d] - v.row(i)[d]), sq.half_step(d) * 1.0001f + 1e-6f) << i << "," << d;
    }
    for (size_t d = dim; d < pd; ++d) ASSERT_EQ(back[d], 0.f);
  }
}

}  // namespace
}  // namespace vdb
