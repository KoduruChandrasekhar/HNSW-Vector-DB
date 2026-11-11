// Microbenchmarks (Google Benchmark): distance kernels at 100, 128, 384 and 960 dims for
// every kernel set this CPU supports, the top-k heap, SQ8 decode-in-loop and PQ scans.
//
//   ./vdb_micro_bench --benchmark_out=bench/results/micro.json --benchmark_out_format=json
//
// Note the scalar row: without -ffast-math the compiler will not reorder the float
// additions, so the reference loop stays scalar. Compare with the SIMD rows.

#include <benchmark/benchmark.h>

#include <random>

#include "distance/distance.h"
#include "index/quant/pq.h"
#include "util/aligned_buffer.h"
#include "util/topk.h"

namespace vdb {
namespace {

void FillRandom(float* p, size_t n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> g;
  for (size_t i = 0; i < n; ++i) p[i] = g(rng);
}

void BM_Distance(benchmark::State& state, const KernelSet* k, bool l2) {
  const size_t dim = static_cast<size_t>(state.range(0));
  const size_t pd = PaddedDim(dim);
  // Many rows so the working set is realistic (and not one cache-resident pair).
  const size_t rows = 4096;
  AlignedBuffer q(pd * sizeof(float)), base(rows * pd * sizeof(float));
  FillRandom(q.floats(), dim, 1);
  for (size_t r = 0; r < rows; ++r) FillRandom(base.floats() + r * pd, dim, r + 2);
  DistFn fn = l2 ? k->l2 : k->dot;
  size_t r = 0;
  float sink = 0;
  for (auto _ : state) {
    sink += fn(q.floats(), base.floats() + r * pd, pd);
    r = (r + 1) & (rows - 1);
  }
  benchmark::DoNotOptimize(sink);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(dim * sizeof(float)));
}

void BM_Sq8(benchmark::State& state, const KernelSet* k) {
  const size_t dim = static_cast<size_t>(state.range(0));
  const size_t pd = PaddedDim(dim);
  std::vector<float> q(pd, 0.f), vmin(pd, 0.1f), scale(pd, 0.01f);
  FillRandom(q.data(), dim, 3);
  std::vector<uint8_t> codes(4096 * pd);
  std::mt19937 rng(4);
  for (auto& c : codes) c = static_cast<uint8_t>(rng());
  size_t r = 0;
  float sink = 0;
  for (auto _ : state) {
    sink += k->sq8_l2(q.data(), codes.data() + r * pd, vmin.data(), scale.data(), pd);
    r = (r + 1) & 4095;
  }
  benchmark::DoNotOptimize(sink);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(dim));
}

void BM_TopK(benchmark::State& state) {
  std::vector<float> d(1 << 16);
  FillRandom(d.data(), d.size(), 5);
  for (auto _ : state) {
    TopK top(static_cast<size_t>(state.range(0)));
    for (size_t i = 0; i < d.size(); ++i) top.push(d[i], static_cast<InternalId>(i));
    benchmark::DoNotOptimize(top.raw().data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(d.size()));
}
BENCHMARK(BM_TopK)->Arg(10)->Arg(100);

void BM_PqScan(benchmark::State& state) {
  const size_t m = static_cast<size_t>(state.range(0));
  const size_t n = 100000;
  ProductQuantizer pq(m * 8, m);
  std::vector<float> train(5000 * m * 8);
  FillRandom(train.data(), train.size(), 6);
  KMeansOptions ko;
  ko.iters = 2;
  if (!pq.train(train.data(), 5000, m * 8, ko, nullptr).ok()) state.SkipWithError("train failed");
  std::vector<uint8_t> codes(n * m);
  std::mt19937 rng(7);
  for (auto& c : codes) c = static_cast<uint8_t>(rng());
  std::vector<float> table(pq.table_size());
  pq.compute_table(train.data(), Metric::L2, table.data());
  for (auto _ : state) {
    TopK top(100);
    for (size_t i = 0; i < n; ++i) top.push(pq.adc(table.data(), codes.data() + i * m), static_cast<InternalId>(i));
    benchmark::DoNotOptimize(top.raw().data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));
}
BENCHMARK(BM_PqScan)->Arg(8)->Arg(16)->Arg(32)->Arg(64);

int RegisterKernels() {
  for (const KernelSet* k : AvailableKernels()) {
    for (bool l2 : {true, false}) {
      std::string name = std::string("BM_") + (l2 ? "L2/" : "Dot/") + k->name;
      benchmark::RegisterBenchmark(name.c_str(), BM_Distance, k, l2)->Arg(100)->Arg(128)->Arg(384)->Arg(960);
    }
    std::string name = std::string("BM_Sq8L2/") + k->name;
    benchmark::RegisterBenchmark(name.c_str(), BM_Sq8, k)->Arg(100)->Arg(128)->Arg(384)->Arg(960);
  }
  return 0;
}
const int kRegistered = RegisterKernels();

}  // namespace
}  // namespace vdb

BENCHMARK_MAIN();
