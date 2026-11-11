#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <random>

#include "util/thread_pool.h"
#include "util/topk.h"

namespace vdb {
namespace {

TEST(TopKTest, KeepsExactlyTheKSmallestIncludingTies) {
  std::mt19937_64 rng(1);
  for (int trial = 0; trial < 200; ++trial) {
    size_t n = 1 + rng() % 500;
    size_t k = 1 + rng() % 50;
    // Few distinct values force many ties; ties are broken by id.
    std::uniform_int_distribution<int> val(0, trial % 2 == 0 ? 5 : 1000);
    std::vector<Neighbor> all;
    TopK top(k);
    for (size_t i = 0; i < n; ++i) {
      Neighbor nb{static_cast<float>(val(rng)), static_cast<InternalId>(i)};
      all.push_back(nb);
      top.push(nb.dist, nb.id);
    }
    size_t m = std::min(k, n);
    std::partial_sort(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(m), all.end());
    all.resize(m);
    EXPECT_EQ(top.sorted(), all) << "trial " << trial;
  }
}

TEST(TopKTest, ThresholdAndEmpty) {
  TopK top(2);
  EXPECT_TRUE(std::isinf(top.threshold()));
  top.push(3, 1);
  top.push(1, 2);
  EXPECT_FLOAT_EQ(top.threshold(), 3);
  top.push(2, 3);
  EXPECT_FLOAT_EQ(top.threshold(), 2);
  TopK zero(0);
  zero.push(1, 1);
  EXPECT_EQ(zero.size(), 0u);
}

TEST(TopKTest, MergeTopK) {
  std::vector<std::vector<Neighbor>> lists = {{{1, 1}, {4, 4}}, {{2, 2}, {3, 3}}, {}};
  auto m = MergeTopK(lists, 3);
  ASSERT_EQ(m.size(), 3u);
  EXPECT_EQ(m[0].id, 1u);
  EXPECT_EQ(m[1].id, 2u);
  EXPECT_EQ(m[2].id, 3u);
}

TEST(ThreadPoolTest, ParallelForCoversEveryIndexOnce) {
  ThreadPool pool(4);
  std::vector<std::atomic<int>> hits(10007);
  pool.parallel_for(0, hits.size(), [&](size_t i) { hits[i].fetch_add(1); }, 13);
  for (auto& h : hits) EXPECT_EQ(h.load(), 1);
}

TEST(ThreadPoolTest, NestedParallelForDoesNotDeadlock) {
  ThreadPool pool(3);
  std::atomic<int> total{0};
  pool.parallel_for(0, 16, [&](size_t) {
    pool.parallel_for(0, 100, [&](size_t) { total.fetch_add(1); });
  });
  EXPECT_EQ(total.load(), 1600);
}

TEST(ThreadPoolTest, SingleThreadPoolRunsInline) {
  ThreadPool pool(1);
  EXPECT_EQ(pool.parallelism(), 1u);
  int x = 0;
  pool.parallel_for(0, 10, [&](size_t i) { x += static_cast<int>(i); });
  EXPECT_EQ(x, 45);
}

}  // namespace
}  // namespace vdb
