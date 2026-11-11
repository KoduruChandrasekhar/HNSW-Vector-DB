#include <gtest/gtest.h>

#include <cmath>

#include "index/hnsw/hnsw_index.h"
#include "index/quant/ivf_pq.h"
#include "index/quant/kmeans.h"
#include "index/quant/pq.h"
#include "test_util.h"
#include "util/thread_pool.h"

namespace vdb {
namespace {

using test::MakeClustered;

double MeanRecall(const Index& idx, const Dataset& qs, const std::vector<std::vector<InternalId>>& gt,
                  const SearchParams& p) {
  std::vector<Neighbor> out;
  double sum = 0;
  for (size_t q = 0; q < qs.rows; ++q) {
    idx.search(qs.row(q), 10, p, nullptr, out, nullptr);
    sum += test::Recall(out, gt[q], 10);
  }
  return sum / static_cast<double>(qs.rows);
}

TEST(KMeansTest, ConvergesWithoutNaNs) {
  Dataset ds = MakeClustered(5000, 16, 8, 1);
  ThreadPool pool(4);
  std::vector<float> cen;
  KMeansStats st;
  KMeansOptions o;
  ASSERT_TRUE(KMeans(ds.buffer.floats(), ds.rows, 16, ds.padded_dim, 32, o, &pool, &cen, &st).ok());
  ASSERT_EQ(cen.size(), 32u * 16u);
  for (float c : cen) ASSERT_TRUE(std::isfinite(c));
  EXPECT_EQ(st.iterations, 25u);
  EXPECT_GT(st.inertia, 0);
}

TEST(KMeansTest, EmptyClustersAreReseeded) {
  // 1000 copies of only 3 distinct points but 16 centroids: most clusters go empty.
  const size_t n = 1000, d = 4;
  std::vector<float> data(n * d);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < d; ++j) data[i * d + j] = static_cast<float>(i % 3) + 0.1f * static_cast<float>(j);
  std::vector<float> cen;
  KMeansStats st;
  KMeansOptions o;
  o.kmeanspp = false;
  ASSERT_TRUE(KMeans(data.data(), n, d, d, 16, o, nullptr, &cen, &st).ok());
  for (float c : cen) ASSERT_TRUE(std::isfinite(c));
  EXPECT_GT(st.empty_reseeds, 0u);
}

TEST(KMeansTest, FewerPointsThanCentroids) {
  std::vector<float> data = {0, 0, 1, 1, 2, 2};
  std::vector<float> cen;
  ASSERT_TRUE(KMeans(data.data(), 3, 2, 2, 8, KMeansOptions{}, nullptr, &cen).ok());
  EXPECT_EQ(cen.size(), 16u);
  for (float c : cen) ASSERT_TRUE(std::isfinite(c));
}

TEST(PqTest, TableDistanceEqualsDistanceToDecodedVector) {
  Dataset ds = MakeClustered(4000, 32, 10, 3);
  ProductQuantizer pq(32, 8);
  ASSERT_TRUE(pq.train(ds.buffer.floats(), ds.rows, ds.padded_dim, KMeansOptions{}, nullptr).ok());
  std::vector<uint8_t> code(8);
  std::vector<float> decoded(32), table(pq.table_size());
  for (Metric m : {Metric::L2, Metric::InnerProduct}) {
    for (size_t q = 0; q < 20; ++q) {
      const float* qv = ds.row(q + 100);
      pq.compute_table(qv, m, table.data());
      for (size_t i = 0; i < 50; ++i) {
        pq.encode(ds.row(i), code.data());
        pq.decode(code.data(), decoded.data());
        float exact = m == Metric::L2 ? L2SqAny(qv, decoded.data(), 32) : -DotAny(qv, decoded.data(), 32);
        EXPECT_NEAR(pq.adc(table.data(), code.data()), exact, 1e-3 * std::max(1.f, std::fabs(exact)));
      }
    }
  }
  EXPECT_FALSE(ProductQuantizer(30, 8).train(ds.buffer.floats(), 10, ds.padded_dim, KMeansOptions{}, nullptr).ok());
}

TEST(PqTest, PqFlatAndIvfPqRecallWithRerank) {
  Dataset base = MakeClustered(20000, 32, 40, 4);
  Dataset qs = MakeClustered(100, 32, 40, 5);
  auto gt = test::GroundTruth(base, qs, 10, Metric::L2);
  ThreadPool pool(4);
  IndexOptions o;
  o.pq_m = 8;
  o.nlist = 64;

  PqFlatIndex pqf(o, Metric::L2, base.view());
  ASSERT_TRUE(pqf.build(&pool).ok());
  SearchParams p;
  p.rerank = 0;
  double no_rerank = MeanRecall(pqf, qs, gt, p);
  p.rerank = 200;
  double with_rerank = MeanRecall(pqf, qs, gt, p);
  EXPECT_GT(with_rerank, no_rerank);
  EXPECT_GE(with_rerank, 0.9);

  IvfPqIndex ivf(o, Metric::L2, base.view());
  ASSERT_TRUE(ivf.build(&pool).ok());
  EXPECT_EQ(ivf.nlist(), 64u);
  size_t total = 0;
  for (size_t l = 0; l < ivf.nlist(); ++l) total += ivf.list_size(l);
  EXPECT_EQ(total, base.rows);
  SearchParams iv;
  iv.rerank = 200;
  iv.nprobe = 1;
  double r1 = MeanRecall(ivf, qs, gt, iv);
  iv.nprobe = 16;
  double r16 = MeanRecall(ivf, qs, gt, iv);
  EXPECT_GE(r16, r1);
  EXPECT_GE(r16, 0.9);
}

TEST(PqTest, IvfPqInnerProduct) {
  Dataset base = MakeClustered(10000, 32, 20, 6, /*normalize=*/true);
  Dataset qs = MakeClustered(100, 32, 20, 7, true);
  auto gt = test::GroundTruth(base, qs, 10, Metric::Cosine);
  IndexOptions o;
  o.pq_m = 16;
  o.nlist = 32;
  IvfPqIndex ivf(o, Metric::Cosine, base.view());
  ASSERT_TRUE(ivf.build(nullptr).ok());
  SearchParams p;
  p.nprobe = 8;
  p.rerank = 100;
  EXPECT_GE(MeanRecall(ivf, qs, gt, p), 0.9);
}

TEST(PqTest, HnswSq8RecallWithRerank) {
  Dataset base = MakeClustered(10000, 64, 30, 8);
  Dataset qs = MakeClustered(100, 64, 30, 9);
  auto gt = test::GroundTruth(base, qs, 10, Metric::L2);
  IndexOptions o;
  HnswSq8Index idx(o, Metric::L2, base.view());
  ASSERT_TRUE(idx.build(nullptr).ok());
  EXPECT_TRUE(idx.graph().check().ok());
  SearchParams p;
  p.ef = 128;
  EXPECT_GE(MeanRecall(idx, qs, gt, p), 0.95);
  EXPECT_EQ(idx.code_bytes(), base.rows * base.padded_dim);  // 4x smaller than floats
}

TEST(PqTest, SmallSegmentsShrinkNlist) {
  Dataset base = MakeClustered(500, 16, 4, 10);
  IndexOptions o;
  o.pq_m = 4;
  o.nlist = 4096;
  IvfPqIndex ivf(o, Metric::L2, base.view());
  ASSERT_TRUE(ivf.build(nullptr).ok());
  EXPECT_EQ(ivf.nlist(), 500u / 39u);
}

}  // namespace
}  // namespace vdb
