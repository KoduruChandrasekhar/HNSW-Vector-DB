#include <gtest/gtest.h>

#include "index/flat_index.h"
#include "test_util.h"
#include "util/dataset.h"
#include "util/thread_pool.h"

namespace vdb {
namespace {

using test::MakeClustered;

class OddRows final : public RowFilter {
 public:
  bool allowed(InternalId id) const override { return id % 2 == 1; }
};

TEST(FlatTest, MatchesBruteForceForEveryMetric) {
  for (Metric m : {Metric::L2, Metric::InnerProduct, Metric::Cosine}) {
    Dataset base = MakeClustered(3000, 37, 10, 1, m == Metric::Cosine);
    Dataset qs = MakeClustered(50, 37, 10, 2, m == Metric::Cosine);
    auto gt = test::GroundTruth(base, qs, 20, m);
    DistFn fn = MetricDistance(m);
    std::vector<Neighbor> out;
    for (size_t q = 0; q < qs.rows; ++q) {
      FlatSearch(base.view(), fn, qs.row(q), 20, nullptr, out);
      ASSERT_EQ(out.size(), 20u);
      EXPECT_GE(test::Recall(out, gt[q], 20), 0.95) << MetricName(m);  // SIMD vs scalar ties
      for (size_t i = 1; i < out.size(); ++i) EXPECT_LE(out[i - 1].dist, out[i].dist);
    }
  }
}

TEST(FlatTest, ParallelAndBatchedMatchSingleThreaded) {
  Dataset base = MakeClustered(20000, 64, 20, 3);
  Dataset qs = MakeClustered(130, 64, 20, 4);
  ThreadPool pool(4);
  DistFn fn = MetricDistance(Metric::L2);
  std::vector<std::vector<Neighbor>> batch;
  FlatSearchBatch(base.view(), fn, qs.buffer.floats(), qs.rows, 10, batch, &pool);
  ASSERT_EQ(batch.size(), qs.rows);
  std::vector<Neighbor> single, par;
  for (size_t q = 0; q < qs.rows; ++q) {
    FlatSearch(base.view(), fn, qs.row(q), 10, nullptr, single);
    FlatSearch(base.view(), fn, qs.row(q), 10, nullptr, par, &pool);
    EXPECT_EQ(single, par);
    EXPECT_EQ(single, batch[q]);
  }
}

TEST(FlatTest, FilterIsRespected) {
  Dataset base = MakeClustered(1000, 16, 4, 5);
  OddRows odd;
  std::vector<Neighbor> out;
  FlatSearch(base.view(), MetricDistance(Metric::L2), base.row(0), 50, &odd, out);
  ASSERT_EQ(out.size(), 50u);
  for (const auto& n : out) EXPECT_EQ(n.id % 2, 1u);
}

TEST(FlatTest, KLargerThanRows) {
  Dataset base = MakeClustered(5, 8, 1, 6);
  std::vector<Neighbor> out;
  FlatSearch(base.view(), MetricDistance(Metric::L2), base.row(0), 100, nullptr, out);
  EXPECT_EQ(out.size(), 5u);
  EXPECT_EQ(out[0].id, 0u);
}

TEST(DatasetTest, FbinRoundTripWithPaddingAndNormalization) {
  test::TempDir dir;
  std::vector<float> rows = {3, 4, 0, 0, 0, 0, 1, 0, 0};  // three rows of dim 3
  ASSERT_TRUE(WriteFbin(dir.sub("x.fbin"), rows.data(), 3, 3, 3).ok());
  Dataset ds;
  ASSERT_TRUE(ReadFbin(dir.sub("x.fbin"), 0, /*normalize=*/true, &ds).ok());
  EXPECT_EQ(ds.rows, 3u);
  EXPECT_EQ(ds.dim, 3u);
  EXPECT_EQ(ds.padded_dim, 16u);
  EXPECT_NEAR(ds.row(0)[0], 0.6f, 1e-6);
  EXPECT_NEAR(ds.row(0)[1], 0.8f, 1e-6);
  EXPECT_EQ(ds.zero_rows, 1u);  // the zero row cannot be normalized
  EXPECT_EQ(ds.row(0)[5], 0.f);
  std::vector<int32_t> gt = {1, 2, 3, 4};
  ASSERT_TRUE(WriteIbin(dir.sub("g.ibin"), gt.data(), 2, 2).ok());
  std::vector<int32_t> back;
  size_t r = 0, c = 0;
  ASSERT_TRUE(ReadIbin(dir.sub("g.ibin"), 0, &back, &r, &c).ok());
  EXPECT_EQ(back, gt);
  EXPECT_EQ(r, 2u);
  EXPECT_EQ(c, 2u);
}

// Exact search reproduces the ground truth shipped with an ANN-Benchmarks dataset when
// one has been converted into data/<name>/ (tools/convert_hdf5.py). Skipped otherwise.
TEST(FlatTest, ReproducesShippedGroundTruthWhenAvailable) {
  const char* root = std::getenv("VDB_DATA_DIR");
  std::string dir = std::string(root != nullptr ? root : "data") + "/fashion-mnist";
  Dataset base, qs;
  if (!ReadFbin(dir + "/base.fbin", 0, false, &base).ok()) GTEST_SKIP() << "no dataset at " << dir;
  ASSERT_TRUE(ReadFbin(dir + "/query.fbin", 200, false, &qs).ok());
  std::vector<int32_t> gt;
  size_t gr = 0, gc = 0;
  ASSERT_TRUE(ReadIbin(dir + "/gt.ibin", 200, &gt, &gr, &gc).ok());
  ThreadPool pool(0);
  std::vector<std::vector<Neighbor>> res;
  FlatSearchBatch(base.view(), MetricDistance(Metric::L2), qs.buffer.floats(), qs.rows, 100, res, &pool);
  double hit = 0;
  for (size_t q = 0; q < qs.rows; ++q) {
    std::set<int32_t> t(gt.begin() + static_cast<std::ptrdiff_t>(q * gc), gt.begin() + static_cast<std::ptrdiff_t>(q * gc + 100));
    for (const auto& n : res[q]) hit += t.count(static_cast<int32_t>(n.id));
  }
  EXPECT_GE(hit / (100.0 * qs.rows), 0.999);
}

}  // namespace
}  // namespace vdb
