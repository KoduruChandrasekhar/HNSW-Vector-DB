// Recall regression: a fixed-seed workload whose recall@10 must not drop more than 0.005
// below the stored value. The synthetic case always runs; the SIFT case runs when a
// 100k subset has been prepared in $VDB_DATA_DIR/sift-100k (bench/prepare_data.sh).

#include <gtest/gtest.h>

#include "index/hnsw/hnsw_index.h"
#include "index/flat_index.h"
#include "test_util.h"
#include "util/dataset.h"
#include "util/thread_pool.h"

namespace vdb {
namespace {

// Stored values (update deliberately, in the same commit as the change that moves them).
constexpr double kSyntheticRecallAtEf32 = 0.993;  // hnswlib on this distribution: 0.993
constexpr double kSiftRecallAtEf64 = 0.985;
constexpr double kTolerance = 0.005;

double HnswRecall(const Dataset& base, const Dataset& qs, const std::vector<std::vector<InternalId>>& gt,
                  size_t ef) {
  IndexOptions o;
  o.M = 16;
  o.ef_construction = 200;
  HnswIndex idx(o, Metric::L2, base.view());
  EXPECT_TRUE(idx.build(nullptr).ok());  // single thread: deterministic
  SearchParams p;
  p.ef = ef;
  std::vector<Neighbor> out;
  double sum = 0;
  for (size_t q = 0; q < qs.rows; ++q) {
    idx.search(qs.row(q), 10, p, nullptr, out, nullptr);
    sum += test::Recall(out, gt[q], 10);
  }
  return sum / static_cast<double>(qs.rows);
}

TEST(RecallRegressionTest, Synthetic) {
  Dataset base = test::MakeClustered(30000, 64, 100, 2024, false, 0.5f);
  Dataset qs = test::MakeClustered(300, 64, 100, 2025, false, 0.5f);
  auto gt = test::GroundTruth(base, qs, 10, Metric::L2);
  double r = HnswRecall(base, qs, gt, 32);
  RecordProperty("recall_at_10", std::to_string(r));
  std::printf("synthetic recall@10 (ef=32): %.4f\n", r);
  EXPECT_GE(r, kSyntheticRecallAtEf32 - kTolerance);
}

TEST(RecallRegressionTest, Sift100k) {
  const char* root = std::getenv("VDB_DATA_DIR");
  std::string dir = std::string(root != nullptr ? root : "data") + "/sift-100k";
  Dataset base, qs;
  if (!ReadFbin(dir + "/base.fbin", 0, false, &base).ok()) GTEST_SKIP() << "no dataset at " << dir;
  ASSERT_TRUE(ReadFbin(dir + "/query.fbin", 1000, false, &qs).ok());
  ThreadPool pool(0);
  std::vector<std::vector<Neighbor>> exact;
  FlatSearchBatch(base.view(), MetricDistance(Metric::L2), qs.buffer.floats(), qs.rows, 10, exact, &pool);
  std::vector<std::vector<InternalId>> gt(qs.rows);
  for (size_t q = 0; q < qs.rows; ++q)
    for (const auto& n : exact[q]) gt[q].push_back(n.id);
  double r = HnswRecall(base, qs, gt, 64);
  std::printf("sift-100k recall@10 (ef=64): %.4f\n", r);
  EXPECT_GE(r, kSiftRecallAtEf64 - kTolerance);
}

}  // namespace
}  // namespace vdb
