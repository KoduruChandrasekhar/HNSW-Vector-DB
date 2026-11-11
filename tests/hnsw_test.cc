#include <gtest/gtest.h>

#include <cmath>

#include "index/hnsw/hnsw_index.h"
#include "storage/env.h"
#include "test_util.h"
#include "util/thread_pool.h"

namespace vdb {
namespace {

using test::MakeClustered;

double MeanRecall(const Index& idx, const Dataset& qs, const std::vector<std::vector<InternalId>>& gt,
                  size_t k, size_t ef, const RowFilter* filter = nullptr) {
  SearchParams p;
  p.ef = ef;
  std::vector<Neighbor> out;
  double sum = 0;
  for (size_t q = 0; q < qs.rows; ++q) {
    idx.search(qs.row(q), k, p, filter, out, nullptr);
    sum += test::Recall(out, gt[q], k);
  }
  return sum / static_cast<double>(qs.rows);
}

class StringFile final : public WritableFile {
 public:
  using WritableFile::append;
  Status append(std::span<const std::byte> d) override {
    data.append(reinterpret_cast<const char*>(d.data()), d.size());
    return Status::OK();
  }
  Status sync() override { return Status::OK(); }
  Status close() override { return Status::OK(); }
  uint64_t size() const override { return data.size(); }
  std::string data;
};

struct Fixture {
  Fixture(size_t n, size_t dim, Metric m) : metric(m) {
    bool norm = m == Metric::Cosine;
    base = MakeClustered(n, dim, 50, 101, norm);
    queries = MakeClustered(200, dim, 50, 202, norm);
    gt = test::GroundTruth(base, queries, 10, m);
  }
  Metric metric;
  Dataset base, queries;
  std::vector<std::vector<InternalId>> gt;
};

TEST(HnswTest, RecallAndGraphInvariantsSingleThread) {
  Fixture f(10000, 32, Metric::L2);
  IndexOptions o;
  o.M = 16;
  o.ef_construction = 200;
  HnswIndex idx(o, f.metric, f.base.view());
  ASSERT_TRUE(idx.build(nullptr).ok());
  ASSERT_TRUE(idx.graph().check().ok()) << idx.graph().check().ToString();
  EXPECT_EQ(idx.size(), f.base.rows);
  double r = MeanRecall(idx, f.queries, f.gt, 10, 100);
  EXPECT_GE(r, 0.95);
  // Larger ef never hurts much and should reach near-perfect recall.
  EXPECT_GE(MeanRecall(idx, f.queries, f.gt, 10, 400), 0.99);
  // About 1 in M nodes reaches layer 1.
  auto hist = idx.graph().level_histogram();
  ASSERT_GE(hist.size(), 2u);
  double frac = 1.0 - static_cast<double>(hist[0]) / static_cast<double>(f.base.rows);
  EXPECT_NEAR(frac, 1.0 / 16, 0.02);
}

TEST(HnswTest, ParallelBuildKeepsRecall) {
  Fixture f(20000, 48, Metric::L2);
  IndexOptions o;
  ThreadPool pool(4);
  HnswIndex idx(o, f.metric, f.base.view());
  ASSERT_TRUE(idx.build(&pool).ok());
  ASSERT_TRUE(idx.graph().check().ok()) << idx.graph().check().ToString();
  EXPECT_GE(MeanRecall(idx, f.queries, f.gt, 10, 128), 0.95);
}

TEST(HnswTest, CosineAndOddDimensions) {
  Fixture f(8000, 100, Metric::Cosine);  // 100 dims: padded to 112
  IndexOptions o;
  HnswIndex idx(o, f.metric, f.base.view());
  ASSERT_TRUE(idx.build(nullptr).ok());
  EXPECT_GE(MeanRecall(idx, f.queries, f.gt, 10, 128), 0.95);
}

TEST(HnswTest, SaveLoadGivesIdenticalResults) {
  Fixture f(5000, 24, Metric::InnerProduct);
  IndexOptions o;
  o.M = 12;
  HnswIndex idx(o, f.metric, f.base.view());
  ASSERT_TRUE(idx.build(nullptr).ok());
  StringFile file;
  ASSERT_TRUE(idx.save(file).ok());
  auto loaded = LoadIndex(file.data, o, f.metric, f.base.view());
  ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
  SearchParams p;
  p.ef = 64;
  std::vector<Neighbor> a, b;
  for (size_t q = 0; q < f.queries.rows; ++q) {
    idx.search(f.queries.row(q), 10, p, nullptr, a, nullptr);
    (*loaded)->search(f.queries.row(q), 10, p, nullptr, b, nullptr);
    ASSERT_EQ(a, b);
  }
  // A flipped byte is caught by the CRC footer.
  file.data[file.data.size() / 2] ^= 0x5a;
  EXPECT_FALSE(LoadIndex(file.data, o, f.metric, f.base.view()).ok());
}

class EvenRows final : public RowFilter {
 public:
  bool allowed(InternalId id) const override { return id % 2 == 0; }
};

TEST(HnswTest, FilteredSearchNeverReturnsDisallowedRows) {
  Fixture f(6000, 16, Metric::L2);
  IndexOptions o;
  HnswIndex idx(o, f.metric, f.base.view());
  ASSERT_TRUE(idx.build(nullptr).ok());
  EvenRows even;
  // Ground truth over even rows only.
  DistFn fn = MetricDistance(Metric::L2, true);
  SearchParams p;
  p.ef = 128;
  std::vector<Neighbor> out;
  double sum = 0;
  for (size_t q = 0; q < f.queries.rows; ++q) {
    TopK top(10);
    for (size_t i = 0; i < f.base.rows; i += 2) top.push(fn(f.queries.row(q), f.base.row(i), f.base.padded_dim), static_cast<InternalId>(i));
    std::vector<InternalId> truth;
    for (auto& n : top.sorted()) truth.push_back(n.id);
    idx.search(f.queries.row(q), 10, p, &even, out, nullptr);
    ASSERT_EQ(out.size(), 10u);
    for (const auto& n : out) ASSERT_EQ(n.id % 2, 0u);
    sum += test::Recall(out, truth, 10);
  }
  EXPECT_GE(sum / f.queries.rows, 0.95);
}

TEST(HnswTest, ExtendKeepsOldGraphAndAddsRows) {
  Fixture f(12000, 32, Metric::L2);
  IndexOptions o;
  VectorView first = f.base.view();
  first.rows = 8000;
  HnswIndex idx(o, f.metric, first);
  ASSERT_TRUE(idx.build(nullptr).ok());
  ThreadPool pool(4);
  auto ext = idx.extend(f.base.view(), &pool);
  ASSERT_TRUE(ext.ok()) << ext.status().ToString();
  EXPECT_EQ((*ext)->size(), f.base.rows);
  auto* h = dynamic_cast<HnswIndex*>(ext->get());
  ASSERT_NE(h, nullptr);
  EXPECT_TRUE(h->graph().check().ok()) << h->graph().check().ToString();
  EXPECT_GE(MeanRecall(**ext, f.queries, f.gt, 10, 128), 0.95);
}

TEST(HnswTest, TinyGraphs) {
  for (size_t n : {1u, 2u, 3u, 17u}) {
    Dataset base = MakeClustered(n, 8, 1, 9);
    IndexOptions o;
    HnswIndex idx(o, Metric::L2, base.view());
    ASSERT_TRUE(idx.build(nullptr).ok());
    EXPECT_TRUE(idx.graph().check().ok());
    std::vector<Neighbor> out;
    SearchParams p;
    idx.search(base.row(0), 5, p, nullptr, out, nullptr);
    EXPECT_EQ(out.size(), std::min<size_t>(n, 5));
    EXPECT_EQ(out[0].id, 0u);
  }
}

TEST(HnswTest, VisitedTagWrapClearsTheArray) {
  SearchContext ctx(4);
  ctx.reset();
  ctx.mark(2);
  EXPECT_TRUE(ctx.seen(2));
  for (int i = 0; i < 65535; ++i) ctx.reset();  // wraps through 0
  EXPECT_FALSE(ctx.seen(2));
  ctx.mark(1);
  EXPECT_TRUE(ctx.seen(1));
  EXPECT_FALSE(ctx.seen(3));
}

}  // namespace
}  // namespace vdb
