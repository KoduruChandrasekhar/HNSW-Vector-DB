#include <gtest/gtest.h>

#include <atomic>
#include <set>
#include <thread>

#include "engine/compaction.h"
#include "test_util.h"
#include "vdb/collection.h"

namespace vdb {
namespace {

constexpr size_t kDim = 24;

CollectionOptions Opts() {
  CollectionOptions o;
  o.dim = kDim;
  o.metric = Metric::L2;
  o.sync = SyncMode::None;
  o.index.M = 12;
  o.index.ef_construction = 100;
  return o;
}

OpenOptions Manual() {
  OpenOptions oo;
  oo.num_threads = 4;
  oo.auto_flush = false;
  oo.auto_compact = false;
  return oo;
}

size_t DiskSegments(const Collection& c) {
  size_t n = 0;
  for (const auto& s : c.Stats().segments) n += s.kind == "disk";
  return n;
}

std::vector<VectorId> Ids(size_t from, size_t to) {
  std::vector<VectorId> v;
  for (size_t i = from; i < to; ++i) v.push_back(i);
  return v;
}

// Real-dims copy of rows [from, to) of a padded dataset.
std::vector<float> Rows(const Dataset& ds, size_t from, size_t to) {
  std::vector<float> out;
  for (size_t i = from; i < to; ++i) out.insert(out.end(), ds.row(i), ds.row(i) + ds.dim);
  return out;
}

double RecallVsLive(const Collection& c, const Dataset& base, const Dataset& qs,
                    const std::set<VectorId>& deleted, size_t ef, bool* saw_deleted) {
  DistFn fn = MetricDistance(Metric::L2, true);
  double sum = 0;
  for (size_t q = 0; q < qs.rows; ++q) {
    TopK top(10);
    for (size_t i = 0; i < base.rows; ++i) {
      if (!deleted.count(i)) top.push(fn(qs.row(q), base.row(i), base.padded_dim), static_cast<InternalId>(i));
    }
    std::set<VectorId> truth;
    for (auto& n : top.sorted()) truth.insert(n.id);
    SearchRequest req;
    req.vector = qs.row(q);
    req.k = 10;
    req.params.ef = ef;
    std::vector<SearchHit> hits;
    EXPECT_TRUE(c.Search(req, &hits).ok());
    for (const auto& h : hits) {
      sum += truth.count(h.id);
      if (deleted.count(h.id)) *saw_deleted = true;
    }
  }
  return sum / (10.0 * qs.rows);
}

TEST(CompactionPolicyTest, Rules) {
  CollectionOptions o = Opts();
  o.tier_base_rows = 1000;
  o.tier_fanout = 4;
  EXPECT_EQ(TierOf(999, 1000, 4), 0u);
  EXPECT_EQ(TierOf(1000, 1000, 4), 1u);
  EXPECT_EQ(TierOf(3999, 1000, 4), 1u);
  EXPECT_EQ(TierOf(4000, 1000, 4), 2u);
  // Nothing to do for one healthy segment.
  EXPECT_FALSE(PickCompaction({{1, 10000, 10000}}, o, 0, true).has_value());
  // Delete-driven.
  auto t = PickCompaction({{1, 10000, 7000}, {2, 100000, 100000}}, o, 0, true);
  ASSERT_TRUE(t);
  EXPECT_EQ(t->reason, "deletes");
  EXPECT_EQ(t->inputs, std::vector<uint32_t>{1});
  // A full tier of four.
  t = PickCompaction({{1, 900000, 900000}, {2, 1500, 1500}, {3, 1600, 1600}, {4, 1700, 1700}, {5, 1800, 1800}}, o, 0, true);
  ASSERT_TRUE(t);
  EXPECT_EQ(t->reason, "tier_1");
  EXPECT_EQ(t->inputs.size(), 4u);
  // Merge into the largest once the rest pass 20% of it.
  t = PickCompaction({{1, 10000, 10000}, {2, 1500, 1500}, {3, 1600, 1600}}, o, 0, true);
  ASSERT_TRUE(t);
  EXPECT_EQ(t->reason, "merge_into_largest");
  EXPECT_TRUE(t->merge_into_largest);
  EXPECT_EQ(t->inputs.front(), 1u);
  EXPECT_FALSE(PickCompaction({{1, 10000, 10000}, {2, 1500, 1500}}, o, 0, true).has_value());
  // Explicit target and empty segments.
  t = PickCompaction({{1, 10000, 10000}, {2, 1500, 1500}}, o, 1, true);
  ASSERT_TRUE(t);
  EXPECT_EQ(t->reason, "explicit");
  t = PickCompaction({{1, 10, 0}, {2, 1500, 1500}}, o, 0, true);
  ASSERT_TRUE(t);
  EXPECT_EQ(t->reason, "drop_empty");
}

TEST(CompactionTest, MutableSegmentFlushesAfterTheAgeLimit) {
  test::TempDir dir;
  CollectionOptions o = Opts();
  o.flush_age_ms = 200;  // far below flush_bytes: only the age can trigger this flush
  OpenOptions oo;
  oo.num_threads = 2;
  oo.auto_compact = false;
  auto c = Collection::Create(dir.sub("c"), o, oo);
  ASSERT_TRUE(c.ok());
  Dataset base = test::MakeClustered(10, kDim, 2, 9);
  auto ids = Ids(0, 10);
  auto rows = Rows(base, 0, 10);
  ASSERT_TRUE((*c)->UpsertBatch(ids.data(), rows.data(), ids.size()).ok());
  for (int i = 0; i < 100 && DiskSegments(**c) == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(DiskSegments(**c), 1u);
  EXPECT_EQ((*c)->Stats().live_rows, 10u);
}

TEST(CompactionTest, BulkLoadMakesExactlyOneSegment) {
  test::TempDir dir;
  Dataset base = test::MakeClustered(10000, kDim, 20, 1);
  auto c = Collection::Create(dir.sub("c"), Opts(), Manual());
  ASSERT_TRUE(c.ok());
  auto ids = Ids(0, base.rows);
  auto rows = Rows(base, 0, base.rows);
  ASSERT_TRUE((*c)->BulkLoad(ids.data(), rows.data(), ids.size()).ok());
  EXPECT_EQ(DiskSegments(**c), 1u);
  EXPECT_EQ((*c)->Stats().live_rows, base.rows);
  // Only new ids: a second load with an existing id fails and changes nothing.
  auto more = Ids(9990, 10010);
  auto mrows = Rows(base, 0, 20);
  EXPECT_EQ((*c)->BulkLoad(more.data(), mrows.data(), more.size()).code(), Status::Code::kAlreadyExists);
  std::vector<VectorId> dup = {20000, 20000};
  EXPECT_FALSE((*c)->BulkLoad(dup.data(), mrows.data(), 2).ok());
  EXPECT_EQ(DiskSegments(**c), 1u);
  EXPECT_EQ((*c)->Stats().live_rows, base.rows);
}

TEST(CompactionTest, DeletesThenCompactionRestoreRecallAndRemoveRows) {
  test::TempDir dir;
  Dataset base = test::MakeClustered(12000, kDim, 30, 2);
  Dataset qs = test::MakeClustered(100, kDim, 30, 3);
  auto c = Collection::Create(dir.sub("c"), Opts(), Manual());
  ASSERT_TRUE(c.ok());
  auto ids = Ids(0, base.rows);
  auto rows = Rows(base, 0, base.rows);
  ASSERT_TRUE((*c)->BulkLoad(ids.data(), rows.data(), ids.size()).ok());
  bool saw = false;
  double r0 = RecallVsLive(**c, base, qs, {}, 32, &saw);
  std::set<VectorId> deleted;
  std::mt19937_64 rng(4);
  std::vector<VectorId> del;
  for (VectorId id = 0; id < base.rows; ++id) {
    if (rng() % 2 == 0) {
      del.push_back(id);
      deleted.insert(id);
    }
  }
  size_t removed = 0;
  ASSERT_TRUE((*c)->RemoveBatch(del.data(), del.size(), &removed).ok());
  EXPECT_EQ(removed, del.size());
  double r_del = RecallVsLive(**c, base, qs, deleted, 32, &saw);
  EXPECT_FALSE(saw) << "a deleted id was returned";
  ASSERT_TRUE((*c)->Compact().ok());
  CollectionStats st = (*c)->Stats();
  EXPECT_EQ(st.deleted_rows, 0u);
  EXPECT_EQ(st.live_rows, base.rows - del.size());
  double r_after = RecallVsLive(**c, base, qs, deleted, 32, &saw);
  EXPECT_FALSE(saw);
  EXPECT_GE(r_after, r0 - 0.03) << "before=" << r0 << " with deletes=" << r_del;
  for (VectorId id : del) ASSERT_FALSE((*c)->Get(id, nullptr, nullptr).ok());
}

TEST(CompactionTest, UpsertsSettleToAtMostTwoSegments) {
  test::TempDir dir;
  CollectionOptions o = Opts();
  o.tier_base_rows = 1000;
  OpenOptions oo;
  oo.num_threads = 4;
  oo.auto_flush = false;  // flush every 500 rows by hand so the segment count is deterministic
  oo.idle_compact_ms = 300;
  auto c = Collection::Create(dir.sub("c"), o, oo);
  ASSERT_TRUE(c.ok());
  Dataset base = test::MakeClustered(15000, kDim, 20, 5);
  for (size_t i = 0; i < base.rows; i += 500) {
    auto ids = Ids(i, i + 500);
    auto rows = Rows(base, i, i + 500);
    ASSERT_TRUE((*c)->UpsertBatch(ids.data(), rows.data(), ids.size()).ok());
    ASSERT_TRUE((*c)->Flush().ok());  // background compaction runs meanwhile
  }
  ASSERT_TRUE((*c)->Compact().ok());
  // The size tiers bound the count while writes continue...
  EXPECT_LE(DiskSegments(**c), 1u + (o.tier_fanout - 1) * 2);
  EXPECT_GT((*c)->Stats().compactions, 0u);
  // ...and once the collection goes idle everything is merged into the largest segment.
  for (int i = 0; i < 100 && DiskSegments(**c) > 2; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_LE(DiskSegments(**c), 2u);
  EXPECT_EQ((*c)->Stats().live_rows, base.rows);
  ASSERT_TRUE((*c)->Compact(1).ok());
  EXPECT_EQ(DiskSegments(**c), 1u);
  EXPECT_EQ((*c)->Stats().live_rows, base.rows);
}

TEST(CompactionTest, MergeIntoLargestKeepsEverything) {
  test::TempDir dir;
  Dataset base = test::MakeClustered(9000, kDim, 20, 6);
  Dataset qs = test::MakeClustered(50, kDim, 20, 7);
  auto c = Collection::Create(dir.sub("c"), Opts(), Manual());
  ASSERT_TRUE(c.ok());
  auto ids = Ids(0, 6000);
  auto rows = Rows(base, 0, 6000);
  ASSERT_TRUE((*c)->BulkLoad(ids.data(), rows.data(), ids.size()).ok());
  for (size_t s = 6000; s < 9000; s += 1000) {
    auto i2 = Ids(s, s + 1000);
    auto r2 = Rows(base, s, s + 1000);
    ASSERT_TRUE((*c)->UpsertBatch(i2.data(), r2.data(), i2.size()).ok());
    ASSERT_TRUE((*c)->Flush().ok());
  }
  std::vector<VectorId> del = {1, 2, 3, 6001, 8999};
  ASSERT_TRUE((*c)->RemoveBatch(del.data(), del.size()).ok());
  EXPECT_EQ(DiskSegments(**c), 4u);
  ASSERT_TRUE((*c)->Compact(1).ok());
  EXPECT_EQ(DiskSegments(**c), 1u);
  std::set<VectorId> deleted(del.begin(), del.end());
  bool saw = false;
  EXPECT_GE(RecallVsLive(**c, base, qs, deleted, 64, &saw), 0.95);
  EXPECT_FALSE(saw);
  EXPECT_EQ((*c)->Stats().live_rows, 9000u - del.size());
}

TEST(CompactionTest, DeletesDuringCompactionAreNotResurrected) {
  test::TempDir dir;
  Dataset base = test::MakeClustered(8000, kDim, 20, 8);
  std::set<VectorId> deleted;
  {
    auto c = Collection::Create(dir.sub("c"), Opts(), Manual());
    ASSERT_TRUE(c.ok());
    for (size_t s = 0; s < base.rows; s += 1000) {
      auto i2 = Ids(s, s + 1000);
      auto r2 = Rows(base, s, s + 1000);
      ASSERT_TRUE((*c)->UpsertBatch(i2.data(), r2.data(), i2.size()).ok());
      ASSERT_TRUE((*c)->Flush().ok());
    }
    std::atomic<bool> done{false};
    std::vector<VectorId> del;
    std::thread deleter([&] {
      VectorId id = 3;
      while (!done.load() && id < base.rows) {
        if ((*c)->Remove(id).ok()) del.push_back(id);
        id += 7;
      }
    });
    ASSERT_TRUE((*c)->Compact(1).ok());
    done.store(true);
    deleter.join();
    deleted.insert(del.begin(), del.end());
    ASSERT_FALSE(deleted.empty());
    for (VectorId id : deleted) ASSERT_FALSE((*c)->Get(id, nullptr, nullptr).ok()) << id;
    EXPECT_EQ((*c)->Stats().live_rows, base.rows - deleted.size());
  }
  auto c = Collection::Open(dir.sub("c"), Manual());
  ASSERT_TRUE(c.ok()) << c.status().ToString();
  for (VectorId id : deleted) ASSERT_FALSE((*c)->Get(id, nullptr, nullptr).ok()) << id;
  EXPECT_EQ((*c)->Stats().live_rows, base.rows - deleted.size());
}

}  // namespace
}  // namespace vdb
