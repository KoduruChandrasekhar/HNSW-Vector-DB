// Stress: 8 reader threads, 2 writer threads, deletes, forced flushes and compactions.
// Run under ThreadSanitizer in CI. Duration: VDB_STRESS_SECONDS (default 5; the plan's
// 10-minute run is VDB_STRESS_SECONDS=600).
//
// Checks: no data races (TSan), no query ever returns an id whose delete was
// acknowledged before the query started, and the final contents match a reference
// model kept by the writers.

#include <gtest/gtest.h>

#include <atomic>
#include <map>
#include <mutex>
#include <random>
#include <thread>

#include "test_util.h"
#include "util/timer.h"
#include "vdb/collection.h"

namespace vdb {
namespace {

constexpr size_t kDim = 16;

std::vector<float> Vec(VectorId id, uint64_t version) {
  std::vector<float> v(kDim);
  for (size_t d = 0; d < kDim; ++d) v[d] = static_cast<float>((id * 13 + d * 3 + version * 7) % 97) / 9.f;
  return v;
}

TEST(ConcurrencyTest, ReadersWritersFlushesAndCompactions) {
  const char* env = std::getenv("VDB_STRESS_SECONDS");
  const double seconds = env != nullptr ? std::atof(env) : 5.0;
  test::TempDir dir;
  CollectionOptions o;
  o.dim = kDim;
  o.metric = Metric::L2;
  o.sync = SyncMode::None;
  o.index.M = 8;
  o.index.ef_construction = 40;
  o.flush_bytes = 32 * 1024;
  o.tier_base_rows = 500;
  OpenOptions oo;
  oo.num_threads = 4;
  auto c = Collection::Create(dir.sub("c"), o, oo);
  ASSERT_TRUE(c.ok());
  Collection& col = **c;

  // Ids are partitioned between the two writers, so each writer's model is exact.
  constexpr int kWriters = 2;
  constexpr VectorId kIdsPerWriter = 3000;
  std::vector<std::map<VectorId, uint64_t>> models(kWriters);  // id -> version
  // Deletes acknowledged so far, each with the time it was acknowledged.
  std::mutex del_mu;
  std::map<VectorId, uint64_t> deleted_at;  // id -> ack time (us); erased on re-insert
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> searches{0}, violations{0}, errors{0};

  auto writer = [&](int w) {
    std::mt19937_64 rng(100 + w);
    auto& model = models[w];
    uint64_t version = 0;
    while (!stop.load()) {
      VectorId id = static_cast<VectorId>(w) * kIdsPerWriter + rng() % kIdsPerWriter;
      if (rng() % 4 == 0) {
        bool found = false;
        if (!col.Remove(id, &found).ok()) {
          errors.fetch_add(1);
          continue;
        }
        model.erase(id);
        std::lock_guard<std::mutex> g(del_mu);
        deleted_at[id] = NowMicros();
      } else {
        ++version;
        {
          // An id stops counting as deleted *before* it can be visible again.
          std::lock_guard<std::mutex> g(del_mu);
          deleted_at.erase(id);
        }
        auto v = Vec(id, version);
        if (!col.Upsert(id, v.data()).ok()) {
          errors.fetch_add(1);
          continue;
        }
        model[id] = version;
      }
    }
  };

  auto reader = [&](int r) {
    std::mt19937_64 rng(200 + r);
    while (!stop.load()) {
      auto q = Vec(rng() % 5000, rng() % 10);
      uint64_t started = NowMicros();
      SearchRequest req;
      req.vector = q.data();
      req.k = 10;
      req.params.ef = 32;
      std::vector<SearchHit> hits;
      if (!col.Search(req, &hits).ok()) {
        errors.fetch_add(1);
        continue;
      }
      searches.fetch_add(1);
      std::lock_guard<std::mutex> g(del_mu);
      for (const auto& h : hits) {
        auto it = deleted_at.find(h.id);
        if (it != deleted_at.end() && it->second < started) violations.fetch_add(1);
      }
    }
  };

  auto maintenance = [&] {
    std::mt19937_64 rng(300);
    while (!stop.load()) {
      if (rng() % 2 == 0) {
        if (!col.Flush().ok()) errors.fetch_add(1);
      } else {
        if (!col.Compact(rng() % 3).ok()) errors.fetch_add(1);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  };

  std::vector<std::thread> threads;
  for (int w = 0; w < kWriters; ++w) threads.emplace_back(writer, w);
  for (int r = 0; r < 8; ++r) threads.emplace_back(reader, r);
  threads.emplace_back(maintenance);
  std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
  stop.store(true);
  for (auto& t : threads) t.join();

  EXPECT_EQ(errors.load(), 0u);
  EXPECT_EQ(violations.load(), 0u) << "a deleted id was returned";
  EXPECT_GT(searches.load(), 100u);

  // Final contents match the model, before and after a reopen.
  auto check = [&](Collection& cc) {
    size_t expected = 0;
    for (int w = 0; w < kWriters; ++w) {
      expected += models[w].size();
      for (const auto& [id, version] : models[w]) {
        std::vector<float> v;
        ASSERT_TRUE(cc.Get(id, &v, nullptr).ok()) << id;
        ASSERT_EQ(v, Vec(id, version)) << id;
      }
    }
    EXPECT_EQ(cc.Stats().live_rows, expected);
  };
  check(col);
  ASSERT_TRUE(col.Flush().ok());
  ASSERT_TRUE(col.Compact(1).ok());
  check(col);
  c->reset();
  auto reopened = Collection::Open(dir.sub("c"), oo);
  ASSERT_TRUE(reopened.ok()) << reopened.status().ToString();
  check(**reopened);
}

}  // namespace
}  // namespace vdb
