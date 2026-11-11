// Model-based testing: random sequences of upsert, delete, search, flush, compact and
// reopen against a std::map plus brute-force search. A flat index keeps search exact, so
// results must match the model exactly (up to distance ties).

#include <gtest/gtest.h>

#include <map>
#include <random>

#include "test_util.h"
#include "vdb/collection.h"

namespace vdb {
namespace {

constexpr size_t kDim = 8;

void RunSeed(uint64_t seed, IndexType type) {
  test::TempDir dir;
  CollectionOptions o;
  o.dim = kDim;
  o.metric = Metric::L2;
  o.sync = SyncMode::None;
  o.index.type = type;
  o.flush_bytes = 4096 * 4;  // tiny segments: many flushes and compactions
  o.tier_base_rows = 64;
  o.schema = {{"g", FieldType::Int64}};
  OpenOptions oo;
  oo.num_threads = 2;
  oo.auto_flush = false;
  oo.auto_compact = false;
  auto c = Collection::Create(dir.sub("c"), o, oo);
  ASSERT_TRUE(c.ok());
  std::unique_ptr<Collection> col = std::move(c).value();
  std::map<VectorId, std::pair<std::vector<float>, int64_t>> model;
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  DistFn fn = MetricDistance(Metric::L2, true);

  for (int step = 0; step < 1500; ++step) {
    int op = static_cast<int>(rng() % 100);
    if (op < 55) {  // upsert (new or existing id)
      VectorId id = rng() % 400;
      std::vector<float> v(kDim);
      for (auto& x : v) x = u(rng);
      int64_t g = static_cast<int64_t>(rng() % 4);
      Metadata md{{"g", g}};
      ASSERT_TRUE(col->Upsert(id, v.data(), &md).ok());
      model[id] = {v, g};
    } else if (op < 75) {  // delete
      VectorId id = rng() % 400;
      bool found = false;
      ASSERT_TRUE(col->Remove(id, &found).ok());
      ASSERT_EQ(found, model.erase(id) > 0) << "step " << step;
    } else if (op < 93) {  // search, sometimes filtered
      std::vector<float> q(kDim);
      for (auto& x : q) x = u(rng);
      bool filtered = rng() % 2 == 0;
      int64_t g = static_cast<int64_t>(rng() % 4);
      std::unique_ptr<Filter> f;
      if (filtered) f = std::move(col->ParseFilter(R"({"field": "g", "eq": )" + std::to_string(g) + "}")).value();
      SearchRequest req;
      req.vector = q.data();
      req.k = 10;
      req.filter = f.get();
      std::vector<SearchHit> hits;
      ASSERT_TRUE(col->Search(req, &hits).ok());
      std::vector<std::pair<float, VectorId>> want;
      std::vector<float> qp(PaddedDim(kDim), 0.f), vp(PaddedDim(kDim), 0.f);
      std::copy(q.begin(), q.end(), qp.begin());
      for (const auto& [id, row] : model) {
        if (filtered && row.second != g) continue;
        std::copy(row.first.begin(), row.first.end(), vp.begin());
        want.push_back({fn(qp.data(), vp.data(), kDim), id});
      }
      std::sort(want.begin(), want.end());
      ASSERT_EQ(hits.size(), std::min<size_t>(10, want.size())) << "step " << step;
      for (size_t i = 0; i < hits.size(); ++i) {
        EXPECT_NEAR(hits[i].distance, want[i].first, 1e-4) << "step " << step << " rank " << i;
        auto it = model.find(hits[i].id);
        ASSERT_NE(it, model.end()) << "deleted id " << hits[i].id << " returned at step " << step;
        if (filtered) {
          EXPECT_EQ(it->second.second, g);
        }
      }
    } else if (op < 97) {
      ASSERT_TRUE(col->Flush().ok());
    } else if (op < 99) {
      ASSERT_TRUE(col->Compact(rng() % 3).ok());
    } else {  // reopen
      col.reset();
      auto r = Collection::Open(dir.sub("c"), oo);
      ASSERT_TRUE(r.ok()) << r.status().ToString();
      col = std::move(r).value();
    }
    if (step % 100 == 99) {
      ASSERT_EQ(col->Stats().live_rows, model.size()) << "step " << step;
    }
  }
  for (const auto& [id, row] : model) {
    std::vector<float> v;
    Metadata md;
    ASSERT_TRUE(col->Get(id, &v, &md).ok()) << id;
    EXPECT_EQ(v, row.first);
    EXPECT_EQ(std::get<int64_t>(md["g"]), row.second);
  }
}

TEST(ModelTest, FlatIndexMatchesTheModel) {
  for (uint64_t seed = 1; seed <= 6; ++seed) {
    SCOPED_TRACE("seed " + std::to_string(seed));
    RunSeed(seed, IndexType::Flat);
    if (HasFatalFailure()) return;
  }
}

}  // namespace
}  // namespace vdb
