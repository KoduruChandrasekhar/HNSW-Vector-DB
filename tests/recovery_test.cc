// Crash recovery: WAL replay, a crash injected at every I/O step of flush, compaction
// and bulk load, simulated power loss, torn log tails and failed fsyncs.

#include <gtest/gtest.h>

#include <map>
#include <random>

#include "storage/fault_injection_env.h"
#include "test_util.h"
#include "vdb/collection.h"

namespace vdb {
namespace {

constexpr size_t kDim = 12;

CollectionOptions Opts(SyncMode sync = SyncMode::Always) {
  CollectionOptions o;
  o.dim = kDim;
  o.metric = Metric::L2;
  o.sync = sync;
  o.index.M = 8;
  o.index.ef_construction = 40;
  o.schema = {{"n", FieldType::Int64}, {"tag", FieldType::Keyword}};
  return o;
}

OpenOptions Quiet() {
  OpenOptions oo;
  oo.num_threads = 2;
  oo.auto_flush = false;
  oo.auto_compact = false;
  return oo;
}

std::vector<float> Vec(VectorId id) {
  std::vector<float> v(kDim);
  for (size_t d = 0; d < kDim; ++d) v[d] = static_cast<float>((id * 31 + d * 7) % 101) / 10.f;
  return v;
}

struct Model {
  std::map<VectorId, std::vector<float>> rows;
  void upsert(VectorId id, std::vector<float> v) { rows[id] = std::move(v); }
  void remove(VectorId id) { rows.erase(id); }
};

// Every acknowledged write is present with exactly the right vector; deleted ids are gone.
void ExpectMatches(Collection& c, const Model& m, const std::set<VectorId>& maybe = {}) {
  CollectionStats st = c.Stats();
  size_t extra = 0;
  for (VectorId id : maybe) {
    std::vector<float> v;
    if (c.Get(id, &v, nullptr).ok()) {
      ++extra;
      EXPECT_EQ(v, Vec(id)) << "half-applied write for id " << id;
    }
  }
  EXPECT_EQ(st.live_rows, m.rows.size() + extra);
  for (const auto& [id, vec] : m.rows) {
    std::vector<float> got;
    Status s = c.Get(id, &got, nullptr);
    ASSERT_TRUE(s.ok()) << "lost acknowledged write id=" << id << ": " << s.ToString();
    ASSERT_EQ(got, vec) << "id " << id;
  }
}

Status UpsertRange(Collection& c, Model& m, VectorId from, VectorId to) {
  for (VectorId id = from; id < to; ++id) {
    auto v = Vec(id);
    Metadata md{{"n", int64_t(id)}, {"tag", std::string(id % 2 ? "odd" : "even")}};
    VDB_RETURN_IF_ERROR(c.Upsert(id, v.data(), &md));
    m.upsert(id, v);
  }
  return Status::OK();
}

TEST(RecoveryTest, ReopenReplaysTheLog) {
  test::TempDir dir;
  Model m;
  {
    auto c = Collection::Create(dir.sub("c"), Opts(), Quiet());
    ASSERT_TRUE(c.ok()) << c.status().ToString();
    ASSERT_TRUE(UpsertRange(**c, m, 0, 500).ok());
    for (VectorId id = 0; id < 500; id += 7) {
      ASSERT_TRUE((*c)->Remove(id).ok());
      m.remove(id);
    }
    auto v = Vec(3000);
    ASSERT_TRUE((*c)->Upsert(3, v.data()).ok());  // overwrite
    m.upsert(3, v);
  }
  auto c = Collection::Open(dir.sub("c"), Quiet());
  ASSERT_TRUE(c.ok()) << c.status().ToString();
  ExpectMatches(**c, m);
  EXPECT_GT((*c)->Stats().last_recovery_records, 500u);
  Metadata md;
  ASSERT_TRUE((*c)->Get(10, nullptr, &md).ok());
  EXPECT_EQ(std::get<int64_t>(md["n"]), 10);
  EXPECT_EQ(std::get<std::string>(md["tag"]), "even");
}

TEST(RecoveryTest, FlushThenReopenIsIdentical) {
  test::TempDir dir;
  Model m;
  std::vector<std::vector<SearchHit>> before;
  {
    auto c = Collection::Create(dir.sub("c"), Opts(), Quiet());
    ASSERT_TRUE(c.ok());
    ASSERT_TRUE(UpsertRange(**c, m, 0, 2000).ok());
    ASSERT_TRUE((*c)->Flush().ok());
    for (VectorId id = 0; id < 2000; id += 5) {
      ASSERT_TRUE((*c)->Remove(id).ok());
      m.remove(id);
    }
    ASSERT_TRUE(UpsertRange(**c, m, 2000, 2600).ok());
    ASSERT_TRUE((*c)->Flush().ok());
    ASSERT_TRUE(UpsertRange(**c, m, 2600, 2700).ok());  // left in the log
    EXPECT_EQ((*c)->Stats().segments.size(), 3u);         // 2 on disk + mutable
    for (VectorId q = 0; q < 20; ++q) {
      auto v = Vec(q * 13);
      SearchRequest req;
      req.vector = v.data();
      req.k = 10;
      req.params.ef = 200;
      std::vector<SearchHit> hits;
      ASSERT_TRUE((*c)->Search(req, &hits).ok());
      before.push_back(hits);
    }
  }
  auto c = Collection::Open(dir.sub("c"), Quiet());
  ASSERT_TRUE(c.ok()) << c.status().ToString();
  ExpectMatches(**c, m);
  for (VectorId q = 0; q < 20; ++q) {
    auto v = Vec(q * 13);
    SearchRequest req;
    req.vector = v.data();
    req.k = 10;
    req.params.ef = 200;
    std::vector<SearchHit> hits;
    ASSERT_TRUE((*c)->Search(req, &hits).ok());
    ASSERT_EQ(hits.size(), before[q].size());
    for (size_t i = 0; i < hits.size(); ++i) EXPECT_EQ(hits[i].id, before[q][i].id);
  }
}

// Runs `op` with a crash injected after k mutating I/O operations, for every k until the
// operation completes, then simulates power loss and checks recovery.
template <class Setup, class Op>
void CrashAtEveryStep(const char* what, Setup setup, Op op, bool tear) {
  int completed_at = -1;
  for (int k = 0; k < 2000 && completed_at < 0; ++k) {
    test::TempDir dir;
    FaultInjectionEnv env;
    Model m;
    std::set<VectorId> maybe;
    {
      auto c = Collection::Create(dir.sub("c"), Opts(), Quiet(), &env);
      ASSERT_TRUE(c.ok()) << c.status().ToString();
      setup(**c, m);
      env.fail_after(k);
      Status s = op(**c, m, maybe);
      if (env.active()) completed_at = k;  // the crash point was never reached
      (void)s;
    }  // closing after the "crash" fails quietly
    env.set_active(true);
    env.fail_after(-1);
    ASSERT_TRUE(env.drop_unsynced(tear).ok());
    auto c = Collection::Open(dir.sub("c"), Quiet(), &env);
    ASSERT_TRUE(c.ok()) << what << " crash at op " << k << ": " << c.status().ToString();
    ExpectMatches(**c, m, maybe);
    // The recovered collection keeps working.
    auto v = Vec(99999);
    ASSERT_TRUE((*c)->Upsert(99999, v.data()).ok());
    ASSERT_TRUE((*c)->Flush().ok());
  }
  EXPECT_GT(completed_at, 5) << what;
}

TEST(RecoveryTest, CrashAtEveryStepOfFlush) {
  CrashAtEveryStep(
      "flush",
      [](Collection& c, Model& m) {
        ASSERT_TRUE(UpsertRange(c, m, 0, 300).ok());
        ASSERT_TRUE(c.Flush().ok());
        ASSERT_TRUE(UpsertRange(c, m, 300, 450).ok());
        for (VectorId id = 0; id < 450; id += 9) {
          ASSERT_TRUE(c.Remove(id).ok());  // deletes on disk and mutable rows
          m.remove(id);
        }
      },
      [](Collection& c, Model&, std::set<VectorId>&) { return c.Flush(); }, false);
}

TEST(RecoveryTest, CrashAtEveryStepOfCompaction) {
  CrashAtEveryStep(
      "compaction",
      [](Collection& c, Model& m) {
        for (int s = 0; s < 3; ++s) {
          ASSERT_TRUE(UpsertRange(c, m, s * 100, s * 100 + 100).ok());
          ASSERT_TRUE(c.Flush().ok());
        }
        for (VectorId id = 0; id < 300; id += 4) {
          ASSERT_TRUE(c.Remove(id).ok());
          m.remove(id);
        }
      },
      [](Collection& c, Model&, std::set<VectorId>&) { return c.Compact(1); }, false);
}

TEST(RecoveryTest, CrashAtEveryStepOfBulkLoadIsAllOrNothing) {
  CrashAtEveryStep(
      "bulk load",
      [](Collection& c, Model& m) { ASSERT_TRUE(UpsertRange(c, m, 0, 50).ok()); },
      [](Collection& c, Model& m, std::set<VectorId>& maybe) {
        std::vector<VectorId> ids;
        std::vector<float> rows;
        for (VectorId id = 1000; id < 1200; ++id) {
          ids.push_back(id);
          auto v = Vec(id);
          rows.insert(rows.end(), v.begin(), v.end());
        }
        Status s = c.BulkLoad(ids.data(), rows.data(), ids.size());
        if (s.ok()) {
          for (VectorId id : ids) m.upsert(id, Vec(id));
        } else {
          maybe.insert(ids.begin(), ids.end());
        }
        return s;
      },
      false);
}

TEST(RecoveryTest, CrashWhileWritingWithTornTail) {
  CrashAtEveryStep(
      "upserts",
      [](Collection& c, Model& m) { ASSERT_TRUE(UpsertRange(c, m, 0, 20).ok()); },
      [](Collection& c, Model& m, std::set<VectorId>& maybe) {
        for (VectorId id = 100; id < 140; ++id) {
          auto v = Vec(id);
          Status s = c.Upsert(id, v.data());
          if (!s.ok()) {
            maybe.insert(id);  // not acknowledged: may or may not survive, never half
            for (VectorId rest = id + 1; rest < 140; ++rest) maybe.insert(rest);
            return s;
          }
          m.upsert(id, v);
        }
        return Status::OK();
      },
      /*tear=*/true);
}

TEST(RecoveryTest, PowerLossLosesNothingAcknowledgedWithSyncAlways) {
  test::TempDir dir;
  FaultInjectionEnv env;
  Model m;
  {
    auto c = Collection::Create(dir.sub("c"), Opts(SyncMode::Always), Quiet(), &env);
    ASSERT_TRUE(c.ok());
    ASSERT_TRUE(UpsertRange(**c, m, 0, 300).ok());
    ASSERT_TRUE((*c)->Flush().ok());
    ASSERT_TRUE(UpsertRange(**c, m, 300, 400).ok());
    env.set_active(false);  // the power goes out
  }
  env.set_active(true);
  ASSERT_TRUE(env.drop_unsynced(/*tear=*/true).ok());
  auto c = Collection::Open(dir.sub("c"), Quiet(), &env);
  ASSERT_TRUE(c.ok()) << c.status().ToString();
  ExpectMatches(**c, m);
}

TEST(RecoveryTest, SyncNoneLosesOnlyTheUnsyncedTailAndRecoversCleanly) {
  test::TempDir dir;
  FaultInjectionEnv env;
  Model durable;
  {
    auto c = Collection::Create(dir.sub("c"), Opts(SyncMode::None), Quiet(), &env);
    ASSERT_TRUE(c.ok());
    ASSERT_TRUE(UpsertRange(**c, durable, 0, 200).ok());
    ASSERT_TRUE((*c)->Flush().ok());  // durable through the segment
    Model scratch;
    ASSERT_TRUE(UpsertRange(**c, scratch, 200, 300).ok());  // only in the page cache
    env.set_active(false);
  }
  env.set_active(true);
  ASSERT_TRUE(env.drop_unsynced(/*tear=*/true).ok());
  std::set<VectorId> maybe;
  for (VectorId id = 200; id < 300; ++id) maybe.insert(id);
  {
    auto c = Collection::Open(dir.sub("c"), Quiet(), &env);
    ASSERT_TRUE(c.ok()) << c.status().ToString();
    ExpectMatches(**c, durable, maybe);
    ASSERT_TRUE(UpsertRange(**c, durable, 500, 520).ok());  // appends after the cut tail
  }
  auto c = Collection::Open(dir.sub("c"), Quiet(), &env);
  ASSERT_TRUE(c.ok()) << c.status().ToString();
  for (VectorId id = 500; id < 520; ++id) EXPECT_TRUE((*c)->Get(id, nullptr, nullptr).ok());
}

TEST(RecoveryTest, FailedFsyncStopsAcceptingWrites) {
  test::TempDir dir;
  FaultInjectionEnv env;
  auto c = Collection::Create(dir.sub("c"), Opts(SyncMode::Always), Quiet(), &env);
  ASSERT_TRUE(c.ok());
  auto v = Vec(1);
  ASSERT_TRUE((*c)->Upsert(1, v.data()).ok());
  env.set_fail_sync(true);
  EXPECT_FALSE((*c)->Upsert(2, v.data()).ok());
  env.set_fail_sync(false);
  Status s = (*c)->Upsert(3, v.data());
  EXPECT_FALSE(s.ok()) << "writes must stay refused after a failed fsync";
  EXPECT_TRUE((*c)->Get(1, nullptr, nullptr).ok());  // reads still work
}

TEST(RecoveryTest, CreateRejectsExistingAndOpenRejectsMissing) {
  test::TempDir dir;
  ASSERT_TRUE(Collection::Create(dir.sub("c"), Opts(), Quiet()).ok());
  EXPECT_EQ(Collection::Create(dir.sub("c"), Opts(), Quiet()).status().code(), Status::Code::kAlreadyExists);
  EXPECT_FALSE(Collection::Open(dir.sub("missing"), Quiet()).ok());
}

TEST(RecoveryTest, CorruptManifestIsReported) {
  test::TempDir dir;
  { ASSERT_TRUE(Collection::Create(dir.sub("c"), Opts(), Quiet()).ok()); }
  std::string cur;
  ASSERT_TRUE(Env::Default()->read_file(dir.sub("c/CURRENT"), &cur).ok());
  cur.pop_back();
  std::string data;
  ASSERT_TRUE(Env::Default()->read_file(dir.sub("c/" + cur), &data).ok());
  data[data.size() / 2] ^= 0x7;
  ASSERT_TRUE(WriteFileSync(Env::Default(), dir.sub("c/" + cur), data).ok());
  EXPECT_TRUE(Collection::Open(dir.sub("c"), Quiet()).status().IsCorruption());
}

}  // namespace
}  // namespace vdb
