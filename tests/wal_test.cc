#include <gtest/gtest.h>

#include "storage/fault_injection_env.h"
#include "storage/wal.h"
#include "test_util.h"

namespace vdb {
namespace {

std::string MakeLog(size_t n, std::vector<size_t>* ends = nullptr) {
  std::string log = WalFileHeader();
  std::string payload;
  for (size_t i = 0; i < n; ++i) {
    payload.clear();
    float v[3] = {static_cast<float>(i), 1, 2};
    EncodeUpsertPayload(&payload, 1000 + i, v, 3, i % 2 == 0 ? "meta" : "");
    AppendWalRecord(&log, i % 3 == 2 ? WalRecordType::kDelete : WalRecordType::kUpsert, i + 1,
                    i % 3 == 2 ? std::string(8, 'x') : payload);
    if (ends != nullptr) ends->push_back(log.size());
  }
  return log;
}

TEST(WalTest, RecordsRoundTrip) {
  std::string log = MakeLog(50);
  std::vector<WalRecord> recs;
  WalParseResult res;
  ASSERT_TRUE(ParseWal(log, [&](const WalRecord& r) { recs.push_back(r); return Status::OK(); }, &res).ok());
  ASSERT_EQ(recs.size(), 50u);
  EXPECT_FALSE(res.torn_tail);
  EXPECT_EQ(res.valid_bytes, log.size());
  for (size_t i = 0; i < recs.size(); ++i) {
    EXPECT_EQ(recs[i].lsn, i + 1);
    if (recs[i].type == WalRecordType::kUpsert) {
      VectorId id;
      uint32_t dim;
      const char* vec;
      std::string_view meta;
      ASSERT_TRUE(DecodeUpsertPayload(recs[i].payload, &id, &dim, &vec, &meta));
      EXPECT_EQ(id, 1000 + i);
      EXPECT_EQ(dim, 3u);
      float v0;
      std::memcpy(&v0, vec, 4);
      EXPECT_EQ(v0, static_cast<float>(i));
      EXPECT_EQ(meta, i % 2 == 0 ? "meta" : "");
    }
  }
}

TEST(WalTest, TornTailIsCutAtTheLastGoodRecord) {
  std::vector<size_t> ends;
  std::string log = MakeLog(10, &ends);
  // Cut the file at every byte inside the last record.
  for (size_t cut = ends[8] + 1; cut < ends[9]; ++cut) {
    std::string torn = log.substr(0, cut);
    size_t count = 0;
    WalParseResult res;
    ASSERT_TRUE(ParseWal(torn, [&](const WalRecord&) { ++count; return Status::OK(); }, &res).ok());
    EXPECT_EQ(count, 9u);
    EXPECT_TRUE(res.torn_tail);
    EXPECT_EQ(res.valid_bytes, ends[8]);
  }
  // Garbage (zeros) after the last record is also a torn tail.
  std::string zeros = log + std::string(100, '\0');
  WalParseResult res;
  ASSERT_TRUE(ParseWal(zeros, [](const WalRecord&) { return Status::OK(); }, &res).ok());
  EXPECT_TRUE(res.torn_tail);
  EXPECT_EQ(res.valid_bytes, log.size());
}

TEST(WalTest, CorruptionFollowedByGoodRecordsIsAnError) {
  std::vector<size_t> ends;
  std::string log = MakeLog(10, &ends);
  log[ends[3] + 20] ^= 0x01;  // inside record 5
  WalParseResult res;
  Status s = ParseWal(log, [](const WalRecord&) { return Status::OK(); }, &res);
  EXPECT_TRUE(s.IsCorruption()) << s.ToString();
}

TEST(WalTest, HeaderChecks) {
  WalParseResult res;
  EXPECT_TRUE(ParseWal("", [](const WalRecord&) { return Status::OK(); }, &res).ok());
  EXPECT_FALSE(res.torn_tail);
  EXPECT_TRUE(ParseWal("VDBW", [](const WalRecord&) { return Status::OK(); }, &res).ok());
  EXPECT_TRUE(res.torn_tail);
  EXPECT_TRUE(ParseWal("XXXXXXXXXXXXXXXXXXXX", [](const WalRecord&) { return Status::OK(); }, &res).IsCorruption());
}

TEST(WalTest, WriterAppendsAndFailsForGoodAfterAFailedSync) {
  test::TempDir dir;
  FaultInjectionEnv env;
  auto w = WalWriter::Create(&env, dir.sub("000001.log"), 1);
  ASSERT_TRUE(w.ok());
  std::string recs;
  AppendWalRecord(&recs, WalRecordType::kDelete, 1, std::string(8, 'a'));
  ASSERT_TRUE((*w)->append(recs).ok());
  ASSERT_TRUE((*w)->sync().ok());
  env.set_fail_sync(true);
  ASSERT_TRUE((*w)->append(recs).ok());
  EXPECT_FALSE((*w)->sync().ok());
  env.set_fail_sync(false);
  EXPECT_TRUE((*w)->failed());
  EXPECT_FALSE((*w)->append(recs).ok());  // no retry after a failed fsync
  EXPECT_FALSE((*w)->sync().ok());
}

TEST(WalTest, FileNames) {
  EXPECT_EQ(WalFileName(14), "000014.log");
  uint64_t n = 0;
  EXPECT_TRUE(ParseWalFileName("000014.log", &n));
  EXPECT_EQ(n, 14u);
  EXPECT_FALSE(ParseWalFileName("00a014.log", &n));
  EXPECT_FALSE(ParseWalFileName("000014.tmp", &n));
}

TEST(FaultInjectionEnvTest, DropsUnsyncedDataAndUndoesUnsyncedRenames) {
  test::TempDir dir;
  FaultInjectionEnv env;
  {
    auto f = env.new_writable(dir.sub("a"));
    ASSERT_TRUE(f.ok());
    ASSERT_TRUE((*f)->append(std::string_view("synced")).ok());
    ASSERT_TRUE((*f)->sync().ok());
    ASSERT_TRUE((*f)->append(std::string_view("-lost")).ok());
  }
  ASSERT_TRUE(env.sync_dir(dir.path()).ok());
  ASSERT_TRUE(WriteFileSync(&env, dir.sub("b.tmp"), "new").ok());
  ASSERT_TRUE(env.rename(dir.sub("b.tmp"), dir.sub("b")).ok());  // no dir sync afterwards
  ASSERT_TRUE(env.drop_unsynced().ok());
  std::string data;
  ASSERT_TRUE(env.read_file(dir.sub("a"), &data).ok());
  EXPECT_EQ(data, "synced");
  EXPECT_FALSE(env.exists(dir.sub("b")));
  EXPECT_FALSE(env.exists(dir.sub("b.tmp")));  // its creation was never made durable either
}

TEST(FaultInjectionEnvTest, FailAfterCrashesAtTheNthOperation) {
  test::TempDir dir;
  FaultInjectionEnv env;
  env.fail_after(2);
  auto f = env.new_writable(dir.sub("x"));  // op 1
  ASSERT_TRUE(f.ok());
  EXPECT_TRUE((*f)->append(std::string_view("1")).ok());  // op 2
  EXPECT_FALSE((*f)->append(std::string_view("2")).ok());  // crash
  EXPECT_FALSE(env.active());
  EXPECT_FALSE((*f)->sync().ok());
  env.set_active(true);
  env.fail_after(-1);
  EXPECT_TRUE((*f)->sync().ok());
}

}  // namespace
}  // namespace vdb
