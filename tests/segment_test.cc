#include <gtest/gtest.h>

#include "meta/schema.h"
#include "storage/manifest.h"
#include "storage/segment.h"
#include "test_util.h"
#include "util/thread_pool.h"

namespace vdb {
namespace {

CollectionOptions Options() {
  CollectionOptions o;
  o.dim = 20;
  o.metric = Metric::L2;
  o.schema = {{"year", FieldType::Int64}, {"tag", FieldType::Keyword}};
  return o;
}

StatusOr<std::shared_ptr<Segment>> Write(const std::string& dir, const Dataset& ds,
                                         const CollectionOptions& o, SegmentMeta* meta) {
  SegmentSource src;
  src.rows = ds.rows;
  src.row = [&](size_t i) { return ds.row(i); };
  src.id = [](size_t i) { return VectorId{5000 + i}; };
  src.meta = [&](size_t i, std::string* out) {
    MetaRow r = {int64_t(2000 + i % 30), std::string(i % 2 == 0 ? "even" : "odd")};
    EncodeMetaRow(r, out);
  };
  SegmentWriteOptions wo;
  wo.id = 7;
  wo.lsn = 99;
  wo.options = &o;
  return WriteSegment(Env::Default(), dir, src, wo, nullptr, meta);
}

TEST(SegmentTest, WriteThenOpenRoundTrips) {
  test::TempDir dir;
  CollectionOptions o = Options();
  Dataset ds = test::MakeClustered(3000, 20, 10, 1);
  SegmentMeta meta;
  auto seg = Write(dir.path(), ds, o, &meta);
  ASSERT_TRUE(seg.ok()) << seg.status().ToString();
  EXPECT_EQ(meta.rows, 3000u);
  EXPECT_TRUE(Env::Default()->is_dir(dir.sub("seg-000007")));
  EXPECT_FALSE(Env::Default()->exists(dir.sub("seg-000007.tmp")));

  auto reopened = Segment::Open(Env::Default(), dir.sub("seg-000007"), meta, o, /*verify_crc=*/true);
  ASSERT_TRUE(reopened.ok()) << reopened.status().ToString();
  const Segment& s = **reopened;
  EXPECT_EQ(s.rows(), 3000u);
  EXPECT_EQ(s.lsn(), 99u);
  EXPECT_EQ(s.external_id(17), 5017u);
  for (size_t d = 0; d < 20; ++d) EXPECT_EQ(s.row(42)[d], ds.row(42)[d]);
  MetaRow r = s.columns().GetRow(5);
  EXPECT_EQ(std::get<int64_t>(r[0]), 2005);
  EXPECT_EQ(std::get<std::string>(r[1]), "odd");
  std::vector<Neighbor> out;
  SearchParams p;
  s.index().search(ds.row(123), 1, p, nullptr, out, nullptr);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].id, 123u);
  // The page-aligned data starts at byte 4096 of vectors.f32.
  auto sz = Env::Default()->file_size(dir.sub("seg-000007/vectors.f32"));
  EXPECT_EQ(*sz, 4096 + 3000 * 32 * sizeof(float));
}

TEST(SegmentTest, CorruptionIsDetected) {
  test::TempDir dir;
  CollectionOptions o = Options();
  Dataset ds = test::MakeClustered(500, 20, 3, 2);
  SegmentMeta meta;
  ASSERT_TRUE(Write(dir.path(), ds, o, &meta).ok());
  const std::string vpath = dir.sub("seg-000007/vectors.f32");
  std::string data;
  ASSERT_TRUE(Env::Default()->read_file(vpath, &data).ok());
  std::string flipped = data;
  flipped[4096 + 1000] ^= 0x10;
  ASSERT_TRUE(WriteFileSync(Env::Default(), vpath, flipped).ok());
  EXPECT_TRUE(Segment::Open(Env::Default(), dir.sub("seg-000007"), meta, o, true).status().IsCorruption());
  // Without the CRC check the data corruption goes unnoticed (that is the tradeoff).
  EXPECT_TRUE(Segment::Open(Env::Default(), dir.sub("seg-000007"), meta, o, false).ok());
  // A damaged header is always caught.
  std::string bad_header = data;
  bad_header[20] ^= 0x01;
  ASSERT_TRUE(WriteFileSync(Env::Default(), vpath, bad_header).ok());
  EXPECT_FALSE(Segment::Open(Env::Default(), dir.sub("seg-000007"), meta, o, false).ok());
  // A manifest that disagrees with the files is rejected.
  ASSERT_TRUE(WriteFileSync(Env::Default(), vpath, data).ok());
  SegmentMeta wrong = meta;
  wrong.rows = 499;
  EXPECT_FALSE(Segment::Open(Env::Default(), dir.sub("seg-000007"), wrong, o, false).ok());
}

TEST(SegmentTest, DeletesFileRoundTrip) {
  test::TempDir dir;
  CollectionOptions o = Options();
  Dataset ds = test::MakeClustered(300, 20, 3, 3);
  SegmentMeta meta;
  ASSERT_TRUE(Write(dir.path(), ds, o, &meta).ok());
  roaring::Roaring del;
  del.add(3);
  del.add(299);
  std::string name;
  FileInfo fi;
  ASSERT_TRUE(WriteDeletesFile(Env::Default(), dir.sub("seg-000007"), 12, del, &name, &fi).ok());
  EXPECT_EQ(name, "deletes-000012.roar");
  meta.deletes_file = name;
  auto s = Segment::Open(Env::Default(), dir.sub("seg-000007"), meta, o, true);
  ASSERT_TRUE(s.ok());
  EXPECT_TRUE((*s)->deletes().test(3));
  EXPECT_TRUE((*s)->deletes().test(299));
  EXPECT_FALSE((*s)->deletes().test(4));
  EXPECT_EQ((*s)->live_rows(), 298u);
}

TEST(ManifestTest, EncodeDecodeAndCommit) {
  test::TempDir dir;
  Manifest m;
  m.number = 9;
  m.options = Options();
  m.next_file_number = 15;
  m.flushed_lsn = 1234;
  m.wal_log_number = 14;
  SegmentMeta s;
  s.id = 6;
  s.rows = 100;
  s.deleted = 3;
  s.lsn = 1000;
  s.deletes_file = "deletes-000011.roar";
  s.files["vectors.f32"] = {4096 + 100 * 128, 0xdeadbeef};
  m.segments.push_back(s);
  ASSERT_TRUE(CommitManifest(Env::Default(), dir.path(), m).ok());
  Manifest back;
  ASSERT_TRUE(ReadCurrentManifest(Env::Default(), dir.path(), &back).ok());
  EXPECT_EQ(back.number, 9u);
  EXPECT_EQ(back.flushed_lsn, 1234u);
  EXPECT_EQ(back.wal_log_number, 14u);
  ASSERT_EQ(back.segments.size(), 1u);
  EXPECT_EQ(back.segments[0].deletes_file, "deletes-000011.roar");
  EXPECT_EQ(back.segments[0].files["vectors.f32"].crc, 0xdeadbeefu);
  EXPECT_EQ(back.options.schema, m.options.schema);
  std::string raw = EncodeManifest(m);
  raw[raw.size() - 5] ^= 0x22;
  EXPECT_TRUE(DecodeManifest(raw, &back).IsCorruption());
}

}  // namespace
}  // namespace vdb
