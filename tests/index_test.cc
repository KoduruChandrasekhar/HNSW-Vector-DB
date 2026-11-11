// Every index type through the common Index interface: build, search, save, load, and
// identical results after reload.

#include <gtest/gtest.h>

#include "index/index.h"
#include "storage/env.h"
#include "test_util.h"

namespace vdb {
namespace {

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

class IndexTypeTest : public ::testing::TestWithParam<IndexType> {};

TEST_P(IndexTypeTest, BuildSearchSaveLoad) {
  Dataset base = test::MakeClustered(4000, 32, 20, 1);
  Dataset qs = test::MakeClustered(50, 32, 20, 2);
  auto gt = test::GroundTruth(base, qs, 10, Metric::L2);
  IndexOptions o;
  o.type = GetParam();
  o.pq_m = 8;
  o.nlist = 32;
  auto idx = CreateIndex(o, Metric::L2, base.view());
  ASSERT_TRUE(idx.ok()) << idx.status().ToString();
  ASSERT_TRUE((*idx)->build(nullptr).ok());
  EXPECT_EQ((*idx)->type(), GetParam());
  EXPECT_EQ((*idx)->size(), base.rows);
  SearchParams p;
  p.ef = 128;
  p.nprobe = 8;
  p.rerank = 100;
  std::vector<Neighbor> out;
  double sum = 0;
  for (size_t q = 0; q < qs.rows; ++q) {
    (*idx)->search(qs.row(q), 10, p, nullptr, out, nullptr);
    sum += test::Recall(out, gt[q], 10);
  }
  EXPECT_GE(sum / qs.rows, 0.85) << IndexTypeName(GetParam());

  StringFile f;
  ASSERT_TRUE((*idx)->save(f).ok());
  auto loaded = LoadIndex(f.data, o, Metric::L2, base.view());
  ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
  EXPECT_EQ((*loaded)->type(), GetParam());
  std::vector<Neighbor> a, b;
  for (size_t q = 0; q < qs.rows; ++q) {
    (*idx)->search(qs.row(q), 10, p, nullptr, a, nullptr);
    (*loaded)->search(qs.row(q), 10, p, nullptr, b, nullptr);
    ASSERT_EQ(a, b) << IndexTypeName(GetParam());
  }
  EXPECT_FALSE(LoadIndex(f.data.substr(0, f.data.size() - 1), o, Metric::L2, base.view()).ok());
}

INSTANTIATE_TEST_SUITE_P(AllTypes, IndexTypeTest,
                         ::testing::Values(IndexType::Flat, IndexType::Hnsw, IndexType::HnswSq8,
                                           IndexType::PqFlat, IndexType::IvfPq),
                         [](const ::testing::TestParamInfo<IndexType>& i) {
                           return std::string(IndexTypeName(i.param));
                         });

TEST(IndexTest, PqMustDivideDim) {
  Dataset base = test::MakeClustered(100, 30, 2, 1);
  IndexOptions o;
  o.type = IndexType::PqFlat;
  o.pq_m = 8;
  EXPECT_FALSE(CreateIndex(o, Metric::L2, base.view()).ok());
}

}  // namespace
}  // namespace vdb
