#include <gtest/gtest.h>

#include <random>

#include "meta/columns.h"
#include "meta/filter.h"
#include "test_util.h"
#include "vdb/collection.h"

namespace vdb {
namespace {

Schema TestSchema() {
  return {{"year", FieldType::Int64},  {"score", FieldType::Float64}, {"lang", FieldType::Keyword},
          {"tags", FieldType::KeywordList}, {"ok", FieldType::Bool}, {"attr", FieldType::Int64}};
}

TEST(FilterParserTest, AcceptsTheDocumentedLanguage) {
  Schema s = TestSchema();
  const char* good[] = {
      R"({"field": "year", "gte": 2020})",
      R"({"and": [{"field": "year", "gte": 2020}, {"field": "tags", "in": ["ml", "db"]}, {"not": {"field": "lang", "eq": "fr"}}]})",
      R"({"or": [{"field": "score", "lt": 0.5}, {"field": "ok", "eq": true}]})",
      R"({"field": "lang", "exists": false})",
      R"({"field": "score", "lte": 3})",
      R"({"field": "year", "in": [1999, 2001]})",
      R"({"field": "tags", "ne": "x"})",
  };
  for (const char* f : good) EXPECT_TRUE(FilterExpr::Parse(f, s).ok()) << f;
}

TEST(FilterParserTest, RejectsBadFilters) {
  Schema s = TestSchema();
  const char* bad[] = {
      "",
      "[1, 2]",
      "{not json",
      R"({"field": "nope", "eq": 1})",
      R"({"field": "year", "eq": "x"})",
      R"({"field": "lang", "gt": "a"})",
      R"({"field": "year"})",
      R"({"field": "year", "eq": 1, "ne": 2})",
      R"({"and": []})",
      R"({"and": [{"field": "year", "eq": 1}], "x": 1})",
      R"({"field": "year", "between": [1, 2]})",
      R"({"field": "ok", "in": [true]})",
      R"({"field": "lang", "exists": 1})",
      R"({"field": "year", "in": []})",
      R"({"not": [1]})",
  };
  for (const char* f : bad) EXPECT_FALSE(FilterExpr::Parse(f, s).ok()) << f;
  std::string deep;
  for (int i = 0; i < 40; ++i) deep += R"({"not": )";
  deep += R"({"field": "year", "eq": 1})";
  for (int i = 0; i < 40; ++i) deep += "}";
  EXPECT_FALSE(FilterExpr::Parse(deep, s).ok());
}

TEST(FilterTest, RowSemantics) {
  Schema s = TestSchema();
  MetaRow r = {int64_t{2021}, 0.25, std::string("en"), std::vector<std::string>{"ml", "ai"}, true, std::monostate{}};
  auto m = [&](const char* f) {
    auto e = FilterExpr::Parse(f, s);
    EXPECT_TRUE(e.ok()) << f;
    return (*e)->Matches(r);
  };
  EXPECT_TRUE(m(R"({"field": "year", "gte": 2020})"));
  EXPECT_FALSE(m(R"({"field": "year", "lt": 2021})"));
  EXPECT_TRUE(m(R"({"field": "tags", "eq": "ml"})"));
  EXPECT_TRUE(m(R"({"field": "tags", "in": ["db", "ai"]})"));
  EXPECT_FALSE(m(R"({"field": "tags", "ne": "ai"})"));
  EXPECT_TRUE(m(R"({"field": "lang", "ne": "fr"})"));
  EXPECT_TRUE(m(R"({"not": {"field": "lang", "eq": "fr"}})"));
  EXPECT_FALSE(m(R"({"field": "attr", "ne": 3})"));  // null: comparisons are false
  EXPECT_TRUE(m(R"({"not": {"field": "attr", "eq": 3}})"));
  EXPECT_TRUE(m(R"({"field": "attr", "exists": false})"));
  EXPECT_TRUE(m(R"({"field": "score", "lt": 0.3})"));
  EXPECT_TRUE(m(R"({"field": "ok", "eq": true})"));
  EXPECT_TRUE(m(R"({"or": [{"field": "ok", "eq": false}, {"field": "year", "eq": 2021}]})"));
}

// The immutable segment's bitmap indexes must agree with row-by-row evaluation.
TEST(FilterTest, ColumnIndexesAgreeWithRowEvaluation) {
  Schema s = TestSchema();
  std::mt19937_64 rng(42);
  const size_t n = 20000;
  std::vector<std::string> enc(n);
  std::vector<MetaRow> rows(n);
  const char* langs[] = {"en", "fr", "de", "es"};
  const char* tags[] = {"ml", "db", "ai", "os", "pl"};
  for (size_t i = 0; i < n; ++i) {
    MetaRow r(s.size(), std::monostate{});
    if (rng() % 10) r[0] = int64_t(1990 + rng() % 35);
    if (rng() % 10) r[1] = static_cast<double>(rng() % 1000) / 100.0;
    if (rng() % 10) r[2] = std::string(langs[rng() % 4]);
    if (rng() % 10) {
      std::vector<std::string> t;
      for (int k = 0; k < static_cast<int>(rng() % 3); ++k) t.push_back(tags[rng() % 5]);
      r[3] = t;
    }
    if (rng() % 10) r[4] = rng() % 2 == 0;
    r[5] = int64_t(rng() % 1000);
    rows[i] = r;
    EncodeMetaRow(r, &enc[i]);
  }
  SegmentColumns cols;
  ASSERT_TRUE(cols.Build(s, n, [&](size_t i) { return std::string_view(enc[i]); }).ok());
  for (size_t i = 0; i < n; i += 997) {
    MetaRow back = cols.GetRow(i);
    EXPECT_EQ(back, rows[i]) << i;
  }
  const char* filters[] = {
      R"({"field": "year", "gte": 2010})",
      R"({"field": "year", "eq": 2000})",
      R"({"field": "year", "ne": 2000})",
      R"({"field": "year", "in": [1991, 1995, 2020]})",
      R"({"field": "score", "lt": 2.5})",
      R"({"field": "score", "gt": 9.0})",
      R"({"field": "score", "lte": 0})",
      R"({"field": "lang", "eq": "fr"})",
      R"({"field": "lang", "in": ["de", "es", "zz"]})",
      R"({"field": "lang", "ne": "en"})",
      R"({"field": "tags", "eq": "ml"})",
      R"({"field": "tags", "in": ["db", "os"]})",
      R"({"field": "tags", "ne": "ml"})",
      R"({"field": "ok", "eq": false})",
      R"({"field": "ok", "exists": true})",
      R"({"field": "attr", "lt": 10})",
      R"({"and": [{"field": "year", "gte": 2000}, {"field": "tags", "in": ["ml", "db"]}, {"not": {"field": "lang", "eq": "fr"}}]})",
      R"({"or": [{"field": "attr", "lt": 5}, {"field": "lang", "eq": "es"}]})",
      R"({"not": {"or": [{"field": "attr", "gte": 500}, {"field": "ok", "eq": true}]}})",
  };
  for (const char* f : filters) {
    auto e = FilterExpr::Parse(f, s);
    ASSERT_TRUE(e.ok()) << f;
    roaring::Roaring bm = cols.Evaluate((*e)->root());
    size_t expect = 0;
    for (size_t i = 0; i < n; ++i) {
      bool want = (*e)->Matches(rows[i]);
      expect += want;
      ASSERT_EQ(bm.contains(static_cast<uint32_t>(i)), want) << f << " row " << i;
    }
    EXPECT_EQ(bm.cardinality(), expect) << f;
  }
}

// ---- filtered search against filtered brute force, every strategy and selectivity

class FilteredSearchTest : public ::testing::Test {
 protected:
  static constexpr size_t kN = 20000;
  static constexpr size_t kDim = 32;

  void SetUp() override {
    CollectionOptions o;
    o.dim = kDim;
    o.metric = Metric::L2;
    o.sync = SyncMode::None;
    o.schema = {{"attr", FieldType::Int64}, {"tag", FieldType::Keyword}};
    OpenOptions oo;
    oo.auto_flush = false;
    oo.auto_compact = false;
    oo.num_threads = 4;
    auto c = Collection::Create(dir_.sub("c"), o, oo);
    ASSERT_TRUE(c.ok());
    col_ = std::move(c).value();
    base_ = test::MakeClustered(kN, kDim, 40, 1);
    queries_ = test::MakeClustered(40, kDim, 40, 2);
    std::mt19937_64 rng(3);
    attr_.resize(kN);
    std::vector<VectorId> ids;
    std::vector<float> rows;
    std::vector<Metadata> md;
    // Most rows bulk-loaded into one HNSW segment, the rest stay in the mutable segment.
    for (size_t i = 0; i < kN; ++i) {
      attr_[i] = static_cast<int>(rng() % 1000);
      Metadata m{{"attr", int64_t{attr_[i]}}, {"tag", std::string(i % 3 == 0 ? "a" : "b")}};
      if (i < kN - 1000) {
        ids.push_back(i);
        rows.insert(rows.end(), base_.row(i), base_.row(i) + kDim);
        md.push_back(m);
      } else {
        ASSERT_TRUE(col_->Upsert(i, base_.row(i), &m).ok());
      }
    }
    ASSERT_TRUE(col_->BulkLoad(ids.data(), rows.data(), ids.size(), &md).ok());
  }

  // Recall of a strategy against exact search over the matching rows.
  double Recall(int threshold, FilterStrategy strat, size_t* min_results, std::string* last_strategy) {
    std::string f = R"({"field": "attr", "lt": )" + std::to_string(threshold) + "}";
    auto filter = col_->ParseFilter(f);
    EXPECT_TRUE(filter.ok());
    DistFn fn = MetricDistance(Metric::L2, true);
    double sum = 0;
    *min_results = 1000;
    for (size_t q = 0; q < queries_.rows; ++q) {
      TopK top(10);
      size_t matches = 0;
      for (size_t i = 0; i < kN; ++i) {
        if (attr_[i] < threshold) {
          ++matches;
          top.push(fn(queries_.row(q), base_.row(i), base_.padded_dim), static_cast<InternalId>(i));
        }
      }
      std::set<VectorId> truth;
      for (auto& n : top.sorted()) truth.insert(n.id);
      SearchRequest req;
      req.vector = queries_.row(q);
      req.k = 10;
      req.filter = filter->get();
      req.params.strategy = strat;
      req.params.ef = 64;
      req.params.prefilter_threshold = 500;
      std::vector<SearchHit> hits;
      SearchExplain ex;
      EXPECT_TRUE(col_->Search(req, &hits, &ex).ok());
      for (const auto& h : hits) {
        EXPECT_LT(attr_[h.id], threshold) << "non-matching row returned";
        sum += truth.count(h.id);
      }
      *min_results = std::min(*min_results, hits.size());
      if (matches >= 10 && strat == FilterStrategy::Auto) {
        EXPECT_EQ(hits.size(), 10u) << "fewer than k results with " << matches << " matches";
      }
      for (const auto& se : ex.segments)
        if (se.kind == "disk") *last_strategy = se.strategy;
    }
    return sum / (10.0 * queries_.rows);
  }

  test::TempDir dir_;
  std::unique_ptr<Collection> col_;
  Dataset base_, queries_;
  std::vector<int> attr_;
};

TEST_F(FilteredSearchTest, PlannerAndExactMatchBruteForceAtEverySelectivity) {
  for (int t : {1, 10, 100, 500, 1000}) {  // 0.1% .. 100%
    size_t min_results = 0;
    std::string strategy;
    double exact = Recall(t, FilterStrategy::PreFilter, &min_results, &strategy);
    EXPECT_DOUBLE_EQ(exact, 1.0) << "exact path at threshold " << t;
    double planned = Recall(t, FilterStrategy::Auto, &min_results, &strategy);
    EXPECT_GE(planned, 0.95) << "planner at threshold " << t << " used " << strategy;
    if (t == 1) {
      EXPECT_EQ(strategy, "prefilter_exact");
    }
    if (t >= 500) {
      EXPECT_EQ(strategy.rfind("filtered_index", 0), 0u) << strategy;
    }
  }
}

TEST_F(FilteredSearchTest, GraphAndPostFilterStrategiesRun) {
  size_t min_results = 0;
  std::string strategy;
  EXPECT_GE(Recall(1000, FilterStrategy::Graph, &min_results, &strategy), 0.9);
  EXPECT_EQ(strategy, "filtered_index");
  EXPECT_GE(Recall(1000, FilterStrategy::PostFilter, &min_results, &strategy), 0.9);
  EXPECT_EQ(strategy, "post_filter");
  // Post-filtering at 0.1% selectivity comes back short: that is its failure mode.
  Recall(1, FilterStrategy::PostFilter, &min_results, &strategy);
  EXPECT_LT(min_results, 10u);
}

TEST_F(FilteredSearchTest, MetadataIsReturned) {
  auto filter = col_->ParseFilter(R"({"field": "tag", "eq": "a"})");
  ASSERT_TRUE(filter.ok());
  SearchRequest req;
  req.vector = queries_.row(0);
  req.k = 20;
  req.filter = filter->get();
  req.include_metadata = true;
  req.include_vector = true;
  std::vector<SearchHit> hits;
  ASSERT_TRUE(col_->Search(req, &hits).ok());
  ASSERT_EQ(hits.size(), 20u);
  for (const auto& h : hits) {
    EXPECT_EQ(h.id % 3, 0u);
    EXPECT_EQ(std::get<std::string>(h.metadata.at("tag")), "a");
    EXPECT_EQ(std::get<int64_t>(h.metadata.at("attr")), attr_[h.id]);
    ASSERT_EQ(h.vector.size(), kDim);
    EXPECT_EQ(h.vector[0], base_.row(h.id)[0]);
  }
}

TEST_F(FilteredSearchTest, InvalidFiltersAndMetadataAreRejected) {
  EXPECT_FALSE(col_->ParseFilter(R"({"field": "attr", "eq": "x"})").ok());
  Metadata bad{{"attr", std::string("x")}};
  EXPECT_FALSE(col_->Upsert(999999, base_.row(0), &bad).ok());
  Metadata unknown{{"nope", int64_t{1}}};
  EXPECT_FALSE(col_->Upsert(999999, base_.row(0), &unknown).ok());
}

}  // namespace
}  // namespace vdb
