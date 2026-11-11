// The HTTP API in-process: contract, error codes and metrics.

#include <httplib.h>
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <thread>

#include "server/binary_format.h"
#include "server/http_api.h"
#include "test_util.h"

namespace vdb {
namespace {

using json = nlohmann::json;

class ServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    OpenOptions oo;
    oo.num_threads = 2;
    mgr_ = std::make_unique<CollectionManager>(dir_.path(), oo);
    ASSERT_TRUE(mgr_->OpenAll().ok());
    api_ = std::make_unique<HttpApi>(mgr_.get(), Config());
    api_->Register(svr_);
    svr_.set_tcp_nodelay(true);
    port_ = svr_.bind_to_any_port("127.0.0.1");
    thread_ = std::thread([this] { svr_.listen_after_bind(); });
    svr_.wait_until_ready();
    cli_ = std::make_unique<httplib::Client>("127.0.0.1", port_);
  }
  void TearDown() override {
    svr_.stop();
    thread_.join();
  }

  virtual ServerConfig Config() {
    ServerConfig cfg;
    cfg.max_batch = 100;
    return cfg;
  }

  uint64_t Counter(const std::string& line_prefix) {
    auto r = cli_->Get("/metrics");
    EXPECT_TRUE(r);
    // Match a sample line, not the "# HELP <name> ..." line above it.
    size_t p = r ? r->body.find("\n" + line_prefix) : std::string::npos;
    if (p == std::string::npos) return 0;
    return std::stoull(r->body.substr(p + 1 + line_prefix.size()));
  }

  json Post(const std::string& path, const json& body, int* status) {
    auto r = cli_->Post(path, body.dump(), "application/json");
    EXPECT_TRUE(r);
    *status = r ? r->status : -1;
    return r ? json::parse(r->body, nullptr, false) : json();
  }
  json Get(const std::string& path, int* status) {
    auto r = cli_->Get(path);
    EXPECT_TRUE(r);
    *status = r ? r->status : -1;
    return r ? json::parse(r->body, nullptr, false) : json();
  }

  void CreateDocs(int* status) {
    Post("/v1/collections",
         {{"name", "docs"}, {"dim", 4}, {"metric", "l2"}, {"index", {{"type", "hnsw"}, {"M", 8}, {"ef_construction", 50}}},
          {"schema", {{"year", "int64"}, {"tags", "keyword_list"}}}, {"sync", "always"}},
         status);
  }

  test::TempDir dir_;
  std::unique_ptr<CollectionManager> mgr_;
  std::unique_ptr<HttpApi> api_;
  httplib::Server svr_;
  int port_ = 0;
  std::thread thread_;
  std::unique_ptr<httplib::Client> cli_;
};

TEST_F(ServerTest, CollectionLifecycleAndSearch) {
  int st = 0;
  CreateDocs(&st);
  ASSERT_EQ(st, 201);
  CreateDocs(&st);
  EXPECT_EQ(st, 409);
  json pts = json::array();
  for (int i = 0; i < 50; ++i) {
    pts.push_back({{"id", i},
                   {"vector", {i * 1.0, 0.0, 0.0, 1.0}},
                   {"metadata", {{"year", 2000 + i}, {"tags", i % 2 ? json{"odd"} : json{"even", "x"}}}}});
  }
  json up = Post("/v1/collections/docs/upsert", {{"points", pts}}, &st);
  ASSERT_EQ(st, 200) << up.dump();
  EXPECT_EQ(up["acknowledged"], 50);
  EXPECT_EQ(up["last_lsn"], 50);

  json res = Post("/v1/collections/docs/search", {{"vector", {10.2, 0, 0, 1}}, {"k", 3}, {"include", {"metadata"}}, {"explain", true}}, &st);
  ASSERT_EQ(st, 200) << res.dump();
  ASSERT_EQ(res["results"].size(), 3u);
  EXPECT_EQ(res["results"][0]["id"], 10);
  EXPECT_EQ(res["results"][0]["metadata"]["year"], 2010);
  EXPECT_TRUE(res.contains("explain"));

  res = Post("/v1/collections/docs/search",
             {{"vector", {10.2, 0, 0, 1}}, {"k", 5},
              {"filter", {{"and", {{{"field", "year"}, {"gte", 2030}}, {{"field", "tags"}, {"in", {"odd"}}}}}}}},
             &st);
  ASSERT_EQ(st, 200) << res.dump();
  for (const auto& h : res["results"]) {
    EXPECT_GE(h["id"].get<int>(), 30);
    EXPECT_EQ(h["id"].get<int>() % 2, 1);
  }

  json batch = Post("/v1/collections/docs/search-batch",
                    {{"queries", {{{"vector", {1, 0, 0, 1}}, {"k", 1}}, {{"vector", {40, 0, 0, 1}}, {"k", 2}}}}}, &st);
  ASSERT_EQ(st, 200);
  ASSERT_EQ(batch["results"].size(), 2u);
  EXPECT_EQ(batch["results"][0][0]["id"], 1);
  EXPECT_EQ(batch["results"][1][0]["id"], 40);

  json pt = Get("/v1/collections/docs/points/7", &st);
  ASSERT_EQ(st, 200);
  EXPECT_EQ(pt["vector"][0], 7.0);
  EXPECT_EQ(pt["metadata"]["year"], 2007);

  json del = Post("/v1/collections/docs/delete", {{"ids", {7, 8, 1000}}}, &st);
  ASSERT_EQ(st, 200);
  EXPECT_EQ(del["deleted"], 2);
  Get("/v1/collections/docs/points/7", &st);
  EXPECT_EQ(st, 404);

  auto r = cli_->Post("/v1/collections/docs/flush", "", "application/json");
  ASSERT_TRUE(r);
  EXPECT_EQ(r->status, 202);
  r = cli_->Post("/v1/collections/docs/compact?target_segments=1", "", "application/json");
  ASSERT_TRUE(r);
  EXPECT_EQ(r->status, 202);

  json info = Get("/v1/collections/docs", &st);
  ASSERT_EQ(st, 200);
  EXPECT_EQ(info["rows"], 48);
  EXPECT_EQ(info["config"]["dim"], 4);
  json list = Get("/v1/collections", &st);
  ASSERT_EQ(list["collections"].size(), 1u);

  r = cli_->Delete("/v1/collections/docs");
  ASSERT_TRUE(r);
  EXPECT_EQ(r->status, 200);
  Get("/v1/collections/docs", &st);
  EXPECT_EQ(st, 404);
}

TEST_F(ServerTest, ErrorsUseTheDocumentedCodes) {
  int st = 0;
  CreateDocs(&st);
  ASSERT_EQ(st, 201);
  json e = Post("/v1/collections/nope/search", {{"vector", {1, 2, 3, 4}}}, &st);
  EXPECT_EQ(st, 404);
  EXPECT_EQ(e["error"]["code"], "not_found");
  e = Post("/v1/collections/docs/search", {{"vector", {1, 2, 3}}}, &st);
  EXPECT_EQ(st, 400);
  EXPECT_NE(e["error"]["message"].get<std::string>().find("dimension"), std::string::npos);
  e = Post("/v1/collections/docs/search", {{"vector", {1, 2, 3, 4}}, {"filter", {{"field", "nope"}, {"eq", 1}}}}, &st);
  EXPECT_EQ(st, 400);
  e = Post("/v1/collections/docs/search", {{"vector", {1, 2, 3, 4}}, {"k", 100000}}, &st);
  EXPECT_EQ(st, 400);
  auto r = cli_->Post("/v1/collections/docs/upsert", "{not json", "application/json");
  ASSERT_TRUE(r);
  EXPECT_EQ(r->status, 400);
  json many = json::array();
  for (int i = 0; i < 101; ++i) many.push_back({{"id", i}, {"vector", {0, 0, 0, 1}}});
  Post("/v1/collections/docs/upsert", {{"points", many}}, &st);
  EXPECT_EQ(st, 400);  // over max_batch
  Post("/v1/collections", {{"name", "bad name!"}, {"dim", 4}}, &st);
  EXPECT_EQ(st, 400);
  Post("/v1/collections", {{"name", "x"}, {"dim", 4}, {"index", {{"type", "ivf_pq"}, {"pq_m", 3}}}}, &st);
  EXPECT_EQ(st, 400);
  r = cli_->Get("/no/such/route");
  ASSERT_TRUE(r);
  EXPECT_EQ(r->status, 404);
}

TEST_F(ServerTest, BinaryEndpointsAndMetrics) {
  int st = 0;
  CreateDocs(&st);
  ASSERT_EQ(st, 201);
  std::vector<VectorId> ids;
  std::vector<float> vecs;
  for (VectorId i = 0; i < 200; ++i) {
    ids.push_back(1000 + i);
    for (int d = 0; d < 4; ++d) vecs.push_back(static_cast<float>(i) + d);
  }
  std::string body = EncodeBinaryBatch(ids.data(), vecs.data(), 100, 4);
  auto r = cli_->Post("/v1/collections/docs/bulk-load", body, "application/octet-stream");
  ASSERT_TRUE(r);
  ASSERT_EQ(r->status, 200) << r->body;
  r = cli_->Post("/v1/collections/docs/bulk-load", body, "application/octet-stream");
  EXPECT_EQ(r->status, 409);  // ids already present
  body = EncodeBinaryBatch(ids.data() + 100, vecs.data() + 400, 100, 4);
  r = cli_->Post("/v1/collections/docs/upsert-binary", body, "application/octet-stream");
  ASSERT_EQ(r->status, 200) << r->body;
  r = cli_->Post("/v1/collections/docs/upsert-binary", body.substr(0, body.size() - 3), "application/octet-stream");
  EXPECT_EQ(r->status, 400);
  json info = Get("/v1/collections/docs", &st);
  EXPECT_EQ(info["rows"], 200);

  r = cli_->Get("/metrics");
  ASSERT_TRUE(r);
  for (const char* m : {"vdb_requests_total", "vdb_request_duration_seconds_bucket", "vdb_search_distance_computations",
                        "vdb_collection_rows{collection=\"docs\"} 200", "vdb_segments", "vdb_deleted_ratio",
                        "vdb_wal_bytes_total", "vdb_wal_fsync_duration_seconds", "vdb_flush_duration_seconds",
                        "vdb_compaction_duration_seconds", "vdb_memory_bytes"}) {
    EXPECT_NE(r->body.find(m), std::string::npos) << m;
  }
  r = cli_->Get("/health");
  EXPECT_EQ(r->status, 200);
}

TEST_F(ServerTest, MetricsMoveUnderLoad) {
  int st = 0;
  CreateDocs(&st);
  ASSERT_EQ(st, 201);
  Post("/v1/collections/docs/upsert", {{"points", {{{"id", 1}, {"vector", {1, 0, 0, 0}}}}}}, &st);
  ASSERT_EQ(st, 200);
  const std::string searches = "vdb_requests_total{route=\"search\",code=\"200\"} ";
  const std::string wal = "vdb_wal_bytes_total ";
  uint64_t s0 = Counter(searches), w0 = Counter(wal);
  for (int i = 0; i < 25; ++i) Post("/v1/collections/docs/search", {{"vector", {1, 0, 0, 0}}, {"k", 1}}, &st);
  Post("/v1/collections/docs/upsert", {{"points", {{{"id", 2}, {"vector", {0, 1, 0, 0}}}}}}, &st);
  EXPECT_EQ(Counter(searches), s0 + 25);
  EXPECT_GT(Counter(wal), w0);
  EXPECT_EQ(Counter("vdb_collection_rows{collection=\"docs\"} "), 2u);
}

// A server with a tiny body limit and no room in the search queue.
class TightServerTest : public ServerTest {
 protected:
  ServerConfig Config() override {
    ServerConfig cfg;
    cfg.max_body_bytes = 2048;
    cfg.max_inflight_searches = 0;  // every search finds the queue full
    return cfg;
  }
};

TEST_F(TightServerTest, BodyOverTheLimitIs413) {
  int st = 0;
  CreateDocs(&st);
  ASSERT_EQ(st, 201);
  std::string big(4096, ' ');
  auto r = cli_->Post("/v1/collections/docs/upsert", "{\"points\": []}" + big, "application/json");
  ASSERT_TRUE(r);
  EXPECT_EQ(r->status, 413);
  json e = json::parse(r->body, nullptr, false);
  EXPECT_EQ(e["error"]["code"], "payload_too_large");
}

TEST_F(TightServerTest, FullSearchQueueIs503WithRetryAfter) {
  int st = 0;
  CreateDocs(&st);
  ASSERT_EQ(st, 201);
  for (const char* path : {"/v1/collections/docs/search", "/v1/collections/docs/search-batch"}) {
    auto r = cli_->Post(path, R"({"vector": [1, 0, 0, 0], "queries": []})", "application/json");
    ASSERT_TRUE(r);
    EXPECT_EQ(r->status, 503) << path;
    EXPECT_EQ(r->get_header_value("Retry-After"), "1");
    EXPECT_EQ(json::parse(r->body)["error"]["code"], "unavailable");
  }
  // Writes are not behind the search queue.
  json up = Post("/v1/collections/docs/upsert", {{"points", {{{"id", 1}, {"vector", {1, 0, 0, 0}}}}}}, &st);
  EXPECT_EQ(st, 200) << up.dump();
}

TEST(BinaryFormatTest, RejectsMalformedBodies) {
  BinaryBatch b;
  EXPECT_FALSE(ParseBinaryBatch("", 4, &b).ok());
  EXPECT_FALSE(ParseBinaryBatch("XXXX0000000000000000", 4, &b).ok());
  VectorId id = 1;
  float v[4] = {1, 2, 3, 4};
  std::string ok = EncodeBinaryBatch(&id, v, 1, 4);
  EXPECT_TRUE(ParseBinaryBatch(ok, 4, &b).ok());
  EXPECT_FALSE(ParseBinaryBatch(ok, 5, &b).ok());
  std::string huge = ok;
  huge[8] = static_cast<char>(0xff);  // count no longer matches the size
  huge[9] = static_cast<char>(0xff);
  EXPECT_FALSE(ParseBinaryBatch(huge, 4, &b).ok());
}

}  // namespace
}  // namespace vdb
