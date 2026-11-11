#pragma once

// HTTP/JSON API (cpp-httplib). Errors are {"error": {"code": "...", "message": "..."}}:
// 400 wrong dimension or bad filter, 404 unknown collection or id, 409 already exists,
// 413 body over the limit, 503 with Retry-After when the search queue is full, 500
// otherwise.

#include <atomic>
#include <cstddef>
#include <string>

#include "server/collection_manager.h"
#include "server/metrics.h"

namespace httplib {
class Server;
struct Request;
struct Response;
}  // namespace httplib

namespace vdb {

struct ServerConfig {
  size_t max_body_bytes = size_t{1} << 30;
  size_t max_k = 1000;
  size_t max_batch = 1024;            // points per upsert, queries per search-batch
  size_t max_inflight_searches = 64;  // the bounded queue in front of search
  double slow_query_ms = 50;
};

class HttpApi {
 public:
  HttpApi(CollectionManager* mgr, ServerConfig cfg) : mgr_(mgr), cfg_(cfg) {}
  void Register(httplib::Server& svr);
  ServerMetrics& metrics() { return metrics_; }

 private:
  CollectionManager* mgr_;
  ServerConfig cfg_;
  ServerMetrics metrics_;
  std::atomic<size_t> inflight_{0};
};

}  // namespace vdb
