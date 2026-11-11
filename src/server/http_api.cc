#include "server/http_api.h"

#include <httplib.h>
#include <spdlog/spdlog.h>

#include <nlohmann/json.hpp>

#include "engine/options_json.h"
#include "meta/schema.h"
#include "server/binary_format.h"
#include "util/timer.h"

namespace vdb {

using json = nlohmann::json;

namespace {

const std::string kName = "([A-Za-z0-9_\\-]+)";

int HttpStatusFor(Status::Code c) {
  switch (c) {
    case Status::Code::kInvalidArgument: return 400;
    case Status::Code::kNotFound: return 404;
    case Status::Code::kAlreadyExists: return 409;
    case Status::Code::kUnavailable:
    case Status::Code::kResourceExhausted:
    case Status::Code::kFailedPrecondition: return 503;
    default: return 500;
  }
}

void SendJson(httplib::Response& res, int status, const json& j) {
  res.status = status;
  res.set_content(j.dump(), "application/json");
}

void SendError(httplib::Response& res, int status, const std::string& code, const std::string& msg) {
  SendJson(res, status, {{"error", {{"code", code}, {"message", msg}}}});
}

void SendStatus(httplib::Response& res, const Status& s) {
  SendError(res, HttpStatusFor(s.code()), CodeName(s.code()), s.message());
}

bool ParseBody(const httplib::Request& req, httplib::Response& res, json* j) {
  *j = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
  if (j->is_discarded() || !j->is_object()) {
    SendError(res, 400, "invalid_argument", "body must be a JSON object");
    return false;
  }
  return true;
}

Status ReadVector(const json& j, size_t dim, std::vector<float>* out) {
  if (!j.is_array()) return Status::InvalidArgument("vector must be a list of numbers");
  if (j.size() != dim) {
    return Status::InvalidArgument("wrong dimension: got " + std::to_string(j.size()) + ", expected " +
                                   std::to_string(dim));
  }
  out->resize(dim);
  for (size_t i = 0; i < dim; ++i) {
    if (!j[i].is_number()) return Status::InvalidArgument("vector must be a list of numbers");
    (*out)[i] = j[i].get<float>();
  }
  return Status::OK();
}

Status ReadId(const json& j, VectorId* id) {
  if (j.is_number_unsigned()) {
    *id = j.get<uint64_t>();
  } else if (j.is_number_integer() && j.get<int64_t>() >= 0) {
    *id = static_cast<uint64_t>(j.get<int64_t>());
  } else {
    return Status::InvalidArgument("id must be a non-negative integer");
  }
  return Status::OK();
}

template <class T>
Status ReadSize(const json& j, const char* key, T* out) {
  if (!j.contains(key) || j[key].is_null()) return Status::OK();
  if (!j[key].is_number_integer()) return Status::InvalidArgument(std::string(key) + " must be an integer");
  int64_t v = j[key].get<int64_t>();
  if (v < 0 && !std::is_signed_v<T>) return Status::InvalidArgument(std::string(key) + " must be >= 0");
  *out = static_cast<T>(v);
  return Status::OK();
}

json HitToJson(const SearchHit& h, bool meta, bool vec) {
  json o = {{"id", h.id}, {"score", h.score}, {"distance", h.distance}};
  if (meta) o["metadata"] = MetadataToJson(h.metadata);
  if (vec) o["vector"] = h.vector;
  return o;
}

json ExplainToJson(const SearchExplain& ex) {
  json segs = json::array();
  for (const auto& s : ex.segments) {
    segs.push_back({{"segment", s.segment_id},
                    {"kind", s.kind},
                    {"index", s.index},
                    {"strategy", s.strategy},
                    {"rows", s.rows},
                    {"live_rows", s.live_rows},
                    {"matches", s.matches},
                    {"selectivity", s.selectivity},
                    {"ef", s.ef},
                    {"distance_computations", s.distance_computations},
                    {"results", s.results},
                    {"micros", s.micros}});
  }
  return {{"segments", segs}, {"distance_computations", ex.distance_computations}, {"total_micros", ex.total_micros}};
}

json StatsToJson(const std::string& name, const Collection& c, bool full) {
  CollectionStats st = c.Stats();
  json j = {{"name", name},
            {"rows", st.live_rows},
            {"segments", st.segments.size()},
            {"disk_bytes", st.disk_bytes}};
  if (!full) return j;
  json segs = json::array();
  for (const auto& s : st.segments) {
    segs.push_back({{"id", s.id}, {"kind", s.kind}, {"index", s.index}, {"rows", s.rows}, {"deleted", s.deleted},
                    {"lsn", s.lsn}, {"memory_bytes", s.memory_bytes}, {"disk_bytes", s.disk_bytes}});
  }
  j["config"] = OptionsToJson(c.options());
  j["total_rows"] = st.total_rows;
  j["deleted_rows"] = st.deleted_rows;
  j["deleted_ratio"] = st.total_rows == 0 ? 0.0 : static_cast<double>(st.deleted_rows) / static_cast<double>(st.total_rows);
  j["segment_list"] = segs;
  j["memory"] = {{"vectors", st.memory.vectors}, {"graph", st.memory.graph}, {"codes", st.memory.codes},
                 {"id_map", st.memory.id_map}, {"metadata", st.memory.metadata}, {"total", st.memory.total()}};
  j["last_lsn"] = st.last_lsn;
  j["flushed_lsn"] = st.flushed_lsn;
  j["flushes"] = st.flushes;
  j["compactions"] = st.compactions;
  j["last_recovery_seconds"] = st.last_recovery_seconds;
  return j;
}

// One search body -> {"results": [...], ...}. Returns the HTTP-facing status.
Status RunSearch(const std::string& name, const Collection& c, const json& j, const ServerConfig& cfg,
                 json* out) {
  if (!j.is_object()) return Status::InvalidArgument("a search must be a JSON object");
  if (!j.contains("vector")) return Status::InvalidArgument("missing \"vector\"");
  std::vector<float> v;
  VDB_RETURN_IF_ERROR(ReadVector(j["vector"], c.options().dim, &v));
  SearchRequest req;
  req.vector = v.data();
  VDB_RETURN_IF_ERROR(ReadSize(j, "k", &req.k));
  if (req.k == 0 || req.k > cfg.max_k) return Status::InvalidArgument("k must be in 1.." + std::to_string(cfg.max_k));
  VDB_RETURN_IF_ERROR(ReadSize(j, "ef", &req.params.ef));
  VDB_RETURN_IF_ERROR(ReadSize(j, "nprobe", &req.params.nprobe));
  VDB_RETURN_IF_ERROR(ReadSize(j, "rerank", &req.params.rerank));
  VDB_RETURN_IF_ERROR(ReadSize(j, "prefilter_threshold", &req.params.prefilter_threshold));
  VDB_RETURN_IF_ERROR(ReadSize(j, "post_filter_alpha", &req.params.post_filter_alpha));
  if (req.params.ef > 10000) return Status::InvalidArgument("ef must be <= 10000");
  if (req.params.post_filter_alpha == 0 || req.params.post_filter_alpha > 1000) {
    return Status::InvalidArgument("post_filter_alpha must be in 1..1000");
  }
  if (j.contains("strategy")) {
    if (!j["strategy"].is_string() || !ParseFilterStrategy(j["strategy"].get<std::string>(), &req.params.strategy)) {
      return Status::InvalidArgument("strategy must be auto, prefilter, graph or postfilter");
    }
  }
  std::unique_ptr<Filter> filter;
  if (j.contains("filter") && !j["filter"].is_null()) {
    auto f = c.ParseFilter(j["filter"].dump());
    if (!f.ok()) return f.status();
    filter = std::move(f).value();
    req.filter = filter.get();
  }
  if (j.contains("include")) {
    if (!j["include"].is_array()) return Status::InvalidArgument("include must be a list");
    for (const auto& e : j["include"]) {
      if (e == "metadata") {
        req.include_metadata = true;
      } else if (e == "vector") {
        req.include_vector = true;
      } else {
        return Status::InvalidArgument("include accepts \"metadata\" and \"vector\"");
      }
    }
  }
  bool explain = j.contains("explain") && j["explain"].is_boolean() && j["explain"].get<bool>();
  Timer t;
  std::vector<SearchHit> hits;
  SearchExplain ex;
  VDB_RETURN_IF_ERROR(c.Search(req, &hits, &ex));
  double ms = t.millis();
  json results = json::array();
  for (const auto& h : hits) results.push_back(HitToJson(h, req.include_metadata, req.include_vector));
  *out = {{"results", results}, {"took_ms", ms}};
  if (explain) (*out)["explain"] = ExplainToJson(ex);
  if (ms > cfg.slow_query_ms) {
    spdlog::warn("slow query: {:.1f} ms on {}: {}", ms, name, ExplainToJson(ex).dump());
  }
  return Status::OK();
}

}  // namespace

void HttpApi::Register(httplib::Server& svr) {
  svr.set_payload_max_length(cfg_.max_body_bytes);
  // cpp-httplib never reads a body sent with GET/HEAD, so on a keep-alive connection those bytes
  // would be parsed as the next request. Serve the request, then close the connection.
  svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response&) {
    if ((req.method == "GET" || req.method == "HEAD") &&
        (req.get_header_value_u64("Content-Length") > 0 || req.has_header("Transfer-Encoding"))) {
      auto& r = const_cast<httplib::Request&>(req);  // the server's own local, not a const object
      r.headers.erase("Connection");
      r.set_header("Connection", "close");
    }
    return httplib::Server::HandlerResponse::Unhandled;
  });
  svr.set_error_handler([](const httplib::Request&, httplib::Response& res) {
    if (!res.body.empty()) return;  // a handler already wrote a JSON error
    std::string code = res.status == 404   ? "not_found"
                       : res.status == 413 ? "payload_too_large"
                       : res.status == 400 ? "invalid_argument"
                                           : "error";
    SendJson(res, res.status, {{"error", {{"code", code}, {"message", httplib::status_message(res.status)}}}});
  });

  using Handler = std::function<void(const httplib::Request&, httplib::Response&)>;
  auto wrap = [this](std::string route, Handler fn) {
    return [this, route = std::move(route), fn = std::move(fn)](const httplib::Request& req, httplib::Response& res) {
      Timer t;
      try {
        fn(req, res);
      } catch (const std::exception& e) {
        SendError(res, 500, "internal", e.what());
      }
      if (res.status == -1) res.status = 200;
      double sec = t.seconds();
      metrics_.Record(route, res.status, sec);
      spdlog::debug("{} {} -> {} in {:.2f} ms", req.method, req.path, res.status, sec * 1e3);
    };
  };
  auto lookup = [this](const httplib::Request& req, httplib::Response& res) {
    std::string name = req.matches[1];
    auto c = mgr_->Get(name);
    if (!c) SendError(res, 404, "not_found", "collection " + name + " not found");
    return c;
  };

  svr.Get("/health", wrap("health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("OK\n", "text/plain");
          }));

  svr.Get("/metrics", wrap("metrics", [this](const httplib::Request&, httplib::Response& res) {
            res.set_content(metrics_.Render(*mgr_), "text/plain; version=0.0.4");
          }));

  svr.Post("/v1/collections", wrap("create_collection", [this](const httplib::Request& req, httplib::Response& res) {
             json j;
             if (!ParseBody(req, res, &j)) return;
             if (!j.contains("name") || !j["name"].is_string()) {
               SendError(res, 400, "invalid_argument", "missing \"name\"");
               return;
             }
             std::string name = j["name"].get<std::string>();
             CollectionOptions o;
             Status s = OptionsFromJson(j, &o);
             if (s.ok()) s = ValidateOptions(o);
             if (!s.ok()) return SendStatus(res, s);
             auto c = mgr_->Create(name, o);
             if (!c.ok()) return SendStatus(res, c.status());
             SendJson(res, 201, {{"name", name}, {"config", OptionsToJson(o)}});
           }));

  svr.Get("/v1/collections", wrap("list_collections", [this](const httplib::Request&, httplib::Response& res) {
            json arr = json::array();
            for (const auto& [name, c] : mgr_->List()) arr.push_back(StatsToJson(name, *c, false));
            SendJson(res, 200, {{"collections", arr}});
          }));

  svr.Get("/v1/collections/" + kName, wrap("get_collection", [lookup](const httplib::Request& req, httplib::Response& res) {
            auto c = lookup(req, res);
            if (c) SendJson(res, 200, StatsToJson(req.matches[1], *c, true));
          }));

  svr.Delete("/v1/collections/" + kName, wrap("drop_collection", [this](const httplib::Request& req, httplib::Response& res) {
               Status s = mgr_->Drop(req.matches[1]);
               if (!s.ok()) return SendStatus(res, s);
               SendJson(res, 200, {{"dropped", std::string(req.matches[1])}});
             }));

  svr.Post("/v1/collections/" + kName + "/upsert", wrap("upsert", [this, lookup](const httplib::Request& req, httplib::Response& res) {
             auto c = lookup(req, res);
             if (!c) return;
             json j;
             if (!ParseBody(req, res, &j)) return;
             if (!j.contains("points") || !j["points"].is_array()) {
               return SendError(res, 400, "invalid_argument", "missing \"points\" list");
             }
             const json& pts = j["points"];
             if (pts.size() > cfg_.max_batch) {
               return SendError(res, 400, "invalid_argument", "at most " + std::to_string(cfg_.max_batch) + " points per request");
             }
             const size_t dim = c->options().dim;
             std::vector<VectorId> ids(pts.size());
             std::vector<std::vector<float>> vecs(pts.size());
             std::vector<Metadata> metas(pts.size());
             std::vector<UpsertPoint> points(pts.size());
             for (size_t i = 0; i < pts.size(); ++i) {
               const json& p = pts[i];
               if (!p.is_object() || !p.contains("id") || !p.contains("vector")) {
                 return SendError(res, 400, "invalid_argument", "each point needs \"id\" and \"vector\"");
               }
               Status s = ReadId(p["id"], &ids[i]);
               if (s.ok()) s = ReadVector(p["vector"], dim, &vecs[i]);
               if (s.ok() && p.contains("metadata")) s = MetadataFromJson(p["metadata"], c->options().schema, &metas[i]);
               if (!s.ok()) return SendStatus(res, s);
               points[i] = UpsertPoint{ids[i], vecs[i].data(), &metas[i]};
             }
             uint64_t lsn = 0;
             Status s = c->UpsertBatch(points, &lsn);
             if (!s.ok()) return SendStatus(res, s);
             SendJson(res, 200, {{"acknowledged", points.size()}, {"last_lsn", lsn}});
           }));

  svr.Post("/v1/collections/" + kName + "/upsert-binary", wrap("upsert_binary", [lookup](const httplib::Request& req, httplib::Response& res) {
             auto c = lookup(req, res);
             if (!c) return;
             BinaryBatch b;
             Status s = ParseBinaryBatch(req.body, c->options().dim, &b);
             if (!s.ok()) return SendStatus(res, s);
             uint64_t lsn = 0;
             s = c->UpsertBatch(b.ids.data(), b.vectors.data(), b.ids.size(), nullptr, &lsn);
             if (!s.ok()) return SendStatus(res, s);
             SendJson(res, 200, {{"acknowledged", b.ids.size()}, {"last_lsn", lsn}});
           }));

  svr.Post("/v1/collections/" + kName + "/bulk-load", wrap("bulk_load", [lookup](const httplib::Request& req, httplib::Response& res) {
             auto c = lookup(req, res);
             if (!c) return;
             BinaryBatch b;
             Status s = ParseBinaryBatch(req.body, c->options().dim, &b);
             if (!s.ok()) return SendStatus(res, s);
             s = c->BulkLoad(b.ids.data(), b.vectors.data(), b.ids.size());
             if (!s.ok()) return SendStatus(res, s);
             SendJson(res, 200, {{"loaded", b.ids.size()}, {"segments", c->Stats().segments.size()}});
           }));

  svr.Post("/v1/collections/" + kName + "/delete", wrap("delete", [this, lookup](const httplib::Request& req, httplib::Response& res) {
             auto c = lookup(req, res);
             if (!c) return;
             json j;
             if (!ParseBody(req, res, &j)) return;
             if (!j.contains("ids") || !j["ids"].is_array()) return SendError(res, 400, "invalid_argument", "missing \"ids\" list");
             if (j["ids"].size() > cfg_.max_batch * 16) return SendError(res, 400, "invalid_argument", "too many ids");
             std::vector<VectorId> ids;
             for (const auto& e : j["ids"]) {
               VectorId id = 0;
               Status s = ReadId(e, &id);
               if (!s.ok()) return SendStatus(res, s);
               ids.push_back(id);
             }
             size_t removed = 0;
             Status s = c->RemoveBatch(ids.data(), ids.size(), &removed);
             if (!s.ok()) return SendStatus(res, s);
             SendJson(res, 200, {{"deleted", removed}});
           }));

  svr.Get("/v1/collections/" + kName + "/points/([0-9]+)", wrap("get_point", [lookup](const httplib::Request& req, httplib::Response& res) {
            auto c = lookup(req, res);
            if (!c) return;
            VectorId id = 0;
            try {
              id = std::stoull(req.matches[2].str());
            } catch (const std::exception&) {
              return SendError(res, 400, "invalid_argument", "bad id");
            }
            std::vector<float> v;
            Metadata md;
            Status s = c->Get(id, &v, &md);
            if (!s.ok()) return SendStatus(res, s);
            SendJson(res, 200, {{"id", id}, {"vector", v}, {"metadata", MetadataToJson(md)}});
          }));

  svr.Post("/v1/collections/" + kName + "/search", wrap("search", [this, lookup](const httplib::Request& req, httplib::Response& res) {
             auto c = lookup(req, res);
             if (!c) return;
             // The bounded queue in front of search: overload answers 503 instead of piling up.
             if (inflight_.fetch_add(1) >= cfg_.max_inflight_searches) {
               inflight_.fetch_sub(1);
               res.set_header("Retry-After", "1");
               return SendError(res, 503, "unavailable", "search queue is full; retry later");
             }
             struct Release {
               std::atomic<size_t>& n;
               ~Release() { n.fetch_sub(1); }
             } release{inflight_};
             json j;
             if (!ParseBody(req, res, &j)) return;
             json out;
             Status s = RunSearch(req.matches[1], *c, j, cfg_, &out);
             if (!s.ok()) return SendStatus(res, s);
             SendJson(res, 200, out);
           }));

  svr.Post("/v1/collections/" + kName + "/search-batch", wrap("search_batch", [this, lookup](const httplib::Request& req, httplib::Response& res) {
             auto c = lookup(req, res);
             if (!c) return;
             if (inflight_.fetch_add(1) >= cfg_.max_inflight_searches) {
               inflight_.fetch_sub(1);
               res.set_header("Retry-After", "1");
               return SendError(res, 503, "unavailable", "search queue is full; retry later");
             }
             struct Release {
               std::atomic<size_t>& n;
               ~Release() { n.fetch_sub(1); }
             } release{inflight_};
             json j;
             if (!ParseBody(req, res, &j)) return;
             if (!j.contains("queries") || !j["queries"].is_array()) {
               return SendError(res, 400, "invalid_argument", "missing \"queries\" list");
             }
             if (j["queries"].size() > cfg_.max_batch) {
               return SendError(res, 400, "invalid_argument", "at most " + std::to_string(cfg_.max_batch) + " queries per batch");
             }
             json all = json::array();
             for (const auto& q : j["queries"]) {
               json out;
               Status s = RunSearch(req.matches[1], *c, q, cfg_, &out);
               if (!s.ok()) return SendStatus(res, s);
               all.push_back(out["results"]);
             }
             SendJson(res, 200, {{"results", all}});
           }));

  svr.Post("/v1/collections/" + kName + "/flush", wrap("flush", [lookup](const httplib::Request& req, httplib::Response& res) {
             auto c = lookup(req, res);
             if (!c) return;
             c->RequestFlush();
             SendJson(res, 202, {{"status", "accepted"}});
           }));

  svr.Post("/v1/collections/" + kName + "/compact", wrap("compact", [lookup](const httplib::Request& req, httplib::Response& res) {
             auto c = lookup(req, res);
             if (!c) return;
             size_t target = 0;
             if (req.has_param("target_segments")) {
               try {
                 target = std::stoul(req.get_param_value("target_segments"));
               } catch (const std::exception&) {
                 return SendError(res, 400, "invalid_argument", "target_segments must be an integer");
               }
             }
             c->RequestCompaction(target);
             SendJson(res, 202, {{"status", "accepted"}, {"target_segments", target}});
           }));
}

}  // namespace vdb
