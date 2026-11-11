#include "server/metrics.h"

#include <cstdio>

#include "server/collection_manager.h"

namespace vdb {
namespace {

std::string Num(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.9g", v);
  return buf;
}

void Type(std::string* out, const char* name, const char* type, const char* help) {
  *out += std::string("# HELP ") + name + " " + help + "\n# TYPE " + name + " " + type + "\n";
}

}  // namespace

void ServerMetrics::Record(const std::string& route, int code, double seconds) {
  std::lock_guard<std::mutex> g(mu_);
  ++requests_[{route, code}];
  auto& h = durations_[route];
  if (!h) h = std::make_unique<Histogram>(ExponentialBuckets(0.0001, 2, 18));
  h->observe(seconds);
}

std::string ServerMetrics::Render(const CollectionManager& mgr) const {
  std::string out;
  {
    std::lock_guard<std::mutex> g(mu_);
    Type(&out, "vdb_requests_total", "counter", "HTTP requests by route and status code.");
    for (const auto& [key, n] : requests_) {
      out += "vdb_requests_total{route=\"" + key.first + "\",code=\"" + std::to_string(key.second) +
             "\"} " + std::to_string(n) + "\n";
    }
    Type(&out, "vdb_request_duration_seconds", "histogram", "HTTP request latency by route.");
    for (const auto& [route, h] : durations_) {
      AppendPrometheusHistogram(&out, "vdb_request_duration_seconds", "route=\"" + route + "\"", h->snapshot());
    }
  }
  EngineMetrics& em = GlobalMetrics();
  Type(&out, "vdb_search_distance_computations", "histogram", "Distance computations per search.");
  AppendPrometheusHistogram(&out, "vdb_search_distance_computations", "", em.search_distance_computations.snapshot());
  Type(&out, "vdb_wal_bytes_total", "counter", "Bytes appended to write-ahead logs.");
  out += "vdb_wal_bytes_total " + std::to_string(em.wal_bytes_total.value()) + "\n";
  Type(&out, "vdb_wal_records_total", "counter", "Records appended to write-ahead logs.");
  out += "vdb_wal_records_total " + std::to_string(em.wal_records_total.value()) + "\n";
  Type(&out, "vdb_wal_fsync_duration_seconds", "histogram", "Log fsync latency.");
  AppendPrometheusHistogram(&out, "vdb_wal_fsync_duration_seconds", "", em.wal_fsync_seconds.snapshot());
  Type(&out, "vdb_flush_duration_seconds", "histogram", "Mutable segment flush duration.");
  AppendPrometheusHistogram(&out, "vdb_flush_duration_seconds", "", em.flush_seconds.snapshot());
  Type(&out, "vdb_compaction_duration_seconds", "histogram", "Compaction duration.");
  AppendPrometheusHistogram(&out, "vdb_compaction_duration_seconds", "", em.compaction_seconds.snapshot());

  std::string rows, segs, ratio, mem;
  for (const auto& [name, c] : mgr.List()) {
    CollectionStats st = c->Stats();
    std::string lbl = "collection=\"" + name + "\"";
    rows += "vdb_collection_rows{" + lbl + "} " + std::to_string(st.live_rows) + "\n";
    segs += "vdb_segments{" + lbl + "} " + std::to_string(st.segments.size()) + "\n";
    double r = st.total_rows == 0 ? 0.0 : static_cast<double>(st.deleted_rows) / static_cast<double>(st.total_rows);
    ratio += "vdb_deleted_ratio{" + lbl + "} " + Num(r) + "\n";
    const std::pair<const char*, size_t> comps[] = {{"vectors", st.memory.vectors},
                                                    {"graph", st.memory.graph},
                                                    {"codes", st.memory.codes},
                                                    {"id_map", st.memory.id_map},
                                                    {"metadata", st.memory.metadata}};
    for (const auto& [comp, bytes] : comps) {
      mem += "vdb_memory_bytes{" + lbl + ",component=\"" + comp + "\"} " + std::to_string(bytes) + "\n";
    }
  }
  Type(&out, "vdb_collection_rows", "gauge", "Live rows per collection.");
  out += rows;
  Type(&out, "vdb_segments", "gauge", "Segments per collection (including the mutable one).");
  out += segs;
  Type(&out, "vdb_deleted_ratio", "gauge", "Deleted rows not yet compacted away, as a fraction.");
  out += ratio;
  Type(&out, "vdb_memory_bytes", "gauge", "Memory by component.");
  out += mem;
  return out;
}

}  // namespace vdb
