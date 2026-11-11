#pragma once

// Lock-free counters and histograms. The engine updates the global ones; the server
// renders them (plus its own) in Prometheus text format.

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vdb {

class Counter {
 public:
  void inc(uint64_t n = 1) { v_.fetch_add(n, std::memory_order_relaxed); }
  uint64_t value() const { return v_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint64_t> v_{0};
};

class Histogram {
 public:
  explicit Histogram(std::vector<double> upper_bounds);
  void observe(double v);

  struct Snapshot {
    std::vector<double> bounds;
    std::vector<uint64_t> counts;  // per bucket (not cumulative); last one is +Inf
    uint64_t count = 0;
    double sum = 0;
  };
  Snapshot snapshot() const;

 private:
  std::vector<double> bounds_;
  std::unique_ptr<std::atomic<uint64_t>[]> counts_;
  std::atomic<uint64_t> count_{0};
  std::atomic<double> sum_{0};
};

std::vector<double> ExponentialBuckets(double start, double factor, size_t count);

struct EngineMetrics {
  Counter wal_bytes_total;
  Counter wal_records_total;
  Counter wal_syncs_total;
  Counter flushes_total;
  Counter compactions_total;
  Counter searches_total;
  Histogram wal_fsync_seconds{ExponentialBuckets(1e-5, 2.5, 14)};
  Histogram flush_seconds{ExponentialBuckets(1e-3, 2.5, 14)};
  Histogram compaction_seconds{ExponentialBuckets(1e-2, 2.5, 14)};
  Histogram search_distance_computations{ExponentialBuckets(16, 2, 18)};
};

EngineMetrics& GlobalMetrics();

// Appends one histogram in Prometheus text format; `labels` is e.g. `route="search"`.
void AppendPrometheusHistogram(std::string* out, const std::string& name,
                               const std::string& labels, const Histogram::Snapshot& s);

}  // namespace vdb
