#include "util/metrics.h"

#include <algorithm>
#include <cstdio>

namespace vdb {

Histogram::Histogram(std::vector<double> upper_bounds) : bounds_(std::move(upper_bounds)) {
  std::sort(bounds_.begin(), bounds_.end());
  counts_ = std::make_unique<std::atomic<uint64_t>[]>(bounds_.size() + 1);
  for (size_t i = 0; i <= bounds_.size(); ++i) counts_[i].store(0);
}

void Histogram::observe(double v) {
  size_t i = static_cast<size_t>(std::lower_bound(bounds_.begin(), bounds_.end(), v) -
                                 bounds_.begin());
  counts_[i].fetch_add(1, std::memory_order_relaxed);
  count_.fetch_add(1, std::memory_order_relaxed);
  double cur = sum_.load(std::memory_order_relaxed);
  while (!sum_.compare_exchange_weak(cur, cur + v, std::memory_order_relaxed)) {
  }
}

Histogram::Snapshot Histogram::snapshot() const {
  Snapshot s;
  s.bounds = bounds_;
  s.counts.resize(bounds_.size() + 1);
  for (size_t i = 0; i <= bounds_.size(); ++i) s.counts[i] = counts_[i].load();
  s.count = count_.load();
  s.sum = sum_.load();
  return s;
}

std::vector<double> ExponentialBuckets(double start, double factor, size_t count) {
  std::vector<double> b;
  b.reserve(count);
  double v = start;
  for (size_t i = 0; i < count; ++i, v *= factor) b.push_back(v);
  return b;
}

EngineMetrics& GlobalMetrics() {
  static EngineMetrics* m = new EngineMetrics();  // never destroyed: used at exit
  return *m;
}

void AppendPrometheusHistogram(std::string* out, const std::string& name,
                               const std::string& labels, const Histogram::Snapshot& s) {
  char buf[64];
  uint64_t cum = 0;
  const std::string sep = labels.empty() ? "" : ",";
  for (size_t i = 0; i < s.bounds.size(); ++i) {
    cum += s.counts[i];
    std::snprintf(buf, sizeof(buf), "%.9g", s.bounds[i]);
    *out += name + "_bucket{" + labels + sep + "le=\"" + buf + "\"} " + std::to_string(cum) + "\n";
  }
  cum += s.counts.back();
  *out += name + "_bucket{" + labels + sep + "le=\"+Inf\"} " + std::to_string(cum) + "\n";
  std::snprintf(buf, sizeof(buf), "%.9g", s.sum);
  std::string lbl = labels.empty() ? "" : "{" + labels + "}";
  *out += name + "_sum" + lbl + " " + buf + "\n";
  *out += name + "_count" + lbl + " " + std::to_string(s.count) + "\n";
}

}  // namespace vdb
