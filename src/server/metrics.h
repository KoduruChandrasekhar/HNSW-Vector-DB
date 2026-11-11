#pragma once

// Prometheus metrics for the server, written by hand in the text exposition format.

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "util/metrics.h"

namespace vdb {

class CollectionManager;

class ServerMetrics {
 public:
  void Record(const std::string& route, int code, double seconds);
  std::string Render(const CollectionManager& mgr) const;

 private:
  mutable std::mutex mu_;
  std::map<std::pair<std::string, int>, uint64_t> requests_;
  std::map<std::string, std::unique_ptr<Histogram>> durations_;
};

}  // namespace vdb
