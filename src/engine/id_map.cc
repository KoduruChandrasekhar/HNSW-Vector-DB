#include "engine/id_map.h"

#include <mutex>

namespace vdb {

std::optional<RowLoc> IdMap::Find(VectorId id) const {
  std::shared_lock<std::shared_mutex> g(mu_);
  auto it = map_.find(id);
  if (it == map_.end()) return std::nullopt;
  return it->second;
}

void IdMap::Put(VectorId id, RowLoc loc) {
  std::unique_lock<std::shared_mutex> g(mu_);
  map_[id] = loc;
}

bool IdMap::Erase(VectorId id) {
  std::unique_lock<std::shared_mutex> g(mu_);
  return map_.erase(id) > 0;
}

bool IdMap::ReplaceIf(VectorId id, RowLoc expected, RowLoc repl) {
  std::unique_lock<std::shared_mutex> g(mu_);
  auto it = map_.find(id);
  if (it == map_.end() || !(it->second == expected)) return false;
  it->second = repl;
  return true;
}

size_t IdMap::size() const {
  std::shared_lock<std::shared_mutex> g(mu_);
  return map_.size();
}

void IdMap::Reserve(size_t n) {
  std::unique_lock<std::shared_mutex> g(mu_);
  map_.reserve(n);
}

size_t IdMap::memory_bytes() const {
  std::shared_lock<std::shared_mutex> g(mu_);
  // Node-based map: key + value + next pointer + cached hash per entry, plus buckets.
  return map_.size() * (sizeof(VectorId) + sizeof(RowLoc) + 2 * sizeof(void*)) +
         map_.bucket_count() * sizeof(void*);
}

}  // namespace vdb
