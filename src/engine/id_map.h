#pragma once

// External id -> (segment, row). Indexes only ever see the 32-bit rows; this map is the
// only place external ids are resolved for writes and point reads. Rebuilt at startup
// from each segment's ids.u64 plus the replayed log.

#include <cstdint>
#include <functional>
#include <optional>
#include <shared_mutex>
#include <unordered_map>

#include "vdb/types.h"

namespace vdb {

struct RowLoc {
  uint32_t segment = 0;
  uint32_t row = 0;
  bool operator==(const RowLoc& o) const { return segment == o.segment && row == o.row; }
};

class IdMap {
 public:
  std::optional<RowLoc> Find(VectorId id) const;
  void Put(VectorId id, RowLoc loc);
  bool Erase(VectorId id);
  // Repoints id only if it still points at `expected` (compaction moving rows).
  bool ReplaceIf(VectorId id, RowLoc expected, RowLoc repl);
  size_t size() const;
  void Reserve(size_t n);
  size_t memory_bytes() const;

 private:
  mutable std::shared_mutex mu_;
  std::unordered_map<VectorId, RowLoc> map_;
};

}  // namespace vdb
