#pragma once

// Compaction policy. Every segment is its own index and every query searches all of
// them, so query cost grows with the segment count. Three rules keep a settled
// collection at one or two segments:
//   1. delete-driven: rebuild a segment once deleted / rows passes delete_compaction_ratio
//   2. size-tiered: tiers of base, base*fanout, base*fanout^2 rows; when a tier holds
//      `fanout` segments, merge them into one segment of the next tier
//   3. merge into the largest: once the other segments together pass
//      merge_into_largest_ratio of the largest, insert their rows into a copy of the
//      largest segment's graph (graph work proportional to the small segments)
// plus an explicit merge to a target segment count (POST .../compact?target_segments=1).

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vdb/options.h"

namespace vdb {

struct SegmentSummary {
  uint32_t id = 0;
  size_t rows = 0;
  size_t live = 0;
};

struct CompactionTask {
  std::vector<uint32_t> inputs;     // segment ids; with merge_into_largest, the largest first
  bool merge_into_largest = false;  // extend the first input's index instead of rebuilding
  std::string reason;
};

size_t TierOf(size_t live_rows, size_t base_rows, size_t fanout);

std::optional<CompactionTask> PickCompaction(const std::vector<SegmentSummary>& segs,
                                             const CollectionOptions& o,
                                             size_t target_segments, bool extend_supported);

}  // namespace vdb
