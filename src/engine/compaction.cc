#include "engine/compaction.h"

#include <algorithm>
#include <cmath>
#include <map>

namespace vdb {
namespace {

double DeletedFraction(const SegmentSummary& s) {
  return s.rows == 0 ? 0.0 : static_cast<double>(s.rows - s.live) / static_cast<double>(s.rows);
}

}  // namespace

size_t TierOf(size_t live_rows, size_t base_rows, size_t fanout) {
  if (base_rows == 0 || live_rows < base_rows) return 0;
  double t = std::log(static_cast<double>(live_rows) / static_cast<double>(base_rows)) /
             std::log(static_cast<double>(fanout));
  return 1 + static_cast<size_t>(std::floor(t + 1e-9));
}

std::optional<CompactionTask> PickCompaction(const std::vector<SegmentSummary>& segs,
                                             const CollectionOptions& o,
                                             size_t target_segments, bool extend_supported) {
  // Fully deleted segments are simply dropped.
  {
    CompactionTask t;
    for (const auto& s : segs)
      if (s.live == 0) t.inputs.push_back(s.id);
    if (!t.inputs.empty()) {
      t.reason = "drop_empty";
      return t;
    }
  }
  if (segs.size() < 1) return std::nullopt;

  std::vector<SegmentSummary> by_size = segs;
  std::sort(by_size.begin(), by_size.end(), [](const SegmentSummary& a, const SegmentSummary& b) {
    return a.live > b.live || (a.live == b.live && a.id < b.id);
  });
  const SegmentSummary& largest = by_size.front();

  // Explicit target (e.g. 1 after a big load).
  if (target_segments > 0 && segs.size() > target_segments) {
    CompactionTask t;
    t.reason = "explicit";
    if (target_segments == 1) {
      for (const auto& s : by_size) t.inputs.push_back(s.id);
      t.merge_into_largest = extend_supported && DeletedFraction(largest) < o.delete_compaction_ratio;
    } else {
      size_t n = segs.size() - target_segments + 1;  // merge the n smallest into one
      for (size_t i = by_size.size() - n; i < by_size.size(); ++i) t.inputs.push_back(by_size[i].id);
    }
    return t;
  }

  // 1. Delete-driven rebuild.
  for (const auto& s : by_size) {
    if (DeletedFraction(s) > o.delete_compaction_ratio) {
      CompactionTask t;
      t.inputs = {s.id};
      t.reason = "deletes";
      return t;
    }
  }

  if (segs.size() < 2) return std::nullopt;

  // 2. Size tiers.
  std::map<size_t, std::vector<SegmentSummary>> tiers;
  for (const auto& s : by_size) tiers[TierOf(s.live, o.tier_base_rows, o.tier_fanout)].push_back(s);
  for (auto& [tier, members] : tiers) {
    if (members.size() < o.tier_fanout) continue;
    std::sort(members.begin(), members.end(), [](const SegmentSummary& a, const SegmentSummary& b) {
      return a.live < b.live || (a.live == b.live && a.id < b.id);
    });
    CompactionTask t;
    for (size_t i = 0; i < o.tier_fanout; ++i) t.inputs.push_back(members[i].id);
    t.reason = "tier_" + std::to_string(tier);
    return t;
  }

  // 3. Merge everything else into the largest.
  size_t others = 0;
  for (size_t i = 1; i < by_size.size(); ++i) others += by_size[i].live;
  if (static_cast<double>(others) > o.merge_into_largest_ratio * static_cast<double>(largest.live)) {
    CompactionTask t;
    for (const auto& s : by_size) t.inputs.push_back(s.id);
    t.merge_into_largest = extend_supported && DeletedFraction(largest) < o.delete_compaction_ratio;
    t.reason = "merge_into_largest";
    return t;
  }
  return std::nullopt;
}

}  // namespace vdb
