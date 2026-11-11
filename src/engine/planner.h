#pragma once

// Per-segment search planning for filtered queries.
//
//   pre-filter + exact   build the bitmap, exact distances for matching rows only.
//                        Right when few rows match; a full scan when many do.
//   filtered graph       index search that walks through non-matching nodes but never
//                        returns them, with ef raised as selectivity drops.
//                        Right when many rows match; recall collapses when few do.
//   post-filter          search k * alpha without the filter, drop non-matching rows.
//                        Right when nearly everything matches; returns fewer than k
//                        when few rows match.
//
// The planner evaluates the filter into a roaring bitmap per segment (minus deleted rows),
// then compares costs in distance computations: exact costs `matches`; a filtered graph
// search costs about 10 * ef / selectivity (fitted on SIFT). It picks exact when that is
// cheaper or matches <= prefilter_threshold, else filtered graph search. Mutable segments
// always scan.

#include <vector>

#include "distance/distance.h"
#include "engine/mutable_segment.h"
#include "meta/filter.h"
#include "storage/segment.h"
#include "vdb/collection.h"

namespace vdb {

struct SegmentHits {
  std::vector<Neighbor> hits;
  SegmentExplain explain;
};

void SearchDiskSegment(const Segment& seg, const float* q, size_t k, const SearchParams& p,
                       const FilterExpr* filter, DistFn fn, SegmentHits* out);

void SearchMutable(const MutableSegment& seg, const float* q, size_t k, const FilterExpr* filter,
                   DistFn fn, const char* kind, SegmentHits* out);

}  // namespace vdb
