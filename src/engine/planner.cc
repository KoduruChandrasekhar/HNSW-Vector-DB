#include "engine/planner.h"

#include <algorithm>
#include <cmath>

#include "util/timer.h"
#include "util/topk.h"

namespace vdb {
namespace {

class DeleteFilter final : public RowFilter {
 public:
  explicit DeleteFilter(const DeleteBits* d) : d_(d) {}
  bool allowed(InternalId id) const override { return !d_->test(id); }

 private:
  const DeleteBits* d_;
};

// Allowed rows: the filter bitmap, which already excludes deleted rows.
class BitmapFilter final : public RowFilter {
 public:
  explicit BitmapFilter(const roaring::Roaring* bm) : bm_(bm) {}
  bool allowed(InternalId id) const override { return bm_->contains(id); }

 private:
  const roaring::Roaring* bm_;
};

void ExactOverBitmap(const Segment& seg, const float* q, size_t k, const roaring::Roaring& bm,
                     DistFn fn, std::vector<Neighbor>& out, uint64_t* dc) {
  TopK top(k);
  const size_t pd = seg.vectors().padded_dim;
  for (uint32_t r : bm) top.push(fn(q, seg.row(r), pd), r);
  *dc += bm.cardinality();
  top.sorted_into(&out);
}

}  // namespace

void SearchDiskSegment(const Segment& seg, const float* q, size_t k, const SearchParams& p,
                       const FilterExpr* filter, DistFn fn, SegmentHits* out) {
  Timer t;
  SegmentExplain& ex = out->explain;
  ex.segment_id = seg.id();
  ex.kind = "disk";
  ex.index = IndexTypeName(seg.index().type());
  ex.rows = seg.rows();
  const size_t deleted = seg.deletes().count();
  ex.live_rows = seg.live_rows();
  IndexSearchStats st;

  if (filter == nullptr) {
    ex.strategy = "index";
    ex.matches = ex.live_rows;
    if (deleted == 0) {
      seg.index().search(q, k, p, nullptr, out->hits, &st);
    } else {
      DeleteFilter df(&seg.deletes());
      seg.index().search(q, k, p, &df, out->hits, &st);
    }
  } else {
    roaring::Roaring bm = seg.columns().Evaluate(filter->root());
    if (deleted > 0) bm -= *seg.deleted_bitmap();
    const size_t matches = bm.cardinality();
    ex.matches = matches;
    ex.selectivity = ex.live_rows == 0 ? 0.0 : static_cast<double>(matches) / static_cast<double>(ex.live_rows);
    FilterStrategy strat = p.strategy;
    if (strat == FilterStrategy::Auto) {
      // Cost model fitted on SIFT (bench/results/filter_*.json): a filtered graph search
      // keeps walking until it holds ef matching rows, so it costs about
      // kGraphCost * ef / selectivity distance computations; exact costs `matches`.
      constexpr double kGraphCost = 10.0;
      const double s = std::max(ex.selectivity, 1e-9);
      const double graph_cost = kGraphCost * static_cast<double>(std::max(p.ef, k)) / s;
      strat = (matches <= p.prefilter_threshold || static_cast<double>(matches) <= graph_cost ||
               seg.index().type() == IndexType::Flat)
                  ? FilterStrategy::PreFilter
                  : FilterStrategy::Graph;
    }
    if (matches == 0) {
      ex.strategy = "empty";
      out->hits.clear();
    } else if (strat == FilterStrategy::PreFilter) {
      ex.strategy = "prefilter_exact";
      ExactOverBitmap(seg, q, k, bm, fn, out->hits, &st.distance_computations);
    } else if (strat == FilterStrategy::Graph) {
      SearchParams pp = p;
      if (p.strategy == FilterStrategy::Auto) {
        // HNSW needs no ef boost: it already walks until it has ef matching rows (raising ef
        // by 1/selectivity too, the first rule tried, cost 5x QPS at 10% selectivity for
        // +0.001 recall). IVF only scans the lists it probes, so it does need more of them.
        double s = std::max(ex.selectivity, 1e-9);
        pp.nprobe = std::max(p.nprobe, static_cast<size_t>(std::ceil(static_cast<double>(p.nprobe) / s)));
      }
      ex.strategy = "filtered_index";
      BitmapFilter bf(&bm);
      seg.index().search(q, k, pp, &bf, out->hits, &st);
      // Never return fewer than k when k rows match: an IVF probe or a disconnected
      // region can come up short, and then exact search over the bitmap is the answer.
      if (p.strategy == FilterStrategy::Auto && out->hits.size() < std::min(k, matches)) {
        ex.strategy = "filtered_index+exact_fallback";
        ExactOverBitmap(seg, q, k, bm, fn, out->hits, &st.distance_computations);
      }
    } else {  // post-filter
      ex.strategy = "post_filter";
      size_t kk = std::min(std::max<size_t>(ex.live_rows, 1), k * std::max<size_t>(1, p.post_filter_alpha));
      SearchParams pp = p;
      pp.ef = std::max(p.ef, kk);
      std::vector<Neighbor> wide;
      if (deleted == 0) {
        seg.index().search(q, kk, pp, nullptr, wide, &st);
      } else {
        DeleteFilter df(&seg.deletes());
        seg.index().search(q, kk, pp, &df, wide, &st);
      }
      out->hits.clear();
      for (const Neighbor& n : wide) {
        if (bm.contains(n.id)) out->hits.push_back(n);
        if (out->hits.size() >= k) break;
      }
    }
  }
  ex.ef = st.ef;
  ex.distance_computations = st.distance_computations;
  ex.results = out->hits.size();
  ex.micros = t.micros();
}

void SearchMutable(const MutableSegment& seg, const float* q, size_t k, const FilterExpr* filter,
                   DistFn fn, const char* kind, SegmentHits* out) {
  Timer t;
  SegmentExplain& ex = out->explain;
  ex.segment_id = seg.id();
  ex.kind = kind;
  ex.index = "flat";
  ex.strategy = filter != nullptr ? "scan_filter" : "scan";
  ex.rows = seg.size();
  ex.live_rows = seg.live_rows();
  uint64_t dc = 0;
  size_t matches = 0;
  seg.Search(q, k, fn, filter != nullptr ? &filter->root() : nullptr, out->hits, &dc, &matches);
  ex.matches = matches;
  ex.selectivity = ex.live_rows == 0 ? 0.0 : static_cast<double>(matches) / static_cast<double>(ex.live_rows);
  ex.distance_computations = dc;
  ex.results = out->hits.size();
  ex.micros = t.micros();
}

}  // namespace vdb
