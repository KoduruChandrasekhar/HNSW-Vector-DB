#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "vdb/types.h"

namespace vdb {

// Keeps the k smallest neighbors seen so far in a max-heap of size k.
class TopK {
 public:
  explicit TopK(size_t k) : k_(k) { heap_.reserve(k + 1); }

  void push(float dist, InternalId id) {
    if (k_ == 0) return;
    Neighbor n{dist, id};
    if (heap_.size() < k_) {
      heap_.push_back(n);
      std::push_heap(heap_.begin(), heap_.end());
    } else if (n < heap_.front()) {
      std::pop_heap(heap_.begin(), heap_.end());
      heap_.back() = n;
      std::push_heap(heap_.begin(), heap_.end());
    }
  }

  bool full() const { return heap_.size() >= k_; }
  size_t size() const { return heap_.size(); }
  size_t capacity() const { return k_; }
  // Distance a candidate must beat to enter; +inf until the heap is full.
  float threshold() const {
    return full() && k_ > 0 ? heap_.front().dist : std::numeric_limits<float>::infinity();
  }

  // Closest first.
  std::vector<Neighbor> sorted() const {
    std::vector<Neighbor> out = heap_;
    std::sort(out.begin(), out.end());
    return out;
  }
  void sorted_into(std::vector<Neighbor>* out) const {
    out->assign(heap_.begin(), heap_.end());
    std::sort(out->begin(), out->end());
  }
  const std::vector<Neighbor>& raw() const { return heap_; }
  void clear() { heap_.clear(); }

 private:
  size_t k_;
  std::vector<Neighbor> heap_;
};

// Merges several closest-first lists into one closest-first top-k list.
inline std::vector<Neighbor> MergeTopK(const std::vector<std::vector<Neighbor>>& lists, size_t k) {
  TopK top(k);
  for (const auto& l : lists)
    for (const auto& n : l) top.push(n.dist, n.id);
  return top.sorted();
}

}  // namespace vdb
