#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "vdb/types.h"

namespace vdb {

// Visited set without clearing: one uint16 tag per node plus the current tag. Each query
// bumps the tag, which turns an O(N) reset into O(1); when the tag wraps to 0 the array
// is cleared once. Also holds the per-search heaps so they are reused between queries.
class SearchContext {
 public:
  explicit SearchContext(size_t n) : tags_(n, 0) {}

  void reset() {
    if (++cur_ == 0) {  // wrapped: stale tags could alias the new value
      std::fill(tags_.begin(), tags_.end(), 0);
      cur_ = 1;
    }
    results.clear();
    candidates.clear();
  }
  bool seen(uint32_t i) const { return tags_[i] == cur_; }
  void mark(uint32_t i) { tags_[i] = cur_; }
  size_t capacity() const { return tags_.size(); }
  uint16_t tag() const { return cur_; }

  std::vector<Neighbor> results;     // max-heap: worst result on top
  std::vector<Neighbor> candidates;  // min-heap: closest candidate on top
  std::vector<uint32_t> links;       // copy of a neighbor list (locked reads during build)
  std::vector<float> scratch;        // decoded vectors (quantized spaces)

 private:
  std::vector<uint16_t> tags_;
  uint16_t cur_ = 0;
};

// Hands one context to each concurrent search.
class SearchContextPool {
 public:
  explicit SearchContextPool(size_t n) : n_(n) {}

  class Handle {
   public:
    Handle(SearchContextPool* pool, std::unique_ptr<SearchContext> ctx)
        : pool_(pool), ctx_(std::move(ctx)) {}
    ~Handle() { pool_->put(std::move(ctx_)); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    SearchContext* operator->() { return ctx_.get(); }
    SearchContext& operator*() { return *ctx_; }

   private:
    SearchContextPool* pool_;
    std::unique_ptr<SearchContext> ctx_;
  };

  Handle get() {
    std::unique_ptr<SearchContext> ctx;
    {
      std::lock_guard<std::mutex> g(mu_);
      if (!free_.empty()) {
        ctx = std::move(free_.back());
        free_.pop_back();
      }
    }
    if (!ctx) ctx = std::make_unique<SearchContext>(n_);
    ctx->reset();
    return Handle(this, std::move(ctx));
  }

  size_t memory_bytes() const {
    std::lock_guard<std::mutex> g(mu_);
    return free_.size() * n_ * sizeof(uint16_t);
  }

 private:
  void put(std::unique_ptr<SearchContext> ctx) {
    std::lock_guard<std::mutex> g(mu_);
    free_.push_back(std::move(ctx));
  }

  size_t n_;
  mutable std::mutex mu_;
  std::vector<std::unique_ptr<SearchContext>> free_;
};

}  // namespace vdb
