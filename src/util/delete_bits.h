#pragma once

// Live deletes: one bit per row in atomic 64-bit words, set with fetch_or, so readers
// need no lock. Words live in blocks of 65536 rows allocated on demand behind a fixed
// directory, so the mutable segment can grow without moving anything readers look at.
// (CRoaring bitmaps are not safe for concurrent reads and writes; they are used only
// for the persisted copy, deletes-<n>.roar.)

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

#include "roaring.hh"

namespace vdb {

class DeleteBits {
 public:
  static constexpr size_t kBlockShift = 16;
  static constexpr size_t kBlockBits = size_t{1} << kBlockShift;
  static constexpr size_t kBlockWords = kBlockBits / 64;

  explicit DeleteBits(size_t max_rows)
      : nblocks_((max_rows + kBlockBits - 1) / kBlockBits),
        blocks_(std::make_unique<std::atomic<std::atomic<uint64_t>*>[]>(nblocks_)) {
    for (size_t i = 0; i < nblocks_; ++i) blocks_[i].store(nullptr);
  }
  ~DeleteBits() {
    for (size_t i = 0; i < nblocks_; ++i) delete[] blocks_[i].load();
  }
  DeleteBits(const DeleteBits&) = delete;
  DeleteBits& operator=(const DeleteBits&) = delete;

  size_t capacity() const { return nblocks_ * kBlockBits; }

  // Allocates storage for rows [0, rows). Call before those rows become visible.
  void ensure(size_t rows) {
    size_t need = (rows + kBlockBits - 1) / kBlockBits;
    if (need > nblocks_) need = nblocks_;
    std::lock_guard<std::mutex> g(mu_);
    for (size_t b = 0; b < need; ++b) {
      if (blocks_[b].load(std::memory_order_relaxed) != nullptr) continue;
      auto* p = new std::atomic<uint64_t>[kBlockWords];
      for (size_t w = 0; w < kBlockWords; ++w) p[w].store(0, std::memory_order_relaxed);
      blocks_[b].store(p, std::memory_order_release);
    }
  }

  bool test(size_t i) const {
    const std::atomic<uint64_t>* p = block(i);
    if (p == nullptr) return false;
    return (p[(i >> 6) & (kBlockWords - 1)].load(std::memory_order_acquire) >> (i & 63)) & 1u;
  }

  // Returns true if the bit was not set before.
  bool set(size_t i) {
    std::atomic<uint64_t>* p = block(i);
    if (p == nullptr) {
      ensure(i + 1);
      p = block(i);
      if (p == nullptr) return false;  // beyond capacity
    }
    const uint64_t bit = uint64_t{1} << (i & 63);
    uint64_t old = p[(i >> 6) & (kBlockWords - 1)].fetch_or(bit, std::memory_order_acq_rel);
    if (old & bit) return false;
    count_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  size_t count() const { return count_.load(std::memory_order_relaxed); }

  roaring::Roaring to_roaring(size_t rows) const {
    roaring::Roaring r;
    for (size_t b = 0; b * kBlockBits < rows && b < nblocks_; ++b) {
      const std::atomic<uint64_t>* p = blocks_[b].load(std::memory_order_acquire);
      if (p == nullptr) continue;
      for (size_t w = 0; w < kBlockWords; ++w) {
        uint64_t bits = p[w].load(std::memory_order_acquire);
        while (bits != 0) {
          size_t row = b * kBlockBits + w * 64 + static_cast<size_t>(__builtin_ctzll(bits));
          if (row < rows) r.add(static_cast<uint32_t>(row));
          bits &= bits - 1;
        }
      }
    }
    return r;
  }

  void set_from(const roaring::Roaring& r) {
    if (r.isEmpty()) return;
    ensure(static_cast<size_t>(r.maximum()) + 1);
    for (uint32_t v : r) set(v);
  }

  size_t memory_bytes() const {
    size_t b = nblocks_ * sizeof(void*);
    for (size_t i = 0; i < nblocks_; ++i)
      if (blocks_[i].load(std::memory_order_relaxed) != nullptr) b += kBlockWords * 8;
    return b;
  }

 private:
  std::atomic<uint64_t>* block(size_t i) const {
    size_t b = i >> kBlockShift;
    if (b >= nblocks_) return nullptr;
    return blocks_[b].load(std::memory_order_acquire);
  }

  size_t nblocks_;
  std::unique_ptr<std::atomic<std::atomic<uint64_t>*>[]> blocks_;
  std::mutex mu_;
  std::atomic<size_t> count_{0};
};

}  // namespace vdb
