#pragma once

// The in-memory, append-only segment that takes new writes. Rows live in fixed-size
// chunks behind a fixed directory, so they never move and readers need no lock. A row is
// published only after it is fully written:
//   writer: write row; count_.store(n, release)
//   reader: n = count_.load(acquire); read only rows below n
// It is searched by brute force and flushed once it holds flush_bytes of vectors.

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "distance/distance.h"
#include "meta/filter.h"
#include "meta/schema.h"
#include "util/aligned_buffer.h"
#include "util/delete_bits.h"
#include "vdb/status.h"
#include "vdb/types.h"

namespace vdb {

class MutableSegment {
 public:
  static constexpr size_t kChunkRows = 1024;
  static constexpr size_t kDefaultMaxRows = size_t{1} << 26;

  MutableSegment(uint32_t id, size_t dim, size_t max_rows = kDefaultMaxRows);
  ~MutableSegment();
  MutableSegment(const MutableSegment&) = delete;
  MutableSegment& operator=(const MutableSegment&) = delete;

  uint32_t id() const { return id_; }
  size_t dim() const { return dim_; }
  size_t padded_dim() const { return padded_; }

  // Single writer (the group-commit leader). `vec` holds padded_dim floats.
  StatusOr<InternalId> Append(VectorId ext, const float* vec, MetaRow meta, uint64_t lsn);

  size_t size() const { return count_.load(std::memory_order_acquire); }
  const float* row(InternalId r) const { return chunk(r)->vecs.floats() + (r % kChunkRows) * padded_; }
  VectorId external_id(InternalId r) const { return chunk(r)->ids[r % kChunkRows]; }
  const MetaRow& meta(InternalId r) const { return chunk(r)->meta[r % kChunkRows]; }
  uint64_t row_lsn(InternalId r) const { return chunk(r)->lsns[r % kChunkRows]; }
  DeleteBits& deletes() const { return *deletes_; }
  const std::shared_ptr<DeleteBits>& deletes_ptr() const { return deletes_; }
  size_t vector_bytes() const { return size() * padded_ * sizeof(float); }
  uint64_t max_lsn() const { return max_lsn_.load(std::memory_order_acquire); }
  // Steady-clock time (us) of the first append, 0 while empty: drives the age-limit flush.
  uint64_t first_append_us() const { return first_append_us_.load(std::memory_order_acquire); }
  size_t live_rows() const { return size() - std::min(size(), deletes_->count()); }

  void freeze() { frozen_.store(true); }
  bool frozen() const { return frozen_.load(); }

  // Exact top-k over the published rows, skipping deleted rows and rows failing `filter`.
  void Search(const float* q, size_t k, DistFn fn, const FilterNode* filter,
              std::vector<Neighbor>& out, uint64_t* distance_computations,
              size_t* matches = nullptr) const;

  size_t memory_bytes() const;

 private:
  struct Chunk {
    explicit Chunk(size_t padded) : vecs(kChunkRows * padded * sizeof(float)) {}
    AlignedBuffer vecs;
    VectorId ids[kChunkRows] = {};
    uint64_t lsns[kChunkRows] = {};
    MetaRow meta[kChunkRows];
  };

  const Chunk* chunk(InternalId r) const {
    return chunks_[r / kChunkRows].load(std::memory_order_acquire);
  }

  uint32_t id_;
  size_t dim_;
  size_t padded_;
  size_t max_chunks_;
  std::unique_ptr<std::atomic<Chunk*>[]> chunks_;
  std::atomic<size_t> count_{0};
  std::atomic<uint64_t> max_lsn_{0};
  std::atomic<uint64_t> first_append_us_{0};
  std::atomic<bool> frozen_{false};
  std::shared_ptr<DeleteBits> deletes_;
};

}  // namespace vdb
