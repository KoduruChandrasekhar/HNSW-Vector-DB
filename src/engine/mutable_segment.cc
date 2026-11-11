#include "engine/mutable_segment.h"

#include <cstring>

#include "util/timer.h"
#include "util/topk.h"

namespace vdb {

MutableSegment::MutableSegment(uint32_t id, size_t dim, size_t max_rows)
    : id_(id),
      dim_(dim),
      padded_(PaddedDim(dim)),
      max_chunks_((max_rows + kChunkRows - 1) / kChunkRows),
      chunks_(std::make_unique<std::atomic<Chunk*>[]>(max_chunks_)),
      deletes_(std::make_shared<DeleteBits>(max_chunks_ * kChunkRows)) {
  for (size_t i = 0; i < max_chunks_; ++i) chunks_[i].store(nullptr);
}

MutableSegment::~MutableSegment() {
  for (size_t i = 0; i < max_chunks_; ++i) delete chunks_[i].load();
}

StatusOr<InternalId> MutableSegment::Append(VectorId ext, const float* vec, MetaRow meta,
                                            uint64_t lsn) {
  const size_t n = count_.load(std::memory_order_relaxed);
  if (n >= max_chunks_ * kChunkRows) return Status::ResourceExhausted("mutable segment is full");
  const size_t ci = n / kChunkRows;
  Chunk* c = chunks_[ci].load(std::memory_order_relaxed);
  if (c == nullptr) {
    c = new Chunk(padded_);
    deletes_->ensure((ci + 1) * kChunkRows);
    chunks_[ci].store(c, std::memory_order_release);
  }
  const size_t slot = n % kChunkRows;
  std::memcpy(c->vecs.floats() + slot * padded_, vec, padded_ * sizeof(float));
  c->ids[slot] = ext;
  c->lsns[slot] = lsn;
  c->meta[slot] = std::move(meta);
  if (lsn > max_lsn_.load(std::memory_order_relaxed)) max_lsn_.store(lsn, std::memory_order_release);
  if (n == 0) first_append_us_.store(NowMicros(), std::memory_order_release);
  count_.store(n + 1, std::memory_order_release);  // publish the fully written row
  return static_cast<InternalId>(n);
}

void MutableSegment::Search(const float* q, size_t k, DistFn fn, const FilterNode* filter,
                            std::vector<Neighbor>& out, uint64_t* distance_computations,
                            size_t* matches) const {
  const size_t n = size();
  TopK top(k);
  uint64_t dc = 0;
  size_t m = 0;
  for (size_t r = 0; r < n; ++r) {
    auto id = static_cast<InternalId>(r);
    if (deletes_->test(r)) continue;
    if (filter != nullptr && !filter->Matches(meta(id))) continue;
    ++m;
    top.push(fn(q, row(id), padded_), id);
    ++dc;
  }
  top.sorted_into(&out);
  if (distance_computations != nullptr) *distance_computations += dc;
  if (matches != nullptr) *matches = m;
}

size_t MutableSegment::memory_bytes() const {
  size_t chunks = (size() + kChunkRows - 1) / kChunkRows;
  return chunks * (kChunkRows * padded_ * sizeof(float) + sizeof(Chunk)) +
         max_chunks_ * sizeof(void*) + deletes_->memory_bytes();
}

}  // namespace vdb
