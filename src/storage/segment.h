#pragma once

// An immutable on-disk segment: seg-<id>/
//   vectors.f32       4 KiB header + padded rows (data starts at byte 4096: page aligned)
//   ids.u64           external id of each row
//   meta.bin          encoded metadata of each row
//   index.<type>      this segment's index (hnsw, hnsw_sq8, pq, ivf_pq, flat)
//   deletes-<n>.roar  rows deleted as of the last checkpoint
//
// vectors.f32 header: magic VDBSEG01, u32 version, u32 metric, u64 rows, u32 dim,
// u32 padded dim, u64 data offset, u64 data length, u32 CRC32C of the data, u64 min LSN,
// u64 max LSN, ..., u32 CRC32C of the header at byte 4092.

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "index/index.h"
#include "meta/columns.h"
#include "storage/env.h"
#include "storage/manifest.h"
#include "util/delete_bits.h"
#include "vdb/collection.h"

namespace vdb {

class ThreadPool;

constexpr size_t kSegmentHeaderSize = 4096;
constexpr std::string_view kSegmentMagic = "VDBSEG01";
constexpr std::string_view kIdsMagic = "VDBIDS01";
constexpr std::string_view kMetaMagic = "VDBMETA1";
constexpr std::string_view kDeletesMagic = "VDBDEL01";

std::string IndexFileName(IndexType t);  // index.hnsw, ...
std::string DeletesFileName(uint64_t number);

class Segment {
 public:
  // Maps the vectors, loads ids, metadata columns, the index and the delete set.
  // `deletes` (optional) is shared with the frozen mutable segment this was flushed from,
  // so deletes that arrive during the flush are never lost.
  // A freshly written segment passes its still-valid mapping and built index so nothing
  // is read back (a rename does not invalidate a mapping).
  static StatusOr<std::shared_ptr<Segment>> Open(Env* env, const std::string& dir,
                                                 const SegmentMeta& meta,
                                                 const CollectionOptions& opts, bool verify_crc,
                                                 std::shared_ptr<DeleteBits> deletes = nullptr,
                                                 std::unique_ptr<MappedRegion> map = nullptr,
                                                 std::unique_ptr<Index> index = nullptr);
  ~Segment();
  Segment(const Segment&) = delete;
  Segment& operator=(const Segment&) = delete;

  uint32_t id() const { return meta_.id; }
  size_t rows() const { return view_.rows; }
  uint64_t lsn() const { return lsn_.load(); }
  void set_lsn(uint64_t l) { lsn_.store(l); }
  const std::string& dir() const { return dir_; }
  const VectorView& vectors() const { return view_; }
  const float* row(InternalId r) const { return view_.row(r); }
  VectorId external_id(InternalId r) const { return ids_[r]; }
  const Index& index() const { return *index_; }
  const SegmentColumns& columns() const { return columns_; }
  DeleteBits& deletes() const { return *deletes_; }
  const std::shared_ptr<DeleteBits>& deletes_ptr() const { return deletes_; }
  size_t live_rows() const { return rows() - std::min(rows(), deletes_->count()); }

  // Deleted rows as a roaring bitmap, cached until the delete count changes.
  std::shared_ptr<const roaring::Roaring> deleted_bitmap() const;

  // Delete count last written to deletes-<n>.roar (and recorded in the manifest).
  size_t persisted_deletes() const { return persisted_deletes_.load(); }
  void set_persisted_deletes(size_t n) { persisted_deletes_.store(n); }
  const std::string& deletes_file() const { return deletes_file_; }
  void set_deletes_file(std::string f) { deletes_file_ = std::move(f); }

  // Once obsolete, the directory is removed when the last reader lets go.
  void mark_obsolete() { obsolete_.store(true); }

  MemoryStats memory() const;
  uint64_t disk_bytes() const;

 private:
  Segment() = default;

  Env* env_ = nullptr;
  std::string dir_;
  SegmentMeta meta_;
  std::atomic<uint64_t> lsn_{0};
  std::unique_ptr<MappedRegion> map_;
  VectorView view_;
  std::vector<VectorId> ids_;
  SegmentColumns columns_;
  std::unique_ptr<Index> index_;
  std::shared_ptr<DeleteBits> deletes_;
  std::atomic<size_t> persisted_deletes_{0};
  std::string deletes_file_;
  std::atomic<bool> obsolete_{false};

  mutable std::mutex cache_mu_;
  mutable std::shared_ptr<const roaring::Roaring> deleted_cache_;
  mutable size_t deleted_cache_count_ = ~size_t{0};
};

// Rows to write into a new segment.
struct SegmentSource {
  size_t rows = 0;
  std::function<const float*(size_t)> row;  // padded row
  std::function<VectorId(size_t)> id;
  std::function<void(size_t, std::string*)> meta;  // appends the encoded MetaRow
};

struct SegmentWriteOptions {
  uint32_t id = 0;
  uint64_t lsn = 0;
  const CollectionOptions* options = nullptr;
  ThreadPool* pool = nullptr;
  // Merge into the largest segment: extend this index (whose rows are the first rows of
  // the source) instead of building one from scratch.
  const Index* extend_from = nullptr;
};

// Flush steps 2-4: write the files into seg-<id>.tmp/, build and write the index, fsync
// every file and the directory, rename to seg-<id>/ and fsync segments/. Then opens it.
StatusOr<std::shared_ptr<Segment>> WriteSegment(Env* env, const std::string& segments_dir,
                                                const SegmentSource& src,
                                                const SegmentWriteOptions& wo,
                                                std::shared_ptr<DeleteBits> deletes,
                                                SegmentMeta* meta_out);

// Writes deletes-<number>.roar into a segment directory (fsynced, directory too).
Status WriteDeletesFile(Env* env, const std::string& seg_dir, uint64_t number,
                        const roaring::Roaring& deleted, std::string* name, FileInfo* info);

}  // namespace vdb
