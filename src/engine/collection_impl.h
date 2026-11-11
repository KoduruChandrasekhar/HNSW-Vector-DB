#pragma once

// The engine behind vdb::Collection.
//
// Write path: validate, then hand the request to the group-commit queue. The writer at
// the front of the queue becomes the leader: it assigns LSNs to every waiting write,
// appends them to the log in one write, fsyncs once (sync mode "always"), applies them
// to the mutable segment and the id map, and wakes everyone. A write is acknowledged
// only after its record is durable.
//
// Read path: copy the current snapshot pointer (nanoseconds under a tiny lock), search
// every segment in it, merge the per-segment top-k lists, map rows to external ids. The
// shared_ptrs keep every segment alive until the reader finishes, so flush and
// compaction build new segments off to the side and publish a new snapshot without ever
// blocking a reader.
//
// Lock order: flush_mu_ / compact_mu_  ->  writer_mu_ (leadership)  ->  manifest_mu_
// ->  snap_mu_. Exclusive sections (freeze, compaction and bulk-load commits) run as a
// special writer in the queue, so no write is applied while they run.

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "distance/distance.h"
#include "engine/compaction.h"
#include "engine/id_map.h"
#include "engine/mutable_segment.h"
#include "meta/filter.h"
#include "storage/env.h"
#include "storage/manifest.h"
#include "storage/segment.h"
#include "storage/wal.h"
#include "util/thread_pool.h"
#include "vdb/collection.h"

namespace vdb {

struct Snapshot {
  std::shared_ptr<MutableSegment> active;                // takes writes; lock-free reads
  std::vector<std::shared_ptr<MutableSegment>> frozen;   // being flushed
  std::vector<std::shared_ptr<Segment>> on_disk;
};

class CollectionImpl {
 public:
  static StatusOr<std::unique_ptr<CollectionImpl>> Create(const std::string& path,
                                                          const CollectionOptions& options,
                                                          const OpenOptions& open, Env* env);
  static StatusOr<std::unique_ptr<CollectionImpl>> Open(const std::string& path,
                                                        const OpenOptions& open, Env* env);
  ~CollectionImpl();

  const CollectionOptions& options() const { return options_; }

  Status Upsert(const std::vector<UpsertPoint>& points, uint64_t* last_lsn);
  Status UpsertRows(const VectorId* ids, const float* vectors, size_t n,
                    const std::vector<Metadata>* metadata, uint64_t* last_lsn);
  Status Remove(const VectorId* ids, size_t n, size_t* removed, bool* first_found,
                uint64_t* last_lsn);
  Status BulkLoad(const VectorId* ids, const float* vectors, size_t n,
                  const std::vector<Metadata>* metadata);

  StatusOr<std::unique_ptr<Filter>> ParseFilter(std::string_view json) const;
  Status Search(const SearchRequest& req, std::vector<SearchHit>* out, SearchExplain* explain,
                bool parallel_segments) const;
  Status SearchBatch(const float* queries, size_t nq, size_t k, const SearchParams& params,
                     const Filter* filter, std::vector<std::vector<SearchHit>>* out) const;
  Status Get(VectorId id, std::vector<float>* vector, Metadata* metadata) const;

  Status Flush();
  void RequestFlush();
  Status Compact(size_t target_segments);
  void RequestCompaction(size_t target_segments);
  CollectionStats Stats() const;
  Status Close();

  std::shared_ptr<const Snapshot> snapshot() const {
    std::lock_guard<std::mutex> g(snap_mu_);
    return snap_;
  }
  ThreadPool* pool() const { return pool_.get(); }

 private:
  struct WriteOp {
    enum Kind : uint8_t { kUpsert, kDelete } kind = kUpsert;
    VectorId id = 0;
    std::vector<float> vec;  // padded, normalized for cosine
    MetaRow meta;
    std::string meta_enc;
    uint64_t lsn = 0;
    bool found = false;
  };
  struct Writer {
    std::vector<WriteOp>* ops = nullptr;
    const std::function<Status()>* exclusive = nullptr;
    Status status;
    bool done = false;
    std::condition_variable cv;
  };

  CollectionImpl(Env* env, std::string dir);

  Status Recover();
  void StartBackground();
  void FlushLoop();
  void CompactLoop();
  void SyncLoop();

  Status PrepareUpsert(VectorId id, const float* vec, const Metadata* md, WriteOp* op) const;
  Status PrepareQuery(const float* v, std::vector<float>* q) const;
  Status Write(std::vector<WriteOp>* ops, uint64_t* last_lsn);
  Status Submit(Writer* w);
  Status RunExclusive(const std::function<Status()>& fn);
  void ApplyOp(WriteOp& op, bool replay, const Snapshot& snap);
  void MarkDeleted(const Snapshot& snap, RowLoc loc);

  Status FlushOnce();
  Status CompactOnce(const CompactionTask& task);
  Status CommitManifestLocked(Manifest m, std::vector<std::string>* garbage);
  void ModifySnapshot(const std::function<void(Snapshot&)>& fn);
  uint64_t NextFileNumber() { return next_file_number_.fetch_add(1); }
  void SetFailed(const Status& s);
  Status failure() const;
  void RemoveObsoleteLogs(uint64_t below);

  Env* env_;
  const std::string dir_;
  const std::string wal_dir_;
  const std::string seg_dir_;
  CollectionOptions options_;
  OpenOptions open_;
  size_t padded_ = 0;
  DistFn dist_ = nullptr;
  std::unique_ptr<ThreadPool> pool_;

  mutable std::mutex snap_mu_;
  std::shared_ptr<const Snapshot> snap_;

  std::mutex writer_mu_;
  std::deque<Writer*> writers_;
  uint64_t last_lsn_ = 0;  // under writer_mu_ / leadership
  std::atomic<uint64_t> last_lsn_pub_{0};
  std::string group_buf_;  // the leader's encode buffer
  mutable std::mutex wal_ptr_mu_;
  std::shared_ptr<WalWriter> wal_;

  IdMap id_map_;
  std::unordered_map<uint32_t, uint64_t> replay_seg_lsn_;  // recovery only

  mutable std::mutex manifest_mu_;
  Manifest manifest_;
  std::atomic<uint64_t> next_file_number_{1};

  std::mutex flush_mu_;
  std::mutex compact_mu_;
  std::mutex bg_mu_;
  std::condition_variable bg_cv_;
  bool stop_ = false;
  bool flush_requested_ = false;
  bool compact_requested_ = false;
  size_t compact_target_ = 0;
  std::thread flush_thread_, compact_thread_, sync_thread_;

  std::atomic<bool> failed_{false};
  mutable std::mutex fail_mu_;
  Status failure_;

  std::mutex close_mu_;
  std::atomic<bool> closed_{false};

  std::atomic<uint64_t> last_write_us_{0};
  uint64_t idle_merged_at_ = ~uint64_t{0};  // compaction thread only
  std::atomic<uint64_t> flushes_{0};
  std::atomic<uint64_t> compactions_{0};
  double recovery_seconds_ = 0;
  uint64_t recovery_records_ = 0;
};

}  // namespace vdb
