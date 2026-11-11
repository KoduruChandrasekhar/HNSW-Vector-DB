#include "engine/collection_impl.h"

#if defined(__linux__)
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_set>

#include "engine/options_json.h"
#include "engine/planner.h"
#include "storage/file_format.h"
#include "util/dataset.h"
#include "util/metrics.h"
#include "util/timer.h"

namespace vdb {
namespace {

constexpr size_t kMaxGroupBytes = 8u << 20;  // one group commit appends at most ~8 MB
constexpr size_t kOpsPerWriter = 1024;       // big batches are split; batches are not atomic
constexpr size_t kMaxK = 10000;

bool EndsWith(const std::string& s, std::string_view suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

CollectionImpl::CollectionImpl(Env* env, std::string dir)
    : env_(env),
      dir_(std::move(dir)),
      wal_dir_(JoinPath(dir_, "wal")),
      seg_dir_(JoinPath(dir_, "segments")) {}

CollectionImpl::~CollectionImpl() { Close(); }

// ===================================================================== create / open

StatusOr<std::unique_ptr<CollectionImpl>> CollectionImpl::Create(const std::string& path,
                                                                 const CollectionOptions& options,
                                                                 const OpenOptions& open,
                                                                 Env* env) {
  if (env == nullptr) env = Env::Default();
  VDB_RETURN_IF_ERROR(ValidateOptions(options));
  if (env->exists(JoinPath(path, "CURRENT"))) {
    return Status::AlreadyExists("a collection already exists at " + path);
  }
  VDB_RETURN_IF_ERROR(CreateDirs(env, path));
  VDB_RETURN_IF_ERROR(env->create_dir(JoinPath(path, "wal")));
  VDB_RETURN_IF_ERROR(env->create_dir(JoinPath(path, "segments")));
  VDB_RETURN_IF_ERROR(env->sync_dir(path));
  VDB_RETURN_IF_ERROR(env->sync_dir(DirName(path)));
  Manifest m;
  m.number = 1;
  m.options = options;
  m.next_file_number = 2;
  VDB_RETURN_IF_ERROR(CommitManifest(env, path, m));
  return Open(path, open, env);
}

StatusOr<std::unique_ptr<CollectionImpl>> CollectionImpl::Open(const std::string& path,
                                                               const OpenOptions& open, Env* env) {
  if (env == nullptr) env = Env::Default();
  std::unique_ptr<CollectionImpl> c(new CollectionImpl(env, path));
  c->open_ = open;
  VDB_RETURN_IF_ERROR(c->Recover());
  c->StartBackground();
  return c;
}

Status CollectionImpl::Recover() {
  Timer timer;
  // 1. CURRENT -> manifest, CRC checked.
  VDB_RETURN_IF_ERROR(ReadCurrentManifest(env_, dir_, &manifest_));
  options_ = manifest_.options;
  padded_ = PaddedDim(options_.dim);
  dist_ = MetricDistance(options_.metric);
  pool_ = std::make_unique<ThreadPool>(open_.num_threads);
  uint64_t next = manifest_.next_file_number;

  // 2. Delete leftovers: *.tmp, old manifests, segments and delete sets the manifest does
  //    not name (a crash between writing them and committing the manifest).
  std::vector<std::string> names;
  VDB_RETURN_IF_ERROR(env_->list_dir(dir_, &names));
  for (const auto& n : names) {
    uint64_t num = 0;
    if (EndsWith(n, ".tmp")) {
      VDB_RETURN_IF_ERROR(env_->remove_dir_all(JoinPath(dir_, n)));
    } else if (ParseManifestFileName(n, &num)) {
      next = std::max(next, num + 1);
      if (num != manifest_.number) VDB_RETURN_IF_ERROR(env_->remove_file(JoinPath(dir_, n)));
    }
  }
  VDB_RETURN_IF_ERROR(env_->create_dir(seg_dir_));
  VDB_RETURN_IF_ERROR(env_->create_dir(wal_dir_));
  std::unordered_map<uint32_t, const SegmentMeta*> live;
  for (const auto& sm : manifest_.segments) live[sm.id] = &sm;
  VDB_RETURN_IF_ERROR(env_->list_dir(seg_dir_, &names));
  for (const auto& n : names) {
    uint32_t id = 0;
    const std::string p = JoinPath(seg_dir_, n);
    if (EndsWith(n, ".tmp")) {
      VDB_RETURN_IF_ERROR(env_->remove_dir_all(p));
      continue;
    }
    if (!ParseSegmentDirName(n, &id)) continue;
    next = std::max<uint64_t>(next, uint64_t{id} + 1);
    auto it = live.find(id);
    if (it == live.end()) {
      VDB_RETURN_IF_ERROR(env_->remove_dir_all(p));
      continue;
    }
    std::vector<std::string> files;
    VDB_RETURN_IF_ERROR(env_->list_dir(p, &files));
    for (const auto& f : files) {
      bool stray_deletes = f.rfind("deletes-", 0) == 0 && f != it->second->deletes_file;
      if (stray_deletes || EndsWith(f, ".tmp")) VDB_RETURN_IF_ERROR(env_->remove_file(JoinPath(p, f)));
    }
  }

  // 3. Open every segment (mmap, header and size checks, optional CRC).
  std::vector<std::shared_ptr<Segment>> segs(manifest_.segments.size());
  std::vector<Status> st(segs.size());
  pool_->parallel_for(0, segs.size(), [&](size_t i) {
    const SegmentMeta& sm = manifest_.segments[i];
    auto s = Segment::Open(env_, JoinPath(seg_dir_, SegmentDirName(sm.id)), sm, options_,
                           open_.verify_checksums);
    if (s.ok()) {
      segs[i] = std::move(s).value();
    } else {
      st[i] = s.status();
    }
  });
  for (const auto& s : st) VDB_RETURN_IF_ERROR(s);

  // 4. Rebuild the external-id map from the segments' ids, oldest segment first.
  std::vector<size_t> order(segs.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return segs[a]->lsn() < segs[b]->lsn() || (segs[a]->lsn() == segs[b]->lsn() && segs[a]->id() < segs[b]->id());
  });
  size_t total = 0;
  for (const auto& s : segs) total += s->live_rows();
  id_map_.Reserve(total);
  uint64_t max_seg_lsn = 0;
  for (size_t i : order) {
    const Segment& s = *segs[i];
    replay_seg_lsn_[s.id()] = s.lsn();
    max_seg_lsn = std::max(max_seg_lsn, s.lsn());
    for (size_t r = 0; r < s.rows(); ++r) {
      if (!s.deletes().test(r)) id_map_.Put(s.external_id(static_cast<InternalId>(r)), RowLoc{s.id(), static_cast<uint32_t>(r)});
    }
  }

  // 5. Replay the logs into a fresh mutable segment.
  std::vector<uint64_t> logs;
  VDB_RETURN_IF_ERROR(env_->list_dir(wal_dir_, &names));
  for (const auto& n : names) {
    uint64_t num = 0;
    if (!ParseWalFileName(n, &num)) continue;
    next = std::max(next, num + 1);
    if (num < manifest_.wal_log_number) {
      VDB_RETURN_IF_ERROR(env_->remove_file(JoinPath(wal_dir_, n)));
    } else {
      logs.push_back(num);
    }
  }
  std::sort(logs.begin(), logs.end());
  next_file_number_.store(next);

  auto snap = std::make_shared<Snapshot>();
  snap->active = std::make_shared<MutableSegment>(static_cast<uint32_t>(NextFileNumber()), options_.dim);
  snap->on_disk = segs;
  {
    std::lock_guard<std::mutex> g(snap_mu_);
    snap_ = snap;
  }
  last_lsn_ = std::max(manifest_.flushed_lsn, max_seg_lsn);
  uint64_t expected = manifest_.flushed_lsn + 1;
  uint64_t replayed = 0;
  for (size_t li = 0; li < logs.size(); ++li) {
    const std::string path = JoinPath(wal_dir_, WalFileName(logs[li]));
    std::string data;
    VDB_RETURN_IF_ERROR(env_->read_file(path, &data));
    WalParseResult res;
    Status ps = ParseWal(
        data,
        [&](const WalRecord& rec) -> Status {
          if (rec.lsn <= manifest_.flushed_lsn) return Status::OK();
          if (rec.lsn != expected) {
            return Status::Corruption("log sequence gap: expected LSN " + std::to_string(expected) +
                                      ", found " + std::to_string(rec.lsn));
          }
          ++expected;
          WriteOp op;
          op.lsn = rec.lsn;
          if (rec.type == WalRecordType::kUpsert) {
            op.kind = WriteOp::kUpsert;
            uint32_t dim = 0;
            const char* vec = nullptr;
            std::string_view meta;
            if (!DecodeUpsertPayload(rec.payload, &op.id, &dim, &vec, &meta) || dim != options_.dim) {
              return Status::Corruption("bad upsert record at LSN " + std::to_string(rec.lsn));
            }
            op.vec.assign(padded_, 0.f);
            std::memcpy(op.vec.data(), vec, dim * sizeof(float));
            if (!DecodeMetaRow(meta, options_.schema, &op.meta)) {
              return Status::Corruption("bad metadata at LSN " + std::to_string(rec.lsn));
            }
          } else {
            op.kind = WriteOp::kDelete;
            if (!DecodeDeletePayload(rec.payload, &op.id)) {
              return Status::Corruption("bad delete record at LSN " + std::to_string(rec.lsn));
            }
          }
          ApplyOp(op, /*replay=*/true, *snap);
          last_lsn_ = std::max(last_lsn_, rec.lsn);
          ++replayed;
          return Status::OK();
        },
        &res);
    if (!ps.ok()) return Status(ps.code(), path + ": " + ps.message());
    if (res.torn_tail) {
      // A torn write from the crash: only acceptable at the very end of the newest log.
      if (li + 1 != logs.size()) {
        return Status::Corruption(path + ": damaged record in a log that is not the newest");
      }
      VDB_RETURN_IF_ERROR(env_->truncate(path, res.valid_bytes));
    }
  }
  if (failed_.load()) return failure();
  replay_seg_lsn_.clear();

  // 6. A new log for new writes.
  uint64_t num = NextFileNumber();
  VDB_ASSIGN_OR_RETURN(auto w, WalWriter::Create(env_, JoinPath(wal_dir_, WalFileName(num)), num));
  VDB_RETURN_IF_ERROR(env_->sync_dir(wal_dir_));
  wal_ = std::shared_ptr<WalWriter>(std::move(w));
  last_lsn_pub_.store(last_lsn_);
  recovery_seconds_ = timer.seconds();
  recovery_records_ = replayed;
  return Status::OK();
}

void CollectionImpl::StartBackground() {
  last_write_us_.store(NowMicros());
  flush_thread_ = std::thread([this] { FlushLoop(); });
  compact_thread_ = std::thread([this] { CompactLoop(); });
  if (options_.sync == SyncMode::Interval) sync_thread_ = std::thread([this] { SyncLoop(); });
  if (open_.auto_flush && snapshot()->active->vector_bytes() >= options_.flush_bytes) RequestFlush();
  if (open_.auto_compact) RequestCompaction(0);
}

Status CollectionImpl::Close() {
  std::lock_guard<std::mutex> cg(close_mu_);
  if (closed_.exchange(true)) return Status::OK();
  {
    std::lock_guard<std::mutex> g(bg_mu_);
    stop_ = true;
  }
  bg_cv_.notify_all();
  if (flush_thread_.joinable()) flush_thread_.join();
  if (compact_thread_.joinable()) compact_thread_.join();
  if (sync_thread_.joinable()) sync_thread_.join();
  Status s;
  std::lock_guard<std::mutex> g(wal_ptr_mu_);
  if (wal_) {
    if (!failed_.load()) s = wal_->sync();
    wal_->close();
  }
  return s;
}

// ===================================================================== background

void CollectionImpl::RequestFlush() {
  {
    std::lock_guard<std::mutex> g(bg_mu_);
    flush_requested_ = true;
  }
  bg_cv_.notify_all();
}

void CollectionImpl::RequestCompaction(size_t target_segments) {
  {
    std::lock_guard<std::mutex> g(bg_mu_);
    compact_requested_ = true;
    if (target_segments > 0) compact_target_ = target_segments;
  }
  bg_cv_.notify_all();
}

void CollectionImpl::FlushLoop() {
  std::unique_lock<std::mutex> l(bg_mu_);
  // Wake periodically to enforce the age limit as well as the size trigger.
  const uint64_t age_us = uint64_t{options_.flush_age_ms} * 1000;
  const auto tick = std::chrono::milliseconds(
      age_us == 0 ? 1000 : std::clamp<uint32_t>(options_.flush_age_ms / 4, 10, 1000));
  for (;;) {
    bool requested = bg_cv_.wait_for(l, tick, [&] { return stop_ || flush_requested_; });
    if (stop_) return;
    flush_requested_ = false;
    bool aged = false;
    if (!requested && open_.auto_flush && age_us > 0) {
      uint64_t first = snapshot()->active->first_append_us();
      aged = first != 0 && NowMicros() - first >= age_us;
    }
    if (!requested && !aged) continue;
    l.unlock();
    FlushOnce();  // failures are recorded by SetFailed
    l.lock();
  }
}

void CollectionImpl::CompactLoop() {
#if defined(__linux__)
  // One compaction at a time, on a low-priority thread: it must not slow down queries.
  setpriority(PRIO_PROCESS, static_cast<id_t>(::syscall(SYS_gettid)), 10);
#endif
  std::unique_lock<std::mutex> l(bg_mu_);
  const auto tick = std::chrono::milliseconds(
      open_.idle_compact_ms == 0 ? 1000 : std::clamp<uint32_t>(open_.idle_compact_ms / 4, 10, 1000));
  for (;;) {
    bool woke = bg_cv_.wait_for(l, tick, [&] { return stop_ || compact_requested_; });
    if (stop_) return;
    size_t target = 0;
    if (woke) {
      compact_requested_ = false;
      target = compact_target_;
      compact_target_ = 0;
    }
    l.unlock();
    if (woke && (open_.auto_compact || target > 0)) Compact(target);
    // Idle for a while: merge everything into the largest segment, once per quiet period.
    uint64_t last = last_write_us_.load();
    if (open_.auto_compact && open_.idle_compact_ms > 0 && last != idle_merged_at_ &&
        NowMicros() - last >= uint64_t{open_.idle_compact_ms} * 1000) {
      if (snapshot()->on_disk.size() > 1) Compact(1);
      idle_merged_at_ = last;
    }
    l.lock();
  }
}

void CollectionImpl::SyncLoop() {
  std::unique_lock<std::mutex> l(bg_mu_);
  const auto interval = std::chrono::milliseconds(std::max<uint32_t>(1, options_.sync_interval_ms));
  for (;;) {
    bg_cv_.wait_for(l, interval, [&] { return stop_; });
    if (stop_) return;
    l.unlock();
    std::shared_ptr<WalWriter> w;
    {
      std::lock_guard<std::mutex> g(wal_ptr_mu_);
      w = wal_;
    }
    if (w) {
      Status s = w->sync();
      if (!s.ok()) SetFailed(s);
    }
    l.lock();
  }
}

void CollectionImpl::SetFailed(const Status& s) {
  std::lock_guard<std::mutex> g(fail_mu_);
  if (!failed_.load()) {
    failure_ = Status(s.code(), "collection stopped accepting writes after an I/O failure: " + s.message());
    failed_.store(true);
  }
}

Status CollectionImpl::failure() const {
  std::lock_guard<std::mutex> g(fail_mu_);
  return failure_;
}

// ===================================================================== write path

Status CollectionImpl::PrepareUpsert(VectorId id, const float* vec, const Metadata* md,
                                     WriteOp* op) const {
  if (vec == nullptr) return Status::InvalidArgument("missing vector");
  op->kind = WriteOp::kUpsert;
  op->id = id;
  op->vec.assign(padded_, 0.f);
  for (size_t i = 0; i < options_.dim; ++i) {
    if (!std::isfinite(vec[i])) return Status::InvalidArgument("vector has a NaN or infinite value");
    op->vec[i] = vec[i];
  }
  if (options_.metric == Metric::Cosine && !NormalizeVector(op->vec.data(), options_.dim)) {
    return Status::InvalidArgument("a zero vector cannot be normalized for cosine");
  }
  if (md != nullptr) {
    VDB_RETURN_IF_ERROR(ToMetaRow(options_.schema, *md, &op->meta));
  } else {
    op->meta.assign(options_.schema.size(), std::monostate{});
  }
  op->meta_enc.clear();
  EncodeMetaRow(op->meta, &op->meta_enc);
  return Status::OK();
}

Status CollectionImpl::Upsert(const std::vector<UpsertPoint>& points, uint64_t* last_lsn) {
  std::vector<WriteOp> ops(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    VDB_RETURN_IF_ERROR(PrepareUpsert(points[i].id, points[i].vector, points[i].metadata, &ops[i]));
  }
  return Write(&ops, last_lsn);
}

Status CollectionImpl::UpsertRows(const VectorId* ids, const float* vectors, size_t n,
                                  const std::vector<Metadata>* metadata, uint64_t* last_lsn) {
  if (metadata != nullptr && metadata->size() != n) {
    return Status::InvalidArgument("metadata count does not match the number of rows");
  }
  std::vector<WriteOp> ops(n);
  for (size_t i = 0; i < n; ++i) {
    VDB_RETURN_IF_ERROR(PrepareUpsert(ids[i], vectors + i * options_.dim,
                                      metadata != nullptr ? &(*metadata)[i] : nullptr, &ops[i]));
  }
  return Write(&ops, last_lsn);
}

Status CollectionImpl::Remove(const VectorId* ids, size_t n, size_t* removed, bool* first_found,
                              uint64_t* last_lsn) {
  std::vector<WriteOp> ops(n);
  for (size_t i = 0; i < n; ++i) {
    ops[i].kind = WriteOp::kDelete;
    ops[i].id = ids[i];
  }
  VDB_RETURN_IF_ERROR(Write(&ops, last_lsn));
  size_t c = 0;
  for (const auto& op : ops) c += op.found ? 1 : 0;
  if (removed != nullptr) *removed = c;
  if (first_found != nullptr) *first_found = !ops.empty() && ops[0].found;
  return Status::OK();
}

Status CollectionImpl::Write(std::vector<WriteOp>* ops, uint64_t* last_lsn) {
  if (closed_.load()) return Status::FailedPrecondition("collection is closed");
  if (failed_.load()) return failure();
  if (ops->empty()) return Status::OK();
  if (ops->size() <= kOpsPerWriter) {
    Writer w;
    w.ops = ops;
    VDB_RETURN_IF_ERROR(Submit(&w));
  } else {
    // Split very large batches so one group commit stays bounded.
    for (size_t b = 0; b < ops->size(); b += kOpsPerWriter) {
      size_t e = std::min(ops->size(), b + kOpsPerWriter);
      std::vector<WriteOp> part(std::make_move_iterator(ops->begin() + static_cast<std::ptrdiff_t>(b)),
                                std::make_move_iterator(ops->begin() + static_cast<std::ptrdiff_t>(e)));
      Writer w;
      w.ops = &part;
      Status s = Submit(&w);
      for (size_t i = b; i < e; ++i) {
        (*ops)[i].lsn = part[i - b].lsn;
        (*ops)[i].found = part[i - b].found;
      }
      VDB_RETURN_IF_ERROR(s);
    }
  }
  if (last_lsn != nullptr) *last_lsn = ops->back().lsn;
  return Status::OK();
}

Status CollectionImpl::RunExclusive(const std::function<Status()>& fn) {
  Writer w;
  w.exclusive = &fn;
  return Submit(&w);
}

Status CollectionImpl::Submit(Writer* w) {
  std::unique_lock<std::mutex> lk(writer_mu_);
  writers_.push_back(w);
  while (!w->done && w != writers_.front()) w->cv.wait(lk);
  if (w->done) return w->status;

  if (w->exclusive != nullptr) {
    w->status = (*w->exclusive)();
    w->done = true;
    writers_.pop_front();
    if (!writers_.empty()) writers_.front()->cv.notify_one();
    return w->status;
  }

  // Group commit: this writer leads every data write queued behind it.
  std::vector<Writer*> group;
  size_t bytes = 0;
  for (Writer* x : writers_) {
    if (x->exclusive != nullptr) break;
    if (!group.empty() && (bytes > kMaxGroupBytes || !open_.group_commit)) break;
    group.push_back(x);
    for (const auto& op : *x->ops) bytes += op.vec.size() * sizeof(float) + op.meta_enc.size() + 64;
  }
  Status s;
  if (failed_.load()) s = failure();
  uint64_t lsn = last_lsn_;
  size_t nrecords = 0;
  std::string& buf = group_buf_;
  buf.clear();
  if (s.ok()) {
    std::string payload;
    for (Writer* x : group) {
      for (WriteOp& op : *x->ops) {
        op.lsn = ++lsn;
        payload.clear();
        if (op.kind == WriteOp::kUpsert) {
          EncodeUpsertPayload(&payload, op.id, op.vec.data(), options_.dim, op.meta_enc);
          AppendWalRecord(&buf, WalRecordType::kUpsert, op.lsn, payload);
        } else {
          EncodeDeletePayload(&payload, op.id);
          AppendWalRecord(&buf, WalRecordType::kDelete, op.lsn, payload);
        }
        ++nrecords;
      }
    }
  }
  std::shared_ptr<WalWriter> wal = wal_;
  lk.unlock();
  if (s.ok()) s = wal->append(buf);
  if (s.ok() && options_.sync == SyncMode::Always) s = wal->sync();
  lk.lock();

  if (!s.ok()) {
    SetFailed(s);
  } else {
    last_lsn_ = lsn;
    last_lsn_pub_.store(lsn);
    GlobalMetrics().wal_records_total.inc(nrecords);
    auto snap = snapshot();
    for (Writer* x : group)
      for (WriteOp& op : *x->ops) ApplyOp(op, /*replay=*/false, *snap);
    last_write_us_.store(NowMicros());
    if (open_.auto_flush && snap->active->vector_bytes() >= options_.flush_bytes) RequestFlush();
  }
  for (Writer* x : group) {
    writers_.pop_front();
    x->status = s;
    x->done = true;
    if (x != w) x->cv.notify_one();
  }
  if (!writers_.empty()) writers_.front()->cv.notify_one();
  return s;
}

void CollectionImpl::MarkDeleted(const Snapshot& snap, RowLoc loc) {
  if (snap.active->id() == loc.segment) {
    snap.active->deletes().set(loc.row);
    return;
  }
  for (const auto& f : snap.frozen) {
    if (f->id() == loc.segment) {
      f->deletes().set(loc.row);
      return;
    }
  }
  for (const auto& s : snap.on_disk) {
    if (s->id() == loc.segment) {
      s->deletes().set(loc.row);
      return;
    }
  }
}

// Applies one write to memory. Runs under writer_mu_ (or single-threaded recovery).
// During replay, a record older than the segment holding the id's current row (a bulk
// load or compaction committed after the record) must not touch that row.
void CollectionImpl::ApplyOp(WriteOp& op, bool replay, const Snapshot& snap) {
  std::optional<RowLoc> old = id_map_.Find(op.id);
  bool old_is_newer = false;
  if (old && replay) {
    auto it = replay_seg_lsn_.find(old->segment);
    old_is_newer = it != replay_seg_lsn_.end() && it->second >= op.lsn;
  }
  if (op.kind == WriteOp::kDelete) {
    if (old && !old_is_newer) {
      MarkDeleted(snap, *old);
      id_map_.Erase(op.id);
      op.found = true;
    }
    return;
  }
  if (old && !old_is_newer) MarkDeleted(snap, *old);
  auto row = snap.active->Append(op.id, op.vec.data(), std::move(op.meta), op.lsn);
  if (!row.ok()) {
    SetFailed(row.status());
    return;
  }
  if (old_is_newer) {
    snap.active->deletes().set(*row);  // superseded by the newer segment's row
  } else {
    id_map_.Put(op.id, RowLoc{snap.active->id(), *row});
  }
}

// ===================================================================== snapshots / manifest

void CollectionImpl::ModifySnapshot(const std::function<void(Snapshot&)>& fn) {
  std::lock_guard<std::mutex> g(snap_mu_);
  auto next = std::make_shared<Snapshot>(*snap_);
  fn(*next);
  snap_ = std::move(next);
}

// Caller holds manifest_mu_. Allocates the manifest number, commits, swaps manifest_.
Status CollectionImpl::CommitManifestLocked(Manifest m, std::vector<std::string>* garbage) {
  m.number = NextFileNumber();
  m.next_file_number = next_file_number_.load();
  Status s = CommitManifest(env_, dir_, m);
  if (!s.ok()) {
    SetFailed(s);  // the on-disk state is ambiguous until the next recovery
    return s;
  }
  garbage->push_back(JoinPath(dir_, ManifestFileName(manifest_.number)));
  manifest_ = std::move(m);
  return Status::OK();
}

void CollectionImpl::RemoveObsoleteLogs(uint64_t below) {
  std::vector<std::string> names;
  if (!env_->list_dir(wal_dir_, &names).ok()) return;
  for (const auto& n : names) {
    uint64_t num = 0;
    if (ParseWalFileName(n, &num) && num < below) env_->remove_file(JoinPath(wal_dir_, n));
  }
}

// ===================================================================== flush

// The order matters; a crash at any step leaves either the old manifest current (the
// log still holds the data) or the new one (records at or below flushed_lsn are skipped).
Status CollectionImpl::FlushOnce() {
  std::lock_guard<std::mutex> fl(flush_mu_);
  if (failed_.load()) return failure();
  std::shared_ptr<MutableSegment> frozen;
  uint64_t flushed_lsn = 0, new_log = 0;
  std::vector<std::pair<std::shared_ptr<Segment>, roaring::Roaring>> disk_deletes;
  roaring::Roaring frozen_deletes;

  // 1. Freeze: the mutable segment becomes read-only, a new one takes writes, a new log
  //    starts, and the delete sets of on-disk segments are captured as of this LSN.
  Status s = RunExclusive([&]() -> Status {
    auto snap = snapshot();
    if (snap->active->size() == 0) return Status::OK();
    flushed_lsn = last_lsn_;
    std::shared_ptr<WalWriter> old_wal = wal_;
    VDB_RETURN_IF_ERROR(old_wal->sync());  // older logs are always complete
    new_log = NextFileNumber();
    VDB_ASSIGN_OR_RETURN(auto w, WalWriter::Create(env_, JoinPath(wal_dir_, WalFileName(new_log)), new_log));
    VDB_RETURN_IF_ERROR(env_->sync_dir(wal_dir_));
    {
      std::lock_guard<std::mutex> g(wal_ptr_mu_);
      wal_ = std::shared_ptr<WalWriter>(std::move(w));
    }
    old_wal->close();
    frozen = snap->active;
    frozen->freeze();
    for (const auto& seg : snap->on_disk) {
      if (seg->deletes().count() != seg->persisted_deletes()) {
        disk_deletes.emplace_back(seg, seg->deletes().to_roaring(seg->rows()));
      }
    }
    frozen_deletes = frozen->deletes().to_roaring(frozen->size());
    auto fresh = std::make_shared<MutableSegment>(static_cast<uint32_t>(NextFileNumber()), options_.dim);
    ModifySnapshot([&](Snapshot& sn) {
      sn.active = fresh;
      sn.frozen.push_back(frozen);
    });
    return Status::OK();
  });
  if (!s.ok()) {
    SetFailed(s);
    return s;
  }
  if (!frozen) return Status::OK();
  Timer timer;
  auto fail = [&](const Status& st) {
    SetFailed(st);
    return st;
  };

  // 2-4. Write the segment files, build its index, fsync, rename, fsync segments/.
  SegmentSource src;
  src.rows = frozen->size();
  src.row = [&](size_t i) { return frozen->row(static_cast<InternalId>(i)); };
  src.id = [&](size_t i) { return frozen->external_id(static_cast<InternalId>(i)); };
  src.meta = [&](size_t i, std::string* out) { EncodeMetaRow(frozen->meta(static_cast<InternalId>(i)), out); };
  SegmentWriteOptions wo;
  wo.id = frozen->id();
  wo.lsn = flushed_lsn;
  wo.options = &options_;
  wo.pool = pool_.get();
  SegmentMeta meta;
  auto seg_or = WriteSegment(env_, seg_dir_, src, wo, frozen->deletes_ptr(), &meta);
  if (!seg_or.ok()) return fail(seg_or.status());
  std::shared_ptr<Segment> seg = std::move(seg_or).value();
  if (!frozen_deletes.isEmpty()) {
    FileInfo fi;
    std::string name;
    Status ds = WriteDeletesFile(env_, seg->dir(), NextFileNumber(), frozen_deletes, &name, &fi);
    if (!ds.ok()) return fail(ds);
    meta.deletes_file = name;
    meta.deleted = frozen_deletes.cardinality();
    meta.files[name] = fi;
  }
  struct Written {
    std::shared_ptr<Segment> seg;
    std::string name;
    FileInfo info;
    size_t count;
  };
  std::vector<Written> written;
  for (auto& [dseg, bm] : disk_deletes) {
    Written wr{dseg, "", {}, static_cast<size_t>(bm.cardinality())};
    Status ds = WriteDeletesFile(env_, dseg->dir(), NextFileNumber(), bm, &wr.name, &wr.info);
    if (!ds.ok()) return fail(ds);
    written.push_back(std::move(wr));
  }

  // 5. Commit the manifest with the new segment, delete sets and flushed_lsn.
  std::vector<std::string> garbage;
  {
    std::lock_guard<std::mutex> mg(manifest_mu_);
    Manifest m = manifest_;
    m.segments.push_back(meta);
    for (auto& wr : written) {
      auto it = std::find_if(m.segments.begin(), m.segments.end(),
                             [&](const SegmentMeta& sm) { return sm.id == wr.seg->id(); });
      if (it == m.segments.end()) continue;  // compacted away meanwhile
      if (!it->deletes_file.empty()) {
        garbage.push_back(JoinPath(wr.seg->dir(), it->deletes_file));
        it->files.erase(it->deletes_file);
      }
      it->deletes_file = wr.name;
      it->deleted = wr.count;
      it->files[wr.name] = wr.info;
    }
    m.flushed_lsn = std::max(m.flushed_lsn, flushed_lsn);
    m.wal_log_number = new_log;
    VDB_RETURN_IF_ERROR(CommitManifestLocked(std::move(m), &garbage));
  }
  seg->set_persisted_deletes(frozen_deletes.cardinality());
  for (auto& wr : written) wr.seg->set_persisted_deletes(wr.count);

  // 6. Swap in memory: readers move from the frozen segment to the mapped files.
  ModifySnapshot([&](Snapshot& sn) {
    sn.frozen.erase(std::remove(sn.frozen.begin(), sn.frozen.end(), frozen), sn.frozen.end());
    sn.on_disk.push_back(seg);
  });

  // 7. Clean up: logs whose records are all at or below flushed_lsn, the old manifest.
  for (const auto& g : garbage) env_->remove_file(g);
  RemoveObsoleteLogs(new_log);
  GlobalMetrics().flush_seconds.observe(timer.seconds());
  GlobalMetrics().flushes_total.inc();
  flushes_.fetch_add(1);
  if (open_.auto_compact) RequestCompaction(0);
  return Status::OK();
}

Status CollectionImpl::Flush() {
  if (closed_.load()) return Status::FailedPrecondition("collection is closed");
  return FlushOnce();
}

// ===================================================================== compaction

Status CollectionImpl::Compact(size_t target_segments) {
  const bool extend_ok = options_.index.type == IndexType::Hnsw || options_.index.type == IndexType::Flat;
  for (int round = 0; round < 64; ++round) {
    if (failed_.load()) return failure();
    std::vector<SegmentSummary> sums;
    // Hold the snapshot: a range-for over snapshot()->on_disk would destroy the temporary
    // shared_ptr before the loop runs (C++20), and a concurrent swap would free the vector.
    auto snap = snapshot();
    for (const auto& s : snap->on_disk) sums.push_back({s->id(), s->rows(), s->live_rows()});
    auto task = PickCompaction(sums, options_, target_segments, extend_ok);
    if (!task) return Status::OK();
    VDB_RETURN_IF_ERROR(CompactOnce(*task));
  }
  return Status::OK();
}

Status CollectionImpl::CompactOnce(const CompactionTask& task) {
  std::lock_guard<std::mutex> cl(compact_mu_);
  if (failed_.load()) return failure();
  Timer timer;

  // Start: the input segments and a copy of their delete sets as of now.
  std::vector<std::shared_ptr<Segment>> inputs;
  {
    auto snap = snapshot();
    for (uint32_t id : task.inputs) {
      auto it = std::find_if(snap->on_disk.begin(), snap->on_disk.end(),
                             [&](const std::shared_ptr<Segment>& s) { return s->id() == id; });
      if (it == snap->on_disk.end()) return Status::OK();  // segments changed; re-plan
      inputs.push_back(*it);
    }
  }
  std::vector<roaring::Roaring> start_del(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) start_del[i] = inputs[i]->deletes().to_roaring(inputs[i]->rows());

  // Copy live rows with new row numbers, remembering (input, old row) for each new row.
  // Merging into the largest keeps all of its rows (deleted ones too) so the copied
  // graph's node ids stay valid.
  struct Src {
    uint32_t input;
    uint32_t row;
  };
  std::vector<Src> rows;
  const bool merge = task.merge_into_largest && !inputs.empty();
  if (merge) {
    for (size_t r = 0; r < inputs[0]->rows(); ++r) rows.push_back({0, static_cast<uint32_t>(r)});
  }
  for (size_t i = merge ? 1 : 0; i < inputs.size(); ++i) {
    for (size_t r = 0; r < inputs[i]->rows(); ++r) {
      if (!start_del[i].contains(static_cast<uint32_t>(r))) rows.push_back({static_cast<uint32_t>(i), static_cast<uint32_t>(r)});
    }
  }

  std::shared_ptr<Segment> out;
  SegmentMeta meta;
  const auto new_id = static_cast<uint32_t>(NextFileNumber());
  if (!rows.empty()) {
    auto bits = std::make_shared<DeleteBits>(rows.size());
    bits->ensure(rows.size());
    if (merge) {
      for (uint32_t r : start_del[0]) bits->set(r);
    }
    SegmentSource src;
    src.rows = rows.size();
    src.row = [&](size_t j) { return inputs[rows[j].input]->row(rows[j].row); };
    src.id = [&](size_t j) { return inputs[rows[j].input]->external_id(rows[j].row); };
    src.meta = [&](size_t j, std::string* o) {
      EncodeMetaRow(inputs[rows[j].input]->columns().GetRow(rows[j].row), o);
    };
    SegmentWriteOptions wo;
    wo.id = new_id;
    wo.options = &options_;
    wo.pool = pool_.get();
    wo.extend_from = merge ? &inputs[0]->index() : nullptr;
    auto seg_or = WriteSegment(env_, seg_dir_, src, wo, bits, &meta);
    if (!seg_or.ok()) {
      SetFailed(seg_or.status());
      return seg_or.status();
    }
    out = std::move(seg_or).value();
  }

  // Commit, with no writer running: catch up deletes that hit the inputs since the
  // start (missing this step is the classic compaction bug where deleted rows come back),
  // persist the delete set, commit the manifest, then swap the id map and the snapshot.
  std::vector<std::string> garbage;
  Status s = RunExclusive([&]() -> Status {
    if (out) {
      DeleteBits& nb = out->deletes();
      for (size_t j = 0; j < rows.size(); ++j) {
        const Src& sr = rows[j];
        if (inputs[sr.input]->deletes().test(sr.row) && !start_del[sr.input].contains(sr.row)) nb.set(j);
      }
      roaring::Roaring del = nb.to_roaring(rows.size());
      if (!del.isEmpty()) {
        FileInfo fi;
        std::string name;
        Status ds = WriteDeletesFile(env_, out->dir(), NextFileNumber(), del, &name, &fi);
        if (!ds.ok()) {
          SetFailed(ds);
          return ds;
        }
        meta.deletes_file = name;
        meta.deleted = del.cardinality();
        meta.files[name] = fi;
      }
      out->set_persisted_deletes(del.cardinality());
      meta.lsn = last_lsn_;  // reflects every record up to now
      out->set_lsn(last_lsn_);
    }
    {
      std::lock_guard<std::mutex> mg(manifest_mu_);
      Manifest m = manifest_;
      m.segments.erase(std::remove_if(m.segments.begin(), m.segments.end(),
                                      [&](const SegmentMeta& sm) {
                                        return std::find(task.inputs.begin(), task.inputs.end(), sm.id) != task.inputs.end();
                                      }),
                       m.segments.end());
      if (out) m.segments.push_back(meta);
      VDB_RETURN_IF_ERROR(CommitManifestLocked(std::move(m), &garbage));
    }
    ModifySnapshot([&](Snapshot& sn) {
      sn.on_disk.erase(std::remove_if(sn.on_disk.begin(), sn.on_disk.end(),
                                      [&](const std::shared_ptr<Segment>& x) {
                                        return std::find(inputs.begin(), inputs.end(), x) != inputs.end();
                                      }),
                       sn.on_disk.end());
      if (out) sn.on_disk.push_back(out);
    });
    if (out) {
      for (size_t j = 0; j < rows.size(); ++j) {
        const Src& sr = rows[j];
        const Segment& in = *inputs[sr.input];
        id_map_.ReplaceIf(in.external_id(sr.row), RowLoc{in.id(), sr.row}, RowLoc{new_id, static_cast<uint32_t>(j)});
      }
    }
    for (auto& in : inputs) in->mark_obsolete();  // files go when the last reader lets go
    return Status::OK();
  });
  if (!s.ok()) {
    if (out) out->mark_obsolete();
    return s;
  }
  for (const auto& g : garbage) env_->remove_file(g);
  GlobalMetrics().compaction_seconds.observe(timer.seconds());
  GlobalMetrics().compactions_total.inc();
  compactions_.fetch_add(1);
  return Status::OK();
}

// ===================================================================== bulk load

Status CollectionImpl::BulkLoad(const VectorId* ids, const float* vectors, size_t n,
                                const std::vector<Metadata>* metadata) {
  if (closed_.load()) return Status::FailedPrecondition("collection is closed");
  if (failed_.load()) return failure();
  if (n == 0) return Status::OK();
  if (metadata != nullptr && metadata->size() != n) {
    return Status::InvalidArgument("metadata count does not match the number of rows");
  }
  if (n >= (size_t{1} << 32) - 1) return Status::InvalidArgument("bulk load too large for one segment");
  {
    std::vector<VectorId> sorted(ids, ids + n);
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
      return Status::InvalidArgument("duplicate id in bulk load");
    }
  }
  for (size_t i = 0; i < n; ++i) {
    if (id_map_.Find(ids[i])) {
      return Status::AlreadyExists("id " + std::to_string(ids[i]) +
                                   " already exists; bulk load accepts only new ids");
    }
  }
  AlignedBuffer buf(n * padded_ * sizeof(float));
  float* rows = buf.floats();
  for (size_t i = 0; i < n; ++i) {
    float* dst = rows + i * padded_;
    for (size_t d = 0; d < options_.dim; ++d) {
      float v = vectors[i * options_.dim + d];
      if (!std::isfinite(v)) return Status::InvalidArgument("vector has a NaN or infinite value");
      dst[d] = v;
    }
    if (options_.metric == Metric::Cosine && !NormalizeVector(dst, options_.dim)) {
      return Status::InvalidArgument("a zero vector cannot be normalized for cosine");
    }
  }
  std::vector<std::string> enc;
  if (metadata != nullptr) {
    enc.resize(n);
    MetaRow mr;
    for (size_t i = 0; i < n; ++i) {
      VDB_RETURN_IF_ERROR(ToMetaRow(options_.schema, (*metadata)[i], &mr));
      EncodeMetaRow(mr, &enc[i]);
    }
  }

  const auto seg_id = static_cast<uint32_t>(NextFileNumber());
  auto bits = std::make_shared<DeleteBits>(n);
  bits->ensure(n);
  SegmentSource src;
  src.rows = n;
  src.row = [&](size_t i) -> const float* { return rows + i * padded_; };
  src.id = [&](size_t i) { return ids[i]; };
  if (!enc.empty()) src.meta = [&](size_t i, std::string* o) { o->append(enc[i]); };
  SegmentWriteOptions wo;
  wo.id = seg_id;
  wo.options = &options_;
  wo.pool = pool_.get();
  SegmentMeta meta;
  VDB_ASSIGN_OR_RETURN(auto seg, WriteSegment(env_, seg_dir_, src, wo, bits, &meta));

  std::vector<std::string> garbage;
  Status s = RunExclusive([&]() -> Status {
    for (size_t i = 0; i < n; ++i) {
      if (id_map_.Find(ids[i])) {
        return Status::AlreadyExists("id " + std::to_string(ids[i]) +
                                     " already exists; bulk load accepts only new ids");
      }
    }
    meta.lsn = last_lsn_;
    seg->set_lsn(last_lsn_);
    {
      std::lock_guard<std::mutex> mg(manifest_mu_);
      Manifest m = manifest_;
      m.segments.push_back(meta);
      VDB_RETURN_IF_ERROR(CommitManifestLocked(std::move(m), &garbage));
    }
    ModifySnapshot([&](Snapshot& sn) { sn.on_disk.push_back(seg); });
    for (size_t i = 0; i < n; ++i) id_map_.Put(ids[i], RowLoc{seg_id, static_cast<uint32_t>(i)});
    return Status::OK();
  });
  if (!s.ok()) {
    seg->mark_obsolete();  // never committed: all or nothing
    return s;
  }
  for (const auto& g : garbage) env_->remove_file(g);
  last_write_us_.store(NowMicros());
  if (open_.auto_compact) RequestCompaction(0);
  return Status::OK();
}

// ===================================================================== read path

StatusOr<std::unique_ptr<Filter>> CollectionImpl::ParseFilter(std::string_view json) const {
  VDB_ASSIGN_OR_RETURN(auto f, FilterExpr::Parse(json, options_.schema));
  return std::unique_ptr<Filter>(std::move(f));
}

Status CollectionImpl::PrepareQuery(const float* v, std::vector<float>* q) const {
  if (v == nullptr) return Status::InvalidArgument("missing query vector");
  q->assign(padded_, 0.f);
  for (size_t i = 0; i < options_.dim; ++i) {
    if (!std::isfinite(v[i])) return Status::InvalidArgument("query has a NaN or infinite value");
    (*q)[i] = v[i];
  }
  if (options_.metric == Metric::Cosine && !NormalizeVector(q->data(), options_.dim)) {
    return Status::InvalidArgument("a zero query cannot be normalized for cosine");
  }
  return Status::OK();
}

Status CollectionImpl::Search(const SearchRequest& req, std::vector<SearchHit>* out,
                              SearchExplain* explain, bool parallel_segments) const {
  out->clear();
  if (req.k == 0 || req.k > kMaxK) return Status::InvalidArgument("k must be in 1..10000");
  const FilterExpr* filter = nullptr;
  if (req.filter != nullptr) {
    filter = dynamic_cast<const FilterExpr*>(req.filter);
    if (filter == nullptr) return Status::InvalidArgument("filter was not created by this library");
  }
  Timer timer;
  std::vector<float> q;
  VDB_RETURN_IF_ERROR(PrepareQuery(req.vector, &q));
  auto snap = snapshot();

  struct Part {
    const MutableSegment* m = nullptr;
    const Segment* d = nullptr;
    const char* kind = "";
    SegmentHits hits;
  };
  std::vector<Part> parts;
  parts.reserve(1 + snap->frozen.size() + snap->on_disk.size());
  parts.push_back(Part{snap->active.get(), nullptr, "mutable", {}});
  for (const auto& f : snap->frozen) parts.push_back(Part{f.get(), nullptr, "frozen", {}});
  for (const auto& s : snap->on_disk) parts.push_back(Part{nullptr, s.get(), "disk", {}});
  auto run = [&](size_t i) {
    Part& p = parts[i];
    if (p.m != nullptr) {
      SearchMutable(*p.m, q.data(), req.k, filter, dist_, p.kind, &p.hits);
    } else {
      SearchDiskSegment(*p.d, q.data(), req.k, req.params, filter, dist_, &p.hits);
    }
  };
  if (parallel_segments && parts.size() >= 4 && pool_->parallelism() > 1) {
    pool_->parallel_for(0, parts.size(), run);
  } else {
    for (size_t i = 0; i < parts.size(); ++i) run(i);
  }

  // Merge the per-segment top-k lists; an id appears once (a concurrent upsert can
  // briefly leave the old and the new row both visible).
  struct Cand {
    float dist;
    uint32_t part;
    InternalId row;
  };
  std::vector<Cand> all;
  uint64_t dc = 0;
  for (size_t i = 0; i < parts.size(); ++i) {
    for (const auto& h : parts[i].hits.hits) all.push_back({h.dist, static_cast<uint32_t>(i), h.id});
    dc += parts[i].hits.explain.distance_computations;
  }
  std::sort(all.begin(), all.end(), [](const Cand& a, const Cand& b) {
    return a.dist < b.dist || (a.dist == b.dist && (a.part < b.part || (a.part == b.part && a.row < b.row)));
  });
  std::unordered_set<VectorId> seen;
  for (const Cand& c : all) {
    const Part& p = parts[c.part];
    VectorId ext = p.m != nullptr ? p.m->external_id(c.row) : p.d->external_id(c.row);
    if (!seen.insert(ext).second) continue;
    SearchHit h;
    h.id = ext;
    h.distance = c.dist;
    h.score = options_.metric == Metric::L2 ? c.dist : -c.dist;
    if (req.include_metadata) {
      h.metadata = FromMetaRow(options_.schema, p.m != nullptr ? p.m->meta(c.row) : p.d->columns().GetRow(c.row));
    }
    if (req.include_vector) {
      const float* r = p.m != nullptr ? p.m->row(c.row) : p.d->row(c.row);
      h.vector.assign(r, r + options_.dim);
    }
    out->push_back(std::move(h));
    if (out->size() >= req.k) break;
  }
  GlobalMetrics().searches_total.inc();
  GlobalMetrics().search_distance_computations.observe(static_cast<double>(dc));
  if (explain != nullptr) {
    explain->segments.clear();
    for (auto& p : parts) explain->segments.push_back(std::move(p.hits.explain));
    explain->distance_computations = dc;
    explain->total_micros = timer.micros();
  }
  return Status::OK();
}

Status CollectionImpl::SearchBatch(const float* queries, size_t nq, size_t k,
                                   const SearchParams& params, const Filter* filter,
                                   std::vector<std::vector<SearchHit>>* out) const {
  out->assign(nq, {});
  std::vector<Status> st(nq);
  pool_->parallel_for(0, nq, [&](size_t i) {
    SearchRequest req;
    req.vector = queries + i * options_.dim;
    req.k = k;
    req.params = params;
    req.filter = filter;
    st[i] = Search(req, &(*out)[i], nullptr, /*parallel_segments=*/false);
  });
  for (const auto& s : st) VDB_RETURN_IF_ERROR(s);
  return Status::OK();
}

Status CollectionImpl::Get(VectorId id, std::vector<float>* vector, Metadata* metadata) const {
  // A compaction swaps the snapshot and then repoints the id map; retry across that gap.
  for (int attempt = 0; attempt < 16; ++attempt) {
    auto snap = snapshot();
    auto loc = id_map_.Find(id);
    if (!loc) return Status::NotFound("id " + std::to_string(id) + " not found");
    auto deliver = [&](const float* r, const MetaRow& m) {
      if (vector != nullptr) vector->assign(r, r + options_.dim);
      if (metadata != nullptr) *metadata = FromMetaRow(options_.schema, m);
    };
    auto try_mutable = [&](const MutableSegment& ms) {
      if (ms.id() != loc->segment || loc->row >= ms.size() || ms.deletes().test(loc->row)) return false;
      deliver(ms.row(loc->row), ms.meta(loc->row));
      return true;
    };
    if (try_mutable(*snap->active)) return Status::OK();
    bool done = false;
    for (const auto& f : snap->frozen) done = done || try_mutable(*f);
    if (done) return Status::OK();
    for (const auto& s : snap->on_disk) {
      if (s->id() == loc->segment && loc->row < s->rows() && !s->deletes().test(loc->row)) {
        deliver(s->row(loc->row), s->columns().GetRow(loc->row));
        return Status::OK();
      }
    }
    std::this_thread::yield();
  }
  return Status::NotFound("id " + std::to_string(id) + " not found");
}

CollectionStats CollectionImpl::Stats() const {
  CollectionStats st;
  auto snap = snapshot();
  auto add_mutable = [&](const MutableSegment& m, const char* kind) {
    SegmentStats s;
    s.id = m.id();
    s.kind = kind;
    s.index = "flat";
    s.rows = m.size();
    s.deleted = std::min(m.size(), m.deletes().count());
    s.lsn = m.max_lsn();
    s.memory_bytes = m.memory_bytes();
    st.memory.vectors += m.vector_bytes();
    st.memory.metadata += s.memory_bytes - std::min(s.memory_bytes, m.vector_bytes());
    st.segments.push_back(s);
  };
  add_mutable(*snap->active, "mutable");
  for (const auto& f : snap->frozen) add_mutable(*f, "frozen");
  for (const auto& d : snap->on_disk) {
    SegmentStats s;
    s.id = d->id();
    s.kind = "disk";
    s.index = IndexTypeName(d->index().type());
    s.rows = d->rows();
    s.deleted = std::min(d->rows(), d->deletes().count());
    s.lsn = d->lsn();
    MemoryStats m = d->memory();
    s.memory_bytes = m.total();
    s.disk_bytes = d->disk_bytes();
    st.memory.vectors += m.vectors;
    st.memory.graph += m.graph;
    st.memory.codes += m.codes;
    st.memory.id_map += m.id_map;
    st.memory.metadata += m.metadata;
    st.disk_bytes += s.disk_bytes;
    st.segments.push_back(s);
  }
  for (const auto& s : st.segments) {
    st.total_rows += s.rows;
    st.deleted_rows += s.deleted;
  }
  st.live_rows = st.total_rows - st.deleted_rows;
  st.memory.id_map += id_map_.memory_bytes();
  st.last_lsn = last_lsn_pub_.load();
  {
    std::lock_guard<std::mutex> g(manifest_mu_);
    st.flushed_lsn = manifest_.flushed_lsn;
  }
  {
    std::lock_guard<std::mutex> g(wal_ptr_mu_);
    if (wal_) st.disk_bytes += wal_->bytes();
  }
  st.flushes = flushes_.load();
  st.compactions = compactions_.load();
  st.last_recovery_seconds = recovery_seconds_;
  st.last_recovery_records = recovery_records_;
  return st;
}

// ===================================================================== Collection

Collection::Collection(std::unique_ptr<CollectionImpl> impl) : impl_(std::move(impl)) {}
Collection::~Collection() = default;

StatusOr<std::unique_ptr<Collection>> Collection::Create(const std::string& path,
                                                         const CollectionOptions& options,
                                                         const OpenOptions& open, Env* env) {
  VDB_ASSIGN_OR_RETURN(auto impl, CollectionImpl::Create(path, options, open, env));
  return std::unique_ptr<Collection>(new Collection(std::move(impl)));
}

StatusOr<std::unique_ptr<Collection>> Collection::Open(const std::string& path,
                                                       const OpenOptions& open, Env* env) {
  VDB_ASSIGN_OR_RETURN(auto impl, CollectionImpl::Open(path, open, env));
  return std::unique_ptr<Collection>(new Collection(std::move(impl)));
}

const CollectionOptions& Collection::options() const { return impl_->options(); }

Status Collection::Upsert(VectorId id, const float* vector, const Metadata* metadata, uint64_t* lsn) {
  return impl_->Upsert({UpsertPoint{id, vector, metadata}}, lsn);
}
Status Collection::UpsertBatch(const std::vector<UpsertPoint>& points, uint64_t* last_lsn) {
  return impl_->Upsert(points, last_lsn);
}
Status Collection::UpsertBatch(const VectorId* ids, const float* vectors, size_t n,
                               const std::vector<Metadata>* metadata, uint64_t* last_lsn) {
  return impl_->UpsertRows(ids, vectors, n, metadata, last_lsn);
}
Status Collection::Remove(VectorId id, bool* found, uint64_t* lsn) {
  return impl_->Remove(&id, 1, nullptr, found, lsn);
}
Status Collection::RemoveBatch(const VectorId* ids, size_t n, size_t* removed) {
  return impl_->Remove(ids, n, removed, nullptr, nullptr);
}
Status Collection::BulkLoad(const VectorId* ids, const float* vectors, size_t n,
                            const std::vector<Metadata>* metadata) {
  return impl_->BulkLoad(ids, vectors, n, metadata);
}
StatusOr<std::unique_ptr<Filter>> Collection::ParseFilter(std::string_view json) const {
  return impl_->ParseFilter(json);
}
Status Collection::Search(const SearchRequest& req, std::vector<SearchHit>* out,
                          SearchExplain* explain) const {
  return impl_->Search(req, out, explain, /*parallel_segments=*/true);
}
Status Collection::SearchBatch(const float* queries, size_t nq, size_t k, const SearchParams& params,
                               const Filter* filter, std::vector<std::vector<SearchHit>>* out) const {
  return impl_->SearchBatch(queries, nq, k, params, filter, out);
}
Status Collection::Get(VectorId id, std::vector<float>* vector, Metadata* metadata) const {
  return impl_->Get(id, vector, metadata);
}
Status Collection::Flush() { return impl_->Flush(); }
void Collection::RequestFlush() { impl_->RequestFlush(); }
Status Collection::Compact(size_t target_segments) { return impl_->Compact(target_segments); }
void Collection::RequestCompaction(size_t target_segments) { impl_->RequestCompaction(target_segments); }
CollectionStats Collection::Stats() const { return impl_->Stats(); }
Status Collection::Close() { return impl_->Close(); }

}  // namespace vdb
