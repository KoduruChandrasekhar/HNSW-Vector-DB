#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "storage/env.h"

namespace vdb {

// An Env for crash tests. It forwards to a real Env but remembers, for every file it
// wrote, how much was synced, and every create or rename not yet made durable by a
// directory sync. drop_unsynced() then simulates power loss: unsynced bytes vanish (or,
// with tear, half of them do) and undurable creates and renames are undone.
//
// fail_after(n) lets n more mutating operations succeed and then "crashes": that
// operation and everything after it fail until set_active(true). Stepping n through
// every value hits a crash between every step of a flush.
class FaultInjectionEnv final : public Env {
 public:
  explicit FaultInjectionEnv(Env* base = Env::Default()) : base_(base) {}

  void set_active(bool active) { active_.store(active); }
  bool active() const { return active_.load(); }
  void fail_after(int64_t ops) { countdown_.store(ops); }
  void set_fail_sync(bool f) { fail_sync_.store(f); }
  void set_fail_writes(bool f) { fail_writes_.store(f); }
  uint64_t mutating_ops() const { return ops_.load(); }
  uint64_t syncs() const { return syncs_.load(); }

  // Simulated power loss. Call after the collection using this env is closed.
  Status drop_unsynced(bool tear = false);

  // Env
  StatusOr<std::unique_ptr<WritableFile>> new_writable(const std::string& path) override;
  Status read_file(const std::string& path, std::string* out) override {
    return base_->read_file(path, out);
  }
  StatusOr<std::unique_ptr<MappedRegion>> map_file(const std::string& path) override {
    return base_->map_file(path);
  }
  bool exists(const std::string& path) override { return base_->exists(path); }
  bool is_dir(const std::string& path) override { return base_->is_dir(path); }
  StatusOr<uint64_t> file_size(const std::string& path) override { return base_->file_size(path); }
  Status list_dir(const std::string& dir, std::vector<std::string>* names) override {
    return base_->list_dir(dir, names);
  }
  Status create_dir(const std::string& dir) override;
  Status remove_file(const std::string& path) override;
  Status remove_dir_all(const std::string& dir) override;
  Status rename(const std::string& from, const std::string& to) override;
  Status sync_dir(const std::string& dir) override;
  Status truncate(const std::string& path, uint64_t size) override;

  // Called by FaultWritableFile.
  Status before_mutation(const char* what);
  void on_append(const std::string& path, uint64_t n);
  Status on_sync(const std::string& path);
  bool fail_writes() const { return fail_writes_.load(); }
  Env* base() const { return base_; }

 private:
  struct FileState {
    uint64_t size = 0;
    uint64_t synced = 0;
  };
  struct DirOp {
    enum Kind { kCreateFile, kCreateDir, kRename } kind;
    std::string path;  // created path, or rename target
    std::string from;  // rename source
    bool had_backup = false;
    std::string backup;  // previous content of an overwritten rename target
  };

  void record_op(DirOp op);
  void rekey_prefix(std::string from, std::string to);  // by value: callers pass aliases

  Env* base_;
  std::atomic<bool> active_{true};
  std::atomic<int64_t> countdown_{-1};
  std::atomic<bool> fail_sync_{false};
  std::atomic<bool> fail_writes_{false};
  std::atomic<uint64_t> ops_{0};
  std::atomic<uint64_t> syncs_{0};

  std::mutex mu_;
  std::map<std::string, FileState> files_;
  std::vector<DirOp> pending_;  // in order; undone in reverse
};

}  // namespace vdb
