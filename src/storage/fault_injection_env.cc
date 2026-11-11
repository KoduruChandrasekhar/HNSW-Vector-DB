#include "storage/fault_injection_env.h"

#include <algorithm>

namespace vdb {
namespace {

class FaultWritableFile final : public WritableFile {
 public:
  FaultWritableFile(FaultInjectionEnv* env, std::string path, std::unique_ptr<WritableFile> base)
      : env_(env), path_(std::move(path)), base_(std::move(base)) {}
  using WritableFile::append;

  Status append(std::span<const std::byte> data) override {
    VDB_RETURN_IF_ERROR(env_->before_mutation("append"));
    if (env_->fail_writes()) return Status::IOError("injected write failure: " + path_);
    VDB_RETURN_IF_ERROR(base_->append(data));
    env_->on_append(path_, data.size());
    return Status::OK();
  }
  Status sync() override {
    VDB_RETURN_IF_ERROR(env_->before_mutation("sync"));
    VDB_RETURN_IF_ERROR(env_->on_sync(path_));
    return base_->sync();
  }
  Status close() override { return base_->close(); }
  uint64_t size() const override { return base_->size(); }

 private:
  FaultInjectionEnv* env_;
  std::string path_;
  std::unique_ptr<WritableFile> base_;
};

bool HasPrefix(const std::string& s, const std::string& prefix) {
  return s.size() > prefix.size() && s.compare(0, prefix.size(), prefix) == 0 &&
         s[prefix.size()] == '/';
}

}  // namespace

Status FaultInjectionEnv::before_mutation(const char* what) {
  if (!active_.load()) return Status::IOError(std::string("injected crash (") + what + ")");
  int64_t c = countdown_.load();
  if (c >= 0) {
    if (c == 0) {
      active_.store(false);
      return Status::IOError(std::string("injected crash (") + what + ")");
    }
    countdown_.store(c - 1);
  }
  ops_.fetch_add(1);
  return Status::OK();
}

void FaultInjectionEnv::on_append(const std::string& path, uint64_t n) {
  std::lock_guard<std::mutex> g(mu_);
  files_[path].size += n;
}

Status FaultInjectionEnv::on_sync(const std::string& path) {
  if (fail_sync_.load()) return Status::IOError("injected fsync failure: " + path);
  std::lock_guard<std::mutex> g(mu_);
  auto& f = files_[path];
  f.synced = f.size;
  syncs_.fetch_add(1);
  return Status::OK();
}

void FaultInjectionEnv::record_op(DirOp op) {
  std::lock_guard<std::mutex> g(mu_);
  pending_.push_back(std::move(op));
}

void FaultInjectionEnv::rekey_prefix(std::string from, std::string to) {
  // Caller holds mu_.
  std::vector<std::pair<std::string, FileState>> moved;
  for (auto it = files_.begin(); it != files_.end();) {
    if (it->first == from || HasPrefix(it->first, from)) {
      moved.emplace_back(to + it->first.substr(from.size()), it->second);
      it = files_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto& m : moved) files_[m.first] = m.second;
  // Pending namespace operations under a renamed directory move with it.
  auto rekey = [&](std::string& p) {
    if (p == from || HasPrefix(p, from)) p = to + p.substr(from.size());
  };
  for (auto& op : pending_) {
    rekey(op.path);
    if (op.kind == DirOp::kRename) rekey(op.from);
  }
}

StatusOr<std::unique_ptr<WritableFile>> FaultInjectionEnv::new_writable(const std::string& path) {
  VDB_RETURN_IF_ERROR(before_mutation("create"));
  bool existed = base_->exists(path);
  VDB_ASSIGN_OR_RETURN(auto f, base_->new_writable(path));
  {
    std::lock_guard<std::mutex> g(mu_);
    files_[path] = FileState{};
  }
  if (!existed) record_op(DirOp{DirOp::kCreateFile, path, "", false, ""});
  return std::unique_ptr<WritableFile>(std::make_unique<FaultWritableFile>(this, path, std::move(f)));
}

Status FaultInjectionEnv::create_dir(const std::string& dir) {
  VDB_RETURN_IF_ERROR(before_mutation("mkdir"));
  bool existed = base_->is_dir(dir);
  VDB_RETURN_IF_ERROR(base_->create_dir(dir));
  if (!existed) record_op(DirOp{DirOp::kCreateDir, dir, "", false, ""});
  return Status::OK();
}

Status FaultInjectionEnv::remove_file(const std::string& path) {
  VDB_RETURN_IF_ERROR(before_mutation("remove"));
  {
    std::lock_guard<std::mutex> g(mu_);
    files_.erase(path);
  }
  return base_->remove_file(path);
}

Status FaultInjectionEnv::remove_dir_all(const std::string& dir) {
  VDB_RETURN_IF_ERROR(before_mutation("remove dir"));
  {
    std::lock_guard<std::mutex> g(mu_);
    for (auto it = files_.begin(); it != files_.end();) {
      it = (it->first == dir || HasPrefix(it->first, dir)) ? files_.erase(it) : std::next(it);
    }
  }
  return base_->remove_dir_all(dir);
}

Status FaultInjectionEnv::rename(const std::string& from, const std::string& to) {
  VDB_RETURN_IF_ERROR(before_mutation("rename"));
  DirOp op{DirOp::kRename, to, from, false, ""};
  if (base_->exists(to) && !base_->is_dir(to)) {
    op.had_backup = true;
    VDB_RETURN_IF_ERROR(base_->read_file(to, &op.backup));
  }
  VDB_RETURN_IF_ERROR(base_->rename(from, to));
  {
    std::lock_guard<std::mutex> g(mu_);
    files_.erase(to);
    rekey_prefix(from, to);
    pending_.push_back(std::move(op));
  }
  return Status::OK();
}

Status FaultInjectionEnv::sync_dir(const std::string& dir) {
  VDB_RETURN_IF_ERROR(before_mutation("sync dir"));
  std::lock_guard<std::mutex> g(mu_);
  // Entries directly inside `dir` are now durable.
  pending_.erase(std::remove_if(pending_.begin(), pending_.end(),
                                [&](const DirOp& op) { return DirName(op.path) == dir; }),
                 pending_.end());
  return Status::OK();
}

Status FaultInjectionEnv::truncate(const std::string& path, uint64_t size) {
  VDB_RETURN_IF_ERROR(before_mutation("truncate"));
  VDB_RETURN_IF_ERROR(base_->truncate(path, size));
  std::lock_guard<std::mutex> g(mu_);
  auto it = files_.find(path);
  if (it != files_.end()) {
    it->second.size = std::min(it->second.size, size);
    it->second.synced = std::min(it->second.synced, size);
  }
  return Status::OK();
}

Status FaultInjectionEnv::drop_unsynced(bool tear) {
  std::lock_guard<std::mutex> g(mu_);
  // 1. Unsynced bytes vanish (or half of them survive, a torn write).
  for (auto& [path, st] : files_) {
    if (st.size <= st.synced || !base_->exists(path)) continue;
    uint64_t keep = tear ? st.synced + (st.size - st.synced) / 2 : st.synced;
    VDB_RETURN_IF_ERROR(base_->truncate(path, keep));
    st.size = keep;
  }
  // 2. Undo creates and renames that no directory sync made durable, newest first.
  for (auto it = pending_.rbegin(); it != pending_.rend(); ++it) {
    const DirOp op = *it;  // a copy: rekey_prefix below rewrites the entries in place
    switch (op.kind) {
      case DirOp::kCreateFile:
      case DirOp::kCreateDir:
        if (base_->exists(op.path)) VDB_RETURN_IF_ERROR(base_->remove_dir_all(op.path));
        for (auto f = files_.begin(); f != files_.end();) {
          f = (f->first == op.path || HasPrefix(f->first, op.path)) ? files_.erase(f) : std::next(f);
        }
        break;
      case DirOp::kRename:
        if (base_->exists(op.path)) {
          VDB_RETURN_IF_ERROR(base_->rename(op.path, op.from));
          rekey_prefix(op.path, op.from);
        }
        if (op.had_backup) {
          VDB_ASSIGN_OR_RETURN(auto f, base_->new_writable(op.path));
          VDB_RETURN_IF_ERROR(f->append(op.backup));
          VDB_RETURN_IF_ERROR(f->close());
        }
        break;
    }
  }
  pending_.clear();
  for (auto& [path, st] : files_) st.synced = st.size;
  return Status::OK();
}

}  // namespace vdb
