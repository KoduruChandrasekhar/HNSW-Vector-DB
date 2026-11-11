#include "server/collection_manager.h"

#include <mutex>

#include "storage/env.h"

namespace vdb {

CollectionManager::CollectionManager(std::string data_dir, OpenOptions open)
    : root_(JoinPath(data_dir, "collections")), open_(open) {}

CollectionManager::~CollectionManager() { CloseAll(); }

bool CollectionManager::ValidName(const std::string& name) {
  if (name.empty() || name.size() > 64) return false;
  for (char c : name) {
    bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
    if (!ok) return false;
  }
  return true;
}

std::string CollectionManager::path(const std::string& name) const { return JoinPath(root_, name); }

Status CollectionManager::OpenAll() {
  Env* env = Env::Default();
  VDB_RETURN_IF_ERROR(CreateDirs(env, root_));
  std::vector<std::string> names;
  VDB_RETURN_IF_ERROR(env->list_dir(root_, &names));
  std::unique_lock<std::shared_mutex> g(mu_);
  for (const auto& n : names) {
    if (!ValidName(n) || !env->exists(JoinPath(path(n), "CURRENT"))) continue;
    auto c = Collection::Open(path(n), open_);
    if (!c.ok()) return Status(c.status().code(), "collection " + n + ": " + c.status().message());
    cols_[n] = std::shared_ptr<Collection>(std::move(c).value());
  }
  return Status::OK();
}

StatusOr<std::shared_ptr<Collection>> CollectionManager::Create(const std::string& name,
                                                                const CollectionOptions& options) {
  if (!ValidName(name)) return Status::InvalidArgument("collection names use [A-Za-z0-9_-], 1 to 64 characters");
  std::unique_lock<std::shared_mutex> g(mu_);
  if (cols_.count(name) > 0) return Status::AlreadyExists("collection " + name + " already exists");
  auto c = Collection::Create(path(name), options, open_);
  if (!c.ok()) return c.status();
  auto shared = std::shared_ptr<Collection>(std::move(c).value());
  cols_[name] = shared;
  return shared;
}

std::shared_ptr<Collection> CollectionManager::Get(const std::string& name) const {
  std::shared_lock<std::shared_mutex> g(mu_);
  auto it = cols_.find(name);
  return it == cols_.end() ? nullptr : it->second;
}

Status CollectionManager::Drop(const std::string& name) {
  std::shared_ptr<Collection> c;
  {
    std::unique_lock<std::shared_mutex> g(mu_);
    auto it = cols_.find(name);
    if (it == cols_.end()) return Status::NotFound("collection " + name + " not found");
    c = std::move(it->second);
    cols_.erase(it);
  }
  c->Close();
  c.reset();  // in-flight requests holding a reference finish first
  return Env::Default()->remove_dir_all(path(name));
}

std::vector<std::pair<std::string, std::shared_ptr<Collection>>> CollectionManager::List() const {
  std::shared_lock<std::shared_mutex> g(mu_);
  return {cols_.begin(), cols_.end()};
}

Status CollectionManager::CloseAll() {
  std::unique_lock<std::shared_mutex> g(mu_);
  Status first;
  for (auto& [name, c] : cols_) {
    Status s = c->Close();
    if (!s.ok() && first.ok()) first = s;
  }
  cols_.clear();
  return first;
}

}  // namespace vdb
