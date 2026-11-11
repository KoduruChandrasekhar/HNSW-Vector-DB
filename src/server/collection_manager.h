#pragma once

// Named collections under <data_dir>/collections/<name>.

#include <map>
#include <memory>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

#include "vdb/collection.h"

namespace vdb {

class CollectionManager {
 public:
  CollectionManager(std::string data_dir, OpenOptions open);
  ~CollectionManager();

  Status OpenAll();
  StatusOr<std::shared_ptr<Collection>> Create(const std::string& name, const CollectionOptions& options);
  std::shared_ptr<Collection> Get(const std::string& name) const;
  Status Drop(const std::string& name);
  std::vector<std::pair<std::string, std::shared_ptr<Collection>>> List() const;
  Status CloseAll();

  static bool ValidName(const std::string& name);

 private:
  std::string path(const std::string& name) const;

  std::string root_;
  OpenOptions open_;
  mutable std::shared_mutex mu_;
  std::map<std::string, std::shared_ptr<Collection>> cols_;
};

}  // namespace vdb
