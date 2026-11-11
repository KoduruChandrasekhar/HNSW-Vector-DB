#pragma once

// All file I/O goes through Env, so tests can swap in FaultInjectionEnv to lose unsynced
// data, fail writes or syncs on demand, or tear the last write in half.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "vdb/status.h"

namespace vdb {

enum class Access : uint8_t { kNormal, kRandom, kSequential };

class WritableFile {
 public:
  virtual ~WritableFile() = default;
  virtual Status append(std::span<const std::byte> data) = 0;
  Status append(std::string_view s) {
    return append(std::as_bytes(std::span<const char>(s.data(), s.size())));
  }
  // fdatasync on Linux; fcntl(F_FULLFSYNC) on macOS, where plain fsync does not flush
  // the drive's own cache.
  virtual Status sync() = 0;
  virtual Status close() = 0;
  virtual uint64_t size() const = 0;
};

// A read-only memory mapping of a whole file.
class MappedRegion {
 public:
  virtual ~MappedRegion() = default;
  virtual const std::byte* data() const = 0;
  virtual size_t size() const = 0;
  virtual void advise(Access) {}
};

class Env {
 public:
  virtual ~Env() = default;
  static Env* Default();  // the process-wide PosixEnv

  // Creates (or truncates) a file for writing.
  virtual StatusOr<std::unique_ptr<WritableFile>> new_writable(const std::string& path) = 0;
  virtual Status read_file(const std::string& path, std::string* out) = 0;
  virtual StatusOr<std::unique_ptr<MappedRegion>> map_file(const std::string& path) = 0;
  virtual bool exists(const std::string& path) = 0;
  virtual bool is_dir(const std::string& path) = 0;
  virtual StatusOr<uint64_t> file_size(const std::string& path) = 0;
  virtual Status list_dir(const std::string& dir, std::vector<std::string>* names) = 0;
  virtual Status create_dir(const std::string& dir) = 0;  // OK if it exists
  virtual Status remove_file(const std::string& path) = 0;
  virtual Status remove_dir_all(const std::string& dir) = 0;
  virtual Status rename(const std::string& from, const std::string& to) = 0;
  // Makes creations, renames and removals inside `dir` durable.
  virtual Status sync_dir(const std::string& dir) = 0;
  virtual Status truncate(const std::string& path, uint64_t size) = 0;
};

std::string JoinPath(const std::string& a, const std::string& b);
std::string DirName(const std::string& path);
std::string BaseName(const std::string& path);

// Creates `path`, writes `data`, fsyncs and closes it.
Status WriteFileSync(Env* env, const std::string& path, std::string_view data);
// Creates the directory and all missing parents.
Status CreateDirs(Env* env, const std::string& dir);

}  // namespace vdb
