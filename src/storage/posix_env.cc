#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include "storage/env.h"

namespace vdb {
namespace {

Status ErrnoStatus(const std::string& what, const std::string& path, int err) {
  return Status::IOError(what + " " + path + ": " + std::strerror(err));
}

class PosixWritableFile final : public WritableFile {
 public:
  PosixWritableFile(std::string path, int fd) : path_(std::move(path)), fd_(fd) {}
  ~PosixWritableFile() override {
    if (fd_ >= 0) ::close(fd_);
  }
  using WritableFile::append;

  Status append(std::span<const std::byte> data) override {
    const auto* p = reinterpret_cast<const char*>(data.data());
    size_t left = data.size();
    while (left > 0) {
      ssize_t n = ::write(fd_, p, left);
      if (n < 0) {
        if (errno == EINTR) continue;
        return ErrnoStatus("write", path_, errno);
      }
      p += n;
      left -= static_cast<size_t>(n);
      size_ += static_cast<uint64_t>(n);
    }
    return Status::OK();
  }

  Status sync() override {
#if defined(__APPLE__)
    if (::fcntl(fd_, F_FULLFSYNC) != 0) return ErrnoStatus("fsync", path_, errno);
#elif defined(__linux__)
    if (::fdatasync(fd_) != 0) return ErrnoStatus("fdatasync", path_, errno);
#else
    if (::fsync(fd_) != 0) return ErrnoStatus("fsync", path_, errno);
#endif
    return Status::OK();
  }

  Status close() override {
    if (fd_ < 0) return Status::OK();
    int r = ::close(fd_);
    fd_ = -1;
    if (r != 0) return ErrnoStatus("close", path_, errno);
    return Status::OK();
  }

  uint64_t size() const override { return size_; }

 private:
  std::string path_;
  int fd_;
  uint64_t size_ = 0;
};

class PosixMappedRegion final : public MappedRegion {
 public:
  PosixMappedRegion(void* p, size_t n) : p_(p), n_(n) {}
  ~PosixMappedRegion() override {
    if (p_ != nullptr) ::munmap(p_, n_);
  }
  const std::byte* data() const override { return static_cast<const std::byte*>(p_); }
  size_t size() const override { return n_; }
  void advise(Access a) override {
    if (p_ == nullptr) return;
    int adv = a == Access::kRandom ? MADV_RANDOM
              : a == Access::kSequential ? MADV_SEQUENTIAL
                                         : MADV_NORMAL;
    ::madvise(p_, n_, adv);
  }

 private:
  void* p_;
  size_t n_;
};

class PosixEnv final : public Env {
 public:
  StatusOr<std::unique_ptr<WritableFile>> new_writable(const std::string& path) override {
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return ErrnoStatus("open", path, errno);
    return std::unique_ptr<WritableFile>(std::make_unique<PosixWritableFile>(path, fd));
  }

  Status read_file(const std::string& path, std::string* out) override {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return ErrnoStatus("open", path, errno);
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
      int e = errno;
      ::close(fd);
      return ErrnoStatus("stat", path, e);
    }
    out->resize(static_cast<size_t>(st.st_size));
    size_t got = 0;
    while (got < out->size()) {
      ssize_t n = ::read(fd, out->data() + got, out->size() - got);
      if (n < 0) {
        if (errno == EINTR) continue;
        int e = errno;
        ::close(fd);
        return ErrnoStatus("read", path, e);
      }
      if (n == 0) break;
      got += static_cast<size_t>(n);
    }
    out->resize(got);
    ::close(fd);
    return Status::OK();
  }

  StatusOr<std::unique_ptr<MappedRegion>> map_file(const std::string& path) override {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return ErrnoStatus("open", path, errno);
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
      int e = errno;
      ::close(fd);
      return ErrnoStatus("stat", path, e);
    }
    auto n = static_cast<size_t>(st.st_size);
    void* p = nullptr;
    if (n > 0) {
      p = ::mmap(nullptr, n, PROT_READ, MAP_SHARED, fd, 0);
      if (p == MAP_FAILED) {
        int e = errno;
        ::close(fd);
        return ErrnoStatus("mmap", path, e);
      }
    }
    ::close(fd);  // the mapping stays valid
    return std::unique_ptr<MappedRegion>(std::make_unique<PosixMappedRegion>(p, n));
  }

  bool exists(const std::string& path) override { return ::access(path.c_str(), F_OK) == 0; }

  bool is_dir(const std::string& path) override {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
  }

  StatusOr<uint64_t> file_size(const std::string& path) override {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) return ErrnoStatus("stat", path, errno);
    return static_cast<uint64_t>(st.st_size);
  }

  Status list_dir(const std::string& dir, std::vector<std::string>* names) override {
    names->clear();
    DIR* d = ::opendir(dir.c_str());
    if (d == nullptr) return ErrnoStatus("opendir", dir, errno);
    while (struct dirent* e = ::readdir(d)) {
      std::string n = e->d_name;
      if (n != "." && n != "..") names->push_back(std::move(n));
    }
    ::closedir(d);
    return Status::OK();
  }

  Status create_dir(const std::string& dir) override {
    if (::mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) return ErrnoStatus("mkdir", dir, errno);
    return Status::OK();
  }

  Status remove_file(const std::string& path) override {
    if (::unlink(path.c_str()) != 0 && errno != ENOENT) return ErrnoStatus("unlink", path, errno);
    return Status::OK();
  }

  Status remove_dir_all(const std::string& dir) override {
    if (!is_dir(dir)) return remove_file(dir);
    std::vector<std::string> names;
    VDB_RETURN_IF_ERROR(list_dir(dir, &names));
    for (const auto& n : names) VDB_RETURN_IF_ERROR(remove_dir_all(JoinPath(dir, n)));
    if (::rmdir(dir.c_str()) != 0 && errno != ENOENT) return ErrnoStatus("rmdir", dir, errno);
    return Status::OK();
  }

  Status rename(const std::string& from, const std::string& to) override {
    if (::rename(from.c_str(), to.c_str()) != 0) return ErrnoStatus("rename", from + " -> " + to, errno);
    return Status::OK();
  }

  Status sync_dir(const std::string& dir) override {
    int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return ErrnoStatus("open dir", dir, errno);
    int r = ::fsync(fd);
    int e = errno;
    ::close(fd);
    if (r != 0) return ErrnoStatus("fsync dir", dir, e);
    return Status::OK();
  }

  Status truncate(const std::string& path, uint64_t size) override {
    if (::truncate(path.c_str(), static_cast<off_t>(size)) != 0) return ErrnoStatus("truncate", path, errno);
    return Status::OK();
  }
};

}  // namespace

Env* Env::Default() {
  static PosixEnv* env = new PosixEnv();  // never destroyed
  return env;
}

std::string JoinPath(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  if (a.back() == '/') return a + b;
  return a + "/" + b;
}

std::string DirName(const std::string& path) {
  size_t p = path.find_last_of('/');
  if (p == std::string::npos) return ".";
  if (p == 0) return "/";
  return path.substr(0, p);
}

std::string BaseName(const std::string& path) {
  size_t p = path.find_last_of('/');
  return p == std::string::npos ? path : path.substr(p + 1);
}

Status WriteFileSync(Env* env, const std::string& path, std::string_view data) {
  VDB_ASSIGN_OR_RETURN(auto f, env->new_writable(path));
  VDB_RETURN_IF_ERROR(f->append(data));
  VDB_RETURN_IF_ERROR(f->sync());
  return f->close();
}

Status CreateDirs(Env* env, const std::string& dir) {
  if (dir.empty() || env->is_dir(dir)) return Status::OK();
  std::string parent = DirName(dir);
  if (parent != dir && parent != "." && parent != "/") VDB_RETURN_IF_ERROR(CreateDirs(env, parent));
  return env->create_dir(dir);
}

}  // namespace vdb
