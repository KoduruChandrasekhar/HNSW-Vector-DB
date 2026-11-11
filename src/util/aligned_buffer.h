#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>

namespace vdb {

// A zero-initialized, 64-byte aligned heap buffer (cache-line and AVX-512 friendly).
class AlignedBuffer {
 public:
  static constexpr size_t kAlign = 64;

  AlignedBuffer() = default;
  explicit AlignedBuffer(size_t bytes) { reset(bytes); }

  void reset(size_t bytes) {
    size_ = bytes;
    if (bytes == 0) {
      ptr_.reset();
      return;
    }
    size_t rounded = (bytes + kAlign - 1) / kAlign * kAlign;  // aligned_alloc requirement
    void* p = std::aligned_alloc(kAlign, rounded);
    if (p == nullptr) throw std::bad_alloc();
    std::memset(p, 0, rounded);
    ptr_.reset(static_cast<std::byte*>(p));
  }

  std::byte* data() { return ptr_.get(); }
  const std::byte* data() const { return ptr_.get(); }
  float* floats() { return reinterpret_cast<float*>(ptr_.get()); }
  const float* floats() const { return reinterpret_cast<const float*>(ptr_.get()); }
  template <class T>
  T* as() {
    return reinterpret_cast<T*>(ptr_.get());
  }
  template <class T>
  const T* as() const {
    return reinterpret_cast<const T*>(ptr_.get());
  }
  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

 private:
  struct Free {
    void operator()(std::byte* p) const { std::free(p); }
  };
  std::unique_ptr<std::byte, Free> ptr_;
  size_t size_ = 0;
};

// Rows of padded_dim floats; the index layer only ever sees this view.
struct VectorView {
  const float* data = nullptr;
  size_t rows = 0;
  size_t dim = 0;         // real dimensions
  size_t padded_dim = 0;  // storage stride in floats (multiple of 16)

  const float* row(size_t i) const { return data + i * padded_dim; }
};

}  // namespace vdb
