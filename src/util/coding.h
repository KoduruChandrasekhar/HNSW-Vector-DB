#pragma once

// Little-endian fixed-width encoding helpers for log records, manifests and files.

#include <bit>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace vdb {

static_assert(std::endian::native == std::endian::little,
              "vdb's on-disk formats assume a little-endian host");

inline void PutU8(std::string* dst, uint8_t v) { dst->push_back(static_cast<char>(v)); }
inline void PutU16(std::string* dst, uint16_t v) { dst->append(reinterpret_cast<const char*>(&v), 2); }
inline void PutU32(std::string* dst, uint32_t v) { dst->append(reinterpret_cast<const char*>(&v), 4); }
inline void PutU64(std::string* dst, uint64_t v) { dst->append(reinterpret_cast<const char*>(&v), 8); }
inline void PutI64(std::string* dst, int64_t v) { dst->append(reinterpret_cast<const char*>(&v), 8); }
inline void PutF32(std::string* dst, float v) { dst->append(reinterpret_cast<const char*>(&v), 4); }
inline void PutF64(std::string* dst, double v) { dst->append(reinterpret_cast<const char*>(&v), 8); }
inline void PutBytes(std::string* dst, const void* p, size_t n) {
  dst->append(static_cast<const char*>(p), n);
}
inline void PutString(std::string* dst, std::string_view s) {
  PutU32(dst, static_cast<uint32_t>(s.size()));
  dst->append(s.data(), s.size());
}

inline void EncodeU32(char* dst, uint32_t v) { std::memcpy(dst, &v, 4); }
inline void EncodeU64(char* dst, uint64_t v) { std::memcpy(dst, &v, 8); }
inline uint32_t DecodeU32(const char* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
inline uint64_t DecodeU64(const char* p) {
  uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}

// Bounds-checked sequential reader. Every getter returns false (and the reader stays
// failed) once the input runs out, so parsers never read past the end.
class ByteReader {
 public:
  explicit ByteReader(std::string_view data) : p_(data.data()), end_(data.data() + data.size()) {}

  size_t remaining() const { return static_cast<size_t>(end_ - p_); }
  bool ok() const { return ok_; }
  const char* position() const { return p_; }

  bool GetU8(uint8_t* v) { return GetRaw(v, 1); }
  bool GetU16(uint16_t* v) { return GetRaw(v, 2); }
  bool GetU32(uint32_t* v) { return GetRaw(v, 4); }
  bool GetU64(uint64_t* v) { return GetRaw(v, 8); }
  bool GetI64(int64_t* v) { return GetRaw(v, 8); }
  bool GetF32(float* v) { return GetRaw(v, 4); }
  bool GetF64(double* v) { return GetRaw(v, 8); }
  bool GetBytes(size_t n, std::string_view* out) {
    if (!ok_ || remaining() < n) return Fail();
    *out = std::string_view(p_, n);
    p_ += n;
    return true;
  }
  bool GetString(std::string* out) {
    uint32_t n = 0;
    std::string_view sv;
    if (!GetU32(&n) || !GetBytes(n, &sv)) return false;
    out->assign(sv.data(), sv.size());
    return true;
  }
  bool Skip(size_t n) {
    if (!ok_ || remaining() < n) return Fail();
    p_ += n;
    return true;
  }

 private:
  bool GetRaw(void* v, size_t n) {
    if (!ok_ || remaining() < n) return Fail();
    std::memcpy(v, p_, n);
    p_ += n;
    return true;
  }
  bool Fail() {
    ok_ = false;
    return false;
  }

  const char* p_;
  const char* end_;
  bool ok_ = true;
};

}  // namespace vdb
