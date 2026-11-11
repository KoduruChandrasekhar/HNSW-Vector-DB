#include "util/crc32c.h"

#include <cstring>

#if defined(__x86_64__)
#include <nmmintrin.h>
#endif

namespace vdb::crc32c {
namespace {

constexpr uint32_t kPoly = 0x82F63B78u;  // reflected Castagnoli polynomial

struct Tables {
  uint32_t t[8][256];
};

constexpr Tables MakeTables() {
  Tables tb{};
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t crc = i;
    for (int b = 0; b < 8; ++b) crc = (crc >> 1) ^ (kPoly & (0u - (crc & 1u)));
    tb.t[0][i] = crc;
  }
  for (uint32_t i = 0; i < 256; ++i) {
    for (int k = 1; k < 8; ++k) {
      tb.t[k][i] = (tb.t[k - 1][i] >> 8) ^ tb.t[0][tb.t[k - 1][i] & 0xffu];
    }
  }
  return tb;
}

constexpr Tables kTables = MakeTables();

#if defined(__x86_64__)
__attribute__((target("sse4.2"))) uint32_t ExtendHardware(uint32_t crc, const uint8_t* p,
                                                          size_t n) {
  uint64_t l = crc ^ 0xffffffffu;
  while (n >= 8) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    l = _mm_crc32_u64(l, v);
    p += 8;
    n -= 8;
  }
  auto l32 = static_cast<uint32_t>(l);
  while (n > 0) {
    l32 = _mm_crc32_u8(l32, *p++);
    --n;
  }
  return l32 ^ 0xffffffffu;
}

bool DetectHardware() { return __builtin_cpu_supports("sse4.2"); }
#else
bool DetectHardware() { return false; }
#endif

const bool kHardware = DetectHardware();

}  // namespace

uint32_t ExtendPortable(uint32_t crc, const void* data, size_t n) {
  const auto* p = static_cast<const uint8_t*>(data);
  uint32_t l = crc ^ 0xffffffffu;
  const auto& t = kTables.t;
  while (n >= 8) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    v ^= l;
    l = t[7][v & 0xff] ^ t[6][(v >> 8) & 0xff] ^ t[5][(v >> 16) & 0xff] ^
        t[4][(v >> 24) & 0xff] ^ t[3][(v >> 32) & 0xff] ^ t[2][(v >> 40) & 0xff] ^
        t[1][(v >> 48) & 0xff] ^ t[0][v >> 56];
    p += 8;
    n -= 8;
  }
  while (n > 0) {
    l = t[0][(l ^ *p++) & 0xffu] ^ (l >> 8);
    --n;
  }
  return l ^ 0xffffffffu;
}

uint32_t Extend(uint32_t crc, const void* data, size_t n) {
#if defined(__x86_64__)
  if (kHardware) return ExtendHardware(crc, static_cast<const uint8_t*>(data), n);
#endif
  return ExtendPortable(crc, data, n);
}

bool HardwareAccelerated() { return kHardware; }

}  // namespace vdb::crc32c
