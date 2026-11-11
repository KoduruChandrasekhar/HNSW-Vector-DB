#pragma once

#include <cstdint>

namespace vdb {

// SplitMix64: a fast, well-mixed 64-bit hash, used to derive per-node random levels
// and per-thread seeds so results do not depend on thread scheduling.
inline uint64_t SplitMix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// Uniform double in [0, 1) from 64 random bits.
inline double ToUnitDouble(uint64_t bits) {
  return static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);
}

}  // namespace vdb
