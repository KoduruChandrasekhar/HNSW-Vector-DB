#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace vdb::crc32c {

// CRC32C (Castagnoli). Extend() continues a CRC returned by a previous call, so
// Extend(Value(a), b) == Value(a + b). Uses SSE4.2 when the CPU has it.
uint32_t Extend(uint32_t crc, const void* data, size_t n);
inline uint32_t Value(const void* data, size_t n) { return Extend(0, data, n); }
inline uint32_t Value(std::string_view s) { return Extend(0, s.data(), s.size()); }

// For tests: the portable table-driven implementation.
uint32_t ExtendPortable(uint32_t crc, const void* data, size_t n);
bool HardwareAccelerated();

}  // namespace vdb::crc32c
