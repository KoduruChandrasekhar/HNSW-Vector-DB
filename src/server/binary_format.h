#pragma once

// Binary body for /upsert-binary and /bulk-load (all little-endian):
//   "VDBB" | u32 version (1) | u32 count | u32 dim | count x u64 ids | count x dim x f32
// JSON float parsing is slow; this format is read with a few memcpys.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "vdb/status.h"
#include "vdb/types.h"

namespace vdb {

struct BinaryBatch {
  uint32_t dim = 0;
  std::vector<VectorId> ids;
  std::vector<float> vectors;  // count * dim
};

constexpr size_t kBinaryHeaderSize = 16;

// expected_dim == 0 accepts any dimension.
Status ParseBinaryBatch(std::string_view body, size_t expected_dim, BinaryBatch* out);
std::string EncodeBinaryBatch(const VectorId* ids, const float* vectors, size_t n, size_t dim);

}  // namespace vdb
