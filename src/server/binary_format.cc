#include "server/binary_format.h"

#include <cmath>
#include <cstring>

#include "util/coding.h"

namespace vdb {

Status ParseBinaryBatch(std::string_view body, size_t expected_dim, BinaryBatch* out) {
  if (body.size() < kBinaryHeaderSize || body.substr(0, 4) != "VDBB") {
    return Status::InvalidArgument("binary body must start with the VDBB header");
  }
  uint32_t version = DecodeU32(body.data() + 4);
  uint32_t count = DecodeU32(body.data() + 8);
  uint32_t dim = DecodeU32(body.data() + 12);
  if (version != 1) return Status::InvalidArgument("unsupported binary format version");
  if (dim == 0 || (expected_dim != 0 && dim != expected_dim)) {
    return Status::InvalidArgument("wrong dimension: got " + std::to_string(dim) + ", expected " +
                                   std::to_string(expected_dim));
  }
  // Check the size with 64-bit math so a huge count cannot overflow.
  uint64_t need = kBinaryHeaderSize + uint64_t{count} * 8 + uint64_t{count} * dim * 4;
  if (need != body.size()) {
    return Status::InvalidArgument("binary body size does not match its header (expected " +
                                   std::to_string(need) + " bytes, got " + std::to_string(body.size()) + ")");
  }
  out->dim = dim;
  out->ids.resize(count);
  out->vectors.resize(size_t{count} * dim);
  const char* p = body.data() + kBinaryHeaderSize;
  if (count > 0) {
    std::memcpy(out->ids.data(), p, size_t{count} * 8);
    std::memcpy(out->vectors.data(), p + size_t{count} * 8, size_t{count} * dim * 4);
  }
  for (float v : out->vectors) {
    if (!std::isfinite(v)) return Status::InvalidArgument("vector has a NaN or infinite value");
  }
  return Status::OK();
}

std::string EncodeBinaryBatch(const VectorId* ids, const float* vectors, size_t n, size_t dim) {
  std::string out = "VDBB";
  PutU32(&out, 1);
  PutU32(&out, static_cast<uint32_t>(n));
  PutU32(&out, static_cast<uint32_t>(dim));
  PutBytes(&out, ids, n * sizeof(VectorId));
  PutBytes(&out, vectors, n * dim * sizeof(float));
  return out;
}

}  // namespace vdb
