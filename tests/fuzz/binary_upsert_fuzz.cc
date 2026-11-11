// The binary upsert / bulk-load body parser must reject malformed sizes without
// allocating huge buffers or reading past the end.
#include <cstddef>
#include <cstdint>
#include <string>

#include "server/binary_format.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  std::string body(reinterpret_cast<const char*>(data), size);
  if (size > 0 && (data[0] & 1)) body = "VDBB" + body.substr(1);
  vdb::BinaryBatch b;
  if (vdb::ParseBinaryBatch(body, 0, &b).ok()) {
    if (b.vectors.size() != b.ids.size() * b.dim) __builtin_trap();
  }
  return 0;
}
