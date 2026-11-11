// The WAL reader must never crash or read out of bounds, whatever the log contains.
#include <cstddef>
#include <cstdint>
#include <string>

#include "storage/wal.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // Half the time, prefix a valid header so the fuzzer explores record parsing.
  std::string log(reinterpret_cast<const char*>(data), size);
  if (size > 0 && (data[0] & 1)) log = vdb::WalFileHeader() + log.substr(1);
  vdb::WalParseResult res;
  vdb::ParseWal(
      log,
      [](const vdb::WalRecord& r) {
        vdb::VectorId id;
        uint32_t dim;
        const char* vec;
        std::string_view meta;
        if (r.type == vdb::WalRecordType::kUpsert) {
          vdb::DecodeUpsertPayload(r.payload, &id, &dim, &vec, &meta);
        } else {
          vdb::DecodeDeletePayload(r.payload, &id);
        }
        return vdb::Status::OK();
      },
      &res);
  if (res.valid_bytes > log.size()) __builtin_trap();
  return 0;
}
