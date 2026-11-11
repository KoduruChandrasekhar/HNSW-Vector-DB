#pragma once

// Write-ahead log.
//
// File: [magic "VDBWAL01"][u32 version][u32 reserved], then records. All integers are
// little-endian. Record layout:
//   offset 0  u32  CRC32C of bytes 4 to the end of the record
//   offset 4  u32  payload length
//   offset 8  u8   type: 1 = UPSERT, 2 = DELETE
//   offset 9  u64  LSN (log sequence number, +1 per record)
//   offset 17      payload
// UPSERT payload: id u64, dim u32, vector dim x f32, metadata length u32, metadata bytes.
// DELETE payload: id u64.

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

#include "storage/env.h"
#include "vdb/types.h"

namespace vdb {

enum class WalRecordType : uint8_t { kUpsert = 1, kDelete = 2 };

struct WalRecord {
  WalRecordType type;
  uint64_t lsn;
  std::string_view payload;
};

constexpr size_t kWalFileHeaderSize = 16;
constexpr size_t kWalRecordHeaderSize = 17;
constexpr std::string_view kWalMagic = "VDBWAL01";

std::string WalFileHeader();
void AppendWalRecord(std::string* dst, WalRecordType type, uint64_t lsn, std::string_view payload);

void EncodeUpsertPayload(std::string* dst, VectorId id, const float* vec, size_t dim,
                         std::string_view meta);
bool DecodeUpsertPayload(std::string_view p, VectorId* id, uint32_t* dim, const char** vec,
                         std::string_view* meta);
void EncodeDeletePayload(std::string* dst, VectorId id);
bool DecodeDeletePayload(std::string_view p, VectorId* id);

struct WalParseResult {
  uint64_t valid_bytes = 0;  // end of the last good record
  uint64_t records = 0;
  bool torn_tail = false;  // a bad or truncated record at the very end
};

// Parses a whole log file, calling cb for each good record in order. A truncated record
// or bad CRC at the very end is a torn write and is reported in `result` (truncate the
// file at valid_bytes). A bad record followed by good ones is real corruption.
Status ParseWal(std::string_view data, const std::function<Status(const WalRecord&)>& cb,
                WalParseResult* result);

std::string WalFileName(uint64_t number);  // "000014.log"
bool ParseWalFileName(std::string_view name, uint64_t* number);

// Appends pre-encoded record batches to one log file. Thread-safe. Once a write or sync
// fails the writer is failed for good: after a failed fsync the kernel may already have
// dropped the dirty pages, so retrying could silently lose data ("fsyncgate").
class WalWriter {
 public:
  static StatusOr<std::unique_ptr<WalWriter>> Create(Env* env, const std::string& path,
                                                     uint64_t number);
  Status append(std::string_view records);
  Status sync();
  Status close();
  uint64_t number() const { return number_; }
  uint64_t bytes() const;
  bool failed() const;
  const std::string& path() const { return path_; }

 private:
  WalWriter(std::unique_ptr<WritableFile> f, std::string path, uint64_t number)
      : file_(std::move(f)), path_(std::move(path)), number_(number) {}

  mutable std::mutex mu_;
  std::unique_ptr<WritableFile> file_;
  std::string path_;
  uint64_t number_;
  uint64_t unsynced_ = 0;
  bool failed_ = false;
};

}  // namespace vdb
