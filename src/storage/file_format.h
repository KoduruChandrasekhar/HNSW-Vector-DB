#pragma once

// Small checksummed files (ids, metadata, delete sets):
//   [magic 8][u32 version][u32 crc32c(payload)][u64 payload length][payload]

#include <cstdint>
#include <string>
#include <string_view>

#include "storage/env.h"

namespace vdb {

constexpr size_t kBlobHeaderSize = 24;

std::string EncodeBlob(std::string_view magic, std::string_view payload);
Status DecodeBlob(std::string_view file, std::string_view magic, std::string_view* payload);

// Writes and fsyncs a blob file; reports the file size and payload CRC for the manifest.
Status WriteBlobFile(Env* env, const std::string& path, std::string_view magic,
                     std::string_view payload, uint64_t* file_size = nullptr,
                     uint32_t* crc = nullptr);
// Reads a blob file and checks its magic, length and CRC. *storage owns the bytes.
Status ReadBlobFile(Env* env, const std::string& path, std::string_view magic,
                    std::string* storage, std::string_view* payload);

}  // namespace vdb
