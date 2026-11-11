#pragma once

// MANIFEST-<n>: the collection config, the segment list, flushed_lsn and file counters,
// as JSON in a checksummed blob. CURRENT names the live manifest. A new manifest becomes
// live only through an atomic rename of CURRENT.tmp, so a crash leaves either the old
// manifest or the new one current, never anything in between.

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "storage/env.h"
#include "vdb/options.h"

namespace vdb {

struct FileInfo {
  uint64_t size = 0;
  uint32_t crc = 0;
};

struct SegmentMeta {
  uint32_t id = 0;
  uint64_t rows = 0;
  uint64_t deleted = 0;
  // The segment's rows and delete set reflect every log record at or below this LSN;
  // recovery never applies an older record to this segment's rows.
  uint64_t lsn = 0;
  std::string deletes_file;  // empty: no persisted deletes
  std::map<std::string, FileInfo> files;
};

struct Manifest {
  uint64_t number = 0;
  CollectionOptions options;
  uint64_t next_file_number = 1;  // shared by segments, logs, manifests, delete files
  uint64_t flushed_lsn = 0;       // every record at or below is in the segments
  uint64_t wal_log_number = 0;    // logs numbered below this are obsolete
  std::vector<SegmentMeta> segments;
};

constexpr std::string_view kManifestMagic = "VDBMAN01";

std::string ManifestFileName(uint64_t number);  // MANIFEST-000009
bool ParseManifestFileName(std::string_view name, uint64_t* number);
std::string SegmentDirName(uint32_t id);  // seg-000006
bool ParseSegmentDirName(std::string_view name, uint32_t* id);

std::string EncodeManifest(const Manifest& m);
Status DecodeManifest(std::string_view data, Manifest* m);

// Step 5 of a flush: write MANIFEST-<n> and fsync it; write CURRENT.tmp, fsync it, rename
// it to CURRENT and fsync the collection directory.
Status CommitManifest(Env* env, const std::string& dir, const Manifest& m);
Status ReadCurrentManifest(Env* env, const std::string& dir, Manifest* m);

}  // namespace vdb
