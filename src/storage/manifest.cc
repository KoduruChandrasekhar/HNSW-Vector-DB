#include "storage/manifest.h"

#include <cinttypes>
#include <cstdio>

#include <nlohmann/json.hpp>

#include "engine/options_json.h"
#include "storage/file_format.h"

namespace vdb {

using json = nlohmann::json;

namespace {

bool ParseNumberSuffix(std::string_view name, std::string_view prefix, uint64_t* number) {
  if (name.size() <= prefix.size() || name.substr(0, prefix.size()) != prefix) return false;
  uint64_t n = 0;
  for (char c : name.substr(prefix.size())) {
    if (c < '0' || c > '9') return false;
    n = n * 10 + static_cast<uint64_t>(c - '0');
  }
  *number = n;
  return true;
}

}  // namespace

std::string ManifestFileName(uint64_t number) {
  char buf[40];
  std::snprintf(buf, sizeof(buf), "MANIFEST-%06" PRIu64, number);
  return buf;
}

bool ParseManifestFileName(std::string_view name, uint64_t* number) {
  return ParseNumberSuffix(name, "MANIFEST-", number);
}

std::string SegmentDirName(uint32_t id) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "seg-%06u", id);
  return buf;
}

bool ParseSegmentDirName(std::string_view name, uint32_t* id) {
  uint64_t n = 0;
  if (!ParseNumberSuffix(name, "seg-", &n) || n > UINT32_MAX) return false;
  *id = static_cast<uint32_t>(n);
  return true;
}

std::string EncodeManifest(const Manifest& m) {
  json segs = json::array();
  for (const auto& s : m.segments) {
    json files = json::object();
    for (const auto& [name, fi] : s.files) files[name] = {{"size", fi.size}, {"crc", fi.crc}};
    segs.push_back({{"id", s.id},
                    {"rows", s.rows},
                    {"deleted", s.deleted},
                    {"lsn", s.lsn},
                    {"deletes_file", s.deletes_file},
                    {"files", files}});
  }
  json j = {{"format_version", 1},
            {"number", m.number},
            {"options", OptionsToJson(m.options)},
            {"next_file_number", m.next_file_number},
            {"flushed_lsn", m.flushed_lsn},
            {"wal_log_number", m.wal_log_number},
            {"segments", segs}};
  return EncodeBlob(kManifestMagic, j.dump());
}

Status DecodeManifest(std::string_view data, Manifest* m) {
  std::string_view payload;
  VDB_RETURN_IF_ERROR(DecodeBlob(data, kManifestMagic, &payload));
  json j = json::parse(payload.begin(), payload.end(), nullptr, false);
  if (j.is_discarded() || !j.is_object()) return Status::Corruption("manifest is not valid JSON");
  try {
    *m = Manifest{};
    if (j.value("format_version", 0) != 1) return Status::Corruption("unknown manifest version");
    m->number = j.at("number").get<uint64_t>();
    VDB_RETURN_IF_ERROR(OptionsFromJson(j.at("options"), &m->options));
    m->next_file_number = j.at("next_file_number").get<uint64_t>();
    m->flushed_lsn = j.at("flushed_lsn").get<uint64_t>();
    m->wal_log_number = j.at("wal_log_number").get<uint64_t>();
    for (const auto& s : j.at("segments")) {
      SegmentMeta sm;
      sm.id = s.at("id").get<uint32_t>();
      sm.rows = s.at("rows").get<uint64_t>();
      sm.deleted = s.at("deleted").get<uint64_t>();
      sm.lsn = s.at("lsn").get<uint64_t>();
      sm.deletes_file = s.at("deletes_file").get<std::string>();
      for (auto it = s.at("files").begin(); it != s.at("files").end(); ++it) {
        sm.files[it.key()] = FileInfo{it.value().at("size").get<uint64_t>(),
                                      it.value().at("crc").get<uint32_t>()};
      }
      m->segments.push_back(std::move(sm));
    }
  } catch (const json::exception& e) {
    return Status::Corruption(std::string("bad manifest: ") + e.what());
  }
  return Status::OK();
}

Status CommitManifest(Env* env, const std::string& dir, const Manifest& m) {
  const std::string name = ManifestFileName(m.number);
  VDB_RETURN_IF_ERROR(WriteFileSync(env, JoinPath(dir, name), EncodeManifest(m)));
  const std::string tmp = JoinPath(dir, "CURRENT.tmp");
  VDB_RETURN_IF_ERROR(WriteFileSync(env, tmp, name + "\n"));
  VDB_RETURN_IF_ERROR(env->rename(tmp, JoinPath(dir, "CURRENT")));
  return env->sync_dir(dir);
}

Status ReadCurrentManifest(Env* env, const std::string& dir, Manifest* m) {
  std::string current;
  VDB_RETURN_IF_ERROR(env->read_file(JoinPath(dir, "CURRENT"), &current));
  while (!current.empty() && (current.back() == '\n' || current.back() == '\r')) current.pop_back();
  uint64_t number = 0;
  if (!ParseManifestFileName(current, &number)) return Status::Corruption("bad CURRENT file");
  std::string data;
  VDB_RETURN_IF_ERROR(env->read_file(JoinPath(dir, current), &data));
  VDB_RETURN_IF_ERROR(DecodeManifest(data, m));
  if (m->number != number) return Status::Corruption("manifest number does not match CURRENT");
  return Status::OK();
}

}  // namespace vdb
