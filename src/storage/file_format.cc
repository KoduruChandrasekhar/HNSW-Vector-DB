#include "storage/file_format.h"

#include "util/coding.h"
#include "util/crc32c.h"

namespace vdb {

std::string EncodeBlob(std::string_view magic, std::string_view payload) {
  std::string out;
  out.reserve(kBlobHeaderSize + payload.size());
  out.append(magic.substr(0, 8));
  out.resize(8, '\0');
  PutU32(&out, 1);
  PutU32(&out, crc32c::Value(payload));
  PutU64(&out, payload.size());
  out.append(payload);
  return out;
}

Status DecodeBlob(std::string_view file, std::string_view magic, std::string_view* payload) {
  if (file.size() < kBlobHeaderSize) return Status::Corruption("file shorter than its header");
  if (file.substr(0, magic.size()) != magic) return Status::Corruption("bad file magic");
  uint32_t version = DecodeU32(file.data() + 8);
  uint32_t crc = DecodeU32(file.data() + 12);
  uint64_t len = DecodeU64(file.data() + 16);
  if (version != 1) return Status::Corruption("unsupported file version");
  if (len != file.size() - kBlobHeaderSize) return Status::Corruption("file length mismatch");
  std::string_view p = file.substr(kBlobHeaderSize);
  if (crc32c::Value(p) != crc) return Status::Corruption("file checksum mismatch");
  *payload = p;
  return Status::OK();
}

Status WriteBlobFile(Env* env, const std::string& path, std::string_view magic,
                     std::string_view payload, uint64_t* file_size, uint32_t* crc) {
  std::string blob = EncodeBlob(magic, payload);
  VDB_RETURN_IF_ERROR(WriteFileSync(env, path, blob));
  if (file_size != nullptr) *file_size = blob.size();
  if (crc != nullptr) *crc = DecodeU32(blob.data() + 12);
  return Status::OK();
}

Status ReadBlobFile(Env* env, const std::string& path, std::string_view magic,
                    std::string* storage, std::string_view* payload) {
  VDB_RETURN_IF_ERROR(env->read_file(path, storage));
  Status s = DecodeBlob(*storage, magic, payload);
  if (!s.ok()) return Status::Corruption(path + ": " + s.message());
  return Status::OK();
}

}  // namespace vdb
