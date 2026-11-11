#include <cinttypes>
#include <cstdio>
#include <cstring>

#include "storage/file_format.h"
#include "storage/segment.h"
#include "util/coding.h"
#include "util/crc32c.h"
#include "util/thread_pool.h"

namespace vdb {
namespace {

template <class T>
void Put(std::string& h, size_t off, T v) {
  std::memcpy(h.data() + off, &v, sizeof(T));
}

}  // namespace

std::string IndexFileName(IndexType t) { return std::string("index.") + IndexTypeName(t); }

std::string DeletesFileName(uint64_t number) {
  char buf[48];
  std::snprintf(buf, sizeof(buf), "deletes-%06" PRIu64 ".roar", number);
  return buf;
}

Status WriteDeletesFile(Env* env, const std::string& seg_dir, uint64_t number,
                        const roaring::Roaring& deleted, std::string* name, FileInfo* info) {
  roaring::Roaring copy = deleted;
  copy.runOptimize();
  std::string payload(copy.getSizeInBytes(true), '\0');
  copy.write(payload.data(), true);
  *name = DeletesFileName(number);
  VDB_RETURN_IF_ERROR(
      WriteBlobFile(env, JoinPath(seg_dir, *name), kDeletesMagic, payload, &info->size, &info->crc));
  return env->sync_dir(seg_dir);
}

StatusOr<std::shared_ptr<Segment>> WriteSegment(Env* env, const std::string& segments_dir,
                                                const SegmentSource& src,
                                                const SegmentWriteOptions& wo,
                                                std::shared_ptr<DeleteBits> deletes,
                                                SegmentMeta* meta_out) {
  const CollectionOptions& o = *wo.options;
  const size_t dim = o.dim;
  const size_t pd = PaddedDim(dim);
  const size_t row_bytes = pd * sizeof(float);
  const std::string name = SegmentDirName(wo.id);
  const std::string tmp = JoinPath(segments_dir, name + ".tmp");
  const std::string final_dir = JoinPath(segments_dir, name);
  if (env->exists(tmp)) VDB_RETURN_IF_ERROR(env->remove_dir_all(tmp));
  VDB_RETURN_IF_ERROR(env->create_dir(tmp));

  SegmentMeta meta;
  meta.id = wo.id;
  meta.rows = src.rows;
  meta.lsn = wo.lsn;

  // ---- vectors.f32: 4 KiB header, then padded rows.
  uint32_t data_crc = 0;
  for (size_t i = 0; i < src.rows; ++i) data_crc = crc32c::Extend(data_crc, src.row(i), row_bytes);
  const uint64_t data_len = static_cast<uint64_t>(src.rows) * row_bytes;
  std::string hdr(kSegmentHeaderSize, '\0');
  std::memcpy(hdr.data(), kSegmentMagic.data(), 8);
  Put<uint32_t>(hdr, 8, 1);
  Put<uint32_t>(hdr, 12, static_cast<uint32_t>(o.metric));
  Put<uint64_t>(hdr, 16, src.rows);
  Put<uint32_t>(hdr, 24, static_cast<uint32_t>(dim));
  Put<uint32_t>(hdr, 28, static_cast<uint32_t>(pd));
  Put<uint64_t>(hdr, 32, kSegmentHeaderSize);
  Put<uint64_t>(hdr, 40, data_len);
  Put<uint32_t>(hdr, 48, data_crc);
  Put<uint64_t>(hdr, 56, 0);
  Put<uint64_t>(hdr, 64, wo.lsn);
  Put<uint32_t>(hdr, kSegmentHeaderSize - 4, crc32c::Value(hdr.data(), kSegmentHeaderSize - 4));
  {
    const std::string path = JoinPath(tmp, "vectors.f32");
    VDB_ASSIGN_OR_RETURN(auto f, env->new_writable(path));
    VDB_RETURN_IF_ERROR(f->append(hdr));
    std::string buf;
    buf.reserve(1u << 20);
    for (size_t i = 0; i < src.rows; ++i) {
      buf.append(reinterpret_cast<const char*>(src.row(i)), row_bytes);
      if (buf.size() >= (1u << 20)) {
        VDB_RETURN_IF_ERROR(f->append(buf));
        buf.clear();
      }
    }
    if (!buf.empty()) VDB_RETURN_IF_ERROR(f->append(buf));
    VDB_RETURN_IF_ERROR(f->sync());
    VDB_RETURN_IF_ERROR(f->close());
    meta.files["vectors.f32"] = FileInfo{kSegmentHeaderSize + data_len, data_crc};
  }

  // ---- ids.u64
  {
    std::string payload(src.rows * sizeof(VectorId), '\0');
    for (size_t i = 0; i < src.rows; ++i) {
      VectorId id = src.id(i);
      std::memcpy(payload.data() + i * sizeof(VectorId), &id, sizeof(VectorId));
    }
    FileInfo fi;
    VDB_RETURN_IF_ERROR(WriteBlobFile(env, JoinPath(tmp, "ids.u64"), kIdsMagic, payload, &fi.size, &fi.crc));
    meta.files["ids.u64"] = fi;
  }

  // ---- meta.bin: u64 rows, u64 offsets[rows + 1], encoded rows.
  {
    std::string bytes;
    std::vector<uint64_t> offs(src.rows + 1, 0);
    for (size_t i = 0; i < src.rows; ++i) {
      offs[i] = bytes.size();
      if (src.meta) src.meta(i, &bytes);
    }
    offs[src.rows] = bytes.size();
    std::string payload;
    payload.reserve(8 + offs.size() * 8 + bytes.size());
    PutU64(&payload, src.rows);
    PutBytes(&payload, offs.data(), offs.size() * sizeof(uint64_t));
    payload.append(bytes);
    FileInfo fi;
    VDB_RETURN_IF_ERROR(WriteBlobFile(env, JoinPath(tmp, "meta.bin"), kMetaMagic, payload, &fi.size, &fi.crc));
    meta.files["meta.bin"] = fi;
  }

  // ---- index, built over the freshly written (memory-mapped) rows.
  VDB_ASSIGN_OR_RETURN(auto map, env->map_file(JoinPath(tmp, "vectors.f32")));
  VectorView view{reinterpret_cast<const float*>(map->data() + kSegmentHeaderSize), src.rows, dim, pd};
  std::unique_ptr<Index> index;
  if (wo.extend_from != nullptr) {
    auto ext = wo.extend_from->extend(view, wo.pool);
    if (ext.ok()) {
      index = std::move(ext).value();
    } else if (ext.status().code() != Status::Code::kNotSupported) {
      return ext.status();
    }
  }
  if (!index) {
    VDB_ASSIGN_OR_RETURN(index, CreateIndex(o.index, o.metric, view));
    VDB_RETURN_IF_ERROR(index->build(wo.pool));
  }
  {
    const std::string iname = IndexFileName(o.index.type);
    VDB_ASSIGN_OR_RETURN(auto f, env->new_writable(JoinPath(tmp, iname)));
    VDB_RETURN_IF_ERROR(index->save(*f));
    VDB_RETURN_IF_ERROR(f->sync());
    meta.files[iname] = FileInfo{f->size(), 0};
    VDB_RETURN_IF_ERROR(f->close());
  }

  // ---- durable: directory entries, then the atomic rename.
  VDB_RETURN_IF_ERROR(env->sync_dir(tmp));
  VDB_RETURN_IF_ERROR(env->rename(tmp, final_dir));
  VDB_RETURN_IF_ERROR(env->sync_dir(segments_dir));

  *meta_out = meta;
  return Segment::Open(env, final_dir, meta, o, /*verify_crc=*/false, std::move(deletes),
                       std::move(map), std::move(index));
}

}  // namespace vdb
