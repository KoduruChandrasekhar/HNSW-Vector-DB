#include <cstring>

#include "storage/file_format.h"
#include "storage/segment.h"
#include "util/coding.h"
#include "util/crc32c.h"

namespace vdb {
namespace {

template <class T>
T Get(const char* h, size_t off) {
  T v;
  std::memcpy(&v, h + off, sizeof(T));
  return v;
}

}  // namespace

StatusOr<std::shared_ptr<Segment>> Segment::Open(Env* env, const std::string& dir,
                                                 const SegmentMeta& meta,
                                                 const CollectionOptions& opts, bool verify_crc,
                                                 std::shared_ptr<DeleteBits> deletes,
                                                 std::unique_ptr<MappedRegion> map,
                                                 std::unique_ptr<Index> index) {
  std::shared_ptr<Segment> s(new Segment());
  s->env_ = env;
  s->dir_ = dir;
  s->meta_ = meta;
  s->lsn_.store(meta.lsn);
  const size_t pd = PaddedDim(opts.dim);
  const std::string vpath = JoinPath(dir, "vectors.f32");
  if (!map) {
    VDB_ASSIGN_OR_RETURN(map, env->map_file(vpath));
  }

  // ---- header and size checks
  if (map->size() < kSegmentHeaderSize) return Status::Corruption(vpath + ": shorter than its header");
  const char* h = reinterpret_cast<const char*>(map->data());
  if (std::memcmp(h, kSegmentMagic.data(), 8) != 0) return Status::Corruption(vpath + ": bad magic");
  if (Get<uint32_t>(h, kSegmentHeaderSize - 4) != crc32c::Value(h, kSegmentHeaderSize - 4)) {
    return Status::Corruption(vpath + ": header checksum mismatch");
  }
  const auto rows = Get<uint64_t>(h, 16);
  const uint64_t data_len = Get<uint64_t>(h, 40);
  if (Get<uint32_t>(h, 8) != 1 || Get<uint32_t>(h, 12) != static_cast<uint32_t>(opts.metric) ||
      rows != meta.rows || Get<uint32_t>(h, 24) != opts.dim || Get<uint32_t>(h, 28) != pd ||
      Get<uint64_t>(h, 32) != kSegmentHeaderSize || data_len != rows * pd * sizeof(float) ||
      map->size() < kSegmentHeaderSize + data_len) {
    return Status::Corruption(vpath + ": header does not match the manifest");
  }
  const char* data = h + kSegmentHeaderSize;
  if (verify_crc && crc32c::Value(data, data_len) != Get<uint32_t>(h, 48)) {
    return Status::Corruption(vpath + ": data checksum mismatch");
  }
  s->view_ = VectorView{reinterpret_cast<const float*>(data), rows, opts.dim, pd};
  bool graph = opts.index.type == IndexType::Hnsw || opts.index.type == IndexType::HnswSq8;
  map->advise(graph ? Access::kRandom : Access::kNormal);
  s->map_ = std::move(map);

  // ---- ids
  std::string storage;
  std::string_view payload;
  VDB_RETURN_IF_ERROR(ReadBlobFile(env, JoinPath(dir, "ids.u64"), kIdsMagic, &storage, &payload));
  if (payload.size() != rows * sizeof(VectorId)) return Status::Corruption(dir + ": ids.u64 size mismatch");
  s->ids_.resize(rows);
  if (rows > 0) std::memcpy(s->ids_.data(), payload.data(), payload.size());

  // ---- metadata columns and filter indexes
  VDB_RETURN_IF_ERROR(ReadBlobFile(env, JoinPath(dir, "meta.bin"), kMetaMagic, &storage, &payload));
  {
    ByteReader r(payload);
    uint64_t n = 0;
    std::string_view offs_bytes;
    if (!r.GetU64(&n) || n != rows || !r.GetBytes((rows + 1) * sizeof(uint64_t), &offs_bytes)) {
      return Status::Corruption(dir + ": bad meta.bin");
    }
    std::vector<uint64_t> offs(rows + 1);
    std::memcpy(offs.data(), offs_bytes.data(), offs_bytes.size());
    std::string_view bytes(r.position(), r.remaining());
    for (size_t i = 0; i < rows; ++i) {
      if (offs[i] > offs[i + 1] || offs[i + 1] > bytes.size()) {
        return Status::Corruption(dir + ": bad meta.bin offsets");
      }
    }
    VDB_RETURN_IF_ERROR(s->columns_.Build(opts.schema, rows, [&](size_t i) {
      return bytes.substr(offs[i], offs[i + 1] - offs[i]);
    }));
  }

  // ---- index
  if (!index) {
    std::string bytes;
    VDB_RETURN_IF_ERROR(env->read_file(JoinPath(dir, IndexFileName(opts.index.type)), &bytes));
    auto loaded = LoadIndex(bytes, opts.index, opts.metric, s->view_);
    if (!loaded.ok()) return Status::Corruption(dir + ": " + loaded.status().message());
    index = std::move(loaded).value();
  }
  s->index_ = std::move(index);

  // ---- deletes
  if (deletes) {
    s->deletes_ = std::move(deletes);
    s->deletes_->ensure(rows);
  } else {
    s->deletes_ = std::make_shared<DeleteBits>(std::max<size_t>(rows, 1));
    s->deletes_->ensure(rows);
    if (!meta.deletes_file.empty()) {
      VDB_RETURN_IF_ERROR(ReadBlobFile(env, JoinPath(dir, meta.deletes_file), kDeletesMagic, &storage, &payload));
      try {
        roaring::Roaring r = roaring::Roaring::readSafe(payload.data(), payload.size());
        if (!r.isEmpty() && r.maximum() >= rows) return Status::Corruption(dir + ": delete set out of range");
        s->deletes_->set_from(r);
      } catch (const std::exception& e) {
        return Status::Corruption(dir + ": bad delete set: " + e.what());
      }
    }
  }
  s->persisted_deletes_.store(s->deletes_->count());
  s->deletes_file_ = meta.deletes_file;
  return s;
}

Segment::~Segment() {
  if (obsolete_.load() && env_ != nullptr) {
    index_.reset();
    map_.reset();
    env_->remove_dir_all(dir_);  // best effort; recovery removes leftovers
  }
}

std::shared_ptr<const roaring::Roaring> Segment::deleted_bitmap() const {
  std::lock_guard<std::mutex> g(cache_mu_);
  size_t c = deletes_->count();
  if (!deleted_cache_ || c != deleted_cache_count_) {
    deleted_cache_ = std::make_shared<const roaring::Roaring>(deletes_->to_roaring(rows()));
    deleted_cache_count_ = c;
  }
  return deleted_cache_;
}

MemoryStats Segment::memory() const {
  MemoryStats m;
  m.vectors = view_.rows * view_.padded_dim * sizeof(float);
  m.graph = index_->graph_bytes();
  m.codes = index_->memory_bytes() - std::min(index_->memory_bytes(), index_->graph_bytes());
  m.id_map = ids_.size() * sizeof(VectorId);
  m.metadata = columns_.memory_bytes() + deletes_->memory_bytes();
  return m;
}

uint64_t Segment::disk_bytes() const {
  uint64_t b = 0;
  for (const auto& [name, fi] : meta_.files) b += fi.size;
  return b;
}

}  // namespace vdb
