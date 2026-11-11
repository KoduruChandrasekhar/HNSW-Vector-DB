#include "index/index.h"

#include "index/flat_index.h"
#include "index/hnsw/hnsw_index.h"
#include "index/quant/ivf_pq.h"
#include "storage/env.h"
#include "util/crc32c.h"
#include "util/topk.h"

namespace vdb {

StatusOr<std::unique_ptr<Index>> Index::extend(const VectorView&, ThreadPool*) const {
  return Status::NotSupported(std::string("extend is not supported by ") + IndexTypeName(type()));
}

StatusOr<std::unique_ptr<Index>> CreateIndex(const IndexOptions& opts, Metric metric,
                                             const VectorView& vectors) {
  switch (opts.type) {
    case IndexType::Flat:
      return std::unique_ptr<Index>(std::make_unique<FlatIndex>(metric, vectors));
    case IndexType::Hnsw:
      return std::unique_ptr<Index>(std::make_unique<HnswIndex>(opts, metric, vectors));
    case IndexType::HnswSq8:
      return std::unique_ptr<Index>(std::make_unique<HnswSq8Index>(opts, metric, vectors));
    case IndexType::PqFlat:
    case IndexType::IvfPq:
      if (opts.pq_m == 0 || vectors.dim % opts.pq_m != 0) {
        return Status::InvalidArgument("pq_m must divide the dimension");
      }
      if (opts.type == IndexType::PqFlat) {
        return std::unique_ptr<Index>(std::make_unique<PqFlatIndex>(opts, metric, vectors));
      }
      return std::unique_ptr<Index>(std::make_unique<IvfPqIndex>(opts, metric, vectors));
  }
  return Status::InvalidArgument("unknown index type");
}

Status WriteIndexFile(WritableFile& f, std::string_view magic, const std::string& body) {
  std::string out;
  out.reserve(magic.size() + body.size() + 4);
  out.append(magic);
  out.append(body);
  PutU32(&out, crc32c::Value(out.data(), out.size()));
  return f.append(out);
}

Status CheckIndexFile(std::string_view bytes, std::string_view* magic, std::string_view* body) {
  if (bytes.size() < 12) return Status::Corruption("index file too short");
  uint32_t stored = DecodeU32(bytes.data() + bytes.size() - 4);
  if (crc32c::Value(bytes.data(), bytes.size() - 4) != stored) {
    return Status::Corruption("index file checksum mismatch");
  }
  *magic = bytes.substr(0, 8);
  *body = bytes.substr(8, bytes.size() - 12);
  return Status::OK();
}

StatusOr<std::unique_ptr<Index>> LoadIndex(std::string_view bytes, const IndexOptions& opts,
                                           Metric metric, const VectorView& vectors) {
  std::string_view magic, body;
  VDB_RETURN_IF_ERROR(CheckIndexFile(bytes, &magic, &body));
  ByteReader r(body);
  if (magic == FlatIndex::kMagic) {
    return std::unique_ptr<Index>(std::make_unique<FlatIndex>(metric, vectors));
  }
  if (magic == HnswIndex::kMagic) {
    auto idx = std::make_unique<HnswIndex>(opts, metric, vectors);
    VDB_RETURN_IF_ERROR(idx->load(r));
    return std::unique_ptr<Index>(std::move(idx));
  }
  if (magic == HnswSq8Index::kMagic) {
    auto idx = std::make_unique<HnswSq8Index>(opts, metric, vectors);
    VDB_RETURN_IF_ERROR(idx->load(r));
    return std::unique_ptr<Index>(std::move(idx));
  }
  if (magic == PqFlatIndex::kMagic) {
    auto idx = std::make_unique<PqFlatIndex>(opts, metric, vectors);
    VDB_RETURN_IF_ERROR(idx->load(r));
    return std::unique_ptr<Index>(std::move(idx));
  }
  if (magic == IvfPqIndex::kMagic) {
    auto idx = std::make_unique<IvfPqIndex>(opts, metric, vectors);
    VDB_RETURN_IF_ERROR(idx->load(r));
    return std::unique_ptr<Index>(std::move(idx));
  }
  return Status::Corruption("unknown index file magic");
}

void Rerank(const VectorView& v, DistFn fn, const float* q, const std::vector<Neighbor>& cands,
            size_t k, std::vector<Neighbor>& out, uint64_t* distance_computations) {
  TopK top(k);
  for (const Neighbor& c : cands) top.push(fn(q, v.row(c.id), v.padded_dim), c.id);
  if (distance_computations != nullptr) *distance_computations += cands.size();
  top.sorted_into(&out);
}

}  // namespace vdb
