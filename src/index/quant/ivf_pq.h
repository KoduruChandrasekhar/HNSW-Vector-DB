#pragma once

#include <vector>

#include "index/index.h"
#include "index/quant/pq.h"

namespace vdb {

// PQ codes for every row, scanned in full with ADC, then the top R re-ranked with the
// float rows.
class PqFlatIndex final : public Index {
 public:
  PqFlatIndex(const IndexOptions& opts, Metric metric, const VectorView& v);
  IndexType type() const override { return IndexType::PqFlat; }
  Status build(ThreadPool* pool) override;
  void search(const float* q, size_t k, const SearchParams& p, const RowFilter* filter,
              std::vector<Neighbor>& out, IndexSearchStats* stats) const override;
  size_t size() const override { return view_.rows; }
  size_t memory_bytes() const override { return codes_.size() + pq_.memory_bytes(); }
  size_t code_bytes() const override { return codes_.size(); }
  Status save(WritableFile& f) const override;
  Status load(ByteReader& r);
  const ProductQuantizer& pq() const { return pq_; }
  const uint8_t* code(size_t i) const { return codes_.data() + i * pq_.code_size(); }

  static constexpr std::string_view kMagic = "VDBPQFL1";

 private:
  IndexOptions opts_;
  Metric metric_;
  VectorView view_;
  DistFn dist_;
  ProductQuantizer pq_;
  std::vector<uint8_t> codes_;
};

// IVF-PQ (IVFADC): a coarse k-means quantizer with nlist inverted lists; each row stores
// the PQ code of its residual (row minus its list's centroid). A search probes the
// nprobe nearest lists, builds the ADC table for the query's residual against each list's
// centroid, scans the codes into a top-R heap, and re-ranks with the float rows.
class IvfPqIndex final : public Index {
 public:
  IvfPqIndex(const IndexOptions& opts, Metric metric, const VectorView& v);
  IndexType type() const override { return IndexType::IvfPq; }
  Status build(ThreadPool* pool) override;
  void search(const float* q, size_t k, const SearchParams& p, const RowFilter* filter,
              std::vector<Neighbor>& out, IndexSearchStats* stats) const override;
  size_t size() const override { return view_.rows; }
  size_t memory_bytes() const override;
  size_t code_bytes() const override;
  Status save(WritableFile& f) const override;
  Status load(ByteReader& r);

  size_t nlist() const { return nlist_; }
  size_t list_size(size_t l) const { return ids_[l].size(); }
  bool precomputed_tables() const { return !term2_.empty(); }

  static constexpr std::string_view kMagic = "VDBIVFP1";
  // Precompute per-list tables only while they stay this small (FAISS's default cap).
  static constexpr size_t kMaxPrecomputedBytes = size_t{1} << 30;

 private:
  // FAISS's precomputed-table trick for L2. With r the residual and c the list's centroid,
  //   ||q - c - r||^2 = ||q - c||^2 + sum_j (||r_j||^2 + 2 <c_j, r_j>) - 2 sum_j <q_j, r_j>
  // The middle term depends only on the list and the code (term2_, built once); the last
  // only on the query (one m x 256 table per query). A probed list then costs m x 256 adds
  // instead of m x 256 sub-vector distances.
  void precompute_tables();

  IndexOptions opts_;
  Metric metric_;
  VectorView view_;
  DistFn dist_;
  size_t nlist_ = 0;
  std::vector<float> coarse_;  // nlist * padded_dim
  ProductQuantizer pq_;
  std::vector<std::vector<uint32_t>> ids_;
  std::vector<std::vector<uint8_t>> codes_;
  std::vector<float> term2_;  // nlist * m * 256, L2 only
};

}  // namespace vdb
