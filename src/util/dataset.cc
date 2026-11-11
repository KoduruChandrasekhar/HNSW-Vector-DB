#include "util/dataset.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>

#include "vdb/types.h"

namespace vdb {
namespace {

struct FileCloser {
  void operator()(FILE* f) const {
    if (f != nullptr) std::fclose(f);
  }
};
using FilePtr = std::unique_ptr<FILE, FileCloser>;

Status ReadHeader(FILE* f, const std::string& path, uint32_t* n, uint32_t* d) {
  uint32_t hdr[2];
  if (std::fread(hdr, sizeof(uint32_t), 2, f) != 2) {
    return Status::Corruption("short header in " + path);
  }
  *n = hdr[0];
  *d = hdr[1];
  if (*d == 0 || *d > (1u << 16)) return Status::Corruption("bad dimension in " + path);
  return Status::OK();
}

}  // namespace

bool NormalizeVector(float* v, size_t dim) {
  double norm = 0;
  for (size_t i = 0; i < dim; ++i) norm += static_cast<double>(v[i]) * v[i];
  if (!(norm > 0) || !std::isfinite(norm)) return false;
  auto inv = static_cast<float>(1.0 / std::sqrt(norm));
  for (size_t i = 0; i < dim; ++i) v[i] *= inv;
  return true;
}

Status ReadFbin(const std::string& path, size_t max_rows, bool normalize, Dataset* out) {
  FilePtr f(std::fopen(path.c_str(), "rb"));
  if (!f) return Status::IOError("cannot open " + path);
  uint32_t n = 0, d = 0;
  VDB_RETURN_IF_ERROR(ReadHeader(f.get(), path, &n, &d));
  size_t rows = (max_rows == 0 || max_rows > n) ? n : max_rows;
  out->rows = rows;
  out->dim = d;
  out->padded_dim = PaddedDim(d);
  out->zero_rows = 0;
  out->buffer.reset(rows * out->padded_dim * sizeof(float));
  float* base = out->buffer.floats();
  // Read in blocks, then spread rows to their padded positions.
  constexpr size_t kBlock = 8192;
  std::vector<float> tmp(kBlock * d);
  for (size_t r = 0; r < rows; r += kBlock) {
    size_t cnt = std::min(kBlock, rows - r);
    if (std::fread(tmp.data(), sizeof(float) * d, cnt, f.get()) != cnt) {
      return Status::Corruption("truncated data in " + path);
    }
    for (size_t i = 0; i < cnt; ++i) {
      float* dst = base + (r + i) * out->padded_dim;
      std::memcpy(dst, tmp.data() + i * d, sizeof(float) * d);
      if (normalize && !NormalizeVector(dst, d)) ++out->zero_rows;
    }
  }
  return Status::OK();
}

Status ReadIbin(const std::string& path, size_t max_rows, std::vector<int32_t>* out,
                size_t* rows, size_t* cols) {
  FilePtr f(std::fopen(path.c_str(), "rb"));
  if (!f) return Status::IOError("cannot open " + path);
  uint32_t n = 0, d = 0;
  VDB_RETURN_IF_ERROR(ReadHeader(f.get(), path, &n, &d));
  size_t r = (max_rows == 0 || max_rows > n) ? n : max_rows;
  out->resize(r * d);
  if (std::fread(out->data(), sizeof(int32_t) * d, r, f.get()) != r) {
    return Status::Corruption("truncated data in " + path);
  }
  *rows = r;
  *cols = d;
  return Status::OK();
}

Status WriteFbin(const std::string& path, const float* data, size_t rows, size_t dim,
                 size_t stride) {
  FilePtr f(std::fopen(path.c_str(), "wb"));
  if (!f) return Status::IOError("cannot create " + path);
  uint32_t hdr[2] = {static_cast<uint32_t>(rows), static_cast<uint32_t>(dim)};
  bool ok = std::fwrite(hdr, sizeof(uint32_t), 2, f.get()) == 2;
  for (size_t i = 0; ok && i < rows; ++i) {
    ok = std::fwrite(data + i * stride, sizeof(float), dim, f.get()) == dim;
  }
  if (!ok) return Status::IOError("write failed: " + path);
  return Status::OK();
}

Status WriteIbin(const std::string& path, const int32_t* data, size_t rows, size_t cols) {
  FilePtr f(std::fopen(path.c_str(), "wb"));
  if (!f) return Status::IOError("cannot create " + path);
  uint32_t hdr[2] = {static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)};
  bool ok = std::fwrite(hdr, sizeof(uint32_t), 2, f.get()) == 2 &&
            std::fwrite(data, sizeof(int32_t), rows * cols, f.get()) == rows * cols;
  if (!ok) return Status::IOError("write failed: " + path);
  return Status::OK();
}

}  // namespace vdb
