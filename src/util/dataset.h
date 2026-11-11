#pragma once

// .fbin / .ibin dataset files: uint32 count, uint32 dimension, then raw little-endian
// float32 (or int32) values, row-major. Written by tools/convert_hdf5.py.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "util/aligned_buffer.h"
#include "vdb/status.h"

namespace vdb {

struct Dataset {
  AlignedBuffer buffer;  // rows * padded_dim floats, zero padded
  size_t rows = 0;
  size_t dim = 0;
  size_t padded_dim = 0;
  size_t zero_rows = 0;  // rows that could not be normalized (cosine only)

  VectorView view() const { return {buffer.floats(), rows, dim, padded_dim}; }
  const float* row(size_t i) const { return buffer.floats() + i * padded_dim; }
};

// Reads up to max_rows rows (0 = all) into 64-byte aligned, zero-padded storage.
// With normalize set, every row is scaled to length 1 (zero rows are left as zeros).
Status ReadFbin(const std::string& path, size_t max_rows, bool normalize, Dataset* out);
Status ReadIbin(const std::string& path, size_t max_rows, std::vector<int32_t>* out,
                size_t* rows, size_t* cols);
Status WriteFbin(const std::string& path, const float* data, size_t rows, size_t dim,
                 size_t stride);
Status WriteIbin(const std::string& path, const int32_t* data, size_t rows, size_t cols);

// Scales v to unit length. Returns false (and leaves v unchanged) for a zero or
// non-finite vector.
bool NormalizeVector(float* v, size_t dim);

}  // namespace vdb
