#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace vdb {

using VectorId = uint64_t;    // external id chosen by the user
using InternalId = uint32_t;  // dense row number inside one segment

// One convention everywhere: smaller distance means closer.
//   L2           -> squared L2
//   InnerProduct -> minus the dot product
//   Cosine       -> vectors normalized to length 1, then minus the dot product
enum class Metric : uint8_t { L2 = 0, InnerProduct = 1, Cosine = 2 };

struct Neighbor {
  float dist;     // smaller = closer, for every metric
  InternalId id;  // row inside a segment or index
  // Ties are broken by id so every heap and sort is deterministic.
  bool operator<(const Neighbor& o) const {
    return dist < o.dist || (dist == o.dist && id < o.id);
  }
  bool operator>(const Neighbor& o) const { return o < *this; }
  bool operator==(const Neighbor& o) const { return dist == o.dist && id == o.id; }
};

// Rows are zero-padded to a multiple of 16 floats so SIMD kernels need no tail loop.
constexpr size_t kPadFloats = 16;
inline constexpr size_t PaddedDim(size_t dim) { return (dim + kPadFloats - 1) / kPadFloats * kPadFloats; }

const char* MetricName(Metric m);
bool ParseMetric(std::string_view s, Metric* out);

}  // namespace vdb
