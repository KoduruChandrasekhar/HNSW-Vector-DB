// Status plus the name <-> enum conversions used by the manifest, the API and tools.

#include "vdb/status.h"

#include "vdb/metadata.h"
#include "vdb/options.h"
#include "vdb/types.h"

namespace vdb {

const char* CodeName(Status::Code code) {
  switch (code) {
    case Status::Code::kOk: return "ok";
    case Status::Code::kInvalidArgument: return "invalid_argument";
    case Status::Code::kNotFound: return "not_found";
    case Status::Code::kAlreadyExists: return "already_exists";
    case Status::Code::kIOError: return "io_error";
    case Status::Code::kCorruption: return "corruption";
    case Status::Code::kFailedPrecondition: return "failed_precondition";
    case Status::Code::kResourceExhausted: return "resource_exhausted";
    case Status::Code::kUnavailable: return "unavailable";
    case Status::Code::kNotSupported: return "not_supported";
    case Status::Code::kInternal: return "internal";
  }
  return "unknown";
}

std::string Status::ToString() const {
  if (ok()) return "OK";
  return std::string(CodeName(code_)) + ": " + msg_;
}

const char* MetricName(Metric m) {
  switch (m) {
    case Metric::L2: return "l2";
    case Metric::InnerProduct: return "ip";
    case Metric::Cosine: return "cosine";
  }
  return "unknown";
}

bool ParseMetric(std::string_view s, Metric* out) {
  if (s == "l2" || s == "L2" || s == "euclidean") {
    *out = Metric::L2;
  } else if (s == "ip" || s == "inner_product" || s == "dot") {
    *out = Metric::InnerProduct;
  } else if (s == "cosine" || s == "angular") {
    *out = Metric::Cosine;
  } else {
    return false;
  }
  return true;
}

const char* IndexTypeName(IndexType t) {
  switch (t) {
    case IndexType::Flat: return "flat";
    case IndexType::Hnsw: return "hnsw";
    case IndexType::HnswSq8: return "hnsw_sq8";
    case IndexType::PqFlat: return "pq";
    case IndexType::IvfPq: return "ivf_pq";
  }
  return "unknown";
}

bool ParseIndexType(std::string_view s, IndexType* out) {
  if (s == "flat") {
    *out = IndexType::Flat;
  } else if (s == "hnsw") {
    *out = IndexType::Hnsw;
  } else if (s == "hnsw_sq8" || s == "sq8") {
    *out = IndexType::HnswSq8;
  } else if (s == "pq" || s == "pq_flat") {
    *out = IndexType::PqFlat;
  } else if (s == "ivf_pq" || s == "ivfpq") {
    *out = IndexType::IvfPq;
  } else {
    return false;
  }
  return true;
}

const char* SyncModeName(SyncMode m) {
  switch (m) {
    case SyncMode::Always: return "always";
    case SyncMode::Interval: return "interval";
    case SyncMode::None: return "none";
  }
  return "unknown";
}

bool ParseSyncMode(std::string_view s, SyncMode* out) {
  if (s == "always") {
    *out = SyncMode::Always;
  } else if (s == "interval") {
    *out = SyncMode::Interval;
  } else if (s == "none") {
    *out = SyncMode::None;
  } else {
    return false;
  }
  return true;
}

const char* FilterStrategyName(FilterStrategy s) {
  switch (s) {
    case FilterStrategy::Auto: return "auto";
    case FilterStrategy::PreFilter: return "prefilter";
    case FilterStrategy::Graph: return "graph";
    case FilterStrategy::PostFilter: return "postfilter";
  }
  return "unknown";
}

bool ParseFilterStrategy(std::string_view s, FilterStrategy* out) {
  if (s == "auto") {
    *out = FilterStrategy::Auto;
  } else if (s == "prefilter" || s == "exact") {
    *out = FilterStrategy::PreFilter;
  } else if (s == "graph") {
    *out = FilterStrategy::Graph;
  } else if (s == "postfilter") {
    *out = FilterStrategy::PostFilter;
  } else {
    return false;
  }
  return true;
}

const char* FieldTypeName(FieldType t) {
  switch (t) {
    case FieldType::Int64: return "int64";
    case FieldType::Float64: return "float64";
    case FieldType::Keyword: return "keyword";
    case FieldType::KeywordList: return "keyword_list";
    case FieldType::Bool: return "bool";
  }
  return "unknown";
}

bool ParseFieldType(std::string_view s, FieldType* out) {
  if (s == "int64" || s == "int") {
    *out = FieldType::Int64;
  } else if (s == "float64" || s == "float" || s == "double") {
    *out = FieldType::Float64;
  } else if (s == "keyword" || s == "string") {
    *out = FieldType::Keyword;
  } else if (s == "keyword_list" || s == "tags") {
    *out = FieldType::KeywordList;
  } else if (s == "bool") {
    *out = FieldType::Bool;
  } else {
    return false;
  }
  return true;
}

}  // namespace vdb
