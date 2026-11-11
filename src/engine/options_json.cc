#include "engine/options_json.h"

#include <nlohmann/json.hpp>

#include "meta/schema.h"

namespace vdb {

using json = nlohmann::json;

namespace {

template <class T>
Status GetUnsigned(const json& j, const char* key, T* out) {
  if (!j.contains(key)) return Status::OK();
  const json& v = j[key];
  if (!v.is_number_unsigned() && !(v.is_number_integer() && v.get<int64_t>() >= 0)) {
    return Status::InvalidArgument(std::string(key) + " must be a non-negative integer");
  }
  *out = static_cast<T>(v.get<uint64_t>());
  return Status::OK();
}

Status GetDouble(const json& j, const char* key, double* out) {
  if (!j.contains(key)) return Status::OK();
  if (!j[key].is_number()) return Status::InvalidArgument(std::string(key) + " must be a number");
  *out = j[key].get<double>();
  return Status::OK();
}

}  // namespace

json OptionsToJson(const CollectionOptions& o) {
  json idx = {{"type", IndexTypeName(o.index.type)},
              {"M", o.index.M},
              {"ef_construction", o.index.ef_construction},
              {"pq_m", o.index.pq_m},
              {"nlist", o.index.nlist},
              {"max_train_points_per_centroid", o.index.max_train_points_per_centroid},
              {"seed", o.index.seed}};
  return json{{"dim", o.dim},
              {"metric", MetricName(o.metric)},
              {"index", idx},
              {"schema", SchemaToJson(o.schema)},
              {"sync", SyncModeName(o.sync)},
              {"sync_interval_ms", o.sync_interval_ms},
              {"flush_bytes", o.flush_bytes},
              {"flush_age_ms", o.flush_age_ms},
              {"tier_base_rows", o.tier_base_rows},
              {"tier_fanout", o.tier_fanout},
              {"merge_into_largest_ratio", o.merge_into_largest_ratio},
              {"delete_compaction_ratio", o.delete_compaction_ratio}};
}

Status IndexOptionsFromJson(const json& j, IndexOptions* o) {
  if (j.is_null()) return Status::OK();
  if (!j.is_object()) return Status::InvalidArgument("index must be an object");
  if (j.contains("type")) {
    if (!j["type"].is_string() || !ParseIndexType(j["type"].get<std::string>(), &o->type)) {
      return Status::InvalidArgument("unknown index type");
    }
  }
  VDB_RETURN_IF_ERROR(GetUnsigned(j, "M", &o->M));
  VDB_RETURN_IF_ERROR(GetUnsigned(j, "ef_construction", &o->ef_construction));
  VDB_RETURN_IF_ERROR(GetUnsigned(j, "pq_m", &o->pq_m));
  VDB_RETURN_IF_ERROR(GetUnsigned(j, "m", &o->pq_m));
  VDB_RETURN_IF_ERROR(GetUnsigned(j, "nlist", &o->nlist));
  VDB_RETURN_IF_ERROR(GetUnsigned(j, "max_train_points_per_centroid", &o->max_train_points_per_centroid));
  VDB_RETURN_IF_ERROR(GetUnsigned(j, "seed", &o->seed));
  return Status::OK();
}

Status OptionsFromJson(const json& j, CollectionOptions* o) {
  try {
    if (!j.is_object()) return Status::InvalidArgument("options must be a JSON object");
    VDB_RETURN_IF_ERROR(GetUnsigned(j, "dim", &o->dim));
    if (j.contains("metric")) {
      if (!j["metric"].is_string() || !ParseMetric(j["metric"].get<std::string>(), &o->metric)) {
        return Status::InvalidArgument("unknown metric (use l2, ip or cosine)");
      }
    }
    if (j.contains("index")) {
      json idx = j["index"];
      if (idx.is_string()) idx = json{{"type", idx.get<std::string>()}};
      VDB_RETURN_IF_ERROR(IndexOptionsFromJson(idx, &o->index));
    }
    if (j.contains("schema")) VDB_RETURN_IF_ERROR(SchemaFromJson(j["schema"], &o->schema));
    if (j.contains("sync")) {
      if (!j["sync"].is_string() || !ParseSyncMode(j["sync"].get<std::string>(), &o->sync)) {
        return Status::InvalidArgument("unknown sync mode (use always, interval or none)");
      }
    }
    VDB_RETURN_IF_ERROR(GetUnsigned(j, "sync_interval_ms", &o->sync_interval_ms));
    VDB_RETURN_IF_ERROR(GetUnsigned(j, "flush_bytes", &o->flush_bytes));
    VDB_RETURN_IF_ERROR(GetUnsigned(j, "flush_age_ms", &o->flush_age_ms));
    VDB_RETURN_IF_ERROR(GetUnsigned(j, "tier_base_rows", &o->tier_base_rows));
    VDB_RETURN_IF_ERROR(GetUnsigned(j, "tier_fanout", &o->tier_fanout));
    VDB_RETURN_IF_ERROR(GetDouble(j, "merge_into_largest_ratio", &o->merge_into_largest_ratio));
    VDB_RETURN_IF_ERROR(GetDouble(j, "delete_compaction_ratio", &o->delete_compaction_ratio));
  } catch (const json::exception& e) {
    return Status::InvalidArgument(std::string("bad options: ") + e.what());
  }
  return Status::OK();
}

Status ValidateOptions(const CollectionOptions& o) {
  if (o.dim == 0 || o.dim > 16384) return Status::InvalidArgument("dim must be in 1..16384");
  if (o.index.M < 2 || o.index.M > 256) return Status::InvalidArgument("M must be in 2..256");
  if (o.index.ef_construction == 0) return Status::InvalidArgument("ef_construction must be > 0");
  if (o.index.type == IndexType::PqFlat || o.index.type == IndexType::IvfPq) {
    if (o.index.pq_m == 0 || o.dim % o.index.pq_m != 0) {
      return Status::InvalidArgument("pq_m must divide dim (dim " + std::to_string(o.dim) + ")");
    }
  }
  if (o.index.type == IndexType::IvfPq && o.index.nlist == 0) {
    return Status::InvalidArgument("nlist must be > 0");
  }
  if (o.tier_fanout < 2) return Status::InvalidArgument("tier_fanout must be >= 2");
  if (o.flush_bytes < 4096) return Status::InvalidArgument("flush_bytes must be >= 4096");
  return ValidateSchema(o.schema);
}

}  // namespace vdb
