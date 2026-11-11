#include "meta/schema.h"

#include <cmath>
#include <limits>
#include <set>

#include <nlohmann/json.hpp>

#include "util/coding.h"

namespace vdb {

using json = nlohmann::json;

int FindField(const Schema& schema, std::string_view name) {
  for (size_t i = 0; i < schema.size(); ++i)
    if (schema[i].name == name) return static_cast<int>(i);
  return -1;
}

Status ValidateSchema(const Schema& schema) {
  if (schema.size() > 1024) return Status::InvalidArgument("too many metadata fields");
  std::set<std::string> names;
  for (const auto& f : schema) {
    if (f.name.empty() || f.name.size() > 128) return Status::InvalidArgument("bad field name");
    if (!names.insert(f.name).second) return Status::InvalidArgument("duplicate field " + f.name);
  }
  return Status::OK();
}

namespace {

Status CheckType(const FieldDef& f, const MetaValue& v, MetaValue* out) {
  if (std::holds_alternative<std::monostate>(v)) {
    *out = v;
    return Status::OK();
  }
  switch (f.type) {
    case FieldType::Int64:
      if (std::holds_alternative<int64_t>(v)) {
        *out = v;
        return Status::OK();
      }
      break;
    case FieldType::Float64:
      if (std::holds_alternative<double>(v)) {
        *out = v;
        return Status::OK();
      }
      if (std::holds_alternative<int64_t>(v)) {
        *out = static_cast<double>(std::get<int64_t>(v));
        return Status::OK();
      }
      break;
    case FieldType::Keyword:
      if (std::holds_alternative<std::string>(v)) {
        *out = v;
        return Status::OK();
      }
      break;
    case FieldType::KeywordList:
      if (std::holds_alternative<std::vector<std::string>>(v)) {
        *out = v;
        return Status::OK();
      }
      break;
    case FieldType::Bool:
      if (std::holds_alternative<bool>(v)) {
        *out = v;
        return Status::OK();
      }
      break;
  }
  return Status::InvalidArgument("field " + f.name + " expects " + FieldTypeName(f.type));
}

}  // namespace

Status ToMetaRow(const Schema& schema, const Metadata& md, MetaRow* row) {
  row->assign(schema.size(), std::monostate{});
  for (const auto& [name, value] : md) {
    int fid = FindField(schema, name);
    if (fid < 0) return Status::InvalidArgument("unknown metadata field " + name);
    VDB_RETURN_IF_ERROR(CheckType(schema[fid], value, &(*row)[fid]));
  }
  return Status::OK();
}

Metadata FromMetaRow(const Schema& schema, const MetaRow& row) {
  Metadata md;
  for (size_t i = 0; i < schema.size() && i < row.size(); ++i) {
    if (!std::holds_alternative<std::monostate>(row[i])) md[schema[i].name] = row[i];
  }
  return md;
}

void EncodeMetaRow(const MetaRow& row, std::string* out) {
  uint16_t count = 0;
  for (const auto& v : row)
    if (!std::holds_alternative<std::monostate>(v)) ++count;
  PutU16(out, count);
  for (size_t i = 0; i < row.size(); ++i) {
    const MetaValue& v = row[i];
    if (std::holds_alternative<std::monostate>(v)) continue;
    PutU16(out, static_cast<uint16_t>(i));
    if (const auto* iv = std::get_if<int64_t>(&v)) {
      PutU8(out, static_cast<uint8_t>(FieldType::Int64));
      PutI64(out, *iv);
    } else if (const auto* dv = std::get_if<double>(&v)) {
      PutU8(out, static_cast<uint8_t>(FieldType::Float64));
      PutF64(out, *dv);
    } else if (const auto* sv = std::get_if<std::string>(&v)) {
      PutU8(out, static_cast<uint8_t>(FieldType::Keyword));
      PutString(out, *sv);
    } else if (const auto* lv = std::get_if<std::vector<std::string>>(&v)) {
      PutU8(out, static_cast<uint8_t>(FieldType::KeywordList));
      PutU32(out, static_cast<uint32_t>(lv->size()));
      for (const auto& s : *lv) PutString(out, s);
    } else if (const auto* bv = std::get_if<bool>(&v)) {
      PutU8(out, static_cast<uint8_t>(FieldType::Bool));
      PutU8(out, *bv ? 1 : 0);
    }
  }
}

bool DecodeMetaRow(std::string_view data, const Schema& schema, MetaRow* row) {
  row->assign(schema.size(), std::monostate{});
  if (data.empty()) return true;
  ByteReader r(data);
  uint16_t count = 0;
  if (!r.GetU16(&count)) return false;
  for (uint16_t n = 0; n < count; ++n) {
    uint16_t fid = 0;
    uint8_t type = 0;
    if (!r.GetU16(&fid) || !r.GetU8(&type) || fid >= schema.size()) return false;
    if (type != static_cast<uint8_t>(schema[fid].type)) return false;
    MetaValue& v = (*row)[fid];
    switch (static_cast<FieldType>(type)) {
      case FieldType::Int64: {
        int64_t x = 0;
        if (!r.GetI64(&x)) return false;
        v = x;
        break;
      }
      case FieldType::Float64: {
        double x = 0;
        if (!r.GetF64(&x)) return false;
        v = x;
        break;
      }
      case FieldType::Keyword: {
        std::string s;
        if (!r.GetString(&s)) return false;
        v = std::move(s);
        break;
      }
      case FieldType::KeywordList: {
        uint32_t cnt = 0;
        if (!r.GetU32(&cnt) || cnt > r.remaining() / 4) return false;
        std::vector<std::string> list(cnt);
        for (auto& s : list)
          if (!r.GetString(&s)) return false;
        v = std::move(list);
        break;
      }
      case FieldType::Bool: {
        uint8_t b = 0;
        if (!r.GetU8(&b)) return false;
        v = b != 0;
        break;
      }
      default:
        return false;
    }
  }
  return r.remaining() == 0;
}

Status MetadataFromJson(const json& j, const Schema& schema, Metadata* md) {
  md->clear();
  if (j.is_null()) return Status::OK();
  if (!j.is_object()) return Status::InvalidArgument("metadata must be a JSON object");
  for (auto it = j.begin(); it != j.end(); ++it) {
    const std::string& name = it.key();
    const json& v = it.value();
    int fid = FindField(schema, name);
    if (fid < 0) return Status::InvalidArgument("unknown metadata field " + name);
    if (v.is_null()) continue;
    const FieldDef& f = schema[fid];
    switch (f.type) {
      case FieldType::Int64:
        if (v.is_number_integer()) {
          (*md)[name] = v.get<int64_t>();
        } else if (v.is_number_float() && std::floor(v.get<double>()) == v.get<double>() &&
                   std::fabs(v.get<double>()) < 9.2e18) {
          (*md)[name] = static_cast<int64_t>(v.get<double>());
        } else {
          return Status::InvalidArgument("field " + name + " expects an integer");
        }
        break;
      case FieldType::Float64:
        if (!v.is_number()) return Status::InvalidArgument("field " + name + " expects a number");
        (*md)[name] = v.get<double>();
        break;
      case FieldType::Keyword:
        if (!v.is_string()) return Status::InvalidArgument("field " + name + " expects a string");
        (*md)[name] = v.get<std::string>();
        break;
      case FieldType::KeywordList: {
        if (!v.is_array()) return Status::InvalidArgument("field " + name + " expects a list");
        std::vector<std::string> list;
        for (const auto& e : v) {
          if (!e.is_string()) return Status::InvalidArgument("field " + name + " expects strings");
          list.push_back(e.get<std::string>());
        }
        (*md)[name] = std::move(list);
        break;
      }
      case FieldType::Bool:
        if (!v.is_boolean()) return Status::InvalidArgument("field " + name + " expects a bool");
        (*md)[name] = v.get<bool>();
        break;
    }
  }
  return Status::OK();
}

json MetadataToJson(const Metadata& md) {
  json j = json::object();
  for (const auto& [name, v] : md) {
    if (const auto* iv = std::get_if<int64_t>(&v)) {
      j[name] = *iv;
    } else if (const auto* dv = std::get_if<double>(&v)) {
      j[name] = *dv;
    } else if (const auto* sv = std::get_if<std::string>(&v)) {
      j[name] = *sv;
    } else if (const auto* lv = std::get_if<std::vector<std::string>>(&v)) {
      j[name] = *lv;
    } else if (const auto* bv = std::get_if<bool>(&v)) {
      j[name] = *bv;
    }
  }
  return j;
}

Status SchemaFromJson(const json& j, Schema* schema) {
  schema->clear();
  if (j.is_null()) return Status::OK();
  if (j.is_object()) {
    for (auto it = j.begin(); it != j.end(); ++it) {
      FieldType t{};
      if (!it.value().is_string() || !ParseFieldType(it.value().get<std::string>(), &t)) {
        return Status::InvalidArgument("bad type for schema field " + it.key());
      }
      schema->push_back({it.key(), t});
    }
  } else if (j.is_array()) {  // [{"name": ..., "type": ...}] keeps field order explicit
    for (const auto& e : j) {
      FieldType t{};
      if (!e.is_object() || !e.contains("name") || !e["name"].is_string() || !e.contains("type") ||
          !e["type"].is_string() || !ParseFieldType(e["type"].get<std::string>(), &t)) {
        return Status::InvalidArgument("bad schema field");
      }
      schema->push_back({e["name"].get<std::string>(), t});
    }
  } else {
    return Status::InvalidArgument("schema must be an object or a list");
  }
  return ValidateSchema(*schema);
}

json SchemaToJson(const Schema& schema) {
  json arr = json::array();
  for (const auto& f : schema) arr.push_back({{"name", f.name}, {"type", FieldTypeName(f.type)}});
  return arr;
}

}  // namespace vdb
