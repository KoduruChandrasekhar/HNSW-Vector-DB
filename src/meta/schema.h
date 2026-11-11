#pragma once

// Metadata values stored per row. Internally a row is a MetaRow: one value per schema
// field, by field id, with std::monostate for null.

#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "vdb/metadata.h"
#include "vdb/status.h"

namespace vdb {

using MetaRow = std::vector<MetaValue>;

int FindField(const Schema& schema, std::string_view name);
Status ValidateSchema(const Schema& schema);

// Name-keyed metadata -> MetaRow, checking names and types against the schema. Integer
// values are accepted for float64 fields.
Status ToMetaRow(const Schema& schema, const Metadata& md, MetaRow* row);
Metadata FromMetaRow(const Schema& schema, const MetaRow& row);

// Compact binary encoding used in log records and segment files:
//   u16 count, then per set field: u16 field id, u8 type, value
//   int64/float64: 8 bytes; bool: 1 byte; keyword: u32 length + bytes;
//   keyword_list: u32 count, then (u32 length + bytes) each
void EncodeMetaRow(const MetaRow& row, std::string* out);
bool DecodeMetaRow(std::string_view data, const Schema& schema, MetaRow* row);

// JSON <-> metadata and schema ({"year": "int64", "tags": "keyword_list"}).
Status MetadataFromJson(const nlohmann::json& j, const Schema& schema, Metadata* md);
nlohmann::json MetadataToJson(const Metadata& md);
Status SchemaFromJson(const nlohmann::json& j, Schema* schema);
nlohmann::json SchemaToJson(const Schema& schema);

}  // namespace vdb
