#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace vdb {

enum class FieldType : uint8_t { Int64 = 0, Float64 = 1, Keyword = 2, KeywordList = 3, Bool = 4 };

struct FieldDef {
  std::string name;
  FieldType type;
  bool operator==(const FieldDef& o) const { return name == o.name && type == o.type; }
};

// Declared when the collection is created; field order defines field ids.
using Schema = std::vector<FieldDef>;

// std::monostate means "null / not set".
using MetaValue =
    std::variant<std::monostate, int64_t, double, std::string, std::vector<std::string>, bool>;

// User-facing metadata: field name -> value.
using Metadata = std::map<std::string, MetaValue>;

const char* FieldTypeName(FieldType t);
bool ParseFieldType(std::string_view s, FieldType* out);

}  // namespace vdb
