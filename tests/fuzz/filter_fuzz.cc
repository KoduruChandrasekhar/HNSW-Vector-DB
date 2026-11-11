// The filter parser must never crash; anything it accepts must evaluate on a row.
#include <cstddef>
#include <cstdint>
#include <string>

#include "meta/filter.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  static const vdb::Schema schema = {{"year", vdb::FieldType::Int64},      {"score", vdb::FieldType::Float64},
                                     {"lang", vdb::FieldType::Keyword},    {"tags", vdb::FieldType::KeywordList},
                                     {"ok", vdb::FieldType::Bool}};
  static const vdb::MetaRow row = {int64_t{2021}, 0.5, std::string("en"), std::vector<std::string>{"ml"}, true};
  static const vdb::MetaRow empty(schema.size(), std::monostate{});
  std::string_view in(reinterpret_cast<const char*>(data), size);
  auto f = vdb::FilterExpr::Parse(in, schema);
  if (f.ok()) {
    (*f)->Matches(row);
    (*f)->Matches(empty);
    (void)(*f)->ToString();
  }
  // Metadata decoding shares the fuzz input.
  vdb::MetaRow r;
  vdb::DecodeMetaRow(in, schema, &r);
  return 0;
}
