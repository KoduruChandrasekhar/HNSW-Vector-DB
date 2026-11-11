#pragma once

// Filter language (JSON):
//   {"and": [
//     {"field": "year", "gte": 2020},
//     {"field": "tags", "in": ["ml", "db"]},
//     {"not": {"field": "lang", "eq": "fr"}}
//   ]}
// Operators: eq, ne, in, lt, lte, gt, gte, exists; combinators: and, or, not.
//
// Semantics: a comparison on a null field is false (so "ne" matches only rows that have
// the field); "not" is the complement over all rows. On keyword_list fields, eq means
// "contains", in means "contains any", ne means "does not contain". Numbers compare as
// doubles (exact for integers up to 2^53).

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "meta/schema.h"
#include "vdb/collection.h"

namespace vdb {

struct FilterNode {
  enum class Op : uint8_t { And, Or, Not, Eq, Ne, In, Lt, Lte, Gt, Gte, Exists };
  Op op = Op::And;
  int field = -1;
  FieldType type = FieldType::Int64;
  std::vector<double> numbers;       // numeric literals (int64, float64)
  std::vector<std::string> strings;  // keyword literals
  bool boolean = false;              // bool literal, or the "exists" flag
  std::vector<std::unique_ptr<FilterNode>> children;

  bool Matches(const MetaRow& row) const;
};

const char* FilterOpName(FilterNode::Op op);

class FilterExpr final : public Filter {
 public:
  static constexpr int kMaxDepth = 32;
  static constexpr size_t kMaxNodes = 4096;

  static StatusOr<std::unique_ptr<FilterExpr>> Parse(std::string_view json, const Schema& schema);
  static StatusOr<std::unique_ptr<FilterExpr>> FromJson(const nlohmann::json& j,
                                                        const Schema& schema);

  const FilterNode& root() const { return *root_; }
  bool Matches(const MetaRow& row) const { return root_->Matches(row); }
  std::string ToString() const override;

 private:
  explicit FilterExpr(std::unique_ptr<FilterNode> root) : root_(std::move(root)) {}
  std::unique_ptr<FilterNode> root_;
};

}  // namespace vdb
