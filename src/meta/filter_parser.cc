#include <algorithm>
#include <cmath>

#include <nlohmann/json.hpp>

#include "meta/filter.h"

namespace vdb {

using json = nlohmann::json;
using Op = FilterNode::Op;

const char* FilterOpName(Op op) {
  switch (op) {
    case Op::And: return "and";
    case Op::Or: return "or";
    case Op::Not: return "not";
    case Op::Eq: return "eq";
    case Op::Ne: return "ne";
    case Op::In: return "in";
    case Op::Lt: return "lt";
    case Op::Lte: return "lte";
    case Op::Gt: return "gt";
    case Op::Gte: return "gte";
    case Op::Exists: return "exists";
  }
  return "?";
}

namespace {

bool ParseOp(const std::string& s, Op* op) {
  static const std::pair<const char*, Op> kOps[] = {
      {"eq", Op::Eq}, {"ne", Op::Ne}, {"in", Op::In},   {"lt", Op::Lt},
      {"lte", Op::Lte}, {"gt", Op::Gt}, {"gte", Op::Gte}, {"exists", Op::Exists}};
  for (const auto& [name, o] : kOps) {
    if (s == name) {
      *op = o;
      return true;
    }
  }
  return false;
}

bool IsNumeric(FieldType t) { return t == FieldType::Int64 || t == FieldType::Float64; }

class Parser {
 public:
  explicit Parser(const Schema& schema) : schema_(schema) {}

  StatusOr<std::unique_ptr<FilterNode>> Node(const json& j, int depth) {
    if (depth > FilterExpr::kMaxDepth) return Status::InvalidArgument("filter nested too deeply");
    if (++nodes_ > FilterExpr::kMaxNodes) return Status::InvalidArgument("filter too large");
    if (!j.is_object()) return Status::InvalidArgument("filter node must be an object");
    auto n = std::make_unique<FilterNode>();
    if (j.contains("and") || j.contains("or")) {
      if (j.size() != 1) return Status::InvalidArgument("and/or must be the only key");
      bool is_and = j.contains("and");
      n->op = is_and ? Op::And : Op::Or;
      const json& arr = is_and ? j["and"] : j["or"];
      if (!arr.is_array() || arr.empty()) {
        return Status::InvalidArgument("and/or needs a non-empty list");
      }
      for (const auto& c : arr) {
        VDB_ASSIGN_OR_RETURN(auto child, Node(c, depth + 1));
        n->children.push_back(std::move(child));
      }
      return n;
    }
    if (j.contains("not")) {
      if (j.size() != 1) return Status::InvalidArgument("not must be the only key");
      n->op = Op::Not;
      VDB_ASSIGN_OR_RETURN(auto child, Node(j["not"], depth + 1));
      n->children.push_back(std::move(child));
      return n;
    }
    if (!j.contains("field") || !j["field"].is_string() || j.size() != 2) {
      return Status::InvalidArgument(
          "a condition needs \"field\" and exactly one operator, e.g. {\"field\": \"year\", "
          "\"gte\": 2020}");
    }
    const std::string fname = j["field"].get<std::string>();
    n->field = FindField(schema_, fname);
    if (n->field < 0) return Status::InvalidArgument("unknown filter field " + fname);
    n->type = schema_[n->field].type;
    std::string opname;
    const json* val = nullptr;
    for (auto it = j.begin(); it != j.end(); ++it) {
      if (it.key() != "field") {
        opname = it.key();
        val = &it.value();
      }
    }
    if (!ParseOp(opname, &n->op)) return Status::InvalidArgument("unknown operator " + opname);
    VDB_RETURN_IF_ERROR(Literal(*n, fname, *val));
    return n;
  }

 private:
  Status Literal(FilterNode& n, const std::string& fname, const json& v) {
    if (n.op == Op::Exists) {
      if (!v.is_boolean()) return Status::InvalidArgument("exists takes true or false");
      n.boolean = v.get<bool>();
      return Status::OK();
    }
    const bool numeric = IsNumeric(n.type);
    const bool range = n.op == Op::Lt || n.op == Op::Lte || n.op == Op::Gt || n.op == Op::Gte;
    if (range && !numeric) {
      return Status::InvalidArgument(std::string(FilterOpName(n.op)) + " needs a numeric field; " +
                                     fname + " is " + FieldTypeName(n.type));
    }
    if (n.type == FieldType::Bool && n.op == Op::In) {
      return Status::InvalidArgument("in is not supported on bool fields");
    }
    std::vector<const json*> items;
    if (n.op == Op::In) {
      if (!v.is_array() || v.empty()) return Status::InvalidArgument("in needs a non-empty list");
      if (v.size() > 1024) return Status::InvalidArgument("in list too long");
      for (const auto& e : v) items.push_back(&e);
    } else {
      items.push_back(&v);
    }
    for (const json* e : items) {
      if (numeric) {
        if (!e->is_number()) return Status::InvalidArgument("field " + fname + " compares with numbers");
        double d = e->get<double>();
        if (!std::isfinite(d)) return Status::InvalidArgument("non-finite number in filter");
        n.numbers.push_back(d);
      } else if (n.type == FieldType::Bool) {
        if (!e->is_boolean()) return Status::InvalidArgument("field " + fname + " compares with true/false");
        n.boolean = e->get<bool>();
      } else {
        if (!e->is_string()) return Status::InvalidArgument("field " + fname + " compares with strings");
        n.strings.push_back(e->get<std::string>());
      }
    }
    return Status::OK();
  }

  const Schema& schema_;
  size_t nodes_ = 0;
};

bool NumericOf(const MetaValue& v, double* out) {
  if (const auto* i = std::get_if<int64_t>(&v)) {
    *out = static_cast<double>(*i);
    return true;
  }
  if (const auto* d = std::get_if<double>(&v)) {
    *out = *d;
    return true;
  }
  return false;
}

bool Contains(const std::vector<std::string>& a, const std::string& s) {
  return std::find(a.begin(), a.end(), s) != a.end();
}

}  // namespace

bool FilterNode::Matches(const MetaRow& row) const {
  switch (op) {
    case Op::And:
      for (const auto& c : children)
        if (!c->Matches(row)) return false;
      return true;
    case Op::Or:
      for (const auto& c : children)
        if (c->Matches(row)) return true;
      return false;
    case Op::Not:
      return !children[0]->Matches(row);
    default:
      break;
  }
  const MetaValue& v = row[field];
  const bool present = !std::holds_alternative<std::monostate>(v);
  if (op == Op::Exists) return present == boolean;
  if (!present) return false;
  switch (type) {
    case FieldType::Int64:
    case FieldType::Float64: {
      double x = 0;
      if (!NumericOf(v, &x)) return false;
      switch (op) {
        case Op::Eq: return x == numbers[0];
        case Op::Ne: return x != numbers[0];
        case Op::In: return std::find(numbers.begin(), numbers.end(), x) != numbers.end();
        case Op::Lt: return x < numbers[0];
        case Op::Lte: return x <= numbers[0];
        case Op::Gt: return x > numbers[0];
        case Op::Gte: return x >= numbers[0];
        default: return false;
      }
    }
    case FieldType::Keyword: {
      const auto* s = std::get_if<std::string>(&v);
      if (s == nullptr) return false;
      switch (op) {
        case Op::Eq: return *s == strings[0];
        case Op::Ne: return *s != strings[0];
        case Op::In: return Contains(strings, *s);
        default: return false;
      }
    }
    case FieldType::KeywordList: {
      const auto* l = std::get_if<std::vector<std::string>>(&v);
      if (l == nullptr) return false;
      switch (op) {
        case Op::Eq: return Contains(*l, strings[0]);
        case Op::Ne: return !Contains(*l, strings[0]);
        case Op::In:
          for (const auto& s : strings)
            if (Contains(*l, s)) return true;
          return false;
        default: return false;
      }
    }
    case FieldType::Bool: {
      const auto* b = std::get_if<bool>(&v);
      if (b == nullptr) return false;
      if (op == Op::Eq) return *b == boolean;
      if (op == Op::Ne) return *b != boolean;
      return false;
    }
  }
  return false;
}

StatusOr<std::unique_ptr<FilterExpr>> FilterExpr::FromJson(const json& j, const Schema& schema) {
  try {
    Parser p(schema);
    VDB_ASSIGN_OR_RETURN(auto root, p.Node(j, 0));
    return std::unique_ptr<FilterExpr>(new FilterExpr(std::move(root)));
  } catch (const json::exception& e) {
    return Status::InvalidArgument(std::string("bad filter: ") + e.what());
  }
}

StatusOr<std::unique_ptr<FilterExpr>> FilterExpr::Parse(std::string_view text, const Schema& schema) {
  json j = json::parse(text.begin(), text.end(), nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded()) return Status::InvalidArgument("filter is not valid JSON");
  return FromJson(j, schema);
}

namespace {

void Describe(const FilterNode& n, std::string* out) {
  if (n.op == Op::And || n.op == Op::Or) {
    *out += "(";
    for (size_t i = 0; i < n.children.size(); ++i) {
      if (i > 0) *out += n.op == Op::And ? " AND " : " OR ";
      Describe(*n.children[i], out);
    }
    *out += ")";
    return;
  }
  if (n.op == Op::Not) {
    *out += "NOT ";
    Describe(*n.children[0], out);
    return;
  }
  *out += "#" + std::to_string(n.field) + " " + FilterOpName(n.op) + " ";
  if (!n.numbers.empty()) {
    for (double d : n.numbers) *out += json(d).dump() + ",";
  } else if (!n.strings.empty()) {
    for (const auto& s : n.strings) *out += json(s).dump() + ",";
  } else {
    *out += n.boolean ? "true" : "false";
  }
  if (out->back() == ',') out->pop_back();
}

}  // namespace

std::string FilterExpr::ToString() const {
  std::string s;
  Describe(*root_, &s);
  return s;
}

}  // namespace vdb
