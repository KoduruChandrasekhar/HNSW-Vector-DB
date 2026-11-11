#include "meta/columns.h"

#include <algorithm>
#include <limits>
#include <numeric>

namespace vdb {

using Op = FilterNode::Op;

namespace {

constexpr size_t kBucketRows = 4096;

void AddSorted(roaring::Roaring* r, std::vector<uint32_t>& rows) {
  if (rows.empty()) return;
  std::sort(rows.begin(), rows.end());
  r->addMany(rows.size(), rows.data());
}

}  // namespace

Status SegmentColumns::Build(const Schema& schema, size_t rows,
                             const std::function<std::string_view(size_t)>& encoded_row) {
  rows_ = rows;
  cols_.assign(schema.size(), Column{});
  for (size_t f = 0; f < schema.size(); ++f) {
    Column& c = cols_[f];
    c.type = schema[f].type;
    if (c.type == FieldType::Int64) c.ints.assign(rows, 0);
    if (c.type == FieldType::Int64 || c.type == FieldType::Float64) c.doubles.assign(rows, 0);
    if (c.type == FieldType::Keyword) c.codes.assign(rows, -1);
    if (c.type == FieldType::KeywordList) c.list_offsets.assign(1, 0);
  }
  auto code_of = [](Column& c, const std::string& s) {
    auto [it, inserted] = c.dict_index.emplace(s, static_cast<int32_t>(c.dict.size()));
    if (inserted) {
      c.dict.push_back(s);
      c.postings.emplace_back();
    }
    return it->second;
  };
  std::vector<std::vector<uint32_t>> present(schema.size()), trues(schema.size()),
      falses(schema.size());
  std::vector<std::vector<std::vector<uint32_t>>> posting_rows(schema.size());
  MetaRow row;
  for (size_t r = 0; r < rows; ++r) {
    if (!DecodeMetaRow(encoded_row(r), schema, &row)) {
      return Status::Corruption("bad metadata at row " + std::to_string(r));
    }
    const auto r32 = static_cast<uint32_t>(r);
    for (size_t f = 0; f < schema.size(); ++f) {
      Column& c = cols_[f];
      const MetaValue& v = row[f];
      bool has = !std::holds_alternative<std::monostate>(v);
      if (has) present[f].push_back(r32);
      switch (c.type) {
        case FieldType::Int64:
          if (has) {
            c.ints[r] = std::get<int64_t>(v);
            c.doubles[r] = static_cast<double>(c.ints[r]);
          }
          break;
        case FieldType::Float64:
          if (has) c.doubles[r] = std::get<double>(v);
          break;
        case FieldType::Keyword:
          if (has) {
            c.codes[r] = code_of(c, std::get<std::string>(v));
            posting_rows[f].resize(c.dict.size());
            posting_rows[f][c.codes[r]].push_back(r32);
          }
          break;
        case FieldType::KeywordList:
          if (has) {
            for (const auto& s : std::get<std::vector<std::string>>(v)) {
              int32_t code = code_of(c, s);
              c.list_codes.push_back(code);
              posting_rows[f].resize(c.dict.size());
              auto& pr = posting_rows[f][code];
              if (pr.empty() || pr.back() != r32) pr.push_back(r32);
            }
          }
          c.list_offsets.push_back(static_cast<uint32_t>(c.list_codes.size()));
          break;
        case FieldType::Bool:
          if (has) (std::get<bool>(v) ? trues[f] : falses[f]).push_back(r32);
          break;
      }
    }
  }
  for (size_t f = 0; f < schema.size(); ++f) {
    Column& c = cols_[f];
    AddSorted(&c.present, present[f]);
    AddSorted(&c.true_rows, trues[f]);
    AddSorted(&c.false_rows, falses[f]);
    for (size_t code = 0; code < posting_rows[f].size(); ++code) {
      AddSorted(&c.postings[code], posting_rows[f][code]);
    }
    for (auto& p : c.postings) p.runOptimize();
    if (c.type == FieldType::Int64 || c.type == FieldType::Float64) {
      // Sorted (value, row) pairs over present rows, then bucket bitmaps.
      std::vector<uint32_t> order(present[f].begin(), present[f].end());
      std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return c.doubles[a] < c.doubles[b] || (c.doubles[a] == c.doubles[b] && a < b);
      });
      NumericIndex& ni = c.num;
      ni.rows = order;
      ni.values.resize(order.size());
      for (size_t i = 0; i < order.size(); ++i) ni.values[i] = c.doubles[order[i]];
      for (size_t s = 0; s < order.size(); s += kBucketRows) {
        ni.starts.push_back(s);
        size_t e = std::min(order.size(), s + kBucketRows);
        std::vector<uint32_t> b(order.begin() + static_cast<std::ptrdiff_t>(s),
                                order.begin() + static_cast<std::ptrdiff_t>(e));
        roaring::Roaring bm;
        AddSorted(&bm, b);
        bm.runOptimize();
        ni.buckets.push_back(std::move(bm));
      }
      ni.starts.push_back(order.size());
    }
  }
  return Status::OK();
}

roaring::Roaring SegmentColumns::NumericIndex::Range(double lo, bool lo_incl, double hi,
                                                     bool hi_incl) const {
  auto first = lo_incl ? std::lower_bound(values.begin(), values.end(), lo)
                       : std::upper_bound(values.begin(), values.end(), lo);
  auto last = hi_incl ? std::upper_bound(values.begin(), values.end(), hi)
                      : std::lower_bound(values.begin(), values.end(), hi);
  roaring::Roaring out;
  if (first >= last) return out;
  size_t b = static_cast<size_t>(first - values.begin());
  size_t e = static_cast<size_t>(last - values.begin());
  std::vector<uint32_t> partial;
  size_t nb = buckets.size();
  for (size_t i = 0; i < nb; ++i) {
    size_t s = starts[i], t = starts[i + 1];
    if (t <= b || s >= e) continue;
    if (s >= b && t <= e) {
      out |= buckets[i];  // fully covered
    } else {
      for (size_t p = std::max(s, b); p < std::min(t, e); ++p) partial.push_back(rows[p]);
    }
  }
  AddSorted(&out, partial);
  return out;
}

roaring::Roaring SegmentColumns::All() const {
  roaring::Roaring r;
  if (rows_ > 0) r.addRange(0, rows_);
  return r;
}

roaring::Roaring SegmentColumns::Posting(const Column& c, const std::string& value) const {
  auto it = c.dict_index.find(value);
  if (it == c.dict_index.end()) return {};
  return c.postings[static_cast<size_t>(it->second)];
}

roaring::Roaring SegmentColumns::Evaluate(const FilterNode& n) const {
  switch (n.op) {
    case Op::And: {
      roaring::Roaring acc = Evaluate(*n.children[0]);
      for (size_t i = 1; i < n.children.size() && !acc.isEmpty(); ++i) acc &= Evaluate(*n.children[i]);
      return acc;
    }
    case Op::Or: {
      roaring::Roaring acc;
      for (const auto& c : n.children) acc |= Evaluate(*c);
      return acc;
    }
    case Op::Not:
      return All() - Evaluate(*n.children[0]);
    default:
      return Leaf(n);
  }
}

roaring::Roaring SegmentColumns::Leaf(const FilterNode& n) const {
  const Column& c = cols_[static_cast<size_t>(n.field)];
  if (n.op == Op::Exists) return n.boolean ? c.present : All() - c.present;
  constexpr double kInf = std::numeric_limits<double>::infinity();
  switch (c.type) {
    case FieldType::Int64:
    case FieldType::Float64: {
      const NumericIndex& ni = c.num;
      switch (n.op) {
        case Op::Eq: return ni.Range(n.numbers[0], true, n.numbers[0], true);
        case Op::Ne: return c.present - ni.Range(n.numbers[0], true, n.numbers[0], true);
        case Op::In: {
          roaring::Roaring acc;
          for (double v : n.numbers) acc |= ni.Range(v, true, v, true);
          return acc;
        }
        case Op::Lt: return ni.Range(-kInf, true, n.numbers[0], false);
        case Op::Lte: return ni.Range(-kInf, true, n.numbers[0], true);
        case Op::Gt: return ni.Range(n.numbers[0], false, kInf, true);
        case Op::Gte: return ni.Range(n.numbers[0], true, kInf, true);
        default: return {};
      }
    }
    case FieldType::Keyword:
    case FieldType::KeywordList:
      switch (n.op) {
        case Op::Eq: return Posting(c, n.strings[0]);
        case Op::Ne: return c.present - Posting(c, n.strings[0]);
        case Op::In: {
          roaring::Roaring acc;
          for (const auto& s : n.strings) acc |= Posting(c, s);
          return acc;
        }
        default: return {};
      }
    case FieldType::Bool:
      if (n.op == Op::Eq) return n.boolean ? c.true_rows : c.false_rows;
      if (n.op == Op::Ne) return n.boolean ? c.false_rows : c.true_rows;
      return {};
  }
  return {};
}

MetaRow SegmentColumns::GetRow(size_t r) const {
  MetaRow row(cols_.size(), std::monostate{});
  for (size_t f = 0; f < cols_.size(); ++f) {
    const Column& c = cols_[f];
    if (!c.present.contains(static_cast<uint32_t>(r))) continue;
    switch (c.type) {
      case FieldType::Int64: row[f] = c.ints[r]; break;
      case FieldType::Float64: row[f] = c.doubles[r]; break;
      case FieldType::Keyword: row[f] = c.dict[static_cast<size_t>(c.codes[r])]; break;
      case FieldType::KeywordList: {
        std::vector<std::string> l;
        for (uint32_t i = c.list_offsets[r]; i < c.list_offsets[r + 1]; ++i)
          l.push_back(c.dict[static_cast<size_t>(c.list_codes[i])]);
        row[f] = std::move(l);
        break;
      }
      case FieldType::Bool: row[f] = c.true_rows.contains(static_cast<uint32_t>(r)); break;
    }
  }
  return row;
}

size_t SegmentColumns::memory_bytes() const {
  size_t b = 0;
  for (const Column& c : cols_) {
    b += c.present.getSizeInBytes() + c.true_rows.getSizeInBytes() + c.false_rows.getSizeInBytes();
    b += c.ints.size() * 8 + c.doubles.size() * 8 + c.codes.size() * 4 + c.list_offsets.size() * 4 +
         c.list_codes.size() * 4;
    b += c.num.values.size() * 8 + c.num.rows.size() * 4;
    for (const auto& bm : c.num.buckets) b += bm.getSizeInBytes();
    for (const auto& p : c.postings) b += p.getSizeInBytes();
    for (const auto& s : c.dict) b += s.size() + 32;
  }
  return b;
}

}  // namespace vdb
