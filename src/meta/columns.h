#pragma once

// Metadata columns and filter indexes of one immutable segment.
//
// Each field is a column: numbers as an array plus a presence bitmap; keywords
// dictionary-encoded (a string table plus one code per row); keyword lists as offsets plus
// codes. Filter indexes: keyword value -> roaring bitmap of rows (an inverted index);
// numbers -> rows sorted by value, cut into buckets that each keep a bitmap, so a range
// is "OR the fully covered buckets, then add the rows of the two partial ones".

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "meta/filter.h"
#include "meta/schema.h"
#include "roaring.hh"

namespace vdb {

class SegmentColumns {
 public:
  // Builds columns from each row's encoded metadata (EncodeMetaRow format).
  Status Build(const Schema& schema, size_t rows,
               const std::function<std::string_view(size_t)>& encoded_row);

  size_t rows() const { return rows_; }
  // Rows matching the filter (deleted rows are not removed here).
  roaring::Roaring Evaluate(const FilterNode& node) const;
  MetaRow GetRow(size_t row) const;
  size_t memory_bytes() const;

 private:
  struct NumericIndex {
    std::vector<double> values;   // sorted ascending
    std::vector<uint32_t> rows;   // row of each sorted value
    std::vector<size_t> starts;   // bucket b covers positions [starts[b], starts[b+1])
    std::vector<roaring::Roaring> buckets;
    roaring::Roaring Range(double lo, bool lo_incl, double hi, bool hi_incl) const;
  };
  struct Column {
    FieldType type = FieldType::Int64;
    roaring::Roaring present;
    std::vector<int64_t> ints;
    std::vector<double> doubles;
    NumericIndex num;
    // keyword / keyword_list
    std::vector<std::string> dict;
    std::unordered_map<std::string, int32_t> dict_index;
    std::vector<int32_t> codes;           // keyword: one per row, -1 = null
    std::vector<uint32_t> list_offsets;   // keyword_list: rows + 1 offsets into list_codes
    std::vector<int32_t> list_codes;
    std::vector<roaring::Roaring> postings;  // per dictionary code
    // bool
    roaring::Roaring true_rows, false_rows;
  };

  roaring::Roaring Leaf(const FilterNode& n) const;
  roaring::Roaring All() const;
  roaring::Roaring Posting(const Column& c, const std::string& value) const;

  size_t rows_ = 0;
  std::vector<Column> cols_;
};

}  // namespace vdb
