// nanobind module vdb._vdb. NumPy arrays are read in place (no copy) and the GIL is
// released during upserts, loads and searches, so Python threads can query in parallel.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "engine/options_json.h"
#include "meta/schema.h"
#include "vdb/collection.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using json = nlohmann::json;
using Ids = nb::ndarray<const uint64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Matrix = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

void Check(const vdb::Status& s) {
  if (s.ok()) return;
  switch (s.code()) {
    case vdb::Status::Code::kInvalidArgument: throw nb::value_error(s.ToString().c_str());
    case vdb::Status::Code::kNotFound: throw nb::key_error(s.ToString().c_str());
    default: throw std::runtime_error(s.ToString());
  }
}

std::vector<vdb::Metadata> ParseMetadata(const std::vector<std::string>& items, const vdb::Schema& schema) {
  std::vector<vdb::Metadata> out(items.size());
  for (size_t i = 0; i < items.size(); ++i) {
    json j = json::parse(items[i], nullptr, false);
    if (j.is_discarded()) throw nb::value_error("metadata must be JSON objects");
    Check(vdb::MetadataFromJson(j, schema, &out[i]));
  }
  return out;
}

class PyCollection {
 public:
  explicit PyCollection(std::unique_ptr<vdb::Collection> c) : c_(std::move(c)) {}

  static PyCollection Create(const std::string& path, const std::string& options_json, size_t threads) {
    json j = json::parse(options_json, nullptr, false);
    if (j.is_discarded()) throw nb::value_error("options must be JSON");
    vdb::CollectionOptions o;
    Check(vdb::OptionsFromJson(j, &o));
    vdb::OpenOptions oo;
    oo.num_threads = threads;
    auto c = vdb::Collection::Create(path, o, oo);
    Check(c.status());
    return PyCollection(std::move(c).value());
  }

  static PyCollection Open(const std::string& path, size_t threads) {
    vdb::OpenOptions oo;
    oo.num_threads = threads;
    auto c = vdb::Collection::Open(path, oo);
    Check(c.status());
    return PyCollection(std::move(c).value());
  }

  uint64_t Upsert(const Ids& ids, const Matrix& vectors, const std::optional<std::vector<std::string>>& metadata) {
    CheckRows(ids, vectors);
    std::vector<vdb::Metadata> md;
    if (metadata) md = ParseMetadata(*metadata, c().options().schema);
    uint64_t lsn = 0;
    vdb::Status s;
    {
      nb::gil_scoped_release release;
      s = c().UpsertBatch(ids.data(), vectors.data(), ids.shape(0), metadata ? &md : nullptr, &lsn);
    }
    Check(s);
    return lsn;
  }

  void BulkLoad(const Ids& ids, const Matrix& vectors, const std::optional<std::vector<std::string>>& metadata) {
    CheckRows(ids, vectors);
    std::vector<vdb::Metadata> md;
    if (metadata) md = ParseMetadata(*metadata, c().options().schema);
    vdb::Status s;
    {
      nb::gil_scoped_release release;
      s = c().BulkLoad(ids.data(), vectors.data(), ids.shape(0), metadata ? &md : nullptr);
    }
    Check(s);
  }

  size_t Remove(const Ids& ids) {
    size_t removed = 0;
    vdb::Status s;
    {
      nb::gil_scoped_release release;
      s = c().RemoveBatch(ids.data(), ids.shape(0), &removed);
    }
    Check(s);
    return removed;
  }

  nb::tuple Search(const Matrix& queries, size_t k, size_t ef, size_t nprobe, int rerank,
                   const std::optional<std::string>& filter_json, const std::string& strategy) {
    if (queries.shape(1) != c().options().dim) throw nb::value_error("query dimension does not match the collection");
    vdb::SearchParams p;
    p.ef = ef;
    p.nprobe = nprobe;
    p.rerank = rerank;
    if (!vdb::ParseFilterStrategy(strategy, &p.strategy)) throw nb::value_error("bad strategy");
    std::unique_ptr<vdb::Filter> filter;
    if (filter_json) {
      auto f = c().ParseFilter(*filter_json);
      Check(f.status());
      filter = std::move(f).value();
    }
    const size_t nq = queries.shape(0);
    auto* ids = new uint64_t[nq * k];
    auto* dists = new float[nq * k];
    nb::capsule ids_owner(ids, [](void* ptr) noexcept { delete[] static_cast<uint64_t*>(ptr); });
    nb::capsule dists_owner(dists, [](void* ptr) noexcept { delete[] static_cast<float*>(ptr); });
    vdb::Status s;
    {
      nb::gil_scoped_release release;
      std::vector<std::vector<vdb::SearchHit>> hits;
      s = c().SearchBatch(queries.data(), nq, k, p, filter.get(), &hits);
      for (size_t q = 0; s.ok() && q < nq; ++q) {
        for (size_t i = 0; i < k; ++i) {
          bool have = i < hits[q].size();
          ids[q * k + i] = have ? hits[q][i].id : std::numeric_limits<uint64_t>::max();
          dists[q * k + i] = have ? hits[q][i].distance : std::numeric_limits<float>::infinity();
        }
      }
    }
    Check(s);
    return nb::make_tuple(nb::ndarray<nb::numpy, uint64_t, nb::ndim<2>>(ids, {nq, k}, ids_owner),
                          nb::ndarray<nb::numpy, float, nb::ndim<2>>(dists, {nq, k}, dists_owner));
  }

  nb::tuple Get(uint64_t id) {
    std::vector<float> v;
    vdb::Metadata md;
    Check(c().Get(id, &v, &md));
    return nb::make_tuple(v, vdb::MetadataToJson(md).dump());
  }

  void Flush() {
    nb::gil_scoped_release release;
    Check(c().Flush());
  }

  void Compact(size_t target) {
    vdb::Status s;
    {
      nb::gil_scoped_release release;
      s = c().Compact(target);
    }
    Check(s);
  }

  std::string Stats() {
    vdb::CollectionStats st = c().Stats();
    json segs = json::array();
    for (const auto& s : st.segments) {
      segs.push_back({{"id", s.id}, {"kind", s.kind}, {"index", s.index}, {"rows", s.rows}, {"deleted", s.deleted}});
    }
    return json{{"rows", st.live_rows},
                {"total_rows", st.total_rows},
                {"deleted_rows", st.deleted_rows},
                {"segments", segs},
                {"last_lsn", st.last_lsn},
                {"flushed_lsn", st.flushed_lsn},
                {"disk_bytes", st.disk_bytes},
                {"memory", {{"vectors", st.memory.vectors}, {"graph", st.memory.graph}, {"codes", st.memory.codes},
                            {"id_map", st.memory.id_map}, {"metadata", st.memory.metadata}}},
                {"config", vdb::OptionsToJson(c().options())}}
        .dump();
  }

  size_t Dim() { return c().options().dim; }

  void Close() {
    if (!c_) return;
    vdb::Status s;
    {
      nb::gil_scoped_release release;
      s = c_->Close();
    }
    c_.reset();
    Check(s);
  }

 private:
  vdb::Collection& c() {
    if (!c_) throw std::runtime_error("collection is closed");
    return *c_;
  }
  void CheckRows(const Ids& ids, const Matrix& vectors) {
    if (ids.shape(0) != vectors.shape(0)) throw nb::value_error("ids and vectors have different lengths");
    if (vectors.shape(1) != c().options().dim) {
      throw nb::value_error(("wrong dimension: got " + std::to_string(vectors.shape(1)) + ", expected " +
                             std::to_string(c().options().dim))
                                .c_str());
    }
  }

  std::unique_ptr<vdb::Collection> c_;
};

}  // namespace

NB_MODULE(_vdb, m) {
  m.doc() = "vdb: a vector database (C++ core)";
  nb::class_<PyCollection>(m, "_Collection")
      .def_static("create", &PyCollection::Create, "path"_a, "options_json"_a, "threads"_a = 0)
      .def_static("open", &PyCollection::Open, "path"_a, "threads"_a = 0)
      .def("upsert", &PyCollection::Upsert, "ids"_a, "vectors"_a, "metadata"_a = nb::none())
      .def("bulk_load", &PyCollection::BulkLoad, "ids"_a, "vectors"_a, "metadata"_a = nb::none())
      .def("delete", &PyCollection::Remove, "ids"_a)
      .def("search", &PyCollection::Search, "queries"_a, "k"_a = 10, "ef"_a = 64, "nprobe"_a = 16, "rerank"_a = -1,
           "filter"_a = nb::none(), "strategy"_a = "auto")
      .def("get", &PyCollection::Get, "id"_a)
      .def("flush", &PyCollection::Flush)
      .def("compact", &PyCollection::Compact, "target_segments"_a = 0)
      .def("stats", &PyCollection::Stats)
      .def("dim", &PyCollection::Dim)
      .def("close", &PyCollection::Close);
}
