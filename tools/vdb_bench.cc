// vdb_bench: library-level benchmark runner.
//
//   vdb_bench index    --dataset sift --index hnsw --M 16 --efc 200 --ef 10,20,40 --threads 1
//   vdb_bench filter   --dataset sift --correlated          (Part 6 selectivity sweep)
//   vdb_bench segments --dataset sift --segments 1,2,5,10,50 (QPS vs segment count)
//   vdb_bench deletes  --dataset sift --deleted 0,10,30,50   (delete decay, before/after compaction)
//   vdb_bench ingest   --dim 128 --n 20000                   (sync modes x writers)
//   vdb_bench recovery --dim 128 --log-mb 10,100,1000        (recovery time vs log size)
//   vdb_bench subset   --dataset sift --n 100000 --name sift-100k
//
// Datasets live in <data-dir>/<name>/{base.fbin,query.fbin,gt.ibin} (tools/convert_hdf5.py).
// Every command prints a table and writes raw JSON (with machine details) to --out.

#include <sys/utsname.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>
#include <set>
#include <sstream>
#include <thread>

#include "distance/distance.h"
#include "index/flat_index.h"
#include "index/hnsw/hnsw_index.h"
#include "index/index.h"
#include "index/quant/kmeans.h"
#include "storage/env.h"
#include "util/dataset.h"
#include "util/thread_pool.h"
#include "util/timer.h"
#include "vdb/collection.h"

namespace vdb::bench {
namespace {

using json = nlohmann::json;

// ------------------------------------------------------------------ helpers

std::vector<size_t> ParseList(const std::string& s) {
  std::vector<size_t> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(std::stoul(item));
  }
  return out;
}

std::string ReadFirstLine(const char* cmd) {
  std::string out;
  if (FILE* p = popen(cmd, "r")) {
    char buf[512];
    if (fgets(buf, sizeof(buf), p) != nullptr) out = buf;
    pclose(p);
  }
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
  return out;
}

json MachineInfo() {
  json m;
  std::ifstream cpu("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpu, line)) {
    if (line.rfind("model name", 0) == 0) {
      m["cpu"] = line.substr(line.find(':') + 2);
      break;
    }
  }
  m["cores"] = std::thread::hardware_concurrency();
  std::ifstream mem("/proc/meminfo");
  while (std::getline(mem, line)) {
    if (line.rfind("MemTotal", 0) == 0) {
      m["ram_kb"] = std::stoull(line.substr(line.find(':') + 1));
      break;
    }
  }
  struct utsname u {};
  if (uname(&u) == 0) m["os"] = std::string(u.sysname) + " " + u.release;
  m["compiler"] = __VERSION__;
  m["kernels"] = Kernels().name;
  const char* commit = std::getenv("VDB_GIT_COMMIT");
  m["git_commit"] = commit != nullptr ? commit : ReadFirstLine("git rev-parse --short HEAD 2>/dev/null");
  return m;
}

double Percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  size_t i = static_cast<size_t>(std::ceil(p / 100.0 * static_cast<double>(v.size()))) ;
  return v[std::min(v.size() - 1, i == 0 ? 0 : i - 1)];
}

void WriteJson(const std::string& path, const json& results, const json& extra = json::object()) {
  if (path.empty()) return;
  json doc = {{"machine", MachineInfo()}, {"results", results}};
  for (auto it = extra.begin(); it != extra.end(); ++it) doc[it.key()] = it.value();
  std::ofstream f(path);
  f << doc.dump(1) << "\n";
  std::printf("wrote %s\n", path.c_str());
}

struct Data {
  std::string name;
  Metric metric = Metric::L2;
  Dataset base, query;
  std::vector<int32_t> gt;
  size_t gt_k = 0;
};

Metric GuessMetric(const std::string& name, const std::string& flag) {
  Metric m = Metric::L2;
  if (!flag.empty() && ParseMetric(flag, &m)) return m;
  if (name.find("glove") != std::string::npos || name.find("angular") != std::string::npos ||
      name.find("rag") != std::string::npos) {
    return Metric::Cosine;
  }
  return Metric::L2;
}

// Exact top-k ids for every query (used for subsets, filters and deletes).
std::vector<int32_t> ExactGroundTruth(const Dataset& base, const Dataset& q, Metric metric, size_t k,
                                      ThreadPool* pool, const RowFilter* filter = nullptr) {
  std::vector<int32_t> gt(q.rows * k, -1);
  DistFn fn = MetricDistance(metric);
  if (filter == nullptr) {
    std::vector<std::vector<Neighbor>> res;
    FlatSearchBatch(base.view(), fn, q.buffer.floats(), q.rows, k, res, pool);
    for (size_t i = 0; i < q.rows; ++i)
      for (size_t j = 0; j < res[i].size(); ++j) gt[i * k + j] = static_cast<int32_t>(res[i][j].id);
    return gt;
  }
  pool->parallel_for(0, q.rows, [&](size_t i) {
    std::vector<Neighbor> out;
    FlatSearch(base.view(), fn, q.row(i), k, filter, out, nullptr);
    for (size_t j = 0; j < out.size(); ++j) gt[i * k + j] = static_cast<int32_t>(out[j].id);
  });
  return gt;
}

Status LoadData(const std::string& root, const std::string& name, const std::string& metric_flag,
                size_t max_base, size_t max_query, ThreadPool* pool, Data* d) {
  d->name = name;
  d->metric = GuessMetric(name, metric_flag);
  bool norm = d->metric == Metric::Cosine;
  std::string dir = JoinPath(root, name);
  Timer t;
  VDB_RETURN_IF_ERROR(ReadFbin(JoinPath(dir, "base.fbin"), max_base, norm, &d->base));
  VDB_RETURN_IF_ERROR(ReadFbin(JoinPath(dir, "query.fbin"), max_query, norm, &d->query));
  size_t rows = 0;
  bool subset = false;
  {
    FILE* f = std::fopen(JoinPath(dir, "base.fbin").c_str(), "rb");
    uint32_t hdr[2] = {0, 0};
    if (f != nullptr) {
      if (std::fread(hdr, 4, 2, f) != 2) hdr[0] = 0;
      std::fclose(f);
    }
    subset = d->base.rows < hdr[0];
  }
  Status gs = ReadIbin(JoinPath(dir, "gt.ibin"), max_query, &d->gt, &rows, &d->gt_k);
  if (!gs.ok() || subset) {
    std::printf("computing exact ground truth for %zu x %zu ...\n", d->query.rows, d->base.rows);
    d->gt_k = 100;
    d->gt = ExactGroundTruth(d->base, d->query, d->metric, 100, pool);
  }
  std::printf("%s: %zu base x %zu dims, %zu queries, metric %s, loaded in %.1f s (%zu zero rows)\n",
              name.c_str(), d->base.rows, d->base.dim, d->query.rows, MetricName(d->metric), t.seconds(),
              d->base.zero_rows);
  return Status::OK();
}

double RecallAt(const Data& d, const std::vector<std::vector<VectorId>>& found, size_t k) {
  if (k > d.gt_k) return -1;
  double sum = 0;
  for (size_t q = 0; q < found.size(); ++q) {
    std::set<int32_t> truth(d.gt.begin() + static_cast<std::ptrdiff_t>(q * d.gt_k),
                            d.gt.begin() + static_cast<std::ptrdiff_t>(q * d.gt_k + k));
    size_t hit = 0;
    for (size_t i = 0; i < std::min(k, found[q].size()); ++i) hit += truth.count(static_cast<int32_t>(found[q][i]));
    sum += static_cast<double>(hit) / static_cast<double>(k);
  }
  return sum / static_cast<double>(found.size());
}

// Runs every query single-threaded (exact per-query latency), then again on all threads.
// fn(query index, results) returns the distance computations of that query.
using QueryFn = std::function<uint64_t(size_t, std::vector<VectorId>*)>;

json Measure(const Data& d, size_t k, ThreadPool* all, const QueryFn& fn, bool multi) {
  const size_t nq = d.query.rows;
  std::vector<VectorId> scratch;
  for (size_t i = 0; i < std::min<size_t>(nq, 1000); ++i) fn(i, &scratch);  // warm-up
  std::vector<std::vector<VectorId>> found(nq);
  std::vector<double> lat(nq);
  uint64_t dc = 0;
  Timer total;
  for (size_t i = 0; i < nq; ++i) {
    Timer t;
    dc += fn(i, &found[i]);
    lat[i] = t.millis();
  }
  double single_s = total.seconds();
  json r = {{"k", k},
            {"recall@1", RecallAt(d, found, 1)},
            {"recall@10", RecallAt(d, found, std::min<size_t>(10, k))},
            {"qps", static_cast<double>(nq) / single_s},
            {"p50_ms", Percentile(lat, 50)},
            {"p95_ms", Percentile(lat, 95)},
            {"p99_ms", Percentile(lat, 99)},
            {"dist_comps", static_cast<double>(dc) / static_cast<double>(nq)}};
  if (k >= 100) r["recall@100"] = RecallAt(d, found, 100);
  if (multi && all != nullptr && all->parallelism() > 1) {
    Timer tm;
    all->parallel_for(0, nq, [&](size_t i) {
      std::vector<VectorId> out;
      fn(i, &out);
    }, 16);
    r["qps_all_threads"] = static_cast<double>(nq) / tm.seconds();
    r["threads_all"] = all->parallelism();
  }
  return r;
}

void PrintRow(const json& r) {
  std::printf("  %-34s recall@10 %.4f  qps %9.1f  p99 %7.3f ms  dist %8.0f", r["label"].get<std::string>().c_str(),
              r["recall@10"].get<double>(), r["qps"].get<double>(), r["p99_ms"].get<double>(), r["dist_comps"].get<double>());
  if (r.contains("qps_all_threads")) std::printf("  qps(all) %9.1f", r["qps_all_threads"].get<double>());
  std::printf("\n");
}

struct Common {
  std::string data_dir = "data";
  std::string dataset = "sift";
  std::string metric;
  std::string out;
  std::string work_dir = "/tmp/vdb_bench_work";
  size_t max_base = 0;
  size_t max_query = 10000;
  size_t threads = 0;  // all-thread runs and builds; 0 = all cores
  size_t runs = 1;
  size_t k = 10;
};

void AddCommon(CLI::App* app, Common* c) {
  app->add_option("--data-dir", c->data_dir, "Directory with <dataset>/{base,query}.fbin, gt.ibin");
  app->add_option("--dataset", c->dataset, "Dataset name (sift, glove-100, fashion-mnist, ...)");
  app->add_option("--metric", c->metric, "l2, ip or cosine (default: guessed from the name)");
  app->add_option("--out", c->out, "Raw JSON results file");
  app->add_option("--work-dir", c->work_dir, "Scratch directory for collections");
  app->add_option("--max-base", c->max_base, "Use only the first N base vectors (exact GT is recomputed)");
  app->add_option("--max-query", c->max_query, "Use only the first N queries");
  app->add_option("--threads", c->threads, "Threads for builds and the all-thread run (0 = all cores)");
  app->add_option("--runs", c->runs, "Repeat every configuration this many times");
  app->add_option("--k", c->k, "Neighbors per query");
}

// ------------------------------------------------------------------ index

struct IndexArgs {
  std::string index = "hnsw";
  std::string M = "16", efc = "200", ef = "10,20,40,80,160,320,640";
  std::string nprobe = "1,2,4,8,16,32,64,128,256", rerank = "-1", pq_m = "16", nlist = "4096";
  size_t build_threads = 0;
  size_t max_train = 256;  // k-means training points per centroid (FAISS's cap)
  bool multi = true;
};

int RunIndex(const Common& c, const IndexArgs& a) {
  ThreadPool pool(c.threads);
  Data d;
  Status s = LoadData(c.data_dir, c.dataset, c.metric, c.max_base, c.max_query, &pool, &d);
  if (!s.ok()) {
    std::fprintf(stderr, "%s\n", s.ToString().c_str());
    return 1;
  }
  IndexType type{};
  if (!ParseIndexType(a.index, &type)) {
    std::fprintf(stderr, "unknown index %s\n", a.index.c_str());
    return 1;
  }
  ThreadPool build_pool(a.build_threads);
  json results = json::array();
  const bool graph = type == IndexType::Hnsw || type == IndexType::HnswSq8;
  const bool pq = type == IndexType::PqFlat || type == IndexType::IvfPq;
  std::vector<size_t> Ms = graph ? ParseList(a.M) : std::vector<size_t>{0};
  std::vector<size_t> efcs = graph ? ParseList(a.efc) : std::vector<size_t>{0};
  std::vector<size_t> pqms = pq ? ParseList(a.pq_m) : std::vector<size_t>{0};
  std::vector<size_t> nlists = type == IndexType::IvfPq ? ParseList(a.nlist) : std::vector<size_t>{0};
  std::vector<int> reranks;
  for (const auto& x : ParseList(a.rerank.substr(a.rerank[0] == '-' ? 2 : 0))) reranks.push_back(static_cast<int>(x));
  if (a.rerank.rfind("-1", 0) == 0) reranks.insert(reranks.begin(), -1);
  if (reranks.empty()) reranks.push_back(-1);

  for (size_t M : Ms) {
    for (size_t efc : efcs) {
      for (size_t pqm : pqms) {
        for (size_t nlist : nlists) {
          IndexOptions o;
          o.type = type;
          if (M) o.M = M;
          if (efc) o.ef_construction = efc;
          if (pqm) o.pq_m = pqm;
          if (nlist) o.nlist = nlist;
          o.max_train_points_per_centroid = a.max_train;
          auto idx_or = CreateIndex(o, d.metric, d.base.view());
          if (!idx_or.ok()) {
            std::fprintf(stderr, "%s\n", idx_or.status().ToString().c_str());
            return 1;
          }
          std::unique_ptr<Index> idx = std::move(idx_or).value();
          Timer bt;
          s = idx->build(a.build_threads == 1 ? nullptr : &build_pool);
          double build_s = bt.seconds();
          if (!s.ok()) {
            std::fprintf(stderr, "build: %s\n", s.ToString().c_str());
            return 1;
          }
          size_t mem = idx->memory_bytes();
          std::string base_label = std::string(IndexTypeName(type));
          json params = json::object();
          if (graph) params.update({{"M", o.M}, {"efc", o.ef_construction}});
          if (pq) params.update({{"pq_m", o.pq_m}});
          if (type == IndexType::IvfPq) params["nlist"] = o.nlist;
          size_t repaired = 0;
          if (auto* h = dynamic_cast<HnswIndex*>(idx.get())) repaired = h->graph().repaired();
          std::printf("%s %s: built in %.1f s on %zu threads, %.1f MB index (%.1f B/vector)%s\n",
                      base_label.c_str(), params.dump().c_str(), build_s, build_pool.parallelism(), mem / 1e6,
                      static_cast<double>(mem) / static_cast<double>(d.base.rows),
                      repaired > 0 ? (" , repaired " + std::to_string(repaired) + " links").c_str() : "");
          // Search configurations.
          std::vector<SearchParams> sps;
          if (type == IndexType::Flat) {
            sps.emplace_back();
          } else if (graph) {
            for (size_t ef : ParseList(a.ef)) {
              for (int rr : type == IndexType::HnswSq8 ? reranks : std::vector<int>{-1}) {
                SearchParams p;
                p.ef = ef;
                p.rerank = rr;
                sps.push_back(p);
              }
            }
          } else {
            for (size_t np : type == IndexType::IvfPq ? ParseList(a.nprobe) : std::vector<size_t>{1}) {
              for (int rr : reranks) {
                SearchParams p;
                p.nprobe = np;
                p.rerank = rr;
                sps.push_back(p);
              }
            }
          }
          for (const SearchParams& p : sps) {
            for (size_t run = 0; run < c.runs; ++run) {
              QueryFn fn = [&](size_t qi, std::vector<VectorId>* out) {
                std::vector<Neighbor> res;
                IndexSearchStats st;
                idx->search(d.query.row(qi), c.k, p, nullptr, res, &st);
                out->clear();
                for (const auto& n : res) out->push_back(n.id);
                return st.distance_computations;
              };
              json r = Measure(d, c.k, &pool, fn, a.multi);
              json sp = params;
              if (graph) sp["ef"] = p.ef;
              if (type == IndexType::IvfPq) sp["nprobe"] = p.nprobe;
              if (type != IndexType::Flat && type != IndexType::Hnsw) sp["rerank"] = p.rerank;
              std::string label = base_label + " " + sp.dump();
              r.update({{"system", "vdb"},
                        {"index", base_label},
                        {"dataset", d.name},
                        {"params", sp},
                        {"label", label},
                        {"run", run},
                        {"threads", 1},
                        {"build_s", build_s},
                        {"build_threads", a.build_threads == 1 ? 1 : build_pool.parallelism()},
                        {"index_bytes", mem},
                        {"bytes_per_vector", (static_cast<double>(mem) +
                                              (type == IndexType::Hnsw || type == IndexType::Flat
                                                   ? static_cast<double>(d.base.rows * d.base.dim * 4)
                                                   : 0.0)) /
                                                 static_cast<double>(d.base.rows)},
                        {"repaired_links", repaired}});
              PrintRow(r);
              results.push_back(r);
            }
          }
        }
      }
    }
  }
  WriteJson(c.out, results, {{"command", "index"}});
  return 0;
}

// ------------------------------------------------------------------ collection helpers

std::string FreshDir(const Common& c, const std::string& name) {
  std::string p = JoinPath(c.work_dir, name);
  Env::Default()->remove_dir_all(p);
  CreateDirs(Env::Default(), c.work_dir);
  return p;
}

std::vector<float> RealRows(const Dataset& ds, size_t from, size_t to) {
  std::vector<float> out;
  out.reserve((to - from) * ds.dim);
  for (size_t i = from; i < to; ++i) out.insert(out.end(), ds.row(i), ds.row(i) + ds.dim);
  return out;
}

json CollectionSearch(const Data& d, const Collection& col, size_t k, const SearchParams& p,
                      const Filter* filter, ThreadPool* pool, bool multi, std::map<std::string, size_t>* strategies = nullptr) {
  std::mutex mu;
  QueryFn fn = [&](size_t qi, std::vector<VectorId>* out) {
    SearchRequest req;
    req.vector = d.query.row(qi);
    req.k = k;
    req.params = p;
    req.filter = filter;
    std::vector<SearchHit> hits;
    SearchExplain ex;  // always requested: it carries the distance-computation count
    col.Search(req, &hits, &ex);
    out->clear();
    for (const auto& h : hits) out->push_back(h.id);
    if (strategies != nullptr) {
      std::lock_guard<std::mutex> g(mu);
      for (const auto& s : ex.segments)
        if (s.kind == "disk") ++(*strategies)[s.strategy];
    }
    return ex.distance_computations;
  };
  return Measure(d, k, pool, fn, multi);
}

std::unique_ptr<Collection> BulkCollection(const Common& c, const Data& d, const std::string& name,
                                           size_t segments, const IndexOptions& io,
                                           const std::vector<Metadata>* md = nullptr, double* load_s = nullptr) {
  CollectionOptions o;
  o.dim = d.base.dim;
  o.metric = d.metric;
  o.index = io;
  o.sync = SyncMode::None;
  if (md != nullptr) o.schema = {{"attr", FieldType::Int64}};
  OpenOptions oo;
  oo.num_threads = c.threads;
  oo.auto_compact = false;
  oo.idle_compact_ms = 0;
  auto col = Collection::Create(FreshDir(c, name), o, oo);
  if (!col.ok()) {
    std::fprintf(stderr, "%s\n", col.status().ToString().c_str());
    std::exit(1);
  }
  Timer t;
  size_t per = (d.base.rows + segments - 1) / segments;
  for (size_t b = 0; b < d.base.rows; b += per) {
    size_t e = std::min(d.base.rows, b + per);
    std::vector<VectorId> ids;
    for (size_t i = b; i < e; ++i) ids.push_back(i);
    auto rows = RealRows(d.base, b, e);
    std::vector<Metadata> part;
    if (md != nullptr) part.assign(md->begin() + static_cast<std::ptrdiff_t>(b), md->begin() + static_cast<std::ptrdiff_t>(e));
    Status s = (*col)->BulkLoad(ids.data(), rows.data(), ids.size(), md != nullptr ? &part : nullptr);
    if (!s.ok()) {
      std::fprintf(stderr, "bulk load: %s\n", s.ToString().c_str());
      std::exit(1);
    }
  }
  if (load_s != nullptr) *load_s = t.seconds();
  return std::move(col).value();
}

// ------------------------------------------------------------------ filter

class AttrFilter final : public RowFilter {
 public:
  AttrFilter(const std::vector<int>* attr, int t) : attr_(attr), t_(t) {}
  bool allowed(InternalId id) const override { return (*attr_)[id] < t_; }

 private:
  const std::vector<int>* attr_;
  int t_;
};

int RunFilter(const Common& c, bool correlated, size_t ef, const std::string& thresholds) {
  ThreadPool pool(c.threads);
  Data d;
  Status s = LoadData(c.data_dir, c.dataset, c.metric, c.max_base, c.max_query, &pool, &d);
  if (!s.ok()) {
    std::fprintf(stderr, "%s\n", s.ToString().c_str());
    return 1;
  }
  std::vector<int> attr(d.base.rows);
  if (correlated) {
    // attr = k-means cluster of the row: a filter selects one region of the space.
    std::vector<float> cen;
    KMeansOptions ko;
    ko.max_points_per_centroid = 64;
    s = KMeans(d.base.buffer.floats(), d.base.rows, d.base.padded_dim, d.base.padded_dim, 1000, ko, &pool, &cen);
    std::vector<uint32_t> assign(d.base.rows);
    AssignNearest(d.base.buffer.floats(), d.base.rows, d.base.padded_dim, d.base.padded_dim, cen.data(), 1000, &pool,
                  assign.data(), nullptr);
    for (size_t i = 0; i < d.base.rows; ++i) attr[i] = static_cast<int>(assign[i]);
  } else {
    std::mt19937_64 rng(12345);
    for (auto& a : attr) a = static_cast<int>(rng() % 1000);
  }
  std::vector<Metadata> md(d.base.rows);
  for (size_t i = 0; i < d.base.rows; ++i) md[i] = {{"attr", int64_t{attr[i]}}};
  IndexOptions io;
  double load_s = 0;
  auto col = BulkCollection(c, d, "filter", 1, io, &md, &load_s);
  std::printf("bulk-loaded with attributes in %.1f s (%s)\n", load_s, correlated ? "correlated" : "uncorrelated");
  json results = json::array();
  for (size_t t : ParseList(thresholds)) {
    AttrFilter af(&attr, static_cast<int>(t));
    size_t matches = 0;
    for (int a : attr) matches += a < static_cast<int>(t);
    Data fd;  // same data with the filtered ground truth
    fd.name = d.name;
    fd.metric = d.metric;
    fd.gt_k = 10;
    fd.gt = ExactGroundTruth(d.base, d.query, d.metric, 10, &pool, &af);
    auto filter = col->ParseFilter(R"({"field": "attr", "lt": )" + std::to_string(t) + "}");
    double sel = static_cast<double>(matches) / static_cast<double>(d.base.rows);
    for (const auto& [sname, strat] : std::vector<std::pair<std::string, FilterStrategy>>{
             {"planner", FilterStrategy::Auto},
             {"prefilter", FilterStrategy::PreFilter},
             {"graph", FilterStrategy::Graph},
             {"postfilter", FilterStrategy::PostFilter}}) {
      SearchParams p;
      p.ef = ef;
      p.strategy = strat;
      std::map<std::string, size_t> used;
      // Measure() reads queries and ground truth from a Data; share the vectors.
      Data view;
      view.name = d.name;
      view.metric = d.metric;
      view.gt = fd.gt;
      view.gt_k = 10;
      view.query.rows = d.query.rows;
      view.query.dim = d.query.dim;
      view.query.padded_dim = d.query.padded_dim;
      view.query.buffer.reset(d.query.rows * d.query.padded_dim * sizeof(float));
      std::memcpy(view.query.buffer.data(), d.query.buffer.data(), d.query.buffer.size());
      json r = CollectionSearch(view, *col, c.k, p, filter->get(), &pool, false, &used);
      json strategies = json::object();
      for (const auto& [k2, v] : used) strategies[k2] = v;
      r.update({{"system", "vdb"},
                {"dataset", d.name},
                {"correlated", correlated},
                {"threshold", t},
                {"selectivity", sel},
                {"strategy", sname},
                {"ef", ef},
                {"strategies_used", strategies},
                {"label", sname + " sel=" + std::to_string(sel)}});
      PrintRow(r);
      results.push_back(r);
    }
  }
  WriteJson(c.out, results, {{"command", "filter"}});
  return 0;
}

// ------------------------------------------------------------------ segments

int RunSegments(const Common& c, const std::string& counts, const std::string& efs) {
  ThreadPool pool(c.threads);
  Data d;
  Status s = LoadData(c.data_dir, c.dataset, c.metric, c.max_base, c.max_query, &pool, &d);
  if (!s.ok()) {
    std::fprintf(stderr, "%s\n", s.ToString().c_str());
    return 1;
  }
  json results = json::array();
  for (size_t segs : ParseList(counts)) {
    double load_s = 0;
    auto col = BulkCollection(c, d, "segments", segs, IndexOptions{}, nullptr, &load_s);
    std::printf("%zu segments loaded in %.1f s\n", segs, load_s);
    for (size_t ef : ParseList(efs)) {
      SearchParams p;
      p.ef = ef;
      json r = CollectionSearch(d, *col, c.k, p, nullptr, &pool, false);
      r.update({{"system", "vdb"}, {"dataset", d.name}, {"segments", segs}, {"ef", ef}, {"load_s", load_s},
                {"label", std::to_string(segs) + " segments ef=" + std::to_string(ef)}});
      PrintRow(r);
      results.push_back(r);
    }
  }
  WriteJson(c.out, results, {{"command", "segments"}});
  return 0;
}

// ------------------------------------------------------------------ deletes

int RunDeletes(const Common& c, const std::string& fractions, size_t ef) {
  ThreadPool pool(c.threads);
  Data d;
  Status s = LoadData(c.data_dir, c.dataset, c.metric, c.max_base, c.max_query, &pool, &d);
  if (!s.ok()) {
    std::fprintf(stderr, "%s\n", s.ToString().c_str());
    return 1;
  }
  json results = json::array();
  for (size_t pct : ParseList(fractions)) {
    auto col = BulkCollection(c, d, "deletes", 1, IndexOptions{});
    std::mt19937_64 rng(pct + 1);
    std::vector<VectorId> del;
    std::vector<uint8_t> dead(d.base.rows, 0);
    for (VectorId i = 0; i < d.base.rows; ++i) {
      if (rng() % 100 < pct) {
        del.push_back(i);
        dead[i] = 1;
      }
    }
    if (!del.empty()) col->RemoveBatch(del.data(), del.size());
    class Live final : public RowFilter {
     public:
      explicit Live(const std::vector<uint8_t>* dd) : d_(dd) {}
      bool allowed(InternalId id) const override { return (*d_)[id] == 0; }

     private:
      const std::vector<uint8_t>* d_;
    } live(&dead);
    Data ld;
    ld.name = d.name;
    ld.metric = d.metric;
    ld.gt_k = 10;
    ld.gt = ExactGroundTruth(d.base, d.query, d.metric, 10, &pool, &live);
    ld.query.rows = d.query.rows;
    ld.query.dim = d.query.dim;
    ld.query.padded_dim = d.query.padded_dim;
    ld.query.buffer.reset(d.query.buffer.size());
    std::memcpy(ld.query.buffer.data(), d.query.buffer.data(), d.query.buffer.size());
    SearchParams p;
    p.ef = ef;
    for (int phase = 0; phase < 2; ++phase) {
      Timer ct;
      if (phase == 1) col->Compact();
      double compact_s = phase == 1 ? ct.seconds() : 0.0;
      json r = CollectionSearch(ld, *col, c.k, p, nullptr, &pool, false);
      r.update({{"system", "vdb"}, {"dataset", d.name}, {"deleted_pct", pct}, {"compacted", phase == 1},
                {"ef", ef}, {"compact_s", compact_s},
                {"label", std::to_string(pct) + "% deleted" + (phase == 1 ? " (compacted)" : "")}});
      PrintRow(r);
      results.push_back(r);
    }
  }
  WriteJson(c.out, results, {{"command", "deletes"}});
  return 0;
}

// ------------------------------------------------------------------ ingest

int RunIngest(const Common& c, size_t dim, size_t n, const std::string& writers_list, const std::string& batches) {
  json results = json::array();
  std::mt19937_64 rng(1);
  std::normal_distribution<float> g;
  std::vector<float> pool_vecs(4096 * dim);
  for (auto& x : pool_vecs) x = g(rng);
  struct Mode {
    std::string name;
    SyncMode sync;
    bool group;
  };
  const Mode modes[] = {{"always+group_commit", SyncMode::Always, true},
                        {"always", SyncMode::Always, false},
                        {"interval", SyncMode::Interval, true},
                        {"none", SyncMode::None, true}};
  for (const Mode& m : modes) {
    for (size_t writers : ParseList(writers_list)) {
      for (size_t batch : ParseList(batches)) {
        CollectionOptions o;
        o.dim = dim;
        o.sync = m.sync;
        OpenOptions oo;
        oo.num_threads = c.threads;
        oo.group_commit = m.group;
        oo.auto_compact = false;
        auto col = Collection::Create(FreshDir(c, "ingest"), o, oo);
        if (!col.ok()) return 1;
        std::atomic<size_t> next{0};
        std::vector<std::vector<double>> lat(writers);
        Timer t;
        std::vector<std::thread> th;
        for (size_t w = 0; w < writers; ++w) {
          th.emplace_back([&, w] {
            std::vector<VectorId> ids(batch);
            for (;;) {
              size_t start = next.fetch_add(batch);
              if (start >= n) return;
              size_t cnt = std::min(batch, n - start);
              for (size_t i = 0; i < cnt; ++i) ids[i] = start + i;
              Timer lt;
              Status s = (*col)->UpsertBatch(ids.data(), pool_vecs.data() + (start % 3000) * dim, cnt);
              lat[w].push_back(lt.millis());
              if (!s.ok()) return;
            }
          });
        }
        for (auto& x : th) x.join();
        double secs = t.seconds();
        std::vector<double> all;
        for (auto& l : lat) all.insert(all.end(), l.begin(), l.end());
        json r = {{"system", "vdb"},
                  {"sync", m.name},
                  {"writers", writers},
                  {"batch", batch},
                  {"vectors", n},
                  {"vectors_per_s", static_cast<double>(n) / secs},
                  {"p50_ms", Percentile(all, 50)},
                  {"p99_ms", Percentile(all, 99)}};
        std::printf("  %-22s writers %3zu batch %4zu: %10.0f vectors/s  p50 %7.3f ms  p99 %7.3f ms\n", m.name.c_str(),
                    writers, batch, r["vectors_per_s"].get<double>(), r["p50_ms"].get<double>(), r["p99_ms"].get<double>());
        results.push_back(r);
      }
    }
  }
  WriteJson(c.out, results, {{"command", "ingest"}});
  return 0;
}

// ------------------------------------------------------------------ recovery

int RunRecovery(const Common& c, size_t dim, const std::string& log_mbs, size_t checksum_rows) {
  json results = json::array();
  std::mt19937_64 rng(2);
  std::normal_distribution<float> g;
  std::vector<float> vecs(1000 * dim);
  for (auto& x : vecs) x = g(rng);
  const double record_bytes = 17.0 + 8 + 4 + dim * 4.0 + 4 + 2;
  for (size_t mb : ParseList(log_mbs)) {
    size_t n = static_cast<size_t>(static_cast<double>(mb) * 1e6 / record_bytes);
    std::string dir = FreshDir(c, "recovery");
    CollectionOptions o;
    o.dim = dim;
    o.sync = SyncMode::None;
    OpenOptions oo;
    oo.num_threads = c.threads;
    oo.auto_flush = false;  // everything stays in the log
    oo.auto_compact = false;
    {
      auto col = Collection::Create(dir, o, oo);
      if (!col.ok()) return 1;
      std::vector<VectorId> ids(1000);
      for (size_t b = 0; b < n; b += 1000) {
        size_t cnt = std::min<size_t>(1000, n - b);
        for (size_t i = 0; i < cnt; ++i) ids[i] = b + i;
        (*col)->UpsertBatch(ids.data(), vecs.data(), cnt);
      }
    }
    if (std::system("sync") != 0) std::printf("sync failed\n");
    Timer t;
    auto col = Collection::Open(dir, oo);
    double secs = t.seconds();
    if (!col.ok()) return 1;
    CollectionStats st = (*col)->Stats();
    json r = {{"system", "vdb"}, {"kind", "log_replay"}, {"log_mb", mb}, {"records", st.last_recovery_records},
              {"seconds", secs}};
    std::printf("  log %5zu MB (%zu records): ready in %.3f s\n", mb, n, secs);
    results.push_back(r);
  }
  if (checksum_rows > 0) {
    std::string dir = FreshDir(c, "recovery_crc");
    CollectionOptions o;
    o.dim = dim;
    o.sync = SyncMode::None;
    OpenOptions oo;
    oo.num_threads = c.threads;
    oo.auto_compact = false;
    {
      auto col = Collection::Create(dir, o, oo);
      if (!col.ok()) return 1;
      std::vector<VectorId> ids(checksum_rows);
      std::vector<float> rows(checksum_rows * dim);
      for (size_t i = 0; i < checksum_rows; ++i) {
        ids[i] = i;
        std::memcpy(rows.data() + i * dim, vecs.data() + (i % 1000) * dim, dim * 4);
      }
      (*col)->BulkLoad(ids.data(), rows.data(), checksum_rows);
    }
    for (bool verify : {false, true}) {
      oo.verify_checksums = verify;
      Timer t;
      auto col = Collection::Open(dir, oo);
      double secs = t.seconds();
      json r = {{"system", "vdb"}, {"kind", "segment_open"}, {"rows", checksum_rows}, {"verify_crc", verify},
                {"seconds", secs}};
      std::printf("  open %zu-row segment, crc %s: %.3f s\n", checksum_rows, verify ? "on " : "off", secs);
      results.push_back(r);
    }
  }
  WriteJson(c.out, results, {{"command", "recovery"}});
  return 0;
}

// ------------------------------------------------------------------ mutable segment size

// The design alternative to a flat mutable segment: index its rows with a small HNSW graph,
// inserted one at a time as they arrive. Measures, on the same n rows, the per-row insert
// cost that would land on the write path, and both structures' search latency.
json CompareIncrementalHnsw(const Data& d, size_t first, size_t n, size_t k, size_t ef) {
  VectorView rows = d.base.view();
  rows.data = d.base.view().row(first);
  rows.rows = n;
  IndexOptions o;  // M = 16, ef_construction = 200
  IndexOptions fo;
  fo.type = IndexType::Flat;
  auto flat = CreateIndex(fo, d.metric, rows);
  auto hnsw = CreateIndex(o, d.metric, rows);
  if (!flat.ok() || !hnsw.ok()) return json::object();
  (*flat)->build(nullptr);
  Timer bt;
  (*hnsw)->build(nullptr);  // single thread: rows are inserted in arrival order
  const double insert_us = bt.seconds() * 1e6 / static_cast<double>(n);
  SearchParams p;
  p.ef = ef;
  const size_t nq = std::min<size_t>(d.query.rows, 1000);
  std::vector<double> flat_ms, hnsw_ms;
  size_t agree = 0;
  for (size_t qi = 0; qi < nq; ++qi) {
    std::vector<Neighbor> a, b;
    IndexSearchStats st;
    Timer t1;
    (*flat)->search(d.query.row(qi), k, p, nullptr, a, &st);
    flat_ms.push_back(t1.millis());
    Timer t2;
    (*hnsw)->search(d.query.row(qi), k, p, nullptr, b, &st);
    hnsw_ms.push_back(t2.millis());
    std::set<InternalId> exact;
    for (const auto& x : a) exact.insert(x.id);
    for (const auto& x : b) agree += exact.count(x.id);
  }
  json j = {{"hnsw_insert_us_per_row", insert_us},
            {"flat_p50_ms", Percentile(flat_ms, 50)},
            {"hnsw_p50_ms", Percentile(hnsw_ms, 50)},
            {"hnsw_recall_vs_flat", static_cast<double>(agree) / static_cast<double>(nq * k)}};
  std::printf("    vs incremental HNSW on the same %zu rows: insert %.0f us/row, search p50 %.3f ms "
              "(flat %.3f ms), recall vs flat %.4f\n",
              n, insert_us, j["hnsw_p50_ms"].get<double>(), j["flat_p50_ms"].get<double>(),
              j["hnsw_recall_vs_flat"].get<double>());
  return j;
}

// Every query scans the whole mutable segment, so its size is a tax on every search. Bulk
// load all but the last max(N) rows, upsert N rows that stay in the mutable segment, and
// measure the latency the scan adds; then time the flush of those N rows.
int RunMutable(const Common& c, const std::string& sizes, size_t ef) {
  ThreadPool pool(c.threads);
  Data d;
  Status s = LoadData(c.data_dir, c.dataset, c.metric, c.max_base, c.max_query, &pool, &d);
  if (!s.ok()) {
    std::fprintf(stderr, "%s\n", s.ToString().c_str());
    return 1;
  }
  std::vector<size_t> ns = ParseList(sizes);
  size_t extra = *std::max_element(ns.begin(), ns.end());
  if (extra >= d.base.rows) {
    std::fprintf(stderr, "need more base rows than the largest mutable size\n");
    return 1;
  }
  const size_t bulk = d.base.rows - extra;
  json results = json::array();
  ns.insert(ns.begin(), 0);
  for (size_t n : ns) {
    CollectionOptions o;
    o.dim = d.base.dim;
    o.metric = d.metric;
    o.sync = SyncMode::None;
    OpenOptions oo;
    oo.num_threads = c.threads;
    oo.auto_flush = false;
    oo.auto_compact = false;
    oo.idle_compact_ms = 0;
    auto col = Collection::Create(FreshDir(c, "mutable"), o, oo);
    if (!col.ok()) return 1;
    std::vector<VectorId> ids(bulk);
    for (size_t i = 0; i < bulk; ++i) ids[i] = i;
    auto rows = RealRows(d.base, 0, bulk);
    (*col)->BulkLoad(ids.data(), rows.data(), bulk);
    for (size_t b = bulk; b < bulk + n; b += 1000) {
      size_t e = std::min(bulk + n, b + 1000);
      std::vector<VectorId> ui;
      for (size_t i = b; i < e; ++i) ui.push_back(i);
      auto ur = RealRows(d.base, b, e);
      (*col)->UpsertBatch(ui.data(), ur.data(), ui.size());
    }
    SearchParams p;
    p.ef = ef;
    json r = CollectionSearch(d, **col, c.k, p, nullptr, &pool, false);
    Timer ft;
    (*col)->Flush();
    r.update({{"system", "vdb"}, {"mutable_rows", n}, {"mutable_mb", n * d.base.padded_dim * 4 / 1e6},
              {"flush_s", n > 0 ? ft.seconds() : 0.0}, {"segments_after_flush", (*col)->Stats().segments.size() - 1},
              {"label", std::to_string(n) + " rows in the mutable segment"}});
    if (n > 0) r["alternative"] = CompareIncrementalHnsw(d, bulk, n, c.k, ef);
    PrintRow(r);
    results.push_back(r);
  }
  WriteJson(c.out, results, {{"command", "mutable"}});
  return 0;
}

// ------------------------------------------------------------------ search loop (for perf)

// Single-threaded searches for a fixed time against a collection kept in --work-dir, so a
// profiler sees only the query path: the first run builds and saves the collection, later
// runs just open it. Prints "searching" when the timed loop starts.
int RunSearchLoop(const Common& c, const std::string& index, size_t ef, size_t nprobe, double seconds) {
  ThreadPool pool(c.threads);
  Data d;
  Status s = LoadData(c.data_dir, c.dataset, c.metric, c.max_base, c.max_query, &pool, &d);
  if (!s.ok()) {
    std::fprintf(stderr, "%s\n", s.ToString().c_str());
    return 1;
  }
  IndexOptions io;
  if (!ParseIndexType(index, &io.type)) {
    std::fprintf(stderr, "unknown index %s\n", index.c_str());
    return 1;
  }
  io.nlist = 4096;
  io.max_train_points_per_centroid = 64;
  const std::string name = "searchloop_" + c.dataset + "_" + index;
  const std::string dir = JoinPath(c.work_dir, name);
  const std::string built = JoinPath(c.work_dir, name + ".built");  // written only after a complete build
  if (!Env::Default()->exists(built)) {
    std::printf("building %s over %zu rows (one-time)\n", index.c_str(), d.base.rows);
    auto col = BulkCollection(c, d, name, 1, io);
    if (!col->Close().ok()) return 1;
    std::ofstream(built) << d.base.rows << "\n";
  }
  OpenOptions oo;
  oo.num_threads = 1;
  oo.auto_compact = false;
  oo.idle_compact_ms = 0;
  auto col = Collection::Open(dir, oo);
  if (!col.ok()) {
    std::fprintf(stderr, "%s\n", col.status().ToString().c_str());
    return 1;
  }
  SearchRequest req;
  req.k = c.k;
  req.params.ef = ef;
  req.params.nprobe = nprobe;
  std::vector<SearchHit> hits;
  size_t queries = 0, found = 0;
  std::printf("searching\n");
  std::fflush(stdout);
  Timer t;
  while (t.seconds() < seconds) {
    for (size_t qi = 0; qi < d.query.rows; ++qi) {
      req.vector = d.query.row(qi);
      (*col)->Search(req, &hits, nullptr);
      found += hits.size();
      ++queries;
    }
  }
  const double secs = t.seconds();
  std::printf("%zu queries in %.1f s: %.0f QPS (%zu hits)\n", queries, secs, queries / secs, found);
  return 0;
}

// ------------------------------------------------------------------ graph reordering

// Stretch goal: renumber nodes in BFS order from the entry point so neighbors sit close
// together in memory, rewrite vectors and links in that order, and compare QPS and
// distance computations before and after (same graph, same answers).
int RunReorder(const Common& c, const std::string& efs, size_t M, size_t efc) {
  ThreadPool pool(c.threads);
  Data d;
  Status s = LoadData(c.data_dir, c.dataset, c.metric, c.max_base, c.max_query, &pool, &d);
  if (!s.ok()) {
    std::fprintf(stderr, "%s\n", s.ToString().c_str());
    return 1;
  }
  using Graph = HnswGraph<FloatSpace>;
  DistFn fn = MetricDistance(d.metric);
  FloatSpace sp{d.base.buffer.floats(), d.base.padded_dim, fn};
  Graph g(sp, d.base.rows, M, efc, 42);
  Timer bt;
  g.build(0, d.base.rows, &pool);
  std::printf("built in %.1f s\n", bt.seconds());
  Timer rt;
  std::vector<uint32_t> order = g.bfs_order();
  AlignedBuffer permuted(d.base.rows * d.base.padded_dim * sizeof(float));
  for (size_t i = 0; i < order.size(); ++i) {
    std::memcpy(permuted.floats() + i * d.base.padded_dim, d.base.row(order[i]), d.base.padded_dim * sizeof(float));
  }
  FloatSpace sp2{permuted.floats(), d.base.padded_dim, fn};
  auto g2 = g.renumbered(order, sp2);
  double reorder_s = rt.seconds();
  std::printf("reordered in %.1f s\n", reorder_s);
  json results = json::array();
  for (size_t ef : ParseList(efs)) {
    for (int variant = 0; variant < 2; ++variant) {
      const Graph& gg = variant == 0 ? g : *g2;
      for (size_t run = 0; run < c.runs; ++run) {
        QueryFn qf = [&](size_t qi, std::vector<VectorId>* out) {
          std::vector<Neighbor> res;
          uint64_t dc = 0;
          gg.search(d.query.row(qi), c.k, std::max(ef, c.k), nullptr, res, &dc);
          out->clear();
          for (const auto& nb : res) out->push_back(variant == 0 ? nb.id : order[nb.id]);
          return dc;
        };
        json r = Measure(d, c.k, &pool, qf, false);
        r.update({{"system", "vdb"}, {"dataset", d.name}, {"ef", ef}, {"reordered", variant == 1}, {"run", run},
                  {"reorder_s", reorder_s},
                  {"label", std::string(variant ? "bfs-reordered" : "insertion order") + " ef=" + std::to_string(ef)}});
        PrintRow(r);
        results.push_back(r);
      }
    }
  }
  WriteJson(c.out, results, {{"command", "reorder"}});
  return 0;
}

// ------------------------------------------------------------------ subset

int RunSubset(const Common& c, size_t n, const std::string& name) {
  ThreadPool pool(c.threads);
  Data d;
  Status s = LoadData(c.data_dir, c.dataset, c.metric, n, c.max_query, &pool, &d);
  if (!s.ok()) {
    std::fprintf(stderr, "%s\n", s.ToString().c_str());
    return 1;
  }
  std::string dir = JoinPath(c.data_dir, name);
  CreateDirs(Env::Default(), dir);
  WriteFbin(JoinPath(dir, "base.fbin"), d.base.buffer.floats(), d.base.rows, d.base.dim, d.base.padded_dim);
  WriteFbin(JoinPath(dir, "query.fbin"), d.query.buffer.floats(), d.query.rows, d.query.dim, d.query.padded_dim);
  WriteIbin(JoinPath(dir, "gt.ibin"), d.gt.data(), d.query.rows, d.gt_k);
  std::printf("wrote %s (%zu base, %zu queries, exact top-%zu)\n", dir.c_str(), d.base.rows, d.query.rows, d.gt_k);
  return 0;
}

}  // namespace
}  // namespace vdb::bench

int main(int argc, char** argv) {
  using namespace vdb::bench;
  CLI::App app{"vdb_bench: benchmarks for the vdb vector database"};
  app.require_subcommand(1);
  Common common;

  IndexArgs ia;
  auto* idx = app.add_subcommand("index", "Recall/QPS sweep of one index type");
  AddCommon(idx, &common);
  idx->add_option("--index", ia.index, "flat, hnsw, hnsw_sq8, pq, ivf_pq");
  idx->add_option("--M", ia.M, "HNSW M values, comma separated");
  idx->add_option("--efc", ia.efc, "ef_construction values");
  idx->add_option("--ef", ia.ef, "ef_search values");
  idx->add_option("--nprobe", ia.nprobe, "IVF nprobe values");
  idx->add_option("--rerank", ia.rerank, "Re-rank depths (-1 = auto, 0 = none)");
  idx->add_option("--pq-m", ia.pq_m, "PQ sub-vector counts");
  idx->add_option("--nlist", ia.nlist, "IVF list counts");
  idx->add_option("--build-threads", ia.build_threads, "Threads for the build (0 = all, 1 = single)");
  idx->add_option("--max-train", ia.max_train, "k-means training points per centroid (IVF/PQ)");
  idx->add_flag("!--no-multi", ia.multi, "Skip the all-thread QPS run");

  bool correlated = false;
  size_t filter_ef = 64;
  std::string thresholds = "1,10,100,500,1000";
  auto* flt = app.add_subcommand("filter", "Filtered search vs selectivity, per strategy");
  AddCommon(flt, &common);
  flt->add_flag("--correlated", correlated, "attr = k-means cluster instead of random");
  flt->add_option("--ef", filter_ef, "Base ef");
  flt->add_option("--thresholds", thresholds, "attr < t for each t (attr in [0, 1000))");

  std::string seg_counts = "1,2,5,10,50", seg_efs = "16,32,64,128,256";
  auto* seg = app.add_subcommand("segments", "QPS vs number of segments");
  AddCommon(seg, &common);
  seg->add_option("--segments", seg_counts, "Segment counts");
  seg->add_option("--ef", seg_efs, "ef values");

  std::string fractions = "0,10,30,50";
  size_t del_ef = 64;
  auto* del = app.add_subcommand("deletes", "Recall and QPS vs deleted fraction, before/after compaction");
  AddCommon(del, &common);
  del->add_option("--deleted", fractions, "Percent deleted");
  del->add_option("--ef", del_ef, "ef");

  size_t dim = 128, n = 20000, checksum_rows = 200000;
  std::string writers = "1,8,64", batches = "1,100", log_mbs = "10,100,1000";
  auto* ing = app.add_subcommand("ingest", "Write throughput per sync mode and writer count");
  AddCommon(ing, &common);
  ing->add_option("--dim", dim);
  ing->add_option("--n", n, "Vectors per configuration");
  ing->add_option("--writers", writers);
  ing->add_option("--batch", batches);

  auto* rec = app.add_subcommand("recovery", "Recovery time vs log size; startup with and without CRC checks");
  AddCommon(rec, &common);
  rec->add_option("--dim", dim);
  rec->add_option("--log-mb", log_mbs);
  rec->add_option("--checksum-rows", checksum_rows, "Rows in the segment for the CRC startup test (0 = skip)");

  std::string mutable_sizes = "5000,10000,20000,100000";
  size_t mutable_ef = 64;
  auto* mut = app.add_subcommand("mutable", "Query latency added by the mutable segment's scan, and flush time");
  AddCommon(mut, &common);
  mut->add_option("--rows", mutable_sizes, "Rows left in the mutable segment");
  mut->add_option("--ef", mutable_ef);

  std::string reorder_efs = "16,32,64,128";
  size_t reorder_M = 16, reorder_efc = 200;
  auto* reo = app.add_subcommand("reorder", "HNSW graph reordering (BFS) for cache locality, before vs after");
  AddCommon(reo, &common);
  reo->add_option("--ef", reorder_efs);
  reo->add_option("--M", reorder_M);
  reo->add_option("--efc", reorder_efc);

  size_t subset_n = 100000;
  std::string subset_name = "sift-100k";
  auto* sub = app.add_subcommand("subset", "Write the first N base vectors with exact ground truth");
  AddCommon(sub, &common);
  sub->add_option("--n", subset_n);
  sub->add_option("--name", subset_name);

  std::string loop_index = "hnsw";
  size_t loop_ef = 80, loop_nprobe = 64;
  double loop_seconds = 20;
  auto* lp = app.add_subcommand("search-loop", "Timed single-threaded searches on a saved collection (for perf)");
  AddCommon(lp, &common);
  lp->add_option("--index", loop_index, "hnsw, hnsw_sq8, ivf_pq, ...");
  lp->add_option("--ef", loop_ef);
  lp->add_option("--nprobe", loop_nprobe);
  lp->add_option("--seconds", loop_seconds);

  CLI11_PARSE(app, argc, argv);
  if (*lp) return RunSearchLoop(common, loop_index, loop_ef, loop_nprobe, loop_seconds);
  if (*idx) return RunIndex(common, ia);
  if (*flt) return RunFilter(common, correlated, filter_ef, thresholds);
  if (*seg) return RunSegments(common, seg_counts, seg_efs);
  if (*del) return RunDeletes(common, fractions, del_ef);
  if (*ing) return RunIngest(common, dim, n, writers, batches);
  if (*rec) return RunRecovery(common, dim, log_mbs, checksum_rows);
  if (*sub) return RunSubset(common, subset_n, subset_name);
  if (*mut) return RunMutable(common, mutable_sizes, mutable_ef);
  if (*reo) return RunReorder(common, reorder_efs, reorder_M, reorder_efc);
  return 1;
}
