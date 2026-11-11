#pragma once

// Hierarchical navigable small-world graph (Malkov & Yashunin, Algorithms 1-5).
//
// The graph is generic over a Space, which answers three questions: the distance from a
// float query to a stored node, a float view of a stored node (to use it as a query
// while inserting), and how to prefetch a node. FloatSpace stores padded float rows;
// Sq8Space (hnsw_index.h) stores 8-bit codes.
//
// Layout: layer 0 is one fixed block per node, [count][links x M0], contiguous for cache
// locality. Upper layers are allocated per node: levels 1..level, each [count][links x M].

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "distance/distance.h"
#include "index/hnsw/visited_pool.h"
#include "index/index.h"
#include "util/coding.h"
#include "util/rng.h"
#include "util/thread_pool.h"

namespace vdb {

struct FloatSpace {
  const float* data = nullptr;
  size_t stride = 0;  // padded dim
  DistFn fn = nullptr;

  float dist(const float* q, uint32_t id) const { return fn(q, row(id), stride); }
  const float* node_vec(uint32_t id, float* /*scratch*/) const { return row(id); }
  void prefetch(uint32_t id) const { Prefetch(row(id)); }
  size_t padded_dim() const { return stride; }
  const float* row(uint32_t id) const { return data + static_cast<size_t>(id) * stride; }
};

template <class Space>
class HnswGraph {
 public:
  static constexpr int kMaxLevel = 15;

  HnswGraph(const Space& space, size_t capacity, size_t M, size_t ef_construction, uint64_t seed)
      : space_(space),
        capacity_(capacity),
        M_(std::max<size_t>(2, M)),
        M0_(2 * M_),
        efc_(std::max(ef_construction, M_)),
        seed_(seed),
        mult_(1.0 / std::log(static_cast<double>(M_))),
        level_(capacity, 0),
        links0_(capacity * (1 + M0_), 0),
        upper_(capacity),
        stripes_(LockStripes(capacity)),
        locks_(std::make_unique<std::mutex[]>(stripes_)),
        ctx_pool_(capacity) {}

  // Copy of `o` with room for new_capacity nodes, over a new space whose first
  // o.capacity() nodes are o's nodes.
  HnswGraph(const HnswGraph& o, const Space& space, size_t new_capacity)
      : HnswGraph(space, new_capacity, o.M_, o.efc_, o.seed_) {
    std::copy(o.level_.begin(), o.level_.end(), level_.begin());
    std::copy(o.links0_.begin(), o.links0_.end(), links0_.begin());
    for (size_t i = 0; i < o.capacity_; ++i) {
      if (o.level_[i] == 0) continue;
      size_t n = static_cast<size_t>(o.level_[i]) * (1 + M_);
      upper_[i] = std::make_unique<uint32_t[]>(n);
      std::copy(o.upper_[i].get(), o.upper_[i].get() + n, upper_[i].get());
    }
    entry_point_ = o.entry_point_;
    max_level_ = o.max_level_;
    size_.store(o.size());
  }

  size_t size() const { return size_.load(std::memory_order_relaxed); }
  size_t capacity() const { return capacity_; }
  size_t M() const { return M_; }
  size_t M0() const { return M0_; }
  size_t ef_construction() const { return efc_; }
  int max_level() const { return max_level_; }
  uint32_t entry_point() const { return entry_point_; }
  int level(uint32_t id) const { return level_[id]; }
  const Space& space() const { return space_; }

  // Inserts nodes [begin, end). Parallel when a pool is given; the result then depends
  // on thread scheduling, so tests that compare exact results use one thread.
  void build(size_t begin, size_t end, ThreadPool* pool) {
    if (begin >= end) return;
    if (pool == nullptr || pool->parallelism() == 1) {
      for (size_t i = begin; i < end; ++i) insert(static_cast<uint32_t>(i));
    } else {
      pool->parallel_for(begin, end, [this](size_t i) { insert(static_cast<uint32_t>(i)); }, 32);
    }
    repaired_ += repair_connectivity();
  }

  // Pruning a full neighbor list can remove a node's last incoming link, leaving it
  // unreachable from the entry point on layer 0 (hnswlib graphs have the same property).
  // This pass links every such node from its nearest reachable node that has a free slot
  // (or, if none has, replaces that node's farthest link and checks again). Returns the
  // number of links added.
  size_t repair_connectivity() {
    const size_t n = size();
    if (n <= 1) return 0;
    size_t added = 0;
    std::vector<uint8_t> reach;
    std::vector<uint32_t> stack;
    auto bfs = [&](uint32_t start) {
      reach[start] = 1;
      stack.assign(1, start);
      while (!stack.empty()) {
        uint32_t c = stack.back();
        stack.pop_back();
        const uint32_t* ll = links_at(c, 0);
        for (uint32_t j = 0; j < ll[0]; ++j) {
          if (!reach[ll[1 + j]]) {
            reach[ll[1 + j]] = 1;
            stack.push_back(ll[1 + j]);
          }
        }
      }
    };
    std::vector<Neighbor> out;
    std::vector<float> s1(space_.padded_dim()), s2(space_.padded_dim());
    for (int pass = 0; pass < 4; ++pass) {
      reach.assign(n, 0);
      bfs(entry_point_);
      bool replaced_any = false;
      for (uint32_t u = 0; u < n; ++u) {
        if (reach[u]) continue;
        search(space_.node_vec(u, s1.data()), efc_, efc_, nullptr, out, nullptr);
        bool linked = false;
        for (const Neighbor& r : out) {  // a search only ever reaches reachable nodes
          if (r.id == u) continue;
          uint32_t* ll = links_at(r.id, 0);
          if (ll[0] < M0_) {
            ll[1 + ll[0]] = u;
            ++ll[0];
            linked = true;
            break;
          }
        }
        if (!linked && !out.empty()) {
          uint32_t r = out.front().id;
          uint32_t* ll = links_at(r, 0);
          const float* rv = space_.node_vec(r, s2.data());
          uint32_t far = 0;
          float far_d = -1;
          for (uint32_t j = 0; j < ll[0]; ++j) {
            float d = space_.dist(rv, ll[1 + j]);
            if (d > far_d) {
              far_d = d;
              far = j;
            }
          }
          ll[1 + far] = u;
          linked = true;
          replaced_any = true;
        }
        if (linked) {
          ++added;
          bfs(u);
        }
      }
      if (!replaced_any) break;  // only additions: nothing can have become unreachable
    }
    return added;
  }

  size_t repaired() const { return repaired_; }

  // Graph reordering for cache locality: breadth-first order from the entry point on
  // layer 0 (unreached nodes appended), so graph neighbors sit close together in memory.
  // Returns new id -> old id.
  std::vector<uint32_t> bfs_order() const {
    const size_t n = size();
    std::vector<uint32_t> order;
    order.reserve(n);
    std::vector<uint8_t> seen(n, 0);
    auto visit = [&](uint32_t start) {
      size_t head = order.size();
      seen[start] = 1;
      order.push_back(start);
      while (head < order.size()) {
        const uint32_t* ll = links_at(order[head++], 0);
        for (uint32_t j = 0; j < ll[0]; ++j) {
          if (!seen[ll[1 + j]]) {
            seen[ll[1 + j]] = 1;
            order.push_back(ll[1 + j]);
          }
        }
      }
    };
    if (n > 0) visit(entry_point_);
    for (uint32_t i = 0; i < n; ++i)
      if (!seen[i]) visit(i);
    return order;
  }

  // A copy with every node renumbered: new id i is old id order[i]. `space` must hold the
  // rows in the new order.
  std::unique_ptr<HnswGraph> renumbered(const std::vector<uint32_t>& order, const Space& space) const {
    const size_t n = size();
    std::vector<uint32_t> new_of(n);
    for (uint32_t i = 0; i < n; ++i) new_of[order[i]] = i;
    auto g = std::make_unique<HnswGraph>(space, n, M_, efc_, seed_);
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t old = order[i];
      g->level_[i] = level_[old];
      if (level_[old] > 0) g->upper_[i] = std::make_unique<uint32_t[]>(static_cast<size_t>(level_[old]) * (1 + M_));
      for (int l = 0; l <= level_[old]; ++l) {
        const uint32_t* src = links_at(old, l);
        uint32_t* dst = g->links_at(i, l);
        dst[0] = src[0];
        for (uint32_t j = 0; j < src[0]; ++j) dst[1 + j] = new_of[src[1 + j]];
      }
    }
    g->entry_point_ = new_of[entry_point_];
    g->max_level_ = max_level_;
    g->size_.store(n);
    return g;
  }

  void insert(uint32_t id) {
    const int lvl = level_for(id);
    level_[id] = static_cast<uint8_t>(lvl);
    if (lvl > 0) upper_[id] = std::make_unique<uint32_t[]>(static_cast<size_t>(lvl) * (1 + M_));

    auto ctx = ctx_pool_.get();
    const size_t pd = space_.padded_dim();
    ctx->scratch.resize(3 * pd);
    float* qbuf = ctx->scratch.data();
    float* s1 = qbuf + pd;
    float* s2 = s1 + pd;
    const float* q = space_.node_vec(id, qbuf);

    // A node that lands above the current top layer holds the entry lock for its whole
    // insert, so no other insert can see a half-linked entry point (rare: ~1 in M^L).
    std::unique_lock<std::mutex> top(entry_mu_);
    if (max_level_ < 0) {
      entry_point_ = id;
      max_level_ = lvl;
      size_.fetch_add(1);
      return;
    }
    const int maxl = max_level_;
    uint32_t cur = entry_point_;
    if (lvl <= maxl) top.unlock();

    uint64_t dc = 0;
    float curd = space_.dist(q, cur);
    for (int l = maxl; l > lvl; --l) greedy<true>(q, cur, curd, l, *ctx, dc);

    std::vector<Neighbor> cands, kept, tmp, kept2;
    for (int l = std::min(lvl, maxl); l >= 0; --l) {
      ctx->reset();
      search_layer<true, false>(q, cur, curd, efc_, l, nullptr, *ctx, dc);
      cands.assign(ctx->results.begin(), ctx->results.end());
      std::sort(cands.begin(), cands.end());
      select_neighbors(cands, M_, kept, s1);
      {
        std::lock_guard<std::mutex> g(lock_for(id));
        uint32_t* ll = links_at(id, l);
        ll[0] = static_cast<uint32_t>(kept.size());
        for (size_t j = 0; j < kept.size(); ++j) ll[1 + j] = kept[j].id;
      }
      for (const Neighbor& nb : kept) add_link(nb.id, id, nb.dist, l, tmp, kept2, s1, s2);
      cur = cands.front().id;
      curd = cands.front().dist;
    }
    if (lvl > maxl) {  // still holding `top`
      entry_point_ = id;
      max_level_ = lvl;
    }
    size_.fetch_add(1);
  }

  // k nearest allowed nodes, closest first. Not safe concurrently with insert().
  void search(const float* q, size_t k, size_t ef, const RowFilter* filter,
              std::vector<Neighbor>& out, uint64_t* dist_count) const {
    out.clear();
    if (max_level_ < 0 || k == 0) return;
    auto ctx = ctx_pool_.get();
    uint64_t dc = 1;
    uint32_t cur = entry_point_;
    float curd = space_.dist(q, cur);
    for (int l = max_level_; l > 0; --l) greedy<false>(q, cur, curd, l, *ctx, dc);
    ef = std::max(ef, k);
    if (filter != nullptr) {
      search_layer<false, true>(q, cur, curd, ef, 0, filter, *ctx, dc);
    } else {
      search_layer<false, false>(q, cur, curd, ef, 0, nullptr, *ctx, dc);
    }
    auto& res = ctx->results;
    while (res.size() > k) {
      std::pop_heap(res.begin(), res.end());
      res.pop_back();
    }
    out.assign(res.begin(), res.end());
    std::sort(out.begin(), out.end());
    if (dist_count != nullptr) *dist_count += dc;
  }

  // Neighbor list of a node on a layer (read-only; not during a parallel build).
  std::vector<uint32_t> neighbors(uint32_t id, int layer) const {
    const uint32_t* ll = links_at(id, layer);
    return std::vector<uint32_t>(ll + 1, ll + 1 + ll[0]);
  }

  // Graph checks: no self-links, no duplicates, no node over its link limit, links only
  // to nodes present on that layer, and every node reachable from the entry point on
  // layer 0. Assumes nodes [0, size()) are inserted.
  Status check() const {
    const size_t n = size();
    if (n == 0) return Status::OK();
    std::vector<uint32_t> tmp;
    for (size_t i = 0; i < n; ++i) {
      for (int l = 0; l <= level_[i]; ++l) {
        const uint32_t* ll = links_at(static_cast<uint32_t>(i), l);
        size_t maxc = l == 0 ? M0_ : M_;
        if (ll[0] > maxc) return Status::Corruption("node over link limit: " + std::to_string(i));
        tmp.assign(ll + 1, ll + 1 + ll[0]);
        for (uint32_t e : tmp) {
          if (e == i) return Status::Corruption("self-link at node " + std::to_string(i));
          if (e >= n) return Status::Corruption("link out of range at node " + std::to_string(i));
          if (level_[e] < l) return Status::Corruption("link to a node below the layer");
        }
        std::sort(tmp.begin(), tmp.end());
        if (std::adjacent_find(tmp.begin(), tmp.end()) != tmp.end())
          return Status::Corruption("duplicate link at node " + std::to_string(i));
      }
    }
    std::vector<uint8_t> seen(n, 0);
    std::vector<uint32_t> stack{entry_point_};
    seen[entry_point_] = 1;
    size_t reached = 1;
    while (!stack.empty()) {
      uint32_t c = stack.back();
      stack.pop_back();
      const uint32_t* ll = links_at(c, 0);
      for (uint32_t j = 0; j < ll[0]; ++j) {
        uint32_t e = ll[1 + j];
        if (!seen[e]) {
          seen[e] = 1;
          ++reached;
          stack.push_back(e);
        }
      }
    }
    if (reached != n) {
      return Status::Corruption(std::to_string(n - reached) +
                                " nodes unreachable from the entry point on layer 0");
    }
    return Status::OK();
  }

  std::vector<size_t> level_histogram() const {
    std::vector<size_t> h(static_cast<size_t>(std::max(0, max_level_)) + 1, 0);
    for (size_t i = 0; i < size(); ++i) ++h[level_[i]];
    return h;
  }

  double average_degree(int layer) const {
    size_t nodes = 0, links = 0;
    for (size_t i = 0; i < size(); ++i) {
      if (level_[i] < layer) continue;
      ++nodes;
      links += links_at(static_cast<uint32_t>(i), layer)[0];
    }
    return nodes == 0 ? 0.0 : static_cast<double>(links) / static_cast<double>(nodes);
  }

  size_t memory_bytes() const {
    size_t b = links0_.size() * sizeof(uint32_t) + level_.size() + upper_.size() * sizeof(void*);
    for (size_t i = 0; i < capacity_; ++i)
      if (level_[i] > 0) b += static_cast<size_t>(level_[i]) * (1 + M_) * sizeof(uint32_t);
    return b + stripes_ * sizeof(std::mutex) + ctx_pool_.memory_bytes();
  }

  void serialize(std::string* out) const {
    PutU64(out, capacity_);
    PutU64(out, size());
    PutU32(out, static_cast<uint32_t>(M_));
    PutU32(out, static_cast<uint32_t>(efc_));
    PutU64(out, seed_);
    PutU32(out, static_cast<uint32_t>(max_level_));
    PutU32(out, entry_point_);
    PutBytes(out, level_.data(), level_.size());
    PutBytes(out, links0_.data(), links0_.size() * sizeof(uint32_t));
    for (size_t i = 0; i < capacity_; ++i) {
      if (level_[i] == 0) continue;
      PutBytes(out, upper_[i].get(), static_cast<size_t>(level_[i]) * (1 + M_) * sizeof(uint32_t));
    }
  }

  static StatusOr<std::unique_ptr<HnswGraph>> Deserialize(ByteReader& r, const Space& space,
                                                          size_t expected_capacity) {
    uint64_t cap = 0, sz = 0, seed = 0;
    uint32_t m = 0, efc = 0, maxl = 0, ep = 0;
    if (!r.GetU64(&cap) || !r.GetU64(&sz) || !r.GetU32(&m) || !r.GetU32(&efc) ||
        !r.GetU64(&seed) || !r.GetU32(&maxl) || !r.GetU32(&ep)) {
      return Status::Corruption("truncated hnsw header");
    }
    if (cap != expected_capacity || sz > cap || m < 2 || m > 4096) {
      return Status::Corruption("hnsw header does not match the segment");
    }
    auto g = std::make_unique<HnswGraph>(space, cap, m, efc, seed);
    g->max_level_ = static_cast<int>(maxl);
    g->entry_point_ = ep;
    g->size_.store(sz);
    if (g->max_level_ > kMaxLevel || (cap > 0 && ep >= cap)) {
      return Status::Corruption("bad hnsw entry point");
    }
    std::string_view bytes;
    if (!r.GetBytes(cap, &bytes)) return Status::Corruption("truncated hnsw levels");
    std::memcpy(g->level_.data(), bytes.data(), cap);
    if (!r.GetBytes(g->links0_.size() * sizeof(uint32_t), &bytes)) {
      return Status::Corruption("truncated hnsw layer 0");
    }
    std::memcpy(g->links0_.data(), bytes.data(), bytes.size());
    for (size_t i = 0; i < cap; ++i) {
      int lvl = g->level_[i];
      if (sz > 0 && lvl > g->max_level_) return Status::Corruption("node above the top layer");
      if (lvl == 0) continue;
      size_t n = static_cast<size_t>(lvl) * (1 + g->M_);
      if (!r.GetBytes(n * sizeof(uint32_t), &bytes)) {
        return Status::Corruption("truncated hnsw upper layers");
      }
      g->upper_[i] = std::make_unique<uint32_t[]>(n);
      std::memcpy(g->upper_[i].get(), bytes.data(), bytes.size());
    }
    // Validate every link so a bad file can never send a search out of bounds.
    for (size_t i = 0; i < cap; ++i) {
      for (int l = 0; l <= g->level_[i]; ++l) {
        const uint32_t* ll = g->links_at(static_cast<uint32_t>(i), l);
        if (ll[0] > (l == 0 ? g->M0_ : g->M_)) return Status::Corruption("bad link count");
        for (uint32_t j = 0; j < ll[0]; ++j)
          if (ll[1 + j] >= cap) return Status::Corruption("bad link target");
      }
    }
    return g;
  }

 private:
  static size_t LockStripes(size_t capacity) {
    size_t s = 1;
    while (s < capacity && s < (1u << 16)) s <<= 1;
    return s;
  }

  std::mutex& lock_for(uint32_t id) const { return locks_[id & (stripes_ - 1)]; }

  uint32_t* links_at(uint32_t id, int layer) {
    if (layer == 0) return links0_.data() + static_cast<size_t>(id) * (1 + M0_);
    return upper_[id].get() + static_cast<size_t>(layer - 1) * (1 + M_);
  }
  const uint32_t* links_at(uint32_t id, int layer) const {
    if (layer == 0) return links0_.data() + static_cast<size_t>(id) * (1 + M0_);
    return upper_[id].get() + static_cast<size_t>(layer - 1) * (1 + M_);
  }

  // Levels come from a hash of (seed, id), so they do not depend on insertion order.
  int level_for(uint32_t id) const {
    double u = ToUnitDouble(SplitMix64(seed_ ^ (static_cast<uint64_t>(id) * 0x9E3779B97F4A7C15ull)));
    int l = static_cast<int>(-std::log(1.0 - u) * mult_);  // 1 - u avoids log(0)
    return std::min(l, kMaxLevel);
  }

  void copy_links(uint32_t id, int layer, std::vector<uint32_t>& out) const {
    std::lock_guard<std::mutex> g(lock_for(id));
    const uint32_t* ll = links_at(id, layer);
    out.assign(ll + 1, ll + 1 + ll[0]);
  }

  template <bool kLocked>
  void greedy(const float* q, uint32_t& cur, float& curd, int layer, SearchContext& ctx,
              uint64_t& dc) const {
    for (bool changed = true; changed;) {
      changed = false;
      const uint32_t* nb;
      uint32_t cnt;
      if constexpr (kLocked) {
        copy_links(cur, layer, ctx.links);
        nb = ctx.links.data();
        cnt = static_cast<uint32_t>(ctx.links.size());
      } else {
        const uint32_t* ll = links_at(cur, layer);
        cnt = ll[0];
        nb = ll + 1;
      }
      for (uint32_t j = 0; j < cnt; ++j) {
        float d = space_.dist(q, nb[j]);
        ++dc;
        if (d < curd) {
          curd = d;
          cur = nb[j];
          changed = true;
        }
      }
    }
  }

  // Beam search on one layer. Leaves up to ef results in ctx.results (a max-heap). With a
  // filter, rejected nodes are still expanded but never enter the results, and the search
  // keeps going until it holds ef allowed results (or runs out of candidates).
  template <bool kLocked, bool kFiltered>
  void search_layer(const float* q, uint32_t ep, float ep_dist, size_t ef, int layer,
                    const RowFilter* filter, SearchContext& ctx, uint64_t& dc) const {
    auto& results = ctx.results;
    auto& cand = ctx.candidates;
    results.clear();
    cand.clear();
    ctx.mark(ep);
    float lower = std::numeric_limits<float>::infinity();
    if (!kFiltered || filter->allowed(ep)) {
      results.push_back({ep_dist, ep});
      lower = ep_dist;
    }
    cand.push_back({ep_dist, ep});
    while (!cand.empty()) {
      const Neighbor c = cand.front();
      if (c.dist > lower && (!kFiltered || results.size() >= ef)) break;
      std::pop_heap(cand.begin(), cand.end(), std::greater<Neighbor>());
      cand.pop_back();

      const uint32_t* nb;
      uint32_t cnt;
      if constexpr (kLocked) {
        copy_links(c.id, layer, ctx.links);
        nb = ctx.links.data();
        cnt = static_cast<uint32_t>(ctx.links.size());
      } else {
        const uint32_t* ll = links_at(c.id, layer);
        cnt = ll[0];
        nb = ll + 1;
      }
      if (cnt > 0) space_.prefetch(nb[0]);
      for (uint32_t j = 0; j < cnt; ++j) {
        const uint32_t e = nb[j];
        if (j + 1 < cnt) space_.prefetch(nb[j + 1]);
        if (ctx.seen(e)) continue;
        ctx.mark(e);
        const float de = space_.dist(q, e);
        ++dc;
        if (results.size() < ef || de < lower) {
          cand.push_back({de, e});
          std::push_heap(cand.begin(), cand.end(), std::greater<Neighbor>());
          if (!kFiltered || filter->allowed(e)) {
            results.push_back({de, e});
            std::push_heap(results.begin(), results.end());
            if (results.size() > ef) {
              std::pop_heap(results.begin(), results.end());
              results.pop_back();
            }
            lower = results.front().dist;
          }
        }
      }
    }
  }

  // The neighbor heuristic: keep a candidate only if it is closer to the base node than
  // to every neighbor already kept, so links point in different directions and long
  // links between clusters survive. `cand` must be sorted closest first.
  void select_neighbors(const std::vector<Neighbor>& cand, size_t max_m,
                        std::vector<Neighbor>& kept, float* scratch) const {
    kept.clear();
    if (cand.size() < max_m) {
      kept = cand;
      return;
    }
    for (const Neighbor& c : cand) {
      if (kept.size() >= max_m) break;
      const float* cv = space_.node_vec(c.id, scratch);
      bool good = true;
      for (const Neighbor& k : kept) {
        if (space_.dist(cv, k.id) < c.dist) {
          good = false;
          break;
        }
      }
      if (good) kept.push_back(c);
    }
  }

  // Adds the back-link n -> id. A full list is re-pruned with the heuristic over its
  // links plus the new node, under n's lock.
  void add_link(uint32_t n, uint32_t id, float d_n_id, int layer, std::vector<Neighbor>& tmp,
                std::vector<Neighbor>& kept, float* s1, float* s2) {
    const size_t maxc = layer == 0 ? M0_ : M_;
    std::lock_guard<std::mutex> g(lock_for(n));
    uint32_t* ll = links_at(n, layer);
    const uint32_t cnt = ll[0];
    for (uint32_t j = 0; j < cnt; ++j)
      if (ll[1 + j] == id) return;
    if (cnt < maxc) {
      ll[1 + cnt] = id;
      ll[0] = cnt + 1;
      return;
    }
    const float* nv = space_.node_vec(n, s1);
    tmp.clear();
    tmp.push_back({d_n_id, id});
    for (uint32_t j = 0; j < cnt; ++j) tmp.push_back({space_.dist(nv, ll[1 + j]), ll[1 + j]});
    std::sort(tmp.begin(), tmp.end());
    select_neighbors(tmp, maxc, kept, s2);
    ll[0] = static_cast<uint32_t>(kept.size());
    for (size_t j = 0; j < kept.size(); ++j) ll[1 + j] = kept[j].id;
  }

  Space space_;
  size_t capacity_;
  size_t M_;
  size_t M0_;
  size_t efc_;
  uint64_t seed_;
  double mult_;  // 1 / ln(M)
  std::vector<uint8_t> level_;
  std::vector<uint32_t> links0_;  // capacity * (1 + M0)
  std::vector<std::unique_ptr<uint32_t[]>> upper_;
  size_t stripes_;
  // Striped node locks (at most 65536 mutexes), held one at a time during a build.
  std::unique_ptr<std::mutex[]> locks_;
  std::mutex entry_mu_;
  uint32_t entry_point_ = 0;
  int max_level_ = -1;
  std::atomic<size_t> size_{0};
  size_t repaired_ = 0;
  mutable SearchContextPool ctx_pool_;
};

}  // namespace vdb
