#include "util/thread_pool.h"

#include <algorithm>
#include <atomic>
#include <memory>

namespace vdb {

ThreadPool::ThreadPool(size_t parallelism) {
  if (parallelism == 0) parallelism = std::max(1u, std::thread::hardware_concurrency());
  workers_.reserve(parallelism - 1);
  for (size_t i = 0; i + 1 < parallelism; ++i) workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> g(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : workers_) t.join();
}

void ThreadPool::submit(std::function<void()> task) {
  if (workers_.empty()) {
    task();
    return;
  }
  {
    std::lock_guard<std::mutex> g(mu_);
    tasks_.push_back(std::move(task));
  }
  cv_.notify_one();
}

void ThreadPool::worker_loop() {
  for (;;) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> l(mu_);
      cv_.wait(l, [this] { return stop_ || !tasks_.empty(); });
      if (tasks_.empty()) return;  // stop_ and drained
      task = std::move(tasks_.front());
      tasks_.pop_front();
    }
    task();
  }
}

void ThreadPool::parallel_for_range(size_t begin, size_t end,
                                    const std::function<void(size_t, size_t)>& fn,
                                    size_t chunk) {
  if (begin >= end) return;
  if (chunk == 0) chunk = 1;
  const size_t nchunks = (end - begin + chunk - 1) / chunk;
  if (workers_.empty() || nchunks == 1) {
    for (size_t b = begin; b < end; b += chunk) fn(b, std::min(end, b + chunk));
    return;
  }

  // Shared state outlives the call: helpers that start late find no work and exit
  // without touching fn.
  struct State {
    std::atomic<size_t> next{0};
    std::atomic<size_t> done{0};
    size_t nchunks = 0, begin = 0, end = 0, chunk = 0;
    const std::function<void(size_t, size_t)>* fn = nullptr;
    std::mutex mu;
    std::condition_variable cv;
  };
  auto st = std::make_shared<State>();
  st->nchunks = nchunks;
  st->begin = begin;
  st->end = end;
  st->chunk = chunk;
  st->fn = &fn;

  auto run = [st] {
    for (;;) {
      size_t c = st->next.fetch_add(1);
      if (c >= st->nchunks) return;
      size_t b = st->begin + c * st->chunk;
      (*st->fn)(b, std::min(st->end, b + st->chunk));
      if (st->done.fetch_add(1) + 1 == st->nchunks) {
        std::lock_guard<std::mutex> g(st->mu);
        st->cv.notify_all();
      }
    }
  };

  size_t helpers = std::min(workers_.size(), nchunks - 1);
  for (size_t i = 0; i < helpers; ++i) submit(run);
  run();
  std::unique_lock<std::mutex> l(st->mu);
  st->cv.wait(l, [&] { return st->done.load() == st->nchunks; });
}

void ThreadPool::parallel_for(size_t begin, size_t end, const std::function<void(size_t)>& fn,
                              size_t chunk) {
  parallel_for_range(
      begin, end,
      [&fn](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) fn(i);
      },
      chunk);
}

}  // namespace vdb
