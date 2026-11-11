#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace vdb {

// A fixed-size pool. `parallelism` counts the calling thread, so ThreadPool(1) has no
// workers and runs everything inline, and ThreadPool(8) keeps 8 cores busy.
class ThreadPool {
 public:
  explicit ThreadPool(size_t parallelism = 0);  // 0 = hardware concurrency
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  size_t parallelism() const { return workers_.size() + 1; }

  // Runs a task on a worker (inline when there are no workers).
  void submit(std::function<void()> task);

  // Calls fn(chunk_begin, chunk_end) over [begin, end) in chunks of `chunk` items. The
  // caller works too and the call returns once every chunk is done. Safe to nest: a
  // caller never waits for a task that has not started.
  void parallel_for_range(size_t begin, size_t end,
                          const std::function<void(size_t, size_t)>& fn, size_t chunk = 1);
  // Calls fn(i) for every i in [begin, end).
  void parallel_for(size_t begin, size_t end, const std::function<void(size_t)>& fn,
                    size_t chunk = 1);

 private:
  void worker_loop();

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> tasks_;
  bool stop_ = false;
};

// Runs fn over [begin, end) on `pool`, or inline when pool is null.
inline void ParallelForRange(ThreadPool* pool, size_t begin, size_t end,
                             const std::function<void(size_t, size_t)>& fn, size_t chunk = 1) {
  if (pool != nullptr) {
    pool->parallel_for_range(begin, end, fn, chunk);
  } else if (begin < end) {
    fn(begin, end);
  }
}

}  // namespace vdb
