#pragma once

#include <chrono>
#include <cstdint>

namespace vdb {

class Timer {
 public:
  Timer() : start_(std::chrono::steady_clock::now()) {}
  void reset() { start_ = std::chrono::steady_clock::now(); }
  double seconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  }
  double micros() const { return seconds() * 1e6; }
  double millis() const { return seconds() * 1e3; }

 private:
  std::chrono::steady_clock::time_point start_;
};

inline uint64_t NowMicros() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count());
}

}  // namespace vdb
