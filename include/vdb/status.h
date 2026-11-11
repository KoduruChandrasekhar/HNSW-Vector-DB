#pragma once

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace vdb {

class Status {
 public:
  enum class Code : uint8_t {
    kOk = 0,
    kInvalidArgument,
    kNotFound,
    kAlreadyExists,
    kIOError,
    kCorruption,
    kFailedPrecondition,
    kResourceExhausted,
    kUnavailable,
    kNotSupported,
    kInternal,
  };

  Status() = default;
  Status(Code code, std::string msg) : code_(code), msg_(std::move(msg)) {}

  static Status OK() { return {}; }
  static Status InvalidArgument(std::string m) { return {Code::kInvalidArgument, std::move(m)}; }
  static Status NotFound(std::string m) { return {Code::kNotFound, std::move(m)}; }
  static Status AlreadyExists(std::string m) { return {Code::kAlreadyExists, std::move(m)}; }
  static Status IOError(std::string m) { return {Code::kIOError, std::move(m)}; }
  static Status Corruption(std::string m) { return {Code::kCorruption, std::move(m)}; }
  static Status FailedPrecondition(std::string m) {
    return {Code::kFailedPrecondition, std::move(m)};
  }
  static Status ResourceExhausted(std::string m) {
    return {Code::kResourceExhausted, std::move(m)};
  }
  static Status Unavailable(std::string m) { return {Code::kUnavailable, std::move(m)}; }
  static Status NotSupported(std::string m) { return {Code::kNotSupported, std::move(m)}; }
  static Status Internal(std::string m) { return {Code::kInternal, std::move(m)}; }

  bool ok() const { return code_ == Code::kOk; }
  Code code() const { return code_; }
  const std::string& message() const { return msg_; }
  std::string ToString() const;

  bool IsNotFound() const { return code_ == Code::kNotFound; }
  bool IsCorruption() const { return code_ == Code::kCorruption; }
  bool IsIOError() const { return code_ == Code::kIOError; }

 private:
  Code code_ = Code::kOk;
  std::string msg_;
};

const char* CodeName(Status::Code code);

// A value or the error that prevented computing it.
template <class T>
class StatusOr {
 public:
  StatusOr(Status s) : status_(std::move(s)) {  // NOLINT: implicit by design
    assert(!status_.ok() && "StatusOr built from an OK status");
    if (status_.ok()) status_ = Status::Internal("StatusOr built from an OK status");
  }
  template <class U, std::enable_if_t<std::is_convertible_v<U&&, T> &&
                                           !std::is_same_v<std::decay_t<U>, Status>,
                                       int> = 0>
  StatusOr(U&& v) : value_(std::forward<U>(v)) {}  // NOLINT: implicit by design

  bool ok() const { return status_.ok(); }
  const Status& status() const { return status_; }

  T& value() & { return *value_; }
  const T& value() const& { return *value_; }
  T&& value() && { return std::move(*value_); }
  T* operator->() { return &*value_; }
  const T* operator->() const { return &*value_; }
  T& operator*() & { return *value_; }
  const T& operator*() const& { return *value_; }

 private:
  Status status_;
  std::optional<T> value_;
};

}  // namespace vdb

#define VDB_CONCAT_IMPL(a, b) a##b
#define VDB_CONCAT(a, b) VDB_CONCAT_IMPL(a, b)

#define VDB_RETURN_IF_ERROR(expr)              \
  do {                                         \
    ::vdb::Status vdb_status_tmp_ = (expr);    \
    if (!vdb_status_tmp_.ok()) return vdb_status_tmp_; \
  } while (0)

#define VDB_ASSIGN_OR_RETURN_IMPL(tmp, lhs, expr) \
  auto tmp = (expr);                              \
  if (!tmp.ok()) return tmp.status();             \
  lhs = std::move(tmp).value()

#define VDB_ASSIGN_OR_RETURN(lhs, expr) \
  VDB_ASSIGN_OR_RETURN_IMPL(VDB_CONCAT(vdb_statusor_, __LINE__), lhs, expr)
