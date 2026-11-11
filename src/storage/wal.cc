#include "storage/wal.h"

#include <cinttypes>
#include <cstdio>

#include "util/coding.h"
#include "util/crc32c.h"
#include "util/metrics.h"
#include "util/timer.h"

namespace vdb {
namespace {

bool TryParseAt(std::string_view data, size_t pos, WalRecord* rec, size_t* len) {
  if (data.size() - pos < kWalRecordHeaderSize) return false;
  const char* p = data.data() + pos;
  uint32_t plen = DecodeU32(p + 4);
  if (plen > data.size() - pos - kWalRecordHeaderSize) return false;
  auto type = static_cast<uint8_t>(p[8]);
  if (type != static_cast<uint8_t>(WalRecordType::kUpsert) &&
      type != static_cast<uint8_t>(WalRecordType::kDelete)) {
    return false;
  }
  if (crc32c::Value(p + 4, kWalRecordHeaderSize - 4 + plen) != DecodeU32(p)) return false;
  rec->type = static_cast<WalRecordType>(type);
  rec->lsn = DecodeU64(p + 9);
  rec->payload = std::string_view(p + kWalRecordHeaderSize, plen);
  *len = kWalRecordHeaderSize + plen;
  return true;
}

}  // namespace

std::string WalFileHeader() {
  std::string h(kWalMagic);
  PutU32(&h, 1);
  PutU32(&h, 0);
  return h;
}

void AppendWalRecord(std::string* dst, WalRecordType type, uint64_t lsn, std::string_view payload) {
  size_t start = dst->size();
  PutU32(dst, 0);  // CRC placeholder
  PutU32(dst, static_cast<uint32_t>(payload.size()));
  PutU8(dst, static_cast<uint8_t>(type));
  PutU64(dst, lsn);
  dst->append(payload);
  uint32_t crc = crc32c::Value(dst->data() + start + 4, dst->size() - start - 4);
  EncodeU32(dst->data() + start, crc);
}

void EncodeUpsertPayload(std::string* dst, VectorId id, const float* vec, size_t dim,
                         std::string_view meta) {
  PutU64(dst, id);
  PutU32(dst, static_cast<uint32_t>(dim));
  PutBytes(dst, vec, dim * sizeof(float));
  PutU32(dst, static_cast<uint32_t>(meta.size()));
  dst->append(meta);
}

bool DecodeUpsertPayload(std::string_view p, VectorId* id, uint32_t* dim, const char** vec,
                         std::string_view* meta) {
  ByteReader r(p);
  uint32_t mlen = 0;
  std::string_view v;
  if (!r.GetU64(id) || !r.GetU32(dim) || *dim == 0) return false;
  if (!r.GetBytes(static_cast<size_t>(*dim) * sizeof(float), &v)) return false;
  if (!r.GetU32(&mlen) || !r.GetBytes(mlen, meta) || r.remaining() != 0) return false;
  *vec = v.data();
  return true;
}

void EncodeDeletePayload(std::string* dst, VectorId id) { PutU64(dst, id); }

bool DecodeDeletePayload(std::string_view p, VectorId* id) {
  ByteReader r(p);
  return r.GetU64(id) && r.remaining() == 0;
}

Status ParseWal(std::string_view data, const std::function<Status(const WalRecord&)>& cb,
                WalParseResult* result) {
  *result = WalParseResult{};
  if (data.size() < kWalFileHeaderSize) {
    // Crashed while creating the file: nothing in it was ever acknowledged.
    result->torn_tail = !data.empty();
    return Status::OK();
  }
  if (data.substr(0, kWalMagic.size()) != kWalMagic || DecodeU32(data.data() + 8) != 1) {
    return Status::Corruption("bad log file header");
  }
  size_t pos = kWalFileHeaderSize;
  while (pos < data.size()) {
    WalRecord rec{};
    size_t len = 0;
    if (TryParseAt(data, pos, &rec, &len)) {
      VDB_RETURN_IF_ERROR(cb(rec));
      pos += len;
      ++result->records;
      continue;
    }
    // A bad record. If any good record follows, this is corruption, not a torn tail.
    for (size_t off = pos + 1; off + kWalRecordHeaderSize <= data.size(); ++off) {
      WalRecord later{};
      size_t later_len = 0;
      if (TryParseAt(data, off, &later, &later_len)) {
        return Status::Corruption("corrupt log record at offset " + std::to_string(pos) +
                                  " followed by valid records");
      }
    }
    result->torn_tail = true;
    break;
  }
  result->valid_bytes = pos;
  return Status::OK();
}

std::string WalFileName(uint64_t number) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%06" PRIu64 ".log", number);
  return buf;
}

bool ParseWalFileName(std::string_view name, uint64_t* number) {
  if (name.size() < 5 || name.substr(name.size() - 4) != ".log") return false;
  uint64_t n = 0;
  for (char c : name.substr(0, name.size() - 4)) {
    if (c < '0' || c > '9') return false;
    n = n * 10 + static_cast<uint64_t>(c - '0');
  }
  *number = n;
  return true;
}

StatusOr<std::unique_ptr<WalWriter>> WalWriter::Create(Env* env, const std::string& path,
                                                       uint64_t number) {
  VDB_ASSIGN_OR_RETURN(auto f, env->new_writable(path));
  VDB_RETURN_IF_ERROR(f->append(WalFileHeader()));
  VDB_RETURN_IF_ERROR(f->sync());
  return std::unique_ptr<WalWriter>(new WalWriter(std::move(f), path, number));
}

Status WalWriter::append(std::string_view records) {
  std::lock_guard<std::mutex> g(mu_);
  if (failed_) return Status::IOError("log " + path_ + " failed earlier; writes are refused");
  Status s = file_->append(records);
  if (!s.ok()) {
    failed_ = true;
    return s;
  }
  unsynced_ += records.size();
  GlobalMetrics().wal_bytes_total.inc(records.size());
  return Status::OK();
}

Status WalWriter::sync() {
  std::lock_guard<std::mutex> g(mu_);
  if (failed_) return Status::IOError("log " + path_ + " failed earlier; writes are refused");
  if (unsynced_ == 0) return Status::OK();
  Timer t;
  Status s = file_->sync();
  GlobalMetrics().wal_fsync_seconds.observe(t.seconds());
  GlobalMetrics().wal_syncs_total.inc();
  if (!s.ok()) {
    failed_ = true;
    return s;
  }
  unsynced_ = 0;
  return Status::OK();
}

Status WalWriter::close() {
  std::lock_guard<std::mutex> g(mu_);
  return file_->close();
}

uint64_t WalWriter::bytes() const {
  std::lock_guard<std::mutex> g(mu_);
  return file_->size();
}

bool WalWriter::failed() const {
  std::lock_guard<std::mutex> g(mu_);
  return failed_;
}

}  // namespace vdb
