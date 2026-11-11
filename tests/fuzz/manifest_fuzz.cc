// The manifest decoder (blob framing, CRC, JSON, option validation) must reject garbage
// cleanly. The fuzzer input is wrapped in a valid blob so the JSON layer is reached.
#include <cstddef>
#include <cstdint>
#include <string>

#include "meta/schema.h"
#include "storage/file_format.h"
#include "storage/manifest.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  std::string_view in(reinterpret_cast<const char*>(data), size);
  vdb::Manifest m;
  vdb::DecodeManifest(in, &m);  // raw bytes: framing and CRC checks
  std::string blob = vdb::EncodeBlob(vdb::kManifestMagic, in);
  if (vdb::DecodeManifest(blob, &m).ok()) {
    // Anything accepted must round-trip.
    vdb::Manifest back;
    if (!vdb::DecodeManifest(vdb::EncodeManifest(m), &back).ok()) __builtin_trap();
    if (back.segments.size() != m.segments.size()) __builtin_trap();
  }
  return 0;
}
