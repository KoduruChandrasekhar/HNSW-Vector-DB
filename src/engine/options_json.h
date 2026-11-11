#pragma once

#include <nlohmann/json_fwd.hpp>

#include "vdb/options.h"
#include "vdb/status.h"

namespace vdb {

// The persisted collection config (manifest) and the server's create-collection body
// share one JSON shape. Missing keys keep their defaults.
nlohmann::json OptionsToJson(const CollectionOptions& o);
Status OptionsFromJson(const nlohmann::json& j, CollectionOptions* o);
Status IndexOptionsFromJson(const nlohmann::json& j, IndexOptions* o);
// Checks dim, metric/index compatibility, schema and compaction knobs.
Status ValidateOptions(const CollectionOptions& o);

}  // namespace vdb
