cmake_minimum_required(VERSION 3.24)
project(vdb VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Some dependencies still declare very old minimum CMake versions; CMake 4 refuses
# those unless a policy floor is given.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

option(VDB_BUILD_TESTS "Build unit tests" ON)
option(VDB_BUILD_BENCHMARKS "Build Google Benchmark microbenchmarks" ON)
option(VDB_BUILD_SERVER "Build vdb-server" ON)
option(VDB_BUILD_TOOLS "Build vdb_bench and other tools" ON)
option(VDB_BUILD_PYTHON "Build the nanobind Python module" OFF)
option(VDB_BUILD_FUZZERS "Build libFuzzer targets (Clang only)" OFF)
option(VDB_WERROR "Treat warnings as errors in vdb's own code" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(cmake/Dependencies.cmake)

# Warnings apply to vdb's own targets only, never to fetched dependencies.
add_library(vdb_warnings INTERFACE)
target_compile_options(vdb_warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow)
# GCC's maybe-uninitialized fires false positives inside libstdc++ (std::regex, pulled in
# by cpp-httplib) when sanitizers change inlining; keep it a warning there.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(vdb_warnings INTERFACE -Wno-error=maybe-uninitialized)
endif()
if(VDB_WERROR)
  target_compile_options(vdb_warnings INTERFACE -Werror)
endif()

# ---------------------------------------------------------------- library
set(VDB_SOURCES
  src/util/status.cc
  src/util/crc32c.cc
  src/util/thread_pool.cc
  src/util/metrics.cc
  src/util/dataset.cc
  src/distance/scalar.cc
  src/distance/cpu_features.cc
  src/index/index.cc
  src/index/flat_index.cc
  src/index/hnsw/hnsw_index.cc
  src/index/quant/kmeans.cc
  src/index/quant/sq8.cc
  src/index/quant/pq.cc
  src/index/quant/ivf_pq.cc
  src/storage/posix_env.cc
  src/storage/fault_injection_env.cc
  src/storage/file_format.cc
  src/storage/wal.cc
  src/storage/segment_writer.cc
  src/storage/segment_reader.cc
  src/storage/manifest.cc
  src/meta/schema.cc
  src/meta/columns.cc
  src/meta/filter_parser.cc
  src/engine/options_json.cc
  src/engine/mutable_segment.cc
  src/engine/id_map.cc
  src/engine/planner.cc
  src/engine/compaction.cc
  src/engine/collection_impl.cc
)

# SIMD kernels: only these files get ISA flags, and the dispatcher picks one at
# runtime, so the binary still runs on CPUs without AVX2.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  list(APPEND VDB_SOURCES src/distance/avx2.cc src/distance/avx512.cc)
  set_source_files_properties(src/distance/avx2.cc PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  set_source_files_properties(src/distance/avx512.cc PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx2;-mfma")
  set(VDB_HAVE_X86_KERNELS ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  list(APPEND VDB_SOURCES src/distance/neon.cc)
  set(VDB_HAVE_NEON_KERNELS ON)
endif()

add_library(vdb STATIC ${VDB_SOURCES})
add_library(vdb::vdb ALIAS vdb)
target_include_directories(vdb
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(vdb
  PUBLIC Threads::Threads
  PRIVATE vdb_warnings nlohmann_json::nlohmann_json vdb_roaring)
if(VDB_HAVE_X86_KERNELS)
  target_compile_definitions(vdb PRIVATE VDB_HAVE_X86_KERNELS=1)
endif()
if(VDB_HAVE_NEON_KERNELS)
  target_compile_definitions(vdb PRIVATE VDB_HAVE_NEON_KERNELS=1)
endif()

# Internal headers (src/) are needed by tests, tools and the server.
add_library(vdb_internal INTERFACE)
target_include_directories(vdb_internal INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(vdb_internal INTERFACE vdb nlohmann_json::nlohmann_json vdb_roaring)

# ---------------------------------------------------------------- server
if(VDB_BUILD_SERVER)
  add_library(vdb_server_lib STATIC
    src/server/http_api.cc
    src/server/metrics.cc
    src/server/binary_format.cc
    src/server/collection_manager.cc)
  target_link_libraries(vdb_server_lib
    PUBLIC vdb_internal httplib::httplib spdlog::spdlog
    PRIVATE vdb_warnings)
  add_executable(vdb-server src/server/main.cc)
  target_link_libraries(vdb-server PRIVATE vdb_server_lib CLI11::CLI11 vdb_warnings)
endif()

# ---------------------------------------------------------------- tools
if(VDB_BUILD_TOOLS)
  add_executable(vdb_bench tools/vdb_bench.cc)
  target_link_libraries(vdb_bench PRIVATE vdb_internal CLI11::CLI11 vdb_warnings)
endif()

if(VDB_BUILD_BENCHMARKS)
  add_executable(vdb_micro_bench bench/micro_bench.cc)
  target_link_libraries(vdb_micro_bench PRIVATE vdb_internal benchmark::benchmark vdb_warnings)
endif()

# ---------------------------------------------------------------- tests
if(VDB_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(VDB_BUILD_FUZZERS)
  add_subdirectory(tests/fuzz)
endif()

# ---------------------------------------------------------------- python
if(VDB_BUILD_PYTHON)
  find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_vdb NB_STATIC python/bindings.cc)
  target_link_libraries(_vdb PRIVATE vdb_internal)
  install(TARGETS _vdb LIBRARY DESTINATION vdb)
endif()
