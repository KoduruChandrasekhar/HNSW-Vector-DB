# Third-party dependencies, pulled with FetchContent at pinned release versions.
include(FetchContent)
set(FETCHCONTENT_QUIET ON)
find_package(Threads REQUIRED)

function(vdb_fetch name url)
  FetchContent_Declare(${name} URL ${url} DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
endfunction()

# nlohmann/json: manifest, API bodies, filter language.
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
vdb_fetch(nlohmann_json https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(nlohmann_json)

# CRoaring: filter results and persisted delete sets.
set(ENABLE_ROARING_TESTS OFF CACHE INTERNAL "")
set(ENABLE_ROARING_MICROBENCHMARKS OFF CACHE INTERNAL "")
set(ROARING_BUILD_STATIC ON CACHE INTERNAL "")
set(ROARING_USE_CPM OFF CACHE INTERNAL "")
vdb_fetch(croaring https://github.com/RoaringBitmap/CRoaring/archive/refs/tags/v4.2.1.tar.gz)
FetchContent_MakeAvailable(croaring)
add_library(vdb_roaring INTERFACE)
target_link_libraries(vdb_roaring INTERFACE roaring)
target_include_directories(vdb_roaring SYSTEM INTERFACE
  ${croaring_SOURCE_DIR}/include ${croaring_SOURCE_DIR}/cpp)

if(VDB_BUILD_TESTS)
  set(INSTALL_GTEST OFF CACHE INTERNAL "")
  set(gtest_force_shared_crt ON CACHE INTERNAL "")
  vdb_fetch(googletest https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz)
  FetchContent_MakeAvailable(googletest)
endif()

if(VDB_BUILD_BENCHMARKS)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "")
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE INTERNAL "")
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE INTERNAL "")
  set(BENCHMARK_ENABLE_WERROR OFF CACHE INTERNAL "")
  vdb_fetch(benchmark https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz)
  FetchContent_MakeAvailable(benchmark)
endif()

if(VDB_BUILD_SERVER OR VDB_BUILD_TOOLS)
  vdb_fetch(cli11 https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.4.2.tar.gz)
  set(CLI11_BUILD_TESTS OFF CACHE INTERNAL "")
  set(CLI11_BUILD_EXAMPLES OFF CACHE INTERNAL "")
  set(CLI11_BUILD_DOCS OFF CACHE INTERNAL "")
  FetchContent_MakeAvailable(cli11)
endif()

if(VDB_BUILD_SERVER)
  set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE INTERNAL "")
  set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE INTERNAL "")
  set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE INTERNAL "")
  set(HTTPLIB_USE_ZSTD_IF_AVAILABLE OFF CACHE INTERNAL "")
  vdb_fetch(httplib https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.18.3.tar.gz)
  FetchContent_MakeAvailable(httplib)

  set(SPDLOG_BUILD_EXAMPLE OFF CACHE INTERNAL "")
  set(SPDLOG_BUILD_TESTS OFF CACHE INTERNAL "")
  # 1.15.3 bundles fmt 11.2, which compiles with Clang 20+ (1.15.0's fmt 11.0 does not).
  vdb_fetch(spdlog https://github.com/gabime/spdlog/archive/refs/tags/v1.15.3.tar.gz)
  FetchContent_MakeAvailable(spdlog)
endif()
