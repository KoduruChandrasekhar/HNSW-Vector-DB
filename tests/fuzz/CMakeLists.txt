# libFuzzer targets (Clang). Run each for a while, e.g. 60 s in CI:
#   ./wal_fuzz -max_total_time=60 corpus_wal/
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR "VDB_BUILD_FUZZERS needs Clang (libFuzzer)")
endif()
foreach(name wal_fuzz manifest_fuzz filter_fuzz binary_upsert_fuzz)
  add_executable(${name} ${name}.cc)
  target_compile_options(${name} PRIVATE -fsanitize=fuzzer,address,undefined)
  target_link_options(${name} PRIVATE -fsanitize=fuzzer,address,undefined)
  target_link_libraries(${name} PRIVATE vdb_internal)
endforeach()
target_link_libraries(binary_upsert_fuzz PRIVATE vdb_server_lib)
