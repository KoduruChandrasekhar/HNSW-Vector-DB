// vdb-server --data-dir ./data --port 8090 --threads 8 --log-level info
// Options can also come from a TOML/INI file: --config server.toml

#include <httplib.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <atomic>
#include <chrono>
#include <csignal>
#include <thread>

#include "server/collection_manager.h"
#include "server/http_api.h"

namespace {

std::atomic<bool> g_stop{false};

void OnSignal(int) { g_stop.store(true); }

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"vdb-server: vector database HTTP server"};
  std::string data_dir = "./data";
  std::string host = "0.0.0.0";
  int port = 8090;
  size_t threads = 64;  // cpp-httplib holds one thread per keep-alive connection
  size_t index_threads = 0;
  std::string log_level = "info";
  size_t max_body_mb = 1024;
  size_t max_inflight = 64;
  size_t max_k = 1000;
  double slow_ms = 50;
  bool verify_checksums = true;
  app.add_option("--data-dir", data_dir, "Directory holding collections/");
  app.add_option("--host", host, "Address to bind");
  app.add_option("--port", port, "Port to listen on");
  app.add_option("--threads", threads, "HTTP worker threads (one per open keep-alive connection)");
  app.add_option("--index-threads", index_threads, "Threads for index builds and batch search (0 = all cores)");
  app.add_option("--log-level", log_level, "trace, debug, info, warn, error");
  app.add_option("--max-body-mb", max_body_mb, "Largest accepted request body");
  app.add_option("--max-inflight-searches", max_inflight, "Searches in flight before answering 503");
  app.add_option("--max-k", max_k, "Largest k a search may ask for");
  app.add_option("--slow-query-ms", slow_ms, "Log searches slower than this, with their explain output");
  app.add_option("--verify-checksums", verify_checksums, "Verify segment CRCs when opening collections");
  app.set_config("--config", "", "Read options from a TOML or INI file");
  CLI11_PARSE(app, argc, argv);

  spdlog::set_level(spdlog::level::from_str(log_level));
  vdb::OpenOptions open;
  open.num_threads = index_threads;
  open.verify_checksums = verify_checksums;
  vdb::CollectionManager mgr(data_dir, open);
  auto t0 = std::chrono::steady_clock::now();
  vdb::Status s = mgr.OpenAll();
  if (!s.ok()) {
    spdlog::critical("recovery failed: {}", s.ToString());
    return 1;
  }
  for (const auto& [name, c] : mgr.List()) {
    vdb::CollectionStats st = c->Stats();
    spdlog::info("opened collection {}: {} rows in {} segments, replayed {} log records in {:.3f} s", name,
                 st.live_rows, st.segments.size(), st.last_recovery_records, st.last_recovery_seconds);
  }
  spdlog::info("recovery took {:.3f} s",
               std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

  httplib::Server svr;
  svr.new_task_queue = [threads] { return new httplib::ThreadPool(threads); };
  svr.set_keep_alive_max_count(1000);
  // Responses are written as headers then body; with Nagle on, the second write waits
  // for the client's delayed ACK and every request costs ~40 ms.
  svr.set_tcp_nodelay(true);
  vdb::ServerConfig cfg;
  cfg.max_body_bytes = max_body_mb << 20;
  cfg.max_inflight_searches = max_inflight;
  cfg.max_k = max_k;
  cfg.slow_query_ms = slow_ms;
  vdb::HttpApi api(&mgr, cfg);
  api.Register(svr);

  // Graceful shutdown on SIGTERM / SIGINT: stop accepting, finish in-flight requests,
  // then sync the logs and close files.
  std::signal(SIGTERM, OnSignal);
  std::signal(SIGINT, OnSignal);
  std::signal(SIGPIPE, SIG_IGN);
  std::thread watcher([&svr] {
    while (!g_stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    spdlog::info("shutting down");
    svr.stop();
  });

  if (!svr.bind_to_port(host, port)) {
    spdlog::critical("cannot bind {}:{}", host, port);
    g_stop.store(true);
    watcher.join();
    return 1;
  }
  spdlog::info("vdb-server listening on {}:{} (data dir {})", host, port, data_dir);
  svr.listen_after_bind();
  g_stop.store(true);
  watcher.join();
  s = mgr.CloseAll();
  if (!s.ok()) spdlog::error("close: {}", s.ToString());
  spdlog::info("bye");
  return 0;
}
