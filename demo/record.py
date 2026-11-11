#!/usr/bin/env python3
"""Run the demo scenes for real and record everything they print, with timestamps, plus
/metrics sampled every 0.5 s. demo/render.py turns the recording into a video.

    python demo/record.py --server build/release/vdb-server --build build/release --out demo/out/cast.json

Nothing is faked: every line in the video is output of the command shown above it. The
renderer only shortens long waits (and labels them with the real duration).
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.request

FILTER = '{"field": "title", "in": ["NASA", "Apollo program", "Space exploration"]}'
FILTER_SHOW = f"python demo/vdbctl.py search 'first people on the moon' --explain --filter '{FILTER}'"
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


class Recorder:
    def __init__(self, a):
        self.a = a
        self.t0 = time.time()
        self.steps = []
        self.metrics = []
        self.lock = threading.Lock()
        self.server = None
        self.stop = False
        self.env = dict(os.environ, PYTHONUNBUFFERED="1", TQDM_DISABLE="1", HF_HUB_OFFLINE="1",
                        HF_HUB_DISABLE_PROGRESS_BARS="1", TRANSFORMERS_VERBOSITY="error",
                        TOKENIZERS_PARALLELISM="false", VDB_URL=f"http://localhost:{a.port}")

    def now(self):
        return time.time() - self.t0

    def emit(self, text):
        with self.lock:
            if self.steps:
                self.steps[-1]["events"].append([round(self.now(), 3), text])

    def begin(self, scene, show):
        with self.lock:
            if self.steps:
                self.steps[-1]["end"] = round(self.now(), 3)
            self.steps.append({"scene": scene, "show": show, "start": round(self.now(), 3), "events": []})
        print(f"\n### {scene}\n$ {show}", flush=True)

    # Library chatter that is not part of what the demo shows (the model is cached locally).
    NOISE = ("Could not reach the Hub",)

    def pump(self, stream):
        """Forward output as it arrives, dropping whole NOISE lines. A partial line is held back
        only while it could still turn out to be one."""
        pending, started = "", False   # started: part of the current line was already emitted
        while True:
            chunk = os.read(stream.fileno(), 4096)
            if not chunk:
                if pending and not pending.startswith(self.NOISE):
                    self.emit(pending)
                return
            print(chunk.decode("utf-8", "replace"), end="", flush=True)
            pending += chunk.decode("utf-8", "replace")
            out = []
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                if started or not line.startswith(self.NOISE):
                    out.append(line + "\n")
                started = False
            if pending and (started or not any(n.startswith(pending) or pending.startswith(n) for n in self.NOISE)):
                out.append(pending)
                pending, started = "", True
            if out:
                self.emit("".join(out))

    def shell(self, scene, show, cmd):
        self.begin(scene, show)
        p = subprocess.Popen(["bash", "-c", cmd], cwd=REPO, env=self.env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        self.pump(p.stdout)
        rc = p.wait()
        if rc != 0:
            self.emit(f"[exit status {rc}]\n")

    def start_server(self, scene, show):
        self.begin(scene, show)
        self.server = subprocess.Popen([self.a.server, "--data-dir", self.a.data, "--port", str(self.a.port),
                                        "--threads", "64", "--log-level", "info"],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=self.env)
        threading.Thread(target=self.pump, args=(self.server.stdout,), daemon=True).start()
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://localhost:{self.a.port}/health", timeout=1).read()
                break
            except OSError:
                time.sleep(0.05)
        time.sleep(0.3)

    def stop_server(self, scene, show):
        """Graceful shutdown: SIGTERM, which finishes in-flight requests and syncs the logs."""
        self.begin(scene, show)
        self.server.send_signal(signal.SIGTERM)
        self.server.wait()
        time.sleep(1.0)
        self.emit(f"[1]+  Done                    vdb-server --data-dir {self.a.data_shown}\n")
        time.sleep(1.0)

    def kill_server(self, scene, show):
        self.begin(scene, show)
        pid = self.server.pid
        os.kill(pid, signal.SIGKILL)
        self.server.wait()
        time.sleep(0.2)
        self.emit(f"[1]+  Killed                  vdb-server --data-dir {self.a.data_shown} (pid {pid})\n")
        time.sleep(0.5)

    def sample_metrics(self):
        want = re.compile(r'^(vdb_[a-z_]+)(\{[^}]*\})? ([0-9.e+-]+)$')
        while not self.stop:
            t = self.now()
            m = None
            try:
                body = urllib.request.urlopen(f"http://localhost:{self.a.port}/metrics", timeout=0.4).read().decode()
                m = {}
                for line in body.splitlines():
                    g = want.match(line)
                    if g:
                        m[g.group(1) + (g.group(2) or "")] = float(g.group(3))
            except OSError:
                pass
            with self.lock:
                self.metrics.append([round(t, 3), m])
            time.sleep(0.5)

    def finish(self, out):
        self.stop = True
        with self.lock:
            if self.steps:
                self.steps[-1]["end"] = round(self.now(), 3)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump({"steps": self.steps, "metrics": self.metrics}, f)
        print(f"\nwrote {out}: {len(self.steps)} scenes, {len(self.metrics)} metric samples")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=os.path.join(REPO, "build/release/vdb-server"))
    ap.add_argument("--build", default=os.path.join(REPO, "build/release"), help="build tree for ctest")
    ap.add_argument("--python", default="python3")
    ap.add_argument("--data", default=os.path.expanduser("~/vdb-demo-data"))
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--out", default=os.path.join(REPO, "demo/out/cast.json"))
    a = ap.parse_args()
    a.data_shown = "~/" + os.path.relpath(a.data, os.path.expanduser("~"))
    shutil.rmtree(a.data, ignore_errors=True)
    py = a.python
    r = Recorder(a)
    threading.Thread(target=r.sample_metrics, daemon=True).start()

    r.shell("Build tree: run the whole C++ test suite", "ctest --test-dir build/release -j3",
            f"ctest --test-dir {a.build} -j3 2>&1 | grep -E 'Test +#|tests passed|Total Test'")
    r.start_server("Start the server (it recovers every collection before listening)",
                   f"vdb-server --data-dir {a.data_shown} --port {a.port} &")
    r.shell("Ingest 40,000 Wikipedia passages with metadata (JSON upserts, fsync before each ack)",
            "python demo/vdbctl.py load-wiki", f"{py} demo/vdbctl.py load-wiki")
    r.shell("Inspect the collection: background flushes and compaction already ran",
            "python demo/vdbctl.py stats wiki", f"sleep 2; {py} demo/vdbctl.py stats wiki")
    r.shell("Semantic search over HTTP", 'python demo/vdbctl.py search "Who discovered penicillin?"',
            f'{py} demo/vdbctl.py search "Who discovered penicillin?"')
    r.shell("Filtered search: the planner picks a strategy per segment (explain)",
            FILTER_SHOW, f"{py} demo/vdbctl.py search 'first people on the moon' --filter '{FILTER}' --explain")
    mock = subprocess.Popen([py, "rag/mock_llm_server.py", "--port", "8000"], cwd=REPO,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    r.shell("RAG: retrieve from vdb-server, answer with citations (LLM_URL = a stand-in mock LLM)",
            'LLM_URL=http://localhost:8000 python rag/ask.py --vdb-url http://localhost:8090 '
            '--collection wiki "When did the Berlin Wall fall?"',
            f'LLM_URL=http://localhost:8000 {py} rag/ask.py --vdb-url http://localhost:{a.port} '
            '--collection wiki "When did the Berlin Wall fall?"')
    mock.terminate()
    r.shell("Load: 8 concurrent clients for 8 seconds (watch the metrics panel)",
            "python demo/vdbctl.py burst --clients 8 --seconds 8",
            f"{py} demo/vdbctl.py burst --clients 8 --seconds 8")
    r.shell("Durability: 500 acknowledged writes that are only in the log (not flushed)",
            "python demo/vdbctl.py write 900000 500", f"{py} demo/vdbctl.py write 900000 500")
    r.kill_server("Crash it: kill -9, no shutdown, no flush", "kill -9 %1")
    r.start_server("Restart: recovery replays the log tail", f"vdb-server --data-dir {a.data_shown} --port {a.port} &")
    r.shell("Every acknowledged write survived", "python demo/vdbctl.py check 900000 500",
            f"{py} demo/vdbctl.py check 900000 500")
    time.sleep(1.0)
    r.stop_server("Graceful shutdown: finish in-flight requests, sync the logs, exit 0", "kill %1")
    r.finish(a.out)


if __name__ == "__main__":
    main()
