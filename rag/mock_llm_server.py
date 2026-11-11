#!/usr/bin/env python3
"""A stand-in for your LLM inference server, for testing rag/ask.py's plumbing only.

    python rag/mock_llm_server.py --port 8000 &
    LLM_URL=http://localhost:8000 python rag/ask.py "Who discovered penicillin?"

It implements POST /v1/chat/completions with "stream": true (server-sent events) and
answers by quoting the first sentence of the first passage it was given, with a [1]
citation. It is not a language model; point LLM_URL at your real server for answers.
"""

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        prompt = body["messages"][-1]["content"]
        m = re.search(r"\[1\][^\n]*\n(.+?)(?:\n\n\[2\]|\n\nQuestion:)", prompt, re.S)
        first = re.split(r"(?<=[.!?])\s", m.group(1).strip())[0] if m else "The passages do not say."
        answer = f"According to the first passage, {first[0].lower() + first[1:] if first else first} [1]"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for word in answer.split(" "):
            chunk = {"choices": [{"delta": {"content": word + " "}, "index": 0}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.01)
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
