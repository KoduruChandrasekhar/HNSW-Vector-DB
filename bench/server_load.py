#!/usr/bin/env python3
"""Open-loop HTTP load generator for vdb-server: fixed offered rates, latency percentiles.

    python bench/server_load.py --url http://127.0.0.1:8090 --collection sift \
        --data-dir data --dataset sift --rates 100,500,1000,2000,4000 --seconds 10

Requests are scheduled at fixed intervals whether or not earlier ones finished (open
loop), so queueing shows up in the latency instead of silently lowering the rate. 503
answers (the bounded search queue) are counted separately.
"""

import argparse
import asyncio
import json
import os
import struct
import time

import numpy as np


def read_fbin(path, max_rows):
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        n = min(n, max_rows)
        return np.fromfile(f, dtype=np.float32, count=n * d).reshape(n, d)


class Pool:
    """Keep-alive HTTP/1.1 connections over asyncio streams."""

    def __init__(self, host, port, size):
        self.host, self.port, self.size = host, port, size
        self.free = asyncio.Queue()

    async def start(self):
        for _ in range(self.size):
            self.free.put_nowait(await asyncio.open_connection(self.host, self.port))

    async def post(self, path, body):
        # The server closes a keep-alive connection after a fixed number of requests;
        # that is not a failed request, so reconnect and retry once.
        st = await self._post_once(path, body)
        return st if st != -1 else await self._post_once(path, body)

    async def _post_once(self, path, body):
        reader, writer = await self.free.get()
        try:
            data = body.encode()
            writer.write(f"POST {path} HTTP/1.1\r\nHost: {self.host}\r\nContent-Type: application/json\r\n"
                         f"Content-Length: {len(data)}\r\n\r\n".encode() + data)
            await writer.drain()
            status_line = await reader.readline()
            status = int(status_line.split()[1])
            length = 0
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b""):
                    break
                k, _, v = line.decode().partition(":")
                if k.lower() == "content-length":
                    length = int(v)
            await reader.readexactly(length)
            self.free.put_nowait((reader, writer))
            return status
        except Exception:
            writer.close()
            self.free.put_nowait(await asyncio.open_connection(self.host, self.port))
            return -1


async def run_rate(pool, path, bodies, rate, seconds):
    lat, codes, tasks = [], {}, []
    interval = 1.0 / rate
    start = time.perf_counter()

    async def one(body, scheduled):
        st = await pool.post(path, body)
        lat.append((time.perf_counter() - scheduled) * 1e3)  # from the scheduled time: includes queueing
        codes[st] = codes.get(st, 0) + 1

    i = 0
    while True:
        scheduled = start + i * interval
        if scheduled - start >= seconds:
            break
        now = time.perf_counter()
        if scheduled > now:
            await asyncio.sleep(scheduled - now)
        tasks.append(asyncio.create_task(one(bodies[i % len(bodies)], scheduled)))
        i += 1
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start
    ok = codes.get(200, 0)
    return {"offered_qps": rate, "achieved_qps": ok / elapsed, "requests": i, "codes": codes,
            "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99))}


async def main_async(a):
    host, port = a.url.split("//")[1].split(":")
    q = read_fbin(os.path.join(a.data_dir, a.dataset, "query.fbin"), 10000)
    bodies = [json.dumps({"vector": row.tolist(), "k": a.k, "ef": a.ef}) for row in q]
    pool = Pool(host, int(port), a.connections)
    await pool.start()
    path = f"/v1/collections/{a.collection}/search"
    results = []
    for rate in [int(r) for r in a.rates.split(",")]:
        r = await run_rate(pool, path, bodies, rate, a.seconds)
        r.update({"system": "vdb-server", "ef": a.ef, "k": a.k, "segments": a.segments})
        print(f"  offered {rate:6d} qps: achieved {r['achieved_qps']:8.1f}  p50 {r['p50_ms']:7.2f} ms  "
              f"p99 {r['p99_ms']:8.2f} ms  codes {r['codes']}", flush=True)
        results.append(r)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--collection", default="sift")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--dataset", default="sift")
    ap.add_argument("--rates", default="100,500,1000,2000,4000")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--connections", type=int, default=64)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef", type=int, default=64)
    ap.add_argument("--segments", type=int, default=1, help="recorded beside the numbers")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    results = asyncio.run(main_async(a))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"command": "server_load", "results": results}, f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
