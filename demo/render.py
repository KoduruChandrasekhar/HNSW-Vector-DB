#!/usr/bin/env python3
"""Render demo/out/cast.json (from demo/record.py) into demo/out/vdb-demo.mp4 and .gif.

Left: the terminal, typed and printed as recorded. Right: the server's /metrics, sampled
live during the recording. Waits longer than about a second are shortened and labelled
with their real length; nothing else is changed.

    python demo/render.py [--cast demo/out/cast.json] [--fps 15]
"""

import argparse
import bisect
import json
import os
import re
import subprocess

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
BG, PANE, PANE2 = (13, 15, 21), (22, 26, 34), (17, 20, 27)
FG, DIM, GREEN, AMBER, RED, BLUE = (216, 222, 233), (120, 128, 145), (126, 200, 140), (230, 170, 90), (230, 100, 100), (110, 170, 240)
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
MONO = ImageFont.truetype(f"{FONT_DIR}/DejaVuSansMono.ttf", 14)
MONO_B = ImageFont.truetype(f"{FONT_DIR}/DejaVuSansMono-Bold.ttf", 14)
SANS = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans.ttf", 15)
SANS_B = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans-Bold.ttf", 16)
BIG = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans-Bold.ttf", 40)
MID = ImageFont.truetype(f"{FONT_DIR}/DejaVuSans.ttf", 20)
CW = MONO.getlength("M")
LH = 18
TERM = (16, 56, 868, 704)          # x0, y0, x1, y1
PANEL = (884, 56, 1264, 704)
COLS = int((TERM[2] - TERM[0] - 20) // CW)
ROWS = (TERM[3] - TERM[1] - 16) // LH
CAP = 0.9                          # longest wait kept as is (s)
TYPE_CPS = 55


# --------------------------------------------------------------------------- timeline

def build_timeline(cast):
    """Compressed-time events: (t, kind, payload); plus (real, compressed) anchors for metrics."""
    events, anchors, skips = [], [], []
    t = 0.0
    for i, st in enumerate(cast["steps"]):
        events.append((t, "scene", (i, st["scene"])))
        anchors.append((st["start"], t))
        typing = min(3.0, max(0.6, len(st["show"]) / TYPE_CPS))
        events.append((t, "type", (st["show"], typing)))
        t += typing + 0.25
        events.append((t, "enter", None))
        anchors.append((st["start"], t))
        prev = st["start"]
        lines = 0
        for rt, text in st["events"]:
            gap = rt - prev
            if gap > CAP + 0.4:
                skips.append((t, t + CAP, gap))
                t += CAP
            else:
                t += max(gap, 0)
            prev = rt
            anchors.append((rt, t))
            events.append((t, "out", text))
            lines += text.count("\n")
        tail = st["end"] - prev
        t += min(max(tail, 0), CAP)
        anchors.append((st["end"], t))
        t += min(7.0, max(2.5, lines * 0.3))   # time to read the output
        anchors.append((st["end"], t))
    return events, anchors, skips, t


def real_time(anchors, tc):
    """Map compressed time back to recording time (piecewise linear)."""
    cs = [a[1] for a in anchors]
    j = bisect.bisect_right(cs, tc)
    if j == 0:
        return anchors[0][0]
    if j >= len(anchors):
        return anchors[-1][0]
    (r0, c0), (r1, c1) = anchors[j - 1], anchors[j]
    return r0 if c1 == c0 else r0 + (r1 - r0) * (tc - c0) / (c1 - c0)


# --------------------------------------------------------------------------- terminal

class Terminal:
    def __init__(self):
        self.lines = [[]]   # each line: list of (text, color)
        self.color = FG

    def write(self, text, color=None):
        color = color or FG
        for ch in text:
            if ch == "\n":
                self.lines.append([])
            elif ch == "\r":
                self.lines[-1] = []
            elif ch == "\x1b":
                continue
            else:
                self.lines[-1].append((ch, color))

    def display(self):
        out = []
        for ln in self.lines:
            if not ln:
                out.append([])
            for s in range(0, len(ln), COLS):
                out.append(ln[s:s + COLS])
        return out[-ROWS:]


def colorize(text):
    """Server log lines dim, errors red, everything else default."""
    return DIM if re.match(r"^\[\d{4}-\d\d-\d\d", text) else FG


# --------------------------------------------------------------------------- metrics

def metric_sum(m, name, **labels):
    tot = 0.0
    for k, v in m.items():
        if not k.startswith(name):
            continue
        rest = k[len(name):]
        if rest and not rest.startswith("{"):
            continue
        if all(f'{a}="{b}"' in rest for a, b in labels.items()):
            tot += v
    return tot


def searches(m):
    return metric_sum(m, "vdb_requests_total", route="search") + metric_sum(m, "vdb_requests_total", route="search_batch")


def quantiles(m0, m1, route, qs=(0.5, 0.99)):
    """Latency quantiles from the histogram buckets added between two samples."""
    pat = re.compile(r'vdb_request_duration_seconds_bucket\{route="' + route + r'",le="([^"]+)"\}')
    b = []
    for k, v in m1.items():
        g = pat.match(k)
        if g:
            le = float("inf") if g.group(1) == "+Inf" else float(g.group(1))
            b.append((le, v - (m0 or {}).get(k, 0)))
    b.sort()
    if not b or b[-1][1] <= 0:
        return None
    total = b[-1][1]
    res = []
    for q in qs:
        need = q * total
        res.append(next((le for le, c in b if c >= need), b[-1][0]))
    return res


class Metrics:
    def __init__(self, samples):
        self.samples = samples
        self.ts = [s[0] for s in samples]

    def at(self, r):
        j = bisect.bisect_right(self.ts, r) - 1
        return j

    def get(self, j):
        return self.samples[j][1] if 0 <= j < len(self.samples) else None

    def qps_series(self, j, n=40):
        out = []
        for i in range(max(1, j - n + 1), j + 1):
            a, b = self.get(i - 1), self.get(i)
            if a is None or b is None:
                out.append(None)
                continue
            dt = self.samples[i][0] - self.samples[i - 1][0]
            out.append(max(0.0, (searches(b) - searches(a)) / dt))
        return out


def fmt_n(x):
    return f"{x / 1e6:.2f}M" if x >= 1e6 else f"{x / 1e3:.1f}k" if x >= 1e4 else f"{x:.0f}"


def fmt_ms(s):
    return "inf" if s == float("inf") else f"{s * 1e3:.2f} ms" if s < 0.01 else f"{s * 1e3:.0f} ms"


def draw_panel(d, mets, r):
    x0, y0, x1, y1 = PANEL
    d.rounded_rectangle(PANEL, 10, fill=PANE2)
    d.text((x0 + 16, y0 + 12), "GET /metrics  (live)", font=SANS_B, fill=FG)
    j = mets.at(r)
    m = mets.get(j)
    y = y0 + 44
    if m is None:
        d.text((x0 + 16, y), "● server down", font=SANS_B, fill=RED)
        return
    d.text((x0 + 16, y), "● server up", font=SANS_B, fill=GREEN)
    note = "counters are per process"
    d.text((x1 - 16 - SANS.getlength(note), y + 1), note, font=SANS, fill=DIM)
    y += 32

    def row(label, value, color=FG):
        nonlocal y
        d.text((x0 + 16, y), label, font=SANS, fill=DIM)
        d.text((x1 - 16 - SANS.getlength(value), y), value, font=SANS, fill=color)
        y += 24

    def head(title):
        nonlocal y
        y += 6
        d.text((x0 + 16, y), title, font=SANS_B, fill=AMBER)
        y += 26

    rows = metric_sum(m, "vdb_collection_rows", collection="wiki")
    segs = metric_sum(m, "vdb_segments", collection="wiki")
    mem = metric_sum(m, "vdb_memory_bytes", collection="wiki")
    head('collection "wiki"')
    row("rows", fmt_n(rows))
    row("segments (incl. mutable)", f"{segs:.0f}")
    row("memory", f"{mem / 1e6:.1f} MB")

    head("requests")
    row("search (+ search-batch)", fmt_n(searches(m)))
    row("upsert", fmt_n(metric_sum(m, "vdb_requests_total", route="upsert")))
    other = metric_sum(m, "vdb_requests_total") - searches(m) - metric_sum(m, "vdb_requests_total", route="upsert") \
        - metric_sum(m, "vdb_requests_total", route="metrics")
    row("other (not counting these scrapes)", fmt_n(other))

    series = mets.qps_series(j)
    cur = next((v for v in reversed(series) if v is not None), 0.0)
    head("search QPS")
    d.text((x1 - 16 - SANS_B.getlength(f"{cur:.0f}/s"), y - 26), f"{cur:.0f}/s", font=SANS_B, fill=BLUE)
    gx0, gy0, gx1, gy1 = x0 + 16, y, x1 - 16, y + 60
    d.rectangle((gx0, gy0, gx1, gy1), outline=(40, 46, 58))
    vals = [v or 0 for v in series]
    top = max(max(vals), 50.0)
    pts = [(gx0 + (gx1 - gx0) * i / max(1, len(vals) - 1), gy1 - (gy1 - gy0 - 4) * v / top) for i, v in enumerate(vals)]
    if len(pts) > 1:
        d.line(pts, fill=BLUE, width=2)
    y = gy1 + 8
    j0 = max(0, j - 4)
    q = quantiles(mets.get(j0), m, "search")
    row("latency p50 / p99", f"{fmt_ms(q[0])} / {fmt_ms(q[1])}" if q else "-")

    head("write path")
    row("WAL records", fmt_n(metric_sum(m, "vdb_wal_records_total")))
    row("fsyncs (group commit)", fmt_n(metric_sum(m, "vdb_wal_fsync_duration_seconds_count")))
    row("flushes / compactions", f"{metric_sum(m, 'vdb_flush_duration_seconds_count'):.0f} / "
                                 f"{metric_sum(m, 'vdb_compaction_duration_seconds_count'):.0f}")


# --------------------------------------------------------------------------- frames

def card(lines, sub=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = 190 if sub is None else 150
    for i, (text, font, color) in enumerate(lines):
        d.text(((W - font.getlength(text)) / 2, y), text, font=font, fill=color)
        y += font.size + 18
    return img


def title_card():
    return card([("vdb", BIG, FG),
                 ("a vector database in C++20, built from scratch", MID, FG),
                 ("", MID, FG),
                 ("HNSW · SQ8 · IVF-PQ  ·  write-ahead log, segments, crash recovery", SANS, DIM),
                 ("metadata filters with a query planner  ·  HTTP server  ·  Python bindings", SANS, DIM),
                 ("", MID, FG),
                 ("Live recording: every line below is real output of the command above it.", SANS, AMBER),
                 ("Long waits are shortened and labelled with their real length.", SANS, AMBER)])


def end_card(test_line):
    return card([("What you just saw", BIG, FG),
                 ("", SANS, FG),
                 (test_line or "C++ test suite: all passing", MID, GREEN),
                 ("40,000 passages ingested with metadata, flushed and compacted in the background", SANS, FG),
                 ("filtered search planned per segment  ·  RAG answers with citations", SANS, FG),
                 ("kill -9 mid-flight: every acknowledged write recovered from the log", SANS, FG),
                 ("", SANS, FG),
                 ("Benchmarks (SIFT1M, one thread): recall@10 0.974 at 3,391 QPS, vs hnswlib 2,601", SANS, DIM),
                 ("SQ8: 292 bytes/vector at the same recall  ·  1,000 kill -9 runs, 0 lost writes", SANS, DIM),
                 ("", SANS, FG),
                 ("README.md · docs/API.md · docs/WRITEUP.md", SANS_B, AMBER)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--cast", default=os.path.join(here, "out", "cast.json"))
    ap.add_argument("--out", default=os.path.join(here, "out", "vdb-demo"))
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--frames", help="also save these frame numbers as PNG, e.g. 100,900")
    ap.add_argument("--scene-frames", action="store_true", help="save the last frame of every scene as PNG")
    a = ap.parse_args()
    cast = json.load(open(a.cast))
    events, anchors, skips, total = build_timeline(cast)
    mets = Metrics(cast["metrics"])
    nsteps = len(cast["steps"])
    test_line = next((ln.strip() for st in cast["steps"] for _, t in st["events"] for ln in t.splitlines()
                      if "tests passed" in ln), None)

    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                           "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out + ".mp4"],
                          stdin=subprocess.PIPE)
    keep = {int(x) for x in a.frames.split(",")} if a.frames else set()
    frame_no = 0

    def put(img, seconds=None):
        nonlocal frame_no
        raw = img.tobytes()
        for _ in range(1 if seconds is None else int(seconds * a.fps)):
            if frame_no in keep:
                img.save(f"{a.out}_frame{frame_no}.png")
            ff.stdin.write(raw)
            frame_no += 1

    put(title_card(), 5.0)

    term = Terminal()
    scene = (0, "")
    typing = None   # (text, start, duration)
    ei = 0
    last_key, last_img = None, None
    n = int(total * a.fps) + 1
    for f in range(n):
        tc = f / a.fps
        while ei < len(events) and events[ei][0] <= tc:
            _, kind, payload = events[ei]
            if kind == "scene":
                if a.scene_frames and last_img is not None:
                    last_img.save(f"{a.out}_scene{scene[0] + 1:02d}.png")
                scene = payload
                if payload[0] > 0:
                    term.write("\n")
            elif kind == "type":
                typing = (payload[0], events[ei][0], payload[1])
            elif kind == "enter":
                term.write("$ ", GREEN)
                term.write(typing[0] + "\n", (240, 240, 200))
                typing = None
            elif kind == "out":
                for part in re.split(r"(\n)", payload):
                    term.write(part, colorize(term_line_prefix(term) + part))
            ei += 1
        skip = next((s for s in skips if s[0] <= tc < s[1]), None)
        r = real_time(anchors, tc)
        shown = None
        if typing:
            k = int(len(typing[0]) * min(1.0, (tc - typing[1]) / typing[2]))
            shown = typing[0][:k]
        key = (len(term.lines), len(term.lines[-1]), shown, scene[0], skip is not None, mets.at(r))
        if key == last_key:
            put(last_img)
            continue
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((16, 16), "vdb", font=SANS_B, fill=FG)
        cap = f"{scene[0] + 1}/{nsteps}  {scene[1]}"
        d.text((60, 17), cap, font=SANS, fill=AMBER)
        d.rounded_rectangle(TERM, 10, fill=PANE)
        disp = term.display()
        if shown is not None:
            disp = disp + [[("$ ", GREEN)] + [(ch, (240, 240, 200)) for ch in shown] + [("█", FG)]]
            disp = disp[-ROWS:]
        y = TERM[1] + 10
        for ln in disp:
            i = 0
            while i < len(ln):  # one draw call per run of same-coloured characters
                j = i
                while j < len(ln) and ln[j][1] == ln[i][1]:
                    j += 1
                d.text((TERM[0] + 10 + i * CW, y), "".join(ch for ch, _ in ln[i:j]), font=MONO, fill=ln[i][1])
                i = j
            y += LH
        if skip:
            label = f"  waited {skip[2]:.1f} s (shortened)  "
            bw = SANS_B.getlength(label)
            d.rounded_rectangle((TERM[2] - bw - 20, TERM[1] + 10, TERM[2] - 10, TERM[1] + 38), 8, fill=(70, 55, 25))
            d.text((TERM[2] - bw - 15, TERM[1] + 14), label, font=SANS_B, fill=AMBER)
        draw_panel(d, mets, r)
        last_key, last_img = key, img
        put(img)

    if a.scene_frames and last_img is not None:
        last_img.save(f"{a.out}_scene{scene[0] + 1:02d}.png")
    put(end_card(test_line), 7.0)
    ff.stdin.close()
    ff.wait()
    print(f"wrote {a.out}.mp4: {frame_no} frames, {frame_no / a.fps:.0f} s")
    # A smaller GIF for the README.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.out + ".mp4", "-vf",
                    "fps=8,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];"
                    "[s1][p]paletteuse=dither=none", a.out + ".gif"], check=True)
    print(f"wrote {a.out}.gif ({os.path.getsize(a.out + '.gif') / 1e6:.1f} MB)")


def term_line_prefix(term):
    return "".join(ch for ch, _ in term.lines[-1])


if __name__ == "__main__":
    main()
