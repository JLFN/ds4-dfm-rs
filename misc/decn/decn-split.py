#!/usr/bin/env python3
"""Split one job A nsys capture by phase and attribute the video VAE decoder.

The capture is the whole render, so the decoder is found by the kernels only it
launches (h3d_attention_mma_f32_kernel, h3d_video_qkv_rope_f32_kernel): their
first start and last end bound the FRAMES phase, and the CLI's own FRAMES line
is the cross-check. Everything inside that window is aggregated by kernel name
and, for the first block, listed in launch order.

Usage: decn-split.py <capture.sqlite> [--blocks N]
"""

import sqlite3
import sys
from collections import defaultdict

DECODER_ONLY = ("h3d_attention_mma_f32_kernel", "h3d_video_qkv_rope_f32_kernel")


def names(cur):
    return {row[0]: row[1] for row in cur.execute("select id, value from StringIds")}


def short(name):
    return name.split("(")[0].removeprefix("void ").removeprefix("cutlass::Kernel2<")


def load(cur, strings):
    rows = cur.execute(
        "select start, end, gridX, gridY, blockX, demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start"
    ).fetchall()
    return [
        (start, end, gx, gy, bx, strings.get(name_id, str(name_id)))
        for start, end, gx, gy, bx, name_id in rows
    ]


def window(kernels):
    starts = [k[0] for k in kernels if short(k[5]) in DECODER_ONLY]
    ends = [k[1] for k in kernels if short(k[5]) in DECODER_ONLY]
    if not starts:
        sys.exit("no decoder kernels found")
    return min(starts), max(ends)


def main():
    path = sys.argv[1]
    blocks = 2
    if "--blocks" in sys.argv:
        blocks = int(sys.argv[sys.argv.index("--blocks") + 1])

    db = sqlite3.connect(path)
    cur = db.cursor()
    strings = names(cur)
    kernels = load(cur, strings)
    first, last = window(kernels)
    inside = [k for k in kernels if k[0] >= first and k[1] <= last]

    span_ms = (last - first) / 1e6
    busy_ms = sum(k[1] - k[0] for k in inside) / 1e6
    print(
        f"decoder window: {span_ms / 1000:.3f} s, {len(inside)} kernels, "
        f"busy {busy_ms / 1000:.3f} s ({100 * busy_ms / span_ms:.1f}%)"
    )
    print()

    gaps = []
    previous_end = first
    previous_name = ""
    for start, end, _gx, _gy, _bx, _name in inside:
        if start > previous_end:
            gaps.append((start - previous_end, previous_end, previous_name, short(_name)))
        previous_end = max(previous_end, end)
        previous_name = short(_name)
    print(f"--- {len(gaps)} gaps, {sum(g[0] for g in gaps) / 1e6:.1f} ms total ---")
    for size, _at, before, after in sorted(gaps, reverse=True)[:14]:
        print(f"{size / 1e6:9.3f} ms   after {before:48s} before {after}")
    print()

    print("--- memcpys in the window ---")
    try:
        copies = cur.execute(
            "select start, end, copyKind, bytes, streamId from CUPTI_ACTIVITY_KIND_MEMCPY "
            "where start >= ? and end <= ? order by start",
            (first, last),
        ).fetchall()
    except sqlite3.OperationalError as error:
        copies = []
        print(f"no memcpy table: {error}")
    kinds = defaultdict(lambda: [0, 0, 0])
    for start, end, kind, count, _stream in copies:
        entry = kinds[kind]
        entry[0] += 1
        entry[1] += (end - start) / 1e6
        entry[2] += count
    for kind, (count, total_ms, bytes_moved) in sorted(kinds.items(), key=lambda kv: -kv[1][1]):
        print(f"copyKind {kind}: {count} copies, {total_ms / 1000:.3f} s, {bytes_moved / 1e9:.2f} GB")
    print()

    per = defaultdict(lambda: [0, 0, 10**9, 0])
    for start, end, gx, gy, bx, name in inside:
        d = (end - start) / 1e6
        key = f"{short(name)} | grid {gx}x{gy} block {bx}" if "--grid" in sys.argv else short(name)
        e = per[key]
        e[0] += 1
        e[1] += d
        e[2] = min(e[2], d)
        e[3] = max(e[3], d)

    print(f"{'kernel':68s} {'calls':>6s} {'total ms':>10s} {'avg ms':>8s} {'min':>8s} {'max':>8s}")
    for name, (calls, total, lo, hi) in sorted(per.items(), key=lambda kv: -kv[1][1]):
        print(f"{name:68s} {calls:6d} {total:10.2f} {total / calls:8.3f} {lo:8.3f} {hi:8.3f}")
    print(f"{'TOTAL':68s} {sum(v[0] for v in per.values()):6d} {sum(v[1] for v in per.values()):10.2f}")
    print()

    print(f"--- first {blocks} block(s) in launch order ---")
    for index, (start, end, gx, gy, bx, name) in enumerate(inside[: 12 * blocks]):
        print(f"{index:4d} {(end - start) / 1e6:9.3f} ms  grid {gx}x{gy} block {bx:5d}  {short(name)}")


if __name__ == "__main__":
    main()
