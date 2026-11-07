#!/usr/bin/env python3
"""Score H3 clips with the reference campaign's own harness code, and gate a change against a baseline clip.

The metrics, models and flag thresholds are not re-implemented here. This imports the r2 campaign's
clip scorer (`harness/eval_clips.py` from the Rapid-H3 reference tree, byte-identical copy, sha256 printed
on every run) and calls its criterion 9 block verbatim -- MUSIQ and CLIP-IQA from pyiqa over 8 frames at
linspace(0, n-1, 8), DOVER's fused technical/aesthetic score, Laplacian variance of the frame, and the
wavelet noise sigma from skimage -- plus its `flow_block` (flicker on static pixels, temporal
high-frequency energy, occlusion-masked warping error, mean RAFT flow, acceleration, camera zoom,
frozen). Its own THRESH and `flags_for()` then judge the change against `--baseline` with the harness's
semantics: a `_d` key is "this clip minus the baseline", an `_r` key is "this clip over the baseline".
A tenth metric is the harness's `comp_ssim`: thumbnail SSIM every 6th frame, i.e. "is this the same
picture".

What is left unscored is what needs inputs this port does not have: no reference image (criterion 7),
no shot script (dialogue, voice), no face model (identity). Those criteria stay out.

Two clips are only comparable at the same canvas: MUSIQ, CLIP-IQA, sharpness and flow all move with
resolution, so a pair that differs in size gets `comp_ssim` left empty and its ratios are not a verdict.

  H3_EVAL_HARNESS=~/ds4-test/harness ~/ds4-test/venv_eval/bin/python tools/h3-clip-score.py \
      --baseline outputs/two-stage/one.mp4 outputs/two-stage/one.mp4 outputs/two-stage/draft.mp4 \
      --out ~/ds4-test/score-route.json

Exit status: 0 when no non-baseline clip carries a flag, 1 when any does.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", help="clips to score, in order")
    ap.add_argument("--baseline", default=None, help="clip the others are judged against (_d / _r)")
    ap.add_argument("--harness", default=os.environ.get("H3_EVAL_HARNESS", os.path.expanduser("~/ds4-test/harness")),
                    help="directory holding the reference eval_clips.py")
    ap.add_argument("--dover-dir", default=os.environ.get("H3_DOVER_DIR", os.path.expanduser("~/ds4-test/DOVER")),
                    help="DOVER checkout with pretrained_weights/DOVER.pth")
    ap.add_argument("--cache", default=os.environ.get("RAPIDH3_EVAL_CACHE", os.path.expanduser("~/rapid-h3/eval_cache")),
                    help="the harness's model cache (its own default: ~/rapid-h3/eval_cache)")
    ap.add_argument("--no-dover", action="store_true", help="skip DOVER, the one metric that is not deterministic")
    ap.add_argument("--no-flow", action="store_true", help="skip the RAFT flow block")
    ap.add_argument("--out", default=None, help="write the records here as JSON")
    a = ap.parse_args()

    for clip in a.clips + ([a.baseline] if a.baseline else []):
        if not os.path.exists(clip):
            sys.exit(f"h3-clip-score: no such clip: {clip}")

    os.environ["RAPIDH3_EVAL_CACHE"] = a.cache  # eval_clips reads it at import, and sets HF_HOME/TORCH_HOME from it
    sys.path.insert(0, a.harness)
    import types

    import eval_clips as E

    E.ARGS = types.SimpleNamespace(dover_dir=a.dover_dir, no_dover=a.no_dover, threads=8, cache=a.cache, force=True)

    import cv2
    import pyiqa
    import torch

    harness = os.path.join(a.harness, "eval_clips.py")
    print(f"harness {harness} sha256={sha256(harness)} pyiqa={pyiqa.__version__} torch={torch.__version__} "
          f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} cache={a.cache}", flush=True)

    thumbs = {}

    def measure(path):
        t0 = time.time()
        frames, fps, _audio, _sr = E.load_video(path)
        n = len(frames)
        h, w = frames[0].shape[:2]
        rec = {"path": path, "frames": n, "fps": fps, "size": [w, h], "seconds": E.rnd(n / fps, 3)}
        pick = [int(round(x)) for x in np.linspace(0, n - 1, 8)]
        thumbs[path] = np.array([cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY),
                                            (E.THUMB, max(1, int(E.THUMB * h / w))),
                                            interpolation=cv2.INTER_AREA) for i in range(n)])
        with torch.no_grad():
            x = E.to_t([frames[i] for i in pick])
            rec["musiq"] = E.rnd(E.iqa("musiq")(x).mean().item(), 2)
            rec["clipiqa"] = E.rnd(E.iqa("clipiqa")(x).mean().item())
        if not a.no_dover:
            try:
                rec.update(E.dover_score(frames))
            except Exception as exc:  # DOVER lives outside the venv: a missing checkout is not a failed clip
                print("dover failed:", repr(exc)[:300], flush=True)
        from skimage.restoration import estimate_sigma
        rec["sharp_frame"] = E.rnd(np.mean([E.lapvar(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)) for i in pick]), 1)
        rec["noise"] = E.rnd(np.mean([estimate_sigma(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float32))
                                      for i in pick]))
        if not a.no_flow:
            rec.update({k: v for k, v in E.flow_block(frames, fps).items() if not k.startswith("_")})
        rec["wall_s"] = E.rnd(time.time() - t0, 1)
        return rec

    base = measure(a.baseline) if a.baseline else None
    if base:
        print("baseline measured", base["path"], flush=True)
    recs = [base if (base and clip == a.baseline) else measure(clip) for clip in a.clips]

    if base:
        from skimage.metrics import structural_similarity
        for rec in recs:
            if rec is base:
                continue
            for key, dkey in E.DELTA_OF.items():
                if rec.get(key) is None or not base.get(key):
                    continue
                rec[dkey] = E.rnd(rec[key] - base[key]) if dkey.endswith("_d") else E.rnd(rec[key] / base[key])
            ta, tb = thumbs[rec["path"]], thumbs[base["path"]]
            if ta.shape[1:] != tb.shape[1:]:  # different canvases: nothing to compare, and SSIM refuses the shapes
                rec["comp_ssim"] = None
            else:
                k = min(len(ta), len(tb))
                rec["comp_ssim"] = E.rnd(np.mean([structural_similarity(ta[j], tb[j], data_range=255)
                                                  for j in range(0, k, 6)]))
    for rec in recs:
        rec["flags"] = E.flags_for(rec, is_ref=(rec is base), has_line=False)

    print("  ".join(["clip", "size", "musiq", "clipiqa", "dover", "sharp", "noise", "flick", "tflick",
                     "warp", "flow", "accel", "zoom", "c_ssim", "flags"]))
    for rec in recs:
        def cell(key, fmt):
            v, d = rec.get(key), rec.get(E.DELTA_OF.get(key, ""))
            s = "" if v is None else fmt.format(v)
            if d is not None and s:
                s += f" ({d:+.3g})" if key in ("musiq", "clipiqa", "dover") else f" (x{d:.3g})"
            return s
        print("  ".join([os.path.basename(rec["path"]), f"{rec['size'][0]}x{rec['size'][1]}",
                         cell("musiq", "{:.2f}"), cell("clipiqa", "{:.3f}"), cell("dover", "{:.3f}"),
                         cell("sharp_frame", "{:.1f}"), cell("noise", "{:.2f}"), cell("flicker", "{:.3f}"),
                         cell("tflick", "{:.3f}"), cell("warp_err", "{:.1f}"), cell("flow_mag", "{:.2f}"),
                         cell("accel", "{:.3f}"), cell("zoom", "{:.3f}"),
                         "" if rec.get("comp_ssim") is None else f"{rec['comp_ssim']:.3f}",
                         ",".join(rec["flags"]) or "-"]))

    for rec in recs:
        if rec is base:
            continue
        name = os.path.basename(rec["path"])
        if rec["flags"]:
            print("FLAG " + name + ": " + ", ".join(rec["flags"]))
        elif base:
            print("PASS " + name + ": no flag against " + os.path.basename(base["path"]))
        else:
            print("PASS " + name + ": scored, no baseline")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"harness_sha256": sha256(harness), "baseline": a.baseline, "clips": recs}, fh, indent=1)
        print("wrote", a.out)
    sys.exit(1 if base and any(r["flags"] for r in recs if r is not base) else 0)


if __name__ == "__main__":
    main()
