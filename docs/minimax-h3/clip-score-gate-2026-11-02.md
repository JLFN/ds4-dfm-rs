# The clip-scoring gate: the reference harness, pointed at two clips from this port (2026-11-02)

Unit: handoff section 4, the instrument the numerics decisions need. Every
numerics arm (E4M3 attention, a fused `amax`+`quantize`, the decode levers)
has to answer one question that PSNR cannot: **does the picture get worse?**
The r2 campaign settled that question for its own renders with
`harness/eval_clips.py`, and until now nothing in this port had ever run it.

It has now, on the clips this port already rendered, with the harness's own
metric code, models and `THRESH` - and it does not agree with the intuitive
answer.

**The verdict first.**

| | |
|---|---|
| the instrument | `tools/h3-clip-score.py`, the reference harness's criterion 9 + `flow_block`, imported not re-implemented |
| harness copy | `harness/eval_clips.py` sha256 `cdc1656442e491fb…`, byte-identical to the reference tree's file |
| cost per clip | 26.6 s (1344x768), 23.0 s, 11.5 s (928x544) on the GB10 |
| deterministic | nine of the ten metrics reproduce to the printed digit across five runs |
| not deterministic | **DOVER: 0.667, 0.670, 0.677, 0.683, 0.691 on one unchanged file** |
| two-stage route pair | **flagged: `sharp_frame_r` x0.500**, and `comp_ssim` 0.147 |
| MUSIQ and CLIP-IQA | **both rise** on the softer clip (+1.23, +0.069) |
| exit status | 1 on a flagged pair, 0 otherwise |

## 1. What runs

`tools/h3-clip-score.py` does not re-implement the metrics. It puts the
reference tree's `harness/eval_clips.py` on `sys.path` and calls the harness's
own functions:

- criterion 9, exactly as `measure_clip` calls it: 8 frames at
  `linspace(0, n-1, 8)` through `to_t`, `iqa("musiq")` and `iqa("clipiqa")`
  (pyiqa), `dover_score` (its `fuse_results` and both heads), `lapvar` on the
  grey frame, and `estimate_sigma` from skimage;
- `flow_block`: RAFT-large at half resolution, forward and backward flow of
  every consecutive pair, then flicker on pixels static both ways, temporal
  high-frequency energy, occlusion-masked warping error, mean flow,
  acceleration, camera zoom, frozen;
- `THRESH` and `flags_for()`, with the campaign's own semantics: a `_d` key is
  "this clip minus the baseline", an `_r` key is "this clip over the baseline";
- `comp_ssim`, the harness's thumbnail SSIM every 6th frame - "is this the
  same picture", which is not in `THRESH` and is reported, not judged.

The copy on the Spark hashes to the same file as the reference tree's
(`sha256sum` both: `cdc1656442e491fb37a95ca08f103ad931580d9525f4ea80fcb87d65d1329cc5`).
Versions are the campaign's own pins: torch 2.14.1+cu130 on the GB10, pyiqa
0.1.16, scikit-image 0.26.0, numpy 2.5.2, av 19.0.2, cv2 5.0.0 (the
`opencv-python-headless==5.0.0.93` wheel), DOVER at `f1ddc96`.
`tools/h3-clip-score-venv.sh` builds that venv.

Post-processing only, on the host: it reads delivered MP4s and touches no
kernel, so it is not part of any render's wall.

## 2. The numbers, pinned

The route pair, both at the delivery canvas, 124 frames at 24 fps
(`outputs/two-stage/one.mp4` against `outputs/two-stage/draft.mp4`):

| metric | one pass | two stage | change | THRESH | flag |
|---|---:|---:|---:|---|---|
| MUSIQ (0-100) | 52.16 | 53.39 | +1.23 | `_d` >= -3.0 | — |
| CLIP-IQA (0-1) | 0.309 | 0.378 | +0.069 | `_d` >= -0.08 | — |
| DOVER (0-1) | 0.667 | 0.661 | -0.006 | `_d` >= -0.08 | — |
| sharp_frame | 49.4 | 24.7 | **x0.500** | 0.8-1.25x | **`sharp_frame_r`** |
| noise (sigma) | 0.546 | 0.437 | x0.800 | <= 1.3x | — |
| flicker | 2.174 | 1.061 | x0.488 | <= 1.25x | — |
| tflick | 0.825 | 0.463 | x0.561 | <= 1.25x | — |
| warp_err | 2.4 | 1.7 | x0.704 | <= 1.25x | — |
| flow_mag | 6.97 | 9.28 | x1.33 | 0.67-1.5x | — |
| accel | 0.176 | 0.195 | x1.11 | <= 1.3x | — |
| zoom (cumulative) | 0.976 | 0.876 | — | needs a scripted camera | not scored |
| comp_ssim | — | **0.147** | | not in THRESH | reported |

And the canvas control this port already had, one pass at the draft canvas
(`draftcanvas.mp4`, 928x544) against the same full-canvas baseline:

| metric | one pass 1344x768 | one pass 928x544 | change | flag |
|---|---:|---:|---:|---|
| MUSIQ | 52.16 | 57.78 | +5.62 | — |
| CLIP-IQA | 0.309 | 0.444 | +0.135 | — |
| sharp_frame | 49.4 | 30.9 | x0.626 | **`sharp_frame_r`** |
| flow_mag | 6.97 | 6.42 | x0.921 | — |
| comp_ssim | — | — (different canvas) | | not comparable |

## 3. What it says, and the trap in it

**No-reference IQA rewards the softer clip.** MUSIQ gains 1.23 and CLIP-IQA
gains 0.069 on the two-stage output, and MUSIQ gains 5.62 and CLIP-IQA 0.135 on
the smaller canvas. Both metrics prefer a smoother image, and a route whose
per-pixel Laplacian energy has halved is exactly that. A gate built on MUSIQ or
CLIP-IQA alone would have passed an arm that loses half the detail: that is why
the campaign's `THRESH` gates sharpness on a ratio (0.8-1.25x) and why this
instrument reports `sharp_frame` beside them.

**Sharpness is the one thing the route loses, and it is not noise.** 49.4 to
24.7 is a factor of 2.0, four times the harness's own acceptance band
(0.8-1.25x). `comp_ssim` 0.147 says the same thing differently: this is not the
one-pass picture at a lower quality, it is another picture - which is what
`two-stage-delivery-2026-11-02.md` section 3 concluded from the stills and the
identical audio hash, and what the flow change (x1.33) and the zoom change
(0.976 to 0.876) show as a different camera move.

**The canvas control suggests the upscale does lose something, but does not
prove it.** The 928x544 one-pass render keeps x0.626 of the full canvas's
sharpness while the route keeps x0.500. Both numbers are measured at native
resolution, so they are not the same measurement: the control tells us the
draft canvas alone is not the whole story, and nothing more. Pricing the
upscale would need a same-canvas control the port does not have yet - the
route's own draft stage, decoded instead of upscaled.

## 4. The instrument's own noise floor

Five runs over the same two files, same binary, same host - the last of them
through the shipped command, from the checkout at `0108d7c4`:

| run | `one.mp4` DOVER | `draft.mp4` DOVER | delta |
|---|---:|---:|---:|
| 1 | 0.683 | 0.662 | -0.021 |
| 2 | 0.670 | — | — |
| 3 | 0.667 | 0.661 | -0.006 |
| 4 | 0.691 | 0.658 | -0.033 |
| 5 | 0.677 | 0.665 | -0.012 |

Every other metric - MUSIQ, CLIP-IQA, sharpness, noise, flicker, tflick,
warping error, flow, acceleration, zoom - reproduced to the printed digit in
all five runs. DOVER moves by up to 0.024 on an unchanged file, and its two
heads show where: the aesthetic head held 0.131 across runs while the
technical head went 0.063 to 0.070. So the campaign's DOVER delta budget
(0.08) is only about 2.4x the largest delta this instrument produced on one
unchanged pair (0.033), and a single DOVER delta sitting near its threshold is
not a verdict.

This is the reason `--no-dover` exists in the driver. Everything else in the
gate is exact, which is what makes it usable on a numerics arm: any change in
the other nine numbers is the arm's, not the instrument's.

## 5. What it does not score

The harness has nine criteria; this driver scores the two that need nothing but
the clip. The others stay out because this port has no inputs for them, not
because they are unimportant:

- **identity, person, face sharpness** need a reference image and InsightFace's
  `buffalo_l` - this port renders text-to-video from a prompt, with no reference
  frame;
- **voice, dialogue, pitch** need a shot script (`shots.json` with the line);
- **lip sync** needs SyncNet and S3FD plus a face;
- **reference fidelity** (criterion 7) needs the reference image.

So the gate judges picture quality and motion, and nothing about who is on
screen or what they say. For a numerics arm that is the right scope: a rounding
change does not cast a different actor.

## 6. Reproducing

The venv (once, ~10-15 min):

```
tools/h3-clip-score-venv.sh ~/ds4-test/venv_eval
```

The harness, copied out of the reference tree (the driver prints its sha256;
it must match `cdc1656442e491fb…`):

```
mkdir -p ~/ds4-test/harness && cp /data/Rapid-H3/harness/eval_clips.py ~/ds4-test/harness/
```

The measurement, and the gate (exit 1 as soon as any non-baseline clip carries
a flag):

```
H3_EVAL_HARNESS=~/ds4-test/harness ~/ds4-test/venv_eval/bin/python tools/h3-clip-score.py \
    --baseline outputs/two-stage/one.mp4 outputs/two-stage/one.mp4 outputs/two-stage/draft.mp4 \
    --out ~/ds4-test/score-route.json
```

The rules that make the comparison meaningful: both sides at the same canvas
(otherwise `comp_ssim` is left empty and the ratios mix resolutions), and one
baseline render per arm, since re-measuring the baseline is how a delta picks
up DOVER's noise.
