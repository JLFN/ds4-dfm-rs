# The E4M3 attention arm: the shape reproduces, and the picture pays for it (2026-11-02)

Unit: handoff section 3, item 1. The previous unit measured an E4M3 K/V
attention block at **421.7 ms** a call against the BF16 block's 502.9, then
withdrew it because that shape's output moved between runs
([`fp8-attention-2026-11-02.md`](fp8-attention-2026-11-02.md) section 4). It was
one line of the preparation pass. Fixed, the shape reproduces - **433.1 ms**
against **514.7** at the delivery sequence, -15.9% - it is deterministic to the
hash, and it takes **24.7 s** off the delivery job's `DENOISE`. It ships
**opt-in** (`H3_DIT_ATTENTION=fp8`) and not as a default, because the clip gate
refuses it: `sharp_frame_r` **0.607** against a 0.8-1.25 band, `musiq_d`
**-7.12** against a -3 budget, detail energy **-23%**. Spark `gx10-e1d2`,
sm_121a, CUDA 13.0, 65 C, no clock touched.

## 1. The defect, and why it looked like arithmetic

The preparation pass writes K's tiles as they are read and V's **transposed**, a
dimension a row: for each `(head, dimension, tile)` one thread writes four
bytes, the four keys of its group. Its rows were `round_up(sequence, 16)` bytes
wide - 37,728 at the benchmark's 37,722 rows, 37,792 at the delivery job's
37,786. The last tile of 37,722 begins at 37,696 and its last group ends at
37,759, so **32 bytes of every dimension's last tile landed in the
next dimension's row**, on keys 28-31 of that row's first tile, which another
thread was writing with real data (at 37,786 it is the same 32 bytes, 37,792 to
37,823). Whichever store retired last won. Nothing
about the preparation's maxima or its scales moved, because the overflow was
past the values they reduce over - which is exactly what the evidence said: K8
byte-identical, V8 differing, differences at tile boundaries, and none of it at
a shape whose key count is a multiple of 64.

The fix is the row width (`native/h3/h3_dit_gpu.cu:4237`): V's rows are every
tile the sequence has, `tiles * 64`, so each tile owns its own bytes and each
byte is written exactly once. 32 bytes a row more, 0.08%. That also retired the
second defect the previous unit named: the ring's copy of a dead tail chunk
pointed at `sequence - 1` to stay in bounds, which is neither the tile's own key
offset nor sixteen-byte aligned, and `cp.async` needs both addresses aligned
whatever its source size. With the rows as wide as the tiles, the tail chunk
copies from its own offset and reads zeros - the values the preparation wrote
there (`:3902`).

The write path is `:3842`; the arm's selector is `:119`, its dispatch `:4312`,
the block `:3909`, the preparation `:3700`.

Two determinism witnesses, at the delivery canvas in both spellings - 37,722 rows
in the benchmark, 37,786 in the job - `tests/h3/attn_bm 37722
56 3` prints `out_fnv=b8d31bd13276ad67` on every process, and the two delivery
renders below produce the same clip md5 (`4124fcf2737c6bd40cd949eebbb6c76a`)
from two separate processes. At engine level the same fact is the gate: the
previous unit's `identical=false` under this arm was this bug seen from
`tests/h3_engine_gate`, whose two renders are the same request through one
resident engine. It now reads `MEDIA identical=true` with the arm set, and
`RESULT PASS residency` - as it does with the arm unset, on the same binary.

## 2. What it buys, and where the price is

`tests/h3/attn_bm`, 56 heads, three calls, one binary, the arm in the
environment and nothing else changed:

| sequence | BF16 block (default) | E4M3 arm | delta |
|---|---:|---:|---:|
| 37,722 (the delivery canvas) | 514.726 ms | **433.128 ms** | **-15.9%** |
| 38,900 | 538.878 ms | **452.616 ms** | **-16.0%** |

The arm's own deviation from the block it replaces, same binary, `attn_bm`'s
dump comparison: at 1,024 rows RMS **7.35e-4**, max 3.91e-3, on 95.3% of
elements; at 37,722 rows RMS **1.21e-4**, max 7.32e-4, on 95.3%. The BF16
tensor-core block itself already deviates from the byte-exact kernel by 4.85e-5
at 1,024 rows, so this is 15x that - three mantissa bits on K, V and P, which is
the dtype and not a bug. (The previous unit's 1.51e-4 at 37,722 was measured on
the corrupt V; the fix lowers it to 1.21e-4.)

The delivery job (`outputs/attn-fp8/`, canvas 1344x768x121, 6 steps, seed 42,
one process an arm, same job line as `outputs/two-stage/one.jobs`):

| arm | `TEXT` | `LOAD` | `DENOISE` | s/step | `VIDEO` | wall | clip md5 |
|---|---:|---:|---:|---:|---:|---:|---|
| BF16 block | 26.909 | 6.609 | 232.951 | 39.11 | 46.618 | 319.334 | `878f0f7d...` |
| E4M3 arm | 27.851 | 6.590 | **208.246** | **34.75** | 46.910 | **295.891** | `4124fcf2...` |

`DENOISE` **-24.705 s (-10.6%)**, step **-4.36 s (-11.2%)**, wall -23.4 s
(-7.3%) with `LOAD` warm and equal in both rows. The render's own sequence is
**37,786** rows, not the benchmark's 37,722 (the job's sequence carries 64 rows
the bench's shape does not), and the arm's line in the log says so: fifty calls
of the bench's 81.6 ms is 4.04 s of the step's 4.36, and the 76-row connector
attention - 128-wide, also on the arm - plus the step's own 0.4 s spread are the
rest, so the bench and the step agree to about 7%. The BF16 row
is the arm's own baseline and is **byte-identical to the route unit's clip**
(`878f0f7d8d8e28dd7b00029623d6b2b8`, `outputs/two-stage/one.mp4`) and carries
its content hashes (`video_fnv=1e11a80f73242ad6`), which is the acceptance's
"a run with the arm unset reproduces the baseline": one binary, one job line, one
variable.

## 3. The quality half, which refuses it

Decoded yuv420p, all 124 frames, the arm against its own baseline
(`~/ds4-test/metrics-pair.sh`):

| instrument | BF16 block | E4M3 arm | |
|---|---:|---:|---|
| PSNR y / u / v | - | 17.219 / 30.810 / 32.346 dB | average 18.899 (min 18.213, max 19.695) |
| SSIM Y / All | - | 0.6465 / 0.7367 | |
| detail sigma 4 | 4.6499 | **3.5560** | **-23.5%** |
| detail sigma 8 | 8.0901 | **6.2897** | **-22.2%** |

And the clip gate on the same pair (`tools/h3-clip-score.py --baseline base.mp4
base.mp4 fp8.mp4`, exit 1):

| metric | baseline | arm | delta / ratio | band | |
|---|---:|---:|---:|---|---|
| `sharp_frame_r` | 49.4 | 30.0 | **x0.607** | 0.8-1.25 | **FLAG** |
| `musiq_d` | 52.16 | 45.04 | **-7.12** | >= -3 | **FLAG** |
| `clipiqa_d` | 0.309 | 0.278 | -0.031 | >= -0.08 | pass |
| `dover_d` | 0.675 | 0.701 | +0.026 | >= -0.08 | pass (never alone) |
| `noise_r` | 0.546 | 0.504 | x0.923 | <= 1.3 | pass |
| `tflick_r` | 0.825 | 0.672 | x0.815 | <= 1.25 | pass |
| `flicker` | 2.174 | not produced | - | <= 1.25x | not judged |
| `warp_err_r` | 2.407 | 2.456 | x1.02 | <= 1.25 | pass |
| `flow_mag_r` | 6.972 | 7.227 | x1.037 | 0.67-1.5 | pass |
| `accel_r` | 0.176 | 0.212 | x1.205 | <= 1.3 | pass |
| `zoom` | 0.976 | 1.027 | - | - | - |
| `comp_ssim` | - | 0.543 | - | reported | same picture |

This is the opposite reading to the route pair, where the IQA metrics *rose* on
the softer clip: here MUSIQ falls 7 points, which is the honest direction, and
it agrees with the instruments that do not reward smoothness - detail energy
down 23%, Laplacian sharpness at 0.607x, luma PSNR 17.2 dB against the FP8
GEMM arm's 22.1. `comp_ssim` 0.543 is the other half: the two clips are the
same shot with the same composition (`0.147` was the route pair's *different
shot*), so this is not "two valid answers" as the projection arm's record put
it - it is one answer with a fifth of its detail energy gone. Chroma holds
(30.8 / 32.3 dB, SSIM U 0.905, V 0.929), so the loss is luma structure.

Why attention loses more than the GEMM arm: P is E4M3 - three mantissa bits on
the softmax's weights, which is the term that carries the picture, where the
projections quantize a matrix product whose result is a sum over 5,376 terms.
The 15x-with-the-bench deviation says the same thing numerically.

## 4. What it is worth, and where

Not the delivery: a fifth of the detail energy for 10.6% of the denoise is a
trade this port has refused before (the width unit's criterion was bit-identity,
this one's is the gate). For scale, the two-stage route's draft clip - a
supported, opt-in mode - measures detail energy **2.5913** against the same
baseline's 4.6499
([`two-stage-delivery-2026-11-02.md`](two-stage-delivery-2026-11-02.md)), a
larger loss than this arm's 3.5560; what the route has and the arm does not is a
refine step that puts some of it back, and the route's ceiling is now that
refine. So the **draft leg** is where the arm could earn its keep - it is
upscaled and refined one step later. That is a measurement for the route unit
and not a claim here: the route's pair was rendered on the BF16 arm.

It ships as an arm because the operator's rule makes every FP8 path opt-in and
nothing about the default changes: unset, the tree is the block it was, clip md5
and all. `H3_DIT_ATTENTION` now reads `scalar`, `tensor` (default), `four`,
`pipe12`, `pipe16`, `fp8` or `e4m3`; `fp8` leaves the width at the eight-warp
default so a refused shared-memory admission falls back to the measured block
rather than to the older synchronous one, and a run that takes the arm prints one
line per distinct shape it runs (`E4M3 attention (mma 8 warps, prepared
operands): sequence=37722 heads=56`) so a log names what produced its clip.

## 5. Reproducing

```
make -j8 CUDA_ARCH=sm_121 tests/h3/attn_bm tests/h3_engine_gate h3-rs
H3_DIT_ATTENTION=fp8 tests/h3/attn_bm 37722 56 3      # 433.128 ms, b8d31bd13276ad67
tests/h3/attn_bm 37722 56 3                            # 514.726 ms, 80652b45a36d0166
H3_MODEL=/mnt/h3-artifacts/MiniMax-H3-FastH3-4step ./tests/h3_engine_gate
H3_MODEL=/mnt/h3-artifacts/MiniMax-H3-FastH3-4step H3_DIT_ATTENTION=fp8 ./tests/h3_engine_gate
```

The pair is `~/ds4-test/attn-fp8-render.sh` (both legs, one process each, the job
line of `outputs/two-stage/one.jobs`), the table `~/ds4-test/metrics-pair.sh`,
and the gate:

```
H3_EVAL_HARNESS=~/ds4-test/harness ~/ds4-test/venv_eval/bin/python tools/h3-clip-score.py \
    --baseline outputs/attn-fp8/base.mp4 outputs/attn-fp8/base.mp4 outputs/attn-fp8/fp8.mp4 \
    --out ~/ds4-test/score-fp8-attn.json
```

Both renders were run twice, from two processes and two binaries, and both md5s
reproduced: `878f0f7d…` for the baseline (which is `one.mp4`'s) and
`4124fcf2…` for the arm.
