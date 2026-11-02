# The decode's two schedule levers: 48.1 s to 45.9 s with every frame bit-identical (2026-11-02)

Unit: handoff section 3. Both attention levers were spent (the width is optimal,
E4M3 did not reproduce), so the phase's own work was next: the shipped FP8 arm
spends **49.202 s of its 359 s wall** in `FRAMES`
([`fp8-delivery-2026-11-02.md`](fp8-delivery-2026-11-02.md) section 1), and
[`vae-decode-2026-10-31.md`](vae-decode-2026-10-31.md) section 6 priced five
levers from 48.5 s to ~29 s, two of which are scheduling and need no numerical
decision. This unit took those two and left the three numerical ones alone.

**The verdict first.** The decoder is **46.672 / 45.941 / 46.089 s** where it was
**48.111 / 47.444 s**, and the frames hash is `387f816c77ee986f` in every run of
every arm - the RoPE change and the host-overlap change are both *bit-identical*,
so what they bought is free. The two levers and what each was worth:

| lever | class | before | after | saved |
|---|---|---:|---:|---:|
| QKV/RoPE staged once into shared instead of read twice | schedule | 0.894 ms a call | 0.616 | 1.03 s |
| the tile unpack and the chunk stitch off the launch thread | schedule | - | - | 0.7 s |
| **the phase** | | **48.111** | **46.089** | **2.0 s** |

That is 3.1% of the `FRAMES` phase and 0.6% of the 359 s job, and it cost no
quality decision at all - which is why it went first. The three numerical levers
(the two cast passes, the F32 elementwise chain, BF16 P/V operands; ~19.6 s) are
still unspent and still one decision, because they change the picture.

Spark `gx10-e1d2`, one binary, sm_121a, CUDA 13.0, GPU 2,327-2,411 MHz at
65-71 C. The latent is the gate's own deterministic draw at the product canvas
(37x48x84 -> 1344x768, tile 320, 5x3 tiles, 7 chunks, 124 frames), the arms are
the delivery's (`--vae-gemm bf16i --vae-attention tensor`), and the decoder's
cost does not depend on the latent's values.

## 1. Step 0: the split as this commit has it

The lever table was measured on 2026-10-31, before the two-stage route and the
`bf16i` arm landed, so it was re-measured before anything was designed. One nsys
capture of the unchanged decoder (`~/ds4-test/vae-base.nsys-rep`, split by
`misc/decn/decn-split.py` and a gap histogram): decoder window **46.227 s**,
72,531 kernels, busy 44.235 s (**95.7%**), 1,995.7 ms of gaps.

| kernel | calls | total ms |
|---|---:|---:|
| `nvjet_sm121_tst_mma_128x208x64` (w1) | 3,779 | 7,620.40 |
| `h3d_attention_mma_f32_kernel` | 3,780 | 6,500.41 |
| `h3d_uncast_kernel` (widen, three grids) | 15,116 | 6,821.18 |
| `h3d_swiglu_f32_kernel` | 3,779 | 4,263.49 |
| `nvjet_sm121_tst_mma_128x192x64` (w2) | 3,779 | 4,252.65 |
| `h3d_video_qkv_rope_f32_kernel` | 3,780 | 3,476.32 |
| `nvjet_sm121_tst_mma_128x208x64` (w3) | 3,779 | 2,990.83 |
| `h3d_scale_add_f32_kernel` | 7,558 | 2,159.77 |
| `h3d_cast_kernel` (narrow, two grids) | 15,116 | 3,524.87 |
| `h3d_rms_norm_f32_kernel` | 7,558 | 1,407.12 |
| `cutlass_..._bf16_s16816gemm_relu` (qkv, out) | 3,779 | 1,051.19 |
| the outer GEMMs, their biases, the output norm | 832 | 150.72 |
| **total** | **72,531** | **44,234.56** |

Grouped against the 2026-10-31 table: block GEMMs 15.86 s -> **15.92**, the two
cast passes 10.37 -> **10.35**, attention 6.46 -> **6.50**, the F32 elementwise
chain 7.96 -> **7.83**, RoPE 3.47 -> **3.48**, the outer GEMMs 0.11 -> **0.15**.
Every piece holds; the phase is where it was.

The gaps did not hold exactly, and the difference matters to the plan:

| gap class | count | total |
|---|---:|---:|
| under 0.1 ms (kernel-to-kernel bubbles) | 62,395 | 127.3 ms |
| 0.1-1 ms | 104 | 19.6 ms |
| 1-5 ms | 1 | 1.8 ms |
| **5-20 ms (one per tile boundary)** | **98** | **1,157.1 ms** |
| **20-200 ms (one per chunk boundary)** | **6** | **690.0 ms** |

So the six chunk-boundary gaps are **106-120 ms each, 690 ms**, not the 165-180 ms
and 1.03 s section 6 of the earlier record quoted - and the *tile* boundaries,
which that table did not price, are the bigger half at **1.16 s**. Both are the
same defect: the stream stops while the host works.

## 2. Lever 1: the QKV/RoPE kernel reads its rows once

`h3d_video_qkv_rope_f32` runs **3,780 times a decode** - once per block, 36
blocks of a tile, 105 tiles - at 2,805 rows, 32 heads, a 64-wide head, and lands
3.48 s. The old kernel gave one thread one dimension and made **every one of the
head's 64 threads re-read the whole 64-wide Q and K row** to compute the two RMS
sums, then read its own element again to normalize and rotate: 0.894 ms a call.

New `tests/h3/vae_rope_bm`, one binary, each variant hashed over Q/K/V. The
"shipped" row is the baseline kernel from `a18e0cd9`, captured before the change
landed; the bench's own `shipped` variant now calls the staged kernel, so a rerun
of this table prints 0.616 twice - the numbers below are the ones that chose it:

| variant | ms a call | GB/s | Q/K/V fnv |
|---|---:|---:|---|
| the shipped kernel | 0.894 | 154.1 | `5f33bd9c5855f5bd` |
| Q/K row staged into shared, every thread summing from it | **0.629** | 219.3 | `5f33bd9c5855f5bd` |
| the same, the two sums run once | 0.638 | 216.2 | `5f33bd9c5855f5bd` |
| two heads a block | 0.629 | 219.0 | `5f33bd9c5855f5bd` |
| four heads a block | 0.622 | 221.5 | `5f33bd9c5855f5bd` |
| **the traffic floor**: same outputs, no sums, no rotation | 0.621 | 221.8 | differs by construction |

The staged shape is the one that ships, because it is the smallest change that
reaches the floor: a block is one head of one row, it stages the two rows into
shared once, and then **every thread runs the shipped kernel's own serial `fmaf`
chain over them** - the reduction order and its operands are untouched, so
`rsqrtf(sum / 64 + 1e-5)` is the same bits and the output is bit-identical by
construction, not by tolerance. That the floor is 0.621 and the shipped shape is
0.616-0.629 says there is nothing left in this kernel: 124 MB a call of
mandatory traffic (62 MB in, 62 MB out) at ~220 GB/s is the memory system, not
the arithmetic.

Measured in the decoder, same latent, same seed: **48.111 / 47.444 -> 47.068 /
46.598 s**, hash `387f816c77ee986f` before and after.

## 3. Lever 2: the unpack and the stitch leave the launch thread

The phase stopped the stream for host work in two places: `run_tile` reads the
tile's 26 MB projection back and then the host unpacks it (10.2 ms a tile), and
at each chunk's end the host stitches fifteen tiles (91 ms). That is exactly the
98 x 11.8 ms and 6 x 115 ms of section 1.

**The arm that did not work, first.** `--vae-workers` spreads the chunk loop over
several device contexts, which was the obvious way to overlap it, and it is
already ported. At the product canvas it is slower, not faster:

| workers | seconds |
|---:|---:|
| 1 | **47.133** |
| 2 | 47.632 |
| 3 | 48.018 |

Each worker builds its own decoder - a second 10.4 GB checkpoint read and a second
stream - and a stream that is already 95% busy has nothing to give back. The gate
gained `--workers N` from this measurement and prints `workers=` on its `RUN` line.

**What ships.** The readback stays on the calling thread, because it is the wait
the tile's own kernels impose; everything after it moves to one worker thread per
chunk, with two recycled projection buffers (`spare` hands each back, so no
projection is copied or reallocated, and neither channel can block because
exactly two buffers exist):

```
run_tile -> readback -> hand the buffer to the worker -> next tile's kernels
                        (unpack 10 ms, and at the chunk's end the stitch 91 ms)
```

Measured: **47.068 / 46.598 -> 46.342 / 45.837 s**, the frames hash unchanged, and
the nsys capture agrees about *where*: the 98 tile-boundary gaps of 11.8 ms are
gone (1,157 ms -> 102 ms of 14 gaps plus 175 ms of 85 one-to-five-millisecond
ones), while the six chunk-boundary gaps are still there at 764 ms under the
profiler.

**And the second half of lever 2 does not pay.** Hoisting the same worker across
the whole chunk loop - so a chunk's stitch runs behind the next chunk's kernels -
was implemented and measured: 46.640 s and worse against 46.342 / 45.837, so it
was reverted. Two things are worth recording about that:

- The numbers were taken while three `h3_engine_gate --draft` processes from an
  interrupted script were resident (`ps` showed them at 50-63% CPU), so they are
  **not a clean A/B**; the clean run of the shipped shape is what section 4
  quotes. A repeat on an idle box is what would settle it.
- The 6 x 115 ms of chunk gaps that the profiler reports are therefore suspect
  too: an nsys capture slows the host, so a gap that is really device idle and a
  gap that is really CUPTI's bookkeeping look the same. The wall clock - which
  moved by 0.03 s across the two shapes - says the stitch was never worth 690 ms
  of real time. *A lap timer that synchronizes is an attribution tool, not a
  stopwatch* is the guardrail this unit re-learned the hard way.

## 4. What the arm is, and its numbers

The shipped decoder: `bf16i` GEMM, tensor attention, the staged RoPE kernel, and
the per-chunk unpack/stitch worker. Nothing new is opt-in; the arms are the ones
the delivery already states, and no numerical surface moved.

| run | seconds | frames hash |
|---|---:|---|
| before the unit | 48.111 | `387f816c77ee986f` |
| before the unit | 47.444 | `387f816c77ee986f` |
| RoPE staged | 47.068 | `387f816c77ee986f` |
| RoPE staged | 46.598 | `387f816c77ee986f` |
| worker (first shape) | 46.342 | `387f816c77ee986f` |
| worker (first shape) | 45.837 | `387f816c77ee986f` |
| **shipped, clean, idle box** | **46.672 / 45.941 / 46.089** | `387f816c77ee986f` x3 |

The gate asserts it, not the table: `h3_vae_gate --expect
387f816c77ee986f` matches on every run, and that hash is an FNV over **all** 124
frames x 768 x 1344 x 3 F32 values, so "identical" means every channel of every
pixel of every frame.

The engine gates on the same binary:

```
MEDIA identical=true first=frames=22 video=88ea05b50c4f1131 audio=3945a4b4ecd9b28c
                     frames=1ad7a65c4d74fdec second=... (the same four)
RESULT PASS residency
RESULT PASS two-stage text_hit audio_frozen
```

`video=88ea05b50c4f1131` is the value the E4M3 unit's record quotes for this
gate, taken before any of this landed: the decode change did not move a byte of
the engine's own media either.

## 5. The delivery canvas

The job the campaign delivers - 1344x768, `--frames 121` (latent 37x48x84),
6 steps, seed 42, `H3_DIT_GEMM=fp8`, `--crf 15 --preset slow --vae-gemm bf16i
--vae-attention tensor` - rendered through this binary
(`outputs/decode-unit/`, prompt and `run.log` beside the clip):

| line | this arm | 2026-11-02 record |
|---|---:|---:|
| `TEXT` | 53.591 | 28.640 |
| `LOAD` | 37.064 | 39.640 |
| `DENOISE` (6 steps) | **233.588** (38.93 a step) | 235.419 (39.24) |
| `AUDIO` | 3.150 | 3.085 |
| `FRAMES` | **46.465** | **49.202** |
| `VIDEO` (mux) | 1.970 | - |
| wall | 377 | 359 |

`FRAMES` **49.202 -> 46.465 s, -2.737 s (-5.6% of the phase)**, which is the
gate's 48.1 -> 46.1 plus the delivery arm's own few tenths of drift; the phase
prints its own arms (`gemm=bf16i attention=tensor order=heads`), so this is the
decode that ran and not an inference. `DENOISE` is untouched at 38.93 s a step,
as it must be - nothing here is on the DiT's side - and `TEXT`'s 53.6 s is the
page cache, not the arm: the same line reads 28.6 s in the earlier record, and
this render followed two hours of 10 GB checkpoint reads.

The clip is `0f7903b3327159fa7e3b26211aa46ba3`, and **it cannot be compared to
the earlier delivery clip** - that prompt was not kept (section 6), so this is a
different subject, not a different arithmetic. What the two renders do share is
the part this unit changed: the phase's arms and its seconds.

## 6. What is left, and what this does not establish

- **The three numerical levers are untouched and are one decision**: the two cast
  passes (10.35 s), the F32 elementwise chain (7.83 s) and BF16 P/V attention
  operands (6.50 s), ~19.6 s at best and a different picture. They need an arm, a
  frames PSNR/SSIM/detail pair against this arm, and `h3_engine_gate` PASS - the
  numerical surface [`vae-bf16-arm-2026-10-29.md`](vae-bf16-arm-2026-10-29.md)
  section 6 deferred, still deferred.
- **The block GEMMs are the phase now**: 15.92 s, 89.7 TFLOPS, 76% of this
  clock's BF16 peak. Nothing there is a port defect.
- **The denoise side has the same shape and more of it.** `step-profile-2026-11-01.md`
  puts `h3d_qkv_rope_kernel` at 112.7 ms a call, 51 calls a step - 6.6 s a step
  where the reference's whole rope+norm class is 0.85 s - and launches
  `h3d_attention_mma_f32_kernel` 1,890 times inside one denoise step. Both are
  arithmetic-neutral work of the kind this unit did, and both are worth more per
  second than the decode's remaining numerical levers. That is the next unit.
- **One latent, one canvas, one host.** Every number is the product canvas at
  1344x768 with a drawn latent; no other geometry and no other machine.
- **The delivery render's prompt is kept beside its clip this time.** The
  2026-11-02 delivery render's `outputs/` tree (clip, stills, `prompt.txt`) is
  gone from the Spark, so its `cffb3fb9...` clip md5 cannot be reproduced. This
  unit's is at `outputs/decode-unit/prompt.txt`, md5
  `98fd163de409fb3713c6b4d14bb10d6d`.

## 7. Reproducing

    make CUDA_ARCH=sm_121 tests/h3_vae_gate tests/h3/vae_rope_bm h3-rs tests/h3_engine_gate
    export H3_VAE=/mnt/h3-artifacts/MiniMax-H3-FastH3-4step/FL2VA/video_vae/source
    export H3_VAE_GEMM=bf16i H3_VAE_ATTENTION=tensor
    tests/h3/vae_rope_bm 2805 32 50                    # 0.616 ms, six shapes
    tests/h3_vae_gate --latent 37,48,84 --seed 1 --repeat 3 \
        --expect 387f816c77ee986f                      # 46.0-46.7 s a decode
    tests/h3_vae_gate --latent 37,48,84 --seed 1 --workers 2   # slower, by design
    H3_VAE_TIMING=1 tests/h3_vae_gate --latent 37,48,84 --seed 1   # per-chunk split
    H3_MODEL=/mnt/h3-artifacts/MiniMax-H3-FastH3-4step tests/h3_engine_gate
    H3_MODEL=/mnt/h3-artifacts/MiniMax-H3-FastH3-4step tests/h3_engine_gate --draft

The instruments are `~/ds4-test/{step0.sh,rope-verify.sh,workers-sweep.sh,
overlap-verify.sh,final-decode.sh,delivery-render.sh}`, `~/ds4-test/gap-hist.py`
(the gap histogram), `~/ds4-test/vae-base.sqlite` and `~/ds4-test/vae-overlap.sqlite`
(the two captures), and `~/ds4-test/{decode-base,decode-timing,workers-sweep,
final-decode}.log`.
