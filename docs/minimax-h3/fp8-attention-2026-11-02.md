# E4M3 attention: the pipe halves, and the cast eats it (2026-11-02)

Unit: handoff section 3, steps 0-2. The width unit before it closed every shape
the block can take - twelve warps +1.0% over eight, sixteen -30% on spills - and
concluded the critical path is the per-row MMA and softmax, which only a
narrower operand shrinks. This unit takes the narrow operand, measures what it
buys and what it costs, and **does not ship it**: the shape that was fast was not
reproducible, and the shape that was reproducible was slower than the BF16 block
it replaced. Nothing from the kernel landed; the tree at `b648f695` is what the
next unit starts from.

**The verdict first, in four numbers.** The pipe halves exactly as predicted
(BF16 123.1 TFLOP/s, E4M3 246.2, measured). The E4M3 arithmetic alone, with the
ring never refilled, is **329.8 ms** a call at S = 37,722 against the BF16
block's **502.9**. Converting K and V on the way into the ring - the handoff's
step 0 - costs **538 ms** on top of that, so the first complete shape ran at
**1,043.8 ms** and, with the tile's reads hoisted over the arithmetic,
**868.0 ms**: both slower than the block they were meant to replace. Preparing
the operands once per call instead got the whole arm to **421.7 ms** - and that
shape produced a different answer on every run, which is where the unit ends.
Spark `gx10-e1d2`, sm_121a, CUDA 13.0.

## 1. What the pipe will do, before writing a kernel

`~/ds4-test/fp8mma.cu`, eight independent accumulator chains a warp, no memory
in the loop, 20,000 iterations, four blocks an SM at 256 threads:

| shape | FLOP an instruction | 8 warps/SM | 16 warps/SM | 4 warps/SM |
|---|---:|---:|---:|---:|
| BF16 `m16n8k16` | 4,096 | **123.1 TF** | 123.1 | 61.5 |
| E4M3 `m16n8k32` | 8,192 | **246.2 TF** | 245.6 | 123.1 |

Two facts govern the unit. E4M3 is exactly twice the arithmetic per instruction
and issues at the same rate, so a block that was MMA-bound would halve. And one
CTA of eight warps already reaches the same 123 TF as four CTAs, which closes
the second-CTA question the width unit left open: the ring can shrink all it
likes and it will not buy a second resident block. The BF16 attention block
measures 80.4 TF of that 123 - 65% - so its headroom is the 35% the softmax and
the fragments take, and only part of the pipe time is available to a narrower
operand.

## 2. The kernel, and the three arithmetic choices it made

Eight warps, 128 query rows, a three-stage ring of **68,608 B** (9,216 B of K a
stage, 10,240 B of V, 1,280 B of P a warp), 209 registers, no spills, one
barrier. Fragments came from `ldmatrix` in all three places: the E4M3 fragment
and the BF16 one want the same 16-byte rows in the same order, four bytes an
element instead of two.

The named deviations, all arithmetic:

- **K and V carry a scale per 64-key tile, Q one per warp.** The score is
  multiplied by the product; the output accumulator carries the V scale as one
  more factor on the rescale it already does each tile
  (`rescale = expf(max_old - max_new) * v_prev / v_tile`), and the last tile's
  scale divides out in the epilogue. That factor has to ride the accumulator: a
  per-tile V scale cannot be folded into a sum that outlives its tile.
- **P is E4M3 against a fixed 448.** The online softmax's weights are already
  `<= 1`, so 448 puts them at the top of the format exactly and costs no
  reduction; the output divides it back out.
- **Scores, the running maximum, the denominator and the accumulator stay
  FP32**, as in the BF16 block.

V was staged **transposed** - a dimension a row, 64 keys a row, padded to 80 -
because the PV fragment wants four keys of one dimension in a register while
V's own rows are four dimensions of one key, and `ldmatrix.trans` has no
byte-granular form. P went through the warp's own 16x80 tile for the same reason
the other way round: the score fragments hold two adjacent keys a lane, the PV
fragments want four, and one shared-memory round trip is cheaper than the
four-lane butterfly that would replace it.

Numerically it was right. Against the BF16 arm of the same binary:

| shape | pair | RMS | max abs | differing |
|---|---|---:|---:|---:|
| S = 1,024 | E4M3 against BF16 tensor-core | 7.35e-4 | 3.91e-3 | 95.32% |
| S = 1,024 | BF16 tensor-core against BF16 scalar | **4.85e-5** | 4.88e-4 | 41.89% |
| S = 37,722 | E4M3 against BF16 tensor-core | **1.51e-4** | 9.61e-4 | 96.23% |

The second row is what the tensor-core block itself already deviates from the
byte-exact reference by, so E4M3 attention is **15.1x** that at 1,024 rows -
three mantissa bits on P, on K and on V, as expected.

## 3. Where the cast has to live, and what it costs there

The handoff's step 0 said to convert on the way into the ring, because the
staging path already writes the tile and the cast would ride a loop that was
going to run anyway. It does not: the BF16 block stages with `cp.async`, which
copies bytes and casts nothing, so a converting staging pass replaces eight
asynchronous copies with register reads whose latency nothing hides. Four
shapes were built and measured at S = 37,722, one binary, three calls:

| shape | ms/call | what it does |
|---|---:|---|
| cast in the staging path | 1,043.775 | the tile's maximum, then the cast, both in front of the arithmetic |
| + the tile's reads hoisted over the arithmetic | 868.041 | the same work, ordered so the latency lands inside the tile |
| + staging neutered, arithmetic only | **329.759** | the ring never refilled; the arithmetic unchanged |
| operands prepared once a call | **421.742** | the shape that was fast, and not reproducible |

Hoisting the reads bought 176 ms of the 538. The rest is the cast itself:
8,192 values a tile turned into E4M3 a thread at a time, against a tile whose
arithmetic is 1,480 cycles of tensor pipe. **A conversion inside this loop costs
more than the MMA halving saves at this shape** - which is the finding that
decides the unit, because the alternative is a pass outside it.

## 4. The preparation pass, and the bug that ended the unit

`h3d_f8_prepare_kernel`, one block a (head, 64-key tile): K's tile maximum and
V's, one barrier, the two scales, then K written as read and V written
transposed a dimension at a time (sixteen threads carry one dimension's
sixty-four keys, so a warp writes two contiguous runs and nothing scatters).
V's prepared rows are padded to a sixteen-byte multiple so that the ring's
`cp.async` of a row is aligned. Scratch: `elements + heads*head_dim*stride + two
scale arrays`, grown on the context and freed with it. About 2.2 GB of traffic,
9 ms at this host's rate, inside the 421.7.

It is not reproducible, and the unit stops there. The evidence, in the order it
was gathered:

- `tests/h3_engine_gate` with `H3_DIT_ATTENTION=fp8` fails `identical=false`,
  while the same binary with the arm unset passes with the recorded hashes
  (`88ea05b50c4f1131`, `RESULT PASS residency`).
- `tests/h3/attn_bm` gives a **different `out_fnv` on every process** at any
  shape whose key count is not a multiple of 64 - 1234 rows, 37,722 rows - and
  the same hash every time at 1,024, which is 16 exact tiles. The BF16 and
  scalar arms are stable at 1234 rows and at every other shape tried.
- A zeroed scratch (`cudaMemsetAsync` over the whole buffer, every call) does
  not change it, so it is not uninitialised global memory.
- Replacing `cp.async` with plain shared-memory stores does not change it, so it
  is not the asynchronous copies or their commit/wait accounting.
- Dumping the prepared buffer from two runs isolates it: **K8 is byte-identical
  and V8 differs**, and in the two-head reproduction the scales are identical
  too, so the difference is in V's transposed write and not in either reduction.
- A tiny shape (128 rows, 2 heads) is identical across runs; 1234 rows with the
  same 2 heads is not. So it is the tile count, not the head count, and the
  differences sit at tile boundaries.

What that pattern says - V's prepared bytes moving while both maxima hold, at
tile granularity - is a write-path defect in the preparation kernel, not an
arithmetic one: the same attention block, fed the same prepared bytes, is
deterministic (the S = 1,024 runs agree). Finding it is the first task of the
unit that resumes this, because the 421.7 ms and the 16.1% are real and worth
having. The buffer also needs the two smaller things this session found: the
dead-tail copies of V point at `sequence - 1`, which is not sixteen-byte
aligned, and the preparation writes only whole words, so V8's bytes past
`sequence` are left as whatever the buffer held.

## 5. What it would be worth if it reproduced

The arm is a 16.1% cut of the attention block, and attention is 42% of the
delivery clip's wall, so the clip would save about **24 s of 359** - against
`FRAMES`' 48.3 s, which needs no numerics decision at all, and against the FP8
**GEMM** arm's 359 s that already shipped opt-in for a picture measured at PSNR
y 22.1, SSIM Y 0.717. That is the honest ranking: this lever is the smaller of
the two still open and it costs 15x the BF16 block's arithmetic error to take.
It would have shipped as an arm and not a default for exactly that reason.

So the ranking stands with the unit closed unsolved: **`FRAMES` is the next
lever**, and the FP8 attention route is a second attempt behind it, with the
prepared-operand shape and the write-path defect named above as where it starts.

## 6. Reproducing

    make CUDA_ARCH=sm_121 tests/h3/attn_bm
    H3_DIT_ATTENTION=tensor H3_ATTN_DUMP=~/ds4-test/t1234.bin \
        tests/h3/attn_bm 1234 56 2
    H3_DIT_ATTENTION=scalar tests/h3/attn_bm 1234 56 2 ~/ds4-test/t1234.bin

The probe of section 1 is `~/ds4-test/fp8mma.cu` on the Spark:
`nvcc -O3 -arch=sm_121a -o fp8mma fp8mma.cu && ./fp8mma 20000 48`. The four
shapes of section 3 and the instrumented dumps of section 4 were built from
copies of `h3_dit_gpu.cu` in `~/ds4-test/` (`f8probe.cu` + `probe.cu`, with
`H3_F8_DUMP` writing the prepared buffer); none of them are arms in the tree,
because none of them earned a place there.
