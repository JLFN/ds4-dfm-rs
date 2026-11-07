# The projections' true shapes, the gap that was bookkeeping, and what FP8 is worth

The previous record (`fp8-projections-2026-10-29.md`) priced the DiT's four
projections and then reported a contradiction: the same GEMMs measured 96.6
TFLOPS alone and 68.6 TFLOPS inside the model. The contradiction was mostly an
error in the bench, and finding it changes the campaign's biggest number. This
record fixes the shape, retires the two hypotheses the earlier record raised
(alignment and kernel choice), quantifies the residual, and re-prices FP8.

## 1. The bench priced fc1 at half its width

`tests/h3/linear_fp8_bm` timed the four projection shapes with
`fc1`'s N set to `ffn` = 14,336. The model's fc1 is the **fused gate and up
pair**: its N is `2 * ffn` = 28,672. Four independent sources:

- the reference's own loader, `h3_dit.c:501-525`: `mlp.fc1.weight` is
  `FFN*2 x HIDDEN`;
- the port's layout validation, `crates/ds4-core/src/minimax_h3/layout.rs:563`,
  which checks the checkpoint's tensor against `&[ffn * 2, hidden]`;
- `dit.rs` `run_block`'s call, `output_dim = self.config.ffn as u32 * 2`, whose
  output then feeds `swiglu`, which halves it back to `ffn`;
- and the model's own trace (below), where the first kernel's grid.y is 28 for
  that shape against 14 for the bench's `ffn`-wide one.

So the bench's `fc1` row, and with it the "50 layers" total, understated the
real per-evaluation projection work by 144.9 TFLOP (2.898 vs 5.795 TFLOP per
layer). The bench is corrected in this commit; the table below is the re-run.

## 2. What the model actually runs, shape by shape

The trace is `~/lev1-staged-4` on the Spark (`--denoise --steps 4`, four
evaluations, 18,799 rows, the staged elementwise pair), re-read per call rather
than summed per kernel name. Grid dimensions separate the shapes without
guessing: 200 calls of each (4 evaluations x 50 layers), three grids on the
CUTLASS kernel and one on NVJet's.

| projection | K | N | grid | calls | med ms/call | TFLOP/call | TFLOPS | ms/eval |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| fc1 (fused gate+up) | 5,376 | 28,672 | (592, 28, 1) | 200 | 60.768 | 5.7955 | 95.4 | 3,040.0 |
| qkv | 5,376 | 21,504 | (592, 21, 1) | 200 | 45.323 | 4.3467 | 95.9 | 2,264.2 |
| fc2 | 14,336 | 5,376 | (592, 6, 1) | 200 | 43.261 | 2.8977 | 67.0 | 2,145.3 |
| out (NVJet) | 7,168 | 5,376 | (632, 6, 1) | 200 | 16.491 | 1.4498 | 87.9 | 820.6 |
| **the four** | | | | **800** | | **14.4897** | **87.6** | **8,270.1** |

Cross-checked against the per-evaluation span instead of the trace-wide sums:
the three steady-state evaluation windows hold 11.283, 11.277 and 11.341 s of
kernels, of which the four projections are 8,257 ms (cutlass 7,438 + NVJet
819). Both readings agree to 0.2%.

**The projection budget was 724.5 TFLOP per evaluation at 87.6 TFLOPS**, not
579.5 at 73.3. That is 73% of the 120.4 TFLOPS `mma.sync` peak implied by the
2.45 GHz this host runs at (81% of the 108 TFLOPS figure quoted at 2.19 GHz),
and 73% of the evaluation's kernel time rather than the 64% the old FLOP count
implied.

Two further checks on the same trace:

- **Nothing overlaps and nothing is inflated.** Per evaluation, the sum of all
  kernel durations is 0.995-0.998 of the span they occupy: one stream, no PDL,
  ~2 us of gap per kernel. The per-kernel durations are therefore real GPU
  time, and the earlier worry that CUPTI might be reporting a wait is closed.
- **The 150/50 call split is a shape split.** 3 shapes x 50 layers on CUTLASS,
  one on NVJet - which is why the attribution record could not say which
  projection was which.

## 3. The same kernel, the same grid, in both places

Re-running the corrected bench under `nsys` and reading its grids identifies
each shape the same way, and the identification is the point:

| shape | bench grid | model grid | bench med ms/call | model med ms/call | ratio |
|---|---|---|---:|---:|---:|
| qkv | (592, 21, 1) | (592, 21, 1) | 41.172 | 45.323 | 1.101 |
| fc1 (N = 14,336 in the bench) | (592, 14, 1) | (592, 28, 1) | 27.526 | 60.768 | 2.208 |
| fc2 | (592, 6, 1) | (592, 6, 1) | 36.589 | 43.261 | 1.182 |
| out (NVJet) | (632, 6, 1) | (632, 6, 1) | 14.720 | 16.491 | 1.120 |

Both columns are nsys per-call medians from the same kind of capture. fc1's
2.208 is the shape gap (2.000, the doubled N) times the same 1.10 penalty the
other three show, which is the corroboration that the bench's fc1 was
half-width. Per-evaluation, the bench's medians give 7,377 ms against the
model's 8,270: **1.121, the model at 89% of the standalone rate.**

Same kernel family, same grid, same block for every shape that both ran. The
model is not falling back to a WMMA path and cuBLASLt is not choosing
differently, so **the two hypotheses the previous record raised are retired**:
an `align8` tensor-op kernel requires 16-byte-aligned operands, and the drop
probe already showed the library refuses the call outright below that, so the
model's operands satisfied the same test the bench's did. The residual is
uniform across shapes and kernel families - 1.101, 1.104 (fc1's 2.208 halved),
1.182 and 1.120 - not concentrated in one shape or one library.

## 4. What the uniform 12% is, and what is left of it

The corrected bench, re-run this session at the same shapes (Spark, sm_121,
18,799 rows, one process per arm):

| shape | K | N | BF16 ms (3 calls) | TFLOPS | BF16 ms (40 calls) | FP8 ms (3) | FP8 ms (40) |
|---|---:|---:|---:|---:|---:|---:|---:|
| qkv | 5,376 | 21,504 | 40.436 | 107.5 | 40.215 | 20.359 | 19.991 |
| out | 7,168 | 5,376 | 14.459 | 100.2 | 14.350 | 6.892 | 6.743 |
| fc1 | 5,376 | 28,672 | 53.240 | 108.9 | 53.166 | 26.950 | 26.498 |
| fc2 | 14,336 | 5,376 | 37.871 | 76.5 | 34.906 | 14.107 | 13.930 |
| **50 layers** | | | **7,300** | 99.3 | **7,132** | **3,415** | **3,358** |

- **A short burst and a sustained run are the same speed.** 40 calls per shape
  (~11 s of GEMM) is 2.3% *faster* than 3, so there is no warm-up or
  thermal-drift story in the bench, and `clocks.sm` sampled at 500 ms holds
  2,483-2,489 MHz at 74-81 W throughout the sustained run.
- **The model runs 90 MHz lower.** The same sampler over the gate's run reads
  2,294-2,411 MHz (mode 2,398) with peaks of 95.4 W - a 3.6% clock deficit,
  which is a third of the 12.1% total gap.
- **About 8% is unexplained.** Nothing in this session's evidence accounts for
  it: the kernels and grids match, the operands pass the same alignment test,
  the trace is not inflated, and the clock sampler cannot see an instantaneous
  clock inside a 500 ms window under a mixed load. The candidates left are
  the per-call instantaneous clock under a workload that draws 95 W rather
  than a bench's 81 W (a power-limited SoC, and this host has no clock
  policy), and the L2/IO state each GEMM inherits from its neighbours. Neither
  is addressable from the port, and both are named here so no later session
  re-opens the question as if it were.

One property of the bench is worth recording because it is a library fact, not
ours: `fc2` runs at 76.5 TFLOPS where `qkv` and `fc1` reach 107-109. Same
kernel, same 128x256x32 tile; `fc2`'s N of 5,376 gives it 21 N-tiles against
`fc1`'s 112, so it has fewer and longer CTAs and worse wave quantization. The
model inherits the same 67-79 TFLOPS band on that shape.

## 5. What FP8 is worth, corrected

FP8 (E4M3 x E4M3, FP32 accumulate, BF16 output, unscaled) at the corrected
shapes: **3,415 ms per evaluation against BF16's 7,300, or 3,358 against
7,132 sustained** - 212 TFLOPS against 99, and the same ratio shape by shape
(213.5/107.5, 215.0/108.9, 210.2/100.2, 205.4/76.5). Inside the bench that is
**3,885 ms off a 7,300 ms evaluation; applied to the model's own 8,270 ms at
the measured FP8/BF16 ratio it is 3,870 ms against 8,270, a saving of 4,400 ms,
39% of an 11,282 ms evaluation.** Over a four-evaluation render that is 17.6 s
of the denoise, where the earlier record's number was 13.0 s.

The activation side is not free and is not in the table: quantizing each A
operand needs one pass that reads it in BF16, reduces the scale and writes it
in E4M3 - about 1.82 GB per layer for the four shapes, ~380 ms per evaluation
at the measured streaming rate. Net of that, the lever is ~4,000 ms per
evaluation, 16 s of a four-evaluation render. The weights can be converted once
at load (15.4 GB read, 7.7 GB of E4M3 written) and cached.

## 6. The release does not ship FP8 weights - it converts at load

This is what makes the lever a port rather than an artifact ask. Read in the
release's Spark runtime:

- `runtime/stage1.py:46` imports `FP8Config`, `FP8QuantizeMethod` and
  `convert_model_to_fp8` from `fastvideo.layers.quantization.fp8_config`;
- `runtime/stage1.py:52` names the route
  `native_FastVideo_tensorwise_W8A8_after_original_BF16_LoRA_merge`;
- `runtime/stage1.py:126-134` (`quantize_after_merge`): the LoRA is merged in
  BF16 first, the wrapper layers are stripped back to their base linears, then
  every linear whose name passes `quantized_linear_name` is tagged
  `FP8Config()` + `FP8QuantizeMethod("tensor")` and `convert_model_to_fp8`
  walks the model;
- `runtime/stage1_ops/lookup.py:15-16` defines the targets as names ending
  `.to_q`, `.to_k`, `.to_v`, `.to_out`, `.ff.fc_in`, `.ff.fc_out` - the six
  projections per block, i.e. exactly the four shapes in section 2, with
  nothing else in the block quantized.

So the reference quantizes **per tensor** (weight scale per tensor, and
FastVideo's tensorwise method quantizes the activation per tensor at run time),
from the BF16 checkpoint it already has. The scheme's exact kernel lives in the
`fastvideo` package, which is not installed on this host (it is the pinned
stage-2/Spark environment), so its implementation is the one thing a port would
have to read from FastVideo itself rather than from this tree.

What is *not* measured anywhere in the campaign: the quality of W8A8 on this
50-block DiT at our 18,799-row, four-update workload. The reference accepts it
at 672x384. That is the gate a port would have to clear, and it is the reason
this stays an operator decision.

## 7. Method

- Model trace: `~/lev1-staged-4.nsys-rep` / `.sqlite` on the Spark, exported
  with `nsys export --type sqlite`; aggregated by `~/gemm_analyze.py` (per
  kernel name with grid and block, count, min/median/max/sum) and
  `~/span_check.py` (per evaluation window: kernel-duration sum against span).
  Both scripts are on the Spark and in this session's `/tmp`.
- Bench: `make CUDA_ARCH=sm_121 tests/h3/linear_fp8_bm` then
  `tests/h3/linear_fp8_bm 18799 {3,40}`, `fc1`'s N corrected to `2 * ffn` in
  this commit. Clocks alongside with
  `nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits -lms 500`.
  One script drove both arms and the gate (`~/fp8v2.sh`, log
  `~/fp8v2.log`, tables `~/fp8v2-{burst,sustained}.txt`, clocks
  `~/clock-{burst,sustained,gate}.txt`), all on the Spark.
- Both arms ran on the Spark in one session, one GPU job at a time, with no
  other process on the device (`pgrep -a ds4` empty, `nvidia-smi` 0% at start).
