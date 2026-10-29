# Where a DiT evaluation goes: the per-kernel table

One evaluation of the fast arm at job A's shape costs 12.35 s of GPU kernels.
The table below is measured, not modelled: it comes from two Nsight Systems
captures of the isolated DiT gate (`tests/h3_dit_gate`, the `h3_dit_gate`
target), at 768x768 with 107 frames, four and two evaluations, with
`H3_DIT_SOL=0.9` — the same sparsity the fast arm serves with — on the Spark at
`ab87a1e7`, 2.4-2.5 GHz, no clock policy on this host.

It re-ranks the campaign: the four projections are already at roughly
two-thirds of the BF16 tensor peak on this hardware, while two elementwise
kernels and the whole sparse-attention path together hold a third of the
evaluation at a small fraction of their own bounds. FP8 is not the only lever,
and by this table it is not the first one.

## 1. Method

The gate refuses `--steps 1`, so the pair is two evaluations and four and the
difference is two evaluations' worth of kernels, halved. Everything the process
pays once — the text encoder, the weight upload, the first sparse search,
cuBLASLt's first algorithm choice — cancels in the difference. Both captures ran
under `nsys profile --trace=cuda --sample=none --cpuctxsw=none`; the `.nsys-rep`
files are kept in `~/phaseB/` on the Spark, so the analysis can be redone
without the GPU.

The gate's own timings bracket the result: `DENOISE` 25.181 s for two
evaluations and 49.846 s for four, and the server's hot render reports 50.550 s
for four. The table sums to 12,345.9 ms per evaluation against those 12.33-12.64.

| kernel | ms/eval | calls | share |
|---|---:|---:|---:|
| `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8` | 7101.99 | 150 | 57.5% |
| `nvjet_sm121_tst_mma_128x240x64_2_32x120x64_tmaAB_bz_TNNN` | 798.97 | 50 | 6.5% |
| `h3d_sol_forward_kernel` | 1311.59 | 50 | 10.6% |
| `h3d_qkv_rope_kernel` | 1030.97 | 50 | 8.4% |
| `h3d_adaln_kernel` | 1013.47 | 102 | 8.2% |
| `h3d_swiglu_kernel` | 383.93 | 50 | 3.1% |
| `h3d_gate_kernel` | 289.19 | 100 | 2.3% |
| `h3d_sol_vsa_kernel` | 216.69 | 50 | 1.8% |
| `h3d_sol_pool_kernel` | 123.95 | 50 | 1.0% |
| `h3d_sol_threshold_kernel` | 66.28 | 50 | 0.5% |
| `h3d_sol_stats_kernel` | 1.29 | 50 | 0.0% |
| `cutlass_80_simt_sgemm_128x128_8x4` and smaller | 5.75 | 3 | 0.0% |

## 2. The four groups

- **The projections: 7,901 ms, 64.0%.** 150 calls of a Blackwell BF16
  tensor-core GEMM plus 50 of NVJet's `sm121` MMA. The gate's log gives 18,799
  rows, and the block geometry gives 4 x (2xRxd_inxd_out) per layer: qkv
  2xR x 5376 x 21504, out 2xR x 7168 x 5376, and both MLP projections
  2xR x 5376 x 14336, which is 11.59 TFLOP per layer and 579.5 TFLOP per
  evaluation over 50 layers. Against the 7.901 s that is 73.3 TFLOPS: 68% of the
  108 TFLOPS `mma.sync` peak at the 2.19 GHz reference clock, or 61% of the
  ~120 TFLOPS that 2.45 GHz implies. The call counts (150 = 3x50, 50 = 1x50)
  say three of the four projections run on the first kernel and one on the
  second; which is which, and why one of them asks for a `relu` epilogue, is a
  reading of the block code, and the FLOP figure above assumes all four are here.
- **The two elementwise kernels: 2,044 ms, 16.6%.** `qkv_rope` (20.6 ms per
  call) at minimum writes q, k and v — 18,799 x 21,504 x 2 bytes, 808 MB — after
  reading the 202 MB input, which is 4.0 ms at 250 GB/s. `adaln` (9.94 ms per
  call, twice per layer) reads and writes 18,799 x 5,376 x 2 bytes, 404 MB, which
  is 1.6 ms. Both sit at about a sixth of their bandwidth floors.
- **The sparse attention path: 1,719 ms, 13.9%.** `sol_forward` runs the
  selected blocks; `sol_vsa`, `sol_pool`, `sol_threshold` and `sol_stats` are
  the fold. Dense attention at this shape is 10.13 TFLOP per call
  (2 x 2 x R^2 x heads x head_dim), so 507 TFLOP per evaluation over 50 calls;
  the log reports the realized share as 0.1370, and `sol_forward` therefore
  does about 69 TFLOP per evaluation in 1.311 s: 53.2 TFLOPS, 49% of the
  108 TFLOPS `mma.sync` peak. The selection did its job - the exact share is
  13.7% of the dense blocks, a 7.3x FLOP ratio - and the kernel converts it:
  26.07 ms per call against the dense kernel's 173-174 and its own all-exact
  control's 150.3 (`sol-forward-2026-10-28.md`, same host), i.e. 58.6 and
  67.4 TFLOPS for those two. 79% of the all-exact arm's rate on 13.7% of its
  work.

  *Corrected 2026-10-29*: this bullet's first version divided a per-call FLOP
  count (1.39 TFLOP) by a per-evaluation time (1.311 s) and reported
  1.06 TFLOPS "about 1% of the tensor peak". That is 50x off, and it is the
  reason the sparse path was ranked as the second lever in section 3 below.
  The per-call and per-evaluation figures are above; the remaining headroom
  in this path is the fold's own constant (about 5.6 ms of a 26.1 ms call)
  and the selector's 410 ms of the evaluation.
- **The rest: 674 ms, 5.4%.** `swiglu` 383.9 ms moves 1.28 GB, 5.1 ms at
  250 GB/s against 7.7 measured, so it is near its floor and not a target;
  `gate` 289.2 ms and the small VSA helpers are the remainder.

## 3. What this decides

Ranked by recoverable time, with each change's numerical class stated as the
kind of work it is:

1. The projections, 7.901 s. Reaching 90% of the peak at 2.45 GHz would make
   them 5.36 s and give back 2.54 s per evaluation. What that takes — a better
   cuBLASLt algorithm and workspace, a different layout, or our own tile — is
   the next measurement, not an assumption. This is also the term FP8 would
   halve, and the two are additive: FP8 at today's 61-68% efficiency would make
   the same 579.5 TFLOP take about 2.7 s.
2. `qkv_rope` and `adaln`, 2.044 s. At their floors they would be about
   0.2 s. Neither is a numerical change: same arithmetic, better access pattern
   and fewer passes.
3. The sparse path, 1.719 s, of which `sol_forward` is 1.311 s. It is not at
   1% of the peak - see the correction above: it is at 53 TFLOPS, 79% of the
   rate its own all-exact control reaches. What is left here is the fold's
   constant and the selector, together under 0.7 s of the evaluation.
4. Everything else, 0.674 s, with `swiglu` already near its floor.

Cold-read work (Phase C's FP8 checkpoint) still halves `LOAD` 66 GiB to 33 GiB,
which is 9-43 s on the first render of a process but nothing at steady state.

## 4. Limits

- The delta is a mean over the two extra evaluations, not a per-evaluation
  trace. Nothing here separates the first evaluation's cold cost from the rest.
- The FLOP figures are derived from the block geometry and the gate's own
  `rows=18799`; the projection-to-kernel assignment is inferred from call counts.
- Kernel times under Nsight are CUPTI-reported GPU durations; the two captures
  differ in total process time (the profiler's own overhead is not uniform), but
  the differencing removes everything that is constant between them.
- `relu` in the first kernel's name is unexplained by this record. If a
  rectifier is being applied where the block does not want one, the arithmetic
  is wrong in a way no timing can see; the block code must be read before the
  projection work starts.
- One host state, one sparsity setting, one shape. The 2.4-2.5 GHz regime on a
  host with no clock policy is not the 2.19 GHz reference used for the peak.
