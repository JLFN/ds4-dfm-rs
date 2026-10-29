# The projections in FP8, and the gap between the same GEMMs alone and in the model

*Corrected 2026-10-29*: the `fc1` row below priced N at `ffn` (14,336) when the
model's fc1 is the fused gate+up pair, N = 2 x `ffn` = 28,672
(`h3_dit.c:501-525`, `layout.rs:563`, and the model's own trace, which shows
grid.y 28 for that shape against the bench's 14). The bench is fixed in the
same commit; the corrected table, the resolution of section 2's contradiction
(it was mostly this error, not a real kernel-selection gap), the retired
alignment hypothesis and the re-priced FP8 lever are in
[`dit-gemm-shape-2026-10-29.md`](dit-gemm-shape-2026-10-29.md). The `bf16 ms`
and `fp8 ms` columns here are unchanged measurements; the `TFLOP` column, the
"50 layers" total and the section 2 conclusions are superseded: the model runs
724.5 TFLOP per evaluation at 87.6 TFLOPS, and FP8 is worth about 4,400 ms per
evaluation, not 3,262.

Two measurements, and a contradiction between them that is worth more than the
FP8 answer itself.

## 1. FP8 at the four projection shapes

`tests/h3/linear_fp8_bm` builds the same cuBLASLt configuration `h3d_linear`
uses (row-major, `TRANSB`, FP32 accumulate, BF16 output, alpha 1) at job A's
shapes and times it in BF16 and in E4M3 x E4M3. Spark, `sm_121a`, busy
2,398-2,450 MHz, three calls per arm, one process:

| gemm | K | N | TFLOP | BF16 ms | BF16 TFLOPS | FP8 ms | FP8 TFLOPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| qkv | 5,376 | 21,504 | 4.347 | 40.404 | 107.6 | 20.154 | 215.7 |
| out | 7,168 | 5,376 | 1.449 | 14.491 | 100.0 | 6.690 | 216.6 |
| fc1 | 5,376 | 14,336 | 2.898 | 27.260 | 106.3 | 13.534 | 214.1 |
| fc2 | 14,336 | 5,376 | 2.898 | 37.824 | 76.6 | 14.367 | 201.7 |
| **50 layers** | | | **579.5** | **5,999** | 96.6 | **2,737** | 211.7 |

FP8 is 2.19x the BF16 rate here (211.7 against 96.6 TFLOPS), and it would take
the four projections from 5,999 to 2,737 ms per evaluation: **3,262 ms off an
11,510 ms evaluation**, 13.0 s off a four-evaluation render's denoise, before
any accuracy consideration and before the activation-quantization cost.

What this does not price: quantizing the activations (a per-row or per-tensor
pass over 202-404 MB per call), the scales and their accuracy, and whether a
W8A8 artifact of this 50-block DiT reproduces the BF16 render's quality. It
also assumes the E4M3 weights can be produced at all - the release ships its
stage 1 in W8A8 FP8, so the recipe exists, but it is the operator's artifact
(decision 2 in the handoff).

## 2. The same GEMMs are 41% slower inside the model

The four shapes above sum to 5,999 ms per evaluation when run alone. The
model's own trace says its four projections take **8,446 ms** per evaluation
(`cutlass_80_tensorop_bf16_s16816gemm...` 7,601.24 ms over 150 calls plus
`nvjet_sm121_tst_mma_128x240x64` 845.24 ms over 50, from the
`--denoise` nsys pair, two independent captures agreeing to 0.06%) - i.e. the
same arithmetic, same shapes, same library, at 68.6 TFLOPS instead of 96.6.

So **2,447 ms of every evaluation is spent on something other than the GEMM
work**, which is larger than what the elementwise pair's whole unit recovered
(1,382 ms). Two candidate causes, neither yet established:

- **The kernel the library picks is not the same.** The DiT window of the
  trace contains `cutlass_80_wmma_tensorop_bf16` family kernels, which the
  standalone bench does not: those are the WMMA path, not the `s16816`
  tensorop one that the bench's 100+ TFLOPS implies. Why cuBLASLt would pick
  differently for the same shape is the question; the operands' alignment is
  one answer (below), their strides another, and the absent workspace
  (`h3d_linear_lt_compute` passes `NULL, NULL, 0`) a third.
- **The operands' alignment.** A probe with the operands shifted by N elements
  shows the library is strict: a 2-byte shift fails the call outright
  (`CUBLAS_STATUS_...`, the bench reports "cuBLASLt gemm failed"), an 8-byte
  shift does too for E4M3, and 16-byte shifts are clean. The model's tensors
  are views into larger allocations, so their alignment is whatever the
  uploader chose - the traced kernel names say `align8`, which is 8-byte
  alignment, and this probe says that is not enough for every kernel family.

That is the next measurement, and it is cheap: the model's weight and
activation pointers can be printed at the `h3d_linear` call, and the bench can
be told to reproduce whatever alignment they have. A 2.4 s per evaluation gap
that turns out to be a pointer is the cheapest thing left in this campaign.

## 3. Method

The bench is `tests/h3/linear_fp8_bm [rows] [calls] [drop]`, `make
CUDA_ARCH=<arch> tests/h3/linear_fp8_bm`; `drop` is the elements the operands
are shifted by, its own probe. The in-model figures are the
`docs/minimax-h3/dit-elementwise-2026-10-29.md` pair's table, which is the same
host, session and binary as the elementwise work.
