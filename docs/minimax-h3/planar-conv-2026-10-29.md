# P2d: the im2col-free convolution, or what the attribution bought (2026-10-29)

Unit P2d of `docs/minimax-h3/ltx-sol-pipeline-plan-2026-10-28.md`, on branch
`feat/sol-h3`. The target was the release's own 2.13 s for upscale + adapter
(3.8% of its 56.17 s) against our 4.150 s. The unit took the upscaler's forward
to **2.219-2.276 s** and the adapter to **0.208-0.210 s** on the Spark, i.e.
2.43 s against the release's 2.13 s (1.14x) and 1.73x faster than where it
started (4.144 s).

Everything here is measured on the DGX Spark (GB10, 48 SMs) at busy 2,405-2,431
MHz, no clock policy, and cross-checked on the workstation (RTX 4070 SUPER) at
2.77 GHz. One process per arm; the numbers repeat inside 1%.

## The measurement that chose the form

`tests/h3_ltx_adapter_gate` with `--arm bf16`, one CUDA trace, its sqlite
aggregated per kernel and split at the upscaler/adapter boundary (the gate
finishes the upscaler before it loads the adapter; `h3b_resize3d` ends at
2,953 ms, `h3b_unshuffle2` starts at 4,144 ms). The upscaler's forward, 3,971 ms
of device work in 2,288 kernels:

| kernel | instances | ms | share |
|---|---|---|---|
| `h3b_im2col_kernel` | 522 | 2,130.2 | 53.6% |
| `nvjet_sm121_tss_mma_256x112x64` | 432 | 1,183.6 | 29.8% |
| `h3b_epilogue_kernel` | 572 | 150.0 | 3.8% |
| `h3b_norm_apply_kernel` | 122 | 87.2 | 2.2% |
| `h3b_pad3d_kernel` | 100 | 85.0 | 2.1% |
| `h3b_add_kernel` | 72 | 72.1 | 1.8% |
| the remaining GEMMs (5 shapes) | 143 | 155.6 | 3.9% |
| norm partial, silu, depthwise, resize, pack, wmma | 313 | 106.1 | 2.7% |

The im2col was not slow in itself; it was slow because of how it read. A warp's
32 lanes take consecutive columns of the matrix, which is `((channel*kd + kt)*kh
+ ky)*kw + kx`, so a warp spans about three `kx` values and then jumps a row: 32
lanes fill roughly 5.5 sectors, and the matrix it writes is 27x the activation
the convolution came from. Per conversion that is 104 GiB of im2col traffic plus
the GEMMs' re-read of it at four N-tiles.

Two forms would remove it: a hand-written implicit GEMM that gathers A in the K
loop, or a layout that makes every tap a dense matmul. The second one is a
`cublasLt` call, so it was tried first, and it turned out to be enough — the
GEMMs it feeds reach 90% of the peak on the dominant shape, which is a hand
kernel's only remaining advantage.

## The form

`h3b_planar` (`native/h3/h3_dit_gpu.cu:5153`, kernel at `:4407`) writes the
padded activation in channel-last form once per kernel column tap:

    planar[t'][y'][x][s][c] = padded[c][t'][y'][x + s]

with `s` the column tap, so the element sits at `((t'*pad_height + y')*out_width
+ x)*K + s*channels + c` for `K = kw*channels`. That layout is the whole trick:
the element (row `m = y*out_width + x`, column `k = s*channels + c`) of the
convolution's A operand for the tap `(kt, ky)` is at

    ((t + kt)*pad_height + y + ky)*out_width*K + m*K + k

which is a plain row-major matrix of leading dimension `K` — the rows are the
padded image's rows, and every tap `(kt, ky)` is the same buffer at a row
offset. `convolve_planar` (`crates/ds4-core/src/minimax_h3/ltx_adapter.rs:869`)
issues the `kd*kh` taps as `h3b_matmul_planar` (`:5225`) calls that accumulate
in f32 (`beta = 1` after the first), then the frame's epilogue transposes the
result into channel-major order. One frame at a time, because the A window a
frame reads — `kd` images of `kh + out_height` rows of `K` halfwords — is what
has to stay near the cache.

Three consequences worth naming:

- **The weight needs a copy.** The checkpoint's `[out][in][kd][kh][kw]` order has
  the `kw` triple contiguous and the channel stride `kd*kh*kw`, which cannot
  express a `(out, kw*in)` window. `h3b_weight_pack` (`:5197`, kernel at `:4472`)
  writes `[kt*kh + ky][out][s*in + c]` once at load. It is a layout copy, not a
  value change, and `h3b_pack_kernel` is checked element for element by the ops
  gate.
- **The 3x3x3 is nine matmuls, not one.** The taps cannot go into one K because
  the A rows of different taps are the same rows at different offsets; that is
  also why the accumulation needs `beta = 1` and a second pass over C.
- **The plan's `relu` epilogue question does not arise here.** This is the latent
  path; the 150-call projection GEMM is the DiT's, in stage 1.

Both paths stay in the tree and `H3_LTX_CONV` selects between them
(`ltx_adapter.rs:839-852`); the classic one is the control and the fallback, and
the scratch sizes the im2col matrix to one element when the planar path is the
one in use (`ltx_adapter.rs:1520`, `:2467`).

## Correctness

`tests/h3_ltx_ops_gate` pins the three new primitives against host arithmetic on
shapes that are not square and not powers of two:

| case | values | worst error |
|---|---|---|
| `planar` (the padded channel-last buffer, `kw` copies) | 1,350 | 0 (exact) |
| `pack` (the pass order) | 144 | 0 (exact) |
| `matmul_planar` (one tap, one window) | 40 | 1.192e-7 |
| `matmul_planar_accumulate` (two taps into one accumulator) | 40 | 2.384e-7 |

`tests/h3_ltx_adapter_gate`, both arms, both paths, both hosts — the reference
is the profile's own frozen tensor, so the two paths are compared against the
same thing:

| host | path | arm | upscaler_forward | adapter_convert | h3_upscaled | adapter_output | refiner_input |
|---|---|---|---|---|---|---|---|
| Spark | classic | bf16 | 3.890 / 3.878 | 0.254 / 0.257 | 6.736e-3 | 1.210e-2 | 1.213e-2 |
| Spark | planar | bf16 | 2.276 / 2.219 | 0.210 / 0.208 | 6.668e-3 | 1.208e-2 | 1.212e-2 |
| Spark | classic | f32 | 12.014 | 0.774 | 2.287e-6 | 5.426e-6 | 5.495e-6 |
| Spark | planar | f32 | 10.302 | 1.006 | 3.002e-6 | 6.127e-6 | 6.203e-6 |
| 4070 | classic | bf16 | 3.379 / 3.396 | 0.191 / 0.200 | 6.678e-3 | 1.206e-2 | — |
| 4070 | planar | bf16 | 2.283 / 2.337 | 0.164 / 0.166 | 6.685e-3 | 1.209e-2 | — |
| 4070 | classic | f32 | 7.978 | 0.491 | 2.874e-6 | 6.076e-6 | 6.154e-6 |
| 4070 | planar | f32 | 7.264 | 0.443 | 2.962e-6 | 6.064e-6 | 6.139e-6 |

The two paths agree with the reference to the same order in every arm; the
differences between them are the bf16 arm's own noise. The f32 adapter is the
one regression (0.774 -> 1.006 s on the Spark): at that size and dtype the
planar build's three copies and the nine passes over the f32 accumulator cost
more than the im2col they replace. Production is bf16, where the adapter is
faster both times (0.254 -> 0.210 on the Spark).

## What the planar path now spends

Same trace, the planar run: 2,660 ms of device work, 2.66 s of a 3.22 s span.

| kernel | instances | ms | share |
|---|---|---|---|
| `cutlass_80_tensorop_s16816gemm_bf16` (M 4,032) | 13,698 | 1,300.9 | 48.9% |
| `nvjet_sm121_tss_mma_96x128x64_3` (M 1,008) | 13,392 | 330.3 | 12.4% |
| `h3b_planar_kernel` | 146 | 228.2 | 8.6% |
| `h3b_epilogue_kernel` | 4,024 | 193.3 | 7.3% |
| `h3b_norm_apply_kernel` | 168 | 97.1 | 3.7% |
| `h3b_add_kernel` | 118 | 86.5 | 3.2% |
| the remaining GEMMs (4 shapes) | 372 | 213.1 | 8.0% |

Three findings:

- **The GEMM runs at 62% of the peak, and the tap loop's accumulator is why.**
  The released shapes are `M 4,032, K 1,536, N 512` (6.34 GFLOP per call,
  13,698 calls in 1,300.9 ms = 94.4 us each) and `M 1,008, K 1,536, N 512`
  (1.59 GFLOP, 13,392 calls in 330.3 ms = 24.7 us each). Both are 62-63% of the
  GB10's 108 TFLOPS `mma.sync` peak, so the compute floor is 58.7 us and 14.7 us
  against 94.4 and 24.7. The excess is the accumulation: `beta = 1` re-reads and
  re-writes the whole f32 accumulator once per tap, 16.5 MiB a call at the wide
  shape, competing with the tap's 12.4 MiB A window for a 24 MiB L2. The tap
  decomposition is what buys the L2-friendly A, and this is what it costs.
- **The apparent idle is the load, not the launches.** 19,502 gaps totalling
  592 ms, of which 550.9 ms is in 86 gaps over 50 us — and those sit in the two
  model-load phases (a 201.7 ms gap at t=0.304 s, 80.7 ms before a `pack`
  kernel, 79.8 ms before a `planar` kernel at t=2.794 s). The 19,416 gaps inside
  the forward have a **median of 2.1 us** and total 41 ms, so per-launch host
  cost is not a target here, and neither is caching the `cublasLt` descriptors.
- **`cublasLtMatmulAlgoGetHeuristic` returns what the library already picks.**
  Measured: `H3_LTX_GEMM_ALGO=heuristic` reproduces the kernel mix instance for
  instance, and the two settings' walls agree inside 0.1% (2.285 vs 2.283 s).
  The explicit query is therefore off by default (`native/h3/h3_dit_gpu.cu:4939`)
  and the plan's item 2 is closed as neutral rather than adopted.

### The block that was tried and rejected

The natural fix for the accumulator is to block the frame's rows so that C and
the tap's A window fit L2 together. Implemented at 1,024 rows (2 MiB of C and
3.2 MiB of A), 2026-10-29, and measured on the workstation: **2,283 -> 3,158 us**
for the upscaler against the classic path's 3,478. The block costs the call its
parallelism — `M 1,024, N 512` on a 128x128 tile is 32 CTAs, 0.67 of a wave on a
56-SM part and 0.67 of one on the GB10's 48 — and losing the waves costs more
than the accumulator's DRAM walk. The taps stay whole-plane, 2.6 waves at the
widest convolution. `h3b_matmul_planar` keeps the `c_offset` the block needed
(`native/h3/h3_dit_gpu.cu:5225`) so the experiment is repeatable.

## The lever that was not there: the temporal chunk

The upscaler runs its 37 latent frames through the node's chunk machinery at
`chunk = 32` with `overlap = 5` (`ltx_adapter.rs:140`), which computes 62 frames
of convolution work — 1.68x the latent's own frame count. Raising the chunk to
64 would put the clip on the node's own `T <= chunk` single-pass branch
(`nodes/minimax_h3_latent_upscaler_3d.py:266`) and cost 1.37 s instead of 2.25 s
on the Spark.

It is wrong. The model's temporal blocks downsample and upsample in time, so a
segment's frame parity decides the intermediate grid, and the single 37-frame
segment lands a materially different latent: **3.806e-1 relative** against the
reference, identical in both convolution paths and both arms (classic and planar
agree to 4 digits, which is what makes it a property of the plan and not of the
convolution). The chunking the node's tooltip calls a VRAM saving is part of the
arithmetic for this model. `UPSCALER_CHUNK` keeps the node's value.

## What is left, and what would take it

0.3 s separates this from the release's own 2.13 s, in three named pieces: the
build (228 ms of the gate, 9% of the upscaler), the epilogue (193 ms, 7%), and
the GEMM's 1.66x over its compute floor (the tap loop's accumulator). A
hand-written kernel that gathers A from the channel-major activation in its K
loop, keeps the `kd*kh` taps in one K and writes the channel-major output
directly would remove all three: no planar buffer to build, no f32 accumulator
to re-walk nine times, no separate transpose. Its ceiling is the 106 TFLOP at
108 TFLOPS = 0.98 s of GEMM, so the whole stage could plausibly land at 1.1-1.3 s
against today's 2.22. That is a 0.9 s gain on a 56 s pipeline, and it costs a
mma kernel with a real correctness surface — which is why it is not the next
unit.

The next unit is P3, the LTX-2.5 refiner (44% of the release's time), with the
stage-2 environment decision the plan already names.

## Reproducing

    make CUDA_ARCH=sm_121 tests/h3_ltx_ops_gate tests/h3_ltx_adapter_gate
    ./tests/h3_ltx_ops_gate
    H3_LTX_CONV=planar ./tests/h3_ltx_adapter_gate ~/ltx-golden \
      /mnt/h3-artifacts/H3-upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors \
      /mnt/h3-artifacts/H3-to-LTX-Latent-Adapter --arm bf16

The two traces are kept on the Spark as `~/p2d-attr.nsys-rep` (the classic
attribution) and `~/p2d-nsys.nsys-rep` (the planar one), with their sqlite
exports and the aggregation scripts beside them (`p2d-attr.sh`, `p2d-split.py`,
`p2d-nsys.sh`, `p2d-gaps.py`).
