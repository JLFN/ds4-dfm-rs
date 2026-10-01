# Qwen-Image-2.1 port — phase S, the decisive spike

[Documentation index](README.md) | [Repository README](../README.md) |
[Plan](qwen-image-2.1-plan.md) | [Recipe](qwen-image-2.1-recipe.md) |
[Roadmap](qwen-image-2.1-roadmap.md) | [Decision](qwen-image-2.1-decision.md)

Status: complete. Three measurements, each against the reference's own kernels on
the same hardware. Verdict: **none of the three shows a gain the port can
capture.** By the decision rule in [roadmap](qwen-image-2.1-roadmap.md) section 4,
the performance case for the port is dead; if the port proceeds it is on
ownership grounds only, and every later phase must be justified that way rather
than by speed.

Date: 2026-10-01. Host and reference: RTX 4070 SUPER (AD104, sm_89, 56 SMs,
11894 MiB usable), CUDA 13.3 (V13.3.73), driver 610.43.03. Reference binary
`/data/imagegen/bin/sd-cli` version `master-890-74988b2-4-g6dcb5bb`, commit
`6dcb5bb`, the pinned revision in the [recipe](qwen-image-2.1-recipe.md) section 1.

## 1. What was run

The reference was executed on its artifact at the real workload shape, then
profiled with Nsight Systems, so every reference number below is the reference's
own kernel on this GPU rather than a reimplementation of it:

```
sd-cli --mode img_gen \
  --diffusion-model .../qwen-image-2.1-Q6_K.gguf \
  --vae            .../qwen_image_2.1_vae_bf16.safetensors \
  --llm            .../Qwen3VL-8B-Instruct-Q4_K_M.gguf \
  --backend te=cpu,diffusion=cuda0,vae=cpu \
  --params-backend te=cpu,diffusion=cuda0,vae=cpu \
  -p "a red cube on a wooden table" -W 1024 -H 1024 --steps 3 --seed 42 --cfg-scale 6.0
```

That is the deployed placement (`/data/imagegen/imagegen.toml:110-113`) with the
model and prompt that reach the DiT shape the roadmap names:

- The scheduler reports `image_seq_len=4096` — the latent grid is 64x64, so the
  image segment is 4096 tokens. With a 128-token text prefix the GEMM width is
  4224. That is the "about 4.3k tokens" of the roadmap, confirmed by measurement
  rather than by the geometry in the recipe.
- DiT weights on device: **5604.32 MB**, 297 tensors, 6 blocks. Compute buffer:
  **2317.45 MB**. See section 6 for why this corrects the plan.
- Steady state: **4.84-4.87 s per DiT step** in the recorded run, **2.93-2.96 s**
  with `--diffusion-fa`. The first step (21.11 s and 18.57 s) is one-time lazy
  load and graph build and is excluded from every number below. An independent
  re-run on the same host, with the clocks in a slightly different state, put the
  same two configurations at 4.75-4.78 s and 2.81-2.82 s, so the *ratio* (1.65x)
  is the reproducible quantity rather than the absolute seconds.
- Around the step: text encode on CPU 13.13 s, VAE decode on CPU 76.93 s.

The ds4-side measurements are two standalone harnesses under
`misc/scratch/phase-s/` (gitignored; the evidence index is section 9). They link
the tree's own vendored objects, so the kernel measured is the production one,
not a copy of it.

Clocks were **not locked**: this host has no passwordless sudo, so `nvidia-smi
-lgc` is unavailable. Back-to-back runs on an otherwise idle GPU were observed at
2565-2805 MHz (idle 210 MHz), sampled at 200 ms during each harness run. Every
number below is therefore a same-session comparison at the same observed clock
band, not a clock-locked absolute.

## 2. M1 — attention at the DiT shape

Question: can ds4's stack beat the reference's attention at 32 heads of 128 over
a 4096-query image segment attending to a 4224-token range, no KV cache?

The reference's answer first, because it is not what the plan assumed. **The
deployed configuration does not use flash attention.** No `flash_attn_ext`
kernel appears in the default trace; the DiT attention runs unfused as two
batched matmuls with a separate score-scaling pass and a masked softmax between
them:

| Kernel (one steady-state step, default config) | Launches | Total | Per launch | Geometry |
| --- | --- | --- | --- | --- |
| `scale_f32` (the score scaling before softmax) | 128 | 681.2 ms | 5.322 ms | gridX 2101760 / 2104832, gridY=gridZ=1, block (256,1,1) |
| `soft_max_f32<(bool)1,(int)0,(int)0,float>` | 128 | 706.3 ms | 5.518 ms | grid (4096,32,1), block (32,1,1) |
| `cutlass::Kernel2<s1688gemm_256x64_16x4_tn>` | 64 | 520.6 ms | 8.134 ms | grid (512,3,32), block (128,1,1) |
| `cutlass::Kernel2<s1688gemm_64x64_16x6_tn>` | 64 | 483.6 ms | 7.555 ms | grid (128,1,32), block (128,1,1) |
| **attention total** | | **2391.7 ms** | | **50.9% of the step** |

The `scale_f32` row is the unfused path's score scaling, and it was missed on the
first pass. The step runs 520 of those kernels in total, in eight distinct grid
shapes; under flash attention 392 remain and exactly four of the eight shapes
disappear, 32 launches each — gridX 11, 29, 2101760 and 2104832. The two large
ones carry 341.1 and 340.1 ms, the two tiny ones nothing measurable, so 681.2 ms
of the step belongs to the attention path and vanishes when it is fused. This
raises the unfused share from 36.4% to 50.9%.

The control settles what that means. Adding the single flag `--diffusion-fa`
replaces all of it with one fused kernel:

| Kernel (one steady-state step, `--diffusion-fa`) | Launches | Total | Per launch | Geometry |
| --- | --- | --- | --- | --- |
| `flash_attn_ext_f16<128,128,64,1,false,false>` (image segment) | 64 | 544.0 ms | 8.500 ms | grid (112,1,1), block (32,4,1), 26624 B smem |
| `flash_attn_ext_f16<128,128,16,1,false,false>` (text segment) | 64 | 0.4 ms | 0.007 ms | grid (32,1,1) |
| **attention total** | | **544.4 ms** | | **18.2% of the step** |

- Flash attention is **4.39x faster than the unfused path for identical
  arithmetic** per step (2391.7 ms to 544.4 ms), and the flag alone takes the
  steady-state step from 4.855 s to 2.945 s (**1.65x**) on a 4697.7 ms to
  2992.5 ms kernel budget.
- The whole-run totals from the two runs (120.92 s and 103.62 s) are **not**
  comparable and are not used as evidence: the text encoder loaded cold in the
  first (13.13 s) and warm in the second (1.46 s), an 11.7 s difference that
  swamps the sampling change. The per-step numbers are the attributable ones.
- The deployed bridge does not pass `--diffusion-fa`: `backend_args` in
  `/data/imagegen/imagegen.toml` is only the two placement flags.
- The two images are not identical (mean absolute difference 1.17 of 255, 93.2%
  of pixels differing by at least 1). Enabling flash attention changes the
  numerics; a parity gate has to name which configuration it compares against.

Is there room left for ds4? The fused kernel does 2 x 2 x 4096 x 4224 x 128 x 32
= 283.5 GFLOP in 8.500 ms, i.e. **33.4 TFLOP/s, or 47% of the measured dense-FP16
rate of this GPU** (70.4 TFLOP/s, section 3). A kernel reaching that measured
rate would take 4.03 ms instead of 8.500, saving 286 ms of a 2990 ms step —
**9.6%, and only by writing an attention kernel better than the upstream one.**

**M1 = no gain available to the port.** The plan's own P2 route is "vendor the
missing `ggml-cuda` ops": it would ship the same `flash_attn_ext` family the
reference already runs, so the port captures 0 by construction. The measured
headroom is 47% to ~100% of the dense rate, and closing it is new kernel work,
not a port, with a 9.6% ceiling.

## 3. M2 — Q6_K through the dense MMQ

Question: does extending the software-pipelined K loop (`cuda/mmq/ds4_mmq_pipe.cuh`,
gated at :38-40 to IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS) to Q6_K pay at the DiT's dense
shapes?

The baseline is `mul_mat_q_case<GGML_TYPE_Q6_K>` — the vendored upstream template
(llama.cpp pin `5c0e9468`), which is the same code the reference's `mul_mat_q`
runs. Both harness and trace numbers are for 4224 columns:

| Shape (M x N x K) | Production Q6_K MMQ | cuBLAS dense FP16 | stock / cuBLAS |
| --- | --- | --- | --- |
| 4096 x 4224 x 4096 (to_q/k/v/out) | 1.988 ms, **71.3 TFLOP/s** | 2.039 ms, 69.5 TFLOP/s | 102.6% |
| 12288 x 4224 x 4096 (mlp gate / proj) | 5.379 ms, **79.1 TFLOP/s** | 5.821 ms, 73.0 TFLOP/s | 108.2% |
| 4096 x 4224 x 12288 (mlp out) | 5.823 ms, **73.0 TFLOP/s** | 5.746 ms, 74.0 TFLOP/s | 98.7% |

The production Q6_K kernel **matches or beats a dense cuBLAS FP16 GEMM at the
same shapes while reading 6.5x fewer weight bytes**. The empirical bound is the
cuBLAS number itself: 69.5-74.0 TFLOP/s is what a tuned dense FP16 GEMM achieves
on this part at these shapes, and the MMQ kernel is at or above it. The observed
clocks (up to 2805 MHz against a 2475 MHz nominal boost) account for the spread.
The kernel is at the SM's tensor-core limit.

Cross-check against the reference. The fractional MMQ shapes are visible in the
trace: grid (1056,1,1) is the `M=4096` group (ntx 33 x nty 32 = 1056 for 4224
columns) and grid (3168,1,1) is `M=12288` (33 x 96). Grouping per step gives 320
launches of the first and 128 of the second, 448 launches of Q6_K dense MMQ
every step in both configurations: **1309.8 ms (703.7 + 606.1) in the default
step and 1410.9 ms in the `--diffusion-fa` step**, 47.1% of the latter. The two
configurations issue identical MMQ work, so the 8% spread between them is run
and clock variance, not a different GEMM set. The harness's eager per-block-pass
(all 7 GEMMs) is 23.185 ms, x64 block-passes = 1484 ms/step — the same number
within 5%, measured through a completely different path.

Correctness. The tree has no dense Q6_K parity test, so the harness establishes
one against the tree's own canonical Q6_K dot product (`ds4.c:4538`,
`ds4_vec_dot_q6_K_f32`, copied verbatim). With activations chosen exactly
representable in Q8_1 — which removes the activation quantization from the
comparison and makes the check decisive for the weight dequant and the MMA —
the relative RMS is **2.9e-07**, i.e. FP32 rounding. With generic activations it
is 3.8e-03, the expected Q8_1 activation-quantization floor.

**M2 = no gain.** The stock loop is already at the hardware limit; there is no
gap for a pipelined K loop to close. Extending the pipe to Q6_K is not worth
building.

## 4. M3 — graph capture, residency and the staging arena

Question: on this 12 GB card with the DiT resident, what do capture and the
offload mode cost or save?

| Measurement | Result |
| --- | --- |
| Launch overhead, 4000 empty kernels | eager 5.090 ms (1.273 us/launch) vs graph replay 2.491 ms (0.623 us). At the measured 4544 kernels/step (4736 unfused) the ceiling is **2.9-3.1 ms/step (0.1%)** |
| One DiT-shaped captured block-pass (the 7 real Q6_K GEMMs, warmed) | eager 23.185 ms vs captured 22.332 ms, i.e. 55 ms/step at the x64 block-pass count; repeated runs put the saving between **3 and 55 ms/step**, so it is the bound below, not this figure, that the decision rests on |
| Host/idle bubble in the real step (union of kernel intervals vs wall) | 164.5 ms (3.4%) unfused, **172.7 ms (5.5%)** with `--diffusion-fa` — the absolute ceiling for capture plus host-work elimination |
| Staging the whole DiT into a fixed VRAM arena | 5604 MiB pinned host to device in 240 ms = 24.5 GB/s = **8.0% of a 2990 ms step** |
| Residency | weights 5604 + compute 2317 = **7921 MiB of 11894 MiB, fits with 3973 MiB spare**; one `cudaMalloc` of the full need succeeds |

Two facts about the reference matter here. First, it issues **4544-4736 individual
kernel launches per step and uses no CUDA graphs at all** (0 of 13632 kernels in
the trace carry a graph id), so capture is genuinely unused today. Second, the
kernels are perfectly serial on the critical path — the union of kernel intervals
equals their sum, 2992.5 ms against 3165.2 ms of wall — so the 172.7 ms is the
whole non-GPU time and nothing hides behind a second stream.

**M3 = a small, bounded win.** Capture is worth at most the measured host bubble,
172.7 ms/step (5.5%), and the directly measured GEMM block-pass saving was not
stable enough to pin (3-55 ms/step). It is real and it is free, but it is an
order of magnitude below the 47% the GEMMs cost and the 18% attention costs. The
offload mode costs 8.0% of the step when the entire DiT is staged per step and
less when segmented and overlapped — it is a fallback with a price, as the plan
already says, not a source of speed.

## 5. The step, measured

One steady-state step, `--diffusion-fa`, of 2992.5 ms of kernel time (3165.2 ms
wall):

| Share | ms | Launches | What |
| --- | --- | --- | --- |
| 47.1% | 1410.9 | 448 | Q6_K dense MMQ (all 7 DiT linears per block) |
| 18.2% | 544.0 | 64 | flash attention, image segment |
| 6.0% | 178.3 | 772 | `k_bin_bcast` op_mul (modulation) |
| 4.7% | 139.5 | 516 | `cpy_scalar` |
| 4.6% | 136.8 | 256 | `k_bin_bcast` op_add |
| 4.5% | 133.7 | 450 | `quantize_mmq_q8_1` (activation quantization for MMQ) |
| 3.8% | 112.7 | 322 | `concat_cont` |
| 3.4% | 101.3 | 64 | SiLU-gated MLP |
| 2.1% | 62.0 | 256 | `k_bin_bcast_unravel` op_repeat |
| rest | 173.2 | | norms, scales |

65.3% of the step is kernels whose ds4 counterparts are the same upstream code
(MMQ and `flash_attn_ext`). The remaining 34.7% is elementwise glue, where ds4
has plausible wins from fusion that the plan has not scoped and this spike did
not measure.

## 6. Corrections to the plan and the roadmap

Both are load-bearing, so they are recorded rather than quietly fixed.

1. **The DiT's footprint is ~7.9 GB, not ~11 GB.** The plan said so twice, in
   appendix A.1 ("the DiT at Q6_K is 297 tensors and about 11 GB") and again in
   its section 14 risk list ("the image stack peaks around 11.5 GB on the same
   12 GB card"). The roadmap never carried a size; an earlier draft of this
   report attributed the claim to it and that attribution was wrong. Measured:
   5604.32 MB of weights plus a 2317.45 MB compute buffer, reported by the
   reference's own model manager and reproduced by the harness's allocation of
   the full need. The ~11 GB figure is the whole three-module stack (text encoder
   4.8 GB + DiT 5.6 GB + VAE 0.6 GB, which is what the risk list's 11.5 GB is
   about); on the deployed placement the encoder and VAE are on the CPU, so only
   5.6 GB of weights are on the device. The 12 GB card has ~4 GB of headroom with
   the DiT fully resident, which is a materially easier constraint than the plan
   assumed. Both plan passages are corrected in place.

2. **The deployment runs flash attention off**, and the plan's premise that
   attention runs as a "vendored masked-attention core" at a good rate is not
   what is deployed. This is outside the port and is the single largest measured
   performance change available on this stack: an operator flag worth 1.65x on
   the step, with no code.

3. The plan's applicability analysis in section 4 is otherwise confirmed by
   measurement: the MMQ pipe does not route Q6_K, MMVQ has no work at these
   widths, and there is no KV. The one thing it could not have known is that the
   dense MMQ is already at the tensor-core limit.

## 7. Decision

The roadmap's Phase S rule: *if none of the three shows a useful gain, the
performance case is dead, the port proceeds only on ownership grounds, and this
document says so instead of implying otherwise.*

Applying it:

- M1: 0% available to the port through its planned route; 9.6% ceiling only by
  writing a better attention kernel than upstream's.
- M2: 0%; the kernel is at the hardware limit and there is nothing to close.
- M3: capture is real but bounded by the 5.5% host bubble; the offload mode costs
  8% when fully staged.

**The performance case for the port is dead.** The port is justified, if at all,
by the ownership argument in the plan's section 3 — one process, one memory
policy, one weight path, no third-party runtime, artifact control — and by the
34.7% of the step that is elementwise glue, which ds4's fusion machinery might
reduce but which no phase of the plan currently owns. No later phase may claim a
speed motive without its own measurement.

Recommendation, in the plan's own terms: do not start P0-P6 on performance
grounds. If the port is wanted for ownership, the first thing to scope is a
fusion pass over the elementwise glue, because that is the only measured
unclaimed headroom left; and the roadmap's M1/M2 items should be marked as
settled rather than open.

Resolved 2026-10-01: the [decision](qwen-image-2.1-decision.md) scoped that
fusion pass at 250-450 ms/step and closed the port, because it is below what the
free `--diffusion-fa` flag already returns. M1/M2 are recorded as settled in the
roadmap's tracker.

## 8. What this does not prove

- A microbenchmark is not the whole workload. The 7-GEMM block-pass is not 32
  blocks with the elementwise glue and the cross-block dependencies; the
  in-situ trace numbers are the workload, and they are the ones the decision
  rests on.
- Clocks were not locked. All comparisons are same-session at an observed
  2565-2805 MHz band. A clock-locked re-run is the honest way to firm up the
  M2 ratios if they are ever contested.
- The Q6_K correctness evidence is relative-RMS against the tree's canonical dot
  product, not a full dense Q6_K parity test — the tree does not have one. The
  exact-activation case (2.9e-07) is the decisive part.
- The 4224-column width is the reference's own trace width for this prompt; a
  different prompt changes the text length and the shape slightly.
- Attention TFLOP/s is derived arithmetic (FLOPs over measured kernel time), not
  a counter reading.
- The offload-mode staging cost is the unsegmented worst case (the entire DiT
  every step). The plan's segmented, overlapped version is cheaper by an amount
  this spike did not measure.
- Nothing here measures the VAE decode, the text encoder, or any end-to-end
  pipeline: the reference's CPU VAE costs 76.93 s per image and dwarfs the
  sampling in the deployed configuration, and this spike did not touch it.

## 9. Evidence index

Everything below lives in `/data/ds4-dfm-rs/misc/scratch/phase-s/` on this host.
`misc/` is gitignored, so the harnesses are not in git; the numbers in this
document are.

| Path | What it is |
| --- | --- |
| `run-ref.sh`, `run-ref-fa.sh` | The two reference invocations, exact flags |
| `ref-run.log`, `ref-fa.log` | Reference stderr: step times, buffer sizes, phases |
| `ref-nsys.nsys-rep`, `ref-fa-nsys.nsys-rep` (+ `.sqlite`) | Nsight Systems traces of both runs |
| `kern_sum.txt`, `fa_kern_sum.txt` | `nsys stats` kernel summaries |
| `sd-cli-help.txt` | The reference binary's flag surface |
| `artifact-hashes.txt` | SHA-256 of the three artifacts |
| `out-ref-s3.png`, `out-ref-fa-s3.png` | The two 1024x1024 images |
| `m2/bench_q6k.cu`, `build.sh`, `bench_q6k.out` | Q6_K dense MMQ harness, its CPU oracle and its output |
| `m3/bench_graph.cu`, `build.sh`, `bench_graph.out` | Capture, residency and staging harness and its output |
| `m2/clocks.csv`, `m3/clocks.csv` | SM clock samples taken during both harness runs |
| `ref/qwen_image_2_1.hpp`, `ref/ggml_extend.cpp` | The reference sources read for the attention path |
| `../../qa-evidence/qa-report.md` (tracked) | The independent QA passes and their verdict |

Artifacts measured (SHA-256 in `artifact-hashes.txt`):
`qwen-image-2.1-Q6_K.gguf` 5876556448 bytes,
`qwen_image_2.1_vae_bf16.safetensors` 675509688 bytes,
`Qwen3VL-8B-Instruct-Q4_K_M.gguf` 5027784800 bytes.
