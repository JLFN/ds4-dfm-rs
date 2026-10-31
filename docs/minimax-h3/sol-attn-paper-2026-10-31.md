# The Sol-Attn paper, and what the port does not have (2026-10-31)

Source read first-hand: arXiv 2607.24027, NVIDIA, "Sol-Attn: Accelerating
Video Generation Inference via On-the-Fly Attention Sparsification" (2026-07-28),
held at /tmp/arxiv-2607.24027.pdf. The reference tree is
/data/Sol-h3-Spark, revision on disk today. Everything below cites one or the
other; nothing is quoted from the page or from a prior note.

## 1. The FP8 stage 1 measured (Phase 0 of the plan)

One command, one prompt, one seed, 672x384x124 into a 1344x768 clip, stage 1
with `H3_DIT_GEMM=fp8`, everything else as the BF16 arm ran it.

| phase | BF16 (8718b3eb) | FP8 (d428fec8) |
|---|---:|---:|
| TEXT | 15.412 | 13.300 |
| LOAD | 39.167 | 8.226 |
| DENOISE (4 steps) | 21.810 | 14.971 |
| AUDIO | 2.964 | 2.939 |
| H3 VAE decode | 65.209 | 63.765 |
| WALL | 150.202 | 108.526 |

Output: 5,167,820 B, 5.175 s, H.264 1344x768 + AAC 32 kHz stereo, sha256
`d428fec8b921a407942df56bec2c9e73f23ed5d4e98bcf5a0016683f12154dc6`, decode
clean. Sharpness at 2.0 s, the record's own measure (variance of the 8-bit
Laplacian, calibrated this session: it reproduces 14.11 on the BF16 clip and
17.52 against the recorded 17.54 on the direct one): **20.64** against 14.11.

Denoise falls 31.4%, matching the 29.3% per-evaluation saving the FP8 record
measured at 768x768x107; LOAD falls 79% because the loader converts each block
to E4M3 and drops the BF16 tensor, as the release drops its parameter.

Anatomy: the FP8 arm's 0.5, 1.0, 2.0 and 3.5 s frames each hold one fox with a
visible head and a single tail. The BF16 arm's buried head at 0.5 s does not
recur, so that defect was a property of that sample, not of the upscale and not
of the sparsity policy (section 5).

The phase line also states the realized mask: `sparsity=0.900 prefix_blocks=7
first call: 9753 rows, 56 heads, exact share 0.1830` - 18.3% of blocks computed
exactly, 81.7% folded from the pooled summaries.

## 2. What the paper adds, and what of it we already run

The paper is the method behind the release's stage-2 sparsifier. Its own
contribution is the pairing of threshold routing with approximate correction in
one online-softmax pass; every piece of that pairing is in this port.

| paper | release | ours |
|---|---|---|
| Eq. (3): proxy score from pooled keys, block means | `techniques/sparse_backends/sol_attn/preprocess.py` | block mean of K and block sum of V, `native/h3/h3_dit_gpu.cu:6261` |
| Eq. (4): per-query threshold mu + beta * sigma | `_diag_threshold_kernel`, `preprocess.py:143` | `h3d_sol_threshold_kernel`, `:6313`; log2-space, 1e-6 floor, same three steps |
| Eq. (5) and App. B: diagonal estimator from pooled-key first and second moments | `_pool_key_kernel` var_diag | `h3d_sol_stats_kernel`, `:6288`, read by the threshold kernel |
| Eq. (8)-(10): zeroth-order contribution of unselected blocks to numerator and denominator | `triton_ref/fwd.py:109-170` | `h3d_sol_forward_kernel`, `:6815`; P @ vc and P * block_length |
| 64-token blocks, 32-block groups, sink block range | `BLOCK_SIZE = 64`, `_sink_block_range` | `H3D_SOL_BLOCK = 64`, `H3D_SOL_GROUP = 32`, sink span in `:6365` |
| beta per step | `configs/default.json` stage2 `taus: [1.0, 1.25, 1.5]`, `threshold: diag`, `dense_layers: [0]` | `TAU_SCHEDULE`, layer 0 dense, `crates/ds4-core/src/minimax_h3/ltx_stage2.rs:215-219` |
| stage 1 stays VSA top-k, tile 64, prefix exempt, no warm-up | `runtime/stage1_ops/vsa.py:19-38`: `dense_layers: []`, `dense_first_n_steps: 0` | `H3_DIT_SOL` = 0.9, prefix exempt, every layer, every step |

So the paper does not describe something we forgot to port for the stage that
uses it. What it does is bound how much that stage can ever give us.

## 3. What we do not have, in order of measured size

1. The Sol-Engine stack around the kernel. The paper's 5.08x on HunyuanVideo is
   kernel fusion 1.11-1.21x, then diffusion step cache 2.59-2.64x, then
   Sol-Attn (section 4.4, Figure 7). We run neither of the first two. The port does carry
   the release engine's own step cache as `--core-reuse` (1 exact, 4 fast, 6
   aggressive; `crates/ds4-cli/src/bin/h3-rs.rs:1128`), and every render so far
   ran it at 1. The direct route spends 502.9 s of its 630.3 s in denoise, so
   this is the only untested lever of that size. It changes numerics, so it is a
   unit of its own, not a flag flip inside another one.
2. Fused routing. The paper routes inside the online-softmax pass, per pooled-key
   chunk, and materialises neither the proxy map nor the mask (section 3.2,
   Algorithm 1, Figure 4). Ours is five launches: pool, statistics, threshold,
   route, forward, with a bitmask in HBM between the last two. Measured at
   18,432 x 56 x 128 on the Spark: pool 2.493 ms, statistics 0.008, threshold
   1.323, route 0.852, total 4.924 ms against 173.163 ms of dense attention
   (`9b2bdc48`); at 768x768x107 the routing passes are 407 ms of a 7,977 ms
   evaluation, 5.1%. The paper's own end-to-end comparison puts the separate
   routing stage at 6.2% of a cuDNN BSA run (Figure 10). Worth low single
   digits, not the gap to 56 s.
3. Sequence length. Our stage 1 is 9,753 tokens and stage 2 is 17,136; the
   paper's whole sweep is 16K to 128K and its 5.41x kernel figure is at 128K
   with 90% sparsity (Figure 5a, H100 against FA3). At 9.7K tokens we are below
   the range in which the kernel multiple is measured at all, and stage 2 sits
   at its bottom edge. Stage 1's attention is also small against its
   projections, which is why the release's own 35.2% stage-1 share survives at
   0.9 sparsity.
4. One unnamed deviation of ours. `h3d_sol_route_kernel` always keeps the +-1
   neighbouring blocks (`native/h3/h3_dit_gpu.cu:6432`, `distance <= 1`).
   Eq. (4), `triton_ref/fwd.py` and the CuTe kernels keep only sinks and
   threshold-selected blocks. The cost is a few exact blocks per row out of 267;
   the correctness cost is that the port is denser than the release for a reason
   no source states. Name it or delete it.

## 4. The hypothesis the paper kills

`runtime/stage1_ops/sol.py` is a Sol policy with `TAUS = (None, 1.0, 1.25, 1.5)`
and `route(step, layer) = None if step == 0 or layer == 0`: step 0 dense,
layer 0 dense, and a receipt that demands 147 Sol calls and 55 dense ones. Read
alone it says our stage 1 is missing its warm-up, and a dense first step in a
four-step render would be a plausible cause for a broken draft.

It is not applicable. `runtime/stage1.py:277` refuses that policy for any
task other than ref2va, and the T2VA path takes `vsa.configure_profile`
(`runtime/stage1.py:293`), whose contract is `dense_layers: []`,
`dense_first_n_steps: 0`, `prefix_mode: "exempt"`, sparsity 0.9, tile 64
(`runtime/stage1_ops/vsa.py:19-38`, enforced at `:44-46`). The release config
agrees: `stage1.attention` is `VSA_cuDNN_BSA`, not `Triton_Sol`. Our stage 1 is
policy-exact, and the draft's anatomy is a step-count and conditioning question
(Phase 1), which the FP8 arm's frames support.

## 5. What the paper does not explain

For LTX 2.3 it reports 89.83% sparsity and 2.4x stage-2 speedup at
PSNR 22.24 / SSIM 0.8704 against dense (Table 1). That is the whole claim that
applies to our refiner, and it is measured on H100 against FA3. The paper does
not state the Spark's stage-2 configuration or wall time, so the page's 24.71 s
for three 22B BF16 passes remains 4.5 TB/s of weight traffic and unexplained.
Phase 4 stands: one nsys profile of one refinement step before any refiner work.

## 6. Next

Order unchanged. Phase 1 (draft quality: steps, then a structured prompt) is
still the quality fix and the paper removes the warm-up explanation. After it,
`--core-reuse` at 2 and 4 on the direct route is the cheapest measurement of
the class the paper credits for 2.6x, and it needs its own acceptance on
content across the clip.
