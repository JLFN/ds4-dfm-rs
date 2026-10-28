# Porting Sol-H3-Spark: what the source gives, and in what order

Read of the reference at `/data/Sol-h3-Spark` (NVlabs/Sana, branch `sol-engine`
@ `6e4a440`) for branch `feat/sol-h3`, on 2026-10-28. This file answers one
question: what does the reference's source contain that this engine can port,
and against which of this engine's own measured phases does each piece land.

Everything below is read from the reference source or from this repository's own
records. Where a number is an estimate rather than a measurement it says so.

## 1. The reference, as its own recipe states it

`models/minimax_h3/Sol-H3-Spark/configs/default.json` is the frozen recipe
(`runtime/config.py:21-25` hashes it and refuses a modified copy). Its own keys:

| Stage | Setting, as written |
|---|---|
| stage1 | 672x384, 124 frames, `qwen: NVFP4_AWQ_resident`, `dit: W8A8_FP8_after_BF16_LoRA_merge`, `lora: FastH3_VSA_DataFree` strength 1.0, `updates: 4`, `video_sigmas [1.0, 0.973, 0.923, 0.8, 0.0]`, `attention: VSA_cuDNN_BSA`, `sparsity: 0.9`, `tile_size: 64`, `video_decode: false` |
| handoff | CPU tensor file, `upscale_before_adapter: true`, `upscale_ratio: 2.0`, `refiner_input_shape [1, 128, 16, 24, 42]` |
| stage2 | `transformer: LTX-2.5_dev_BF16`, `lora: distilled_lora_450` strength 0.8, `updates: 3`, `attention: Triton_Sol`, `taus [1.0, 1.25, 1.5]`, `threshold: diag`, `dense_layers [0]`, `decoder: official_Conv_VAE`, tiles 448x768x128 with 64/24 overlaps, `direct_nhwc: true`, `writer_chunk_frames: 16` |
| output | 1344x768, 121 frames, 24 fps, H264 crf 19, AAC 32 kHz |

Four facts follow from it that the earlier recon did not pin down:

1. The reference's **H3 stage is not the Sol-Attn stage**. Stage 1 uses VSA at
   90% sparsity on cuDNN's BSA backend; Sol-Attn is stage 2's attention, on the
   LTX refiner (`default.json:12` against `:30`). The task switch confirms it:
   `runtime/config.py:50-63` only gives stage 1 the `FA4_Sol_text_audio_sink`
   policy for `ref2va`, and pops `sparsity` and `tile_size` when it does.
2. The step counts are **4 (H3 draft) and 3 (LTX refiner)**, and both come from
   distilled adapters, not from a scheduler trick (`updates` plus the LoRA keys
   in both stages).
3. Their Sigmas are fixed per update (`stage1.video_sigmas`, `stage2.sigmas`),
   so both stages are a fixed-step distilled schedule rather than a
   configurable sampler sweep.
4. The pixel-space round trip is removed on purpose: stage 1 sets
   `video_decode: false` and the latent goes through an x2 learned upscaler and
   an adapter before the refiner (`runtime/latent_transfer.py`,
   `runtime/stage2_ops/h3_upscale.py`). `runtime/stage2.py:258` renames the
   phase `official_input_video_vae_encode_s` to `h3_upscale_and_adapter_s`, and
   its own receipt asserts `input_video_vae: 0` calls (`:262`).

The reference does **not** claim a throughput number in the checkout: its README
attributes "56 s hot E2E" to the project page and adds "This is not a throughput
or cold-start claim" (`README.md:236-239`). The 56.17 s figure this campaign
carries in its handoff comes from that page by way of the recon, and cannot be
re-derived from the local tree.

## 2. Why 56 s is not a ratio over our wall

Our nearest recorded job is 768x768, 107 frames, two evals, one binary, both
arms: 821.994 s scalar / 277.019 s tensor (`speed-2026-10-27.md:520-537`). The
reference's job is a different canvas, frame count, step count, model set and
decoder, so no quotient of the two means anything. What is comparable is each
side's own split of its own wall. Ours, tensor arm:

| Phase | Job A (768x768, 107 frames, 2 evals) | Share |
|---|---:|---:|
| TEXT | 27.533 s | 9.9% |
| LOAD | 7.160 s | 2.6% |
| DENOISE | 40.294 s | 14.5% |
| AUDIO | 2.577 s | 0.9% |
| FRAMES (video VAE decode) | 197.945 s | 71.5% |
| VIDEO | 0.623 s | 0.2% |
| Total | 277.019 s | |

FRAMES is the video VAE decoder (`speed-2026-10-27.md:173`). So on the job this
campaign measures, the term that decides the wall is our own decoder, and
neither Sol-Attn nor the LoRA touches it. On a 20-step job the balance moves the
other way: 20 evals of the same step cost ~403 s against FRAMES' ~198, which is
why the denoise levers matter more there.

## 3. The levers, ranked against our measured phases

| # | Lever | Phase it touches | Our size | What it needs | Basis |
|---|---|---|---|---|---|
| L1 | VAE decode: tensor-core attention + BF16 GEMM | FRAMES | 197.9 s | code already written on `feat/h3-vae-options`; a numerics decision | **measured**: FRAMES 196.873 -> 34.494 s (`vae-options-2026-10-28.md:175-181`) |
| L2 | Few-step distilled LoRA | DENOISE | 40.3 s (2 evals) / ~403 s (20) | 5.34 GB FastH3 VSA DataFree LoRA plus a merge path | their recipe; our 5x is an extrapolation from their 4-step schedule |
| L3 | FP8 W8A8 weights | DENOISE | as L2 | a quantizer or a pre-quantized checkpoint | their recipe (`W8A8_FP8_after_BF16_LoRA_merge`) |
| L4 | Text encoder to NVFP4 | TEXT | 27.5 s | new weights (their Qwen is NVFP4-AWQ, 15.69 GB) | unmeasured here |
| L5 | Sparsity: Sol-Attn (stage 2) or VSA (stage 1) | DENOISE attention | ~17.3 s of the 40.3 s, **estimate** (50 body layers x 173.163 ms per call x 2 evals) | code only | our search step is measured at 2.8% of dense (`speed` record); their 3.1x is against FA4 |
| L6 | Latent transfer and a second stage | new pipeline | - | gated LTX-2.5 (42.02 GB transformer, 8.90 GB LoRA) | not a port; a second engine |

L1 is the largest and the cheapest, and it is ours rather than the reference's:
their decode is a different decoder (LTX `ConvVideoDecoder`,
`stage2_ops/models.py:116-128`) on tiles of 448x768x128 with `direct_nhwc`
pixel shuffles and a 16-frame streaming writer, so their decode's 12.5% share
does not transfer as a technique. What does transfer is the reuse of H3's own
decode arm options already built here.

L5 is the piece this branch started. Its consumer does not exist yet, so the
estimate above stands on our measured dense call and the reference's claimed
ratio, not on our own measurement. The layer count is not assumed: our DiT has
50 blocks (`crates/ds4-core/src/minimax_h3/dit.rs:102`, mirroring the
reference's `h3_dit.c:15-30`), the same count the reference's own H3 stage
asserts (`stage1_ops/sol.py:108`).

## 4. The Sol-Attn algorithm, as the reference defines it

Read first-hand for the port (`techniques/sparse_backends/sol_attn/`):

- Blocks are 64 tokens, query and key alike (`triton_ref/fwd.py:20`), and the
  forward walks the key blocks in groups of 32 (`:21`). One program handles one
  64-row query block of one head, with a value tile of 128 or 64 (`:206-210`,
  `:179-183`).
- `kc` is the block **mean** of K (`preprocess.py:45`) and `vc` the block
  **sum** of V (`preprocess.py:82`), both stored in the input dtype.
- The per-head statistics are `kc_mean` and `kc_var_diag`, the mean and variance
  of the block centroids over blocks (`preprocess.py:96-132`).
- The `diag` threshold, which is what the runtime asks for
  (`stage1_ops/sol.py:137`), is built from the query block's **F32 centroid**:
  `q_centroid = sum(q, axis=0) / q_len` in FP32 (`preprocess.py:166`), then
  `mean = sum(q_centroid * kc_mean) * log2_scale` and
  `threshold = mean + TAU * sqrt(max(sum(q_centroid^2 * kc_var) * log2_scale^2, 0) + 1e-6)`
  (`preprocess.py:178-187`), stored per query block per head.
- The routing predicate inside the kernel is
  `(sum(scores, axis=0) / q_len > threshold) | (abs(q_block - block) <= 1)`,
  with the sink range OR'd in when there is one, and `& valid`
  (`triton_ref/fwd.py:96-104`). `scores` is `q @ kc^T * scale * log2(e)`.
- **Approximated blocks are not skipped: they are folded into the same online
  softmax from the pooled summaries** (`fwd.py:109-131`). Their probability is
  `exp2(scores - new_max)` computed from the centroid scores, the output gains
  `P @ vc`, and the running denominator gains `P * block_length`, because `vc`
  is a sum over the block. This is the part a port must get right: a skipped
  block still contributes mass to the softmax denominator.
- Exact blocks are then walked one at a time as ordinary FlashAttention blocks
  against real K and V (`fwd.py:132-170`), in the same running-max form.
- The output is `output / row_sum` in the input dtype (`fwd.py:171-173`).

Two consequences for this engine:

1. The reference computes routing **inside** the attention kernel and never
   materialises a mask (README: "avoiding a materialized routing map"). This
   branch's search step materialises one (`plan->mask`, `plan->count`). That is
   a named deviation: it costs the 4.924 ms measured search per call and buys a
   consumer that does not have to recompute the predicate. It is also what lets
   the search step be tested against a CPU oracle on its own.
2. The `diag` threshold needs the query centroid, which is why the plan carries
   `query_bar` (`native/h3/h3_dit_gpu.h:65-74`); the consumer needs `kc`, `vc`,
   `mask` and the sink range, and no threshold.

## 5. What the source does not give us

- **The VSA kernel.** Stage 1's 90%-sparsity attention is the reference's H3-side
  choice, and its implementation is not in this checkout: `stage1_ops/vsa.py`
  imports `fastvideo` and pins it by commit (`vsa.py:12`) and by a source sha256.
  What is here is the policy: `sparsity 0.9`, `tile_size 64`, a text/audio
  prefix kept exempt, `topk = ceil((1 - 0.9) * num_video_tiles)` selected per
  query (`vsa.py:45-57`), all 50 layers and all 4 steps
  (`vsa.py:107-118` asserts 200 calls per request). Its kernel is a dependency
  read, not a file here.
- **A portable decode technique.** Their decoder is LTX's, the checkpoint is
  gated, and its 12.5% share is a property of a 1.45 GB conv VAE whose weights
  we would not use.
- **Anything for our FRAMES phase.** See section 2: the source has no lever for
  the term that is 71.5% of our measured wall.

## 6. Port order

1. **Land the measured VAE arm's wiring decision** (L1). The measurement exists;
   what is missing is the decision to make a numerically different decode the
   default or to keep it an option. This is worth more than everything else on
   this page for job A and needs no weights.
2. **Finish Sol-Attn** (L5): the sparse forward, the consumer of the mask the
   search step already produces. It is self-contained, needs no new weights, and
   is the only lever here whose cost is code alone.
3. **Then the weights levers** (L2, L3, L4) as the operator's artifact decisions:
   a distilled LoRA, FP8 W8A8, and an NVFP4 text encoder. Each needs an artifact
   this repository does not have, and the H3 revision the reference pins
   (`9bfb6693...`) differs from the one this engine serves (`42ed227e`).

L6 is a separate-engine project and stays out of the port.

## Citations

- Reference recipe and keys: `models/minimax_h3/Sol-H3-Spark/configs/default.json:1-42`,
  `runtime/config.py:21-63`.
- Artifact manifest with sizes, sha256s and licences:
  `configs/checkpoints.json` (`vsa_lora` 5,339,117,712 B; `transformer` 42,018,190,584 B
  gated; `refiner_lora` 8,899,889,568 B gated; `qwen_checkpoint` 15,687,142,551 B;
  `h3_upscaler_checkpoint` "No explicit license declared at pinned revision").
- Sol-Attn algorithm: `techniques/sparse_backends/sol_attn/triton_ref/fwd.py:20-173`,
  `preprocess.py:45,82,96-132,166,178-187`, `interface.py:134-142`,
  `stage1_ops/sol.py:12-18,120-149`.
- VSA policy: `runtime/stage1_ops/vsa.py:12,24,45-57,107-118`.
- Our phase table and the FRAMES identification: `speed-2026-10-27.md:173,520-537`.
- Our VAE arm measurements: `vae-options-2026-10-28.md:56-60,143-146,175-181`.
- Our search-step measurement: `docs/minimax-h3/sol-search-step-2026-10-28.md`.
