# The Sol-H3-Spark pipeline: what it takes to run it here

Goal: our engine produces the setup's product — 1344x768, 5 s, 121 frames, with
audio — at its speed and quality, on one DGX Spark. Today we produce 768x768,
107 frames, 88.99 s hot, with no refiner and with a 34.1 s H3 VAE decode in the
critical path.

Sources read first-hand for this plan: the release page
(`nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark`), the pinned checkout
`/data/Sol-h3-Spark` @ `6e4a440` (`configs/default.json`,
`configs/checkpoints.json`, `configs/dependencies.json`, `docs/setup.md`,
`docs/validation.md`, `runtime/stage2.py`, `runtime/stage2_ops/models.py`), and
the safetensors headers of the four LTX files the operator added
(`~/ltx-structure.txt`, 1912 lines).

## 1. What the setup is

Stage 1 drafts 672x384x124 through MiniMax-H3 with W8A8 FP8 weights, the
four-step FastH3 VSA LoRA and VSA attention at 90% sparsity in 64-token blocks,
Qwen NVFP4-AWQ resident, and no video decode at all (`stage1.video_decode` is
false). Then a learned 3D latent upscaler doubles the latent to 48x84, a learned
adapter maps it to the refiner's space (crop to 16 latent frames), LTX-2.5
refines for three updates against a cached fixed prompt, and the official
convolutional VAE decodes to pixels. Their split of 56.17 s: Qwen 2.4%, H3 35.2%
(19.77 s), upscaler + adapter 3.8% (2.13 s), LTX 44.0% (24.71 s), LTX VAE decode
12.5% (7.02 s), mux 1.3% (0.73 s), overhead 0.9% (0.51 s).

Four properties make it fit one Spark, and each is a change from what we run:

- No inter-stage pixel round trip: the H3 VAE decoder and the LTX VAE encoder
  are both absent ("24 s saved" on their page). Our 34.1 s `FRAMES` term is
  exactly that decode.
- The refiner's prompt is one content-agnostic sentence, encoded once offline
  through Gemma and the connector, cached; Gemma is never resident and stage 2
  does no text encoding. Stage 1 still encodes the real prompt with Qwen per
  request.
- Low-bit weights: NVFP4 Qwen (we have it), FP8 H3 DiT (we do not), and the
  LTX side loaded BF16.
- 116.9 GiB resident of the 119.68 GiB pool, so nothing is constructed per
  request.

## 2. What we already have, and what is missing

Present and running: the H3 DiT at four steps, the FastH3 VSA LoRA merged
artifact (`MiniMax-H3-FastH3-4step`), our VSA implementation at sparsity 0.9,
the NVFP4-AWQ Qwen encoder (`--text-nvfp4`), the H3 video VAE decoder
(tensor-core, BF16 GEMM), the audio VAE, the muxer, the CLI and the server with
per-phase reporting.

Added by the operator and byte-count-matched to `checkpoints.json`: the LTX-2.5
22B dev transformer (42,018,190,584 B BF16), the distilled LoRA 450
(8,899,889,568 B), the Conv video VAE (1,452,269,922 B), the audio VAE
(364,866,540 B).

Still needed, from `configs/checkpoints.json`:

| artifact | bytes | why |
|---|---:|---|
| H3 latent upscaler (`minimax_h3_latent_upscaler_3d_bf16`) | 690,592,992 | the 2x latent upscale before the adapter |
| H3-to-LTX adapter (`model.safetensors` + `config.json`) | 389,541,112 | maps the upscaled H3 latent into LTX space |
| offline Gemma 4 12B int8 (`gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot`) | 15,372,969,374 | one-time: builds the cached refiner prompt |
| offline connector int8 (`ltx-2.5-22b-dev-transformer-comfy-int8-convrot`) | 21,504,034,224 | one-time: same |
| the prompt cache itself | small | `runtime` validates its recipe and tensors |
| an FP8 H3 DiT checkpoint | ~15.5 GB | stage 1 is W8A8 FP8 in their recipe |

The LTX-2 model source is already in the pinned tree, outside the checkout
cone: `models/ltx25/GB200/environment/LTX-2/`, upstream `Lightricks/LTX-2`
release v1.2.0 at commit `7954dcb0d986bdc36ef272564a9789ade07fcc65`
(`models/ltx25/GB200/SOURCE_SNAPSHOT.json`), 263 files across `packages/ltx-core`
(160), `ltx-kernels` (57) and `ltx-pipelines` (46), with its `uv.lock`,
python 3.13.11, torch 2.11.0+cu130 and triton 3.6.0. The Spark profile pins
upstream `d1511477` instead (`configs/dependencies.json`), so the local copy is
a sibling snapshot: port against it, diff the block forward against the Spark
pin, and fetch that pin only if the two differ.

Two more local references: `models/ltx2.5-refiner/GB200/` (their refiner driver,
`refiner.toml`, `sol_attention.py`, `refiner_head_cp.py`) and
`models/minimax_h3/super_acceleration/` — the v2 end-to-end lineage, whose
`STAGE2_CONTRACT.md` freezes the algorithm (48 layers; video self-attention
dense in layer 0 and strict Sol-Attn in layers 1-47 with taus 1.0, 1.25, 1.5;
cross-attention and audio attention dense; three updates at
`[0.909375, 0.725, 0.421875, 0.0]`; H.264 1344x768 121 frames 24 FPS plus
stereo AAC 32 kHz) and whose `BENCHMARK_REFERENCE.json` carries their GB200
formal run (stage-2 service median 2.447 s on a GB200, a different silicon
and profile from the Spark number).

Read those paths with `git show HEAD:<path>` or a `git archive` extraction into
a scratch directory; the pinned checkout stays untouched. The only source we
still need from outside is the H3 upscaler node,
`LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler` @ `d7c01b90...`
(`nodes/minimax_h3_latent_upscaler_3d.py`, sha256 `744063b4...`).

## 3. The architecture we are porting

From the header dump (48 blocks, 4,349 tensors, 4,059 BF16 + 290 F32):

- Video stream 4,096 wide, audio 2,048, 32 heads of 128, per-head gate logits
  (`to_gate_logits` is `(32, width)`) — the learned gating LTX-2.5 uses for
  sparse refinement.
- Per block: video self-attention (`attn1`), video cross-attention (`attn2`),
  audio self- and cross-attention, the two cross-modal attentions
  (`audio_to_video_attn`, `video_to_audio_attn`, each with its own
  `to_gate_logits`), both with q/k RMS norms per head; video FF 4,096→16,384→
  4,096 and audio FF 2,048→8,192→2,048 (SwiGLU); AdaLN tables of 9 rows
  (per block), 5 rows (`scale_shift_table_a2v_ca_video`/`_audio`, the AV cross
  attention), and 2 rows (prompt).
- Outside the blocks: per-stream timestep embedders (256→width→width), the
  `adaln_single` projections (36,864 = 9x4096, 18,432 = 9x2048, 16,384, 8,192),
  the AV gate AdaLN singles, `patchify_proj` (4,096, 128) and `proj_out`
  (128, 4,096), `keyframes_abs_pos_embedding`, and the
  `video_embeddings_connector` with eight 1D blocks (4,096 wide, FF 16,384) plus
  its audio counterpart — the second-stage connector behind the cached context.
- The LoRA: 3,320 tensors, rank 450 on the main linears and 256 on the
  timestep embedders, applied at strength 0.8.
- The Conv VAE: 170 tensors, three up blocks, NHWC direct; the audio VAE: 1,329
  tensors, four up blocks.

Nothing here is a primitive we do not already run for H3: BF16 GEMMs, SDPA
variants, AdaLN, RoPE, patchify, RMS norms. What is new is the two-stream joint
attention with per-head gates, the six attention modules per block, and the
volume (48 blocks at 22B parameters).

## 4. Phases

Each phase ports, then measures the same way our H3 stage is measured now
(Nsight Systems capture, per-kernel table, each entry against its own floor),
then optimizes the top entry, then re-measures end to end. A phase is done when
its own record carries the port, the table and the gate.

| # | Phase | Port | Measure | Gate |
|---|---|---|---|---|
| P0 | Reference and artifacts | extract the local vendored LTX-2 source and the Spark runtime's stage-2 ops; write the weights map | tensor inventory (done), parameter and FLOP counts per phase | the map predicts each stage's share; mismatches are named |
| P1 | Cached prompt | build it once offline with their recipe (int8 Gemma + int8 connector), store the two context tensors as an artifact | one-time only, no runtime cost | shapes match the connector's output; the source's own bitwise check (`docs/validation.md` §"Single-Spark T2VA") |
| P2 | Latent transfer | x2 3D upscaler + adapter + crop to 16 frames, from the pinned node source | their 3.8%, 2.13 s | our latent in, refiner-shaped latent out, against their recorded conventions |
| P3 | LTX-2.5 refiner | 48 dual-stream blocks, joint AV attention with per-head gates, both timestep paths, patchify/proj_out, the connector's eight 1D blocks, LoRA merge at 0.8, the three-update sampler `[0.909375, 0.725, 0.421875, 0.0]` and the tiles (frames 128/24, height 448/64, width 768/64) | their 44.0%, 24.71 s for three updates | a runnable reference (P0's golden decision) or the port is unverified |
| P4 | Decoders and mux | Conv VAE (direct NHWC, 16-frame chunks), audio VAE, H.264 crf 19 veryfast + AAC 32 kHz stereo | their 12.5%, 7.02 s | 121 frames out, audio muxed, same container contract |
| P5 | Pipeline and serving | one resident session, stage 1 with `video_decode` off, transfer, refiner, decode, mux; per-phase report lines | hot end-to-end against 56.17 s, resident against 116.9 GiB | three consecutive requests, one warmup, no per-request construction (their integration list) |
| P6 | Performance | stage 1: FP8 DiT, GEMM efficiency, `qkv_rope`, AdaLN, VSA; stage 2: the joint attention, the gate path, the CFG-free single pass; both: CUDA graphs | per-kernel tables per phase, each against its floor | each adopted change carries its own before/after and no quality regression |
| P7 | Quality | — | 1344x768 output for the fixed prompts, detail energy and paired comparisons | the merge record's instrument must separate detail from noise before any adoption |

## 5. What is already known about stage 1's cost shape

The Phase B capture on our current job A (18,799 rows, four evaluations, SOL
0.9) gives the per-evaluation kernel table, and it re-ranks the levers:

| kernel | ms/eval | calls | share |
|---|---:|---:|---:|
| `cutlass_80_tensorop_bf16_s16816gemm_relu_...128x256` | 7,102.0 | 150 | 57.5% |
| `nvjet_sm121_tst_mma_128x240x64_tmaAB` | 799.0 | 50 | 6.5% |
| `h3d_sol_forward_kernel` (sparse attention) | 1,311.6 | 50 | 10.6% |
| `h3d_qkv_rope_kernel` | 1,031.0 | 50 | 8.4% |
| `h3d_adaln_kernel` | 1,013.5 | 102 | 8.2% |
| `h3d_swiglu_kernel` | 383.9 | 50 | 3.1% |
| `h3d_gate_kernel` | 289.2 | 100 | 2.3% |
| `h3d_sol_vsa_kernel` | 216.7 | 50 | 1.8% |
| `h3d_sol_pool_kernel` | 123.9 | 50 | 1.0% |
| `h3d_sol_threshold_kernel` | 66.3 | 50 | 0.5% |

Sum 12,345.9 ms against the run's own 12.59 s per evaluation. The GEMM pair is
64% and runs at roughly 61-68% of the BF16 tensor peak; the two hand-written
elementwise kernels (`qkv_rope`, `adaln`) together take 16.6% while sitting at
about a sixth of their bandwidth floors; the whole VSA path is 13.9% and reaches
about 1% of the tensor peak. Those four numbers are the stage-1 agenda, and they
apply whether or not the FP8 checkpoint arrives.

## 6. Decisions this plan needs

1. Approve the phases and their order.
2. The three transfers: the upscaler checkpoint (690 MB), the adapter (390 MB)
   and — if we build the prompt cache here — the offline Gemma and connector
   (36.9 GB). Alternatively the operator supplies the cache the source expects.
3. The golden: install the pinned stage-2 environment on the Spark once
   (`torch 2.13.0+cu132`, the three LTX packages, its `all2all_cpp` extension,
   the Triton Sol backend) and dump per-update latents and contexts for one
   seed, or accept that the refiner is gated on outputs alone. The first gives
   the port a real reference; the second leaves a 22B model unverified.
4. Whether stage 1 goes to FP8 as well: their recipe needs an FP8 H3 DiT
   (~15.5 GB) that we would quantize from the merged LoRA artifact, exactly as
   the reference does.

## 7. Risks

- Size. P3 is a 22B two-stream diffusion transformer with six attention modules
  and nine AdaLN tables per block; it is the largest single thing this repo has
  ported. The phases are ordered so that P0-P2 and P4 land measurable value and
  unblock the pipeline shape before P3's cost is fully known.
- Reference. The LTX-2 model source, the refiner driver and the frozen v2
  contract are local, so the port has a real reference for structure. What is
  missing is a numeric one: `docs/validation.md` records structural and CPU
  checks and one Spark integration, not hashes of stage-2 latents, and the
  vendored copy is a sibling revision of the pinned one. Without a runnable
  stage-2 environment the refiner's arithmetic is gated on outputs alone.
- Memory. 116.9 GiB resident of 119.68 GiB leaves under 3 GiB; our H3 stack is
  BF16 today and the host has 121 GiB total. Stage 1's FP8 is what makes room.
- Licenses. The LTX-2.x Community License is gated and the H3 upscaler declares
  no license at its pin. Local use is the operator's call; nothing here is
  published.
- The cache's provenance. `runtime` "validates its recipe and tensors"; a cache
  built with a different Gemma or connector revision is a silent quality change.
