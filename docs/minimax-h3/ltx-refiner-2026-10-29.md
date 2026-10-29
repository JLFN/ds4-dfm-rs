# P0: the LTX-2.5 refiner's own source, read at the pin the release runs

The unit the handoff asked for: read the second stage's own source with file:line
citations, write the weights map and check a cost model against the release's
share table, and diff the two pins the campaign had been carrying. No GPU job and
no build was run for this record; the Spark was idle and every weight fact below
was read from the four files' headers over the gvfs mount, not from a GPU.

Two of the unit's five steps turned out to be already landed: the weights map of
the *artifacts* and the latent path are [the reference map](ltx-reference-map-2026-10-28.md)
and [the latent path record](latent-path-2026-10-29.md), both older than this
work. What is new here is the pin diff (step 3), the read of the *model* source at
that pin (step 1), and a weights map that goes one level deeper than the earlier
one - the LoRA's per-module ranks and the loader's fusion arithmetic.

## 1. Which tree is the port's authority, and the pin diff

The profile does not vendor a model library. It puts the pinned checkout's three
packages on `sys.path` and imports from there:

- `models/minimax_h3/Sol-H3-Spark/runtime/stage2_ops/models.py:25-36` -
  `public_refiner_modules` inserts `ltx_root/packages/{ltx-core,ltx-pipelines,ltx-kernels}/src`.
- `configs/dependencies.json:50-59` pins that source at
  `Lightricks/LTX-2` revision `d151147788a9284cca791edc6ce898007e727fe6`.
- `runtime/stage2.py:79-80` and `:113` import `ltx_core.model.audio_vae` and
  `ltx_core.types`; `stage2_ops/conv_direct_nhwc.py:22` imports
  `ltx_core.model.video_vae.memory_efficient_decode`; `stage2_ops/models.py:49`
  imports `ltx_core.model.transformer.model.LTXModelType`.

**The pin is the v1.2.0 tag's commit.** Verified live 2026-10-29:
`git ls-remote https://github.com/Lightricks/LTX-2.git refs/tags/v1.2.0` returns
`d151147788a9284cca791edc6ce898007e727fe6`. The pristine tree is extracted to
`/data/ltx2-pin/packages` from a blob-filtered clone at `/data/ltx2-upstream`, and
**every** `ltx-core`/`ltx-pipelines` line in this record is from that tree.

**The two pins differ by one file, not by code.** `models/ltx25/GB200/SOURCE_SNAPSHOT.json` names
upstream release v1.2.0 with `upstream_commit 7954dcb0d986bdc36ef272564a9789ade07fcc65`;
`git log 7954dcb0..d1511477` is a single commit, `d151147 "core: Pin transformers
below 5.15."`, and `git diff --stat` across it is one file,
`packages/ltx-core/pyproject.toml`, 6 insertions and 1 deletion:
`transformers>=5.8.0` becomes `transformers>=5.8.0,<5.15`, with the comment that
5.15.0 routes config attribute access through a heterogeneity layer that raises
`AmbiguousGlobalPerLayerAttributeError` on global `config.head_dim`, which its own
gemma4 model code still reads. So step 3 is answered two ways: **no source
difference exists between the pins**, and the Gemma-4 encoder is pinned below
transformers 5.15 - a fact P1 needs.

**The vendored GB200 copy is a different lineage.** `models/ltx25/GB200/
environment/LTX-2/packages` is *not* v1.2.0, and nothing may be cited from it:

- `diff -rq` against the pin (ignoring `__pycache__`, `.venv`, egg-info and
  `uv.lock`) reports **145** entries that differ or exist in only one tree.
- Spot blobs by content hash: `feed_forward.py`, `attention.py` and
  `patchifiers.py` are byte-identical to the pin; `components/diffusion_steps.py`
  is not (pin 252 lines against 186, 66 changed lines).
- Subsystems exist in one and not the other: pristine-only
  `quantization/nvfp4`, `video_vae/transformer/{chunked,combined,dsl_kernels,det_attn_rope,rope_math,apply,config}.py`,
  `text_encoders/gemma/gemma_assets.py`; vendored-only `ltx_core/opt/`,
  `video_vae/tiling.py`, `video_vae/transformer/swiglu_efficient.py`.
- It was vendored by `549e11f "Add self-contained LTX-2.5 GB200 runtime"`
  (2026-08-14) in `NVlabs/Sana`, and its `optimized_commit ccedf847…` is not in
  the upstream repository at all (`git cat-file -t` refuses it), so it is that
  fork's private tip.

Consequence: [the reference map](ltx-reference-map-2026-10-28.md)'s "the vendored
LTX-2 source at v1.2.0 @ 7954dcb0, verified this session" is wrong, and its
`transformer.py:254-420`-style citations came from a tree whose line numbers the
port will not compile against. This record re-anchors them.

## 2. The weights, measured from the files

All four operator files are byte-exact against the manifest
(`configs/checkpoints.json`, entries at `:71-80` transformer, `:83-92` refiner
LoRA, `:95-104` video VAE, `:107-116` audio VAE), each pinned with its sha256:

| artifact | bytes (manifest = measured) | sha256 (manifest) |
|---|---:|---|
| `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | 42,018,190,584 | `792a2bad…c8e7584` |
| `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | 8,899,889,568 | `86370bbf…31a9bbf5` |
| `vae/ltx-2.5-video-vae-conv-bf16.safetensors` | 1,452,269,922 | `685b06ee…97dfce8d` |
| `vae/ltx-2.5-audio-vae-bf16.safetensors` | 364,866,540 | `c52733d3…0d54837a5` |

All four come from `Lightricks/LTX-2.5` at revision
`5e6e71018ee1756ed329b697a7b4aedc934dfce9`.

### 2.1 The transformer: 4,349 tensors, 48 blocks, 84 per block

Keys are ComfyUI-flavoured: `model.diffusion_model.` prefixes the whole model and
`transformer_blocks.N.` names a block
(`ltx-core/src/ltx_core/model/transformer/model_configurator.py:222-225`). 4,059
BF16 + 290 F32; 48 x 84 = 4,032 in the blocks, 317 outside.

Per block, in the checkpoint's own shapes (`weight` is `(out, in)`):

| module | tensors | shapes |
|---|---|---|
| `attn1`, `attn2` (video self, video text cross) | 12 each | `to_q/to_k/to_v/to_out.0` 4096->4096 with bias; `q_norm`, `k_norm` (4096,); `to_gate_logits` (32, 4096) with bias |
| `audio_attn1`, `audio_attn2` | 12 each | the same at 2048; gate (32, 2048) |
| `audio_to_video_attn` | 12 | `to_q` (2048, 4096), `to_k`/`to_v` (2048, 2048), `to_out.0` (4096, 2048), all biased; norms (2048,); gate (32, 4096) |
| `video_to_audio_attn` | 12 | `to_q` (2048, 2048), `to_k`/`to_v` (2048, 4096), `to_out.0` (2048, 2048); gate (32, 2048) |
| `ff` | 2 | `net.0.proj` (16384, 4096), `net.2` (4096, 16384), **no biases** |
| `audio_ff` | 4 | `net.0.proj` (8192, 2048) + bias (8192,), `net.2` (2048, 8192) + bias (2048,) |
| six AdaLN tables, F32 | 6 | `scale_shift_table` (9, 4096), `audio_scale_shift_table` (9, 2048), `scale_shift_table_a2v_ca_video` (5, 4096), `..._audio` (5, 2048), `prompt_scale_shift_table` (2, 4096), `audio_prompt_scale_shift_table` (2, 2048) |

So `attention_bias true` reaches q/k/v/out on all six attention modules, and the
video FF is bias-free exactly where the audio FF is not - the checkpoint's
`ff_bias false` with the code's `audio_ff_bias=True` default.

Outside the blocks, 317 tensors:

- Eight `AdaLayerNormSingle` modules, each `emb.timestep_embedder.linear_1`
  (d -> d), `linear_2` (d -> d) and a `linear` (d -> coefficient x d):
  `adaln_single` 36864 = 9 x 4096, `audio_adaln_single` 18432 = 9 x 2048,
  `prompt_adaln_single` 8192 = 2 x 4096, `audio_prompt_adaln_single` 4096 = 2 x
  2048, `av_ca_video_scale_shift_adaln_single` 16384 = 4 x 4096,
  `av_ca_audio_scale_shift_adaln_single` 8192 = 4 x 2048,
  `av_ca_a2v_gate_adaln_single` 4096 = 1 x 4096, `av_ca_v2a_gate_adaln_single`
  2048 = 1 x 2048.
- Two connector stacks, `video_embeddings_connector` and
  `audio_embeddings_connector`, each 8 x `transformer_1d_blocks.N` with the same
  per-block set as a transformer block's `attn1` (q/k/v/out with biases, q/k
  norms, a (32, d) gate) plus an `ff` (net.0.proj, net.2), and one
  `learnable_registers` of (128, 4096) and (128, 2048).
- The edges: `patchify_proj` (4096, 128) + bias, `proj_out` (128, 4096) + bias,
  `audio_patchify_proj` (2048, 128) + bias, `audio_proj_out` (128, 2048) + bias,
  and `keyframes_abs_pos_embedding` (1, 4096).
- No caption projection: the 22B checkpoint carries `caption_proj_before_connector
  true`, and `model_configurator.py:202-215` returns `(None, None)` for it, so the
  projection lives in the text encoder file - which is why its name carries
  `with-proj`. `caption_channels 3840` in the config is the caption width the
  connector is fed, not a transformer weight.

### 2.2 The LoRA: 3,320 tensors, four ranks, fused without an alpha

Measured from the LoRA's header, by module family:

| target | rank | tensors |
|---|---:|---:|
| every block linear named `attn1/attn2/audio_attn1/audio_attn2/audio_to_video_attn/video_to_audio_attn` `to_q/to_k/to_v/to_out.0`, and `ff.net.0.proj`/`ff.net.2`/`audio_ff.net.0.proj`/`audio_ff.net.2` | 450 | 2,688 (48 x 56) |
| the six `to_gate_logits` per block, `A` (32, d) and `B` (32, 32) | 32 | 576 (48 x 12) |
| `patchify_proj`, `proj_out`, `audio_patchify_proj`, `audio_proj_out` | 128 | 8 |
| each of the eight AdaLN singles' `emb.timestep_embedder.linear_1` | 256 | 16 |
| each AdaLN single's `..._linear_2` and `..._linear` | 450 | 32 |

The fusion is the loader's, not a PEFT wrapper:
`loader/fuse_loras.py:185-186` collects the affected keys from their `.lora_A.weight`
suffix, `:196-198` pairs A with B, and `:110-116` accumulates
`delta = B * strength @ A` in BF16 (`matmul(...).to(dtype)` for the first product,
`addmm_` for the rest), so the merged weight is `W + delta`. The key map is
`loader/sd_ops.py:135-137` (`diffusion_model.` stripped) and `:139-145` (`.lora_A.weight`
and `.lora_B.weight` both become `.weight`). **There is no alpha and no rank
scaling anywhere in the loader** - no `.alpha` key is read - so the release's
strength 0.8 (`runtime/stage2_ops/models.py:76`, `runtime/stage2.py:205`) is the
whole scale, and rank only sets A and B's inner dimension. `single_gpu_model_builder.py:97`
assigns by name into a meta model (`load_state_dict(..., strict=False, assign=True)`),
and `:209` appends the LoRA before the build; nothing is transposed or permuted
anywhere in `loader/`.

### 2.3 The two VAEs

The video VAE is 170 tensors: `conv_in.conv` (1024, 128, 3, 3, 3), three
`up_blocks`, `conv_out` (48, 128, 3, 3, 3) = 3 RGB x 4^2, and a mirrored encoder
whose `conv_out.conv` is (129, 1024, 3, 3, 3) - the 129th channel is still
unexplained, and it does not matter on this route because the adapter replaces the
encoder. The profile decodes with `ConvVideoDecoder` only
(`runtime/stage2_ops/models.py:128-129`), tiled by `TILES` at `:20-22`
(frames 128/24, height 448/64, width 768/64, validated at `:121-122` against
`tiling_scale_factors_for_vae`), emitting NHWC directly
(`runtime/stage2_ops/writer_chunking.py:11` asserts `(768, 1344, 3)`), and muxes at
CRF 19 `veryfast` (`runtime/stage2.py:123-125`).

The audio VAE is 1,329 tensors, and on this route only its **encoder** runs:
`runtime/stage2.py:78-93` encodes the H3 PCM and `:95-98` freezes the result
(`latent = clean_latent`, zero denoise mask, `frozen=True`), while the mux keeps
the original H3 waveform. The vocoder, the bandwidth-extension stage and the audio
decoder are out of scope, which is why their `output_sampling_rate` and BWE
configuration need no resolution now.

## 3. The refiner's graph, exactly

Citations are `ltx-core/src/ltx_core/...` (shortened to `core/`) and
`ltx-pipelines/src/ltx_pipelines/...` (`pipelines/`) inside `/data/ltx2-pin/packages`.

### 3.1 What the profile constructs

`runtime/stage2_ops/models.py:47-134`:

- geometry `VideoPixelShape(batch=1, frames=121, height=768, width=1344, fps=24)`
  at `:69`, its latent at `:70`, and `VideoLatentTools(VideoLatentPatchifier(patch_size=1),
  latent_shape, 24.0, scale_factors=VIDEO_SCALE_FACTORS)` at `:71-72` - the `24.0`
  is the fps that turns the time axis into seconds for RoPE.
- `DiffusionStage.from_checkpoint(transformer, bfloat16, cuda, loras=(lora,),
  quantization=None, ...)` at `:77-80`; an assertion that the built config is 48
  layers, 32 heads, 128 head_dim at `:83-85`; `AttentionManager(max_tokens=65536,
  num_heads=32, head_dim=128)` at `:86-87`; a sequence-parallel builder at `:95-97`;
  `BatchSplitAdapter(max_batch_size=1)` at `:109`.
- `Stage2SolAttention(transformer, ...)` at `:99-100` with a hard requirement that
  the backend is the original Triton one at `:101-102`, and a `LTXModelType.AudioVideo`
  check at `:105-108`.
- the audio encoder and `AudioLatentTools(AudioPatchifier(patch_size=1),
  AudioLatentShape.from_video_pixel_shape(pixel_shape))` at `:110-114`, then the
  tiles and the Conv decoder at `:116-129`.

The orchestrator is the release's own: `compat.OfficialCompatRefiner`
(`super_acceleration/stage2/official_compat_h3_refiner_diagnostic.py`), which the
profile subclasses at `runtime/stage2.py:33-37` to override `prepare_input`,
`encode_prompt_multimodal`, `upscale`, `convert`, `encode_source_audio`,
`prepare_multimodal_states`, `_checked_attention_stats` and `write_full_vae_video`.
Its sequence is `run_diagnostic` (`:506`): prepare input, encode prompt, video
encode, source audio, `prepare_multimodal_states` (`:313`), `denoise_multimodal`
(`:363`), then `stream_full_vae` (`:384`) and the writer.

### 3.2 Geometry and tokens

- Video: the refiner latent is `(1, 128, 16, 24, 42)` (`runtime/stage2_ops/h3_upscale.py`
  `REFINER_INPUT`, checked at `runtime/stage2.py:76`), patchified at patch size 1
  to **16,128 tokens** of width 128 (`core/components/patchifiers.py:12-18` fixes
  `(1, p, p)`, `:31-38` is the flatten `b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)`,
  so the token vector is channel-major and one token is one latent voxel; with
  `p=1` the token count is `frames x height x width`, `core/types.py:100-101`).
- Audio: `AudioLatentShape.from_video_pixel_shape` (`core/types.py:185-201`) with
  its defaults - channels 8, mel bins 16, 16 kHz, hop 160, downsample 4 - gives
  `round(121 / 24 * 16000 / 160 / 4)` = **126 frames**, so the latent is
  `(1, 8, 126, 16)` and its tokens are `(1, 126, 128)`: the audio patchifier packs
  channels and mel bins into one token per frame (`b c t f -> b t (c f)`,
  `core/components/patchifiers.py:300-304`), and 8 x 16 = 128 is exactly
  `audio_patchify_proj`'s input width. Token count is the frame count
  (`core/components/patchifiers.py:213-214`).
- So one step's attention set is 16,128 video tokens plus 126 audio tokens, well
  inside the 65,536-token manager budget.

### 3.3 Preprocessing

- **Timesteps.** `timestep_scale_multiplier` is 1000 (`core/model/transformer/model.py:62`),
  applied before the AdaLN MLP (`core/model/transformer/transformer_args.py:177-180`,
  `core/model/transformer/adaln.py:27-45`): sinusoidal 256-wide embedding, SiLU,
  `linear_1`, SiLU, `linear_2`, then the coefficient-wide `linear`
  (`adaln.py:37`), returning both the modulation and the `embedded_timestep` that
  drives the output stage. `Modality.timesteps` is `denoise_mask * sigma`
  (`pipelines/utils/helpers.py:484`), so conditioning positions see zero timestep.
- **Prompt path.** `prompt_adaln_single` (coefficient 2) is fed from the modality's
  own sigma (`core/model/transformer/transformer_args.py:276-279`) and modulates
  the cross-attention **keys and values**: the 2-row table's rows are combined with
  its own timestep projection (`core/model/transformer/transformer.py:441-443`) and
  applied as `context * (1 + scale_kv) + shift_kv` (`:446`), where row 0 is the
  shift and row 1 the scale. The query side of that attention is modulated by rows
  6..8 of the 9-row block table (`:239-241`, `get_ada_values(..., slice(6, 9))`).
- **Cross-modality AdaLN.** `av_ca_video_scale_shift`, `av_ca_audio_scale_shift`
  (coefficient 4) and `av_ca_a2v_gate`, `av_ca_v2a_gate` (coefficient 1) are built
  at `core/model/transformer/model.py:269-288` and fed at
  `core/model/transformer/transformer_args.py:393-409`: the scale/shift input is
  the *own* modality's timesteps times 1000, the gate input is the *other*
  modality's sigma times 1000 times `av_ca_timestep_scale_multiplier /
  timestep_scale_multiplier` (1.0 here). `model.py:505-506` prepares video with
  audio as the cross modality and vice versa. The cross-axis RoPE uses the time
  axis only, `[cross_pe_max_pos]` and the midpoint grid, at width
  `audio_cross_attention_dim` = 2048 (`transformer_args.py:365-370`).
- **RoPE.** theta 10000, video `max_pos [20, 2048, 2048]`, audio `[20]`, split
  rotation, midpoints grid, `frequencies_precision float64`
  (`core/model/transformer/model.py:60-71`, `:105`, `:118`). The frequency grid is
  built in numpy float64 and only then cast, `core/model/transformer/rope.py:86-108`;
  positions are the patch *midpoints*, `(start + end) / 2` (`rope.py:149-155`), and
  the angle is `index * (2 * fractional_position - 1)` (`rope.py:159`); the split
  rotation is a half-swap with `addcmul_` (`rope.py:66-77`). Latent coordinates are
  computed in pixel space - scale factors per axis, a first-frame causal fix, then
  a division by fps for the time axis (`core/components/patchifiers.py:71-78`,
  `:158-170`). `causal_temporal_positioning` is a *connector* flag in this
  revision, not a transformer one.
- **Patchify projection and keyframes.** `patchify_proj` then, if
  `use_keyframes_abs_pos_embedding`, adding `keyframes_abs_pos_embedding` where the
  keyframes mask is set (`core/model/transformer/transformer_args.py:268-269`,
  `:22-43`), with the parameter initialized to zeros (`model.py:217-218`).

### 3.4 One block, in order

One class, `BasicAVTransformerBlock` (`core/model/transformer/transformer.py:87`),
replicated 48 times with no per-layer variation
(`core/model/transformer/model.py:407-417`, `:442`). **There is no dense or
single-stream layer in this revision**: what the contract calls "layer 0 dense" is
a *Sol-Attn dispatch policy* (`super_acceleration/STAGE2_CONTRACT.md:97-101`,
`stage2/sol_attention.py:4`), not a different block.

1. **Video self-attention.** `get_ada_values(scale_shift_table, ..., slice(0, 3))`
   returns `(shift, scale, gate)` (`transformer.py:271-273`); the modulated input is
   `rms_norm(x) * (1 + scale) + shift` (`core/model/transformer/ops.py:50-58`);
   `attn1` runs; the result is gated into the residual by
   `post_sa_function(x, out, None, eps, gate)` (`:285`).
2. **Video text cross-attention** (`:288`), with the prompt table on the context
   (3.3) and rows 6..8 on the query.
3. **Audio self-attention and audio text cross-attention** (`:300-325`), the same
   two steps at width 2048.
4. **A2V then V2A**, both from a snapshot taken before either mutates anything:
   `vx_pre_av = vx`, `ax_pre_av = ax` with the comment "V2A's video keys/values must
   use the pre-A2V state so direction order doesn't bias the result"
   (`transformer.py:331-334`). A2V is `audio_to_video_attn(video_query,
   context=audio_kv, pe=video.cross_pe, k_pe=audio.cross_pe)` scaled by
   `gate_out_a2v * cross_attn_perturbation_mask` (`:341-364`); V2A mirrors it
   (`:367-394`). The scaling of the inputs uses the 5-row per-direction table with
   `slice(0, 2)` per direction - rows 0,1 for A2V and rows 2,3 for V2A - and the
   gate is row 4 (`core/model/transformer/transformer.py:201-220`).
5. **Feed-forward** with rows 3..5 of the block table (`:400-414`):
   `rms_norm(x) * (1 + scale) + shift`, then `Linear -> GELU(tanh) -> Linear`
   (`core/model/transformer/feed_forward.py:5-15`, `gelu_approx.py:10-11`), gated
   into the residual. `mult` is 4, so 4096 -> 16384 -> 4096 for video and
   2048 -> 8192 -> 2048 for audio.

Attention internals (`core/model/transformer/attention.py`): `inner_dim = heads *
dim_head` (`:499`), `q_norm`/`k_norm` are RMSNorm **over the whole inner dim, before
the head split** (`:505-506`), `to_q`/`to_k`/`to_v` are biased linears (`:508-510`),
the gate is `to_gate_logits = Linear(query_dim, heads)` (`:514`), and the per-head
gate is `2 * sigmoid(logits) * out` taken from the block's raw input, before
`to_out` (`ops.py:94-106`), so **the output projection is not gated**
(`attention.py:578-580` then `:582`). No rotary is applied to the text
cross-attention's keys - `pe` is the self-attention's or the AV pair's.

### 3.5 The output

`model.py:472-489` and `:520-536`: a LayerNorm **without affine parameters**
(the config's `norm_elementwise_affine false`, and the checkpoint has no
`norm_out` tensors), then the 2-row model table combined with the
`embedded_timestep`, `x * (1 + scale) + shift` with row 0 the shift and row 1 the
scale (`:486-488`), then `proj_out` to 128 channels (`:489`). Then unpatchify.

### 3.6 The sampler

Deterministic Euler, three updates from the literal schedule, no CFG:

- `STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]`
  (`pipelines/utils/constants.py:20`), three updates because the loop is
  `sigmas[:-1]` (`pipelines/utils/samplers.py:73`).
- One update: `sigma = sigmas[i]`, `sigma_next = sigmas[i+1]`, `dt = sigma_next -
  sigma`, `velocity = (x - x0) / sigma`, `x + velocity * dt` with `x` and
  `velocity` cast to float32 for the product
  (`core/components/diffusion_steps.py:35-40`, `core/utils.py:25,35-36`).
- The mask blend runs **before** the step: `post_process_latent(denoised,
  denoise_mask, clean)` = `denoised * mask + clean.float() * (1 - mask)`
  (`pipelines/utils/helpers.py:462-463`), called at `pipelines/utils/samplers.py:35-36`.
- The denoiser is `SimpleDenoiser` (`pipelines/utils/denoisers.py:214-253`), one
  transformer call per step, `sigma` expanded per batch (`:237`, `:248`); the
  profile passes the cached video and audio contexts and no guidance
  (`super_acceleration/stage2/official_compat_h3_refiner_diagnostic.py:377`).
- Start: `GaussianNoiser` seeded per request (`...diagnostic.py:322-323`) and
  `noise_scale = stage2_sigmas[0]` = 0.909375 (`:330`, `:339`), where the noise
  scale is a two-way `lerp` weight, `lerp(latent, noise, noise_scale)` then
  `lerp(clean, that, denoise_mask)` (`core/components/noisers.py:32-33`).
- Frozen audio: the audio latent is replaced by its clean value with a zero mask
  and `frozen=True` (`runtime/stage2.py:95-98`), which forces `Modality.sigma` to
  zero (`pipelines/utils/helpers.py:479-480`). Audio is still carried through every
  block and its refinement discarded - it conditions video through the AV pair.

### 3.7 The Sol-Attn policy

Video self-attention only: layer 0 dense, layers 1-47 Sol-Attn, cross and audio
attention dense (`super_acceleration/STAGE2_CONTRACT.md:97-99`,
`super_acceleration/stage2/sol_attention.py:4`, `:207`, `:209`). The tau is **per
step, not per layer**: `STAGE2_TAUS = (1.0, 1.25, 1.5)` (`sol_attention.py:20`)
indexed by the step counter (`:188`), advanced once per step at layer 1's entry
(`:67-68`, `:160-165`). The assertion the release runs is exact - 3 dense calls,
141 Sol calls, which is 47 x 3 (`runtime/stage2.py:101-107`) - so a study of the
policy must land on those counts or say why not. The kernel behind it is
`techniques/sparse_backends/sol_attn_backend._run_sol_attn_bthd` from the Sana
techniques tree (`sol_attention.py:173-194`), with `thresh_type="diag"` and
`kv_splits="auto"`; that backend is the next read this port owes.

### 3.8 Conditioning, decode and mux

- The prompt is a **fixed string** and the context is an offline cache:
  `FIXED_PROMPT = "4K, refined, high quality, cinematic detail, clean textures,
  natural motion."` (`runtime/prompt_cache.py:15`), contexts
  `{"video": (1, 1024, 4096), "audio": (1, 1024, 2048)}` in BF16 (`:29-30`), with a
  recipe that names both int8 artifacts and is validated for exact equality
  (`:16-28`, `:38-60`). Gemma is never resident (`runtime/stage2.py:209`).
- Decode: the Conv VAE, the tiles above, NHWC, 16-frame chunks
  (`runtime/stage2.py:110-130`), then H.264 CRF 19 `veryfast` at 24 fps with the
  original 32 kHz H3 audio.

## 4. The cost model, re-derived, against the release's share table

[The reference map](ltx-reference-map-2026-10-28.md) §4 carries the FLOP table.
Re-deriving it independently from the geometry in 3.2 and the shapes in 2.1, with
a linear `n x k -> m` at `2nkm` and dense attention at `4nkd` FLOPs, reproduces
every row: `attn1` 2,168.9 GFLOP of projection (qkv 1,623.5, out 541.2, gate 4.2)
plus 4,261.7 of attention; `attn2` 1,155.3 (q 541.2, k 34.4, v 34.4, out 541.2,
gate 4.2) plus 270.6; video FF 4,329.3; the AV pair 543.3 each plus 16.6 of
attention. The one discrepancy is 4.2 GFLOP: the earlier table puts the per-head
gate's `2 x 16128 x 4096 x 32` on one of the two AV directions only, where both
carry it - 0.03% of the block, recorded rather than fixed.

Per evaluation that is **640.1 TFLOP dense**, and with the Sol policy - layer 0
dense, layers 1-47 sparse at 10% of the keys - **445.1 TFLOP**, of which 420.9 is
projection and FF. Three updates are 1,335.3 TFLOP, which against the release's
own 24.71 s is **54.0 TFLOPS, 45% of the ~120 TFLOPS BF16 peak at 2.45 GHz**, so
the stage is a GEMM problem (1,262.6 of those TFLOP) with a sparse-attention
problem on top. Our own DiT already runs its projections at 73.3 TFLOPS
(`denoise-attribution-2026-10-28.md`), which is the evidence that the release's
number is reachable rather than a floor. Weight streaming is not the limit: 21.0 G
parameters is 39.1 GiB, and one pass per evaluation at 250 GB/s is 168 ms against
a per-evaluation budget near 8.2 s.

The share table's remaining rows are consistent with the same measurement: the
audio tokens are 126 against video's 16,128, so the audio-only rows are noise
(0.0-0.2% of a block) - which is exactly what the table says.

## 5. What this changes in the campaign's records

- **The vendored LTX-2 copy is not the pin.** The reference map's §1 claim and any
  citation drawn from it (its §7 "what the port must mirror" list) must be read
  against `/data/ltx2-pin`. The step-3 diff's answer is: the two pins differ by one
  dependency bound, nothing else, so the campaign was reading a *different* tree
  than it believed rather than a *different version* of the right one.
- **There is no single-stream or dense block.** The plan's "layer 0 dense" is the
  Sol-Attn dispatch policy; a port that builds a second block class for layer 0
  would be inventing architecture.
- **The video FF is GELU-tanh, not SwiGLU**, with no biases, while the audio FF has
  biases - so `h3d_swiglu_kernel` does not transfer, as the reference map already
  said, and the bias difference is per stream.
- **The LoRA is fused without an alpha**, at four distinct ranks, and 576 of its
  tensors are a rank-32 update on the per-head gate logits - which a merge that
  assumed "rank 450 everywhere except the timestep embedders" would drop.
- **The taus are per step.** The GB200 contract's `taus 1.0/1.25/1.5` reads as a
  per-layer schedule; the Spark code that runs applies one tau per update to all
  47 Sol layers.
- **P1's dependency is pinned**: transformers below 5.15, for the gemma4
  `config.head_dim` failure the pin's own comment describes.

## 6. What is now specified, and what is still open

Specified and ready to implement: the block (3.4), the preprocessing (3.3), the
sampler (3.6), the edges and the weights (2.1), the LoRA merge (2.2), the
conditioning contract (3.8).

Still open, in the order the phases need them:

1. **The Sol-Attn kernel itself** - `techniques/sparse_backends/sol_attn_backend`
   in the pinned Sana checkout, `_run_sol_attn_bthd` with `thresh_type="diag"`:
   the 10%-of-keys selection, its tile size and its numerical contract. Nothing in
   the ported tree corresponds to it yet.
2. **A numeric reference** for the refiner, which needs the pinned stage-2
   environment (decision 2 of the handoff) - the same decision this read makes more
   concrete, because the Sol policy has no dense fallback path in the port.
3. **The caption path** (P1): the int8 Gemma pair, or the cache. The connector's
   own structure is in 2.1 and its arithmetic in
   `core/text_encoders/gemma/embeddings_connector.py:95-189` (8 blocks, 128
   learnable registers tiled when `seq_len % 128 == 0`, one final sequence
   RMSNorm, no `norm_output` symbol anywhere in the tree) and
   `feature_extractor.py:48-64` (`PER_TOKEN_RMS` normalizes over the layer axis of
   `[B, T, D, L]` and zeroes padding), with the tokenizer at a fixed 1024
   (`gemma_assets.py:161`).
4. **The decoder's tiling**, ported against `ConvVideoDecoder` and the explicit
   `TILES` contract rather than the public `TilingConfig.default()`.

The artifacts are unchanged from the handoff's list: the four above are present
and byte-exact, and the int8 Gemma pair plus the FP8 H3 DiT remain the two stores
this stage cannot run without.
