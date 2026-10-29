# P0: the LTX-2.5 reference, its exact bridge, and the cost model

P0 of [the pipeline plan](ltx-sol-pipeline-plan-2026-10-28.md): read the LTX-2.5
sources first-hand, write the weights map, and check a cost model against the
release's own share table, so every later phase ports a measured object instead
of a guessed one. Nothing here was built or run on a GPU; the Spark is idle.

## 1. What was read

- The vendored LTX-2 source, extracted from the pinned checkout to
  `/data/ltx2-reference` by a sparse, blob-filtered clone at
  `6e4a440` (`sol-engine`), cone over
  `models/ltx25/GB200/environment/LTX-2`, `models/ltx2.5-refiner/GB200`,
  `models/minimax_h3/super_acceleration`, `models/ltx25/prompts`
  (407 files, 9.4 MiB; `packages/{ltx-core,ltx-pipelines,ltx-kernels,ltx-trainer}`
  = 158/43/10+43/23+21+9 files). The clone is separate so `/data/Sol-h3-Spark`
  stays untouched.
- The Spark profile itself, which is the actual port target and is materialised
  in the pinned checkout at `/data/Sol-h3-Spark/models/minimax_h3/Sol-H3-Spark/`
  (72 tracked files: `configs/`, `runtime/`, `stage1_ops/`, `stage2_ops/`) —
  not in `/data/ltx2-reference`, whose cone covers only the vendored LTX-2
  packages, the GB200 refiner and the super-acceleration lineage.
- The four LTX-2.5 checkpoints the operator added, by safetensors header only:
  `ltx-2.5-22b-dev-transformer-bf16.safetensors` (42,018,190,584 B),
  `ltx-2.5-22b-distilled-lora-450-bf16.safetensors` (8,899,889,568 B),
  `ltx-2.5-video-vae-conv-bf16.safetensors` (1,452,269,922 B),
  `ltx-2.5-audio-vae-bf16.safetensors` (364,866,540 B). All four byte counts
  equal `configs/checkpoints.json`'s `bytes` fields exactly
  (`checkpoints.json:76-117`), so this is the release's own weight set.
- The release page, re-fetched: `nvlabs.github.io/Sana/Sol-Engine/Sol-H3-Spark`.

## 2. The bridge, exactly

The Spark route is not the public LTX two-stage pipeline; it replaces the
official latent upsampler. `configs/default.json:15-21` declares
`upscale_before_adapter: true`, `upscale_ratio: 2.0`, and
`refiner_input_shape: [1,128,16,24,42]`; `runtime/stage2.py:58-76`
(`video_encode`) runs upscale, then adapter, then crop, shape-checking each
seam with `checked()`, and `runtime/stage2_ops/h3_upscale.py:24-27`
pins the four shapes:

    H3_INPUT       (1,  24, 37, 24, 42)   normalized H3 latent, 124 pixel frames
    H3_UPSCALED    (1,  24, 37, 48, 84)   author node D(model(N(x))), scale 2.0
    ADAPTER_OUTPUT (1, 128, 17, 24, 42)   LTX-space latent
    REFINER_INPUT  (1, 128, 16, 24, 42)   `latent[:, :, :16]`

Read against the frozen geometry in
`runtime/stage2_ops/h3_ltx_adapter/constants.py:3-11` (H3: 24 channels, 16x
spatial, 4x temporal, clip 17, drop 3; LTX: 128 channels, 32x spatial, 8x
temporal), the whole route is exact and needs no interpolation:

- H3 latent frames for `n` pixel frames are
  `ceil(n/17) * ceil(17/4) - 3` (`geometry.py:13-25`): 124 -> 37, and our own
  job A's 107 -> 32. That reconciles our port's recorded job-A latent
  (32 x 48 x 48, 18,432 target rows = 32 x 24 x 24 with the 2x2 DiT patch) with
  the release's `H3_INPUT`, from the same 16x/4x VAE.
- The H3 latent of 384x672 is 24x42. At the LTX 32x factor on the 1344x768
  canvas the LTX latent is also 24x42. The H3 upscale to 48x84 followed by a
  fixed 2x pixel-unshuffle (`geometry.py:102-112`, `scale_h == scale_w == 2`
  is asserted at `:105`) is what converts one grid into the other, and it
  multiplies the channels by 4: `cat((linear, packed))` along channels
  (`geometry.py:99-101`) gives 24 + 24x3 = 96, and 96 x 4 = 384 = the adapter's
  `in_channels`.
- Temporal alignment is `temporal_resample` (linear interpolation between the
  37 H3 positions and the LTX positions of the padded 129 frames,
  `geometry.py:39-53`) concatenated with `temporal_nearest_pack` into 3 slots
  (`geometry.py:56-79`), giving the 17 LTX latent frames that are cropped to 16.
- The adapter itself is a plain Conv3D resnet stack
  (`h3_ltx_adapter/model.py:27-42`) and is now fully specified by its own
  `config.json` next to its header: `in_channels 384`, `out_channels 128`,
  `width 752`, `num_blocks 22`, `expansion 2`, `groups 16`, `model_type tiny`,
  194,759,504 BF16 parameters in 228 tensors, `format
  h3_ltx_adapter_safetensors_v1`. Its map, confirmed against the header: a 1x1
  `skip (128,384,1,1,1)`, a 3x3x3 `stem (752,384,3,3,3)`, 22
  `FactorizedResidualBlock`s of 10 tensors each, then `final_norm` (GroupNorm
  16), SiLU and a 1x1 `head (128,752,1,1,1)`; each block is `GroupNorm(16) ->
  SiLU -> spatial (752,752,1,3,3) -> depthwise temporal (752,1,3,1,1) -> SiLU ->
  1x1 up to 3008 -> chunk(value,gate) -> value*SiLU(gate) -> 1x1 back to 752`,
  added to its input. `build_model` (`model.py:45-55`) rejects any checkpoint
  that enables the unsupported training variants, all zero here.

So the release's "2 -> 0 inter-stage VAE" claim is mechanically this: the H3 VAE
decoder and the LTX VAE encoder are both replaced by one small learned Conv3D
whose input is an H3 latent and whose output is an LTX latent. The upscale half
is the author's own ComfyUI node (`LBH-123-AI/Minimax_h3_latent_Upscaler`, the
`normalization`, `LatentResizer3D` and `zero_module` symbols `h3_upscale.py:22`
expects), applied as `D(model(N(x)))` on the normalized H3 latent, taking
24 -> 24 channels in place and doubling H and W.

## 3. The LTX-2.5 model maps

From the checkpoint metadata config block of the transformer (the authority;
`transformer.py`'s dataclass defaults are not what ships):

    num_layers 48   num_attention_heads 32   attention_head_dim 128
    audio_num_attention_heads 32   audio_attention_head_dim 64
    in/out_channels 128   cross_attention_dim 4096   audio_cross_attention_dim 2048
    activation_fn gelu-approximate   ff_bias false   audio_ff_bias (absent -> true)
    cross_attention_adaln true   use_prompt_adaln_single true   av_cross_ada_norm true
    apply_gated_attention true   qk_norm/standardization_norm rms_norm   norm_eps 1e-6
    rope_type split   frequencies_precision float64   positional_embedding_theta 10000
    positional_embedding_max_pos [20,2048,2048]   audio [20]   use_middle_indices_grid true
    timestep_scale_multiplier 1000   av_ca_timestep_scale_multiplier 1000.0
    causal_temporal_positioning true   use_keyframes_abs_pos_embedding true
    caption_proj_before_connector true   text_encoder_norm_type PER_TOKEN_RMS
    connector: 8 blocks, heads 32, head_dim 128 (audio 32 x 64), 128 learnable
    registers, norm_output true, gated attention true, max_pos [4096]

Tensor inventory: 4,349 tensors, 21,004,025,600 parameters, 4,059 BF16 and 290
F32. The 290 F32 are exactly 2 global `(2, dim)` tables plus 48 blocks x 6
tables: `scale_shift_table (9,4096)`, `audio_scale_shift_table (9,2048)`,
`scale_shift_table_a2v_ca_{video,audio} (5,4096)/(5,2048)`,
`prompt_scale_shift_table (2,4096)`, `audio_prompt_scale_shift_table (2,2048)`.
Nine rows is `cross_attention_adaln`'s coefficient (`adaln.py:6-16`).

Block 0 has 84 tensors and mirrors the code exactly: `attn1`, `attn2`,
`audio_attn1`, `audio_attn2` (all `4096->4096` video, `2048->2048` audio, with
`q_norm`/`k_norm` RMSNorm weights, `to_gate_logits (32, dim)`),
`audio_to_video_attn` (q `4096->2048`, k/v `2048->2048`, out `2048->4096`),
`video_to_audio_attn` (q `2048->2048`, k/v `4096->2048`), `ff` `4096->16384->4096`
and `audio_ff` `2048->8192->2048`. `ff` carries no biases and `audio_ff` does,
which is what `ff_bias false` plus the code's `audio_ff_bias=True` default
predicts.

Outside the blocks: `patchify_proj (4096,128)`, `proj_out (128,4096)`,
`audio_patchify_proj (2048,128)`, `audio_proj_out (128,2048)`,
`keyframes_abs_pos_embedding (1,4096)`, and the eight `AdaLayerNormSingle`
modules sized `9*d`, `2*d`, `4*d`, `1*d` for video and audio at the coefficients
`model.py:169-232` computes - `adaln_single (36864,4096)`,
`audio_adaln_single (18432,2048)`, `prompt_adaln_single (8192,4096)`,
`audio_prompt_adaln_single (4096,2048)`,
`av_ca_video_scale_shift (16384,4096)`, `av_ca_audio_scale_shift (8192,2048)`,
`av_ca_a2v_gate (4096,4096)`, `av_ca_v2a_gate (2048,2048)`. Every AdaLN is a
`256 -> d -> d` SiLU MLP followed by a `d -> coefficient*d` Linear
(`adaln.py:19-43`, `timestep_embedding.py:114-142`).

Video VAE, from its checkpoint metadata `config.vae`: `CausalVideoAutoencoder`,
3D, in/out 3, **latent_channels 128**, `patch_size 4`, `norm_layer pixel_norm`,
`spatial_padding_mode zeros`, `causal_decoder false`,
`timestep_conditioning false`, base channels 128. The encoder block list is
`res_x(4) -> compress_space(2) -> res_x(6) -> compress_time(2) -> res_x(4) ->
compress_all(2) -> res_x(2) -> compress_all(1) -> res_x(2)`, and the decoder the
mirror `res_x(4) -> compress_space(2) -> res_x(6) -> compress_time(2) ->
res_x(4) -> compress_all(1) -> res_x(2) -> compress_all(2) -> res_x(2)`. That is
two spatial steps times `patch_size 4` = **32x spatial** and three temporal
steps = **8x temporal**, i.e. `F = 8*(F'-1)+1`, `H = 32*H'`, `W = 32*W'` -
exactly the factors the release's geometry assumes. 170 tensors: `conv_in
(1024,128,3,3,3)`, nine `up_blocks` (res blocks at 1024/512/512/256/128 and
depth-to-space convs `(4096,1024)`, `(4096,512)`, `(512,512)`, `(512,256)`),
`conv_out (48,128,3,3,3)` = 3 RGB x 4^2, and a mirrored encoder whose
`conv_in (128,48,...)` re-packs 4x4 RGB patches and whose `conv_out` is
**(129,1024,...)**. The encoder's 129th channel is not accounted for by the
decoder's 128 and is left open here.

Audio VAE, from its metadata: `z_channels 8`, `ch 128`, `ch_mult [1,2,4]`,
`num_res_blocks 2`, `mel_bins 64`, `in_channels 2`, `causality_axis height`,
`norm_type pixel`, `sampling_rate 16000`, hop 160, and a vocoder chain of
`upsample_rates [5,2,2,2,2,2]` plus a BWE stage `[6,5,2,2,2]` whose
`output_sampling_rate` is 48000. 1,329 tensors, 364,866,540 B.

The Spark profile uses only the audio **encoder** and only the video **decoder**:
`stage2.py:78-93` encodes the H3 PCM, `stage2.py:96-99` freezes that latent
(`denoise_mask` zero, `frozen`), and `stage2.py:111-130` muxes the **original**
H3 PCM as AAC. `STAGE2_CONTRACT.md:74-78` says the same: the jointly denoised
stage-2 audio is discarded. So the audio vocoder, the BWE stage and the audio
decoder are out of scope; so is the video VAE encoder on this route (the
adapter replaces it), which leaves `conv_out`'s 129th channel a curiosity rather
than a blocker.

## 4. The cost model and the share table

Geometry at the Spark profile's output: canvas 768x1344x121 -> video latent
16x24x42 = **16,128 tokens**; audio ~126 tokens. 48 layers, 32 heads x 128.
FLOPs are forward MACs at 2 each, counted per component per evaluation:

    component                              proj GFLOP  attn GFLOP   total   share
    video self attn (attn1)                    2168.9      4261.7   6430.6   48.2%
    video text cross (attn2)                   1155.3       270.6   1425.9   10.7%
    video FF                                   4329.3         0.0   4329.3   32.5%
    audio self attn (audio_attn1)                 4.2         0.1      4.4    0.0%
    audio text cross (audio_attn2)               19.3         1.1     20.4    0.2%
    audio -> video (audio_to_video_attn)        547.5        16.6    564.2    4.2%
    video -> audio (video_to_audio_attn)        543.3        16.6    559.9    4.2%
    per block                                  8767.9      4566.7  13334.6

Per evaluation that is 640.1 TFLOP dense, or **445.1 TFLOP** with layer 0 dense
and layers 1-47 sparse (the release's Sol-Attn policy: video self-attention
only, at sparsity 0.9, i.e. 10% of each query's keys retained); three updates
are 1,335.3 TFLOP sparse. Against the release's 24.71 s for LTX (`44.0%` of
56.17 s) that is **54.0 TFLOPS, 45.0% of the ~120 TFLOPS BF16 `mma.sync` peak
at 2.45 GHz** on 48 SMs. The conventions, for anyone re-deriving it: a linear
`n x k -> m` is `2nkm` FLOPs, and dense attention over `n` queries and `k` keys
with total head width `d` is `2nkd` MACs = `4nkd` FLOPs for QK^T plus AV. The
per-component table above reproduces from those two formulas and the geometry to
0.1 GFLOP.

Three consequences decide the port:

1. Sol-Attn removes only 30.5% of the FLOPs (194.9 of 640.1 TFLOP per
   evaluation). The remaining 445.1 TFLOP is 420.9 TFLOP of projections and
   video FF plus 24.2 TFLOP of sparse attention: **1,262.6 TFLOP of dense GEMM
   over the three updates, 94.5% of the sparse total**. The LTX stage is a GEMM
   problem first and a sparsity problem second.
2. Our own H3 DiT already runs its projections at 73.3 TFLOPS (61% of the 2.45
   GHz peak, `denoise-attribution-2026-10-28.md`). At that efficiency the same
   1,262.6 TFLOP takes 17.2 s, against the release's 24.71 s for the whole stage.
   That is a projection from our measured efficiency and their operator mix, not
   a measurement of a port that does not exist yet; it says the release's LTX
   number is a reachable target, not a floor.
3. Weight streaming is not the limit: 21.0 G parameters is 39.1 GiB, one pass at
   250 GB/s is 84 ms per evaluation (168 ms counting the LoRA-merged stream
   twice). Activation traffic is ~132 MB per tensor at this token count. The
   stage is compute-bound.

The remainder of the release's 56.17 s, with the page's own percentages applied
to it: Qwen 1.35 s, H3 four steps 19.77 s, H3 upscaler + adapter 2.13 s, LTX
three steps 24.71 s, LTX VAE decode 7.02 s, MP4 + audio mux 0.73 s, overhead
0.51 s. The VAE decode is 124.9 Mpx in 7.02 s = **17.8 Mpx/s**, which is the
rate our own decoder work has to reach: our H3 decoder does 1.85 Mpx/s, so the
LTX conv decoder is nearly 10x faster per pixel and is not portable from the H3
one.

Stage 1 checks out against our measurement independently: the reference row
count is 9,324 (37 x 12 x 21 latent tokens at 672x384 with the 2x2 patch), 4
evaluations in 19.77 s = 530 us per row per evaluation, against our 672 us on
job A (18,799 rows x 4 in 50.55 s). We are **1.27x slower per row**, and we have
already attributed that 21% (`denoise-attribution-2026-10-28.md`: the 7,901 ms
projection GEMM, `qkv_rope`, `adaln`).

## 5. Corrections to earlier records

- The video FF is **GELU-tanh, not SwiGLU**: `FeedForward` is
  `GELUApprox -> Identity -> Linear` (`feed_forward.py:6-13`, `gelu_approx.py:4-11`)
  and the config's `activation_fn` is `gelu-approximate`. Our H3
  `h3d_swiglu_kernel` does not transfer; the LTX FF needs a gelu-tanh epilogue.
- The LTX video VAE is **32x spatial / 8x temporal / 128 latent channels**, and
  is a Conv VAE (`_class_name CausalVideoAutoencoder`), not the TAEHV decoder
  named in `models/ltx2.5-refiner/GB200/refiner.toml:50`. `configs/default.json`
  sets `stage2.decoder = official_Conv_VAE`, so the Spark profile's 7.02 s
  decode is the convolutional VAE above. TAEHV belongs to the GB200 profile.
- The Spark **stage 2 loads no text encoder at all**:
  `stage2.py:209` records `stage2_gemma_loaded: False` and
  `stage2_connector_loaded: False`, and `prompt_cache.py:15` fixes the prompt
  to `"4K, refined, high quality, cinematic detail, clean textures, natural
  motion."` with a bounded 64 MiB cache (`prompt_cache.py:29`) carrying
  `video (1,1024,4096)` and `audio (1,1024,2048)` BF16 contexts
  (`prompt_cache.py:30`). Gemma is an offline concern, exactly as the release
  page states. It is not a speed lever and it is not needed resident.
- `TILES` in `stage2_ops/models.py:20-22` is `frames 128/24, height 448/64,
  width 768/64` and is asserted against the checkpoint's own scale factors
  (`models.py:118-128`), with `tiling_source "original_Conv128_448x768"`
  (`models.py:123`). In pixel units against a 768x1344x121 output that is
  2x2x1 = 4 decode tiles. The public GB200 pipeline instead uses
  `TilingConfig.default()` (768/64 px spatial, 80/24 frame), so the tile
  numbers are profile-specific and the Spark ones are the contract to port.
- The sparsity policy has a **machine-checked receipt**, not just prose:
  `stage2.py:101-111` (`_checked_attention_stats`) aborts the run unless the
  backend reports exactly `completed_steps 3`, `dense_calls 3`, `sol_calls 141`,
  `kernel_calls 141`, `dense_guard_calls 0`, backend `triton`,
  architecture `sm121-triton`. One dense call and 47 Sol calls per update is
  precisely the contract's "layer 0 dense, layers 1-47 strict Sol-Attn"; a port
  that deviates produces 140 or 142 and fails this check.
- The three-update schedule is a literal constant, not a scheduler:
  `STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]`
  (`ltx-pipelines/utils/constants.py:19`), four endpoints giving three Euler
  updates because the loop iterates `sigmas[:-1]` (`samplers.py:75`). The update
  is `x + ((x - x0_hat)/sigma) * (sigma_next - sigma)` in float32
  (`diffusion_steps.py:24-40`). The refiner runs with **no CFG** - only stage 1
  in the two-stage pipelines is guided - and the release's stage 2 is a
  single conditioned pass per step.
- The prompt cache carries the pair, not the video tensor alone:
  `CONTEXT_SHAPES` (`prompt_cache.py:30`) is video `(1,1024,4096)` and audio
  `(1,1024,2048)`, because the joint AV model consumes an audio context too.
  The Gemma attention mask is consumed inside `process_hidden_states` and is not
  a denoiser input (`prompt_cache.py:1-7`).

## 6. Artifacts

All checks below are of the Spark's own store, read over the gvfs mount at
`/mnt/h3-artifacts` (the workstation's `/mnt/h3-artifacts` is empty; the store
is on the measurement host).

**Present**, sha256 matched against `configs/checkpoints.json`'s pinned values
where it carries one:

| artifact | bytes | sha256 |
|---|---:|---|
| LTX-2.5 22B dev transformer bf16 | 42,018,190,584 | — (byte count matched) |
| LTX-2.5 distilled LoRA 450 bf16 | 8,899,889,568 | — (byte count matched) |
| LTX-2.5 video Conv VAE bf16 | 1,452,269,922 | — (byte count matched) |
| LTX-2.5 audio VAE bf16 | 364,866,540 | — (byte count matched) |
| H3 latent upscaler `..._3d_bf16.safetensors` | 690,592,992 | `4f57821f…46a5e6` verified |
| H3-to-LTX adapter `model.safetensors` + `config.json` | 389,541,112 | `170199a3…0724ac` verified |

Both verified sha256 values equal the release's own: the upscaler's is the
constant in `stage2_ops/h3_upscale.py:20-21`, and the adapter's is both
`checkpoints.json`'s and the one its own `config.json` declares under
`model_sha256`. The adapter's declared parameter count also reproduces exactly
from its header (194,759,504 BF16 in 228 tensors).

**Missing**, with the exact repo, revision and byte count from
`checkpoints.json`, confirmed live against the Hugging Face model/tree API:

| artifact | repo @ revision | bytes | gating |
|---|---|---:|---|
| `diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors` | `Lightricks/LTX-2.5` @ `5e6e7101` | 21,504,034,224 | `gated: auto` |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | `Lightricks/LTX-2.5` @ `5e6e7101` | 15,372,969,374 | `gated: auto` |
| the fixed-prompt cache itself | derived | < 64 MiB | built once from the row above |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` (alternative) | `Lightricks/LTX-2.5` @ `5e6e7101` | 26,263,858,182 | `gated: auto` |
| an FP8 H3 DiT checkpoint (stage 1) | not in this set | ~15.5 GB | — |

The store shows the operator already staging the gap: `LTX-2.5/diffusion_models/`
and `LTX-2.5/latent_upscale_models/` exist and are empty, so the connector and
the official LTX upscalers are anticipated, while `text_encoders/` has not been
created. Two further facts narrow the gate. The upscaler file exists under that
name only at the pinned revision - the current default branch shows it moved to
`minimax_h3_latent_upscaler_3d_conv_v1/` as `..._3d_conv_v1_bf16.safetensors` -
so the pinned revision is the one to fetch, not `main`. And `Lightricks/LTX-2.5`
is gated for *download* but not for *metadata*: its tree listing is public, and
the operator's four present files come from that same repo at that same
revision, so access is proven rather than merely possible.

The prompt cache is the only real gate, and it gates conditioning alone. Its
shape is fixed and so is its recipe: `validate_payload`
(`prompt_cache.py:39-62`) rejects any payload whose `recipe` dict differs from
the module's `RECIPE` at all, so a cache built from the BF16 Gemma (26.3 GB)
would be refused by the release loader unless the recipe is rewritten - a named
deviation whose numeric effect (int8-convrot contexts vs BF16 contexts) would
have to be measured against the release's own cache. The int8 pair is the
faithful route and lives behind the licence the operator already accepted.

## 7. What the port must mirror

Ordered by what the plan's phases need, with the file that defines each:

1. **Store and sampler** (`diffusion_steps.py:24-40`, `samplers.py:37-121`):
   Euler, three updates, the literal sigma list, `to_velocity` and `to_denoised`
   (`utils.py:21-35`), the mask blend in `post_process_latent`
   (`helpers.py:257-259`). Stage 2's `noise_scale` is `sigmas[0]` = 0.909375.
2. **The AV block** (`transformer.py:254-420`): AdaLN modulation
   `rms_norm(x) * (1+scale) + shift` (`ops.py:50-58`), self-attention,
   `x + attn*gate` with the gate from the same AdaLN triple, text cross-attention
   with the query-side AdaLN and the prompt table's K/V modulation
   (`transformer.py:223-253`), the A-V pair with its 5-row tables, the MLP, and
   per-head gating `2*sigmoid(logits) * out` (`ops.py:95-107`). Note the
   A2V/V2A ordering invariant: both directions read pre-A2V state
   (`transformer.py:330-333`, with the comment stating why). The 5-row AV table
   splits 4 scale/shift rows + 1 gate row (`transformer.py:202-216`,
   `num_scale_shift_values=4`), and `get_ada_values`
   (`transformer.py:191-200`) is a table row plus a reshaped timestep
   projection - the same `scale_shift_table` pattern all six F32 tables use.
3. **Preprocessing** (`transformer_args.py:113-270`, `model.py:47-79`):
   patchify (`patchifiers.py`, one token per latent voxel), the two timestep
   paths (`timestep_scale_multiplier` 1000 and its own prompt path), the
   cross-modality AdaLN inputs, and RoPE with `theta 10000`, `max_pos
   [20,2048,2048]`, `use_middle_indices_grid` midpoints and the float64 numpy
   frequency grid (`rope.py:86-108`, selected by `frequencies_precision
   float64`).
4. **The connector** (`text_encoders/gemma/embeddings_connector.py`): 8 blocks per
   modality, learnable registers, `PER_TOKEN_RMS` feature extractor
   (`feature_extractor.py:100-129`). Only needed offline, once.
5. **The adapter and upscaler** (`stage2_ops/h3_ltx_adapter/`,
   `stage2_ops/h3_upscale.py`): pure Conv3D, shape-checked at every seam.
6. **The Conv VAE decoder** with the profile's tiles, emitting NHWC directly
   (`stage2_ops/conv_direct_nhwc.py`), then the mux at CRF 19
   `veryfast` (`stage2.py:124-126`).

The stage-1 side needs no reference reading: the four-step H3 draft at 672x384
is our own DiT at a token count we have already measured, with FP8 weights and
the FastH3 VSA LoRA, and the three stage-1 levers already queued in the
attribution record are the same 1.27x.
