# Unit 7: the LTX-2.5 refiner's edges - the preprocessing and the output head

Unit 6 gave a verified block that consumes `Signals`; nothing produced them and
nothing consumed the block's output. Both ends are now in the port: `prepare`
turns the two latents, the step's sigma and the two prompt contexts into the
states and `Signals` the 48 blocks move, and `head` runs `_process_output` on
both modalities. What is NOT in this unit is named in section 7: the sampler,
the Sol policy, the cache plumbing and the audio unpatchify.

Source of truth read first-hand this unit: `/data/ltx2-pin/packages` (the pin
the release runs, `d151147788a9284cca791edc6ce898007e727fe6` = tag v1.2.0),
`ltx-core/src/ltx_core/model/transformer/{transformer_args,adaln,model,
timestep_embedding,rope}.py`, `components/patchifiers.py`, `tools.py`,
`types.py`, and the release's own stage-2 wiring at
`/data/Sol-h3-Spark/models/minimax_h3/Sol-H3-Spark/runtime/stage2_ops/models.py`
plus `ltx-pipelines/src/ltx_pipelines/utils/{helpers,denoisers}.py`.

## 1. What landed

| file | what |
|---|---|
| `crates/ds4-core/src/minimax_h3/ltx_refiner.rs` | `Geometry`, the positions, `rope_tables`, `sinusoid`, `AdaLn`, `EdgeScratch`, `Head`, `Edges::{open,prepare,head}`, `unpatchify_video` |
| `crates/ds4-core/examples/h3_ltx_refiner_gate.rs` | the two new probes, the `--edges` mode, the keyframe check, the head transcription |
| `native/h3/h3_dit_gpu.{cu,h}` | `h3d_audio_patchify`, `h3d_keyframe_add` |
| `crates/ds4-sys/src/h3_dit.rs` | the two externs and their safe wrappers |

The 59 tensors outside the blocks load through the same `Reader` that checks
each descriptor against the graph before anything reaches the device: the two
patchify projections, the keyframe embedding, the eight AdaLN singles, the two
model-level `scale_shift_table`s and the two output projections. That is
854,954,496 B against the blocks' 37.1 GB.

## 2. The preprocessing, in the reference's own order

1. **Tokens.** The video's `b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)`
   at patch size 1 is the transpose unit 6 already landed
   (`patchifiers.py:27-38`); the audio's `b c t f -> b t (c f)`
   (`:293-304`) is its own kernel, because the `bins` axis stays innermost.
2. **Positions** are patch bounds in seconds. `get_patch_grid_bounds`
   (`:71-133`) gives `[f, h, w]` integer bounds; `get_pixel_coords`
   (`:137-170`) scales them by `SpatioTemporalScaleFactors.default()` =
   `(8, 32, 32)` (`types.py:31-33`) and applies the causal first-frame fix
   `(t * 8 + 1 - 8).clamp(min=0)` on the temporal axis only; `tools.py:174`
   divides that axis by 24 fps. The audio patchifier is already in seconds
   (`:215-283`, mel frame `i * 4`, causal offset 1, `160 / 16000` seconds per
   mel frame) and the audio tools never scale it (`tools.py:246-278`).
3. **RoPE tables.** `precompute_freqs_cis` (`rope.py:198-223`) with the float64
   grid (`:86-108`, the config's `frequencies_precision: "float64"`):
   `generate_freqs` (`:137-160`) takes each patch's midpoint (`:149-155`),
   divides by `max_pos`, and forms `index * (2 * fractional - 1)` over the
   per-axis angles. Video: `4096 // 6 = 682` angles over 3 axes = 2046, and
   `split_freqs_cis` (`:163-184`) prepends **2 padding entries** (cos 1, sin 0)
   to reach the 2048 half-width before the head split; audio: 1024 and the
   cross `pe` 2048 at one axis, both without padding. All cast to BF16.
   One layout note: the reference reshapes to `(b, heads, t, half)` and swaps
   because it rotates a head-major tensor (`:179-183`); this port keeps the
   states token-major and writes the tables `[row][head][half]` - the same
   values in the layout `h3d_ltx_rope_pair` reads.
4. **The AdaLN singles.** `timestep * 1000` through the 256-wide sinusoid
   (`timestep_embedding.py:6-52` at `:133`'s parameters: `flip_sin_to_cos`
   true, `downscale_freq_shift` 0), then `linear_1 -> SiLU -> linear_2` is the
   `embedded_timestep` and `SiLU -> linear` on top of it the
   `coefficient * dim` modulation (`adaln.py:19-45`). Eight of them: 9 for the
   block, 4 for the A-V scale/shift, 1 for each gate and 2 for each prompt
   (`model.py:221-287`). The A-V gate is fed the **other** modality's sigma
   (`transformer_args.py:388-410`); `av_ca_timestep_scale_multiplier` is the
   same 1000.0, so `av_ca_factor` is 1.0.
5. **The keyframe term.** `apply_keyframes_absolute_embedding`
   (`:22-43`) adds the learned `(1, 4096)` row where the mask is set, and
   `_first_frame_keyframes_mask` (`tools.py:186-198`) marks the first latent
   frame unconditionally - 1,008 of 16,128 tokens at this geometry.
6. **The context** is the cached `(1, 1024, dim)`; `caption_projection` is
   None, so `_prepare_context` only reshapes it (`transformer_args.py:186-196`)
   and the port borrows it rather than copying.

The release's own path is the one exercised: `SimpleDenoiser` expands the
scalar schedule value to `(B,)` for both modalities (`denoisers.py:236-246`),
`timesteps_from_mask` multiplies the all-ones `denoise_mask` by that sigma
(`helpers.py:494-505`), so every token carries the same timestep and the
per-token modulation's rows are identical. It is the *shape* - `[rows, 9, dim]`
and `[rows, 4, dim]` - that the kernels validate, and it is the shape the
reference's own expressions produce.

## 3. The output head

`_process_output` (`model.py:472-490`): the model-level table combined with the
embedded timestep - **row 0 the shift, row 1 the scale** (`:486-488`) - the
affine-free LayerNorm (`:487`), `x * (1 + scale) + shift` per token (`:488`),
and `proj_out` to 128 channels (`:489`). The audio head is the same at 2048
(`:528-536`). `unpatchify_video` is the transpose again, so the head's
`[tokens, 128]` returns to a `(1, 128, 16, 24, 42)` latent.

## 4. The two new kernels

| kernel | reference | rounding chain |
|---|---|---|
| `h3d_audio_patchify` | `patchifiers.py:287-304` | a pure gather, no arithmetic |
| `h3d_keyframe_add` | `transformer_args.py:36-43` | the mask is exactly 0 or 1, so `mask * embedding` is exact and the whole expression is one BF16 add |

## 5. The measurements

Both hosts, the same seed and the same shard
(`/mnt/h3-artifacts/LTX-2.5/ltx-2.5-22b-dev-transformer-bf16.safetensors`):
sm_89 is the workstation's RTX 4070 SUPER (CUDA 13.3), sm_121 the GB10
(CUDA 13.4).

### 5.1 Probes: all thirteen bit-identical on both hosts

The eleven unit-6 probes plus `audio_patchify` and `keyframe_add`, each against
a host transcription of the same expression: `differ=0`, `rel_rms=0.000e0`,
including the two that end in `rsqrtf`.

### 5.2 The keyframe term, isolated and oracle-free

Against a shard whose `keyframes_abs_pos_embedding` is zero the state *is* the
projected latent, so the real state must differ from it by exactly the
embedding row - `bf16(base + embedding)` - on exactly the rows the mask marks
and nowhere else. Both hosts: `KEYFRAME marked_rows=2 rows=8 differ=0`.

### 5.3 The head against a host transcription

The transcription stops before `proj_out` and hands its own modulated values to
the *device's* linear with the same weights, so both sides run one
`h3d_linear` and the comparison is bit-level apart from the LayerNorm's
`rsqrtf`. sm_89: `head_video` `rel_rms=1.787e-3`, `head_audio=1.948e-3`;
sm_121: `head_audio=1.992e-3`. Against a 2e-2 ceiling, and three orders below
the order-1 deviation a swapped table row or a missing norm would produce.

### 5.4 Cross-host residuals, release geometry

`h3d_linear` is cuBLASLt and not bit-stable across architectures, so the two
hosts' *hashes* differ while their extrema agree exactly. Against the 2e-2
ceiling:

| buffer | elements | differ | rel_rms |
|---|---|---|---|
| video state | 66,060,288 | 436 | 8.012e-6 |
| audio state | 258,048 | 2 | 3.696e-6 |
| video modulation | 594,542,592 | 2,112,768 | 1.378e-4 |
| video cross scale/shift | 264,241,152 | 177,408 | 2.339e-4 |
| video cross gate | 4,096 | 0 | 0.000e0 |
| video prompt modulation | 8,192 | 0 | 0.000e0 |
| video self cos / sin | 33,030,144 each | 0 | 0.000e0 |
| video cross cos | 16,515,072 | 0 | 0.000e0 |
| audio modulation | 2,322,432 | 882 | 1.473e-4 |
| audio self cos | 129,024 | 0 | 0.000e0 |
| audio cross scale/shift | 1,032,192 | 33,894 | 4.721e-4 |
| block 0 video state | 66,060,288 | 4,955,801 | 1.074e-3 |
| head video | 2,064,384 | 403,787 | 1.650e-3 |
| head audio | 16,128 | 6,072 | 2.890e-3 |

The four RoPE table pairs, the two gates and the prompt modulation are
**bit-identical** across hosts: they are host arithmetic and one-row linear
work, and this is the strongest cross-host agreement the campaign has.

At the small geometry (8 video rows, 4 audio rows) the same table gives
`video state` 436 -> 1 differing element (5.293e-6), `audio state` 0, and the
same modulation residues, so the residue is a property of the linear layers,
not of the row count.

### 5.5 The first resident 48-block run

All 48 blocks loaded on the Spark (37.1 GB of block weights plus 855 MB of
edges), one step of the release's own geometry (16,128 video + 126 audio
tokens, sigma 0.909375, the first of `STAGE2_SIGMAS`):

| phase | sm_89 | sm_121 |
|---|---|---|
| edges load (59 tensors, 855 MB) | 9.749 s over gvfs | 0.207-0.631 s |
| prepare | 0.759 s | 0.905-0.926 s |
| 48 blocks | 114.8 s (2.392 s each) | 141.4-141.8 s (2.95 s each) |
| head | under a millisecond | 0.007-0.026 s |

Three independent Spark samples of the 48-block loop: 141.415, 141.837 and
141.435 s, a 0.3% spread. One evaluation is therefore ~142 s on GB10 against
the release's own **8.2 s per step** (24.71 s for three updates with the Sol
policy, the figure unit 6 recorded from the reference's Spark run). Dense
attention at every layer is the whole difference: unit 6 measured it at ~65% of
a block, which is what the Sol policy replaces.

The 48-block load is not separately timed by the gate; the running process's
wall clock minus its own work puts it at ~24 s, i.e. ~1.5 GB/s off `/dev/sda`
for the 37.9 GB.

**Footprint.** `nvidia-smi` reports `[N/A]` on this box and the process's own
resident set is 871 MB - GB10 is unified memory and the device allocations are
outside it - so the number comes from the machine's own view while the blocks
run: 2 GiB used and 119 GiB available at idle, **44-45 GiB used and 76 GiB
available at the peak**, and 2 GiB used again after the process exits (the
allocator returns everything). The predicted 45 GB - 37.9 GB of weights, 3.3 GB
of block scratch, 1.3 GB of edge scratch and 2.5 GB of prepared signals - is
what the machine reports.

The per-block state extrema grow from `high=268` at block 0 to `high=33280,
low=-576` at block 47. That is the model's own behaviour on the gate's
synthetic latents (an LCG draw, not the reference's input distribution), not a
fault: block 0's extrema are identical on both hosts, and the growth is smooth
(about 1.05x per block, never a jump).

### 5.6 The preprocessing is not free

`prepare` is 0.759 s of the 142 s at the release geometry, and most of it is
the per-token AdaLN MLPs: 16,128 rows through `256 -> 4096 -> 4096 -> 36,864`
for the block modulation alone, plus a second per-token pass for the A-V
scale/shift. The buffers are real memory too: the modulation is 1.19 GB, the
cross scale/shift 528 MB, the two embedded timesteps 264 MB.

## 6. What this unit does not prove

- That the *values* match the reference. Every gate here is either a host
  transcription of the reference's own expression (which cannot catch a shared
  misreading) or a cross-host comparison (which cannot catch one either). The
  owed golden dump is still the only thing that can, and it is still owed.
- The per-token/per-request distinction in the head: the release's denoise
  mask is all ones, so every row's embedded timestep is the same value and a
  `scale_rows` mistake would be invisible in the values. The kernels' own size
  validation is what pins the shapes.
- Anything about the sampler: sigma is an input here, and the port has run one
  step, not a schedule.
- The audio unpatchify, the generated-keyframe slots, conditioning items and
  the `attention_mask` path: not ported, not exercised.

## 7. What is still not ported of the refiner

The sampler (`STAGE_2_DISTILLED_SIGMA_VALUES`, the mask blend, `GaussianNoiser`,
the frozen-audio path), the Sol-Attn policy and its backend, the decoder's
tiling, the cache-to-latent plumbing (`stage2_ops/h3_upscale.py`), the audio
unpatchify, and the LoRA merge the release applies at load
(`LoraPathStrengthAndSDOps(..., 0.8, LTXV_LORA_COMFY_RENAMING_MAP)`).

## 8. Evidence and reproduction

```sh
# workstation, sm_89
rm -f native/h3/*.o && make CUDA_ARCH=sm_89 tests/h3_ltx_refiner_gate
tests/h3_ltx_refiner_gate --probe
M=.../ltx-2.5-22b-dev-transformer-bf16.safetensors
tests/h3_ltx_refiner_gate --edges --transformer $M --blocks 1 --rows 8 --audio-rows 4 \
  --seed 7 --signals --keyframe-zeroed /tmp/refiner-edges-zero.safetensors
tests/h3_ltx_refiner_gate --edges --transformer $M --blocks 1 --rows 16128 \
  --audio-rows 126 --seed 7 --signals --dump /tmp/edges-full
tests/h3_ltx_refiner_gate --edges --transformer $M --blocks 1 --rows 16128 \
  --audio-rows 126 --seed 7 --signals --residual /tmp/edges-full-spark

# Spark, sm_121 (script ~/ds4-test/unit7-spark.sh, log unit7-spark.log)
rm -f native/h3/*.o && make CUDA_ARCH=sm_121 tests/h3_ltx_refiner_gate
tests/h3_ltx_refiner_gate --edges --transformer $M --blocks 48 --rows 16128 \
  --audio-rows 126 --seed 7

# hygiene
cargo test -p ds4-core --lib 3<&-   # 755 passed, 7 ignored
cargo clippy -p ds4-core --example h3_ltx_refiner_gate --features h3-gates
```

## 9. Owed operator decisions (unchanged)

1. **The written cache has never been checked with `torch.load`**; the write
   direction's acceptance is argued from torch's source (unit 5). A torch
   2.14.1+cu130 tree exists in the Spark's uv cache.
2. **The golden dump** for units 2b-7: every value gate this campaign has is a
   host transcription of the reference's own expression, which cannot catch a
   shared misreading (5b was exactly that).
3. **The INT8 connector file**: is the `distilled` file acceptable, or must
   `dev` (sha256 `2edbdb44...caf92b`) be fetched?
4. **Whether to build the delivered cache on the Spark** with the processed
   weights (SM121-only per `runtime/cache_builder.py:105-107`).
5. **The decoder's route**: unset (F32, the reference) 110.4 s, `bf16` 30.4 s,
   `bf16i` 24.4 s on job A.
