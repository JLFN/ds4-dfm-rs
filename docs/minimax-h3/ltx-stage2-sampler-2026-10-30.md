# The LTX-2.5 stage-2 sampler, and the first evaluation (unit 8)

Date: 2026-10-30. Branch `feat/sol-h3`, commits `3df3780a` (the sampler),
`30cc86c1` (the unpatchify fix). Reference pin `/data/ltx2-pin/packages`
(Lightricks/LTX-2 `d1511477` = tag v1.2.0); release runtime
`/data/Sol-h3-Spark` @ `6e4a440`.

Unit 7 ended with the whole per-step graph running: preprocessing, 48 blocks,
head. Nothing walked the schedule, nothing turned a velocity into an x0, and
nothing got a latent in or out. This unit is the schedule plus that plumbing, so
the port can run an evaluation of the release's own geometry end to end.

## 1. What landed

- `crates/ds4-core/src/minimax_h3/ltx_stage2.rs`: `SIGMAS`, `Generator`,
  `StageScratch`, `Stage2::{open, noised_state, step, run, latent,
  latent_grid, noise, seed, generator}`.
- Seven device kernels and their `ds4-sys` wrappers: `h3d_randn_bf16`,
  `h3d_noise_lerp`, `h3d_timesteps`, `h3d_to_denoised`, `h3d_to_velocity`,
  `h3d_euler_step`, `h3d_post_latent`.
- The refiner's route became video-only, with the reference's four guards; the
  patchify moved out of `prepare` into `patchify_video`/`patchify_audio`.
- A `--stage` mode in the gate, plus seven probes for the new kernels.
- `unpatchify_video`'s swapped transpose arguments, found by the first call it
  ever had.

## 2. The reference route

`RefinerEncoderAblationStage2` denoises with **no audio state**:

- `prepare_denoise(upscaled, seed)`
  (`super_acceleration/stage2/refiner_encoder_ablation_single_gpu.py:691-701`)
  builds one generator, calls `create_noised_state(... noise_scale=
  STAGE2_SIGMAS[0], initial_latent=latent)` and returns it.
- `denoise(state, context)` (`:704-714`) runs
  `euler_denoising_loop(sigmas, video_state=state, audio_state=None,
  stepper=EulerDiffusionStep(), transformer=self.wrapped_transformer,
  denoiser=SimpleDenoiser(v_context=context, a_context=None))`.
- `finish_denoise` (`:718-721`) is `clear_conditioning` then `unpatchify`.
- The schedule is `SIGMAS = (0.909375, 0.725, 0.421875, 0.0)`
  (`runtime/stage2_ops/models.py:19`), and `euler_denoising_loop` iterates
  `sigmas[:-1]` (`ltx-pipelines/src/ltx_pipelines/utils/samplers.py:73`), so
  three intervals.

One interval is five reference expressions, each transcribed as one kernel or
one host step:

| step | reference | where |
|---|---|---|
| the noise | `torch.randn(x.shape, generator=g, dtype=x.dtype)` | `components/noisers.py:23-29` in `GaussianNoiser`, reached from `create_noised_state` (`utils/helpers.py:442-445`) |
| the two lerps | `torch.lerp(latent.f32, noise.f32, noise_scale)`, then `torch.lerp(clean.f32, latent, mask)` | `noisers.py:30-37` |
| the timestep | `timesteps_from_mask`: `denoise_mask * sigma` | `utils/helpers.py:494-501` |
| x0 | `to_denoised(latent, vx, video.timesteps)` | `model/transformer/model.py:590-605` |
| the blend | `post_process_latent`: `denoised*mask + clean.float()*(1-mask)` | `utils/helpers.py:461-464` |
| the step | `EulerDiffusionStep.step`: `to_velocity(sample, sigma, denoised)` then `sample + velocity*dt` | `ltx-core/src/ltx_core/components/diffusion_steps.py:32-40` |

`to_denoised` and `to_velocity` themselves are
`ltx-core/src/ltx_core/utils.py:39-52` and `:21-35`.

Two details are easy to lose and are transcribed rather than simplified:

- `to_denoised` takes **`video.timesteps`**, the per-token `mask * sigma`, not
  the scalar sigma the block AdaLN takes. With the release's all-ones mask the
  two are the same number, which is exactly why the port keeps the per-token
  tensor: `StageScratch::timesteps` is `h3d_timesteps`'s output and both the
  kernel and the probe read it.
- `torch.lerp` is not `a + w*(b-a)` unconditionally. `aten/src/ATen/native/
  Lerp.h:22-33` selects `self + w*(end - self)` when `|w| < 0.5` and
  `end - (end - self)*(1 - w)` otherwise, so the branch is part of the value.
  `noise_scale` is 0.909375 and takes the second arm; the mask is exactly 1,
  where that arm is the identity (`end - (end - self)*0`). Both arms are in
  `h3d_lerp`.

## 3. The video-only correction

Unit 6's block ran every branch, and its comment said so: "Every branch runs:
`run_vx` and `run_ax` are always true on this route". That was wrong. The
release's stage-2 denoise passes `audio_state=None`, so
`BasicAVTransformerBlock.forward`'s guards
(`model/transformer/transformer.py:265-269`) come out

```
run_vx  = video is not None and video.enabled and vx.numel() > 0   -> true
run_ax  = audio is not None and audio.enabled and ax.numel() > 0   -> false
run_a2v = run_vx and (audio is not None and ax.numel() > 0)        -> false
run_v2a = run_ax and (video is not None and vx.numel() > 0)        -> false
```

so only the video self-attention, its text cross-attention and its video
feed-forward run. The audio state does not exist and the AV cross-attention
never fires. `MultiModalTransformerArgsPreprocessor.prepare` likewise returns
the simple args untouched for `cross_modality=None`
(`model/transformer/transformer_args.py:355-356`), so the AV cross AdaLN, the
two gates and the cross RoPE are not computed either.

The port now mirrors that: `Request.audio`/`audio_context` are `Option`,
`Prepared.audio`/`audio_signals` are `Option`, `Signals.cross` is
`Option<Cross>`, and `Block::apply` takes `audio: Option<(&Tensor, &Signals)>`
whose presence is exactly the four guards above (`Some` implies all four true).

The same correction applies to the input: the reference patchifies **once**, in
`VideoLatentTools.create_initial_state` (`ltx-core/src/ltx_core/tools.py:139-184`),
and the loop re-feeds that patchified state unchanged - `patchify_proj` reads
`modality.latent`, which is already `[tokens, 128]`
(`transformer_args.py:268`). So `Request.video` is now the patchified state and
the transpose moved to `patchify_video`/`patchify_audio`, whose only caller is
the state's creation.

## 4. The noise is torch's own draw

`GaussianNoiser._sample_noise` is `torch.randn(*latent.shape, device=...,
dtype=latent.dtype, generator=self.generator)` on a bfloat16 state, which
reaches `normal_kernel<BFloat16>` (`ATen/native/cuda/DistributionTemplates.h:461-476`)
and `normal_and_transform<BFloat16, float>` with `curand_normal4`
(`:443-455`), inside `distribution_elementwise_grid_stride_kernel<accscalar_t,
4>` (`:66-89`). `h3d_randn_bf16_kernel` is that kernel with the same
`calc_execution_policy` grid (`:50-62`), the same `curand_init(seed, idx,
offset)`, the same `float4`-wide `#pragma unroll` and the same
`__float2bfloat16` store (`c10/util/BFloat16-inl.h` -> `torch/headeronly/util/
BFloat16.h:125-130`; `transformation::normal<float>(v, 0, 1)` is `v*1 + 0`,
`ATen/core/TransformationHelper.h:96-99`).

The `(seed, offset)` pair is torch's too: `manual_seed(seed)` stores the seed
and zeroes `philox_offset_per_thread_`
(`CUDAGeneratorImpl::set_current_seed`, and an empty capture id), and
`philox_cuda_state` hands out the current offset before adding the increment -
so `Generator` stores `(seed, offset)` and the device returns the increment the
host adds.

This has to run on the device. `curand_normal4` is `curand_box_muller4`
(`curand_normal.h:165-179`) over `curand4`: `sqrtf`, `logf` and `__sincosf` run
on the device's own units, and the reference ran on GB10. A host
reimplementation of Box-Muller with libm would differ in the last bits of every
value, which is why the port calls curand's own headers instead of
reimplementing Philox and Box-Muller.

## 5. The kernels

Each is one reference expression, float32 roundings and branch selection
included. `h3d_lerp` is the two-arm `torch.lerp`; `h3d_timesteps` is
`denoise_mask * sigma`; `h3d_to_denoised` is `latent - velocity*timestep`;
`h3d_to_velocity` is `(sample - denoised)/sigma`; `h3d_euler_step` is
`sample + velocity*(sigma_next - sigma)`; `h3d_post_latent` is
`denoised*mask + clean*(1-mask)`; `h3d_noise_lerp` is the noiser's two lerps.
`to_velocity` and the step are deliberately two kernels: the reference rounds
the velocity to bfloat16 before the sum, and one fused kernel would hide that.

## 6. Measurements

Probes: 24, on sm_89 and on sm_121, all pass, no FAIL on either host.

| probe | sm_89 | sm_121 |
|---|---|---|
| `timesteps`, `to_velocity`, `patchify_pair` | exact | exact |
| `to_denoised`, `euler_step`, `post_latent`, `noise_lerp` | `differ=0` (residual verdict) | `differ=0` |
| `randn` | `unfilled=0`, `sd=0.996252`, `fnv=8cb73e5f217fe213` | `unfilled=0`, `sd=0.996252`, `fnv=8cb73e5f217fe213` |

The four steps reported as residues are in fact bit-identical on both hosts:
nvcc did not contract them, or the contraction did not survive the bfloat16
round. `patchify_pair` is the round trip that caught the swapped
`unpatchify_video` (§8).

**The noise is bit-identical across the two architectures**: 4097 elements of
`torch.randn` at seed 20251025 hash to the same value on the 4070 SUPER
(sm_89) and on GB10 (sm_121). That is the strongest evidence this campaign can
get without the golden: it says the draw is decided by the algorithm rather
than by the device's `sqrtf`/`logf`/`__sincosf`, so the only remaining risk in
it is the transcription of torch's *harness* - the grid formula, the subsequence
and the offset bookkeeping - which is read from torch's own sources (§4) and
not from a guess.

`--stage --layers 48` on the Spark, seed 20251030, synthetic latent and
context, release geometry (16,128 tokens, 128 wide), two runs:

```
run 1: load=6.692 s  steps=125.774 / 126.837 / 127.806 s  denoise=380.437 s
run 2: load=6.710 s  steps=126.022 / 127.106 / 127.995 s  denoise=381.143 s
       wall=389 s, /usr/bin/time -v maximum RSS 645968 kB
noise    fnv=2eac5cb8bb2aa957   high=5.031250  low=-5.281250
state0   fnv=a7991fc33188f068
step0    fnv=2f0eb3015a958e30
step1    fnv=d22edcb9bf77ac7d
step2    fnv=03a31966cef75544
latent   fnv=4266f9e3f93be058   high=4.500000  low=-4.718750
generator offset after one draw = 28
```

The two runs agree to 0.2% per step and the state hashes are identical, which
is what a deterministic path should look like. The final latent's extremes are
step 2's, as they must be: unpatchify is a permutation.

Memory peak 44 GiB used of 121 GiB (`free -g`); `/usr/bin/time -v`'s maximum
RSS is 631 MB and says nothing about the device, as in unit 7. 126.3 s per step
against unit 7's 141.4 s is the video-only route skipping the audio branch and
the AV cross-attention - not an optimization, the removal of work the reference
does not do.

## 7. What it does not prove

Every value gate here is still a host transcription of the reference's own
expression, so a shared misreading survives it - and this unit adds two shared
assumptions the earlier ones did not have:

1. **The generator's `(seed, offset)`.** `manual_seed` -> `set_current_seed`
   and `philox_cuda_state`'s "hand out then advance" are read from
   `CUDAGeneratorImpl.cpp` at the version the Spark's wheel holds, not run. A
   wrong offset shifts the whole noise field and nothing here would notice,
   because every noise check is a self-consistency check.
2. **`curand_normal4` is the reference's own.** The kernel calls the same
   headers, and §6 shows the values are architecture-independent between sm_89
   and sm_121, so the failure mode is the harness rather than the transform -
   but no value here is compared against a torch run. The probes pin a full
   fill, determinism at one generator state and a fresh draw at the advanced
   offset; they cannot pin "these are torch's numbers" without the golden.

The final latent has no correctness check at all beyond its own round trip:
neither producer of the two inputs is run by this gate, so the latent and the
context are synthetic and the values are meaningless even though every kernel
on the path is exercised at the release's shapes and the per-step cost is real.
(Corrected 2026-10-30: an earlier draft of this section said the latent
upsampler "is not ported". It is — `minimax_h3::ltx_adapter`'s `Upscaler` and
`Adapter`, gated by `h3_ltx_adapter_gate`, landed in
[`latent-path-2026-10-29.md`](latent-path-2026-10-29.md). What is missing is
that the stage mode does not run it.)

## 8. The `unpatchify_video` defect

`unpatchify_video` called `h3d_channel_transpose(latent, output, ...)` while
that entry point takes `(output, input)` first, so it wrote the grid into the
state and left the grid untouched. Nothing called it until this unit's
`latent_grid`, whose first output read back as all zeros -
`STAGE stage-latent ... high=0.000000 low=0.000000`, a silent failure. It is
fixed in `30cc86c1`, and the new `patchify_pair` probe takes a state through
unpatchify and back through patchify: it reports `differ=638` of 640 on the
swapped pair and `differ=0` on the fix.

This is the campaign's own lesson repeating: a path with no caller has no gate.

## 9. What is not ported

The Sol-Attn policy (unit 9, and it is the 17x: 126.7 s against the release's
8.2 s per step), the decoder, the audio unpatchify, the generated keyframe
slots, the conditioning items, the attention_mask path, the `res2s` and
ancestral samplers, and the LoRA merge at strength 0.8. (The latent upsampler
is ported; the stage mode does not run it. See section 7.)

## 10. Reproduction

```
# workstation (sm_89)
make tests/h3_ltx_refiner_gate CUDA_ARCH=sm_89
./tests/h3_ltx_refiner_gate --probe
cargo test -p ds4-core --lib 3<&-

# Spark (sm_121)
rm -f native/h3/*.o && make CUDA_ARCH=sm_121 tests/h3_ltx_refiner_gate
./tests/h3_ltx_refiner_gate --probe
./tests/h3_ltx_refiner_gate --edges --transformer \
  /mnt/h3-artifacts/LTX-2.5/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  --rows 8 --audio-rows 4 --blocks 1 --seed 3
./tests/h3_ltx_refiner_gate --stage --transformer \
  /mnt/h3-artifacts/LTX-2.5/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  --layers 48 --seed 20251030 --dump ~/ds4-test/stage8-dump
```

## 11. Owed decisions (unchanged)

1. **The golden dump** for units 2b-8. The sampler raises the stakes: the RNG
   and the schedule are new shared assumptions, and nothing in this campaign
   can check them.
2. **A real input latent and context.** The stage mode runs synthetic ones; the
   upsampler and the connector are the two producers.
3. The written cache's `torch.load` acceptance (unit 5's direction is argued
   from torch's source).
4. The INT8 connector file: `distilled`, or must `dev` be fetched.
5. The decoder's route: unset (F32, the reference) 110.4 s, `bf16` 30.4 s,
   `bf16i` 24.4 s on job A.
