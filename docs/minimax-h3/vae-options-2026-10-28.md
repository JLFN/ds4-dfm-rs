# The video VAE decoder's bench options, 2026-10-28

This unit gives the largest untouched block of the H3 job an option surface and
measures the arms. The block is the F32 video VAE decoder, which the 768x768
107-frame 2-step job spends 197.945 s inside (71.5% of its 277.019 s wall,
[speed-2026-10-27.md](speed-2026-10-27.md)); the campaign's remaining plan at
that point was deeper work on the DiT attention, which is 40.294 s of the same
wall. Nothing here changes a default: every option is off unless a job asks, so
a run that sets nothing is the reference path.

Branch `feat/h3-vae-options` off `509ac111`. `h3-rs` sha256
`c6a32ebe7802d07be8b9feacad0e986d162576afa2c5a83a9fc549c86fb20b12`, 5,177,960
bytes, built on the Spark with `rm -f native/h3/*.o && make h3-rs` under
`CUDA_ARCH := sm_121`.

## The option surface

Read once per context in `h3d_create` from the environment, and settable per
run with `h3d_set_vae_option(context, name, value)` from the host, which refuses
an unknown name or value rather than defaulting silently.

| Variable | Values | Default | What it selects |
|---|---|---|---|
| `H3_VAE_ATTENTION` | `scalar`, `tensor` | `scalar` | the decoder's F32 attention kernel |
| `H3_VAE_SDPA_ORDER` | `heads`, `rows` | `heads` | that kernel's CTA order only |
| `H3_VAE_GEMM` | `fp32`, `tf32`, `bf16` | `fp32` | the decoder's F32 matmul compute mode |

Only the decoder reads them. The matmul option is a separate entry point
(`h3d_vae_linear`, called from `minimax_h3::vae_video`) rather than a flag on
`h3d_linear`, because the video VAE encoder and the audio VAE share the F32
matmul and their gates compare against the reference bit for bit; a global
switch could have broken them silently.

## The workload model this unit corrects

The campaign's records describe the decoder's attention as "12 tiles, 432 calls
per decode" at "16,133 rows for the 768 job". Neither is what the decoder runs.
`configured_tile_pixels` searches 256..=320 pixels and picks the smallest
`tiles x pixels^2`, which at 768x768 is a 304-pixel tile - the reference's own
"tiles 3x3 at 304 px" in the phase table. `tile_axis` then gives three tiles per
axis, so a tile is 19x19 latents and `CHUNK_LATENT_TIME * 19 * 19 + SUFFIX` =
2,532 rows, and `chunk_plan` gives `(32 - 2) / 5` = 6 chunks: 54 tile
submissions, 1,944 block calls. The 512x512 job takes 288-pixel tiles, 18x18,
four per chunk over three chunks: 12 submissions of 2,273 rows, 432 calls.
The 16,133 and 7,173 figures are untiled full-plane shapes and 432 calls belongs
to the 512 job; 432 x 1.698 s = 734 s, which does not fit the 197.945 s phase
they were quoted against. `crates/ds4-core/src/minimax_h3/vae_video.rs:156-205`,
pinned by `tile_plan_matches_the_reference_at_the_released_shapes`, and the
reproducer header now say so.

## The F32 attention at the shapes the decoder passes

`tests/h3/attn_f32_bm <rows> 32 5`, three runs an arm, on the Spark. Shapes
2,532 and 2,273 had never been measured before this unit.

| Rows | Scalar | Tensor core | Ratio |
|---|---:|---:|---:|
| 2,532 (768 job tile) | 41.29 / 41.35 / 41.19 ms | 1.333 / 1.333 / 1.331 ms | 31.0x |
| 2,273 (512 job tile) | 33.42 / 33.35 / 33.33 ms | 1.087 / 1.089 / 1.087 ms | 30.7x |
| 16,133 (the campaign's bound) | 1,681.7 / 1,692.2 / 1,703.3 ms | 52.5 / 52.4 / 52.5 ms | 32.2x |

At 1,944 calls the 768 job's attention is 80.3 s scalar against 2.59 s tensor,
which is 40.6% of the 197.945 s phase - the share the earlier attribution could
only extrapolate. The gate hash holds throughout: the scalar arm prints
`56468a4a8fd2a2de` at 16,133 on both hosts.

The tensor kernel's own resources, `ptxas -v` at sm_121a: 124 registers, no
spills, no stack, 18,432 bytes of shared memory, against the BF16 DiT kernel's
168 registers and 34,816 bytes.

## Deviation and quality

`tests/h3/attn_f32_bm 2532 32 5 --compare`, the byte-exact arm against the
tensor arm on the same inputs in one process:

| Shape | max absolute | rms | values differing |
|---|---:|---:|---:|
| 2,532 | 1.5842e-04 | 2.8383e-05 | 100% |
| 2,273 | 1.6882e-04 | 2.9628e-05 | 100% |

The class is the DiT tensor arm's: Q, K and V round to BF16 at staging, P rounds
to BF16 for the second `mma`, and the scores reduce through `mma` rather than
the reference's lane partial and shuffle. That is the same order of deviation
the recorded BF16 kernel carries (1.2207e-04 max, 1.1489e-05 rms at 18,432).

End to end on the yardstick job, two runs of one binary with
`H3_VAE_ATTENTION` the only difference:

| Phase | Default | Tensor arm | Delta |
|---|---:|---:|---:|
| TEXT | 28.288 s | 28.299 s | +0.011 s |
| LOAD | 23.640 s | 12.447 s | page cache, not code |
| DENOISE | 39.211 s | 40.139 s | +0.928 s, 2.4% |
| AUDIO | 2.583 s | 2.600 s | +0.017 s |
| FRAMES | 196.732 s | 114.856 s | -81.876 s, -41.6% |
| VIDEO | 0.600 s | 0.631 s | +0.031 s |
| Render sha256 | `5cb28765...` | `e213b641...` | differs, as the arm intends |

FRAMES is the phase's own measurement and needs no control; the wall does, and
these two runs cannot supply it because their LOAD rows differ by 11.193 s from
the page cache. The two runs are one each: this project's recorded spread is
0.609% on the denoise and 0.136% on FRAMES between two runs of one binary, so
the -41.6% is far outside it and the +0.928 s on DENOISE is not.

Quality, FFmpeg's own metrics between the two renders, which differ only by
this arm: PSNR y 50.28 dB, u 54.21, v 54.39, average 51.25, min 49.26; SSIM Y
0.993174, U 0.996581, V 0.996790, all 0.994345. Against the same binary's own
default, this arm is two orders of magnitude closer to it than the DiT tensor
default is to the scalar path (36.97 dB / 0.9779 on rain at 20 steps), which is
what a deviation confined to the last stage of the pipeline should look like.

At the production step count, 512x512 56 frames 20 steps seed 42 `--reuse 1`,
two runs of one binary again:

| Phase | Default | Tensor arm | Delta |
|---|---:|---:|---:|
| TEXT | 27.906 s | 27.991 s | +0.085 s |
| LOAD | 7.350 s | 7.406 s | +0.056 s |
| DENOISE, 20 evals | 63.286 s | 64.708 s | +1.422 s |
| AUDIO | 1.415 s | 1.425 s | +0.010 s |
| FRAMES | 37.010 s | 22.270 s | -14.740 s, -39.8% |
| VIDEO | 0.325 s | 0.319 s | -0.006 s |
| Total, TEXT line to exit | 137.292 s | 124.119 s | -13.173 s, -9.6% |
| Render sha256 | `ed8a1f03...` | `21a94ab3...` | differs, as the arm intends |

Here the LOAD rows agree to 0.056 s, so the wall comparison is usable: -9.6% on
a 20-step job, where the denoise is a much larger share. Quality between the two
renders: PSNR y 46.93, u 51.82, v 51.10, average 47.98, min 47.11; SSIM Y
0.990578, U 0.994784, V 0.994421, all 0.991919.

The DENOISE rows are +0.928 s at 768 and +1.422 s at 512, both positive and
both a phase this option has no call path into. Two runs cannot separate that
from run order, and in both pairs the arm ran second, so the five-run sequence
in the next section brackets it with a repeated default run.

## The matmul compute mode

`tests/h3/vae_gemm_bm <rows> <calls>` times one decoder block's four matmuls
(2048->6144, 2048->2048, 2048->16384, 8192->2048) at the tile shape, so a
per-block figure times 1,944 prices the block. On the Spark, three calls a run,
two runs:

| Rows | `fp32` | `tf32` | `bf16` |
|---|---:|---:|---:|
| 2,532 (768 job tile) | 88.410 / 88.006 ms | 12.693 / 12.635 ms | 11.335 / 11.232 ms |
| 2,273 (512 job tile) | 97.985 ms | 11.103 ms | 9.954 ms |

The two runs agree to 0.5% on every arm, so the first arm's cold cache is not
what the figures are made of. The 2,273-row fp32 figure being 11% above the
2,532-row one is unexplained: 2,273 is odd and the same three calls were made,
so a cuBLAS algorithm selection is the candidate, not the port.

Deviation, `--compare` at the same shapes, against the `fp32` arm on the same
inputs and in the same process:

| Rows | Pair | max absolute | rms | rel_rms | differing |
|---|---|---:|---:|---:|---:|
| 2,532 | tf32 vs fp32 | 0.0445 | 4.3720e-03 | 2.75e-05 | 99.99% |
| 2,532 | bf16 vs fp32 | 0.0445 | 4.3720e-03 | 2.75e-05 | 99.99% |
| 2,532 | **bf16 vs tf32** | **0** | **0** | **0** | **0.00%** |
| 2,273 | bf16 vs tf32 | 0 | 0 | 0 | 0.00% |

`CUBLAS_COMPUTE_32F_FAST_TF32` and `CUBLAS_COMPUTE_32F_FAST_16BF` are one
computation on this toolchain, bit for bit, on both hosts. That is not the
port's plumbing collapsing them: a raw cuBLASLt probe with the same row-major
TRANSB layouts, no port code in the path, returns status 0 for all three
compute types and differences of exactly 0 between the two FAST ones
(`CUBLAS_COMPUTE_32F` = 68, `_FAST_TF32` = 77, `_FAST_16BF` = 75 in
`cublas_api.h`). So the two reduced modes cannot be A/B'd against each other
and are one arm; only (`fp32`, `FAST`) is a real comparison. Their deviation
from `fp32` is 2.75e-05 relative rms, four orders below the value scale.

End to end, the five-run sequence on the Spark, one binary, fixed order:

| Run | Arm | DENOISE | FRAMES | Render sha256 (first 24) |
|---|---|---:|---:|---|
| 1 | default | 39.896 s | 196.873 s | `5cb28765208c49d73d1b1789` |
| 2 | `H3_VAE_ATTENTION=tensor` | 40.768 s | 114.875 s | `e213b6410f64dd04ecf5b6a1` |
| 3 | default | 41.110 s | 197.856 s | `5cb28765208c49d73d1b1789` |
| 4 | `tensor` + `H3_VAE_GEMM=tf32` | 40.909 s | 36.595 s | `337365713afee62f1d4b337b` |
| 5 | `tensor` + `H3_VAE_GEMM=bf16` | 40.506 s | 34.494 s | `337365713afee62f1d4b337b` |

Three things this settles:

- The arm does not touch the denoise. Run 3 is the same configuration as run 1
  and is 1.214 s slower, which is larger than run 2's +0.872 s over run 1, so
  the DENOISE rows move with the session and not with the option. The earlier
  two-run pairs' +0.928 s and +1.422 s are the same effect.
- FRAMES is reproducible: 196.873 against 197.856 for the same configuration,
  0.5%, and the renders are byte-identical across the repeats of both arms.
- Runs 4 and 5 are one computation (see above) and their 2.1 s spread is
  run-to-run noise at this scale, so the honest figure for the reduced mode is
  about 35 s against 196.9 s, and the 6% between tf32 and bf16 must not be read
  as an arm difference.

Quality against the default render: run 2 measures PSNR y 50.28 dB / SSIM all
0.994345 (above), runs 4 and 5 measure PSNR y 50.16, u 54.03, v 54.22, average
51.13, min 49.22, SSIM Y 0.992976, U 0.996434, V 0.996659, all 0.994166; run 3
is identical to run 1 (`inf` PSNR, SSIM 1.000000), which is the control that
says the comparison itself is sound.

### The gap this unit could not close

The harness prices one block's four matmuls at 88.006 ms at the 768 tile, which
over 1,944 block calls is 171.1 s of matmul alone. The job's FRAMES is 196.873 s
in the default arm and 114.875 s with the tensor attention and the same `fp32`
matmuls, so 1,944 block calls cannot contain 171.1 s of matmul: the harness and
the job disagree by roughly a factor of two, in the direction that makes the
harness the outlier. Either the harness prices a block differently from the way
the decoder runs one, or the decoder runs fewer block calls than the tile plan
implies - the plan itself is checkable and says 54 submissions of 36 blocks, and
the reference's own profile line records 54 video VAE submissions. Nothing in
this unit settles which, and no profile has been taken since the tensor default
landed. One nsys capture over the FRAMES range, per kernel, is the measurement
that decides it, and until it exists the job-level FRAMES rows above are the
only admissible numbers for this block; the per-block table is a kernel-class
measurement, not an attribution.


`H3_VAE_SDPA_ORDER=rows` launches the byte-exact F32 kernel as
`grid(query blocks, heads)` instead of the reference's `grid(heads, query
blocks)`, so the CTAs resident together stay on one head and read that head's
staged keys and values from L2. It changes no arithmetic: the arm prints the
gate hash `74c6a4b2fa3cba6c` at 2,532 and `--compare` reports max_abs 0, rms 0,
0% differing.

At 2,532 rows: heads 41.44 / 41.54 / 41.69 ms against rows 41.22 / 41.24 /
41.32 ms, 0.7% with the ranges disjoint. Small, free, and it costs no memory or
arithmetic, so it is a candidate to adopt on the reference path regardless of
which attention kernel runs. It is left as an option in this unit because the
campaign's adoption rule asks for the measurement before the adoption, and one
kernel-level measurement at one shape is thin evidence for changing a released
order.

## Refuted: folding the bias into the matmul epilogue

The advice for this block included replacing the matmul-plus-bias-kernel pair
with one `cublasLtMatmul` carrying `CUBLASLT_EPILOGUE_BIAS`, which would remove
7,830 launches and a read-modify-write of every activation. It was implemented,
and `cublasLtMatmul` refuses it here: the descriptor attributes all accept, and
the matmul returns `CUBLAS_STATUS_NOT_SUPPORTED` (15), while the same matmul
without the epilogue returns 0. Reproduced on both hosts - GB10 sm_121 with
CUDA 13.0 and the workstation's sm_89 with CUDA 13.3 - with the bias pointer,
the epilogue and the three row-major layouts the decoder uses. The option was
removed rather than shipped as a switch that cannot be taken; the separate bias
kernel stays, in every mode.

## What this unit does not establish

- The noise floor is still missing. Every comparison here is against this
  binary's own default, and no benign configuration change on this host has
  produced a non-zero deviation to bound the arms with. That gap is the DiT
  tensor default's too, and this unit does not close it.
- The tensor arm's output is not arch-portable: the scalar arm prints
  `74c6a4b2fa3cba6c` at 2,532 on both an RTX 4070 SUPER and the GB10, while the
  tensor arm prints `07e00791e357b262` and `a96a18d539d41084` respectively. The
  deviation is the same size on both, so only the last-bit rounding differs, but
  a hash gate for this arm cannot be written across architectures.
- No profile has been taken since the tensor default landed. The 80.3 s scalar
  figure prices the attention from the kernel's own measured cost at the real
  shape; the matmul and elementwise shares of FRAMES, and the effect of
  `H3_VAE_GEMM`, still rest on the 2026-10-27 nsys run, which predates both the
  tensor default and the k=8/r=2 restructure. One nsys capture is the next
  measurement.
- One run per arm at the job level. The `--compare` control and the three-run
  kernel sweeps carry the repeatability here.
- The FRAMES delta is the arm's effect on this job. It does not price the arm's
  effect on a 20-step job, where FRAMES is a smaller share of the wall, and it
  does not price the GEMM option.

## Open

- Whether to make the tensor VAE attention the default, and whether to adopt
  the query-major CTA order. Both are operator decisions; this unit leaves them
  as options.
- The GEMM arms (`tf32`, `bf16`) are correctness-gated
  (`tests/h3/native_test`: the default path is bit-identical to `h3d_linear`,
  both reduced modes stay finite and inside a structural bound) and were not
  benchmarked in this unit.
- The `H3_VAE_ATTENTION` and `H3_VAE_GEMM` options reach the CLI and the HTTP
  API only through the environment today. A CLI flag and a create-body field,
  as the DiT attention has, are the next step if these arms are kept.
