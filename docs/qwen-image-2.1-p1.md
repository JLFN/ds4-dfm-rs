# Qwen-Image-2.1 port — phase P1, the CPU oracle (stages 1 and 2)

[Documentation index](README.md) | [Repository README](../README.md) |
[Plan](qwen-image-2.1-plan.md) | [Recipe](qwen-image-2.1-recipe.md) |
[Roadmap](qwen-image-2.1-roadmap.md) | [Phase S](qwen-image-2.1-phase-s.md) |
[P0](qwen-image-2.1-decision.md) | [P1 recipe](qwen-image-2.1-recipe.md)

Status: in progress, stages 1 and 2 of 3 done. Stage 1 is the part of P1 the plan
calls exact by construction — the initial noise, the positions, the RoPE table
inputs, the segment masks and the flow schedule. Stage 2 is the DiT evaluation
itself, driven by the reference's own conditioning and gated on the reference's
own dumped velocity. The VAE decode and the image are stage 3 and are not in this
document.

Date: 2026-10-01. Host: RTX 4070 SUPER (sm_89, 12282 MiB nominal), driver
610.43.03. Reference: `/data/imagegen/bin/sd-cli`, version
`master-890-74988b2-4-g6dcb5bb`, commit `6dcb5bb` (the [recipe](qwen-image-2.1-recipe.md)
section 1 pin).

## 1. The harness: the reference dumps its own intermediates

A staged port needs the reference's numbers, not a second opinion about them.
The deployed `sd-cli` carries dump hooks that are not in the pinned upstream
source (upstream at `6dcb5bb` contains no `SD_DUMP` string). They were found in
the binary and their behaviour was established by running them:

| variable | writes | content |
| --- | --- | --- |
| `SD_DUMP_IDS` | nothing (logs) | the token ids of each conditioning pass |
| `SD_DUMP_COND=<prefix>` | `<prefix>.<n>.bin` | the conditioning tensor; 0 is the prompt, 1 the negative |
| `SD_DUMP_STEPS=<prefix>` | `<prefix>.step<n>.in.bin`, `<prefix>.step<n>.pred.bin` | the latent entering each step and the velocity the sampler applies |
| `SD_DUMP_LLM=<prefix>` | `<prefix>.layer<n>.bin` | the text encoder's layers (P7's, unused here) |

File format, read off the dumps: an ASCII header `dims <rank> <d0> <d1> ...` then
one `\n`, then raw little-endian F32 in GGML axis order (ne0 fastest). The
tensors this document uses are `step1.in.bin` (16384 values, `dims 4 16 16 64 1`)
and `step1.pred.bin` (the same shape), at 256x256.

The fixture run, the same placement the phase-S report used:

```
cd /data/imagegen
SD_DUMP_IDS=1 SD_DUMP_COND=$D/run1.cond SD_DUMP_STEPS=$D/run1 \
  bin/sd-cli --diffusion-model models/diffusion_models/qwen-image-2.1-Q6_K.gguf \
             --vae models/vae/qwen_image_2.1_vae_bf16.safetensors \
             --llm models/text_encoders/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
             --backend te=cpu,diffusion=cuda0,vae=cpu \
             --params-backend te=cpu,diffusion=cuda0,vae=cpu \
             -p "a red cube on a white table" -H 256 -W 256 --steps 2 \
             --cfg-scale 6.0 -s 42 --rng cuda -o $D/run1.png
```

The dumps live in `misc/scratch/p1/refdump/` (gitignored, like every other
scratch harness). They are singletons of this host's reference, not artifacts the
project ships.

## 2. What the reference's own numbers settle

Read out of the run log and the dumps, in the reference's own words:

- `running in FLOW mode` (`diffusion_engine.cpp:1428`): the denoiser is
  `DiscreteFlowDenoiser` with shift 3.0, so `c_skip = 1`, `c_out = -sigma`,
  `c_in = 1` and `sigma_to_t` is `sigma * 1000`.
- `Flux scheduler: image_seq_len=256, steps=2, mu=0.500` (`denoiser.hpp:767`).
- The text lengths are not the token counts: the conditioning dumps are
  `[4096, 15]` (positive prompt) and `[4096, 9]` (negative) while the token ids
  number 29 and 23. The template prefix and suffix are stripped before the DiT
  sees the context; the layout is built from the resulting width, so those are
  the two prefix lengths, not 29 and 23.
- The latent grid is `16x16` for a 256x256 image, so the VAE's spatial
  compression is 16x, and the phase-S reading of `image_seq_len=4096` at 1024
  square (64x64) follows from it rather than being assumed.
- The first sigma is exactly 1 and `c_in` is 1, so `step1.in.bin` is the raw
  initial noise and the first step's update is `x1 = x0 + pred0 * (sigma1 - 1)` —
  in that order's arithmetic only, as section 3 shows: the reference computes
  it in a different order, and that difference is measurable.
  Least squares over the dumped `pred0` recovers `(sigma1 - sigma0)/sigma0` as
  `-0.377540648`, which is exactly the F32 step the port's own schedule
  computes, `f32(0.622459352) - 1`. The double-precision ideal of the same
  expression is `-0.377540669`; the two differ by the F32 spacing, which is why
  the agreement is claimed at the width the comparison is made in and not as a
  digit count.

## 3. Stage 1 results

| stage | quantity | tolerance | measured |
| --- | --- | --- | --- |
| noise | Philox randn, seed 42, 16384 values | byte-identical | **16384/16384 byte-identical** |
| positions | text prefix 15, image grid 16x16 | exact | prefix 15, temporal id 15 at the first image token, spatial ids -8..7 |
| masks | text segment causal, image unmasked | exact | lower-triangular, length 15 |
| schedule | mu, sigmas, step timesteps | exact F32 | mu 0.500, sigma [1, 0.622459352, 0]; t [1000, 622.459352] |
| Euler step | `x1` from `x0, pred0` | byte-identical | **16384/16384 byte-identical** |

The RoPE table's pairing convention (`rope_interleaved`) is the one item the
recipe lists as unsettleable by reading. Both pairings are implemented behind
`RopePairing`; section 4 settles it by measurement.

## 4. Stage 2: the DiT forward

One complete F32 DiT evaluation, driven by the reference's own conditioning dumps
and compared with the reference's own velocity. The gate is the dumped
`step1.pred.bin` from a `--cfg-scale 6.0` run, which is
`uncond + 6*(cond - uncond)` over the two guidance passes: `DiscreteFlowDenoiser`
has `c_skip = 1` and `c_out = -sigma`, so the Euler velocity the sampler applies
is the model output itself (`denoised = x - sigma*model_out`,
`d = (x - denoised)/sigma`).

Measured on this host (release build, 271 joint tokens, 15 + 256, both passes
and the combination; the wall time is one run on this host and moves with load):

| pairing | correlation | relative RMS | max abs | wall time |
| --- | --- | --- | --- | --- |
| `RopePairing::Interleaved` | **0.999973** | **7.56e-3** | 1.37 | 106 s |
| `RopePairing::HalfSplit` | 0.742873 | 6.69e-1 | 30.2 | 108 s |

The pairing is therefore settled by measurement and not by reading: the port
applies the interleaved pairing, the reference's own default, and the other
pairing is 0.257 away in correlation. The gate asserts the winner's identity,
its correlation and relative RMS, and a clear separation from the loser.

### Why the residual is a tolerance and not a bug

Two independent F32 implementations of this forward — this port and a from-source
numpy port (`misc/scratch/p1/dit_numpy.py`, gitignored) — agree with each other
to **1.2e-6 relative RMS** on the conditioned pass and 4.1e-7 on the
unconditioned one (correlation 1.0000000, max |diff| 2.4e-5, both measured
through the fixture test's `DS4_QWEN_IMAGE_DIT_OUT` dumps). On the same run both
sit 5.2e-3 relative RMS from the reference's conditioned pass and 7.6e-3 from
its CFG combination. F32 summation-order noise is three orders of magnitude
below the residual, so the difference is not how the sums are ordered: it is the
reference's own arithmetic. The reference evaluated this fixture on CUDA
(`diffusion=cuda0`) with Q6_K weights through its quantized matmul, and the
manual attention path it selected (`flash_attn: false` in the run log — the F16
casts of k and v live in `build_kqv`, which only runs under flash or sage
attention) still leaves the rest of the graph on kernels this port does not
model. Which kernel or numeric type accounts for the residual is not isolated
here, and not claimed.

### The comparison harness

`misc/scratch/p1/dit_numpy.py` is a second, independent implementation of the
same forward, written in numpy from the pinned source rather than ported from
this one; it is scratch and gitignored, not a shipped artifact. Comparing the
two stage by stage during development is what exposed three layout errors in
this port — the matmul's output order, patchify's channel order and the
unpatchify order — each of which left the magnitudes correct and the content
scrambled. The reproducible part of that comparison is section 4's pass-level
agreement, through the fixture test's own dumps.

### The dequantizers are checked against a third implementation

The two weight layouts the forward needs are Q6_K (`ql[128] | qh[64] |
scales[16] i8 | d f16`, 210 bytes per 256 values) and BF16. Both were compared
against the `gguf` Python package's dequantizer, whose Q6_K comes from llama.cpp
rather than from this port: `img_in.weight` (BF16), `attn.to_q.weight` (Q6_K),
`timestep_embedder.linear_2.weight` and `modulation.1.weight` all match with a
maximum difference of zero. The comparison is `DS4_QWEN_IMAGE_EXPORT=<tensor>` on
the fixture test, which writes one dequantized tensor for an external check.

### What stage 2 does not establish

1. Only step 1 of the fixture, at the fixture's 16x16 latent grid. Step 2 is a
   second evaluation of the same function at a different sigma and is not run.
2. Text-to-image only: a joint sequence with reference latents is refused by name
   (`DitError::Unsupported`), which is the img2img path and belongs with its own
   gate.
3. No VAE, no latent statistics, no PNG — stage 3.
4. No kernel, no GPU path, no timing worth quoting: the forward is a reference
   implementation, and its 106 s for two passes is the CPU's, not a serving
   number.

### The operation order of a step is measurable

The Euler step is *not* insensitive to how it is written. The reference denoises
first and takes the velocity from that — `denoised = x - sigma*pred`,
`d = (x - denoised)/sigma`, `x += d*(sigma_next - sigma)` — and this chain
rebuilds `step2.in.bin` from `step1.in.bin` and `step1.pred.bin` bit for bit,
16384 of 16384 values. The algebraically equal one-expression form
`x + pred*(sigma_next - sigma)` differs in 692 of those values, because
`x - (x - sigma*pred)` is not exactly `sigma*pred` in F32. The oracle therefore
exposes `euler_step` in the reference's order, and the parity test asserts both
that the reference's order is bit-exact and that the one-expression form is not,
so a later simplification cannot quietly cost the exactness.

### The other trap: Box-Muller is mixed precision

The reference's Box-Muller is a **mixed-precision** computation, and an
otherwise faithful single-precision port is not byte-identical:

```cpp
float u = x * two_pow32_inv + two_pow32_inv / 2;   // f32
float v = y * two_pow32_inv_2pi + ...;             // f32
float s = sqrt(-2.0f * log(u));                    // log/sqrt resolve to DOUBLE
float r1 = s * sin(v);                             // double product, one rounding
```

Measured against the reference's own noise dump, over 16384 samples:

| chain | byte-identical |
| --- | --- |
| all F32 (`logf`, `sqrtf`, `sinf`) | 11242 |
| double throughout, one final rounding | 12136 |
| double `log`/`sqrt`, F32 `s`, double product | **16384** |

The port mirrors the third form. This is exactly the class of quiet error the
oracle exists to prevent: the all-F32 version is wrong in 5142 of 16384 samples
(4774 by one ulp, 360 by two, 8 by three), a difference no downstream tolerance
would ever have flagged.

## 5. Code and tests

- `crates/ds4-core/src/qwen_image/oracle.rs`: `Philox`, `initial_noise`,
  `Layout`/`Segment`/`build_layout`/`LayoutError`, `rope_table`, `apply_rope` +
  `RopePairing`, `text_mask`, `flux_time_shift`/`flux_mu`/`flux_sigmas`/
  `flow_timestep`/`noise_scaling`/`euler_step`. 12 unit tests, model-free.
- `crates/ds4-core/src/qwen_image/dit.rs`: `DitWeights` (`open`, `linear`,
  `dequantize_f32`), `DitPass`, `DitError`, `forward`, `cfg_combine`, `Parity` /
  `parity`, with the Q6_K and BF16 dequantizers, the F32 matmul and the
  elementwise kernels the forward needs. 9 unit tests, model-free.
- `crates/ds4-core/tests/qwen_image_oracle.rs`: 9 tests. Six evaluate the
  reference's dumps and return early unless `DS4_QWEN_IMAGE_ORACLE=<prefix>` is
  set (the DiT gate also needs `DS4_QWEN_IMAGE_DIT=<gguf>`), one is the dequant
  cross-check hook gated on `DS4_QWEN_IMAGE_EXPORT=<tensor>`, and two are
  model-free and always run. Cargo runs the test binary from the package
  directory, so the fixture prefix must be an absolute path.

```
cargo test -p ds4-core --lib qwen_image::oracle                          # 12 passed
cargo test -p ds4-core --lib qwen_image::dit                             # 9 passed
D=/data/ds4-dfm-rs/misc/scratch/p1/refdump/run1
DS4_QWEN_IMAGE_ORACLE=$D cargo test -p ds4-core --test qwen_image_oracle # 9 passed
DS4_QWEN_IMAGE_ORACLE=$D \
DS4_QWEN_IMAGE_DIT=/data/imagegen/models/diffusion_models/qwen-image-2.1-Q6_K.gguf \
  cargo test -p ds4-core --release --test qwen_image_oracle dit_forward_reproduces
```

## 6. Artifacts

Re-verified this session (`misc/scratch/p1/p1-artifact-hashes.txt`):

| artifact | sha256 |
| --- | --- |
| `qwen-image-2.1-Q6_K.gguf` | `a3a0d39bb03cda26302fc048b49d019baaea1c381cbda3994f6f2a7826344fb9` |
| `qwen_image_2.1_vae_bf16.safetensors` | `bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9` |
| `misc/scratch/p0/vae-decode-bf16.gguf` | `d3feefed174e69d51f380c71bb500ab92033d85fc83f5d87c9811a8775725372` |

## 7. What this does not establish

1. Nothing about the VAE decode, the latent statistics, the conversions or the
   PNG, which are stage 3.
2. No kernel, no GPU path and no serving timing. P1 is a reference, not a
   measurement.
3. The conditioning intake consumes dumps; the text encoder that produces them
   is P7's and is not exercised here.
4. Stage 2's limits are listed in section 4.
