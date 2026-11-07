# The draft leg's E4M3 attention: 3.6 s, no clip-gate flag, and audio that moves (2026-11-02)

Unit: handoff section 3. The E4M3 attention block was measured and withdrawn
once already: it is **15.9%** faster on the attention class, deterministic to the
hash, and the clip gate refused it when it ran the whole delivery - a fifth of
the detail energy for 10.6% of the denoise
([`fp8-attention-arm-2026-11-02.md`](fp8-attention-arm-2026-11-02.md) section 3).
The question this unit asked was whether the **draft leg** is a place the arm
can still pay, since the delivered picture sees that leg only through the
upscaler and one refine step at sigma 0.72 (`x = 0.72 * noise + 0.28 *
upscaled`). The arm's own record ended by naming this leg as the place it could
earn its keep - "what the route has and the arm does not is a refine step that
puts some of it back" - so this is the measurement it asked for.

The answer is yes, and it is the first FP8 arm in this port to pass the clip
gate: the draft's `50.483 s` becomes **46.83-46.93 s** (-7.1%), `DENOISE`
`89.335 s` becomes **85.66-85.80 s** (-4.1%), and the delivered clip is
`PASS ... no flag` with `musiq_d -2.12` against a -3 budget, detail energy
**-3.8%**, and `comp_ssim` **0.897**. What the gate does not see is the audio:
the draft is the frozen stage, and the arm moves it - RMS x**1.195** (+1.55 dB),
SNR 6.0 dB, correlation 0.915. The arm ships **opt-in**
(`--draft-attention e4m3`) and not as a default. Spark `gx10-e1d2`, sm_121a,
CUDA 13.0, 61-64 C, no clock touched.

## 1. An environment word is a process word; a stage needs a context word

`H3_DIT_ATTENTION` is read when the DiT's context is created
(`native/h3/h3_dit_gpu.cu:120`, `:173`), and the two-stage route creates two
contexts - the draft at 928x544 and the refine at 1344x768
(`engine.rs` `load_stage`, one per prepared graph). So the environment can turn
the arm on for the whole process and nothing else: there was no way to say "the
draft leg only", which is exactly the shape the unit needed.

The tree now says it at the layer that owns the word. `h3d_set_attention_type`
(`native/h3/h3_dit_gpu.h:104`, `:188` in the `.cu`) states the data type on one
context after `h3d_set_attention` has read the environment; `Gpu::
set_attention_type` (`crates/ds4-sys/src/h3_dit.rs:1122`) exposes it;
`LoadOptions::attention_type` (`dit.rs:877`) carries it into the load, where a
`Scalar` + `E4M3` request is refused rather than half-honoured (the scalar kernel
is the byte-exact one and has no quantized path); and `RenderRequest::
draft_attention` (`engine.rs:118`) is the only place a caller can name it. The
refine's key states `attention_type: None` explicitly (`engine.rs:557`) - it is
the delivery, so it keeps the environment's word whatever the draft says - and
the field is part of `PreparedKey` (`engine.rs:222`) so two arms are two graphs.

Three proofs that the confinement is real, not intended:

- the arm's own per-shape line (`h3_dit_gpu.cu:4235`) printed **76** (the
  connector) and **18,731** (the draft's DiT sequence) in the arm's render, and
  the refine's **37,786** never printed - from the same process, minutes later,
  at the sequence the delivery always runs;
- the refine's seconds did not move: **38.499, 38.829, 38.876, 38.853** across
  the control, both arm runs and the warm tail (section 2);
- the delivered audio hash is the draft's, carried through: the render reports
  `audio_frozen=true` with `draft_audio=fd6c3aaa59876a2b` equal to
  `audio=fd6c3aaa59876a2b`, while the control's is `0e170b4d53054721`.

The environment's own word is untouched. `H3_DIT_ATTENTION` unset still produces
the preset's clip bit-for-bit (`md5 e2126e620ed1bd5d3d524388be4c4c6d`, section
2), and `H3_DIT_ATTENTION=fp8` still selects the arm for every context in the
process, which is what the arm's record measured.

Two spellings of the new flag, both deliberate: `--draft-attention e4m3` is the
arm on the draft leg with the refine on BF16, and `--draft-attention bf16` pins
the draft while `H3_DIT_ATTENTION=fp8` carries the rest - a run that wants the
arm on the refine and not on the draft. `fp8` is accepted as the arm's other
name. Without `--draft` the flag is refused (`h3-rs.rs:582`, `engine.rs:332`):
on a single pass the draft leg is the delivery, which is the clip the gate
already refused.

## 2. Where the 3.6 s comes from, and what the wall can and cannot say

Four renders, one binary, one session, the delivery job's own line
(`outputs/two-stage/one.jobs`), `1344x768x121`, `steps=4`, `--draft 0.7
--refine-sigma 0.72`, seed 42, `H3_DIT_ATTENTION` unset throughout:

| run | `LOAD` | `DENOISE` | draft | upscale | refine | `VIDEO` | wall | clip md5 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `ref-head` (cold) | 56.525 | 88.401 | 49.903 | 3.979 | 38.499 | 52.143 | 235.539 | `e2126e62...` |
| **arm**, `--draft-attention e4m3` | 19.927 | **85.655** | **46.826** | 3.745 | 38.829 | 46.728 | 191.334 | `7eeac032...` |
| **arm**, repeat | 14.464 | **85.802** | **46.926** | 3.756 | 38.876 | 46.866 | 183.855 | `7eeac032...` |
| `ref-tail` (warm) | 13.660 | 89.335 | 50.483 | 3.753 | 38.853 | 47.011 | 187.297 | `e2126e62...` |

Warm against warm (the tail is the reference; the arm's two runs are both warm):

| arm | draft | step | `DENOISE` | wall |
|---|---:|---:|---:|---:|
| arm #1 | -3.657 (-7.2%) | -0.914 | -3.680 (-4.1%) | +4.037 |
| arm #2 | -3.557 (-7.0%) | -0.889 | -3.533 (-4.0%) | -3.442 |

**The wall does not resolve it.** The four walls span 52 s and the two warm arms
straddle the even warmer tail by +/-4 s, so 3.6 s of `DENOISE` is inside this
box's own spread today - the same lesson the scale ladder recorded (its arm
pairs moved 25 s of wall on identical stage sums). What the numbers do say is
the two stages: 0.9 s a step off the draft, every step, twice, and a refine that
never moved. The route's wall is still ~187 s and this arm is 1.9% of it.

**The class, priced at the draft's own sequence.** The render's arm line named
it - **18,731 rows** - so the bench can be asked the same shape
(`tests/h3/attn_bm 18731 56 3`, three calls each, one binary):

| shape | BF16 block | E4M3 arm | delta | `out_fnv` |
|---|---:|---:|---:|---|
| 18,731 rows | 126.291 ms | **109.147 ms** | **-13.6%** | `7b1f1d6cb1c5ed6a` |
| 18,731 rows, second process | - | **109.114 ms** | - | `7b1f1d6cb1c5ed6a` |

Fifty calls a step - one per block - put **6.315 s** of the draft's 12.62 s step
in the attention class, **50.0%**, and the arm's 17.1 ms a call is **3.43 s**
over the four steps against the render's measured **3.61 s** (-5%, which is the
connector's 76-row calls and the bench's own spread). The class percentage is
the same 15.9%-ish the delivery measured; what changes with the canvas is the
denominator, and it is the sequence's square, not the class.

## 3. The picture: the refine washes most of it out

Same canvas, same seed, same prompt, decoded yuv420p over all 124 frames, the
arm against the warm reference (`~/ds4-test/route-compare.sh`):

| instrument | `ref-tail` | arm | delta |
|---|---:|---:|---:|
| PSNR y / u / v | - | 24.814 / 37.316 / 39.237 dB | average 26.476 (min 23.634, max 29.291) |
| SSIM Y / All | - | 0.8851 / 0.9148 | |
| detail sigma 4 | 2.3266 | **2.2373** | **-3.8%** |
| detail sigma 8 | 4.1298 | **3.9252** | **-5.0%** |
| `comp_ssim` | - | **0.897** | same shot |
| bytes | 4,351,509 | 4,269,101 | -82,408 (-1.9%) |

And the clip gate, `tools/h3-clip-score.py --baseline ref-tail.mp4` (**exit 0**,
both arm runs `PASS ... no flag`):

| metric | `ref-tail` | arm | delta / ratio | band | |
|---|---:|---:|---:|---|---|
| `musiq_d` | 51.80 | 49.68 | **-2.12** | >= -3 | pass |
| `clipiqa_d` | 0.366 | 0.358 | -0.008 | >= -0.08 | pass |
| `dover_d` | 0.642 | 0.664 | +0.022 | >= -0.08 | pass (never alone) |
| `sharp_frame_r` | 22.9 | 22.2 | x0.969 | 0.8-1.25 | pass |
| `noise_r` | 0.41 | 0.41 | x0.99 | <= 1.3 | pass |
| `flicker` | 1.112 | 0.970 | x0.872 | <= 1.25 | pass |
| `tflick_r` | 0.436 | 0.471 | x1.08 | <= 1.25 | pass |
| `warp_err_r` | 1.7 | 1.6 | x0.946 | <= 1.25 | pass |
| `flow_mag_r` | 11.12 | 11.20 | x1.01 | 0.67-1.5 | pass |
| `accel_r` | 0.191 | 0.207 | x1.08 | <= 1.3 | pass |
| `zoom` | 0.749 | 0.773 | - | - | - |
| `comp_ssim` | - | 0.897 | - | reported | same picture |

Read against the arm's own refusal on the delivery: there `sharp_frame_r` was
**0.607**, `musiq_d` **-7.12**, detail **-23.5%** at `comp_ssim` 0.543. Here the
same kernel on the same shot costs **-3.8%** of the detail energy and two MUSIQ
points, and `comp_ssim` 0.897 says the frame is the same frame and not a re-roll
of the composition (the scale ladder's 0.6 rung read 0.327; the route against
the one-pass clip read 0.147). What is left of the arm's damage is the part the
refine's `0.72` re-noise and its one BF16 step could not put back.

**Determinism, at this shape.** The arm's two renders are the same clip - `md5
7eeac032dc74bb4ced2c8063e306791b` both, PSNR `inf`, and the same fingerprints
(`video_fnv bd1b68b65a066189`, `audio_fnv fd6c3aaa59876a2b`, `frames_fnv
d970c25325ec9a79`) - and the bench's `out_fnv` at 18,731 rows agrees across two
processes. The control is the other half: `ref-head` and `ref-tail` are
bit-identical to each other and to the ladder's `scale07b.mp4` (`e2126e62...`,
PSNR `inf`, detail 2.3266 / 4.1298 both), so the arm off is the tree it was.

## 4. The audio the gate does not score

The draft is the frozen stage: the refine receives the draft's audio and returns
it bit-for-bit (`audio_frozen=true`), so whatever the arm does to the audio rows
during the draft is delivered untouched. The clip gate is video-only, so the
measurement is `~/ds4-test/audio-compare.sh` against the same tail:

| | value |
|---|---:|
| RMS reference -> candidate | 0.002724 -> 0.003256 |
| RMS ratio | **1.1953** (**+1.55 dB**) |
| SNR | 6.01 dB |
| correlation | 0.9148 |

For scale: the scale ladder's 0.6 draft leg moved the same audio by x1.449
(+3.2 dB) and 0.5 by x1.666 (+4.4 dB), so the arm's 1.55 dB is smaller than a
canvas change and larger than a rounding difference - it is a different
realization of the soundtrack, not a level shift, which is what 6 dB of SNR and
0.915 correlation describe. This is the number that keeps the arm opt-in: a
1.9% wall cut is not worth silently re-rolling the audio of every shot.

## 5. The engine gate, at its own shape

`tests/h3_engine_gate`, `512x512x22`, `steps=4`, `--draft` (the route's own
0.7 / 0.72), two renders per process:

| arm | draft | upscale | refine | media | |
|---|---:|---:|---:|---|---|
| control | 1.368 | 0.766 | 0.685 | `2c224536f5f00113 / 11a34732bcb63274 / 16a19b26f173619d` | `MEDIA identical=true`, `RESULT PASS two-stage text_hit audio_frozen` |
| `--draft-attention e4m3` | 1.368 | 0.421 | 0.681 | `838b53160f66e9a4 / 56a8353c5caf6c16 / ffcf408ff6dfea00` | `MEDIA identical=true`, `RESULT PASS two-stage text_hit audio_frozen` |
| `--draft-attention e4m3`, no `--draft` | - | - | - | - | `FAIL render 1: the draft attention arm needs a two-stage route (--draft)`, exit 1 |

The control reproduces the ladder's own gate hashes exactly, so the new field
changes nothing it does not name. The arm is deterministic at the gate's shape
too (`identical=true` over two renders of one resident engine), its two shape
lines are `sequence=15` (the connector) and `sequence=936` (the draft's DiT), and
the refine's ~3.8k-row sequence never printed. At 936 rows the arm's second is
inside the step's noise (1.368 both), which is the small end of the law the
section above prices.

## 6. What ships, and what it is worth

Ships, all opt-in:

- `--draft-attention bf16|e4m3` on `h3-rs` (`fp8` accepted for `e4m3`), refused
  without `--draft`;
- the same word on `tests/h3_engine_gate`, so the gate can run one leg;
- `LoadOptions::attention_type`, `RenderRequest::draft_attention` and the
  per-context `h3d_set_attention_type`, which is the general form of the arm for
  any two-stage render;
- one engine test (`the_attention_type_is_part_of_the_prepared_key`) beside the
  kernel-choice test it mirrors.

Not a default, for three reasons: the wall does not resolve the 3.6 s; the arm
moves the delivered audio by 1.55 dB where the gate cannot see it; and the
operator's rule keeps every FP8 path opt-in. Nothing about a run that does not
pass the flag changes - `md5 e2126e62...`, `video_fnv f24ba94a9e9506d6`, and
the engine gate's three hashes all reproduce.

Where it is worth the most is the largest draft: the saving is the sequence's
square, so this 3.6 s is what the 0.7 leg pays and the smaller rungs pay less
(the scale ladder's 0.6 draft carries 71% of the 0.7 leg's latent tokens, 0.5
carries 51%, before the canvas-fixed text and audio rows are subtracted from
both). It is the right arm to reach for when a render's draft leg is long and
the soundtrack is not the deliverable.

## 7. Reproducing

```
make -j8 CUDA_ARCH=sm_121 h3-rs tests/h3_engine_gate tests/h3/attn_bm
bash ~/ds4-test/route-draft-attn.sh        # the four renders
bash ~/ds4-test/attn-evidence.sh           # audio, ffprobe, three pairs, the gate
bash ~/ds4-test/attn-bm-draft.sh           # 18,731 rows, BF16 once, arm twice
bash ~/ds4-test/gate-draft-attn.sh         # the three engine-gate arms
cargo test -p ds4-core --lib minimax_h3::engine::tests
```

The render line is the campaign's (`H3_DIT_GEMM=fp8 H3_VAE_GEMM=bf16i
H3_VAE_ATTENTION=tensor`, `--crf 15 --preset slow`, `H3_DIT_ATTENTION` unset):

```
./target/release/h3-rs -d /mnt/h3-artifacts/MiniMax-H3-Turbo-v4 \
    --jobs outputs/draft-attn/attn-draft.jobs --draft 0.7 --refine-sigma 0.72 \
    --draft-attention e4m3 --crf 15 --preset slow
```

Artifacts: `outputs/draft-attn/` (four clips, their logs and job lines),
`~/ds4-test/{route-draft-attn,attn-evidence,attn-bm-draft,gate-draft-attn}.{sh,log}`,
`~/ds4-test/score-draft-attn.json`.
