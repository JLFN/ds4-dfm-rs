# The route's 4-step draft: -25.5 s, a picture the gate passes, and the audio it changes (2026-11-02)

Unit: handoff section 3. The delivery checkpoint is a **4-step distillation**
and the two-stage route ran **six** draft steps plus one upscale plus one refine
- the reference's *base-model* schedule, not this checkpoint's own. Two fewer
draft steps are **-25.4 s** of the route, and the route has been the campaign's
biggest lever since it landed at 1.61x
([`two-stage-delivery-2026-11-02.md`](two-stage-delivery-2026-11-02.md)).

The unit also had to answer a question that is not arithmetic: the preset's
refine sigma is **0.72**, and `host.rs:581` pins why - it re-enters above the
**six**-step draft's terminal **0.7059**. At four evaluations the same `serving`
grid ends at **0.8**, so a 4-step job line with the preset's sigma refines
*below* where its own draft stopped. The plausible fix was the "consistent
ladder", a sigma above 0.8. It was measured, and it **loses**: the preset's 0.72
is the better pairing for a 4-step draft, and the reference does the same thing
for the same reason (section 4).

Spark `gx10-e1d2`, sm_121a, CUDA 13.0, 61-65 C, no clock touched. Every render
is the delivery job - the fox prompt, seed 42, **1344x768x121**, `--crf 15
--preset slow`, `H3_DIT_GEMM=fp8 H3_VAE_GEMM=bf16i H3_VAE_ATTENTION=tensor`.

## 1. Four renders, one binary, one session

`~/ds4-test/route-four.sh`. The six-step route is rendered twice - first cold and
last warm - so the 4-step arms sit between two rows of the same reference and the
wall comparison has a warm partner. `LOAD` is the page-cache share of the wall
and is not a property of the arm.

| run | steps | refine sigma | LOAD | DENOISE | draft | upscale | refine | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| one-pass (direct) | 6 | - | 7.188 | 232.699 | - | - | - | 347.611 |
| ref6, cold | 6 | 0.72 | 47.808 | 113.497 | 74.877 | 4.034 | 38.619 | 246.484 |
| ref6, warm | 6 | 0.72 | 13.256 | 114.820 | 75.851 | 3.676 | 38.969 | 212.462 |
| four-preset | 4 | 0.72 | 31.823 | 89.082 | 50.374 | 3.674 | 38.708 | 212.129 |
| four-ladder | 4 | 0.82 | 13.097 | 89.359 | 50.456 | 3.602 | 38.903 | 187.483 |

Warm against warm (ref6 warm -> four-ladder): **wall -24.979 s (-11.8%)**,
**DENOISE -25.461 s (-22.2%)**, **draft -25.395 s (-33.5%)**. Against the
one-pass clip the route is now **347.611 -> 187.483 s, 1.85x**. The pinned
216.068 s row of the route record is reproduced: the same job at the same six
steps is 212.5 s warm, and its clip is **bit-identical** to the pinned
`outputs/two-stage/draft.mp4` (`md5 d8fcf9f171952c260041f1c6dafc4d1c`, and
`video_fnv=15caddb0cdeebf23`, `audio_fnv=0b8436940a1faf88`,
`frames_fnv=d1ce47a36ce76921` all equal) across two processes and two binaries.

The 25 s is exactly the draft's step arithmetic: **12.5-12.8 s a step** at
928x544 (six: 12.551, 12.618, 12.667, 12.703, 12.733, 12.754; four: 12.536,
12.570, 12.608, 12.660), so two steps are 25.2-25.5 s, and nothing else in the
route moves. The refine's own cost is **sigma-independent** - 38.708 at 0.72,
38.903 at 0.82, 38.969 on the six-step reference - which is the check that the
sigma arm changed a noise level and not a schedule length.

## 2. The picture: both arms pass, and the preset is the better of them

Same canvas, same seed, same prompt, so every pair here is comparable. PSNR,
SSIM and detail energy are `~/ds4-test/route-compare.sh`; the clip gate is
`tools/h3-clip-score.py` against the six-step route clip as baseline.

| arm | PSNR y/u/v (avg) | SSIM Y | SSIM All | detail sigma=4 | detail sigma=8 | comp_ssim |
|---|---|---:|---:|---:|---:|---:|
| ref6 (reference) | - | - | - | 2.5913 | 4.5724 | - |
| four-preset | 20.225/32.324/33.985 (21.874) | 0.8011 | 0.8535 | 2.3266 (-10.2%) | 4.1298 (-9.7%) | 0.719 |
| four-ladder | 20.259/32.712/34.401 (21.917) | 0.8006 | 0.8536 | 2.3013 (-11.2%) | 4.0458 (-11.5%) | 0.717 |

The gate's verdict, both arms against `ref6.mp4` (**exit 0**):

| arm | musiq | clipiqa | dover | sharp | noise | flicker | tflick | warp | flow | accel | zoom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| four-preset | -1.59 | -0.012 | -0.001 | x0.927 | x0.947 | x1.05 | x0.942 | x0.986 | x1.20 | x0.979 | 0.749 |
| four-ladder | -2.89 | -0.003 | -0.017 | x0.992 | x0.945 | x1.14 | x1.08 | x0.962 | x1.22 | x1.16 | 0.742 |

Every budget holds (`musiq_d >= -3.0`, `sharp_frame_r` 0.8-1.25, `noise_r <=
1.3`, the flicker family and `warp_err_r <= 1.25`, `flow_mag_r` 0.67-1.5,
`accel_r <= 1.3`). This is the campaign's **first arm the gate passes** - the
E4M3 attention arm was refused at `sharp_frame_r` 0.607 and `musiq_d` -7.12
([`fp8-attention-arm-2026-11-02.md`](fp8-attention-arm-2026-11-02.md)) - so the
result needs its control: **the gate's own numbers say "same shot, slightly
softer", not "a different shot"**. `comp_ssim` is 0.719/0.717 against the route
reference, while the route itself against the one-pass clip is **0.147** (PSNR
average 14.2 dB, `sharp_frame_r` x0.5, `flow_mag_r` x1.33 - the route family is
flagged against the one-pass clip as a different take, which is what
[`two-stage-delivery-2026-11-02.md`](two-stage-delivery-2026-11-02.md) called "a
different shot"). A 10% detail loss on top of the route's own 44% is the whole
cost of the 25 s, and it is inside the gate's band on every metric but the ones
that pass it.

At engine level the same arm passes on the binary that rendered, twice:

| gate | shape | MEDIA | RESULT |
|---|---|---|---|
| `h3_engine_gate --draft` | 512x512x22, steps=4, draft 0.7, refine 0.72 | `identical=true` (`video=2c224536f5f00113`, `audio=11a34732bcb63274`) | PASS two-stage text_hit audio_frozen |
| `h3_engine_gate --draft --refine-sigma 0.82` | same, refine 0.82 | `identical=true` (`video=bb7daf73356fe845`, `audio=11a34732bcb63274`) | PASS two-stage text_hit audio_frozen |

The gate's default `steps` is already **4**, so `--draft` there is this unit's
shape without a flag; and the two rows carry the **same audio hash** while their
video hashes differ - the frozen-audio contract, seen from the engine rather than
from the clips (section 4).

## 3. What the 25 s does not buy: the refine sigma

The "consistent ladder" arm - refine at **0.82**, i.e. just above the 4-step
draft's own terminal 0.8, keeping the preset's 2% re-noise margin - is the arm
that *looks* right, and it is the worse picture:

- `musiq_d` **-2.89** against the preset arm's **-1.59**, `dover_d` -0.017
  against -0.001, `accel_r` **x1.16** against x0.979, `flicker_r` x1.14 against
  x1.05. `comp_ssim` is a hair lower (0.717 against 0.719). The only budgets it
  wins are the ones where it sits closer to 1.0 because it changed less overall -
  `sharp_frame_r` x0.992 and `tflick_r` x1.08 - and even those come with lower
  detail energy (2.3013 against 2.3266), i.e. the sharpness metric is reading
  the extra noise.
- The reference agrees, and for the same reason. Its own accelerated-draft
  presets take the opposite move: `p4_r1` runs a **4-NFE** draft and pairs it
  with `refine_sigmas="0.55, 0"` - *below* the draft's terminal, not above
  (`/data/Rapid-H3/harness/fast_test.py:58`; the PDD mechanism is documented at
  `harness/workflow.py:101`). `p6_r1` (6 NFE) and `p8_r1` (8 NFE) carry the same
  0.55. So the reference tunes the refine's sigma **per draft schedule** and does
  not derive it from the draft's terminal; entering below is a shape it ships.

The lesson recorded here, because the handoff argued the other way: the refine's
sigma is a **preset number that belongs to a draft**, not an invariant of the
sigma grid. `host.rs:581`'s inequality is true of the preset pair it names, and
it is not a rule for a shorter draft. Measuring the "obviously consistent" arm
was the only way to know which of the two shapes the picture prefers, and it
answered against the intuition.

## 4. The audio, which the video gate cannot see

The route freezes the draft's audio (`freeze_audio`,
`harness/workflow.py:50`), so a delivered clip's audio is **entirely a product of
the draft leg**, and the clip gate is video-only. `~/ds4-test/audio-compare.sh`
decodes each clip to 48 kHz mono and reports RMS and SNR/correlation, with the
caveat that the audio is a stochastic joint denoise and a correlation is
therefore a weak instrument. The hard facts:

- The two 4-step arms carry **bit-identical audio**
  (`audio_fnv=0e170b4d53054721` both), and both take the preset's refine sigma
  path - proof that the sigma arm never touches the audio, as the frozen-audio
  contract says. The engine gate says the same thing at a different shape: its
  two refine sigmas give different video hashes and the **same** audio hash
  (`11a34732bcb63274`, section 2).
- The 4-step draft's audio is **0.321x** the six-step draft's RMS (**-9.9 dB**),
  correlation **0.511**: quieter, and not the same signal. The refine cannot
  restore it, because it never re-noises the audio.
- For scale, the route's six-step audio already differs from the one-pass clip's
  (2.35x RMS, correlation 0.390) - so the route always hands over "draft audio",
  and step count is one more turn of that dial.
- Reproducibility is the reason to trust the first two facts at all: the same
  route rendered twice, in two processes on one binary, is bit-identical
  (`md5 d8fcf9f1...`), so the differences above are the step count's and not the
  run's.

A 4-step draft therefore buys 25 s of **video** that the gate passes and a
**10 dB drop in audio level** that nothing in this campaign's instrument set
scores. The honest recommendation: ship the option, and judge the audio by ear
before anyone makes 4 the default for a shot with a sound design.

## 5. What ships

Nothing is defaulted and nothing is hardcoded: the route stays an arm (`--draft`),
the step count stays the job line's own column, and the sigma stays a flag.
What this unit adds is what it learned:

- `crates/ds4-cli/src/bin/h3-rs.rs:114` now says what the preset's 0.72 belongs
  to: the six-step draft, and a shorter draft needs its own sigma
  (`p4_r1`'s 0.55 is the reference's answer to the same question).
- `crates/ds4-core/src/minimax_h3/host.rs` gains a test that pins the arithmetic
  this unit turned on - `serving(4)` ends at **0.8**, above the preset's refine
  sigma, so a 4-step route with `--refine-sigma` unset is a *shape*, not a bug,
  and a future "fix" that moves it has to delete a test that says why.

## 6. Reproducing

```
cd ~/ds4-dfm-rs
PROMPT=$(cut -f1 outputs/two-stage/one.jobs)
printf '%s\t42\t6\t1344\t768\t121\toutputs/two-stage/ref6.mp4\n'        "$PROMPT" > outputs/two-stage/ref6.jobs
printf '%s\t42\t4\t1344\t768\t121\toutputs/two-stage/four-preset.mp4\n' "$PROMPT" > outputs/two-stage/four-preset.jobs
printf '%s\t42\t4\t1344\t768\t121\toutputs/two-stage/four-ladder.mp4\n' "$PROMPT" > outputs/two-stage/four-ladder.jobs
export H3_DIT_GEMM=fp8 H3_VAE_GEMM=bf16i H3_VAE_ATTENTION=tensor
export H3_VAE=/mnt/h3-artifacts/MiniMax-H3-FastH3-4step/FL2VA/video_vae/source
export H3_UPSCALER_DIR=/mnt/h3-artifacts/models/latent_upscale
bash ~/ds4-test/route-four.sh
bash ~/ds4-test/route-four-evidence.sh
bash ~/ds4-test/audio-compare.sh outputs/two-stage/ref6.mp4 outputs/two-stage/four-preset.mp4
./tests/h3_engine_gate --draft
./tests/h3_engine_gate --draft --refine-sigma 0.82
```

The gate's default shape is already this unit's: `steps=4` at 512x512x22, so
`--draft` there is a 4-step route, and the second invocation is the ladder arm's
frozen-audio and two-turn identity check. Instruments on the Spark:
`~/ds4-test/route-four.sh` (the four renders), `route-four-evidence.sh` (pairs,
probe, both gate runs), `audio-compare.sh`, `route-compare.sh`. Logs:
`outputs/two-stage/{ref6,four-preset,four-ladder}.log`, `~/ds4-test/
{route-four,route-four-evidence,audio-four,gate-four}.log`, and the gate records
`~/ds4-test/score-four-{abs,vsref}.json`.
