# The route's draft scale: -16.6 s and -31.1 s, and the shot they cost (2026-11-02)

Unit: handoff section 3, the operator's own question - *why is the draft so big,
should it be small resolution and then upscale*? The route already does exactly
that: the draft runs at the preset's **0.7 linear = 928x544**, 49% of the
delivery pixels, and the learned latent resizer takes it to **1344x768** before
one refine step. The open number was *how small*, and the timing said go small:
0.6 is **-16.6 s** of a 186.7 s route and 0.5 is **-31.1 s**, up to **2.23x**
against the one-pass clip.

Both arms were measured, and neither ships: below 0.7 the route stops delivering
the take it was asked for. PSNR against the 0.7 clip is **13.0 dB** at 0.6 and
**12.7 dB** at 0.5 - not "softer", different - and the clip gate flags both
(`musiq_d` -3.77 and -3.13 against a -3.0 budget, `warp_err_r` 1.53 and 1.38).
The mechanism is in the graph, not in a kernel: the draft's latent is a
different shape, so `Rng::seed(42)` draws a different realization for it
(`engine.rs:447-450`), and the reference does the same by handing `RandomNoise`
the draft's own latent (`harness/workflow.py:342`, `:408-409`). **Scale is a shot
knob, not a resolution knob**, and 0.7 is the smallest rung on which the shot
survives.

Spark `gx10-e1d2`, sm_121a, CUDA 13.0, 61-64 C, no clock touched. Every render
is the delivery job - the fox prompt, seed 42, **1344x768x121**, four draft
steps, refine sigma **0.72** held fixed, `--crf 15 --preset slow`,
`H3_DIT_GEMM=fp8 H3_VAE_GEMM=bf16i H3_VAE_ATTENTION=tensor`.

## 1. Four renders, one binary, one session: the ladder's price

`~/ds4-test/route-ladder.sh`. The 0.7 arm is rendered **first and last** (the
last unit's rule: a wall comparison needs its own reference twice), so the two
smaller drafts sit between two rows of the same reference. `LOAD` is the
page-cache share of the wall and is not a property of the arm.

| run | draft canvas | latent | LOAD | DENOISE | draft | upscale | refine | wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.7 head, cold | 928x544 | 58x34 | 50.236 | 88.250 | 49.729 | 3.983 | 38.521 | 227.747 |
| 0.6 | 800x448 | 50x28 | 15.813 | 69.297 | 30.622 | 3.444 | 38.675 | 170.115 |
| 0.5 | 672x384 | 42x24 | 12.731 | 58.705 | 19.790 | 3.233 | 38.915 | 155.574 |
| 0.7 tail, warm | 928x544 | 58x34 | 12.721 | 89.409 | 50.477 | 3.638 | 38.932 | 186.669 |

Warm against warm (0.7 tail):

| arm | wall | DENOISE | draft | upscale | draft+upscale |
|---|---:|---:|---:|---:|---:|
| 0.6 | -16.554 (-8.9%) | -20.112 (-22.5%) | -19.855 (-39.3%) | -0.194 | **-20.049** |
| 0.5 | -31.095 (-16.7%) | -30.704 (-34.3%) | -30.687 (-60.8%) | -0.405 | **-31.092** |

Against the one-pass clip (**347.611 s**) the route is 1.86x at 0.7, **2.04x at
0.6** and **2.23x at 0.5**. Nothing else in the route moves: the refine is
38.5-38.9 s at every rung (the last unit's sigma-independence, now scale-
independence too), and text encode, load, decode and mux are canvas-fixed.

**The draft's cost is area^1.4, not area.** A draft step is **12.619 s** at
1,972 latent tokens, **7.656 s** at 1,400 (71.0% of the tokens) and **4.948 s**
at 1,008 (51.1%), i.e. 6.40, 5.47 and 4.91 ms a token-step. Attention is
quadratic in the sequence, so the bigger canvas pays more per token - which is
why the ladder beat the handoff's area-proportional prediction (35.8 s and
25.8 s) by **5.2 s and 6.0 s**. Price a canvas on two rungs, not on its pixels.

**Where the wall goes.** The canvas-fixed stages are ~131 s on the 0.7 row
(text 27.7, load 12.7, refine 38.9, audio 3.0, video 47.0, mux 1.7). At 0.7 the
draft (50.5 s) is the largest single stage; at **0.6 the decode takes that
place** (video 46.9 s against draft 30.6) - one rung earlier than the handoff
guessed, and it makes the decode the next target, with the draft's per-step cost
right behind it.

**Determinism, per canvas.** The 0.7 head and tail are bit-identical
(`md5 e2126e620ed1bd5d3d524388be4c4c6d` both, PSNR `inf`, `video_fnv
f24ba94a9e9506d6`, `audio_fnv 0e170b4d53054721`, `frames_fnv
f2a5f6a267cba18a`), and that md5 is the 4-step unit's `four-preset.mp4`, so
this session reproduced the previous one's bytes exactly, on a rebuilt binary.
The 0.6
arm was rendered twice: `md5 4ef5d4a7a9fb9360a84062639c28bd4a` both times, and
the fingerprints (`video_fnv 2a3508895e36d402`, `audio_fnv c1b5b8e69f3cf0da`,
`frames_fnv 9e9c77202cc28307`) agree. The 0.5 arm was rendered once at the
delivery canvas; the engine gate ran it twice at 256x256 and printed identical
media (section 4).

## 2. The picture: not the same shot, and the energy that rises is noise

Same canvas, same seed, same prompt, so every pair is comparable. PSNR, SSIM and
detail energy are `~/ds4-test/route-compare.sh`; the clip gate is
`tools/h3-clip-score.py` against `scale07b.mp4` as baseline.

| arm | PSNR y/u/v (avg) | SSIM Y | SSIM All | detail sigma=4 | detail sigma=8 | comp_ssim | bytes |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.7 (reference) | - | - | - | 2.3266 | 4.1298 | - | 4,351,509 |
| 0.6 | 13.001/28.513/30.856 (14.714) | 0.6315 | 0.7326 | 2.8684 (+23.3%) | 4.9345 (+19.5%) | **0.327** | 6,251,656 |
| 0.5 | 12.690/27.792/30.293 (14.399) | 0.6246 | 0.7299 | 2.9708 (+27.7%) | 5.1329 (+24.3%) | **0.261** | 6,273,565 |

For scale: the 4-step draft against the six-step one was **20.2 dB y (21.9
average) with comp_ssim 0.719**, and the route against the one-pass clip was
comp_ssim 0.147.
A two-step reduction was a softer take of the same shot; a one-rung scale
reduction is **between the two**, and it is not a quality trade - the frames show
the fox framed larger with the tree line re-arranged, and the CRF-15 bitrate
rises **44%** because there is more high-frequency content to encode.

The gate's verdict against `scale07b.mp4` (**exit 1**, both arms):

| arm | musiq | clipiqa | dover | sharp | noise | warp_err | flow | accel | zoom | static_frac | flags |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 51.80 | 0.366 | 0.654 | 22.9 | 0.414 | 1.671 | 11.12 | 0.191 | 0.749 | 0.337 | - |
| 0.6 | 48.03 (-3.77) | 0.308 (-0.058) | 0.658 (+0.004) | x1.249 | x1.07 | 2.555 (x1.53) | x0.971 | x1.11 | 0.596 | 0.002 | `warp_err_r, musiq_d` |
| 0.5 | 48.67 (-3.13) | 0.360 (-0.006) | 0.603 (-0.051) | x1.288 | x1.14 | 2.303 (x1.38) | x0.953 | x0.749 | 0.760 | 0.005 | `warp_err_r, sharp_frame_r, musiq_d` |

Two of the gate's families agree with the picture: `warp_err_r` is 1.38-1.53
against a 1.25 budget (occlusion-masked warping error, i.e. frame-to-frame
coherence), and the **static fraction collapses from 0.337 to 0.002-0.005** - at
0.6 both `flicker` and `tflick` come back `null` and at 0.5 `flicker` does,
because there are almost no static pixels left to measure flicker on. A noisy
frame is a moving frame to the warper.

And the detail column is the trap: **detail energy is +23%/+28%, not -10%**. The
4-step unit read a 10% *loss* as the price of a pass; here a 23% *gain* is the
signature of the resizer's high-frequency noise, and it arrives with `musiq`
down 3.8 and `sharp_frame_r` at the 1.25 ceiling. Detail energy is not monotone
in quality: quote it beside `musiq` and `comp_ssim`, never alone.

## 3. The audio, again from the frozen leg

The refine holds the draft's audio (`freeze_audio`), so the delivered audio is
the smaller draft's own take. `~/ds4-test/audio-compare.sh`, 0.7 against each:

| pair | RMS ratio | SNR | correlation | per-second RMS |
|---|---:|---:|---:|---|
| 0.7 -> 0.6 | **1.449** (+3.2 dB) | -0.09 dB | 0.7306 | 0.0041 0.0037 0.0036 0.0039 0.0044 vs 0.0029 0.0027 0.0025 0.0024 0.0030 |
| 0.7 -> 0.5 | **1.666** (+4.4 dB) | -1.64 dB | 0.7084 | 0.0047 0.0043 0.0043 0.0044 0.0049 vs 0.0029 0.0027 0.0025 0.0024 0.0030 |

The whole span is louder, not one section of it, and the correlation is the same
~0.71 the 4-step draft produced (0.511) in the other direction. The video gate
cannot see any of this: it is video-only, and the audio's own fingerprints do
change per canvas (`audio_fnv` 0e170b4d53054721 at 0.7, c1b5b8e69f3cf0da at
0.6, d09d2fc1e3d6c2df at 0.5). A frozen stage remains the instrument set's blind
spot, and this unit is the second measurement of it.

## 4. The engine gate, at three canvases

The gate could not take a draft scale before this unit (`--draft` was the
preset's 0.7 with no argument), so it does now: `h3_engine_gate --draft [SCALE]`,
and the scale is a property of the route's canvas rather than of the preset.
At the gate's own shape, 512x512x22, steps=4:

| `--draft` | draft canvas | draft | upscale | video | audio | frames | verdict |
|---|---|---:|---:|---|---|---|---|
| 0.7 | 352x352 | 1.382 | 0.768 | `2c224536f5f00113` | `11a34732bcb63274` | `16a19b26f173619d` | PASS `text_hit audio_frozen` |
| 0.6 | 320x320 | 1.104 | 0.409 | `217d18fdb821aa43` | `e96f8e2d2873f688` | `3587a8d09c65f1d7` | PASS `text_hit audio_frozen` |
| 0.5 | 256x256 | 0.810 | 0.401 | `f981591f3400cdb6` | `bbb378611f1abb40` | `60cebec41583100c` | PASS `text_hit audio_frozen` |

Every arm holds the route's two invariants at **its own** canvas - the audio the
refine conditioned on is the audio the decoder received (`draft_audio ==
audio`), and two turns of the same request print identical media - and the 0.7
hashes reproduce the 4-step unit's row exactly. The audio hash differs per
canvas and is stable within one, which is the per-canvas determinism the ladder
needed and could not get from a single render.

The upscaler's own cost tracks the **output** canvas, not the ratio: 0.768 s for
352x352 -> 512x512, 0.409 for 320 -> 512, 0.401 for 256 -> 512. At the delivery
canvas that is 3.6, 3.4 and 3.2 s - the draft's scale barely reaches it.

## 5. What ships

- **`h3_engine_gate --draft [SCALE]`**, because the route's canvas is a
  measurement surface now (`crates/ds4-core/examples/h3_engine_gate.rs:108`).
- **A test pinning the ladder's middle canvas** -
  `a_sixty_percent_draft_is_the_ladders_middle_canvas`
  (`engine.rs:1079`): 0.6 of 1344x768 is 800x448, of 768x1344 is 448x800, and of
  the gate's 512x512 is 320x320.
- **Nothing defaulted.** 0.7 stays the route's scale and `--draft` still carries
  the number; the reference's own presets all use 0.7 (`workflow.py:50`) and its
  `half` is the spelling of an experiment (`:393`), not a preset.
- **The guidance this unit adds:** 0.7 is the smallest rung the shot survives,
  so the route's remaining cost is the draft's *per-step* work (12.62 s a step,
  the E4M3 conversion arm's target) and the delivery decode (47.0 s, now the
  largest single stage), not its canvas.

## 6. Reproducing

```bash
cd ~/ds4-dfm-rs && git pull --ff-only
export H3_VAE=/mnt/h3-artifacts/MiniMax-H3-FastH3-4step/FL2VA/video_vae/source
export H3_UPSCALER_DIR=/mnt/h3-artifacts/models/latent_upscale
export H3_DIT_GEMM=fp8 H3_VAE_GEMM=bf16i H3_VAE_ATTENTION=tensor
bash ~/ds4-test/route-ladder.sh        # 0.7 cold, 0.6, 0.5, 0.7 warm (~13 min)
bash ~/ds4-test/ladder-evidence.sh     # audio, pairs, clip gate (~10 min)
H3_MODEL=/mnt/h3-artifacts/MiniMax-H3-FastH3-4step ./tests/h3_engine_gate --draft 0.7
H3_MODEL=/mnt/h3-artifacts/MiniMax-H3-FastH3-4step ./tests/h3_engine_gate --draft 0.6
H3_MODEL=/mnt/h3-artifacts/MiniMax-H3-FastH3-4step ./tests/h3_engine_gate --draft 0.5
```

Clips and jobs are `outputs/two-stage/scale0{7a,7b,6,5}.{mp4,jobs,log}`, plus
`scale06b.mp4` (the determinism re-render); evidence is
`~/ds4-test/{route-ladder.log,ladder-evidence.log,gate-ladder.log,score-ladder.json}`.
