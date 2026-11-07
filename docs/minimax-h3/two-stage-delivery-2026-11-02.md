# The delivery job on the two-stage route: 1.61x on the wall, and a different shot (2026-11-02)

Unit: handoff section 3, the delivery through the two-stage route. The route has
been in the engine since `two-stage-route-2026-11-01.md` and the gate has always
asserted its freeze; what had never happened is **the delivery job rendered
through it on the shipped arms**. It has now, beside the same job without the
route in the same session on the same binary.

**The verdict first.**

| | one pass | two stage | |
|---|---:|---:|---|
| wall | 347.611 s | **216.068 s** | **1.61x** |
| `DENOISE` | 232.699 s | **115.031 s** | **2.02x** |
| of which draft | — | 76.045 s (6 steps, 12.674 s/step) | 0.327 of a step |
| of which upscale | — | 4.047 s | at 928x544 -> 1344x768 |
| of which refine | — | 38.986 s (1 step) | = the one-pass step |
| `LOAD` | 7.188 s | 16.018 s | +8.83 s, two graphs |
| `TEXT` | 54.397 s | 27.683 s | variance, not the route |
| `FRAMES` | 47.051 s | 47.064 s | unchanged |
| PSNR y / avg | — | **12.53 / 14.23** | against the one pass |
| SSIM Y / All | — | **0.541 / 0.666** | against the one pass |
| detail sigma 4 | 4.6499 | **2.5913** | 0.56 of the one pass |
| detail sigma 8 | 8.0901 | **4.5724** | 0.57 of the one pass |
| clip md5 | `878f0f7d8d8e28dd…` | `d8fcf9f171952c26…` | different by construction |

So the route buys **1.61x on the wall** (2.02x on the denoise it touches,
less 8.8 s of extra load), and it returns **a different shot**: not a softer
version of the same frame, a different composition at 12.5 dB. Section 3 shows
which stage makes that choice, and it is the draft canvas, not the upscale.

## 1. One job line, two routes

The delivery job has only ever run one-shot, and the route is reachable only
through `--jobs`, so the job got a line (7 TAB-separated fields,
`jobs.rs:7-10`):

```
prompt <TAB> 42 <TAB> 6 <TAB> 1344 <TAB> 768 <TAB> 121 <TAB> outputs/two-stage/<route>.mp4
```

The prompt is `outputs/decode-unit/prompt.txt` (the fox, md5
`98fd163de409fb3713c6b4d14bb10d6d`) joined to one line, because a job line is one
line; the words are unchanged and `TEXT` hashes are therefore not comparable
with the earlier one-shot records. Both routes took the delivery's own arms -
`H3_DIT_GEMM=fp8`, `--vae-gemm bf16i --vae-attention tensor`, `--crf 15
--preset slow` - and differed in one flag: `--draft 0.7`.

```
JOB n=1 canvas=1344x768x121 steps=6 seed=42 text=54.397s load=7.188s denoise=232.699s
  draft=0.000s upscale=0.000s refine=0.000s audio_frozen=true audio=3.019s video=47.051s
  mux=1.822s wall=347.611s video_fnv=1e11a80f73242ad6 audio_fnv=be226ce13ca93358
  frames_fnv=a4318ed80e026ff7 out=outputs/two-stage/one.mp4
JOB n=1 canvas=1344x768x121 steps=6 seed=42 text=27.683s load=16.018s denoise=115.031s
  draft=76.045s upscale=4.047s refine=38.986s audio_frozen=true audio=3.082s video=47.064s
  mux=1.691s wall=216.068s video_fnv=15caddb0cdeebf23 audio_fnv=0b8436940a1faf88
  frames_fnv=d1ce47a36ce76921 out=outputs/two-stage/draft.mp4
```

Both write 124 frames of 1344x768 at 24 fps. `--draft 0.7` on a 1344x768 job is
a **928x544** draft canvas (`round(1344*0.7/32)*32 = 928`,
`round(768*0.7/32)*32 = 544`; `engine.rs:55`), which is 0.489 of the pixels and
0.239 of the attention work once its rows are squared.

## 2. Where the 117.7 s came from, and where 26.7 s did not

The route's own effect is the denoise figure: 232.699 -> 115.031 s, a **117.67 s**
saving. Its parts are arithmetic, not luck:

- **The draft step is 12.674 s against a delivery step's 38.783**, i.e. 0.327.
  That is *below* the pixel ratio 0.489 because 68% of a step is attention, which
  scales with the square of the row count: a control render at 928x544 one-pass
  measured 75.045 s over the same six steps (12.508 s/step), so the draft stage
  inside the route costs the same as a standalone render at its canvas (+1.3%).
- **The refine step is 38.986 s against the one-pass step's 38.783** (+0.5%), as
  it must be: it is one step at the delivery canvas.
- **The upscale is 4.047 s.** `two-stage-route-2026-11-01.md` measured 4.6 s at
  768x1344x141; 121/141 of the frames is 3.95 s, so the two agree.
- **The load grew by 8.83 s** because the two stages are two prepared graphs and
  neither is resident across them.
- The remaining 26.7 s of the wall delta is **`TEXT`, not the route**: 54.397 in
  the one-pass run against 27.683 in the route run, while the control run
  measured 59.046. Prompt encode on this host moves between about 28 and 59 s
  across invocations, so a wall A/B that is not interleaved must be read on
  `DENOISE`, exactly as the campaign's other records do.

## 3. The picture: a different shot, and the draft canvas chose it

The clips cannot be equal - the refine re-noises rather than continuing
(`two-stage-route-2026-11-01.md` section 1.1, citing `comfy/samplers.py:993`) -
but the size of the
difference is larger than "softer", and the stills say why: the one-pass clip is
a low close-up with the fox filling the left third, and the route's clip is a
wider shot with the fox smaller, facing the other way, on much smoother snow.

| still | one pass | two stage |
|---|---|---|
| frame 60 | close-up, fox fills the left third | wide, fox mid-frame, right-facing |

**The control localizes it.** A one-pass render of the same job at the *draft
canvas* (928x544, 6 steps, seed 42) produced the same composition as the route's
clip, and - more decisively - the **same audio hash**, `0b8436940a1faf88`, which
is also the route's own `audio_fnv`. The draft stage is therefore bit-faithful
to a standalone render at its canvas, the audio crosses the route untouched, and
the composition difference is the 0.7 canvas's choice, not a defect of the
upscaler or the refine. Value-level, the upscale cannot put back detail that the
draft never sampled: detail energy is 2.5913 against 4.6499 at sigma 4 and
4.5724 against 8.0901 at sigma 8, at the same 1344x768.

This is the reference's own trade-off - `ts_h2` "gives up close-up detail and
weakens push-ins" - now measured on this port instead of inferred from one pair
of frames (`two-stage-route-2026-11-01.md` section 6).

## 4. The gate, on the binary that rendered

```
./tests/h3_engine_gate --draft
STAGES turn=1 draft=1.333s upscale=0.395s refine=0.682s load=11.921s text_hit=false
  prepared_hit=false draft_audio=11a34732bcb63274 audio=11a34732bcb63274 frozen=true
STAGES turn=2 draft=1.327s upscale=0.123s refine=0.694s load=11.962s text_hit=true
  prepared_hit=false draft_audio=11a34732bcb63274 audio=11a34732bcb63274 frozen=true
MEDIA identical=true first=frames=22 video=2c224536f5f00113 audio=11a34732bcb63274
RESULT PASS two-stage text_hit audio_frozen
```

`draft_audio == audio` on both turns, the repeat is media-identical, and the
report's `audio_frozen=true` agrees with the delivery run's own line.

## 5. What this establishes, and what the decision is

- **The route is worth 1.61x on the wall at the delivery job**, 2.02x on the
  denoise it replaces, and it costs 8.8 s of extra load and 4.0 s of upscale.
- **It is a different picture at these settings**, and the settings are the
  reference's: scale 0.7, sigma 0.72, one refine step. They are not knobs to
  tune (`two-stage-route-2026-11-01.md`), so the honest consequence is that
  `--draft` stays **an arm, never the default**: a caller who wants this shot
  and this wall asks for it, and the one-pass arm remains the delivery default.
- **What is not established:** one prompt, one seed, one clip pair. The
  campaign's scoring harness (`harness/eval_clips.py`) is not wired to this
  port, so the PSNR/SSIM/detail table is a measurement of this job and not a
  quality verdict across prompts. The 4-step draft arm (the checkpoint is a
  4-step distillation) was not run at the delivery job; six-plus-one is what the
  reference does and what the direct arm does, which is what makes the
  comparison fair.

## 6. Reproducing

```
cd ~/ds4-dfm-rs
printf '%s\t42\t6\t1344\t768\t121\toutputs/two-stage/one.mp4\n'   "$PROMPT" > outputs/two-stage/one.jobs
printf '%s\t42\t6\t1344\t768\t121\toutputs/two-stage/draft.mp4\n' "$PROMPT" > outputs/two-stage/draft.jobs
printf '%s\t42\t6\t928\t544\t121\toutputs/two-stage/draftcanvas.mp4\n' "$PROMPT" > outputs/two-stage/control.jobs
export H3_DIT_GEMM=fp8 H3_VAE_GEMM=bf16i H3_VAE_ATTENTION=tensor
export H3_VAE=/mnt/h3-artifacts/MiniMax-H3-FastH3-4step/FL2VA/video_vae/source
export H3_UPSCALER_DIR=/mnt/h3-artifacts/models/latent_upscale
./target/release/h3-rs -d /mnt/h3-artifacts/MiniMax-H3-Turbo-v4 --jobs outputs/two-stage/one.jobs   --crf 15 --preset slow
./target/release/h3-rs -d /mnt/h3-artifacts/MiniMax-H3-Turbo-v4 --jobs outputs/two-stage/draft.jobs --draft 0.7 --crf 15 --preset slow
./target/release/h3-rs -d /mnt/h3-artifacts/MiniMax-H3-Turbo-v4 --jobs outputs/two-stage/control.jobs --crf 15 --preset slow
bash ~/ds4-test/route-compare.sh outputs/two-stage/one.mp4 outputs/two-stage/draft.mp4
./tests/h3_engine_gate --draft
```

Instruments on the Spark: `~/ds4-test/route-delivery.sh` (both routes),
`~/ds4-test/route-compare.sh` (PSNR/SSIM/detail of one clip against another;
`metrics.sh` hardcodes the two clips of the FP8 decision, and the arm it
compared against is no longer on disk - `outputs/` was wiped before this unit, so
the baseline was re-derived in the same session instead). Logs:
`outputs/two-stage/{one,draft,control}.log`, `~/ds4-test/route-delivery.log`,
`~/ds4-test/route-compare.log`, `~/ds4-test/gate-draft2.log`. Stills:
`outputs/two-stage/stills/{one,draft,control}-{40,60,100}.png`.
