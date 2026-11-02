# Render profiles: the recipe as a file, and the draft→upscale it names

Two things landed together: the route's recipe became a named, versioned file
(`tools/h3-profiles.json`) instead of a set of script defaults, and the recipe
that file documents was verified line by line against the reference it came from.
The verification is the substance; the file is what makes it reusable.

## 1. The recipe is the reference's, stage for stage

`/data/Rapid-H3`'s default preset is `ts_h2`
(`harness/workflow.py:50-52`):

```python
_H2 = dict(_RAPIDH3, steps=6, draft="0.7", refine_sigmas="0.72, 0",
           freeze_audio=True, decode_save="overlap")
"ts_h2": dict(_H2, encode_once="draft", rapidh3_attn=190, te_quant="int8_rot")
```

and the two-stage graph it builds (`:392-415`) sizes the draft as
`round(w*0.7/32)*32`, samples the whole schedule there, then hands the draft's
**video latent** to a learned resizer with `mode="target dimensions"` and the
delivery width and height.

| stage | reference | ours |
|---|---|---|
| draft sizing | `round(w*f/32)*32`, `f = 0.7` → 928x544 | `tools/h3-rapid.sh:192-195`, same round → `PLAN delivery=1344x768 draft=928x544 scale=0.7` |
| upscale | `MinimaxH3LatentUpscaler3D`, `mode="target dimensions"`, delivery dims, `align 32`, fp16, no temporal chunking | `Upscaler::upscale(latent, t, h, w, out_h, out_w)` called with the delivery dims (`engine.rs:524-533`), checkpoint `minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors` |
| resizer network | `LatentResizer3D`: conv_in 24→512, 12 `ResBlockEmb3D` interleaved with 6 `TemporalConv`, trilinear resize, 12 more, `norm_out`, conv_out 24, GroupNorm 32, conditioned on `scale-1` | the same 322-tensor network (`latent_upscale.rs`), released `latents_mean`/`latents_std`, `scale = mean(out_w/w, out_h/h)` at `:659` |
| refine | `refine_sigmas="0.72, 0"` = one step from sigma 0.72 | `--refine-sigma 0.72`, `Schedule::refine`, `refine=38.4 s` |
| audio | `freeze_audio=True` — the draft's audio is kept | `audio_frozen=true`, `audio_fnv` identical to the draft's |
| text | `encode_once="draft"` — one Qwen pass | second stage reports `text=0.000s (hit)` |

So the answer to "doesn't it do a smaller draft and then just scale up" is: yes,
and we do the same thing, but **the scale-up is not a resize of pixels and not a
plain latent interpolation** — it is a learned 3D resizer that takes the H3
24-channel latent in and returns the same 24-channel latent out, with the
convolutions either side of its internal trilinear resize re-synthesizing the
detail the resize cannot invent. That is why it costs 3.77 s (ours) / ≈3 s
(theirs) instead of the minutes a decode-and-re-encode would.

**The "×2" in the reference's README is the model's nominal factor, not the
factor in use.** The node is driven by target dimensions, so with a 0.7 draft the
real ratio is `1/0.7 ≈ 1.43`: latent 34x58 → 48x84. The network is told the real
ratio (`scale = 1.430` = mean of 84/58 and 48/34), which is what the `scale - 1`
embedding exists for. A port that hard-coded ×2 would double the draft and be
1.96x too large.

**0.7 is the floor on both sides.** The reference rejects 0.6 and 0.5 drafts
because "smaller drafts crop in to close-ups" (`README.md:172`); this campaign
measured the same failure before reading that line (0.6/0.5 re-roll the shot:
PSNR 13.0/12.7 dB, `comp_ssim` 0.33/0.26, `draft-scale-ladder-2026-11-02.md`).

## 2. The step count, measured on one binary in one session

The reference's `ts_h2` runs 6 draft steps; our route default is 4. Both clips
below are the same prompt, seed 42, 1344x768x120, draft 0.7, refine 0.72, on the
`bf16a` decode:

| leg | steps | draft | refine | decode | wall | detail σ4 | detail σ8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rapid` | 4 | 49.7 | 38.4 | 28.1 | **181.2 s** | 2.7528 | 4.6945 |
| `ts_h2` | 6 | 74.3 | 38.4 | 28.1 | 193.2 s | **2.8980** | **4.9547** |

- Two draft steps cost **+24.6 s of arithmetic** (12.3 s each); the walls differ
  by 12.0 s only because the page-cache load differed by 11.7 s, which is not a
  property of the arm.
- **The six-step leg buys +5.3% detail energy (σ4) and +5.5% (σ8)** - and pays
  MUSIQ **-2.01**, CLIP-IQA -0.021, DOVER -0.002, sharpness x0.976. The clip gate
  **passes it with no flag** (`musiq_d` -2.01 against a -3.0 budget).
- `comp_ssim` 0.746 and SSIM 0.785 say these are two different takes of the shot,
  not one picture at two qualities, so the pair is a judgement call the campaign's
  own gate is allowed to refuse - and it does not.

**Verdict: 4 steps is the right default.** The checkpoint is a 4-step distill
(`MiniMax-H3-FastH3-4step`); two more steps make it sharper by 5% and less
pleasant by the perceptual metrics, for +25 s. The reference reaches 6 steps from
the *stock* checkpoint, which is a different starting point - copying its step
count onto a distilled model is not following its recipe, it is misreading it.

## 3. The profile file

`tools/h3-profiles.json`, one entry per recipe, keys named after the script's own
flags:

```bash
tools/h3-rapid.sh --list-profiles
tools/h3-rapid.sh --profile rapid   -p "…" --seconds 5 -o outputs/x.mp4
tools/h3-rapid.sh --profile ts_h2   -p "…"                          # the reference's count
tools/h3-rapid.sh --profile one-pass -p "…"                         # its ts_s6 shape
```

Shipped: `ts_h2` (6 steps, the reference's preset), `rapid` (4 steps, the speed
point), `rapid-e4m3` (the draft on the E4M3 arm), `one-pass` (no draft, the
`ts_s6` shape, 347.6 s), `portrait` (768x1344) and `ipanema` (a whole shot).

A profile may also carry `prompt` or `prompt_file`, which makes it a complete
shot rather than only its settings - `ipanema` holds its scene text, so
`--profile ipanema` is one word for a full 9:16 render. An explicit `-p` still
wins, and a profile that carries no prompt is settings only, as the first five
are.

Precedence, and it matters: **a flag beats a profile, and a profile beats the
environment.** The route runs `target/release/h3-rs`, and a shell that still
exports `H3_VAE_GEMM=bf16i` would otherwise silently override the recipe - which
is exactly how the first render of the previous unit measured the wrong arm. Every
render prints its resolved recipe:

```
PLAN route=two-stage profile=ts_h2
PLAN steps=6 seed=42 sigma=0.72 draft-attention=bf16
PLAN arms H3_DIT_GEMM=fp8 H3_VAE_GEMM=bf16a H3_VAE_ATTENTION=tensor
```

The loader (`tools/h3-profile.py`) is standard library only, refuses an unknown
key by name rather than ignoring it, and `shlex.quote`s every value so a `doc`
with an apostrophe cannot break the `eval`.

## 4. Run for real, twice

The profile path was not only dry-run. Both renders below are the same prompt,
seed 42, five seconds, on `2cd8e5f0`:

| command | canvas | draft | steps | draft | refine | decode | wall | md5 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `--profile ts_h2` | 1344x768 | 928x544 | 6 | 74.4 | 38.5 | 28.1 | 192.3 s | `89392d81e25979439e3c0af13678ba2d` |
| `--profile portrait` | 768x1344 | 544x928 | 4 | 50.0 | 38.7 | 28.2 | 168.1 s | (its own, `frames_fnv 4c77fcbeefab146f`) |
| `--profile ipanema` | 768x1344 | 544x928 | 4 | 49.9 | 38.5 | 28.2 | 167.3 s | `7b2caec8d8611c86f594d066675e73e7` |

- **The profile is the flags it expands to, bit for bit**: `--profile ts_h2`
  reproduced the earlier `--steps 6` render exactly - same md5, same
  `video_fnv=eaac4c9dd7d0911c`, same `frames_fnv=c0a9c6d218f5884e`, same
  6,105,975 bytes. A recipe that renders something slightly different from the
  flags it claims to be would be worse than no recipe.
- **The canvas does not change the arithmetic**: 768x1344 costs the same as
  1344x768 at equal pixels (draft 50.0 against 49.7, refine 38.7 against 38.4,
  decode 28.2 against 28.1). The 181.2 s of the 16:9 `rapid` run in §2 against
  this 168.1 s is the page-cache load (26.2 against 13.6), not the shot.
- One correction this run produced: the `portrait` profile's own doc said its
  draft lands at 544x960; `round(1344*0.7/32)*32` is **544x928**. The recipe was
  right and its description was wrong, which is the failure mode a profile file
  exists to prevent - fixed in the same commit.

## 5. Traps

- Recipes describe, they do not define: the `portrait` doc carried a wrong draft
  size (544x960 for 544x928) through a commit, and only running it caught that.
  `--dry-run` prints the resolved geometry for exactly this reason.
- A dry run used to write the `.jobs` file - and the path may be a pinned
  clip's own job line. It clobbered `outputs/rapid/rapid.jobs` with a throwaway
  prompt during this unit's smoke test (restored). A dry run now prints the line
  instead of writing it.
- `--dry-run` still creates nothing: the earlier behaviour `mkdir -p`'d the
  output directory; that moved behind the same check.
- The profile is applied by `eval`ing generated shell assignments **before** the
  flag loop, not after: applied afterwards it would silently beat an explicit
  flag, which is the opposite of what a default is for.
- Recipes live in the repo, so a clip's recipe is reproducible from git; the
  clips themselves stay in `outputs/` (gitignored) and the pinned set is stale
  (`vae-decode-chain-2026-11-02.md` §5).
