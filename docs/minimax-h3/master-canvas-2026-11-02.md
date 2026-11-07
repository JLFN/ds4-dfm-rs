# The cleanest master at the released canvas

2026-11-02. Branch `feat/rapid-h3`, checkout `295a2d52`. The operator asked for
the Ipanema clip "pixel free ... perfect image", and separately for something
bigger than 768x1344. This unit answers the first and records why the second is
not a knob.

## 1. Verdict

The canvas is the ceiling: `MAX_PIXELS = 768 * 1344` (`render.rs:33`,
`h3_host.h:8`), and the engine refuses a larger one by name
(`canvas exceeds the released 768*1344 pixel limit`). Neither 2K nor 4K is a
render this checkpoint can make, so the 2K/4K profiles are removed.

Within the release there are exactly three levers that change delivered pixels:
the projection arm, the decoder's rounding, and the encoder. Both were taken to
their clean end as named recipes:

| profile | latent | wall | clip |
|---|---|---:|---|
| `master-fp8` | **the same** as `quality-portrait` (`video_fnv=8de00389fe365e9b`) | 414.4 s of work | `outputs/master-fp8/five-seconds.mp4`, md5 `855a3532…`, 23,481,394 B |
| `master` | a new one (`video_fnv=8b6860b4b7a81da0`) | 455.3 s | `outputs/master/five-seconds.mp4`, md5 `26689b2d…`, 23,297,673 B |

`master-fp8` is the operator's own shot with its pixels redone. `master` is a
different take on the same prompt and seed, because the arm that generates it is
different.

## 2. The pair that isolates pixels

`ipanema-quality` (`quality-portrait`, fp8, `bf16i`, crf 12/veryslow) and
`master-fp8` (fp8, `fp32`, crf 6/veryslow) printed the **same video
fingerprint**: `video_fnv=8de00389fe365e9b`. The denoised latent is identical;
only the decode and the mux differ, which makes the pair a clean measurement of
"the pixels, not the picture":

| pair | PSNR y | PSNR avg | SSIM all | verdict |
|---|---:|---:|---:|---|
| same latent, `bf16i`/crf 12 → `fp32`/crf 6 | 48.200 | **48.929** | **0.9911** | the same picture, redrawn |
| same recipe, fp8 arm → bf16 arm | 19.730 | **21.236** | 0.8055 | a different picture |

The second row reproduces the recorded arm cost (`fp8-row-scales-2026-10-29.md`
§4 measured 21.720 dB for fp8 against bf16): within 0.5 dB, on a different
canvas and checkpoint pair, which is what two independent measurements of the
same effect should look like.

## 3. What the harness scores

Both pairs were scored with the reference's own harness (`tools/h3-clip-score.py`,
`eval_clips.py` sha256 `cdc16564…`), 124 frames at 768x1344. Neither run
carries a flag.

Against `ipanema-quality` (the clip the operator kept), for `master-fp8`:

| metric | value | delta |
|---|---:|---|
| MUSIQ | 60.70 | +0.49 |
| CLIP-IQA | 0.360 | -0.009 |
| DOVER | 0.782 | +0.000 |
| sharpness (Laplacian var) | 26.9 | x1.03 |
| noise sigma | 0.47 | x1.15 |
| temporal HF energy | 0.815 | x1.24 |
| `comp_ssim` | — | **1.000** |

So the clean decode and the crf-6 encode are worth **+0.49 MUSIQ and +3%
sharpness** on an unchanged picture, at 2.8x the bytes (23.5 MB against 8.5 MB)
and +1.7 s of mux. The `noise x1.15` is the crf keeping grain the old encode
smoothed; `tflick x1.24` is more per-frame detail, not more flicker.

Against `master-fp8`, for the bf16 `master`:

| metric | value | delta |
|---|---:|---|
| MUSIQ | **62.29** | +1.59 |
| DOVER | **0.806** | +0.026 |
| sharpness | **28.3** | x1.05 |
| flicker (static-pixel) | **3.138** | **x0.412** |
| `comp_ssim` | — | 0.566 |

The bf16 arm scores best of the four clips and is **2.4x steadier on static
pixels**, which is the one difference large enough to read as quality rather
than as a number. It is also, at `comp_ssim` 0.566, a different take: a viewer
who wants *that* shot does not get it from this arm.

## 4. What each part costs

| stage | `master` (bf16) | `master-fp8` (fp8) |
|---|---:|---:|
| text | 28.013 s | 28.735 s |
| load | 9.811 s | 808.637 s (cold, see §5) |
| denoise, 6 steps | 268.587 s (44.76 s/step) | 233.226 s (38.87 s/step) |
| audio | 3.061 s | 3.046 s |
| video decode | **136.880 s** (`fp32`) | **141.770 s** (`fp32`) |
| mux, crf 6 veryslow | 7.417 s | 7.600 s |
| wall | **455.3 s** | 414.4 s of work |

Two costs are worth naming. The **`fp32` decode is 136.9 s against 46.9 s** for
`bf16i` at this canvas (`ipanema-quality`'s own `video=46.903s`), so 90 s of the
master's wall buys whatever share of the 48.9 dB the rounding contributes; the
encode is the rest and has not been separated (that needs one more decode of the
same latent at `bf16i`/crf 6). The **bf16 denoise is +35.4 s** over fp8
(44.76 against 38.87 s a step), which is the price of the picture the arm
delivers.

## 5. The trap: a second arm pays a page-in

`master-fp8`'s load was **808.637 s**. Nothing was wrong with the render — its
steps ran at 38.7-39.2 s, exactly the `ipanema-quality` numbers. What happened is
that the bf16 leg had put ~40 GB of anonymous pages on a 121 GB box, and the
fp8 leg's checkpoint was no longer in page cache: it came off `/dev/sda` at
30-70 MB/s (`/proc/PID/io` static, `/proc/diskstats` advancing, the loading
thread in `folio_wait_bit_common`, RSS climbing 43.5 → 64.6 GB).

The consequence is operational, not numerical: **two different arms in one
sitting cost about a quarter of an hour of page-in on this box**, while two
renders of the same arm pay 8-10 s. It is the same class as the two-decodes-at-once
hazard in `vae-decode-bytes-2026-11-02.md` §5, and it argues for batching a
sitting by arm rather than by shot.

## 6. What is left, and what is not a setting

The 48.9 dB pair says the old delivery was already near-transparent: crf 12 on
veryslow was not costing the picture anything a viewer can see. The measurements
that moved are the ones that change what is *generated* or how steadily it
moves — the bf16 arm's +1.59 MUSIQ and its x0.412 flicker.

So "perfect image" at this canvas is not a remaining knob. What remains is
outside the recipe: the fp8 GEMM's `amax`+`quantize` fusion (queued, 2.40 s a
step), the canvas-fixed stages (text 27-28 s, audio 3.0 s), and the int8
weight path the reference gets its last ~18 s from, which needs a rotation this
checkpoint does not carry.

## 7. Reproduce

```bash
cd ~/ds4-dfm-rs
export H3_MODEL=/mnt/h3-artifacts/MiniMax-H3-FastH3-4step
tools/h3-rapid.sh --profile master     --prompt-file ~/ds4-test/prompt-ipanema.txt \
  -o outputs/master/five-seconds.mp4
tools/h3-rapid.sh --profile master-fp8 --prompt-file ~/ds4-test/prompt-ipanema.txt \
  -o outputs/master-fp8/five-seconds.mp4
```

Through the API the same recipes are `"profile": "master"` / `"master-fp8"`, but
only on a server started with the matching `H3_DIT_GEMM` — the arm is read once
per load, and a recipe that names the other one is refused with
`profile_needs_reload` rather than served as something else.

Evidence on the Spark: `~/ds4-test/render-master.log`, `measure-master.log`,
`score-master.json`, `score-master-fp8.json`.
