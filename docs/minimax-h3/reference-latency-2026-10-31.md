# Where the reference's 56 s goes, and where ours goes

Status: open. One optimization landed (`7a5e9b04`), the sharpness cause is
measured, and one profile is missing before any latency target is promised.

## 1. The claim and its scope

`/data/Sol-h3-Spark/README.md:46`: "On a single NVIDIA DGX Spark at 1344x768,
T2V end-to-end latency is 56 s for a 5-second output."

`runtime/pipeline.py` defines the clock the claim is about: one monotonic
interval from request entry to the completed muxed MP4 (`:175-189`), recorded
per request as `e2e_s` beside `qwen_s`, `stage1_s`, `stage2_s` and
`stage2_phases_s` (`:194-197`). Warmup is excluded; the first formal request
includes the first Qwen encode (`absolute_paths`: "first_formal_request_
included": True). No `results.json` ships with the tree, so the 56 s has no
per-phase evidence on this unit. Everything below that is marked measured was
measured here; everything derived is labelled.

## 2. What the reference does differently, read from its code

| # | Mechanism | Source | What it removes |
|---|---|---|---|
| 1 | The H3 video is never decoded: Stage 1 saves no video, returns no frames, and the capture asserts `video_decode_calls == 0` and `h3_decoder_calls == 0` | `runtime/stage1.py` (`save_video=False`, `return_frames=False`, capture checks; `resolved_config` records `"video_decode": False`); `README.md` "The latent adapter replaces an intermediate H3 VAE decode and LTX VAE encode" | Our 109.2 s H3 VAE decode at 672x384x124 |
| 2 | Models stay resident between requests; Stage 2 prompt contexts are computed offline, so Gemma and the connector never load online | `README.md` "Why the models can remain resident"; `docs/validation.md` "Gemma and the connector are absent from online inference" | Our 28.2 s `TEXT` and 41.7 s `LOAD` per process |
| 3 | Stage 1 attention is 90% sparse over 64-token tiles on cuDNN BSA, 200 calls and 200 fused merges per request | `runtime/stage1_ops/vsa.py` `configure_profile`; `docs/validation.md` | Part of our denoise (our VSA exists as `H3_DIT_SOL`, off by default) |
| 4 | Stage 1 DiT is W8A8 FP8 after the BF16 LoRA merge | `runtime/stage1.py` `resolved_config`: "native W8A8 after BF16 LoRA merge; exact BF16 AdaLN lookup" | Our 57.5 % BF16 GEMM share per evaluation |
| 5 | Stage 2 is three BF16 updates with Triton Sol (141 Sol calls, 3 dense), the distilled LoRA at 0.8, and the decoder streamed in 16-frame chunks straight into ffmpeg | `runtime/stage2.py` `_checked_attention_stats`, `write_full_vae_video`; `runtime/stage2_ops/writer_chunking.py`; `docs/validation.md` | Our 110.5 s denoise and our whole-clip decode-and-mux |

## 3. Our phases, measured on this Spark

Shapes: draft 672x384x124, refine 1344x768x121. Dense BF16 stage 1 (no
`H3_DIT_SOL`), Sol-Attn and the distilled LoRA at 0.8 in stage 2.

| Arm | Phases | Sum |
|---|---|---|
| 4-step FastH3 draft, with video | 28.22 + 39.35 + 28.41 + 3.09 + 109.16 + 0.60 | 208.8 s |
| 4-step FastH3 draft, `--latent-only` | 28.24 + 41.73 + 28.76 + 3.11 + 0.00 | 101.8 s |
| 20-step base H3 draft, with video | 27.32 + 41.25 + 143.29 + 2.98 + 109.06 + 0.51 | 324.4 s |
| Transfer only | 0.31 + 2.83 + 15.14 + 0.83 | 19.1 s |
| Refiner, 3 updates | 2.83 + 41.0 + 36.6 + 33.0 + 15.7 + 0.79 | 129.9 s |

The latent-only flag is `7a5e9b04`. Its measurement is the first two rows:
one decode and one mux, 107.0 s of the 208.8, removed without touching text,
load, denoise or audio.

## 4. The sharpness finding, measured

Laplacian variance of the frame at 2.0 s, the same operator on every row:

| Input / output | 672x384 | 1344x768 |
|---|---|---|
| 4-step draft | 25.31 | - |
| 20-step draft | 35.93 | - |
| bicubic 2x of the same draft frame | - | 2.71 (4-step) / 3.65 (20-step) |
| transfer-only output | - | 3.82 (4-step) / 4.63 (20-step) |
| transfer-only output, projected back to 672x384 | 8.65 / 15.91 | - |
| 3-step refiner output | - | 10.42 |

Two conclusions. The transfer path is barely sharper than a bicubic resize:
the LTX VAE is asked to decode a latent the refiner was meant to denoise
first, and that direct decode softens it. The refiner is 2.25x sharper than
the transfer, and it changes the animal: at 2.0 s the draft's tall pointed
ears and narrow muzzle come back as short rounded ears and a blunt open
muzzle. So the sharp route alters and the faithful route is soft, on the same
latent.

## 5. The question that decides the plan

One refiner step must read the transformer it uses: 48 blocks sum to 34.6 GiB
BF16 in the header of `ltx-2.5-22b-dev-transformer-bf16.safetensors`, of which
the video-only branch is about 24 GiB. Our 36.8 s per step then implies an
effective 0.7-1.0 TB/s, three to four times the 246 GB/s this host streams in
our own measurement (`docs/gb10-hardware.md`). One of the two numbers is
wrong, and the difference decides whether the refiner has 3x of headroom or
none. It needs one profile of a single step before anyone promises 56 s here.

## 6. Next, in order

1. Residency: serve the draft through `h3-server-rs` with the NVFP4 prompt
   encoder. Our own server record already shows `TEXT` 0.001 s and `LOAD`
   0.000 s on the second render in a process
   (`docs/minimax-h3/steady-state-2026-10-28.md`).
2. Profile one refiner step with `nsys`, then `ncu` on its largest kernel, to
   settle section 5.
3. Make the stage-2 sigma schedule a runtime value and test a one-step pass
   from a low sigma as the detail pass: the anchor stays, the model supplies
   the texture the direct decode loses.
