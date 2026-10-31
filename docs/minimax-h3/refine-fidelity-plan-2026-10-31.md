# Refined-clip fidelity: the fox becomes a feline

Status: open. The stage-1 draft is a fox; the refined output is a cat/fox hybrid.
This plan isolates the cause before any speed work resumes.

## What was observed

- Draft: /mnt/h3/draft.mp4, 672x384, 124 frames, 4 updates, seed 42, from
  MiniMax-H3-FastH3-4step. Frame at 1.0 s is a clean red fox.
- Refined: /mnt/h3/refined.mp4, 1344x768, 121 frames, 3 updates, Sol-Attn,
  distilled LoRA, sha256 c2d368a94de595bfa164754717bc8257d20ce688b20c6ab30f522cdd87c9c5bf.
  Frame at 1.0 s is a feline hybrid: same body colour and pose, wrong ears and face.
- Frames used for comparison: /mnt/h3/draft-1s.png and /mnt/h3/refined-1s.png,
  extracted with ffmpeg -ss 1.0 -frames:v 1.
- The draft stage is not suspect in shape or scheduling: it produced coherent
  video and audio, and the refined clip keeps the draft's pose and palette.

A cat that keeps the fox's pose and colour is the signature of a latent that
carries the scene but has lost identity detail, not of a decode or layout fault:
a wrong layout would produce noise, and a wrong temporal crop would shift the
pose instead of the species.

## Settings, reference against this run

Reference: /data/Sol-h3-Spark/models/minimax_h3/Sol-H3-Spark/configs/default.json.

| Setting | Reference | This run | Status |
|---|---|---|---|
| Stage-2 distilled LoRA strength | 0.8 | 1.0, hardcoded at h3-sol-rs.rs:171 | Confirmed mismatch |
| Stage-1 attention | VSA, 90% sparsity, 64-token blocks | dense (H3_DIT_SOL unset) | Confirmed mismatch |
| Stage-1 DiT precision | W8A8 FP8 after BF16 LoRA merge | BF16 path | Confirmed mismatch (numerics may differ) |
| Stage-1 updates | 4 | 4 | Match |
| Stage-1 sigmas | [1.0, 0.972972989, 0.923076928, 0.8, 0.0] | engine schedule from h3_dit_schedule | Unverified |
| Stage-2 sigmas | [0.909375, 0.725, 0.421875, 0.0] | same (ltx_stage2.rs:43) | Match |
| Stage-2 taus / dense layer | [1.0, 1.25, 1.5], layer 0 dense | same (TAU_SCHEDULE, enable_sol) | Match |
| Stage-2 prompt/context | fixed prompt, INT8 Gemma + INT8 connector, cached | same, fingerprint d37b38a9... | Match |
| Upscaler normalization | author N/D around the network | port's forward applies N/D (ltx_adapter.rs:1773) | Match by code |
| Adapter input normalization | "normalized" (no-op), then align | convert = align + model | Match by code |
| Refiner input | first 16 of 17 latent frames | crop_frames keeps 16 | Match by code |

The three confirmed mismatches are enough to explain a fidelity failure; the
LoRA strength is the cheapest and most likely first cause because the refiner
is distilled for exactly three updates at 0.8.

## Results so far (2026-10-31)

- Dense attention at LoRA 1.0: feline (/mnt/h3/refined-dense.mp4, 126.0 and
  127.5 s for the first two steps).
- Sol attention at LoRA 0.8: still feline (/mnt/h3/refined-lora08-sol.mp4,
  111.6 s denoise, 15.2 s VAE). The strength and the routing are therefore not
  sufficient causes.
- The VAE's own `latents_mean`/`latents_std` (FL2VA/video_vae/config.json) equal
  the port's `H3_LATENTS_MEAN`/`H3_LATENTS_STD` exactly, so the draft latent and
  the upscaler's N/D share one normalization. That suspect is closed.

### The transfer is faithful; the refiner is what changes the animal

`--transfer-only` (new in this branch) decodes the upscaler + adapter output
through the LTX VAE without the refiner. Both drafts, VSA and dense, come back
as foxes at 2.0 s (/mnt/h3/transfer-vsa.mp4, /mnt/h3/transfer-dense.mp4, 15.2 s
decode each, 2.8 s transfer). So the "resize" is not altering the image: the
learned x2 upscaler, the adapter and the LTX decoder preserve the subject.

The reference does not promise a pure resize. Its stage 2 is a diffusion
refiner: `noised_state` adds noise at `SIGMAS[0] = 0.909375`, then three Euler
updates run against the text context and the clean, frozen H3 audio
(`runtime/stage2.py`, `stage2_ops/models.py:19`, `configs/default.json`). It
re-synthesises detail by construction; what keeps the subject is the draft
latent as conditioning, the 0.8 LoRA, and the joint audio state.

So the user-visible requirement "resize only, never alter" is exactly what
`--transfer-only` provides; the refined route is a different product and needs
the reference's conditioning to stay faithful.

### A structural deviation found while reading the release

The port's stage 2 implements `RefinerEncoderAblationStage2`
(`super_acceleration/stage2/refiner_encoder_ablation_single_gpu.py`), which
denoises with `audio_state=None` and runs video-only
(crates/ds4-core/src/minimax_h3/ltx_stage2.rs:5-10, :318-326). The release
runtime that produced the published numbers is `runtime/stage2.py`, and its
`prepare_multimodal_states` builds a **frozen audio state** from the H3 PCM
(`encode_source_audio` through the LTX audio VAE, then `denoise_mask=zeros`,
`frozen=True`), so the joint AV transformer runs its audio branches too.

The ablation class exists in the same release, but it is the class used to
measure what the audio encoder contributes; it is not the product route. The
port has no LTX audio VAE encoder module (only the H3 `vae_audio.rs`), so the
joint route was never wired.

This is now the leading candidate: a coherent animal with the right pose and a
wrong head is what a refiner can produce when its sequence is missing the
conditioning it was trained with. It is not proven until the joint route runs.

## Phases

### Phase 0 - freeze the repro (one hour)

- Keep /mnt/h3/draft.bin and /mnt/h3/draft.pcm read-only fixtures; every stage-2
  A/B consumes them unchanged.
- Record commands, timings, hashes and frame extractions under
  /mnt/h3/quality/<date>-<tag>/.
- Commit the current branch state so every later result names a commit.
- Gate: rerunning stage 2 with the same seed reproduces the feline frame. If it
  does not, record the nondeterminism before drawing conclusions.

### Phase 1 - configuration parity (half day)

1. Add --lora-strength (default 0.8, the reference value) and pass it to
   Lora::open; keep 1.0 reachable for the A/B.
2. Draft again with the reference's stage-1 attention: H3_DIT_SOL=0.9 with the
   FastH3 artifact, 64-token blocks.
3. Read the engine's 4-update schedule and compare it numerically with the
   reference's video_sigmas and audio_sigmas. If they differ, add an explicit
   schedule argument rather than approximating.
4. Leave the BF16/FP8 question for Phase 3; do not change two numeric causes at
   once.

Gate: the refined frame at 1.0 s is a fox, at 0.8 LoRA strength, on the same
draft and seed; side-by-side against draft-1s.png.

### Phase 2 - isolate the stage-2 cause (half day)

Run the fixed draft through four combinations at seed 42, same cache:

| LoRA | Attention | Output |
|---|---|---|
| 0.8 | Sol | refined-lora08-sol.mp4 |
| 0.8 | dense | refined-lora08-dense.mp4 |
| 1.0 | Sol | refined-lora10-sol.mp4 (the current failure) |
| 1.0 | dense | refined-lora10-dense.mp4 |

Record per run: step seconds, frame at 1.0 s, identity verdict, and any other
visible change (detail, motion, colours). Keep the ledger in the run directory.

Gate: at least one combination preserves the fox. That combination becomes the
default, and the others stay behind flags for the record.

Note: dense attention is not a free baseline here. With LoRA 1.0 it measured
126.0 s and 127.5 s for the first two steps, against 41.3, 37.1 and 33.5 s with
Sol, so run the LoRA 0.8 dense case last if time matters.

### Phase 3 - localize to stage 1 or stage 2 (one day)

The decisive test is cross-feeding, not more visual guessing:

1. If the reference's own stage-1 payload exists (h3_normalized + audio, the
   format latent_transfer.py accepts), run our h3-sol-rs on that payload. A fox
   means our stage-2 path is faithful and the fault is our draft latent.
   A cat means the fault is in stage 2.
2. If no payload exists, run the reference stage 2 on our draft.bin on the same
   box (the Sol-H3-Spark runtime is present at /data/Sol-h3-Spark). Same
   conclusion, opposite direction.
3. Only if both cross-feeds agree does the BF16-versus-W8A8 stage-1 difference
   become the next numeric candidate to measure.

Gate: the fault is attributed to one stage with a captured artifact from the
other side.

### Phase 4 - numeric dumps at the attributed stage (one day)

- Stage 1 attributed: dump the draft latent and compare per-channel statistics
  and a few patch values against the reference's captured latent, then decide
  between VSA, sigma schedule and FP8.
- Stage 2 attributed: dump the upscaler output, the adapter output and the
  step-0 latent grid from our run, and the reference's own dumps for the same
  input; use the gates' dump modes. Compare per stage, not end to end: a match
  at the adapter and a divergence after step 0 points at the sampler, a
  divergence at the adapter points at the transfer.

Gate: per-stage relative error below the BF16 noise floor that the lora-fusion
record used, with the same frame identity verdict.

### Phase 5 - fix, re-qualify, then performance (half day)

- Apply the localized fix as one commit with its dump evidence.
- Re-run the fixed prompt at seed 42 and at a second seed; record frames and
  timings in the same ledger.
- Only then return to the stage-2 speed gap (111.9 s here against the reference
  transcript's 24.71 s for three updates).

## Rules for the campaign

- One changed cause per A/B; the draft fixture never changes within a phase.
- No speed tuning while fidelity is open; a faster wrong picture is not a gain.
- Every run keeps its log, its guard log, and its 1.0 s frame under
  /mnt/h3/quality/.
- Flags default to the behaviour that is proven; unproven variants stay opt-in.

## Rollback

Each phase is a commit; the stage-2 CLI keeps the current path as the default
until a variant wins. The draft fixtures are read-only, so every A/B can be
repeated from the same bytes.
