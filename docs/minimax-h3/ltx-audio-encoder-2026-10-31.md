# LTX audio encoder gate and frozen stage-2 state (in progress)

Date: 2026-10-31. Branch: `feat/sol-h3`, parent `b19090c6`. This is a
phase-A execution receipt and a phase-B build receipt, not an end-to-end
stage-2 acceptance.

## Reference contract

The release's `runtime/stage2.py:78-99` encodes the stage-1 PCM with the LTX
audio VAE, conforms its time axis to the refiner shape, then holds the clean
audio latent with `frozen=True` and a zero denoise mask. Pinned LTX-2
`utils/helpers.py:466-489` sets a frozen modality's sigma to zero; pinned
`transformer_args.py:388-410` uses a modality's own timesteps for cross
AdaLN and the other modality's sigma for the cross gate. The audio prompt
comes from the existing `prompt_cache.audio` BF16 `[1, 1024, 2048]`.

The Rust port's `ltx_audio_vae.rs` computes stereo resampling, 64-bin log-mel,
the causal encoder and per-channel normalization. Its 130-frame latent is
cropped to 126 frames, converted to BF16 and passed with the cached audio
prompt to every refiner update. The video-only ablation remains available via
`h3-sol-rs --no-audio-state`.

## Phase A: measured encoder gate

On the Spark, the gate built with `make tests/h3_ltx_audio_gate`. The successful
command, run from `~/ds4-sol-h3`, was:

```sh
./tests/h3_ltx_audio_gate \
  /mnt/h3-artifacts/LTX-2.5/ltx-2.5-audio-vae-bf16.safetensors \
  /mnt/h3/recipe.pcm \
  --golden-latent scratch/agold/golden_latent.f32 \
  --golden-mel scratch/agold/golden_mel.f32 --relative 1e-4
```

The gate exited 0. Its input was 165600 stereo samples at 32000 Hz, yielding
518 x 64 x 2 log-mel values in 35.395 ms. The encoder loaded 46 tensors,
42666528 bytes, and passed through 128 x 518 x 64, 128 x 259 x 32,
256 x 130 x 16 and 512 x 130 x 16 stages. Its 8 x 130 x 16 output took
1.065 s. Against the F32 goldens, log-mel rel-RMS was 6.211e-7 with maximum
absolute error 2.527e-5; encoder latent rel-RMS was 4.497e-5 with maximum
absolute error 5.982e-3. The conformed 8 x 126 x 16 state had mean -0.60722
and standard deviation 1.29170 as printed by the gate.

This comparison proves the port against the F32 golden, not bit identity with
the release's BF16 encoder. The port widens BF16 checkpoint weights to F32,
computes the convolution and intermediate activations in F32, then rounds to
BF16 at the refiner boundary. The release constructs its audio encoder at
BF16 (`runtime/stage2_ops/models.py:110-115`). The magnitude of the resulting
BF16-vs-F32 stage-2 difference has not been measured.

## Phase B: built, live gate pending

`ltx_stage2.rs` now forwards an optional clean audio latent and prompt every
step, setting audio sigma to zero independently of video sigma.
`ltx_refiner.rs` applies that audio sigma to audio AdaLN, prompt and the A2V
gate, while the V2A gate still receives video sigma. `h3-sol-rs` requires
`--audio-vae` for the full route and prints the conformed state statistics;
`--no-audio-state` keeps the previous video-only path.

The Spark's `make h3-sol-rs` exited 0 (`~/ds4-test/phase-b-build.log`).
`./h3-sol-rs --help` exited 0 and lists both flags. With the other required
flags supplied and no `--audio-vae`, the CLI exited 2 with
`h3-sol-rs: missing --audio-vae` before creating a device. The Spark's
`--no-audio-state` with the same dummy inputs passed option validation and
stopped at the expected empty latent-file size check before device creation.
On the workstation, `cargo check -p ds4-cli --bin h3-sol-rs --features native`,
the refiner gate example check and the audio-VAE/refiner unit tests passed;
`git diff --check`
reported no errors. No full refiner run or audio-on/off A/B has been performed
with these changes. Picture quality, step times and resident memory are
unverified.

## Execution blocker and resume condition

Read-only inspection on the Spark this session found
`systemctl show nvidia-clocks.service` reporting `LoadState=not-found` and
`ActiveState=inactive`; `/etc/systemd/system/nvidia-clocks.service` is absent.
The host was `gx10-e1d2`, boot ID
`122818b0-ddaf-4426-8f80-43a43ad2c061`; the idle SM clock was 208 MHz
at 0% utilization. There is no observed busy-clock measurement for this boot.
The policy recorded in `docs/gb10-hardware.md` requires 300-2200 MHz to
reduce trip risk and stabilize timings. The reported application clock and
maximum supported clock do not verify a configured 2200 MHz cap. No service
or clock setting was modified, and no full-model job was started after this
discovery. The operator needs to restore or confirm the policy before the
phase-B full-stage A/B and phase-C recipe run; verify service state and busy
clocks read-only first. The `--no-audio-state` switch is the behavioral
rollback if the new conditioning branch fails its gate.

## Shipping decision

The operator explicitly waived the AI QA-tester for this pending push and
chose to ship with the full-stage audio-on/off A/B gate still pending. This
CUDA unit is not lightweight; the waiver is a conscious exception to the
project's normal QA requirement, not a completed QA review. The
exception covers the three earlier local commits `56242e84`, `410597a2`,
`b19090c6` and the audio-state changes in this report. It is not a QA PASS:
there is no `qa-evidence/qa-report.md`, no `tests/qa-gate.sh`, and no full-stage
quality, step-time or resident-memory result for the audio state. The two
pre-existing untracked ELF binaries are explicitly excluded from the push.
Dependency scan: OSV batch query on 2026-10-31 scanned 140 packages in
`Cargo.lock` with zero findings; this is a dated advisory snapshot, not a
proof against future findings. Once the clock policy is restored, run the
same-seed audio-on/off comparison before treating the new branch as qualified.
