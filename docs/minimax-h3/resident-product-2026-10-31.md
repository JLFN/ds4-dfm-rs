# The product route in one process: a job list on the resident engine (2026-10-31)

Handoff section 3 of the H3 campaign. The product route is two processes
(`h3-rs --latent-only`, then `h3-sol-rs --h3-video upscaled`), so every clip
pays the prompt encode, the 62 GiB DiT read and the upscaler load again. This
unit puts the whole route in one process behind one engine, states the request
list as a file, and measures what the second clip costs.

Spark `gx10-e1d2`, sm_121a, CUDA 13.0, `nvidia-clocks.service` absent
(persistence Enabled), so the numbers below carry the host's boost clocks and
nothing else.

## 1. What was added

- **The job list** (`crates/ds4-core/src/minimax_h3/jobs.rs`): one request per
  line, seven TAB-separated fields — prompt, seed, steps, width, height,
  frames, output — with blank and `#` lines skipped. A tab and not a comma
  because a prompt carries commas, quotes and spaces, and because a list has to
  be writable from a shell without an escape language. The list is this port's
  own surface: the reference CLI has no batch mode (`h3_cli.c`), so the only
  rule it inherits is the engine's, that one request runs at a time.
- **`--jobs FILE`** (`h3-rs`): reads the list, validates every canvas and frame
  count before the first weight is read, then renders the requests through one
  `EngineHandle`. Per-request flags on the command line are refused rather than
  dropped: the line owns the request.
- **`--upscaler FILE`**: with `--jobs`, every request takes the release's route
  — the line's canvas is the draft's, the latent is doubled in H3 space by the
  upscaler, and the doubled canvas is what the video VAE decodes
  (`runtime/stage2.py:58-76`). Without `--jobs` the flag is refused, because
  the one-shot path has no upscale stage.
- **The engine's upscale stage** (`engine.rs`): `RenderRequest.upscale` names
  the checkpoint; the geometry follows from the request's own canvas and frame
  count, so a request cannot name a shape the upscaler was not built for. The
  upscaler is held under (checkpoint, geometry), the same rule the prepared DiT
  follows, and its load and forward are reported separately.
- **`Upscaler::upscale`** (`ltx_adapter.rs`): the cross-context handoff in one
  call — F32 latent in, rounded at the BF16 boundary, doubled, F32 out. The
  rounding is the one the two-process route writes with `--save-latent`, so
  both routes feed the network the same bytes.
- **The decoder's arms are options now** (`RenderRequest.vae_attention`,
  `vae_gemm`): `None` leaves the context on `H3_VAE_ATTENTION`/`H3_VAE_GEMM`'s
  word instead of forcing an arm. Without that, a job list could not run the
  `bf16i` arm this host measured at 48.2-48.7 s (`vae-decode-2026-10-31.md`).

## 2. The defect the route exposed

A second upscale call in one process returned different bytes than the first,
with the same input, the same weights and the same arm.

The chunked upscaler's accumulator is a sum: `h3b_accumulate_kernel` does
`accumulator[...] += segment * weight` (`native/h3/h3_dit_gpu.cu:8107`), and
the scratch that holds it was allocated once per load and never cleared.
`cudaMalloc` returns uninitialized memory, so the first call read whatever the
driver handed back — on this host zeroed pages, which is why the two-process
route never showed it — and the second call added its segments to the first's.

Evidence, one engine-gate run each at 512x512x22 (draft 256x256x22,
`bf16`/`tensor`), same seed and same latent (`video=e0100bb6f97a5f14`,
`audio=8670924cbd80af9a` in both runs):

| run | request 1 frames | request 2 frames |
|---|---|---|
| before `h3b_zero` | `210875f83aef6008` | `b80b3de3b5fb5bc0` |
| after `h3b_zero` | `5cff7cecdf260655` | `5cff7cecdf260655` |

The first call's value moved too, so it was reading device memory it had not
written. The fix is `h3b_zero` — a `cudaMemsetAsync` on the context's stream
(`native/h3/h3_dit_gpu.cu`), bound as `Gpu::zero` — applied to the accumulator
at the top of every `Upscaler::forward`. `h3b_accumulate` is the only kernel
that accumulates into a persistent tensor in this path; the adapter has no
equivalent.

The single-call path was not necessarily wrong, only *unproven*: it depended on
the driver returning zeroed pages, which is why the adapter gate and this host's
upscale artifacts have been reproducible.

## 3. The acceptance

`tools/h3-jobs-accept.sh` (`H3_ARTIFACTS` required, nothing hardcoded) runs the
same request three ways at one arm:

1. the list twice, in one process (`h3-rs --jobs scratch/jobs.txt --upscaler`);
2. the two-process route the list replaces: `h3-rs --latent-only`, then
   `h3-sol-rs --h3-video upscaled --hash` (the `--hash` flag is new; it prints
   the decoded frames' FNV, the engine's own `frames_hash`).

Command (Spark, `H3_VAE_GEMM=bf16i H3_VAE_ATTENTION=tensor`):

```sh
H3_ARTIFACTS=/mnt/h3-artifacts H3_VAE_GEMM=bf16i H3_VAE_ATTENTION=tensor \
  bash tools/h3-jobs-accept.sh
```

One request: the release's draft prompt at 672x384x124, 4 steps, seed 42,
doubled to 1344x768x124. Phase table, seconds:

| phase | job 1 | job 2 |
|---|---|---|
| prompt encode (NVFP4) | 12.015 | 0.000 (hit) |
| DiT load | 7.074 | 0.000 (hit) |
| denoise, 4 steps | 21.352 | 21.705 |
| audio VAE | 2.953 | 3.151 |
| upscaler load | 0.371 | 0.000 (hit) |
| upscale forward | 2.214 | 2.241 |
| video VAE decode | 54.317 | 51.049 |
| mux | 1.039 | 0.953 |
| **wall** | **100.158** | **78.273** |

Two clips in one process: `JOBS wall=178.431s` (stage wall 179.424 s including
the guard wrapper). The two-process route on the same request and arm:

| stage | wall |
|---|---|
| stage 1 (`--latent-only`) | 56.143 |
| stage 2 (upscale + H3 decode) | 54.524 |
| **one clip** | **110.667** |

So the second clip costs **78.3 s instead of 110.7 s** — 32.4 s less, 29% — and
two clips cost 179.4 s instead of 221.3 s. The equality the route has to keep,
all three renders at the same arm:

```text
frames job1=395e1b371570e399 job2=395e1b371570e399 two-process=395e1b371570e399
RESULT PASS resident product route
```

The latent and audio hashes agree in the same way (`video_fnv=199fad94f9df0ea7`,
`audio_fnv=ea7070c8d4cddbeb` in both jobs).

Four runs of this acceptance on the same tree (the last four build/rebuild
steps of the unit) printed the same three hashes; the table is the last one,
measured with `h3-rs` and `h3-sol-rs` built from `56242e84` and `410597a2`.

Job 1's own loads are page-cache dependent: a run on cold pages measured
`text=15.639s load=12.788s wall=111.346s` for the same request, and stage 1 on
warm pages still spends 15.1 s on the prompt and 15.5 s on the DiT read. That
is the work the second clip no longer repeats.

## 4. What it does not establish

- **The video decoder is not resident.** Each job loads the 9.8 GB decoder
  inside `video`; the second clip's 51.1 s includes that read. Holding it beside
  the resident DiT is a separate memory decision, not attempted here.
- **The decode's cost is not explained.** The isolated `bf16i` arm measured
  48.2-48.7 s (`vae-decode-2026-10-31.md`); the second clip decoded in 51.1 s
  and the two-process stage 2 in 51.3 s, while job 1's decode took 57.5 s.
  Job 1's extra ~6 s is unexamined — cold page cache for the decoder weights,
  or contention with the resident DiT — and this unit does not attribute it.
- **One request, one shape, one seed, one arm.** A second prompt or a second
  canvas would miss the embedding and the prepared DiT by design; the list's
  hit rules are the engine's `PreparedKey`, unchanged.
- **The list carries text only.** No references, anchors, first/last frames or
  media; the engine's own fields exist, the file format does not state them.
- **No clock policy.** `nvidia-clocks.service` is absent on this host, so
  absolute times move with the boost clocks; the two-process and resident runs
  were interleaved within one minute of each other, which is what makes their
  comparison usable.
- **The MP4 bytes are not compared**, only the decoded frames' hash; ffmpeg
  muxing remains its own surface.
