# The decoder's readback: 1.87 s of pageable D2H, and the route it hid behind

The attribution left two items in FRAMES: 2.97 s of GPU idle between one tile and
the next, and 1.65 GB of device-to-host copies that took 1.87-1.97 s of stream
time (`vae-decoder-attribution-2026-10-29.md` section 6). This unit fixes the
second - a pageable readback that the driver stages itself - and, while measuring
it, found that the campaign's FRAMES figures have all been the *arm's*: the
decoder's default GEMM mode is the reference's F32 route, which is 3.6x slower
than the BF16 arm and 4.5x slower than `bf16i` on this job.

## 1. The mechanism

`h3d_read` copied straight into the caller's buffer:

```c
cudaMemcpyAsync(host, tensor->data, bytes, cudaMemcpyDeviceToHost, context->stream);
cudaStreamSynchronize(context->stream);
```

For a destination that is not pinned, that `cudaMemcpyAsync` is not an
asynchronous DMA: the driver stages the data through its own pinned buffer, and
the call does not return until the copy has landed (CUDA Runtime API, "API
Synchronization behavior", rule 1). The port already knew the pattern - the
*upload* side has staged through an 8 MiB pinned buffer since the decoder's
first unit, and its own comment records why: 126,551 MB in 2,210 copies took
105.4 s at 1.20 GB/s before the staging, "against 4.44 s for the same bytes out
of pinned memory" (`native/h3/h3_dit_gpu.cu:281-289`). The read side never got
the same treatment, so every one of the 54 tile readbacks paid 0.88 GB/s.

## 2. The change

Three parts, all schedule-only:

- **`h3d_read_staged`** (`h3_dit_gpu.cu:371-420`), the mirror of
  `h3d_upload_range`: DMA into the pinned staging buffer in bounded pieces, wait
  for that DMA to land, `memcpy` out. `h3d_read` calls it and keeps its final
  `cudaStreamSynchronize`, so the function's contract ("waiting for the stream")
  and the byte content are unchanged. The staging buffer and its event are now
  allocated by one helper both directions share (`h3d_stage_ready`).
- **`Gpu::read_f32`** (`crates/ds4-sys/src/h3_dit.rs`): the same copy into an
  `f32` destination with no conversion. The host and the device are both
  little-endian and every bit pattern is a valid `f32`, so the per-element
  `f32::from_le_bytes` pass the decoder ran over 7.8 million words per tile was
  pure cost.
- **The rows buffer is the caller's** (`vae_video.rs`): `run_tile` takes
  `out: &mut Vec<f32>` and sizes it once instead of building a fresh 31 MB
  `Vec<u8>` and a second 31 MB `Vec<f32>` per tile, 54 times per render.
- **`H3_VAE_TIMING=1`** prints the per-chunk split the measurement needed:
  extract and normalize, the graph call with its readback wait, the unpack, and
  the stitch.

## 3. Measured, on the route that matters

Job A (768x768x107, seed 42, four steps, `H3_DIT_SOL=0.9 H3_DIT_GEMM=fp8`,
NVFP4 text encoder), one binary built from `6e0696b6` against the pre-change
binary built from `14664786`, interleaved, `H3_VAE_GEMM=bf16`:

| rep | before | after | delta |
|---|---:|---:|---:|
| r1 | 30.247 s | 28.370 s | -1.877 |
| r2 | 30.461 s | 28.581 s | -1.880 |
| mean | **30.354** | **28.476** | **-1.878 s, -6.2%** |

All four renders reproduce the campaign's golden digest for this arm,
`4611c9478c00ee28c159e04d9a057ee19bb08fe827aad7d79ef2f855392532aa`, and 0 of 107
frames differ in every pairing (before against after, and each arm against its
own repeat; PSNR `inf` on every mux pair). The change is byte-identical, which is
what a schedule change has to be.

The after arm's own split (6 chunks x 9 tiles): `extract 0.001s graph 4.36s
unpack 0.079s stitch 0.053s` per chunk. The readback now costs 0.029 s in the
capture's copy table instead of 1.87 s.

## 4. Measured, on the default route

The same A/B with no `H3_VAE_GEMM` at all - the decoder's actual default - and
the same binary pair:

| rep | before | after | delta |
|---|---:|---:|---:|
| r1 | 110.610 s | 108.615 s | -1.995 |
| r2 | 110.134 s | 108.191 s | -1.943 |
| mean | **110.372** | **108.403** | **-1.969 s, -1.8%** |

The same 1.65 GB is recovered, so the absolute saving is the same; it is 1.8% of
this phase rather than 6.2% because the phase is 3.6x longer. Frames and digests
are identical again (`893eb042f2f7...`, 0 of 107 frames differing).

The capture's copy table, which is the direct evidence for the mechanism:

| direction | copies | bytes | time |
|---|---:|---:|---:|
| DtoH (after) | 227 | 1,696,270,976 | **0.029 s** |
| DtoH (before, the arm's capture) | 53 | 1.65 GB | **1.867 s** |
| HtoD (after) | 11,095 | 126,600,037,408 | 2.160 s (58.6 GB/s) |
| DtoD (peer register copies) | 187 | 1,295,529,984 | 0.010 s |

The 227 copies are the decoder's four 8 MiB pieces per tile (216) plus the DiT's
and the text encoder's readbacks; the *bytes* are mode-independent, which is why
the before figure (taken on the bf16 arm) and the after figure (taken on the
default route) are comparable. `h3d_upload_range` gets the same treatment for
free: the weight uploads in the same capture run at 58.6 GB/s.

## 5. The correction this measurement produced

**The decoder's GEMM default is the reference's F32 route, not BF16.**
`h3d_vae_options_from_env` starts from `H3D_VAE_GEMM_FP32` and says so -
"Unset means the reference path, so an arm is always a comparison against the
same binary's own default" (`h3_dit_gpu.cu:90-97`) - and it has been that since
the options landed (`1a715fca`). `bf16`, `tf32` and `bf16i` are arms.

Every FRAMES figure this campaign has recorded was measured with an arm set:
the decoder's own record compares `bf16` against `bf16i` (30.2 against 24.4 s)
and the attribution captured the `bf16` arm (its GEMMs are TF32-class, which is
what `bf16` resolves to). Against the default route the same three options
measure:

| route | FRAMES, this job | against the default |
|---|---:|---:|
| unset (F32, the reference) | 110.4 s | - |
| `H3_VAE_GEMM=bf16` | 30.4 s | **3.6x** |
| `H3_VAE_GEMM=bf16i` | ~24.4 s (the arm's record, same job) | **4.5x** |

So the operator's decision 1 is not a 5.85 s choice between two arms; it is a
~80 s choice between the reference's arithmetic and an arm, on this job alone.
That is only partly this unit's finding - the numbers were all measured before,
the missing piece was the baseline - and it changes what the decision costs.

## 6. What the host time actually was

The attribution read the 2.97 s of GPU idle between tiles as `extract_latent_tile`,
`prepare_input_rows`, `unpack_frames` and `stitch_tiles` plus the readback wait.
The measurement now splits it, and the Rust work is small: per chunk, extract
0.001 s, unpack 0.079 s, stitch 0.053 s - **0.81 s per render** over six chunks,
0.7% of even the default route's phase. So:

    2.97 s of GPU idle  =  1.87 s of DtoH activity (a copy is not a kernel, so
                           the earlier window-minus-kernels split counted it as
                           idle)  +  0.81 s of host Rust work  +  ~0.3 s of
                           launch and allocator gaps

After this change the phase still carries about 1.1 s of that idle time, and the
readback's 1.87 s is gone. The same arithmetic explains why the reused buffer
(part three of the change) is worth only the ~0.13 s the two deltas differ by:
a 31 MB allocation and a vectorized 7.8-million-word conversion are milliseconds,
not seconds, per tile. It stays because it is free, not because it paid.

## 7. What this does not establish

- **One geometry, one job.** 768x768x107, six chunks, nine tiles per chunk. The
  copy total scales with the projected rows, so the saving scales with it; the
  per-chunk split does not change shape.
- **Nothing about the F32 GEMM route's own levers.** The default route's 106.4 s
  of graph time is mostly its GEMMs, and its bench price (88.98 ms/block against
  `bf16`'s 11.24) is the arm's own record. Whether the F32 route is worth
  optimizing at all is the operator's quality decision, not a schedule question.
- **The ~1.1 s of remaining idle was not chased.** It is launch and allocator
  overhead between tiles; nothing in this unit's measurements says how it splits.
- **The conversion removal is unmeasured in isolation.** It is inside the delta,
  not separately priced; the estimate above comes from the arithmetic, not from a
  variant that only changed that.

## 8. Reproducing

`~/tile.sh` (default route, four renders, the frame comparison, the digest, and
the capture with its copy table) and `~/tile2.sh` (the same on the `bf16` arm);
both write their renders, frames and logs into the Spark's home. The pre-change
binary is `~/h3-rs-tilebefore`, the post-change one is the tree's own `h3-rs`.
`H3_VAE_TIMING=1` prints the per-chunk split; without it the phase prints exactly
what it printed before.
