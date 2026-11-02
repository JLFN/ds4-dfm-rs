# The decode's BF16 activation chain: 45.8 s to 27.7 s

The unit that `vae-decode-bytes-2026-11-02.md` priced. That record measured the
delivery decode's byte budget at the package's bandwidth and concluded that only
deleting a pass pays: the `bf16i` arm's narrows and widens move 689.6 MB a decode
at 251 GB/s for 10.4 s of 46.2, the F32 elementwise chain 1.91 GB at 240 GB/s for
another 8, and the BF16 K/V staging was worth 3.2 more. This is that deletion.

Arm: `H3_VAE_GEMM=bf16a` (`H3D_VAE_GEMM_BF16_CHAIN`), also `--vae-gemm bf16a`.
Nothing is default: unset runs the reference path, as before.

## 1. The numbers

Pinned shape, one binary, `tests/h3_vae_gate --latent 37,48,84 --repeat 3`, the
delivery canvas 1344x768, 124 frames, tile 320, five by three tiles, seven chunks:

| arm | run 0 | run 1 | run 2 | hash |
|---|---:|---:|---:|---|
| control `bf16i` | 46.048 | 45.436 | 45.660 | `387f816c77ee986f` |
| control `bf16i`, second session | 46.401 | 45.779 | 45.899 | `387f816c77ee986f` |
| **chain `bf16a`** | **28.202** | **27.547** | **27.677** | `3726eab1d005cb6e` |

- **Warm 45.78 s to 27.68 s: -18.1 s, 1.65x.** Cold 46.05 to 28.20.
- The control reproduced its hash from the record it was pinned in, across two
  sessions, so the F32 and `bf16i` paths are unchanged by this unit: the six
  primitives are templated on the element type and their F32 instantiations
  compile to the code they were.
- The chain is deterministic: three runs, one hash.
- `first`/`last` move from `0.287888765`/`0.349735707` to
  `0.288428068`/`0.351913393`, the rounding this arm is.

## 2. What the arm does

The block's activations become BF16, so the narrows and widens do not exist:

| buffer | `bf16i` | `bf16a` |
|---|---|---|
| `hidden` (residual stream) | F32 | BF16 |
| `norm`, `qkv`, `query`, `key`, `value`, `heads` | F32 | BF16 |
| `branch`, `ff1`, `activated` | F32 | BF16 |
| `projected` (readback) | F32 | F32 |
| block weights | BF16 | BF16 |
| casts a decode | 8 a block call, 3,780 calls | 1 a tile, 105 tiles |

Six primitives gained a BF16 instantiation rather than a twin kernel, so the
arithmetic between the load and the store is the F32 path's in both:
`h3d_rms_norm_kernel<T>`, `h3d_swiglu_kernel<T>`, `h3d_scale_add_kernel<T>`,
`h3d_video_qkv_rope_kernel<T>`, `h3d_attention_mma_kernel<TIn, TOut>` and
`h3d_layer_norm_kernel<T>` (BF16 in, F32 out: `proj_out` runs on F32 operands).

Two pieces are load-time or per-tile instead of per-projection: the register
tokens and the zero suffix row are narrowed once a decode (`TileScratch`), and the
tile's F32 embed output is narrowed once a tile. The chain's attention requires
`H3_VAE_ATTENTION=tensor` and says so at `Decoder::new` rather than failing inside
a launch; its K/V staging is a 16-byte copy where the F32 path rounds eight
values, which is the 3.2 s this record's predecessor attributed to it.

## 3. What the reference says

`/data/Rapid-H3` (`custom_nodes/rapid_h3/vae_fast.py`, `docs/kernels.md`) runs its
decode on fp16 activations end to end — `qkv` fp16 into the int8 quantizer, `h`
fp16 out of the SwiGLU epilogue, fp16 rope tables — and its stock fp16 VAE is
"≈37 s" at 124 frames against our 45.8 s at the same canvas, so the activation
dtype is where the gap starts. Its own quality anchor for an fp16-activation
chain feeding a fused w2 quantizer is **60.6 dB PSNR against the fp16 VAE** with
0.7% of w2 input codes moving by one; that is the class this arm is priced in, and
it is why the arm is judged by the clip gate rather than by a hash.

Its remaining 18 s is int8 weight quantization (`int8_convrot`), which is a
different project: it changes the weights, not just the activations, and it needs
the rotation the checkpoint does not have.

## 4. The picture

The route's five-second job line (`A red fox trots…`, seed 42, 4 steps,
1344x768x120) rendered twice with one binary, only `H3_VAE_GEMM` differing:

| clip | arm | wall | video | md5 |
|---|---|---:|---:|---|
| `outputs/chain-ctl/five-seconds.mp4` | `bf16i` | 221.1 | 47.51 | `0709ea641e82f04f35a42563d128f6c5` |
| `outputs/rapid-chain/five-seconds.mp4` | `bf16a` | **181.2** | **28.30** | `83af9a0a59301fdb6ad41f79e49f2914` |

Both took the same `video_fnv=f95d68560e363fbe`, so the diffusion output is one
latent and the pair isolates the decode. **-39.9 s of route wall, -19.2 s of it
the decode, the rest the resident-stage load the shorter decode leaves less to
hide behind.**

- **PSNR 46.90 dB average** (luma 47.9), **SSIM 0.9883 all / 0.9864 luma**.
- `tools/h3-clip-score.py` against the `bf16i` control: **PASS, no flag**.
  MUSIQ 63.40 → 63.54 (+0.14), CLIP-IQA 0.482 → 0.490 (+0.008), DOVER 0.670 →
  0.666 (-0.004), sharpness x1.00, noise x0.995, `comp_ssim` 1.000. The gate sees
  the same picture; the arm's cost is grain, not structure.
- 46.9 dB is below the reference's 60.6 dB for its fp16 chain, and the reason is
  arithmetic: this arm rounds at every primitive's store (two RMS norms, the
  rotation, the attention output, SwiGLU, two residual adds = seven a block, 252
  a tile) where theirs rounds once per GEMM output. A single-rounding variant is
  the next lever if 46.9 dB is ever judged too far.
- The audio is untouched: `audio_fnv=f20b8671f1220ae3` on both.

## 5. The pinned baseline is stale, and that is a campaign finding

`outputs/rapid/five-seconds.mp4` (`6697bd3453c160502a869520f547a075`, the clip the
five-second measurement was pinned on) does **not** reproduce on this tree with
its own job line: the same binary and the same `bf16i` arm give
`0709ea64…` and **18.99 dB** against it. The pair differs by the generation, not
by the decode — and the pipeline is deterministic across runs (`video_fnv` was
`f95d68560e363fbe` on every render this unit made, three of them). So something
between that clip and `55cb932d` moved the picture, and scoring any arm against
that file would have read the drift as the arm's cost. The first scoring run of
this unit did exactly that and returned `comp_ssim` 0.632 for a change that is
1.000 against a same-binary control.

Re-pinning the baseline set (`outputs/attn-fp8/`, `outputs/draft-attn/`,
`outputs/two-stage/`, `outputs/rapid/`) against one commit is queued in the
handoff as a correctness precondition, not a convenience.

## 6. The other gates

- `tests/h3_engine_gate`: **RESULT PASS residency**, `MEDIA identical=true`
  between its two requests, both on the `bf16` arm the gate picks itself. The
  paths this unit did not touch still reproduce.
- Determinism: three chain runs, one hash (`3726eab1d005cb6e`); `video_fnv`
  identical on every render this unit made.
- `--expect` must not be used on this arm: it is a rounding arm, so the hash
  *should* move. `387f816c77ee986f` remains the pin for `bf16i`.

## 7. Traps

- The ambient `H3_VAE_GEMM` in the shell wins: `decode-chain.sh` reads
  `${H3_VAE_GEMM:-bf16a}`, so a pane that already exported `bf16i` re-ran the
  control and looked like a no-op arm. The first chain run was that, not a result.
- `vae_gemm_name` first printed `gemm=fp32` for the chain: the name table had no
  case for mode 4. The arm ran; the log lied. Fixed in the same unit.
- A profile and a timing run are two sessions. Two decodes at once inflate every
  kernel by about 1.9x (`vae-decode-bytes-2026-11-02.md` §5).
- `make` needs `CUDA_ARCH=sm_121` or ptxas rejects `.bf16`.
- **The CLI is its own target.** `make h3-rs` builds `target/release/h3-rs` and
  copies it over `./h3-rs`; `tools/h3-rapid.sh` runs the former and falls back to
  the latter. Building only `tests/h3_vae_gate` left both stale, and the stale
  parser read `bf16a` with the old first-letter rule as plain `bf16`, so the first
  route render (233.0 s) silently measured the wrong arm and its clip is not a
  result for this unit.
