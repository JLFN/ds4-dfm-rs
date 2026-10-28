# The reference's NVFP4-AWQ prompt encoder

Job A throughout: 768x768, 107 frames, seed 42, the prompt "a red fox walking
through deep snow at sunrise, cinematic", one binary, `sm_121a`, 2.4-2.5 GHz
(the host has no clock policy). Commit `bd572b3c`, and the layer-count fix and
its re-run below.

The unit adds the reference's own stage-1 encoder as a second weight source for
the prompt path. It is a read-time change: the layer walk, the kernels and the
numerics of the forward are the same code on both arms, which is what makes the
two directly comparable and the BF16 arm bit-reproducible.

## 1. What was added

- `crates/ds4-core/src/minimax_h3/nvfp4.rs`: the artifact's reader and decoder.
  It resolves the same eleven roles per layer plus the embedding, checks dtype,
  shape and the `comfy_quant` marker before a byte is read, and hands the
  forward BF16 bytes, exactly what the BF16 shard path hands it.
- `text_forward::Encoder` (`Bf16` default, `Quantized(path)`), selected by
  `--text-nvfp4` on `h3-rs`, `--nvfp4` on `h3_text_gate`, and `H3_TEXT_NVFP4`
  on the server. `encode_with_layers` carries the layer count across both
  stores, so the E4 bisection means the same thing on either side.
- The engine's embedding cache key gains the encoder: two checkpoints answer one
  prompt with two different embeddings.
- `safetensors::Dtype` gains `F8E4M3`. The reference's enum has no FP8 because
  it reads BF16 checkpoints only; the artifact names its block scales
  `F8_E4M3`.

The artifact is the one `configs/checkpoints.json:65` names,
`text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`,
15,687,142,551 B, sha256 `35a88d51...`: 2054 tensors, layers 0..49, no final
norm and no `lm_head`, because the prompt path reads neither.

## 2. The conventions, and what decided them

Sources, read first-hand: ComfyUI v0.30.0 (`b1693ecb`, the revision
`configs/dependencies.json:40-47` pins as `comfy_root`) `comfy/ops.py:1178-1183`
and `:1350-1352`, `comfy/text_encoders/minimax.py:1-19`; comfy-kitchen v0.2.26
(`255a4387`, `requirements.txt:25`) `backends/eager/quantization.py:118-222` and
`float_utils.py` (`pack_uint4`, `to_blocked`, `from_blocked`).

Two of them are invisible in the safetensors header, and both were measured
against the BF16 weights on the Spark (`~/nvfp4-verify.py`):

| projection | reading | corr(f, s) | rel. residual | median \|f/s - 1\| |
|---|---|---:|---:|---:|
| q_proj | plain per-16 | 0.859 | 0.400 | 0.144 |
| q_proj | unswizzled | **0.981** | **0.097** | **0.009** |
| gate_proj | plain per-16 | 0.048 | 0.563 | 0.311 |
| gate_proj | unswizzled | 0.085 | **0.104** | **0.010** |
| up_proj | plain per-16 | 0.057 | 0.528 | 0.267 |
| up_proj | unswizzled | 0.091 | **0.103** | **0.010** |

`f` is the per-column least-squares factor between the dequantized weights and
the BF16 source, `s` the fold's per-input-channel factor. The first convention
is the nibble order: even columns are the high nibble (`pack_uint4`,
`hi_first=True`). The second is that `weight_scale` is stored in cuBLAS's 128x4
tiled *blocked* layout, which keeps the shape and permutes the contents:
`quantize_nvfp4` ends in `to_blocked` and `dequantize_nvfp4` starts with
`from_blocked`. Read plainly, the scales leave 0.40-0.56 relative residual; the
shape is identical either way, so nothing in the header reveals it. Unswizzled,
the residual is 0.10, which is E2M1's own block-16 error, and the per-column
factor matches the fold to 1% on the median column.

`gate_proj`'s low correlation is the instrument, not the decode: its fold
reference is `post_attention_layernorm`'s ratio, whose spread is wide and whose
small entries dominate the correlation while the median error stays at 1%.
The fold itself is confirmed on all three projections by the median column, and
on `q_proj` by the correlation.

The fold's activation side lives in two places, and the artifact carries exactly
those: the preceding layernorm's weight for the five norm-fed projections, and
`pre_quant_scale` for `down_proj` and `o_proj`, the two no layernorm feeds
(`comfy/ops.py:1350-1352` multiplies the activation by it). The reader requires
that split, so a mis-mapped tensor fails at open.

**Deviation, named.** The reference multiplies the BF16 activation by
`pre_quant_scale` and its eager dequantizer rounds in BF16. This source folds
`pre_quant_scale` into the weight columns and accumulates in F32, rounding once
to BF16, because the forward it feeds is the existing BF16 one. The gate below
is the check on that choice.

## 3. The gate

One binary, one job. `TEXT` is the phase the change touches; `LOAD` is the
host's page-cache state and is not comparable across runs (see
`lora-merge-2026-10-28.md` §5).

### The embedding

`h3_text_gate`, eleven tokens, fifty layers:

| arm | FNV over the BF16 embedding | seconds |
|---|---|---:|
| BF16 shards (recorded value) | `6dc82144cbf37691` | 27.7-28.7 |
| NVFP4-AWQ | `d97d6ce16b160dbf` | **15.2-15.5** |

The BF16 arm reproduces the campaign's recorded embedding exactly, so the gate
itself is unchanged. The quantized arm differs, as a lossy 4-bit artifact must;
the edges agree to the lead digits
(`400c.c007.bf40.c089...` against `400c.c000.bf55.c087...`), and §3's renders are
the acceptance test, not this hash.

A first pass of this table also carried `--layers 1` rows, which were wrong:
`encode_with` had dropped the layer count, so both ran fifty layers. Fixed
before the record, and the re-run in §4.

### The render

The recorded fast-path arm is job A on the derived checkpoint, four steps, the
sparse attention at 0.9 and the fast decoder; the two BF16 runs are the
repeated default arm the campaign's noise rule asks for.

| arm | TEXT | LOAD | DENOISE | FRAMES | wall | MP4 sha256 |
|---|---:|---:|---:|---:|---:|---|
| BF16 a | 27.427 | 52.316 | 50.082 | 35.718 | 171 s | `093d6fcd...` |
| NVFP4 | **15.462** | 14.632 | 50.679 | 39.600 | 124 s | `5b5e735c...` |
| BF16 b | 28.200 | 12.026 | 50.341 | 35.149 | 130 s | `093d6fcd...` |

- The two BF16 runs produced the **same MP4**, so the render floor for this arm
  is zero and the quantized arm's different file is the change's effect.
- `TEXT` falls from 27.4-28.2 s to **15.5 s**: 12.0-12.7 s off the phase.
- The wall difference (130 against 124 s) is smaller than the phase's, because
  `FRAMES` ran 4.5 s slower on the quantized arm; both are inside the host's
  run-to-run spread. `TEXT` is the measurement; the wall is not.
- `5fe31c57...` is the *dense default* arm of the merge record, not this one, so
  it is not this arm's expected hash. The dense arm is re-run in §4 as the
  BF16-path regression check.

### Quality

Detail energy, the whole-render mean of `|frame - luma-4 blur|`, the campaign's
instrument (`~/detail.sh`):

| render | detail energy |
|---|---:|
| BF16 a | 3.1893 |
| BF16 b | 3.1893 |
| NVFP4 | 3.4338 |

For scale, the same instrument on the recorded renders: the fast four-step
sparse arm 3.1893, the twenty-step released arm 3.2824, the dense four-step arm
4.5626. The quantized arm sits 7.7% above its own BF16 arm and above the
twenty-step reference, in the band the merge record calls finished and the dense
arm over-sharpened. The mid frames were looked at
(`nvfp4-frame-bf16-a-mid.png`, `nvfp4-frame-nvfp4-mid.png`): same framing, same
pose, same light, no artefact, the quantized frame marginally harder in branch
and fur contrast. That is a small, bounded difference on one job, not a quality
verdict.

## 4. The BF16 path, and where the phase's time goes

Two follow-ups, both run after the layer-count fix:

### The BF16 path is bit-unchanged

The recorded dense-default arm (`lora-merge-2026-10-28.md` runs 3 and 6: the
derived checkpoint, four steps, default arms, dense attention) reproduces its
recorded MP4 on this binary:
`5fe31c57c06bc3848bd210e5709088abd18a84e605b9c1e13675a6dc3edab736`, with
`DENOISE` 79.403 s against the record's 79.391 and 81.147, and `FRAMES` 198.091 s
against 200.985 and 202.391. The second store is reached only through the
switch, and the BF16 path the campaign measured is untouched.

### The localizer

`--layers 1` on either arm, eight lanes, layers 1: the embedding and the first
block only, so a decode fault is visible at the first output rather than
accumulated over fifty layers.

| arm | FNV | first eight words | seconds |
|---|---|---|---:|
| BF16 | `2b45149d9b286329` | `bdd3.bf12.3dd1.be1c.bd62.3e5e.3dc0.be8c` | 1.504 |
| NVFP4 | `406663740240f362` | `bde3.bf0b.3dc1.be16.bd7a.3e35.3db4.be8b` | 2.882 |

The two agree to a percent or two on every leading word, so the quantized
weights decode to the same function and the difference is the 4-bit error, not
a layout mistake: a wrong scale layout or nibble order produces garbage at the
first block, not a 1% shift.

The layer-1 pair is also the one place the quantized arm is *slower*, 2.882 s
against 1.504 s. That is the embedding: 778M `int8` values decoded on the
calling thread, where the BF16 arm copies a mapped span. It is 1.4 s of every
quantized encode and it is the one part of the decode the ring's lanes do not
share. Both arms were measured with the encoder trees out of page cache.

### Where the phase's time goes

The ring's lanes do the decode on the host, so the phase scales with
`H3_QWEN_PREFETCH` while the decode is the cost and saturates at the device's
rate when it is not. Fifty layers, one run each, same tree state:

| arm | 1 lane | 4 lanes | 8 lanes | bytes read |
|---|---:|---:|---:|---:|
| NVFP4-AWQ | 25.537 s | 10.953 s | 10.781 s | 15.69 GB |
| BF16 | 28.733 s | - | 53.014 s | 52.4 GB |

The quantized arm is 2.4x slower on one lane, so the decode is real work and the
ring is what hides it; at four lanes it has saturated at 10.8 s, against a 8.7 s
floor for 15.69 GB at the disk's measured 1.8 GB/s. The BF16 arm behaves the
other way round - 28.7 s on one lane against 53.0 s on eight - which is the
external SSD, not the lanes: eight concurrent shard streams on `/dev/sda` read
slower than one. That is why the two arms' `TEXT` may only be compared from one
cache and concurrency regime, and why §3's paired render arms, taken minutes
apart in one session, are the number to quote and these standalone sweeps are
the attribution.

Note the spread this puts on the quantized phase: 10.8 s warm and saturated,
15.5 s in the paired render, both the same binary and artifact. Host cache
state, not the decoder, is the larger term.

## 5. What this does not establish

- **The read floor.** §4's sweep puts the quantized phase at 10.8 s saturated
  against a 8.7 s floor for 15.69 GB at the disk's measured 1.8 GB/s, so on four
  or more lanes the host decode is already hidden by the read and a device-side
  decoder would recover at most about 2 s. The 1.4 s the embedding costs on the
  calling thread is the part that is *not* hidden, and it is the cheaper target
  of the two.
- **The vision tower and the reference paths.** `--ref-image`, `--ref-video` and
  anchors take `condition::anchor_embedding` and `condition::reference_embedding`,
  which read the BF16 directory and its tower. The switch does not reach them,
  so a reference request with `--text-nvfp4` still encodes through BF16.
- **The server's steady state.** `H3_TEXT_NVFP4` is wired into the engine's
  request, but the two-render resident number this branch still owes was not
  taken. The encoder is loaded per encode, as the reference does; whether it
  should stay resident is that measurement's question.
- **Quality beyond one job and one instrument.** Detail energy ranks finished
  pictures loosely; the frames were looked at once. A second prompt, and a
  longer clip, would test the 7.7% more than this does.
