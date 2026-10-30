# Unit 12a — the LTX-2.5 video VAE decoder, ported and gated

State carrier: `ds4-dfm-rs-handoff.md` (section 3). This record is the unit
boundary. Plan `dc22d0bf` (unit 11) -> this unit.

Landed on `feat/sol-h3`:

- `crates/ds4-core/src/minimax_h3/ltx_vae.rs` — the decoder, its loader and its
  plan;
- `native/h3/h3_dit_gpu.{cu,h}` — `h3b_planar`'s `replicate_t` mode and the
  `h3v_*` section (pixel norm, denormalize, depth-to-space, unpatchify,
  to-rgb);
- `crates/ds4-sys/src/h3_dit.rs` — the four wrappers and the planar mode;
- `crates/ds4-core/examples/h3_ltx_vae_gate.rs` + `tests/h3_ltx_vae_gate`.

Source of truth: `/data/ltx2-pin/packages/ltx-core/src/ltx_core/model/video_vae/`
(pin `d1511477` = tag v1.2.0) and the artifact's own header. The release
reaches it at `runtime/stage2_ops/models.py:116-129`, which builds
`compat.VideoDecoder(...)` and refuses any type but `ConvVideoDecoder`.

## 1. What the port is

`ConvVideoDecoder.forward` (`conv_video_decoder.py:263-357`) over the whole
latent, channel-major `[channel][frame][height][width]` BF16 throughout:

| step | reference |
|---|---|
| `x * std + mean` per channel | `ops.py:74-78` |
| `conv_in` 128 -> 1024 | `:209-218` |
| nine `up_blocks`, the config's list reversed | `:222-238` |
| res group = `PixelNorm, SiLU, 3x3x3, PixelNorm, SiLU, 3x3x3, +` | `resnet.py:122-186` |
| upsampler = 3x3x3 then `(c p1 p2 p3) -> c (d p1) (h p2) (w p3)`, first output frame cut when `p1 = 2` | `sampling.py:66-124` |
| `conv_norm_out` (a `PixelNorm`), `SiLU`, `conv_out` 128 -> 48 | `:328-351` |
| `unpatchify` 4x4, 48 -> 3 | `ops.py:34-60` |
| `(x + 1) / 2` clamped, `[c f h w] -> [f h w c]` | `:497-499` |

Every convolution is `CausalConv3d` with `causal = False`
(`convolution.py:302-312`): **one replicated frame each side in time, one zero
frame each side in space**. The temporal halo is the artifact's own choice of
`spatial_padding_mode: zeros` for the spatial axes (`:299`), so the two are not
the same policy and the port could not reuse `h3b_pad3d`'s zero pad in time.
`h3b_planar` gained a `replicate_t` argument; the adapter and the ops gate pass
`0` and their behaviour is unchanged (section 5).

Rounding policy, the campaign's own: torch's elementwise ops on BF16 compute in
F32 and round once each, so the chain is reproduced op by op — `PixelNorm` is
five roundings (`normalization.py:31-38`), `un_normalize` is two, `to_rgb` is
two, and the convolutions accumulate in F32 with one rounding at the bias
(`h3b_epilogue`). A fused single-rounding form is a different function and the
probe's control shows it differing (`denormalize-control`, section 3).

## 2. The loader asserts the artifact's own contract

`LtxVae::open` reads `__metadata__.config.vae` first and refuses the shard
unless it says: `_class_name` CausalVideoAutoencoder, `causal_decoder`,
`timestep_conditioning`, `use_quant_conv` and `normalize_latent_channels` all
false, `latent_log_var: uniform`, `norm_layer: pixel_norm`,
`spatial_padding_mode: zeros`, `latent_channels` 128, `patch_size` 4,
`decoder_base_channels` 128, and a `decoder_blocks` list equal to
[`DECODER_BLOCKS`] name by name with the same layer counts and multipliers.

Every weight is then pulled by the name the plan derives, against the shape the
plan derives: `decoder.conv_in.conv`, `decoder.up_blocks.{0..8}` (res groups at
`res_blocks.{n}.conv{1,2}.conv`, upsamplers at `conv.conv`),
`decoder.conv_out.conv`, and `per_channel_statistics.{mean,std}-of-means`. The
plan's channel walk is 1024 -> 4096/512 -> 4096/512 -> 512/256 -> 512/128 ->
128 -> 48, which is the artifact's own tensor list (`up_blocks.1.conv.conv` is
`[4096, 1024, 3, 3, 3]`, `up_blocks.5` is `[512, 512, 3, 3, 3]`): a wrong block
order or a wrong reversal fails the load by shape, not by looking plausible.

Load: **86 tensors, 814,338,656 B** (the 84 decoder tensors plus the two
statistics) out of the file's 170 tensors and 1,452,269,922 B.

## 3. The gate

`tests/h3_ltx_vae_gate --probe` (sm_89, 2026-10-30): every new kernel against a
host transcription of the reference's own expression, element by element.

    PROBE denormalize values=24 differ=0 rel_rms=0.000e0 verdict=exact pass
    PROBE denormalize-control differ=4 verdict=pass
    PROBE pixel_norm values=35 differ=0 rel_rms=0.000e0 verdict=exact pass
    PROBE depth_to_space_2x2x2 values=288 differ=0 rel_rms=0.000e0 verdict=exact pass
    PROBE depth_to_space_2x1x1 values=288 differ=0 rel_rms=0.000e0 verdict=exact pass
    PROBE depth_to_space_1x2x2 values=384 differ=0 rel_rms=0.000e0 verdict=exact pass
    PROBE unpatchify values=576 differ=0 rel_rms=0.000e0 verdict=exact pass
    PROBE to_rgb values=72 differ=0 rel_rms=0.000e0 verdict=exact pass
    PROBE conv3d values=96 differ=0 rel_rms=0.000e0 verdict=residual pass
    PROBE conv3d-control differ=95 rel_rms=6.504e-1 verdict=pass

- The five data-movement and elementwise kernels are **bit identical** to their
  transcriptions, so the channel orders, the temporal drop and every rounding
  are the reference's, not merely close.
- `conv3d` is a residual verdict because the order inside a tap is cuBLASLt's;
  on this draw it came back identical anyway.
- **Two controls prove the comparisons can fail.** The `denormalize` control is
  the same expression with one rounding instead of two and differs at 4 of 24
  values (so two roundings are observable). The `conv3d` control is the same
  convolution with a zero temporal halo and differs at 95 of 96 values at
  rel_rms 6.5e-1 (so the probe separates the two halo policies; the gate fails
  when the separation drops under 1e-2).
- Three of the strides are probed because `(c p1 p2 p3) -> c (d p1) (h p2)
  (w p3)` is not symmetric in the axes, and only the `p1 = 2` cases cut a frame.

`--shapes` (host only, no device): the artifact's block list, the scale factors
the blocks imply (`types.py:41-57`, `(8, 32, 32)`), and the release's geometry
round-tripping — `latent_shape(121, 768, 1344) = (16, 24, 42)` and
`pixel_shape(16, 24, 42) = (121, 768, 1344)`.

The whole decoder over a synthetic latent at `--decode-shape 2,4,4`
(sm_89, 2026-10-30):

    VAE tensors=86 bytes=814338656 load=8.601 scratch=21200896
    LATENT channels=128 frames=2 height=4 width=4 source=synthetic
    DECODE frames=9 height=128 width=128 values=442368 seconds=0.053 fnv=dacd8a778e6cab51 first=0.466796875 last=0.527343750
    RANGE outside=0
    FRAME path=/tmp/vae-frame.png frames=9 height=128 width=128

The 8.6 s load is the gvfs mount at ~110 MB/s, not the artifact; the decode
itself is 53 ms. `RANGE outside=0` is the fold's own check — the reference
clamps, so any value outside `[0, 1]` would mean the fold is not its own.

What the whole-graph run does **not** prove, said plainly: there is no
value-level reference for the composition, because the only real latent in hand
is a draw. `refiner-latent10.bf16` is `h3_ltx_adapter_gate --dump-latent`'s
output, and the adapter's input is the oracle's own
`h3_input = torch.randn(1, *h3_shape, dtype=dtype)`
(`tools/minimax_h3/ltx/latent_oracle.py:175-176`, seed 1234), so every network
downstream of it - the upscaler, the adapter and now this decoder - is being
asked to image white noise. Its statistics say so: 128 x 16 x 24 x 42 values,
mean 0.0054, std 0.8693, and its neighbouring tokens differ by 0.71 / 0.71 /
0.65 (w / h / f) against that std, i.e. as much as random pairs do. Decoding it
gives a band-limited colourful field (the frame is in the record's own directory
after unit 12b), which is what a working decoder produces from noise - not a
picture. A real latent needs the chain's entry point (a source image through the
H3 encoder) or the LTX VAE's own encoder half, and both are owed.

### 3.1 A guard that was tried and rejected, with its number

The first structural guard was the obvious one: the graph is stride-1 with
symmetric halos and regular upsampling, so an impulse at one latent cell should
move the frame's response by exactly one cell (32 pixels) when the cell moves
one step, and not at all across axes. It does not. Measured at
`--decode-shape 2,8,16` (a 9-frame 256x512 decode, four decodes, 0.13 s each):

    IMPULSE cell=(4,13) step_down=(22.41,-0.50) step_right=(23.77,-0.46) scale=32 spread=(106.2,105.2)

The cross-axis steps are zero (so an axis swap *would* show), but the along-axis
step is 0.7 of a cell, and the reason is in the same line: the impulse
response's own spread is 106 pixels against a 32-pixel cell. This decoder's
receptive field is essentially global - twelve 3x3x3 convolutions at the pixel
scale, and four res groups at 1/8 and 1/4 resolution before them - so most of a
response is a diffuse field centred on the frame and the centroid is damped
towards it. The instrument stays in the gate as `--impulse-check`, reporting
only; the geometry is pinned by the exact `depth_to_space` and `unpatchify`
probes instead, which is where it belongs.

## 4. The tiling decision, named

**The decode is untiled.** The release decodes through `TILES`
(`models.py:20-22`: 128/24 frames, 448/64 and 768/64 pixels) resolved by
`_prepare_tiles`/`tiled_decode` (`conv_video_decoder.py:359-482`), which at
`(121, 768, 1344)` is four tiles blended by trapezoidal masks. This port
computes the same convolution stack over the whole latent, and the arithmetic
inside every tile is identical; the values in the overlap zones are not, since
the release averages them there. The tile plan is **not ported** and that is
this unit's one real deviation.

What it costs, computable without a device (`--shapes`, section 3):

    SCRATCH state bytes=1998323712
    SCRATCH aux bytes=1998323712
    SCRATCH work bytes=1998323712
    SCRATCH conv_out bytes=1998323712
    SCRATCH planar bytes=6221021184
    SCRATCH temp bytes=66060288
    SCRATCH rgb bytes=749371392
    SCRATCH total bytes=15029747712

14.0 GiB of working set at the release geometry, against ~1.6 GiB of weights.
The largest single item is the planar buffer: 512 channels at `121 x 96 x 168`
is 3.1 G elements of channel-last padded activation for `up_blocks.7`, and the
planar path holds every frame of it at once. A tiled decode would hold one
tile's worth. This is the number the next unit either pays or removes.

The second named deviation is the release's own memory-efficient decode
(`memory_efficient_decode.py`, applied for the conv VAE by
`blocks.py:1090-1095` with `CHANNELS_LAST_3D_WEIGHTS`): it is a layout and
in-place policy over the same expression, so its values may differ in the last
ulp. The port keeps the reference's channel-major expression.

## 5. What else moved, and what did not

- `h3b_planar` gains `replicate_t` (C ABI, ds4-sys wrapper, two gates and the
  adapter's one call site). The adapter passes `0`, so its arithmetic is
  unchanged; `tests/h3_ltx_ops_gate` passes all 22 probes after the change
  (including `planar values=1350 max=0.000e0`), and its `--residual` guards are
  the unit-11 ones.
- `Shape::elements`, `release_latent`, `scratch_plan`, `latent_shape`,
  `pixel_shape` and `scale_factors` are public because the gate is a separate
  crate; everything else in the module is private.
- `cargo test -p ds4-core --lib`: **768 passed, 0 failed, 7 ignored** (7 of
  them this unit's). `mux_pipe_lifetimes` asserts fd 3 is closed in its child
  and fails only when the invoking shell holds fd 3; with `exec 3<&-` it
  passes, and it is unrelated to this unit.

## 6. Reproduction

    make CUDA_ARCH=sm_89 tests/h3_ltx_vae_gate tests/h3_ltx_ops_gate
    ./tests/h3_ltx_vae_gate --shapes
    ./tests/h3_ltx_vae_gate --probe
    ./tests/h3_ltx_vae_gate --vae <ltx-2.5-video-vae-conv-bf16.safetensors> \
        --decode-shape 2,4,4 --frame /tmp/vae-frame.png

`--latent FILE` takes a raw BF16 `[128][f][h][w]` latent, which is what
`h3_ltx_adapter_gate --dump-latent` writes. The release geometry
(`--decode-shape 16,24,42`) needs 14 GiB of scratch and belongs on the Spark;
the real latent on the Spark is `~/ds4-test/refiner-latent10.bf16`, 4,128,768 B
= 128 x 16 x 24 x 42, and it is the *adapter's* output (the latent stage 2
denoises from), not a denoised one.

## 7. Owed

1. The Spark run at the release geometry: the real latent decoded, the frame on
   disk, the peak footprint measured, and the probes re-run on sm_121 (unit
   12b).
2. The tile plan (`TILES` + `masks_are_complementary` + the masked accumulate),
   or a temporal-slabbed planar path that removes the 6.2 GiB buffer while
   staying untiled. Which of the two is a decision for the next unit, not this
   one.
3. The audio VAE and the mux follow the video decoder; the upscaler's implicit
   GEMM is still owed from unit 8.
