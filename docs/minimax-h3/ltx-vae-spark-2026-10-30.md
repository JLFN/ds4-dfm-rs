# Unit 12b — the video VAE decoder on the Spark

Continues `docs/minimax-h3/ltx-vae-decoder-2026-10-30.md` (unit 12a). Plan
`043383cc` -> this unit. Landed: the impulse instrument (`043383cc`) and the
frame's pixel-step report (`04e3f5ae`), both run on sm_121 at the release
geometry with a peak-footprint sample.

## 1. The hosts agree bit for bit on everything that has no cuBLAS in it

`./tests/h3_ltx_vae_gate --probe` on the Spark (sm_121, 2026-10-30) is
identical to the workstation's run (sm_89, section 3 of the 12a record), line
for line: the five elementwise and movement ops at `differ=0`, `conv3d` at
`differ=0` as well, and both controls (`denormalize-control differ=4`,
`conv3d-control differ=95 rel_rms=6.504e-1`). The planar path's matmuls are
cuBLASLt and therefore not bit-stable across architectures in general; on this
probe's three-channel, 96-element sums the transcription and the device agree on
both hosts anyway, so the probe's verdict is the same either way.

## 2. The whole decoder at 121 x 768 x 1344, three latents

All three decodes are one binary, one build, 121 frames of `768 x 1344` and
374,685,696 values. `--expect` is the hash the previous Spark build produced;
each reproduced exactly, so the decode is deterministic across a rebuild of the
native object and the Rust host.

| latent | source | seconds | fnv | first | last |
|---|---|---|---|---|---|
| `refiner-latent10.bf16` | `h3_ltx_adapter_gate --dump-latent` | 14.46 | `5ad08268f3b2acee` | 0.238281250 | 0.296875000 |
| `stage10-dense/stage-latent.bf16` | the 48-layer stage-2 run's output | 14.41 | `9be00ee0f10f4241` | 0.417968750 | 0.265625000 |
| synthetic (the gate's own draw) | `--seed 1` | 14.38 | `14a8e079577e3255` | 0.503906250 | 0.367187500 |

`RANGE outside=0` on all three: the RGB fold's own clamp holds, which is the
reference's `(x + 1) / 2` clamped to `[0, 1]` (`conv_video_decoder.py:499`).

The frames are in this directory as `vae-frame-adapter-2026-10-30.png` and
`vae-frame-stage-2026-10-30.png` (frame 0 of each). They are not pictures, and
that is measured rather than assumed:

- both latents are draws, not images. `refiner-latent10.bf16` is the adapter's
  output for the oracle's own `torch.randn` input
  (`tools/minimax_h3/ltx/latent_oracle.py:175-176`, seed 1234), and
  `stage-latent.bf16` is the stage-2 denoiser's output from the noised version
  of that;
- their statistics say so. The adapter latent: mean 0.0054, std 0.8693, and
  neighbouring tokens differ by 0.711 (w) / 0.710 (h) / 0.646 (f). The stage-2
  latent: mean 0.0211, std 0.8866, neighbours 0.977 / 0.976 / 0.976. For
  independent Gaussian draws `E|X - Y| = 2 sigma / sqrt(pi) = 1.128 sigma`, so
  the stage-2 latent is within 2% of white noise in every axis and the adapter
  latent is at 0.74-0.82 of that, i.e. only modestly smoother;
- and the decodes have the right *shape* of wrongness: the adapter's frame is
  coarse and colourful, the stage-2 frame is fine and grey. That is what the
  4x4 unpatchify does to decorrelated channels (section 3) on latents whose
  coarse spatial structure differs.

So this unit still has no value-level reference for the composition, and says
so: the entry point of the chain (a real source image through the H3 encoder,
then the upscaler and the adapter) or the LTX VAE's own encoder half is owed.
What the run does establish is that the graph executes at the release geometry,
deterministically, inside the fold's own range, at a measured cost.

## 3. The frame's own block structure

Each decode now reports the mean absolute difference between pixels one, four
and sixteen apart, over the frame's own standard deviation:

| latent | step_1 | step_4 | step_16 |
|---|---|---|---|
| adapter | 0.2006 | 0.3124 | 0.7055 |
| stage-2 | 0.1393 | 0.4323 | 0.8627 |
| synthetic | 0.1449 | 0.4320 | 0.8163 |
| synthetic, `(2, 8, 16)` on the workstation | 0.1428 | 0.4229 | 0.8098 |

One pixel apart is the *small* step and four pixels apart is three times it.
That is the unpatchify's own block structure: sixteen channels are interleaved
into every 4x4 block (`ops.py:56-58`), so pixels inside a block share a latent
cell and are correlated, while pixels one block apart are driven by different
cells - which on an iid latent are independent. The step grows with distance
towards the 1.128 sigma white-noise limit, as an independent-cell input
requires.

This is a report, not a verdict, and the reason is in the numbers: a
*transposed* sub-position order (the shuffle's `(h q)` read as `(w r)`) would
move the block structure to the other axis and an isotropic latent would not
show it. The ordering itself is pinned by the exact `unpatchify` probe, which
transcribes `c p^2 + r p + q` for row `h p + q` and column `w p + r`
element by element.

## 4. The impulse at the release geometry

`--impulse-check` at `(16, 24, 42)`, reporting only (12a recorded the reason: a
centroid is damped by this decoder's near-global receptive field):

    IMPULSE cell=(12,39) step_down=(23.99,-0.20) step_right=(25.16,-0.25) scale=32 spread=(130.6,131.5)

The cross-axis steps are 0.2 pixels (no axis swap), the along-axis steps are
0.75 of the 32-pixel scale factor, and the response's own spread is 130.6
pixels against a 32-pixel cell - on a bigger frame the spread grows with the
frame, which is the receptive field and not a geometry error.

## 5. The footprint

`free` sampled every 0.5 s through the whole 12b run (`~/ds4-test/u12b-free.log`,
287 samples): used memory rose from 3,040 MB to a peak of **23,398 MB**, and
free fell from 46,328 MB to a minimum of 26,553 MB - about 20 GB resident, of
which the preallocated working set is 15,029,747,712 B (14.0 GiB, section 4 of
the 12a record), the decoder's weights are 814,338,656 B and their planar pack
is a second copy of most of them. The rest is cuBLASLt's own workspace.

That is the untiled deviation's price, on a host with 121 GB. A tiled decode
(the release's own four tiles) or a temporal-slabbed planar path is what would
take it back down, and 12a's section 7 leaves that choice open.

## 6. Reproduction

    # workstation (sm_89)
    make CUDA_ARCH=sm_89 tests/h3_ltx_vae_gate
    ./tests/h3_ltx_vae_gate --shapes
    ./tests/h3_ltx_vae_gate --probe
    ./tests/h3_ltx_vae_gate --vae <vae.safetensors> --decode-shape 16,24,42 \
        --latent <latent.bf16> --frame /tmp/frame.png --impulse-check

    # Spark (sm_121): ~/ds4-test/u12c-spark.sh, u12b-spark.sh
    cd ~/ds4-sol-h3 && export PATH=$HOME/.cargo/bin:$PATH
    rm -f native/h3/*.o tests/h3_ltx_vae_gate
    make CUDA_ARCH=sm_121 tests/h3_ltx_vae_gate

The Spark's `--latent` files: `~/ds4-test/refiner-latent10.bf16` (the adapter's
output) and `~/ds4-test/stage10-dense/stage-latent.bf16` (stage 2's). Both are
raw BF16 `[128][16][24][42]`, 4,128,768 B.

## 7. Owed, in order of what it buys

1. **The LTX VAE's encoder half** (the other 84 tensors in the same artifact):
   encode a real image, decode it, and the round trip is a value-level proof of
   the whole pair with no other unit in the path. Cheapest decisive test.
2. **The chain's entry point**: a source image through the ported H3 encoder,
   the upscaler and the adapter, then this decoder - the product path, and the
   only one that produces a frame worth looking at.
3. The tile plan or a slabbed planar path (section 5), which is the memory
   decision 12a left open.
4. The audio VAE and the mux, then the release's `dev` INT8 connector file
   still owed from unit 9.
