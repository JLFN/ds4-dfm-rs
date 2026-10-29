# P2: the H3-to-LTX latent path, ported and gated

Measured 2026-10-29 on the DGX Spark (192.168.1.91, GB10 sm_121, no clock
policy: 208 MHz idle, 2431 MHz during the run; 40 C, 43.8 W at the tail) and
on the workstation's RTX 4070 SUPER for the daily gate. The code is
`ff7c3fbf` and `a60db304` on `feat/sol-h3`.

## 1. What was ported

The profile's stage-2 latent path, which has no C reference — its authority is
the Sol-H3-Spark Python and the pinned author node:

| op | reference | port |
|---|---|---|
| x2 latent upscaler | `stage2_ops/h3_upscale.py:112-126` around `LatentResizer3D` (`nodes/minimax_h3_latent_upscaler_3d.py` `d7c01b90`) | `Upscaler` |
| alignment | `stage2_ops/h3_ltx_adapter/geometry.py:81-112` | `Adapter::align` |
| 22-block adapter | `stage2_ops/h3_ltx_adapter/model.py:28-42` | `Adapter::model` |

`Upscaler` and `Adapter` are separate types called in the order
`runtime/stage2.py:58-76` calls them; nothing wraps them in a layer, and no
part of the path is Python at run time.

Device side, `h3b_*` in `native/h3/h3_dit_gpu.cu`: zero pad, im2col, two
cuBLASLt matmuls into an f32 accumulator (im2col layouts for a convolution,
channel-major operands for a 1x1, which copies nothing), a tiled-transpose
epilogue that adds the bias and rounds once, a volume GroupNorm with the
ResBlock epilogues folded in, SiLU, add, a depthwise temporal convolution, a
trilinear resize, the alignment's gather/lerp/offset copy, a pixel unshuffle,
the chunked accumulation and its weighted average, the gate multiply and the
node's N and D. Each kernel is templated on the element type, so one binary
carries the bf16 path the reference runs and an f32 arm used for gating.

Two design points worth keeping:

- **A 3x3x3 is im2col plus cuBLASLt, in temporal slabs.** The whole im2col of
  the widest convolution (48x84 positions x 13,824 columns) would be 4.1 GiB;
  four frames at a time keeps it bounded and still gives a 16,128-row GEMM.
  The slab is the lesser of what the matrix and the f32 accumulator hold, so
  the bound is enforced rather than assumed.
- **No weight is ever permuted.** PyTorch stores a conv weight `[out][in][kd]
  [kh][kw]`, which flattened is exactly the `(out, in*kd*kh*kw)` matrix both
  matmuls want, and the im2col writes its columns in that same order.

## 2. The gates

`h3_ltx_ops_gate` drives all eighteen primitives against a host implementation
of the same arithmetic, on shapes whose dimensions all differ. It is not
ceremony: it found the two bugs that the end-to-end run could only report as
"wrong".

- The epilogue's channel and spatial roles were traded: the store phase read
  its tile with the block indices swapped, so every convolution wrote a
  transposed, partly clipped result.
- `N` must **divide** by the deviation (`(x - mean) / std`,
  `minimax_h3_latent_upscaler_3d.py:586`) and `D` multiply
  (`x * std + mean`, `:592`). One `scale` tensor used both ways left the
  upscaler's input wrong by a factor of std^2 per channel.

`h3_ltx_adapter_gate` runs the whole path at the release's geometry (124
frames, a 384x672 draft) against the frozen float32 vector the reference's own
Python produced, and reports each stage's relative RMS. It also runs the model
alone on a host alignment and the alignment alone, which is what separated the
alignment's channel stride (it must be the *aligned* plane, 48x84, not the LTX
latent's 24x42) from the model's.

Both gates pass on both hosts.

Spark, bf16 arm, this run:

```
h3_upscaled    relative 6.736e-3   max 9.30e-2
aligned_h3     relative 1.775e-3
adapter_output relative 1.210e-2   max 1.32e-1
refiner_input  relative 1.213e-2
```

Workstation, against the same vector: the f32 arm reproduces every stage to
~1e-6 relative (upscaler 2.874e-6, alignment 5.805e-9, adapter 6.076e-6,
refiner input 6.154e-6) and the bf16 arm at 6.7e-3, 1.8e-3, 1.2e-2, 1.2e-2.

The f32 arm is the *structural* gate: it is the same arithmetic as the
reference's own float32 run, so only the summation order differs. The bf16 arm
is a precision statement against that same vector, not a second structural
gate: the reference runs bf16 too, so its own result would sit the same
distance from its f32 run.

How far that distance reaches is measured, not assumed. Feeding the adapter an
input whose last bits differ (an f32 alignment rounded once instead of the
device's bf16 pipeline) moves its output by 7.4e-3 on the Spark and 1.13e-1 on
the 4070 — the model amplifies a last-bit input difference through 22 blocks,
and the two devices differ because cuBLASLt picks different kernels per arch.
Any future A/B of this path at bf16 must expect an output difference of that
order, and a mean must be compared, not a single element.

## 3. Speed, against the release

The release quotes 2.13 s for "upscale + adapter" (3.8% of its 56.17 s
end-to-end, on the same Spark). The port, same geometry, bf16 arm, this run:

| stage | Spark | 4070 SUPER |
|---|---|---|
| upscaler forward | 3.895 s | 3.512 s |
| adapter convert | 0.255 s | 0.201 s |
| total | **4.150 s** | 3.713 s |

So the port is correct and 1.95x off the release on the same host. That is the
expected shape of the number and it is not a surprise: this path is
bandwidth-bound and the Spark has less of it (273 GB/s) than the workstation
card (504 GB/s), which is why the Spark is *slower* despite more compute.

The cost is where the design put it. Per conversion the out_blocks' im2col
alone is ~83 GiB written and read again for 42.5 TFLOP of GEMM; the in_blocks
add ~21 GiB for 10.6 TFLOP. At 273 GB/s the im2col traffic is ~0.76 s and the
GEMMs ~0.49 s at peak (108 TFLOPS), so 3.9 s means roughly 40% GEMM efficiency
and a large share of the wall time in materializing 104 GiB of matrix.

That is the optimization target, and it is a real one: an implicit-GEMM
convolution (the im2col folded into the GEMM's K loop, accumulators in
registers) removes the 104 GiB entirely and keeps the arithmetic order. The
ceiling after that is ~0.8-1.1 s, i.e. comfortably inside the release's 2.13 s.

## 4. What this does not cover

- Only the release's geometry is gated (124 frames, 384x672 draft, which is
  the *chunked* upscaler path). Job A's own shape (107 frames, 768x768) has no
  frozen vector yet, and it is the non-chunked branch.
- No numeric reference for the LTX refiner, the Conv VAE or the mux: P3 and P4
  are unported. The path ends where the profile's `video_encode` ends.
- The offline int8 Gemma pair and the prompt cache are still missing from the
  store, so the refiner cannot be run even once it is ported.

## 5. Reproducing

Workstation:

```
make CUDA_ARCH=native tests/h3_ltx_ops_gate tests/h3_ltx_adapter_gate
./tests/h3_ltx_ops_gate
./tests/h3_ltx_adapter_gate /tmp/bridge-golden-real \
  /data/h3-latent-ckpt/minimax_h3_latent_upscaler_3d_bf16.safetensors \
  /data/h3-latent-ckpt/adapter --arm f32
```

Spark (`~/ds4-sol-h3`, code over a `git bundle`, checkpoints in
`/mnt/h3-artifacts`, the frozen vector copied to `~/ltx-golden`):

```
make CUDA_ARCH=sm_121 tests/h3_ltx_ops_gate tests/h3_ltx_adapter_gate
./tests/h3_ltx_ops_gate
./tests/h3_ltx_adapter_gate ~/ltx-golden \
  /mnt/h3-artifacts/H3-upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors \
  /mnt/h3-artifacts/H3-to-LTX-Latent-Adapter --arm bf16
```

The frozen vector is produced by `tools/minimax_h3/ltx/latent_oracle.py`, which
is a test-vector generator from the reference's own Python; it is not part of
the port and nothing at run time depends on it.
