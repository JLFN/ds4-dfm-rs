# Sol-Attn's sparse forward, ported and measured (2026-10-28)

Branch `feat/sol-h3`. Unit: `h3d_sol_forward`, the consumer of the routing mask
`h3d_sol_search` builds. Commits `93573bfd` (the forward, ABI, tests, bench),
`90efac76` (the bench's control arms), `5b3a0f55` (the launch bound). All
measurements on the DGX Spark, `sm_121a`, CUDA 13.0, busy clocks 2.4-2.5 GHz
(the host has no clock policy: `nvidia-clocks.service` does not exist there, so
every figure below is at a higher clock than the campaign's 2184-2190 MHz
reference range). `ncu` is still blocked by `ERR_NVGPUCTRPERM`, so every
attribution here is a timing inference, not a counter.

## 1. What the reference does, and what this walks

NVIDIA Sana `sol-engine` @ `6e4a440`, `techniques/sparse_backends/sol_attn/`:

- `preprocess.py:45,82` - `kc` is the block **mean** of K and `vc` the block
  **sum** of V, both derated to BF16 (`:287-291`).
- `triton_ref/fwd.py:88-108` - per group of `GROUP = 32` KV blocks, the pooled
  scores `q @ kc^T * scale*log2(e)`, and the group's exact/approximated split.
- `fwd.py:109-131` - the approximated blocks are **folded into the same online
  softmax**, not skipped: `o += bf16(p) @ vc` and
  `row_sum += p * block_length`. That is why `vc` is a sum: the fold carries the
  block's whole contribution, and the length factor restores the denominator.
- `fwd.py:132-170` - the group's exact blocks walked in increasing block order,
  one `BLOCK = 64` key tile at a time, with a maximum and a rescale per tile.
- `fwd.py:171-173` - `o / row_sum` stored BF16.
- `fwd.py:256-262` - a group with no approximated block skips the folded pass
  entirely, leaving the running maximum untouched.

The port is `h3d_sol_forward` (`native/h3/h3_dit_gpu.cu`, declared in
`h3_dit_gpu.h` beside `h3d_sol_search`). It reads `kc`, `vc` and `mask` only:
the mask already carries the diagonal band and the sink range from the search
step (`h3d_sol_route_kernel`), so the forward adds no routing policy of its own.

Structure: one CTA per (query block, head), four warps of 16 query rows;
head_dim 128 = 8 k16 steps; output tile BV = 128. Two shared tiles of 64 rows x
136 BF16 alias per stage - `shared_left` holds `kc` then the exact block's key,
`shared_right` holds `vc` then its value - which keeps the CTA inside the 48 KiB
static shared limit. Per exact block: `__syncthreads`, cooperative staging of K
and V, `__syncthreads`, 64 score MMAs and 64 output MMAs per warp.

## 2. Correctness

`tests/h3/native_test`: **102 PASS / 0 FAIL** on the Spark (`sm_121a`) and on
the workstation (`sm_89`), up from 73 before this unit.

The oracle in the test file transcribes `fwd.py:88-173` and consumes the mask
**as an input**, so the consumer is gated apart from the router. Measured
agreement over all live elements:

| case | folded share | worst element | relative RMS |
|---|---|---|---|
| router mask, tau 0.5 | 22.9% | 7.812e-03 | 5.3e-05 |
| router mask, tau 1.5 | 22.9% | 7.812e-03 | 5.3e-05 |
| all blocks exact | 0% | 7.812e-03 | 4.7e-05 |
| all blocks exact, against `h3d_sdpa` | 0% | 7.812e-03 | 5.5e-05 |
| 33 blocks over two words | 51.1% | 7.812e-03 | 1.6e-04 |
| hand-built, two far blocks, no band | 75.0% | 7.812e-03 | 1.2e-04 |
| hand-built, nothing exact | 100% | 3.906e-03 | 1.5e-05 |
| hand-built, one word exact, one folded | 3.0% | 7.812e-03 | 3.2e-04 |

The worst deltas are one BF16 ULP on values of order 1 (BF16's step near 1.0 is
7.8125e-03), and only 1 to 416 of 65,536 to 537,600 elements differ. The gate is
`relative RMS < 1e-3`.

The all-exact case is the strong one: the sparse walk over the same tiles as
`h3d_sdpa` agrees with it to 5.5e-05 relative RMS, which is the two kernels'
`expf(scale*s)` against `exp2f(log2(e)*scale*s)` and nothing else.

Mutation check, as the campaign rule requires: with the folded `P @ vc` removed
(a `false &&` on the fold's MMA loop) 6 of the new checks fail - the four arms
that fold anything, at relative RMS 4.3e-02 to 1.0 - while the two all-exact
arms still pass. The fold is therefore the property under test, and the test
fails when it is wrong.

## 3. Performance, and the defect the first measurement exposed

`tests/h3/sol_forward_bm` (new target, `make tests/h3/sol_forward_bm`), 3 to 5
back-to-back calls each, one sync, dense arm first and repeated around the
control arms so run order and clocks cannot explain a gap.

18432 x 56 x 128 (the denoise shape: 288 blocks, 9 mask words per query block):

| arm | exact | forward | + search | vs dense |
|---|---|---|---|---|
| dense `h3d_sdpa` | - | 173.0-174.3 ms | - | 1.00x |
| sparse, tau 0.5 | 49.1% | 76.0 ms | 80.9 ms | **2.14x** |
| sparse, tau 1.25 | 47.2% | 73.4 ms | 78.3 ms | **2.22x** |
| sparse, tau 3.0 | 42.6% | 67.6 ms | 72.5 ms | **2.40x** |
| control: all blocks exact | 100% | 150.3-152.1 ms | - | 0.87x |
| control: same count, contiguous window | 47.2% | 72.1 ms | - | - |

4352 x 56 x 128 (512x512 denoise: 68 blocks):

| arm | exact | forward | + search | vs dense |
|---|---|---|---|---|
| dense `h3d_sdpa` | - | 9.85-10.00 ms | - | 1.00x |
| sparse, tau 1.25 | 48.5% | 4.46 ms | 5.57 ms | **1.77x** |
| control: all blocks exact | 100% | 8.82 ms | - | 0.88x |

The forward now scales the way the algorithm says it should:
`forward ~= exact_share x all_exact_time + 2-5 ms`, checked at three shares
(0.426 x 150.3 + 3.6 = 67.6 measured 67.6; 0.472 x 152.1 + 1.6 = 73.4 measured
73.4; 0.491 x 150.6 + 2.1 = 76.0 measured 76.0). The constant is the nine folded
passes, about 4.5 tile-equivalents.

### The defect, and what fixed it

The first measurement of this kernel had it at only **1.19x** the dense
attention at 47.2% exact, and 0.94x end to end at 4352 - slower than dense. Two
hypotheses were killed by measurement before the right one was found:

- **Register spills**: refuted. `ptxas -v` on the Spark reported 176 registers
  and 0 bytes of spill stores and loads for `h3d_sol_forward_kernel`, against
  168 registers for `h3d_attention_mma_kernel`; both with the same 34,816 bytes
  of shared memory.
- **Scattered against sequential K/V addresses**: refuted. The bench's second
  control moves every query block's exact set to a contiguous window of the same
  size - same tile count per CTA, same total tile count, same MMAs and barriers
  - and it measured 72.1 ms against the router's scattered 73.4 ms, 2% apart.
  Address pattern is not a factor at this shape.
- **The register boundary was the cause.** 168 registers is three CTAs of 128
  threads in the 64K register file; 176 is two. `__launch_bounds__(128, 3)`
  brought the kernel to 165 registers with zero spills, and the same binary,
  same host, same minutes:

| arm | two CTAs (176 regs) | three CTAs (165 regs) | change |
|---|---|---|---|
| dense | 173.7 ms | 173.1 ms | unchanged |
| sparse, 47.2% exact | 145.0 ms | 73.1-73.4 ms | **2.0x faster** |
| all blocks exact | 317.4 ms | 150.3-152.1 ms | **2.1x faster** |
| windowed control | 144.3 ms | 72.1 ms | 2.0x faster |

The all-exact control is what made the cause visible: at 100% exact the sparse
walk does the dense kernel's own work - the same 288 sequential tiles, the same
MMAs, the same staging, the same two barriers per tile, no fold - and it took
1.83x the dense kernel. On the workstation at 2,048 tokens the same comparison
is 0.91x, so the penalty was the target shape's latency exposure meeting an
occupancy one CTA lower, not anything in the walk.

Cost of the bound where it does not help: on `sm_89` at 2,048 tokens the
router-mask arm moves 0.210 -> 0.218 ms and the all-exact arm 0.352 -> 0.397 ms,
a 4-13% regression on a host the campaign never measures on. The Spark is the
release reference and gains 2.0x from it.

## 4. What this changes for the campaign

The attention is no longer a primary cost of the denoise. At the recorded job A
shape (768x768, 107 frames, 2 evals) the denoise is 40.294 s of the 277.019 s
wall, and the DiT's attention share of it was estimated at ~17.3 s from our own
50-layer count and a measured 173.163 ms dense call. At 47% exact that share
falls to roughly 8 s before any of the weights levers.

What this unit does **not** establish:

- The realized sparsity is a property of the fixture, not of the model. This
  bench fills K, V and Q with a per-block signal, and the tau routing then
  marks 42-49% of block pairs exact. The reference's stage-1 sparsity (90%) is
  its **VSA** policy - `runtime/stage1_ops/vsa.py:24-33`, tile 64, prefix
  exempt, `topk = ceil(0.1 * video tiles)` - which is a different rule from the
  tau threshold, and its stage-2 Sol-Attn taus were tuned on real LTX latents.
  Measuring the model's own sparsity needs a denoise on real latents, which is
  the next thing to wire.
- At 4352 the routing pass is 1.10 ms against a 4.46 ms forward: a quarter of
  the sparse attention time. It is priced separately and untouched here.
