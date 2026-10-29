# The Gemma4 text tower's primitives: built and gated (2026-10-29)

Unit 2's first half. The tower's linear is unit 1's INT8 convrot route
([`gemma-int8-convrot-2026-10-29.md`](gemma-int8-convrot-2026-10-29.md)); this is
everything the forward does around it. The graph it serves is the source read in
[`gemma4-tower-port-map-2026-10-29.md`](gemma4-tower-port-map-2026-10-29.md), and
every kernel below cites the reference line it mirrors.

## 1. What landed

`native/h3/h3_dit_gpu.{h,cu}`, a new section after the INT8 route:

| Entry point | Reference |
|---|---|
| `h3d_embedding` | `embed_tokens(ids) * embed_scale` (`modeling_gemma4.py:1476-1477`) |
| `h3d_gelu_tanh` | ATen's `GeluType::Tanh` (`ActivationGeluKernel.cu:21-31`) |
| `h3d_mul` | `act_fn(gate_proj(x)) * up_proj(x)` (`:1089`) |
| `h3d_scale_bf16` | `hidden_states *= self.layer_scalar` (`:1462`) |
| `h3d_gemma_rope_table` | `Gemma4TextRotaryEmbedding.forward` (`:1164-1181`) |
| `h3d_gemma_rope` | `apply_rotary_pos_emb` + `rotate_half` (`:786-811`) |
| `h3d_gemma_mask` | `eager_mask` (`masking_utils.py:545-611`) |
| `h3d_gemma_attention` | `eager_attention_forward` (`:824-858`) at `scaling = 1.0` |

`h3d_rms_norm` grew the weightless form instead of gaining a second kernel:
Gemma's `v_norm` is built `with_scale=False` (`:1220`), so `weight == NULL`
skips the multiply (`:210-214`). Its `rsqrtf` was already the reference's
primitive: `pow(mean + eps, -0.5)` on CUDA IS `rsqrtf` (`PowKernel.cu:170-179` ->
`UnaryOpsKernel.cu:96` -> `CUDAMathCompat.h:122`).

Reached from Rust through `crates/ds4-sys/src/h3_dit.rs`: `Gpu::embedding`,
`gelu_tanh`, `mul`, `scale_bf16`, `gemma_rope_table`, `gemma_rope`,
`gemma_mask`, `gemma_attention`, and `rms_norm`'s `Option<&Tensor>` weight.

## 2. The numerical contract

BF16 in and out with an F32 intermediate, which is the reference's own path:
`inputs_embeds` fixes the dtype (`:1679`) and torch's half-precision elementwise
kernels compute in float and round once. The port reproduces that rounding
*structure* rather than keeping one wide accumulator:

- the RoPE pair does three roundings (`x*cos`, `rotate_half(x)*sin`, the sum),
  because torch's two multiplies and one add each round;
- attention rounds the dot to BF16 before the mask add, rounds the masked sum
  again, softmaxes the BF16 values in F32, rounds the normalised weight to BF16
  before each product with V, and rounds the output once.

The deviations are the reduction trees (this port's, since no reference tree is
reproducible here) and the libm: `rsqrtf`, `tanhf`, `expf`, `powf`, `cosf` and
`sinf` are the device's implementations in the kernel and the host's in the
gate, and the two disagree in the last bit on a small fraction of inputs. Those
fractions are measured below, not argued.

## 3. The gate: `tests/h3/gemma_tower_bm`

Host transcriptions of the same expressions, with the kernels' own reduction
trees reproduced so that a check with no transcendental in it is bit-exact.
`make CUDA_ARCH=sm_89 tests/h3/gemma_tower_bm`, then
`./tests/h3/gemma_tower_bm <rows> [padded]` - `padded` is the left padding the
release's tokenizer adds, default `rows / 8`, and 0 turns it off, which is how
the fully masked rows were separated from the race in section 4. On vizzio's RTX
4070 SUPER (sm_89, CUDA 13.3) and on the Spark's GB10 (sm_121, CUDA 13.0) every
check passes; the numbers below are the sm_89 run's own output.

Exact, `rows = 1024`:

| Check | Result |
|---|---|
| embedding (gather + BF16 scale), 3,932,160 values | 0 differ |
| RMS norm without a weight | 0 differ |
| tanh GELU, 15,728,640 values | 0 differ |
| the MLP product, 15,728,640 values | 0 differ |
| the layer scalar, 15,728,640 values | 0 differ |
| rotated query and key, both layer shapes | 0 differ |
| the additive mask, both windows, padded | 0 differ of 1,048,576 |

Where a transcendental sits between the kernel and the oracle, the count moved:

| Check | Result |
|---|---|
| RMS norm with a weight, 3,932,160 values | 11 differ, rel RMS 1.04e-05 |
| sliding cos/sin table, 262,144 values each | 6 / 20 differ, rel RMS 8.5e-06 / 1.7e-05 |
| full cos/sin table, 524,288 values each | 44 / 28 differ, rel RMS 7.8e-06 / 2.2e-05 |
| full attention, window 0, 8,388,608 values | 191 differ, rel RMS 9.72e-06 |
| sliding attention, both windows | 0 differ |

The 191 are one (row, head) of the 16,384: a single softmax whose weights moved
by a last-bit `expf` and whose 512 outputs then re-rounded on either side of a
BF16 boundary. Nothing else in the attention moved. Every tolerance check gates
at rel RMS 1e-3, which is above BF16's own half-ulp (3.9e-03 on the mantissa but
~2e-03 on a value's own scale) and far below any layout or ordering error: the
two real bugs the gate caught both showed rel RMS above 0.35.

At `rows = 128` the same 22 checks pass with the embeddings, products and masks
exact and 1, 0, 2 and 0 differing values in the four transcendental rows.

## 4. The three bugs the gate caught

Worth recording, because two were the same mistake, and the third only a second
host could find:

1. `h3d_gemma_rope_kernel` read `x[partner]` and wrote `x[dimension]` in the same
   block with no barrier between the reads and the writes, so the rotation read a
   half-updated row. `rotate_half` reads across the two halves of the row and the
   two partners are in the same block by construction, so the fix is a shared
   snapshot of the row, one `__syncthreads()`, then the writes. rel RMS 0.508
   before, 0 after.
2. The host oracle had the same bug: it wrote `x[dimension]` in place and then
   read `x[dimension - half]` for the upper half, which it had already
   overwritten. Fixed by snapshotting the row first, as torch materialises
   `rotate_half` before the products.
3. `h3d_gemma_attention_kernel` reused one shared array for the block's maximum
   and then for its sum, and the sum's first write was not separated from the
   read of the maximum. A warp that reached its own (empty) sum loop ahead of a
   straggler's read of `partial[0]` let that read pick up the sum - 128.0 for a
   128-key row - as the maximum, which drives every `exp` in the row to zero and
   changes the output completely. One `__syncthreads()` between the two.

The third is the reason the Spark's sm_121 is not optional. It read as clean on
sm_89 across four runs and failed on sm_121 in the first one; with the harness
able to turn the left padding off, the same local command then disagreed with
itself across consecutive runs, which is what confirmed it as timing rather than
architecture. Eight consecutive runs of each configuration agree after the fix.

The first run also had a harness bug (a `float` uploaded into a BF16 `[1]`),
which read as a 100%-wrong layer scalar.

## 5. Limits, checked and refused

- `h3d_gemma_attention`: `rows <= 4096` and `head_dim <= 512`, because one row's
  score vector is staged in shared memory (16 KB + 2 KB + 1 KB); `head_dim` must
  be a multiple of 256 (8 warps x 32 lanes) so every lane owns whole elements.
  The release's tokenizer truncates to 1024, so the encoder never asks for more.
- `h3d_gemma_rope_table`: `angles <= dim / 2`, `base > 0`.
- `h3d_gemma_rope`: `head_dim <= 512` and even. The launch is exactly `head_dim`
  threads, which is what makes the unguarded barrier in the kernel safe.

## 6. Corrections to the earlier records

- The port map called the mask an F32 `-inf` additive mask. It is a BF16
  `finfo(bfloat16).min` mask (`masking_utils.py:942,990,608-610`). The port map
  now says so, and the consequence matters: a fully masked row softmaxes to
  uniform, not to NaN, and the attention's mask add rounds to BF16.
- The port map listed "a BF16 elementwise multiply exists only as
  `h3b_gate_mul`". It carries a SiLU gate, not a plain product, so `h3d_mul` is
  its own op rather than a reuse.

## 7. What unit 2 still owes

The tower graph itself: `crates/ds4-core/src/minimax_h3/gemma.rs`, loading the
INT8 checkpoint (the sibling pair's 328 linears, four norms per layer, the
`layer_scalar`) and running the 48 layers, then handing over the 49 hidden states
the extractor stacks. Its gate is the operator's golden dump; until that lands
the tower is gated against these primitives plus the record's own numbers.
