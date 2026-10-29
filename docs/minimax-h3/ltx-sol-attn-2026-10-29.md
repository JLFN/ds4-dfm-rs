# The refiner's sparse attention: Sol-Attn, read from its source

Step 1-2 of the handoff's unit after the refiner read. The point was to name the
one mechanism the refiner needs that a dense port would have to invent. The
answer inverts the assumption: **the port already implements it** - the H3 stage's
own sparse path is this same algorithm, ported in the `sol-h3` work with citations
to these exact files - so what the refiner needs is a composition layer and a
shape, not a kernel.

## 1. What runs on the Spark

- The release refuses to run unless the backend is the Triton one:
  `runtime/stage2_ops/models.py:101-102` raises when
  `base._actual_sol_backend() != "triton"`, and the receipt records
  `architecture "sm121-triton"` (`runtime/stage2.py:103`). The CuTe SM89/90/100/120
  kernels are not the path here; the README's dispatch table sends an unknown
  capability to Triton.
- The call the profile makes, once per video self-attention per step:
  `super_acceleration/stage2/sol_attention.py:173-194` reshapes q/k/v to BTHD and
  calls `_run_sol_attn_bthd(..., tau=STAGE2_TAUS[self._step], thresh_type="diag",
  kv_splits="auto", dense_fn=...)`.
- `techniques/sparse_backends/sol_attn_backend.py:119-170` then validates
  (contiguous, BF16, head_dim 128, CUDA), resolves `kv_splits="auto"` to **1**
  (`:94-108`: split 4 is SM90-only), and calls the kernel with **no `scale`** - so
  the kernel takes its default `head_dim ** -0.5` (`sol_attn/triton_ref/fwd.py:392`,
  `sol_attn_backend.py:31`) - and **no sink** (both default to zero), which is
  right for this call: `attn1` is video self-attention and has no text span.
- `SOL_ATTN_STRICT=1` is set for the whole profile
  (`runtime/stage2_ops/models.py:38`), so an ineligible call raises instead of
  falling back to `scaled_dot_product_attention` (`sol_attn_backend.py:165-170`).
  The 3/141 receipt (`runtime/stage2.py:101-107`) is therefore a count of real
  kernel entries: one dense layer-0 call and 47 Sol calls per update.

## 2. The algorithm, in four stages

Blocks are 64 tokens and the forward walks them in groups of 32
(`sol_attn/triton_ref/preprocess.py:10-12`, `triton_ref/fwd.py:21-22`); `NT` is
`ceil(T/64)`.

1. **Block summaries** (`sol_attn/preprocess.py:25-88`, the TMA pair the Spark
   uses; the pointer pair `triton_ref/preprocess.py:24-53` is the same math):
   `kc[block] = mean over the block of K` and `vc[block] = SUM over the block of V`,
   both stored BF16. The mean/sum asymmetry is the method: an approximated block's
   contribution is `exp2(score(centroid)) * vc`, which is exactly the block's true
   contribution with `exp2` pulled out of the sum - so `kc` carries the centroid
   and `vc` carries the un-divided sum (any fix that made `vc` a mean would scale
   every approximated block by 1/64).
2. **Centroid statistics** (`sol_attn/preprocess.py:90-135`): per head, the mean
   and the diagonal variance of the `kc` centroids over all blocks, computed in
   F32 from the BF16 summaries.
3. **The per-query-block threshold** (`sol_attn/preprocess.py:143-186`, the `diag`
   variant): the query block's F32 centroid `q_bar`, scores in log2 units
   (`log2_scale = scale * log2(e)`, `:176`), then

       mean     = <q_bar, kc_mean> * log2_scale
       variance = sum_d q_bar_d^2 * kc_var_diag_d * log2_scale^2
       threshold = mean + tau * sqrt(max(variance, 0) + 1e-6)

   One scalar per (query block, head). `thresh_type="exact"` replaces the diagonal
   variance with the full `D x D` second-moment matrix (`:215-278`); the Spark
   passes `"diag"`, so the port's diagonal model is the released one.
4. **Routing and the forward** (`triton_ref/fwd.py:82-173`, pointer twin
   `:238-357`): per group of 32 KV blocks, compute the proxy score block
   `scores = q @ kc^T * log2_scale` (64x32) and route per KV block:

       exact = (mean over the query block of scores > threshold) | (|q_block - n| <= 1)

   (`fwd.py:93-107`; the CuTe selector states the same rule as
   `sol_attn/common/selector.py:132-142`, plus the sink range `:134-142` of
   `interface.py`, which this call does not use). A block is "exact" if its
   centroid score beats the threshold, or it is the query's own or the adjacent
   block. Then:
   - **approximated** blocks are folded in one pass from the summaries:
     `p = exp2(scores - m)`, `o += p @ vc`, `row_sum += p * block_length`
     (`fwd.py:109-131`);
   - **exact** blocks are walked in increasing block order, one at a time and one
     maximum per 64-key block, with the real K and V (`fwd.py:132-170`);
   - the accumulator is a running online softmax (`row_max`, `alpha = exp2(m_old -
     m_new)`) shared by both paths, and the output is `o / row_sum` in BF16
     (`fwd.py:171-173`).
   The tail block's length is `T - (NT-1)*64` (`fwd.py:82`), and the denominator
   uses lengths rather than counting exponentials, which is what makes the
   approximated path's softmax internally consistent.

`tau` is the whole policy: the README says a higher value "selects fewer KV blocks
for exact attention", the threshold is `mean + tau * std`, and the three-update
schedule raises it per step (`1.0, 1.25, 1.5`, `stage2/sol_attention.py:20`,
`:188`), so the last update is the sparsest.

## 3. What the approximation is, numerically

- The decision is made from the **proxy** score, not from the real scores: routing
  compares the threshold with `q @ kc^T`, so a block whose keys vary a lot inside it
  can be misrouted in either direction. That is inherent to the method, not a port
  defect.
- An approximated block contributes `exp2(score_of_centroid) * sum(V)` instead of
  `sum(exp2(score_t) * V_t)`, i.e. the error is second-order in the within-block key
  spread - which is exactly what the `mean + tau*std` threshold budgets against
  (the variance of the centroid score over blocks stands in for the within-block
  spread).
- `kc`/`vc` are BF16, the threshold, centroids and accumulators F32, and `P` is
  cast to BF16 before both products as the reference does (`fwd.py:129`, `:164`).
- So the output is **not bit-identical to dense attention** and is not meant to be.
  Like the campaign's mixed-quant arms, a refiner arm using Sol-Attn has to be
  gated on output quality (paired NLL / the token stream), with the dense run as
  the reference, not on bit identity.

## 4. The port already has this - and already cites these files

`native/h3/h3_dit_gpu.{cu,h}` carries the same four stages, with the reference's
line numbers in its comments:

| port | reference |
|---|---|
| `h3d_sol_pool_kernel` `h3_dit_gpu.cu:4058-4080` (kc mean, vc sum, row order preserved) | `preprocess.py:64`, `:66`, `:82` |
| `h3d_sol_stats_kernel` `:4084-4107` | `preprocess.py:96-135` |
| `h3d_sol_threshold_kernel` `:4110-...` (centroid, log2, `mean + tau*std`) | `preprocess.py:177-186`, `:189-213` |
| the router `h3d_sol_search` (`h3_dit_gpu.h:550-553`) | `preprocess.py:280-322`, `triton_ref/fwd.py:250-266`, `selector.py:132-142`, `interface.py:134-142` |
| `h3d_sol_forward_kernel` `h3_dit_gpu.cu:4612-4930` - the group walk, the fold from `vc`, the exact-block walk, the tail, `output/row_sum` | `fwd.py:88-108`, `:109-131`, `:132-170`, `:171-173` |

It is **live, not aspirational**: `crates/ds4-core/src/minimax_h3/dit.rs:2751` and
`:2772` call `sol_vsa` (the selection) and `sol_forward` (the walk) per block on
every job A run, which is why the denoise attribution lists
`h3d_sol_forward_kernel` (1,311.59 ms per evaluation)
and `h3d_sol_vsa_kernel` (216.69 ms). Stage 1's policy is the *other* one in the
same tree - VSA's top-k at sparsity 0.9 rather than tau
(`h3d_sol_vsa`, `h3_dit_gpu.h:580-588`) - and it reuses the same pooling, statistics
and threshold kernels (`h3_dit_gpu.cu:4515-4524`).

The port also names its deviations, and one of them is instructive: `h3d_sol_search`
routes from the query-block **centroid** (`(sum_q q / len) . kc`) where the
reference computes the raw-row mean (`sum_q (q . kc) / len`, `fwd.py:98`). The two
are equal in exact arithmetic and differ in rounding; the port keeps the reference's
own raw-row order as the oracle in `tests/h3/test_native.cu` and measures the
substitution in `tests/h3/sol_route_bm`. A refiner that reuses `h3d_sol_search`
inherits that deviation and its measurement.

Constants match by construction: `H3D_SOL_BLOCK = 64`, `H3D_SOL_GROUP = 32`,
`H3D_SOL_HEAD_DIM = 128` (`h3_dit_gpu.h:48`) against the reference's `BLOCK = 64`,
`GROUP = 32`, head dim 128.

## 5. What the refiner still needs

Nothing algorithmic. In order:

1. **The composition layer.** Route only `attn1` to the sparse path, exactly as
   `stage2/sol_attention.py:3-12` describes: layer 0 dense, layers 1-47 sparse,
   `attn2` and the four audio/AV attentions dense, and no sink (no text span in
   `attn1`). The port needs a dense attention primitive for these shapes either
   way - that is the fallback and the layer-0 path.
2. **The tau path wired.** The production DiT uses `h3d_sol_vsa`; the refiner wants
   `h3d_sol_search` with `tau` from the step index. `h3d_sol_search` exists but is
   not called from Rust today, so this is the first wiring of it - which is also
   what `tests/h3/native` and `sol_route_bm` were built to gate.
3. **The shapes and the layout.** 16,128 tokens x 32 heads x 128, batch 1, from
   `(1, 128, 16, 24, 42)` patchified. The port's layout is head-major
   `[heads][sequence][128]`; the reference's is `[batch, tokens, heads, 128]`, one
   transpose away, and that deviation is already named in the port's comment.
4. **Shape coverage in the gates.** The existing gates were built on the H3 job's
   18,432-token shape; the refiner's 16,128 (a 42-wide axis, 24 rows, 16 frames) and
   the group tail (`16128 = 252 x 64` exactly, so no partial block) should be added
   before the arm is trusted. `16128 / 64 = 252`, `252 / 32` leaves a partial group
   of 28 - the same partial-group path the H3 shape exercises.
5. **The dense/Sol switch and its cost.** The release has no "dense Sol" mode, but
   the port's A/B discipline wants one: `H3_LTX_ATTN=sol|dense` (or the existing
   `SOL_ATTN_STRICT` analogue), so the refiner's quality can be gated against its own
   dense run on the same weights.

Acceptance for that unit: the composition in the host, the shapes in the gates, and
a measured route share (the port already reads back the exact-block share,
`crates/ds4-sys/src/h3_dit.rs:2022`) at tau 1.0/1.25/1.5 - plus the honest
statement that the refiner cannot be *numerically* gated until the prompt cache or
an FP8 H3 DiT exists.

## 6. Corrections this read makes

- **"The port has no sparse-attention path" was wrong** (it was in this session's
  first handoff draft, and in the section 3 of the previous one by implication):
  the sparse path is ported, live and measured on stage 1. The refiner's unit is a
  composition and a shape, not a kernel.
- **Three of the port's citations in this area were off**, and are corrected here:
  the fixed-shape comment cited `preprocess.py:11-13` and `fwd.py:23` where the
  tree at `6e4a440` has `preprocess.py:10-12` (BLOCK_SIZE 64, HEAD_DIM 128) and
  `fwd.py:20-21` (BLOCK 64, GROUP 32); and the forward's deviation note cited
  `fwd.py:129,164` for the two BF16 `P` casts, which are at `:122` and `:166`. The
  rest of the citations in this area land within a line or two of the file. The
  sparse-backend tree was *added* by `6e4a440` (2026-10-28), the revision the
  campaign pinned, so the offsets are citation slips rather than a moved file -
  but the lesson is the campaign's own: re-read the file, do not re-quote the
  comment.
- **The `h3d_sol_search` comment's second half was stale** - it called the sparse
  forward "not built" while `h3d_sol_forward` is live on the H3 path - and is
  corrected with this record. Its first half stands: the tau policy is not yet
  wired to a caller, which is exactly the refiner's unit.
- **The exact-vs-diag choice is a released decision**: the refiner runs the diagonal
  variance model; the full-covariance one is in the source and unused here.
