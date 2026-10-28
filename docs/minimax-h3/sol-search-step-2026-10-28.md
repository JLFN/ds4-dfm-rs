# Sol-Attn's search step: ported, tested, measured, 2026-10-28

This unit ports the routing half of Sol-Attn — the decision that picks which KV
blocks attend exactly and which are approximated from their pooled summaries —
into `native/h3/h3_dit_gpu.cu`, and gates it. It is the first piece of the
sparse path the [recon](sol-h3-spark-recon-2026-10-28.md) names as portable.

What it is not: there is no sparse forward yet. The mask this unit produces has
no consumer in this engine, so nothing here changes an inference path, a
default, or an existing gate. The port is scoped to the reference's own step
boundary: pool the summaries, compute the threshold, route. That is what
upstream calls before its mainloop, and it is independently verifiable because
its output is discrete.

Branch `feat/sol-h3` off `17e87b81`; `native/h3/h3_dit_gpu.cu` md5
`c633b3ce567394a014a7e5b3b6c67185`, `tests/h3/test_native.cu` md5
`bbd6fa59dea7d4984032b90453ca347d`, `tests/h3/sol_route_bm.cu` md5
`133ba8aa226dc49aa0f10de515463562` (identical on the Spark and the
workstation). Spark binaries: `tests/h3/native_test` sha256
`e0b9f401717566d3b2f4adfd73c4da50bea001da475571b8775e669d695c8695`,
`tests/h3/sol_route_bm` sha256
`c69a32012e8220aa255c833819731756c43a5a38f4d74213a4738294c536f6f9`.

## The reference rule, cited

NVIDIA Sana `sol-engine` @ `6e4a440`, `techniques/sparse_backends/sol_attn/`:

1. `preprocess.py:280-322`, `triton_ref/preprocess.py:34-68`: per 64-token KV
   block and head, `kc` is the block's mean K row and `vc` its *summed* V row,
   both stored BF16. The last block divides by its own length, not by 64.
2. `preprocess.py:96-135`: `kc_mean` and `kc_var` (the diagonal variance), per
   head, over the blocks, read back from the BF16 `kc`.
3. `preprocess.py:143-186`: per (query block, head),
   `threshold = q_bar . kc_mean * log2_scale + tau * sqrt(max(q_bar^2 . kc_var
   * log2_scale^2, 0) + 1e-6)` with `q_bar` the query block's mean row and
   `log2_scale = scale * log2(e)`.
4. `triton_ref/fwd.py:250-266`: `col_mean[n]` is the column mean of
   `(q @ kc[n]^T) * log2_scale` over the query block's rows.
5. `selector.py:132-142`, `interface.py:134-142`: a block is exact iff
   `col_mean[n] > threshold` or it lies within one block of the query block,
   within the sequence; the sink block range is forced exact.
6. `models/minimax_h3/Sol-H3-Spark/runtime/stage1_ops/sol.py:12-18`: tau is
   `None` on step 0 and layer 0, then `1.0`, `1.25`, `1.5` on the three later
   steps; the sparse call is `thresh_type="diag"`, `kv_splits=1`, with a sink
   range covering the text and audio rows.

## The port

Four kernels and one entry point, `h3d_sol_search`, with the plan's buffers
allocated and shape-checked by the caller (`struct h3d_sol_plan`).

| Kernel | Grid | Work |
|---|---|---|
| `h3d_sol_pool_kernel` | (blocks, heads), 128 threads | `kc`, `vc`, one thread per dimension, sequential over the block's rows so the sum order is the reference's row order |
| `h3d_sol_stats_kernel` | (heads), 128 threads | `kc_mean`, `kc_var` per head |
| `h3d_sol_threshold_kernel` | (query blocks, heads), 32 threads | `q_bar`, `threshold` |
| `h3d_sol_route_kernel` | (query-block tiles of 64, heads), 32 threads | the mask and its count |

The route kernel is the one with a design choice. A CTA covers 64 query blocks
of one head, one lane per KV block of the current 32-block mask word, and the
word's `kc` rows staged in shared, so each summary is read once however many
query blocks the CTA covers. The column mean is the query block's centroid dot
the summary, which is the reference's own sum:

    sum_q (q . kc) / len  =  (sum_q q / len) . kc  =  q_bar . kc

That is the same number for 64x fewer multiplies, and it puts both sides of the
comparison on one F32 centroid: the threshold's mean term is `q_bar . kc_mean`
and the route's column mean is `q_bar . kc[n]`. The reference's raw-row order is
kept as the oracle in `tests/h3/test_native.cu`, which computes both forms and
reports what the substitution moves.

Deviations, named:

- Head-major `[heads][sequence][128]` tensors with batch 1, the port's layout
  for every H3 attention primitive, against the reference's `[B, T, H, 128]`.
- `query_bar` is F32, which is what the reference's diagonal path already is:
  `_diag_threshold_kernel` builds its centroid as
  `tl.sum(q_values.to(tl.float32), axis=0) / q_len` (`preprocess.py:161-170`),
  so this is not a deviation. The BF16 `q_bar` at `preprocess.py:401-405`
  belongs to the separate `thresh_type="exact"` path (`:383-410`), which this
  port does not implement. An earlier draft of this file cited that BF16
  `q_bar` as the reference's form and claimed the F32 store avoided a
  BF16-to-F32 deviation; both halves of that were wrong, and the Astra review
  caught it.
- The accumulation orders are this port's. The reference's Triton reduction
  trees are not reproducible here without its runtime; the pooled summaries
  match a sequential-order transcription bit for bit, and the threshold's own
  order sensitivity is measured below rather than asserted.
- The reference's column-mean order is not an engine arm: a faithful standalone
  version needs the reference's tensor-core `q @ kc^T` tile, and the centroid
  form is the same decision for 64x fewer multiplies. Its row-order form stays
  where it can prove something, in the oracle.

## Tests

`tests/h3/test_native.cu`: nineteen check names across nine cases, 52 checks,
added to the existing suite. The oracle is a transcription of the six cited
steps, including the row-order column mean.

| Check | What it proves |
|---|---|
| `search pools kc and vc exactly as the reference` | bit equality of both summaries, so the BF16 store, the block-length divisor, the tail block and `vc` being a sum are all right. The synthesized V is an order of magnitude above K, so a summary that is a mean where the reference takes a sum cannot compare equal |
| `search threshold matches the reference's formula` | worst relative delta 2.6e-7 to 3.9e-7 over four tau/case combinations |
| `search routes the reference's exact block set` | the mask equals the oracle's rule applied to the reference's row-order column mean |
| `search counts the reference's exact blocks` | the count equals the oracle's |
| `the centroid order cannot move a route wider than the tightest margin` | the two column-mean orders differ by 1.9e-6 to 3.3e-6 while the tightest threshold margin on the same data is 0.17 to 0.36 |
| `search never routes a block past the sequence` | the reference's `valid` term |
| `search count is the mask's own population` | the count and the mask agree |
| `search routes every sink block exactly` | the sink range |
| `constant summaries route the diagonal band and nothing else`, `constant risk threshold matches its closed form` | arithmetic, not a second transcription: constant K and Q have no variance, so the threshold is `mean + tau*sqrt(1e-6)` and only the diagonal band can be exact. The threshold matched its closed form to 0.000e+00 |
| `a spiked summary is routed exactly and no other far block is` | one block's summary spikes in dimension 0 with Q one-hot in that dimension: exactly that block plus the band |
| `raising tau only removes exact blocks` | a tau sweep with no oracle at all: 7 bits removed, 0 added |
| `an empty sink range forces no block: the band is the whole route`, `an empty sink range routes exactly as no sink at all` | `_sink_block_range`'s zero case: zero `sink_tokens` is the empty range, never a forced block 0. Both assertions failed before the fix, which is the point of them |
| `the search step refuses a wrapping sink or shape` | `sink_start + sink_tokens` wrapping to a small number, a sink longer than the sequence, and a `sequence` whose block rounding overflows are all refused before a launch. It fails against the old comparison and passes against the difference |
| `the multiword fixture is 70 blocks over three words` and the two `multiword` comparisons | a second mask word, a second query CTA and a 30-row final block against the oracle at two taus; tightest margin 3.747e-03 against a 5.245e-06 order delta |
| `a column mean equal to the threshold is not routed` | the strict `>`: with Q one-hot on dimension 0 and summaries [1, 0, 3, 0] there, block 0's column mean equals the threshold exactly at tau=0 and is not routed |

The two analytic cases are the ones that catch a wrong formula rather than a
wrong transcription: the constant case pins the `+1e-6` floor and the strict
`>` comparison, the spike case pins the mean/std balance.

Results, identical on both hosts and on both architectures: `tests/h3/native_test`
73 PASS / 0 FAIL — the 40 pre-existing checks plus 33 from this unit.
Model-free Rust gates also clean: `cargo fmt --check` (no Rust touched),
`cargo test -p ds4-core --lib` 696 passed with `3<&-` (the harness's own fd 3
breaks `mux_pipe_lifetimes`), and `tests/h3-test_host_portable`,
`tests/h3-test_av_mux`, `tests/h3-test_tokenizer_portable` all ok.

## Measurements

`tests/h3/sol_route_bm` at the two denoise shapes, on the Spark, sm_121a,
CUDA 13.0.88, driver 580.178.04, event-timed over back-to-back launches. The
dense attention in the last row is `h3d_sdpa` on the tensor arm at the same
shape, the number the search step has to be cheap against.

| Kernel | 18,432 x 56 x 128 | 4,352 x 56 x 128 |
|---|---|---|
| pool (`kc`, `vc`) | 2.493 ms | 0.592 ms |
| `kc` statistics | 0.008 ms | 0.004 ms |
| threshold | 1.323 ms | 0.323 ms |
| route | 0.852 ms | 0.126 ms |
| search total | 4.924 ms | 1.158 ms |
| dense attention, same shape | 173.163 ms | 9.935 ms |
| search share | 2.8% | 11.7% |

Two runs at the job shape, 3 and 5 launches each, agree to the figure:
pool 2.488 / 2.493 ms, statistics 0.008 / 0.008, threshold 1.327 / 1.323,
route 0.856 / 0.852, total 4.924 / 4.924, dense attention 173.097 / 173.163.
The 512-shape run reported pool 0.592 ms, statistics 0.004, threshold 0.323,
route 0.126, total 1.158 ms.

Where the time goes, against the ceilings this campaign measured:

- pool: 528.4 MB of K and V (252 MiB = 264.2 MB per tensor) in 2.493 ms,
  212 GB/s, 89% of the 236.8 GB/s DRAM ceiling. It is a streaming kernel and
  nothing else is left in it.
- threshold: 264.2 MB of Q in 1.323 ms, 200 GB/s, 84% of the same ceiling. The
  query centroid needs every query row once, so this read is inherent to the
  step.
- route: not bandwidth-bound at all. Each of the 5 query tiles stages all 288
  `kc` rows once, so its global reads are 5 x 4.13 MB = 20.6 MB, which at
  0.852 ms is 24 GB/s. What it does instead is 8,192 shared-memory summary
  reads per lane per mask word against 64 staging pieces, so it is instruction-
  and bank-bound inside the SM - and the padding of that shared array decides
  how many wavefronts those reads cost (see the A/B below).
  An earlier draft of this file claimed 1.19 GB of L2 traffic and an L2 bound
  at 1.4 TB/s; that figure counted repeated reads of the *shared* copy, not
  global traffic, so it was wrong. The Astra review caught this one too.
- statistics: 4.13 MB in 8 us, 516 GB/s, above the DRAM ceiling because the pool
  kernel wrote that tensor a moment earlier.

So the step is streaming-bound, and its floor at this shape is the 792.7 MB of
K, V and Q it must read once: 3.35 ms at 236.8 GB/s against the 4.9 ms measured.
A per-request figure
for a Sol-H3-shaped Stage 1 (147 sparse calls) is therefore about 0.72 s of
routing against the 25.4 s the dense attention it routes would take at 18,432
tokens. The model's own Stage 1 runs at 672x384x124, a different token count
that this bench does not cover.

The bench's synthetic logits route 47.2% of query-block/KV-block pairs exact,
52.8% skipped, at the job shape. That share is a property of the data, not of
the reference: upstream's own sparsity figures come from other selectors (VSA's
90% video-tile top-k in Stage 1, Sol-Attn's tau schedule here), and only the
model's own logits decide the real share.

Cross-host note: on identical inputs the route differs by 2 of 4,644,864
decisions between the workstation's sm_89 build and the Spark's sm_121a build.
That is the FP order of the two compilations against pairs sitting within
1e-6 of the threshold, which is the count the noise predicts (4.6e6 pairs,
~2e-6 absolute column-mean order difference, margins spread over ~3). Any
route A/B must therefore compare two runs on one host.

## The route kernel's bank conflict — predicted, then refuted (2026-10-28)

The Astra review of this unit argued from the source layout that `s_kc` reads
carry a four-way shared-memory conflict: a row is `H3D_SOL_KLD = 136` bf16, so
its stride is 68 words, and lane `l` reads word `l*68 + dim/2`; 68 mod 32 = 4,
so lanes eight apart share a bank, over the 8,192 summary reads each
query-block tile makes per mask word.

Two changes were made to test that, and both were measured on the Spark against
a baseline binary kept from the same tree, interleaved in one session at a
steady 2,444 MHz with five back-to-back launches per figure, at 18,432 tokens x
56 heads:

| Arm | KLD | staging | route, round 1 | round 2 |
|---|---|---:|---:|---:|
| baseline | 136 (68 words) | uint4 | 0.851 ms | 0.853 ms |
| control | 136 (68 words) | uint | 0.929 ms | 0.923 ms |
| conflict-free | 130 (65 words, odd) | uint | 0.964 ms | 0.936 ms |

The conflict-free arm is 11-13% **slower**, and the control separates the two
things it changed: with the stride put back to 68 words and only the staging
widened to four-byte pieces, it still costs 0.923-0.929 ms. The staging width
is therefore the whole regression, and the stride moves nothing measurable.

That refutes the review's bank-conflict prediction as a performance claim: at
this shape the route kernel's time is not made of conflicts on `s_kc`. The
reading that fits the three arms is instruction-bound — uint4 staging emits 16
shared stores per lane per mask word against 64 for four-byte pieces, and the
wider rows stay 16-byte aligned, so `ptxas` keeps issuing wide shared accesses.
Both arms are reverted (`ec2d3751` reverts `2dc880cc`) and the baseline is
retained. The route masks were bit-identical in all three arms: the suite
compares whole mask words against the row-order oracle, and every arm reports
92 PASS / 0 FAIL.

What is still unmeasured is the counter view. `ncu` cannot read performance
counters on the Spark without operator action — `ERR_NVGPUCTRPERM`, no
passwordless sudo, and the restriction is the driver default — so the
attribution above is inference from a controlled A/B, not a counter
measurement, and is worth no more than that.

## What this unit does not establish

- No reference run. The rule is checked against a transcription of the
  reference source, not against NVIDIA's Triton or CuTe kernels, which need the
  pipeline of the recon's Phase 2. Until Phase 3 runs, "matches the reference"
  means "matches its source", and the recon's own Phase 4A spec read is still
  open for `sm120/mainloop.py` and `sm90/exact.py`.
- No sparse forward. The mask has no consumer; the approximate path's cost per
  skipped block (one 128-dimension dot against the pooled summary plus the
  length-weighted softmax update) is unaudited, and it decides whether the
  sparse path pays at all. The search step is 2.8% of the dense attention at
  the job shape, so routing is not the prize — the skipped blocks are, and this
  unit prices only the routing.
- No profile. The per-kernel figures are event-timed back-to-back launches, and
  no counter capture exists for any of the four kernels: `ncu` cannot read
  performance counters on this host without operator action (see below). One
  nsys capture over the step would attribute the pool's 212 GB/s and the
  route kernel's instruction stream rather than infer them from byte counts
  and a controlled A/B.
- The host's clock policy changed since the campaign (below), so these absolute
  figures are not directly comparable to `speed-2026-10-27.md`'s.

## Host state, and a clock finding

Spark `gx10-e1d2`: CUDA 13.0.88, driver 580.178.04, `sm_121a`, built with
`rm -f native/h3/*.o && make tests/h3/native_test tests/h3/sol_route_bm`.

`nvidia-clocks.service` does not exist on this host:

```
$ systemctl status nvidia-clocks.service
Unit nvidia-clocks.service could not be found.
$ systemctl is-enabled nvidia-clocks.service
not-found
```

The clock policy the repository requires for benchmarks (300-2200 MHz, the
range `docs/gb10-hardware.md` records) is therefore not applied here, and
`nvidia-smi` reports `clocks.max.sm` 3003 MHz with 2424 MHz observed during
this unit's runs. Two consequences: the absolute figures above are taken at a
higher clock than the campaign's (2184-2190 MHz busy) and should be re-taken
under the policy before any adoption decision; and the reported DGX Spark trip
risk that the policy exists to reduce is unmitigated on this host right now.
The measurement is still internally consistent: the dense attention costs
173.163 ms per call here against the campaign's ~187 ms, and 173 x 2424/2184 =
192 ms, inside the campaign's own repeat range. Fixing the policy is the
operator's call — this unit neither started nor reset anything.
