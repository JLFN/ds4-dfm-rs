# The sparse attention reaches a render (2026-10-28)

Branch `feat/sol-h3`. Units: `h3d_sol_vsa`, the reference's own selection rule
for the H3 stage (commit `95bdd896`), its instruction-level overhaul
(`8617afdb`), and the DiT arm that makes it reachable from a render
(`8027bdf2`). All times on the DGX Spark, `sm_121a`, CUDA 13.0, busy clocks
2.4-2.5 GHz - **the host has no clock policy**, so these are not the campaign's
2184-2190 MHz regime and no absolute figure here is comparable to
`speed-2026-10-27.md`'s. `ncu` is still blocked by `ERR_NVGPUCTRPERM` on that
host, so every attribution below is a timing inference, not a counter.

## 1. What the selector is, and why it is a different rule

The tau router the previous unit measured is Sol-Attn's - the reference's **LTX
refiner** stage. The stage this engine actually serves, the 50-block H3 DiT,
selects `VSA_cuDNN_BSA` at sparsity 0.9, tile 64, prefix exempt
(`Sol-H3-Spark/configs/default.json:12`) - a per-row top-k of the video tiles.
The rule and its provenance are recorded in `vsa-policy-2026-10-28.md`; this
record is its port, its price and its first render.

`h3d_sol_vsa` reuses the tau path's pooling, statistics and centroid kernels
unchanged and adds one selection kernel. The `[heads][blocks][blocks]` score
matrix the reference materialises (`video_sparse_attn_h3.py:629`) never exists:
one CTA per (query block, head), one warp, each column's score computed in
shared from `query_bar` and its `kc` row, then ranked against the kept set. A
column is exact when fewer than `keep` video columns outrank it, ties going to
the lower index.

Two deviations, both named in the header:

- `torch.topk`'s tie order is undocumented, so ties go by ascending column
  index here.
- The reference's tiles are 3D `(4,4,4)` tokens in a reordered sequence; this
  port's blocks are linear 64-token spans of its own head-major sequence.
  Whether the two groupings coincide is a property of the engine's layout and
  has not been checked, so a VSA mask over this port's blocks is a *valid*
  sparsity pattern, not proven to be the reference's.

## 2. What the gate holds, and what it cost to hold it

`tests/h3/native_test`: **124 PASS / 0 FAIL** on `sm_89` and on `sm_121a`, up
from 102. Six new cases - the sparsity sweep, the reference's own setting at
18,432 tokens, a keep of one, an all-tie fixture, sparsity monotonicity and the
API boundary - gated against a CPU transcription of `_build_block_mask`
(`:387-410`) and `compute_topk` (`video_sparse_attn.py:161-163`) applied to the
summaries the port's own pooling wrote, so the selection is gated apart from the
pooling.

Mutation checks: letting the prefix columns compete in the rank fails 14
checks; widening the keep set by one (`rank < keep` -> `rank <= keep`) fails 17.

Two findings the gate itself produced:

- **The sparsity must cross the ABI as a double.** 0.9 rounded to F32 turns
  `ceil((1 - s) * 280)` into 29 where the reference's own Python double gives
  28. The job-shape case caught it as a keep-count failure before it could
  become a mask that is quietly one block denser than the reference's.
- **The mask is a discrete function of the scores**, so the oracle has to score
  with the kernel's own reduction tree (four dimensions per lane, then the
  butterfly offsets 16/8/4/2/1). Scoring with a sequential 128-dimension dot
  flipped one column's rank at the keep boundary at 18,432 tokens. What the F32
  order moves *against the reference's raw-row order* is a deviation to measure,
  and `sol_vsa_bm` reports it instead: **0 of 49,536 mask bits differ** (18,432
  tokens, 4 of 56 heads sampled) and 0 of 5,536 at 4,352. The centroid
  substitution moves nothing measurable on these fixtures.

## 3. Performance

`tests/h3/sol_vsa_bm` (new target), three to five back-to-back calls each with
one sync. "Selector" is the whole VSA search step minus the tau path's own
search step: both launch the same three prefix kernels, so the difference is the
selection kernel.

| arm | 18,432 x 56 | 4,352 x 56 |
|---|---:|---:|
| dense `h3d_sdpa` | 173.7 ms | 9.82 ms |
| tau search (threshold) | 4.87 ms | 1.15 ms |
| VSA search (top-k) | 10.49 ms | 1.36 ms |
| selector, first version | 5.62 ms | 0.21 ms |
| sparse forward after the mask | 27.47 ms | 3.45 ms |
| realized exact share | 14.93% | 29.93% |
| layer total, first version | 37.96 ms (4.58x dense) | 4.81 ms (2.04x) |

The first version's selector was issue-bound: one warp per CTA leaves little to
hide a dependent chain behind, and both the per-column shuffle reduction and the
rank accumulation are serial. Two semantics-preserving changes - two columns
scored per warp pass, and the rank summed in four independent counters - brought
it down and were adopted on their own A/B:

| arm | before | after |
|---|---:|---:|
| VSA search, 18,432 x 56 | 10.49 ms | 7.49 ms |
| selector over tau search | 5.62 ms | **2.61 ms** |
| layer total, 18,432 | 37.96 ms | **34.90 ms (4.93x dense)** |
| VSA search, 4,352 x 56 | 1.36 ms | 1.16 ms |
| selector over tau search | 0.21 ms | 0.006 ms |
| layer total, 4,352 | 4.81 ms | **4.62 ms (2.14x dense)** |

The mask is unchanged by construction and by measurement: same realized share,
same zero-bit deviation.

The forward's own cost model still holds exactly at the reference's setting:
`0.1493 x 150.8 ms + 4.5 ms = 27.0` measured 27.47. The remaining selector cost
is the rank pass re-reading the column scores once per mask word at one warp per
CTA; a CTA covering several query blocks per head would cut both that and the
`kc` re-read, and is the next candidate.

## 4. The DiT arm

`H3_DIT_SOL=<sparsity>` routes the DiT's full attention at `dit.rs:2424` through
the selector and the forward; unset, `h3d_sdpa` is untouched. It is an
environment diagnostic, not a wire field and not a default - the sparse path is
an approximation of the engine's own, and the campaign's rule is that arms are
switchable and never silently adopted.

Two guards, neither exercised by a test yet: the arm is refused when token
pairing is on (halved video rows move every block boundary, so the plan's block
count and prefix span are not the pairing layout's), and it is skipped if the
row count at the call site is not the one the plan was sized for.

The prefix is `video_target_start` rounded **up** to whole 64-token blocks: a
block holding a prefix row is treated as prefix, which is the conservative
reading of a span the reference keeps tile-aligned. At the probe shape that is
2 blocks of 9.

One line per process reports the first sparse call's realized share, read back
from the mask:

    533 rows, 56 heads, prefix_blocks=2 -> exact share 0.4815

## 5. The first renders

`h3-rs`, one binary, one host, three renders per shape with the default repeated
between the arms (run order moves a phase by more than a small arm does). The
dense arm is byte-reproducible: `dense-1` and `dense-2` produced the same MP4
sha256 (`c193dab69505...32177`).

### The probe shape, 256 x 256, 22 frames, 2 steps, seed 1

| run | TEXT | LOAD | DENOISE | AUDIO | FRAMES | VIDEO | wall | MP4 sha256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dense-1 | 27.485 | 7.304 | 0.682 | 0.638 | 3.422 | 0.190 | 40 s | `c193dab6...32177` |
| sol-0.9 | 27.393 | 7.207 | 0.687 | 0.641 | 3.406 | 0.196 | 41 s | `ca7f44e4...db3c42` |
| dense-2 | 27.344 | 7.285 | 0.694 | 0.642 | 3.431 | 0.160 | 40 s | `c193dab6...32177` |

The arm costs nothing and saves nothing here: the whole denoise is 0.68 s of a
40 s render, so attention is invisible. What the run did produce is the
model's own realized share, and it is not the fixture's:

    prefix_blocks=2, 533 rows, 56 heads -> exact share 0.4815

Nine blocks, seven of them video, and `keep = max(1, ceil(0.1 x 7)) = 1`: two
prefix blocks plus one video block of nine is 33% on every video row and 100% on
every prefix row, 48.15% overall. The rule is a count over tiles, so at nine
tiles it cannot express 90% sparsity however the scores fall. Nothing is wrong
with the port; the shape is simply too small for the policy to bite, and it is
the reason a small render cannot validate this arm.

### The job shape, 768 x 768, 107 frames, 2 steps, seed 42

| run | TEXT | LOAD | DENOISE | AUDIO | FRAMES | VIDEO | wall | MP4 sha256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dense-1 | 27.371 | 7.429 | 40.153 | 2.563 | 196.234 | 0.592 | 275 s | `c37882cf...71c958` |
| sol-0.9 | 27.298 | 7.594 | **25.809** | 2.589 | 197.628 | 0.590 | **263 s** | `d7080865...cafa5cb` |
| dense-2 | 27.281 | 8.171 | 40.699 | 2.605 | 197.444 | 0.586 | 277 s | `c37882cf...71c958` |

The denoise phase is the only one the arm touches, and it falls **14.6 s
(36%)**: 40.153 and 40.699 dense against 25.809. The kernel numbers predict
`100 calls x (173.7 - 34.9) ms = 13.9 s`, so the end-to-end phase agrees with
the bench to within 5%. The wall falls 275/277 -> 263 s, **-4.6%**, because
FRAMES (196-198 s) is untouched by any attention lever.

The arm also reported the model's own routing at this shape:

    prefix_blocks=6, 18799 rows, 56 heads -> exact share 0.1370

294 blocks of which 6 are prefix, `keep = ceil(0.1 x 288) = 29`, so
`(6 x 294 + 288 x 35) / 294^2 = 0.1370`. The synthetic fixture at the same keep
gave 14.93%, so the *scores* did not have to be the model's for the share to be
close - but the fixtures differ in block count, so this is agreement, not proof.

The row count matters: the attention call runs **18,799** rows, not the 18,432
the campaign's per-step law quotes. The 2% difference is inside that law's own
11% fit, and the bench shape stands, but the exact figure is 18,799 and the
prefix at this shape is 367 rows (6 blocks).

## 6. What this does not establish

- No counter attribution: `ncu` remains blocked on the Spark, so "issue-bound"
  is inferred from the shape of the kernel and the A/B, not observed.
- The realized share is still a property of the input: 48% at the probe shape
  and 14.93% at the job shape come from *different* block counts, and the rule
  is coarse where blocks are few.
- The `(4,4,4)`-against-linear-block question is unanswered, so the mask is not
  proven equivalent to the reference's.
- No weight-level lever (LoRA, FP8, NVFP4) is in this line, and the reference's
  own 56.17 s is a different job with a different model set.
