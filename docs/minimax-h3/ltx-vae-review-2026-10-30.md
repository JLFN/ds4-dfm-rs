# Unit 12c — the decoder's review round and its whole-graph transcription

Continues `ltx-vae-decoder-2026-10-30.md` (12a) and `ltx-vae-spark-2026-10-30.md`
(12b). Landed as `19937e37` on `feat/sol-h3`, pushed to `fork`. This record also
carries the corrections made to 12a and 12b after the same review; those two
records are edited in place, their numbers unchanged.

Two external reviews of unit 12 (Gpt-6-Astra and agy/Claude Sonnet 5.5, both
2026-10-30) returned the same split: several code findings, which `19937e37`
fixes, and several record overclaims, which this commit fixes. The reviews
agreed with the port's arithmetic and disagreed only about the next unit (the
entry point vs the encoder half); 12a's section 3 keeps both.

## 1. The five code findings, all fixed in `19937e37`

Each one is a hazard on an *unhealthy* input or a load this port does not build;
none changes a value on a released input. `cargo test -p ds4-core --lib`: 768
passed, and the `(2, 4, 4)` decode's hash is unchanged.

- **The `to_rgb` clamp is two comparisons, not `fminf`/`fmaxf`.** Both of those
  return the numeric operand when the other is a NaN, so a NaN activation would
  have become 0 or 1 and passed the caller's `RANGE` check; a comparison with a
  NaN is false, so a NaN stays a NaN and the gate fails. `torch.clamp` also
  propagates the NaN, so this is the reference's behaviour, not a stricter one.
- **`silu` takes the live element count, not the scratch capacity.** The
  buffers are sized for the graph's widest stage; the old call walked the whole
  capacity, up to a 2 GiB unwritten tail per silu, per decode.
- **`decode` refuses a shape whose channel count is not 128, and a latent
  tensor smaller than the shape it is told.** Both were implicit before: a
  129-channel or short latent would have indexed past the end.
- **`assert_config` refuses any block-dict key this port does not build**, and
  an exact name match replaces `starts_with`. `residual` is the one that matters:
  `sampling.py:98-122` turns the upsampler into `x + shuffle(conv(x))` with a
  repeat-tile skip, which adds no tensor at all and would survive every shape
  check; `inject_noise` and `timestep_conditioning` were already rejected by
  name. The artifact's own config carries only `num_layers`/`multiplier`, so
  nothing is rejected today - the guard is for the next checkpoint.
- **The gate's decode timer is bounded by a synchronize at each end.** The
  native entries only report launch errors, so a timer that stopped at
  submission did not delimit the decode.

## 2. The whole-graph transcription (`--transcribe`)

The per-op probes in 12a compare the device to a transcription of one op; they
cannot see a same-shaped substitution, a residual added from the wrong
activation, or a norm applied twice, because each probe's output shape is what
the plan already assumes. `--transcribe` is a second *implementation of the
composition*: it reads the artifact's weights through the same plan, walks the
whole graph host-side from the reference's own expressions with each rounding
where the expression has it, and compares element-wise to the device's frame.

Fires at both probe shapes, fresh this commit (workstation, sm_89):

    --decode-shape 1,2,2   values=12288   differ=4828  rel_rms=3.932e-3  worst=0.00586  over=0  8.1 s
    --decode-shape 2,2,2   values=110592  differ=25791 rel_rms=2.798e-3  worst=0.00781  over=0  44.5 s

and on the Spark (sm_121, `~/ds4-test/u12d.log`):

    --decode-shape 1,2,2   values=12288   differ=4277  rel_rms=3.611e-3  worst=0.00391  over=0  10.3 s
    --decode-shape 2,2,2   values=110592  differ=24724 rel_rms=2.753e-3  worst=0.00781  over=0  56.4 s

The verdict is a **band, not an equality**: every op but the convolution is
exact, and a convolution whose f32 sum lands on the other side of a bf16
rounding boundary flips by one ulp, two dozen deep. At 0.5, one bf16 ulp is
`2^-9 = 0.00195`, so the two shapes' worst elements are 2-3 and 4 ulps, and
`over=0` at a ten-ulp tolerance (`TRANSCRIBE_TOLERANCE = 0.02`) on both hosts.
`(2, 2, 2)` is the shape that exercises all three temporal frame drops, which is
where a wrong `causal` policy would show; it does not.

A note on the two hosts, kept honest: the device-to-transcription comparison is
not a hash and never crosses hosts. The `(1, 2, 2)` decode itself does differ
between them (`afb062b13d564a77` on sm_89 vs `fe8ccd7956b14520` on sm_121,
`first`/`last` agreeing to 1 ulp) because the planar matmuls are cuBLASLt and
its sums are not bit-stable across architectures. That is why the transcription
is the gate and the hash is not.

## 3. What the transcription still does not prove

It transcribes the *port's* reading of the reference, so it proves the device
matches that reading, not that the reading is the reference's. The reading's
fidelity rests on the citations in 12a's section 1, and on the config/shape
assertions the loader makes. A value-level reference for the composition - a
real latent, decoded by the reference implementation - remains owed, and it
arrives with the chain's entry point (unit 13) or the encoder half.

## 4. Corrections to 12a and 12b

Both advisors read the same two records and flagged overclaims in the same
sentences. They are corrected in place; the measurements are untouched.

1. **12a, the tiling sentence.** It said "the arithmetic inside every tile is
   identical" and put the difference only in the overlap. Wrong: the release
   computes each tile with a halo of its own zero padding, so a tile's values
   differ from the untiled values even away from the blended zones. Rewritten.
2. **12a, "are the reference's".** The probes compare device to transcription,
   not transcription to reference; the sentence now says exactly that.
3. **12a, the memory-efficient-decode deviation.** "may differ in the last ulp"
   was an untested guess; it is now marked **unverified**, since no measurement
   was made against `memory_efficient_decode.py`.
4. **12a, the padding sentence (39-41).** It conflated `spatial_padding_mode`
   with the temporal halo; the rewritten sentence keeps them separate (temporal
   replication from `causal = False`, spatial zeros from the artifact's config).
5. **12b, "the right shape of wrongness".** Demoted to a report: a pleasing
   frame would not have proved the port, and a noise field does not either.
6. **12b, "the hosts agree bit for bit".** Bounded to the probed ops at their
   probe sizes (24-384 values), with a pointer to the 1-2 ulp whole-graph claim.
7. **12b, the encoder-half round trip.** Softened from "a value-level proof of
   the whole pair" to a check of the pair, with the compensating-error caveat
   that is Astra's reason for preferring the entry point.
8. **12a section 3.1 and 12b section 4, the impulse explanation.** The damped
   centroid was explained as a near-global receptive field. The radius is
   structural (about twelve cells, from the convolution count); the measured
   spread is a centroid statistic, not that radius, and the probe sits three
   cells from the frame edge, so boundary truncation confounds it. Both records
   now say so, and name translation equivalence - shift a cell away from the
   borders, the response should shift by exactly one cell - as the interior form
   (agy's suggestion).
9. **12b section 3, block independence.** Pixels one block apart are not
   independent; the convolutions between the shuffle and the output overlap.
10. **12b section 5, the footprint.** The host-wide `free` rise was attributed
    to the decoder. It is a system-wide measurement; the remainder beyond the
    working set and weights is unattributed.

## 5. Guards

- `cargo test -p ds4-core --lib`: 768 passed, 7 ignored (section 5 of 12a on
  the fd-3 caveat).
- `tests/h3_ltx_vae_gate --probe`: 8 ops + 2 controls, unchanged by this commit
  (the diff touches `to_rgb`'s clamp expression and `silu`'s count only, and the
  `to_rgb` probe stays `differ=0`).
- The transcription runs at both probe shapes; the `(2, 2, 2)` shape guards the
  three temporal frame drops.

## 6. Reproduction

    make CUDA_ARCH=sm_89 tests/h3_ltx_vae_gate          # or sm_121 on the Spark
    ./tests/h3_ltx_vae_gate --probe
    ./tests/h3_ltx_vae_gate --vae <vae.safetensors> --decode-shape 1,2,2 --transcribe
    ./tests/h3_ltx_vae_gate --vae <vae.safetensors> --decode-shape 2,2,2 --transcribe

Spark: `~/ds4-test/u12d-spark.sh`, log `u12d.log` (probes, both transcriptions,
and the three release-geometry decodes reproducing
`5ad08268f3b2acee` / `9be00ee0f10f4241` / `14a8e079577e3255`).
