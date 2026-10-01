QA report: feat/image — the P1 oracle's DiT forward (Qwen-Image-2.1, stage 2),
final re-verification of the amended unit. Independent falsification pass,
rule 19. Unit: commit bc23830 "feat(image): add the P1 oracle's DiT forward"
(amended from e7becd8, 84d2d4d, b3b5c79), branch feat/image, one commit ahead of
fork/feat/image = 6d613d6. Refs inspected: bc23830, e7becd8, 84d2d4d, b3b5c79,
6d613d6, the pinned reference copies, and upstream leejet/stable-diffusion.cpp at
6dcb5bb. Nothing was committed or pushed; the only file written is this report.

Verdict: PASS. Both remaining findings are fixed and I reproduced them live: all
four section-5 commands now run verbatim from the repo root with the annotated
outputs, and the "nothing runs" sentence is gone. Every measured claim in
docs/qwen-image-2.1-p1.md and in the commit message checks out against my own
measurement or re-derivation. A short list of minor, non-blocking imprecisions
is recorded below as findings (an undefined shell variable in the reference-run
block, and three rounded/representative figures); none changes a result or
misleads a reader about one, so none is a defect.

Covered surfaces (each on its own line, exact string from the gate)

crates/ds4-core/src/qwen_image/dit.rs
crates/ds4-core/src/qwen_image.rs
crates/ds4-core/tests/qwen_image_oracle.rs
docs/qwen-image-2.1-p1.md
pub const Q6K_BLOCK_BYTES
pub const QK_K
pub enum DitError
pub fn cfg_combine
pub fn dequantize_f32
pub fn forward
pub fn linear
pub fn open
pub fn parity
pub fn token
pub struct DitPass
pub struct DitWeights
pub struct Parity

All seventeen present (dit.rs lines 37, 39, 275/288, 331, 337, 388, 415, 459,
710, 840, 846, 856; qwen_image.rs "pub mod dit;"; the test and doc). Amendment
scope: git diff --stat 84d2d4d bc23830 is "docs/qwen-image-2.1-p1.md | 26 ...,
1 file changed, 13 insertions(+), 13 deletions(-)" and nothing else; git diff
84d2d4d bc23830 -- crates/ is empty, so the code and tests are byte-identical to
the first reviewed tree.

Findings D and E re-verified.

Finding D (doc commands unusable) FIXED. Section 5 now defines
D=/data/ds4-dfm-rs/misc/scratch/p1/refdump/run1 and passes the absolute path. I
ran all four documented commands verbatim from the repo root:
  cargo test -p ds4-core --lib qwen_image::oracle              -> 12 passed  (# 12 passed)
  cargo test -p ds4-core --lib qwen_image::dit                 ->  9 passed  (# 9 passed)
  D=/data/ds4-dfm-rs/misc/scratch/p1/refdump/run1;
  DS4_QWEN_IMAGE_ORACLE=$D cargo test -p ds4-core --test qwen_image_oracle
                                                               ->  9 passed  (# 9 passed)
  DS4_QWEN_IMAGE_ORACLE=$D DS4_QWEN_IMAGE_DIT=<gguf> \
    cargo test -p ds4-core --release --test qwen_image_oracle dit_forward_reproduces
                                                               ->  1 passed (the DiT gate)
The gate printed "interleaved: correlation 0.999973, relative RMS 7.5578e-3,
max |diff| 1.3715e0, 119.3s" and "half-split: 0.742873, 6.6874e-1, 3.0159e1,
115.6s". The prose now also states correctly that cargo runs the test binary
from the package directory, which I independently confirmed: with
DS4_QWEN_IMAGE_ORACLE=../../misc/scratch/p1/refdump/run1 the suite passes 9;
with the repo-root-relative form it fails.

Finding E ("Nothing runs without the environment") FIXED. Section 5 now says
"two are model-free and always run". Verified: with no environment variables
cargo test -p ds4-core --test qwen_image_oracle runs 9 tests, 9 passed, and the
two model-free tests (noise_scaling_is_identity_at_the_first_sigma,
philox_is_seed_and_call_position_dependent) execute rather than return early.

Live measurements (release gate; the independent numpy port run on the same
DS4_QWEN_IMAGE_DIT_OUT dumps)

  comparison                              corr        rel RMS      max |diff|
  rust cond   vs numpy cond               1.0000000   1.1948e-06   2.3961e-05
  rust uncond vs numpy uncond             1.0000000   4.1175e-07   2.8610e-06
  rust cond   vs refdump/cfg1 pred        0.9999874   5.1500e-03   2.2745e-01
  numpy cond  vs refdump/cfg1 pred        0.9999874   5.1504e-03   2.2745e-01
  rust cfg    vs refdump/run1 pred        0.9999732   7.5578e-03   1.3715e+00
  numpy cfg   vs refdump/run1 pred        0.9999732   7.5583e-03   1.3715e+00

The doc's and the commit's 1.2e-6, 4.1e-7, 1.0000000, 2.4e-5, 5.2e-3 and 7.6e-3
all reproduce. The argument holds: the residual (5.15e-3) is ~4.3e3 times the
implementation-to-implementation F32 agreement (1.19e-6).

Stage-1 numbers re-derived this pass (code unchanged): least squares
sum((x1-x0)*p0)/sum(p0^2) = -0.377540647882 (doc -0.377540648); the reference's
Euler chain rebuilds step2.in.bin with 0 of 16384 differing while the
one-expression form differs in 692; sigma1 = 0.622459352016449; the token ids
behind the two dumps are 29 and 23 (run1.log:190, :221); the Box-Muller table is
exact - all-F32 11242 byte-identical with ulp distances {1: 4774, 2: 360, 3: 8}
(total 5142), double-throughout 12136, and the mixed form the port mirrors
16384/16384.

Other claims re-verified

- Artifact hashes (doc section 6): qwen-image-2.1-Q6_K.gguf
  a3a0d39b...4fb9, qwen_image_2.1_vae_bf16.safetensors bb21f747...6b7c9, and
  misc/scratch/p0/vae-decode-bf16.gguf d3feefed...5372 all match the doc and
  misc/scratch/p1/p1-artifact-hashes.txt.
- Dequantizers: to_q (Q6_K) max|diff| 0.0 against gguf.quants.dequantize; the
  doc's "maximum difference of zero" holds. Q6_K arithmetic matches ds4.c:4538
  and the layout ds4.c:1163-1168.
- Layout conventions and reference semantics are unchanged from the first pass
  (crates/ byte-identical) and still check out against the pinned sources.
- The doc's scope claims hold: forward() refuses reference latents with
  DitError::Unsupported, and stage 3 (VAE/PNG) and any GPU path are excluded.
- Commit message: the removed "stage by stage / 1e-6 or less at every stage"
  and F16 claims are absent; the trailer "Unit: 5 complete" is the last
  non-empty body line.
- Regression: cargo test -p ds4-core --test tokenizer still fails
  tokenizer_families_match_c_oracle (qwen35 family 12 vs C oracle 11) at
  crates/ds4-core/tests/tokenizer.rs:174; the amendment touches no tokenizer
  file, so it is pre-existing and not this unit's regression.

Minor, non-blocking imprecisions (findings, not defects)

- docs section 1's reference-run code block uses $D (for the dump prefix and the
  output PNG) without defining it; the intended directory is named in the
  sentence that follows. The block is illustrative (it also needs the
  proprietary sd-cli and internal model paths), so it does not block anything.
- "5.6 GiB" (commit message; dit.rs's module doc says the same): the artifact is
  5876556448 bytes = 5.47 GiB. A rounding, ~2% high.
- "dequantizes one Q6_K row at a time into a 16 KiB buffer" (commit message):
  16 KiB is the 4096-wide row; the 12288-wide img_mlp.out rows are 48 KiB. The
  bounded-buffer claim holds; the figure is representative.
- Section 5's "the fixture prefix must be an absolute path": a path relative to
  the package directory also resolves; absolute is the practical instruction.

Unverified items (findings, not passes)

- The doc's claim that the deployed sd-cli's SD_DUMP_* hooks are absent from
  upstream at 6dcb5bb: I confirmed the pinned reference copies contain no
  SD_DUMP string (and existence of the dumps), not the deployed binary itself.
- The reference-internal line citations (diffusion_engine.cpp:1428,
  denoiser.hpp:767, sample_euler, guidance.cpp:171) were checked against the run
  log and the pinned sources for the items this unit uses; the rest of the
  reference's internals is outside this unit.

verdict: overall PASS
