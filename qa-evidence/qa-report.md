QA report: feat/image — the P1 oracle's exact-numerics stage (Qwen-Image-2.1).
Independent falsification pass, rule 19, covering HEAD 7b2726d (the third
amendment of this unit). Nothing was committed or pushed and no source file was
modified; the only file written in the repository is this report.

Verdict: PASS. This report covers HEAD 7b2726d. It confirms the two defects
found on the first pass (6566d07) are corrected, that the corrections were
independently reproduced (Euler byte-identical in the reference's operation
order, 692 of 16384 differing under the simplified form; least squares
recovering -0.377540648, the port's F32 step), and that this third amendment
changed documentation wording only. Every measured claim in
docs/qwen-image-2.1-p1.md now checks out.

Unit identification

Repository /data/ds4-dfm-rs, branch feat/image. HEAD 7b2726d "feat(image): add
the P1 oracle's exact-numerics stage", amended twice from 6566d07 via 4c449ea,
one commit ahead of fork/feat/image = ddf6d88. Working tree has one modified
file, this report. Diff: crates/ds4-core/src/qwen_image/oracle.rs (+644),
crates/ds4-core/tests/qwen_image_oracle.rs (+265),
docs/qwen-image-2.1-p1.md (+171), docs/README.md (+1),
crates/ds4-core/src/qwen_image.rs (+1).

Delta pass on this amendment (4c449ea -> 7b2726d)

git show 7b2726d --stat and git diff 4c449ea 7b2726d: the delta is exactly two
wording changes inside docs/qwen-image-2.1-p1.md and nothing else. No source,
test, constant or measured value changed.

1. Section 2 now qualifies the first step's formula: "the first step's update
   is x1 = x0 + pred0 * (sigma1 - 1) — in that order's arithmetic only, as
   section 3 shows: the reference computes it in a different order, and that
   difference is measurable." This resolves the internal contradiction I
   recorded after the previous amendment (section 2 stated the simplified form
   unqualified while section 3 showed it is not the reference's order).
2. The all-F32 Box-Muller sentence now reads "wrong in 5142 of 16384 samples
   (4774 by one ulp, 360 by two, 8 by three)" instead of "wrong by one ulp per
   sample". I recomputed the distribution for this pass and it is exact:
   11242 byte-identical, 4774 differ by 1 ulp, 360 by 2, 8 by 3, total
   differing 5142, maximum 3 ulp.

Commands run, with actual results

  cargo test -p ds4-core --lib qwen_image::oracle
    running 12 tests ... test result: ok. 12 passed; 0 failed; 0 ignored
  DS4_QWEN_IMAGE_ORACLE=/data/ds4-dfm-rs/misc/scratch/p1/refdump/run1 \
    cargo test -p ds4-core --test qwen_image_oracle
    running 7 tests ... test result: ok. 7 passed; 0 failed

Earlier passes on 4c449ea (unchanged by this amendment; the code and tests are
byte-identical between 4c449ea and 7b2726d):
  cargo test -p ds4-core --lib                      332 passed, 4 ignored
  cargo test -p ds4-core --test qwen_image            8 passed (P0, unchanged)
  cargo test -p ds4-core --test qwen_image_oracle      7 passed (early return,
                                                       no env var)

Independent numeric re-derivation (python3 + glibc libm via ctypes, and a
standalone Rust f32 replica in /tmp; neither uses the unit's code)

Noise, seed 42, counter offset 0, 16384 values of run1.step1.in.bin:
  all-F32 (logf/sqrtf/sinf)             11242 / 16384 byte-identical
  double throughout, one final rounding 12136 / 16384 byte-identical
  mixed precision (the port's form)     16384 / 16384 byte-identical
All-F32 ulp distances from the reference: {0: 11242, 1: 4774, 2: 360, 3: 8};
5142 differ in total, maximum 3 ulp. This matches the amended sentence exactly.

Schedule bits. Recomputing flux_mu/flux_time_shift/flux_sigmas in F32:
  sigma1        bits 0x3f1f597f (value 0.622459352016449)
  sigma1-sigma0 bits 0xbec14d02
Both are the pins the unit test asserts and the test passes, so the code
produces those bits.

Euler step. Recomputing the reference chain denoised = x - sigma*pred,
d = (x - denoised)/sigma, x + d*(sigma_next - sigma):
  exact chain differs from run1.step2.in.bin       : 0 / 16384 (byte-identical)
  one-expression form x + pred*(sigma1 - sigma0)   : 692 / 16384 differ
The doc's and the commit's 692 is exact. The integration test asserts both the
0 and the >0, and passes.

Least squares. sum((x1-x0)*pred0)/sum(pred0^2) = -0.377540647882; the
per-element ratio (x1-x0)/pred0 has median -0.377540648. This is exactly the
port's F32 step f32(0.622459352) - 1, not the double ideal -0.377540669, as the
amended paragraph now says.

Defect 1 from the first pass (Euler figure "worst 2.4e-7"): FIXED. The table
reads "byte-identical 16384/16384", reproduced above.
Defect 2 from the first pass (least-squares "-0.377540669 ... nine digits"):
FIXED. The paragraph now gives -0.377540648 and explains the double ideal's F32
spacing, reproduced above.
Both wording points recorded after the second pass: FIXED by this amendment,
recomputed above.

Per-surface list (every surface the gate requires)

crates/ds4-core/src/qwen_image/oracle.rs — read in full against the reference
sources in misc/scratch/p1/ref/.
crates/ds4-core/src/qwen_image.rs — exactly "pub mod oracle;" beside convert;
unchanged by this amendment.
crates/ds4-core/tests/qwen_image_oracle.rs — the flow test calls pub fn
euler_step and asserts the reference order is byte-identical while the
one-expression form is not; 7 tests early-return without the env var.
docs/qwen-image-2.1-p1.md — the two wording changes of this amendment are its
only edits; re-read in full, no measured claim is now overstated.
docs/README.md — the single P1 index row, unchanged and accurate.

pub const FLOW_SHIFT — 3.0, matches diffusion_engine.cpp:1372.
pub const FLUX_BASE_SHIFT, pub const FLUX_MAX_SHIFT — 0.5, 1.15; match
FluxScheduler (denoiser.hpp:732-733).
pub enum LayoutError — Invalid / SlotShapeMismatch / MissingRefSlots, in the
reference's order (qwen_image_2_1.hpp:78/100/115).
pub enum RopePairing — Interleaved / HalfSplit; both rotations, distinguishable,
left unresolved as the recipe requires.
pub fn apply_rope — out0 = x0*cos - x1*sin, out1 = x0*sin + x1*cos; matches
rope.hpp:1110-1151 for both branches.
pub fn build_layout — matches QwenImage21Layout::build (text-only and
reference-image cases, including the error order).
pub fn euler_step — denoised = x - sigma*pred; d = (x - denoised)/sigma;
x + d*(sigma_next - sigma). Byte-exact against run1.step2.in.bin (0/16384),
independently and by the passing integration test; the terminal-sigma unit test
is valid at sigma = 1.
pub fn flow_timestep — sigma*1000; matches DiscreteFlowDenoiser::sigma_to_t
(denoiser.hpp:1316-1318).
pub fn flux_mu — line through (256, base_shift) and (4096, max_shift); mu(256)
= 0.5.
pub fn flux_sigmas — matches FluxScheduler::get_sigmas; the F32 pin is verified.
pub fn flux_time_shift — exp(mu)/(exp(mu)+(1/t-1)^sigma) in F32.
pub fn initial_noise, pub struct Philox, pub fn new, pub fn seed, pub fn randn —
Philox 4x32, ten rounds, matches rng_philox.hpp:26-96; 16384/16384 byte-exact.
pub fn noise_scaling — latent*(1-sigma)+noise*sigma, or noise*sigma with no
init latent; matches denoiser.hpp:1341.
pub fn rope_table — the per-pair [cos,-sin,sin,cos] layout; all 512 entries of
my position/axis test match an independent Rope::embed_nd.
pub fn text_mask — [key][query] storage, -inf where key > query; matches
qwen_image_2_1.hpp:346-353 for the start-0 text segment this unit exercises.
pub struct Layout, pub struct Segment — fields match the reference structs.

Claims not verified, and whether the gap is acceptable

1. The SD_DUMP_* hooks live only in the deployed binary, not the pinned source;
   I verified their product (the dumps) rather than their disassembly.
   Acceptable.
2. The DiT forward, VAE decode, PNG, kernels/GPU and the text encoder are
   explicitly out of scope (doc section 6) and not quietly included: no FFI, no
   GPU, no model load in the module. Acceptable.
3. The RoPE pairing is intentionally unresolved; the recipe itself lists it as
   unsettleable by reading and needing a probe. Acceptable.
4. text_mask reproduces the reference only for a text segment starting at 0
   (the text-to-image fixtures here); a segment after a vision-slot run would
   need the reference's [end, end-start] shape. Not claimed otherwise; a note
   for stage 2. Acceptable.
5. Artifact hashes (section 5) are a prior-session record; I did not re-hash the
   model artifacts, which are outside this unit's code. Acceptable.

verdict: overall PASS
