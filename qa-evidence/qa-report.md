QA report: feat/image — the P1 oracle's VAE decode (Qwen-Image-2.1 P1 stage 3).
Independent falsification pass, rule 19. This report was refreshed after the unit
was amended in place; it describes commit 7e5a145. Nothing was committed or
pushed by me; the only repo file written is this report.

Unit: commit 7e5a14517641b55131a21b16c6f635aba71af802 "feat(image): add the P1
oracle's VAE decode" (amended from e195e4d, which remains reachable), branch
feat/image, one commit ahead of the QA base fork/feat/image = 1396ce1 (verified:
git rev-parse fork/feat/image = 1396ce17cc30504e6ffd3f156ea5b27e46457627; git log
1396ce1..HEAD lists only 7e5a145). Diff stat 1396ce1..HEAD: +1503/-21 across
exactly the same five files. Amendment delta e195e4d..HEAD: vae.rs +14/-2 (the
f16_round subnormal path now rounds the magnitude at vae.rs:412, with a comment
at :406-411 and six new sign assertions in f16_round_follows_binary16,
vae.rs:1248-1253), docs/qwen-image-2.1-p1.md and docs/qwen-image-2.1-roadmap.md
rewritten with the corrected numbers. The oracle test file and the qwen_image.rs
module line are byte-identical between the two commits (git diff e195e4d..HEAD
for them is empty), so the earlier live falsification of the gate carries over.
Amended vae.rs sha256
bea404b09a78c4c5f77e1f1663966558b6f309e148af210564e647f4bf36d08a. Commit message
now records 9.2 s, PSNR 70.2 dB, relative RMS 3.2568e-4, 1312 differing bytes and
the subnormal correction.

Refs and files inspected: 7e5a145 (full message), e195e4d, 1396ce1, git diff
1396ce1..HEAD and e195e4d..HEAD for crates/ds4-core/src/qwen_image.rs,
crates/ds4-core/src/qwen_image/vae.rs (whole 1278-line file), both docs,
crates/ds4-core/tests/qwen_image_oracle.rs, crates/ds4-core/tests/qwen_image.rs,
tests/qa-gate.sh, the previous qa-evidence/qa-report.md.
Artifacts: misc/scratch/p0/vae-decode-bf16.gguf (sha256 d3feefed...5372),
/data/imagegen/models/diffusion_models/qwen-image-2.1-Q6_K.gguf,
/data/imagegen/models/vae/qwen_image_2.1_vae_bf16.safetensors (bb21f747...).
Fixtures: misc/scratch/p1/refdump/run1.{png,log,step1.in.bin,step1.pred.bin,step2.in.bin,step2.pred.bin}.
Reference harness: /tmp/sdref-build (git 74988b2; vae_harness.cpp, vae_stages.cpp,
src/model/vae/wan_vae.hpp, src/core/tensor_ggml.hpp, src/runtime/preprocessing.hpp,
src/model/vae/vae.hpp, ggml/src/ggml-impl.h, build/CMakeCache.txt GGML_F16C=OFF,
sdref-build.log) and its outputs /tmp/{ref_out_a..f.f32, ref_vae_out.f32,
z_a_mulstd_plusmean.bin, z_ref.bin, refstage_run1_11.f32, refstage_run2_11.f32,
ref_harness.png}; /tmp/vaedec.py. My probes: /tmp/qa_f16_probe.rs (the amended
and the e195e4d f16_round side by side), /tmp/qa_f16.rs (e195e4d only),
/tmp/qa_refconv.c (linked against the harness build's libggml-base.a).

Surfaces covered (each on its own line, exact string from the gate)

crates/ds4-core/src/qwen_image.rs
crates/ds4-core/src/qwen_image/vae.rs
crates/ds4-core/tests/qwen_image_oracle.rs
docs/qwen-image-2.1-p1.md
docs/qwen-image-2.1-roadmap.md
pub enum VaeError
pub fn channels
pub fn decode
pub fn diffusion_to_vae
pub fn height
pub fn new
pub fn open
pub fn token
pub fn to_rgba8
pub fn vae_to_diffusion
pub fn values
pub fn width
pub fn write_png
pub fn zeros
pub struct Plane
pub struct VaeWeights

Findings

1. Re-verified on the amended tree (7e5a145), all live:
- cargo test -p ds4-core --lib qwen_image::vae -> 15 passed, 0 failed.
- cargo test -p ds4-core --lib qwen_image -> 54 passed, 0 failed.
- cargo test -p ds4-server --test image_cli -> 5 passed, 0 failed.
- DS4_QWEN_IMAGE_ORACLE=/data/ds4-dfm-rs/misc/scratch/p1/refdump/run1
  DS4_QWEN_IMAGE_VAE=/data/ds4-dfm-rs/misc/scratch/p0/vae-decode-bf16.gguf
  cargo test -p ds4-core --release --test qwen_image_oracle
  vae_decode_reproduces_the_reference_image -- --nocapture -> ok:
  "vae decode: 256x256x4 in 9.4s"; "decode vs reference: max |diff| r 0.00392
  g 0.00784 b 0.00784 a 0.01961, PSNR 70.2 dB, relative RMS 3.2568e-4, differing
  bytes r 324 g 364 b 410 a 214 of 65536". Sum 1312, exactly the number the
  amended docs and the commit message record, and the per-channel maxima are
  r 1/255, g 2/255, b 2/255, a 5/255. The recorded 9.2 s is one run of a band I
  observe at 9.2-9.5 s on this host (9.4 s this pass, 9.4-9.5 s in the previous
  pass); no number contradicts.
- These amended numbers are exactly what my pre-amendment temporary abs() probe
  measured (PSNR 70.2, 1312 bytes, b max 2/255), which is the direct evidence
  that the committed fix is the change that was probed.

2. Falsification evidence, and its carry-over to the amendment:
- The gate was falsified live on the pre-amendment tree: dropping f16_round at
  both conv call sites FAILED at PSNR 16.1 dB and 254/255; flipping the range map
  at to_rgba8 FAILED at 130/255; both reverted byte-exactly and the pass
  restored. The amended tree changes neither the test file nor the two tolerance
  asserts or their lines (grep: worst <= 6.0/255 at test:550, psnr >= 55.0 at
  test:555, unchanged), so the gate's non-vacuity carries unchanged.
- New check specific to the amendment: the new negative-subnormal assertions were
  executed against the e195e4d function body in a standalone probe. The amended
  function returns the correctly rounded values on all four cases; the e195e4d
  function returns -0.0 on all four (MISMATCH on every case). So the new test
  would have failed on the old code; it is a real gate on the fix.

3. Cheat audit of the amended diff: no hardcoded output, no dead path, no
unfireable assert (both tolerance asserts and the new sign assertions are
demonstrably failable), no reference-file cheat beyond run1.png being the
reference's own image by design. The new assertions' expected values are the
reference conversion's own rounded results, independently checked in finding 8.

4. The strongest claim (the reference harness reproducing run1.png) was
re-derived and re-run live in the previous pass on the same fixtures; the
amendment touches none of it: I rebuilt the harness input byte-identically from
the dumps (65573 bytes), re-ran /tmp/sdref-build/vae_harness to a byte-identical
output (sha256 0b8333cd64d8e09e8521ca76b8b322c5cf82866ab6b76ded4f7c9d6d0034b25d),
and re-derived with the reference's own float_to_u8 semantics (preprocessing.hpp:27-35,
(x+1)/2 clamp, NaN to 1.0) a pixel-byte-exact match to run1.png, 65536 of 65536
pixels; the five alternative latent conversions fail at 5.06/10.78/14.08/11.98/
9.72 dB, inside the claimed 5-14 dB. The amended roadmap's "raw, doubled and
inverse maps fail at 5-14 dB" is consistent with that. "Byte-exactly" holds at
the image-data level, not the PNG container (encoder difference); /tmp/ref_harness.png
is a stale preview of the raw-latent run, not the byte-exact image.

5. The 6% saturation claim is unchanged and holds: the pre-scale head output
(/tmp/refstage_run1_11.f32, verified to reproduce ref_out_a exactly through the
map) carries 15488 NaN values = 5.91% of 262144, i.e. 3872 of 65536 pixels
(5.91%) with all four channels NaN, 0 inf at that stage. The amended roadmap's
reworded range "about [-0.99, +2.77] with ~6% NaN" matches my measurement of the
finite range [-0.9928, +2.7696] and that 5.91%.

6. Artifact and contract use, re-checked live on the amended tree: the GGUF
sha256 matches the doc (d3feefed...5372); identify_vae with DS4_QWEN_IMAGE_VAE
set yields 134 tensors, all BF16 (tests/qwen_image.rs:166-175), and the contract
pins every conv at kT=1 (qwen_image.rs:347-440), so "every Conv3d weight has
ne[2]==1" is established by the exact-dims match; the wrong artifact is rejected,
not mis-loaded (identify_vae(DiT) fails with a contract problem list, and the
oracle loader fails with Contract { tensor: "conv2.weight", why: "missing from
the artifact" }). The 64-channel statistics constants (qwen_image.rs:89-108) are
element-identical to the reference's VERSION_QWEN_IMAGE_2_1 tables
(wan_vae.hpp:1373-1390).

7. The three doc imprecisions recorded by the previous pass are FIXED, and each
corrected number matches my measurement:
- p1.md:223 now reads "max abs diff r 0.00392 g 0.00784 b 0.00784 a 0.01961;
  PSNR 70.2 dB; relative RMS 3.26e-4; 1312 of 262144 channel bytes differ" -
  measured 3.2568e-4 and 1312, identical.
- roadmap:315 now reads "max 5/255, on the alpha channel; 1-2/255 on RGB" -
  measured r 1/255, g 2/255, b 2/255, a 5/255; both clauses true.
- roadmap:320 now reads "about [-0.99, +2.77] with ~6% NaN (not a clean [-1,1])" -
  measured finite range [-0.9928, +2.7696] and 5.91% NaN; accurate.
- p1.md:246-248's new sentence about the negative-subnormal correction matches
  finding 8. The tracker line (roadmap:513) and the commit message carry 70.2 dB,
  consistent with the table. No remaining unreproduced number in the amended docs.

8. The previous pass's defect is FIXED. Evidence:
- Code: vae.rs:412 is now (value.abs() * 16_777_216.0).round_ties_even(), with a
  comment at :406-411 naming the signed-cast saturation it avoids; the new
  assertions at :1248-1253 cover -2^-24, -0.75*2^-24, -1e-5 (= -168*2^-24),
  -6e-5 (= -1007*2^-24) and the sign.
- Standalone probe of the amended function against the e195e4d body, using the
  new test's own expressions: amended returns -5.96046448e-8 for -2^-24 and
  -0.75*2^-24 (both round to -1 ulp of the grid, correctly), -1.00135803e-5 for
  -1e-5, -6.00218773e-5 for -6e-5, and the sign is negative; the e195e4d body
  returns -0.0 for all four.
- Reference check: the reference's own conversion (probe linked against the
  harness build's libggml-base.a, F16C off) returns -1.00135803e-05 for -1e-5
  (bits 0x80a8), -6.00218773e-05 for -6e-5, and -5.96046448e-08 for
  -4.47034836e-8; numpy float16 agrees. The amended function matches the
  reference on all four cases; the e195e4d function did not.
- Gate effect, measured: the amended tree reads PSNR 70.2 dB with 1312 differing
  bytes against the pre-amendment 70.3 dB / 1246; both are far inside the adopted
  6/255 and 55 dB bound, and the amended docs record the amended numbers.
This finding is closed as FIXED.

9. No other defect found. Checked and clean: no hardcoded outputs, no tolerance
inflation, no dead code path in the decode chain, no unfireable assert, no
reference-file cheat, no scope creep (the unit touches exactly the five files
above; qwen_image.rs adds the one module line).

Unverified items

- The stage-2 DiT numbers quoted in the roadmap's P1 gate result (correlation
  0.999973, relative RMS 7.6e-3) were not re-measured here; they belong to the
  previous unit. Its tests pass in this tree, but I did not re-run the DiT
  fixture gate against the DiT artifact.
- The original run1 generation (the reference pipeline that produced run1.png and
  the dumps) was not re-run; I verified only the decode stage and the harness
  from those fixtures.
- I did not count how many conv operands actually land in the negative-subnormal
  range during the decode; the reach is bounded (6.1e-5 per operand) but not
  enumerated. This no longer affects correctness, since the rounding now matches
  the reference for that class.
- The "overflow to infinity" half of the saturation claim is measured at the
  final head stage as NaN only; inf was not counted at intermediate levels.
- Decode wall time is a band (9.2-9.5 s observed across runs on this host); the
  docs' single 9.2 s figure is one sample of it, not a contradiction.

verdict: overall PASS
