QA report: feat/image, unit 9, the DiT rope and segmented attention kernels (Qwen-Image-2.1 P2, second slice).
Independent falsification pass, rule 19. Falsify, do not approve: every claim below was
re-run live on this box. Nothing was committed or pushed; the only repo file written by
this pass is this report. Two probe edits were made in cuda/qwen_image_attn.cuh, each was
observed to fail the gate, and the file was restored byte-exactly after each (cmp and
sha256 equal to the git blob, see finding 3). No tracked file is modified at the end
(git status: only the pre-existing untracked graphify-rs-out/, run-bonsai-spark.sh,
tests/test_qwen35_bonsai_rows; git diff empty).

Unit: commit e6b37ebdc1da32afd097bbe753d4fa7c793904ce "feat(image): add the DiT rope and
segmented attention kernels", branch feat/image, exactly one commit ahead of the QA base
fork/feat/image = bda12f6c40d2ff8ab9742e851ef5dfd07ff4d123 (verified: git rev-parse
fork/feat/image; git log fork/feat/image..HEAD lists only e6b37eb; base is an ancestor).
Diff stat bda12f6..HEAD: 6 files, +638/-31 (Makefile, cuda/qwen_image_attn.cuh,
docs/qwen-image-2.1-roadmap.md, ds4_gpu.h, ds4_qwen_image_gpu.cuh,
tests/test_qwen_image_primitives.c). No crates/ file is touched.

Refs and files inspected: e6b37eb (full message and patch), bda12f6; cuda/qwen_image_attn.cuh
(whole 186-line file, re-read and hashed after every probe), cuda/qwen_image_primitives.cuh
(whole, for kThreads and block_sum), ds4_qwen_image_gpu.cuh (whole 170-line file),
ds4_gpu.h:5140-5206, ds4_cuda.cu:48454 (include), Makefile:740,:893-901,:60-62,
tests/test_qwen_image_primitives.c (whole 643-line file), tests/qa-gate.sh (whole file),
docs/qwen-image-2.1-roadmap.md:349-375 and :537-540, .ds4-cuda-config.mk
(CUDA_ARCH=sm_89 persisted). Oracle: crates/ds4-core/src/qwen_image/oracle.rs:183-261
(build_layout/append_image), :270-284 (linspace/rope_omega), :292-312 (rope_table),
:322-328 (RopePairing), :332-361 (apply_rope), :370-377 (text_mask);
crates/ds4-core/src/qwen_image/dit.rs:138-152 (dot8), :212-220 (rms_norm_row), :257-268
(softmax_row), :523-583 (segment_attention), :591-648 (attention), :740 (rope_table call);
crates/ds4-core/src/qwen_image.rs:39-54 (DIT_HIDDEN/HEAD_DIM/HEADS/AXES_DIM/ROPE_THETA);
crates/ds4-core/tests/qwen_image_oracle.rs:355-400 (the fixture-gated pairing measurement).
Second reference, local checkout /tmp/sdref-build: src/model/diffusion/qwen_image_2_1.hpp:17
(axes_dim 16/56/56), :340-357 (mask construction), :348 (embed_nd, theta 10000),
src/model/common/rope.hpp:1110-1151 (apply_rope, interleaved default, pe [L, d_head/2, 2, 2]).

Artifacts: cuda/qwen_image_attn.cuh sha256 1f6d5d5f1c8e6f686dcb4cc114ff7e93606394a97e586e09285262bdc37d521f
(equals git cat-file -p HEAD:cuda/qwen_image_attn.cuh at the end). tests/test_qwen_image_primitives.c
sha256 00d084a851ef2a4797c6914af80402a1dce8496b963569d9aa16431f3464b94f. Rebuilt live this
pass: ds4_cuda.o, tests/test_qwen_image_primitives, ds4-server (forced relink through the
Rust host). Environment: nvcc 13.3.73, RTX 4070 SUPER sm_89, CUDA_ARCH=sm_89.

Surfaces covered (each on its own line, exact string from the gate)

cuda/qwen_image_attn.cuh
docs/qwen-image-2.1-roadmap.md
ds4_gpu.h
ds4_gpu_qwen_image_attn_segment_tensor
ds4_gpu_qwen_image_rope3d_tensor
ds4_qwen_image_gpu.cuh
Makefile
tests/test_qwen_image_primitives.c

Findings

1. Gate re-run on the committed tree, numbers reproduced exactly.
- make test-qwen-image-primitives CUDA_ARCH=sm_89 -> 14 cases ok, exit 0. Raw lines:
  layernorm rows=1 max_abs=7.451e-09 rel_rms=1.194e-09; rows=4224 4.768e-07 / 5.853e-08;
  modulate (4 cases) 0.000e+00/0.000e+00 (plain) and 4.768e-07/5.130e-08,
  0.000e+00/0.000e+00, 4.768e-07/5.144e-08 (gated+residual);
  mlp_gated 9.537e-07/5.767e-08 and 1.907e-06/5.992e-08; mlp_gated_fused 3.815e-06/5.921e-08
  and 7.629e-06/5.990e-08;
  rope3d q heads=32 tokens=4224 max_abs=0.000e+00 rel_rms=0.000e+00 (bit-exact);
  rope3d k same 0.000e+00/0.000e+00;
  attn text prefix (causal) 128 max_abs=1.788e-07 rel_rms=1.253e-07;
  attn image (unmasked) 4096x4224 max_abs=1.416e-07 rel_rms=1.089e-06.
  All four claims in the commit message reproduce to the digit (14 ok; rope bit-exact;
  attention 1.788e-07/1.253e-07 and 1.416e-07/1.089e-06 against 1e-4/1e-5).
- make ds4-server CUDA_ARCH=sm_89 -> exit 0 (Rust host rebuilt and relinked after the
  probe restore). nm -C ds4-server shows both symbols T:
  ds4_gpu_qwen_image_rope3d_tensor and ds4_gpu_qwen_image_attn_segment_tensor.
- tests/test_qwen_image_primitives and ds4_cuda.o are sm_89 SASS (cuobjdump --list-elf).

2. The critical check: the host mirror is the oracle, element by element. I diffed
tests/test_qwen_image_primitives.c against oracle.rs/dit.rs and the second reference:
- axis widths/theta: mirror axes {16,56,56} and theta 10000.0f at
  tests/test_qwen_image_primitives.c:229-230; oracle DIT_AXES_DIM=[16,56,56] and
  DIT_ROPE_THETA=10000.0 at crates/ds4-core/src/qwen_image.rs:50,54, consumed by
  rope_table at dit.rs:740; reference config axes_dim {16,56,56} at
  /tmp/sdref-build/src/model/diffusion/qwen_image_2_1.hpp:17, theta 10000.f at :348. AGREE.
- pair-axis concatenation: mirror pair_offset accumulates the three half-widths in axis
  order (:233-258); oracle rope_table pair_offset = sum of previous halves (:298-299). AGREE.
- omega ladder: oracle linspace(0, (dim-2)/dim, half) then 1/theta^s (oracle.rs:270-284);
  mirror end=(dim-2)/dim, step=end/(half-1), omega=1/powf(theta, j*step) (:237-241). For
  start 0 the two are the same f32 expression (0 + j*step vs j*step). AGREE.
- pairing: mirror applies (2j, 2j+1) with table [cos,-sin,sin,cos] and cos at +0, sin at +2
  (:245-255,:271-279); oracle RopePairing::Interleaved (:324,:350-358) and rope_table
  values (:305-308). The oracle's choice of Interleaved is itself the fixture-measured
  winner (tests/qwen_image_oracle.rs:375-399), not a stylistic default; the mirror matches
  that branch. AGREE.
- mask: mirror masks key > query only when causal, over keys [0,end), (:322-323); kernel
  passes causal=1 for the text span [0,128) and causal=0 for the image span [128,4224)
  (:571-582). Oracle text_mask is -inf for k>q (oracle.rs:370-377), applied only when the
  segment has image_index<0 (dit.rs:639-644); the second reference builds the same
  [end, end-start] mask with -INFINITY for k>q (qwen_image_2_1.hpp:340-357). AGREE.
- scale: mirror 1.0f/sqrtf(head_dim) (:316), oracle 1.0f32/(head_dim as f32).sqrt()
  (dit.rs:536); kernel uses rsqrtf(head_dim) (cuda/qwen_image_attn.cuh:133), a fast-math
  rounding variant of the same constant (MUFU.RSQ). Same math, kernel rounding inside gate.
- softmax accumulation: oracle serial max, exp(x-max), sum in key order, divide
  (dit.rs:257-268); mirror identical (:328-335); kernel does a block-tree max and a
  block-tree sum over the same key set (cuda/qwen_image_attn.cuh:136-156), masked keys
  contribute -INFINITY and exp 0. Semantically identical; only the reduction order differs.
  This is the dominant residual of the two attention cases (1.25e-07 and 1.089e-06 rel
  RMS), as the header claims.
- dot8: mirror (:284-297), oracle (dit.rs:138-152) and kernel (cuda/qwen_image_attn.cuh:76-92)
  have the identical structure: eight accumulators, the same fixed horizontal grouping
  ((0+1)+(2+3)) + ((4+5)+(6+7)), then the scalar tail.
- layouts: q/k head-major [head][token][head_dim] (kernel q_row at attn.cuh:135, mirror
  :319), matching dit.rs's qh/kh reshape (:611-622); v/out feature-fastest with the head
  offset (mirror v[(key)*hidden+hd+d] / out[query*hidden+hd+d] at :341-352; oracle
  v[d+head_dim*(head+heads*key)] / row[head*head_dim+d] at dit.rs:568-579; kernel
  v_head[(key)*hidden+d] / out_head[d] at attn.cuh:163-181). AGREE.
- positions: mirror text ids 0..127 on all three axes then image temporal=128, spatial
  h-32, w-32 (:203-223); oracle build_layout/append_image: one position per text token,
  temporal = running position (128 after the prefix), center = 64 - 64/2 = 32 (oracle.rs:183-261).
  AGREE.
- No semantic divergence found in any of the listed elements. The rope bit-exactness is
  consistent with the kernel SASS (FMUL.FTZ/FFMA.FTZ for the two rotation expressions) and
  the correspondingly contracted host arithmetic; it is not a coincidence of a shared bug
  I could find.

3. Mask falsification, both directions, independent of the coder's probe.
- Probe A, no-op causal mask: cuda/qwen_image_attn.cuh:140 changed from
  "if (causal && key > query) { score = -INFINITY; }" to "if (causal && key > end) ..."
  (always false, key < end). Rebuilt and ran: attn text prefix (causal) 128
  max_abs=1.127e+00 rel_rms=8.990e-01 FAIL; attn image unchanged at 1.416e-07/1.089e-06 ok;
  the other 12 cases untouched; make exit 2 (test exit 1). Matches the commit's 8.99e-1.
- Probe B, unconditional mask: the same line changed to "if (key > query) ...". Rebuilt and
  ran: attn text unchanged at 1.788e-07/1.253e-07 ok; attn image max_abs=2.263e-01
  rel_rms=1.618e+00 FAIL; make exit 2. Matches the commit's 1.62e0.
- Restore: after each probe the file was copied back from a pre-probe copy and verified with
  cmp and sha256 = 1f6d5d5f1c8e6f686dcb4cc114ff7e93606394a97e586e09285262bdc37d521f; the
  final file hash equals git cat-file -p HEAD:cuda/qwen_image_attn.cuh exactly. The final
  rebuild+run after the restore: all 14 cases ok, exit 0, identical numbers to finding 1.
  (No git checkout was used; two cp restores only. git status shows no tracked modification.)

4. The pe-on-host decision: re-measured, the claim holds.
- Probe /tmp/qa_cosf.cu: 4,194,304 samples over [0, 4224] rad, sm_89, compared against
  double cos of the same f32 angle. With the tree's flags (--use_fast_math):
  max_abs=5.527008e-04 at x=4211.344238. Without --use_fast_math: max_abs=8.546459e-08.
  Host libm cosf in the same sweep: 3.238561e-08. So the device fast-math cosf is indeed
  ~5.5e-4, five-and-a-half times the 1e-4 absolute gate; the commit number 5.5e-4 is
  reproduced (5.53e-4), and the error is the fast-math lowering, not angle quantization.
- The kernel consumes a host-built pe table and computes no trig: rope3d_rows takes the
  pe pointer as an argument and only loads cos/sin from it (cuda/qwen_image_attn.cuh:99-115);
  the header contains no sinf/cosf/__sinf/__cosf call (grep, only comments); the SASS of
  qwen_image_cuda::rope3d_rows contains no MUFU.COS/MUFU.SIN (its only MUFU ops are two
  MUFU.RCP from the integer-division sequence used for pair indexing); the rotation is
  FMUL.FTZ/FFMA.FTZ. ds4_qwen_image_gpu.cuh:106-132 just validates and launches with
  pe->ptr; the test builds the table on the host (tests/test_qwen_image_primitives.c:227-259).

5. ABI: ds4_gpu.h:5176-5204 vs ds4_qwen_image_gpu.cuh:106-170. A mechanical balanced-paren
parse of both prototypes gives identical parameter lists:
  rope3d: (ds4_gpu_tensor *x, const ds4_gpu_tensor *pe, uint32_t tokens, uint32_t n_head,
  uint32_t head_dim), return int, extern "C" on the definition.
  attn_segment: (ds4_gpu_tensor *out, const ds4_gpu_tensor *q, const ds4_gpu_tensor *k,
  const ds4_gpu_tensor *v, uint32_t tokens, uint32_t n_head, uint32_t head_dim,
  uint32_t start, uint32_t end, uint32_t causal), return int.
No name, order, type or return divergence. Both symbols are T in the gate binary and in
ds4-server. ds4_gpu.h now declares exactly six ds4_gpu_qwen_image_* entry points, matching
the roadmap's "six" claim.

6. kMaxSegmentKeys = 8192 (cuda/qwen_image_attn.cuh:55). The entry checks
"if (end > qwen_image_cuda::kMaxSegmentKeys) return 0;" (ds4_qwen_image_gpu.cuh:152),
before any launch or tensor work. Live probe (/tmp/qa_bound_probe.c, linked against the
tree's own objects): end=8193, tokens=9000 -> return 0 (refused); end=9000 -> return 0;
end=8192 exactly -> return 1 (launched). The real 4224-key span is exercised by the gate's
image case and passes. Shared footprint at the bound is (8192+256)*4 = 33792 bytes, inside
the 48 KiB default dynamic limit. Note for precision: the refusal is a silent return 0 (no
stderr line); the "by name" part is the named constant and the early return, not a printed
message. That is the same failure convention as the other five entries in this header.

7. Cheat audit: no cheat found.
- Tolerances are meaningful: the falsification probes move the metric from ~1e-6/1e-7 to
  0.9 and 1.6 rel RMS, and the observed worst case (1.089e-06 rel, 1.416e-07 abs) sits an
  order inside the stated 1e-5/1e-4. The compare() gate was observed failing (probes A/B).
- No hardcoded pass, no unreachable assert (the test has no asserts; compare() increments
  g_failures and the process exits 1 when it fires).
- No kernel left unlaunched: rope3d is launched twice and attn_segment twice per run, with
  the return values checked (exit(1) on refusal); the primitive cases launch their entries too.
- Inputs are not degenerate: splitmix64 uniform q/k/v/norm/param over O(1) ranges; the
  attention cases run 128 queries x 128 keys (text) and 4096 x 4224 (image), and both mask
  directions change the output at O(1) scale when broken, so a wrong or absent mask cannot
  pass. The image case is not uniform and not a single key.
- The two rope cases share one pe table and one input pattern; that is adequate for the
  rotation arithmetic, and the attention cases consume the separately verified roped q/k
  (the isolation is deliberate and stated in the test header).

Observations (not defects, for the next slice):
- dit.rs reshapes q/k to head-major before norm/rope (dit.rs:611-622); no CUDA counterpart
  of that transpose exists yet, so the entries require head-major q/k. The next P2 slice
  must feed head-major from the projections or add the transpose. The commit and roadmap
  acknowledge the remaining glue.
- The mirror's table is built with host gcc cosf/sinf; the Rust oracle uses f32::cos/sin.
  A ~1 ulp table difference is immaterial against 1e-4/1e-5, but the two tables were not
  numerically diffed (see unverified).
- The kTextTokens=128 prefix is one fixture shape; the real joint length 4224 matches
  128 + 64x64 latent tokens.

Unverified items
- The fixture-gated test that measured Interleaved vs HalfSplit (tests/qwen_image_oracle.rs:355-400)
  was not run: it needs DS4_QWEN_IMAGE_ORACLE dumps that are not in this tree. The pairing
  choice is carried by that recorded measurement and by the reference default, not re-measured here.
- The end-to-end DiT path using these kernels (norm_q/norm_k on the existing weighted-RMS ops,
  the head-major reshape, timestep/patch glue) is outside this commit and was not exercised.
- Host libm cosf vs Rust f32::cos/sin equality of the pe table was not numerically diffed.
- Only sm_89 was tested; no other architecture in this pass.

Risk analysis
Decision: accept e6b37eb as PASS for the feat/image QA gate (it may become the next pushed
state). No push, publish or deploy is done by this pass.
External risks: none network-facing; the evidence depends on the local CUDA 13.3 toolkit and
the RTX 4070 SUPER driver/device state (probed: nvcc 13.3.73, sm_89, gate ran green). A
different arch or a JIT-only build was not tested (unverified).
Decision risks: the PASS rests on the test's host mirror being the oracle; checked by a
line-level diff of every element listed in finding 2 (axes, theta, pairing, mask, scale,
softmax, dot8, layout, positions) plus the second reference source; not checked by an oracle
dump run (fixtures absent). The mask falsification covers only the mask, not every mirrored
constant; a constant shared between kernel and mirror in the same wrong way would not be
caught by this harness (mitigated, not eliminated, by the diff).
Measurement limits: the cosf sweep is a 4.2M-sample grid over [0,4224]; the observed
5.527e-4 already exceeds the claimed 5.5e-4, so the claim is safe directionally, but a
different sample grid could in principle find a slightly larger value.
Prevention/rollback: this report is written before the gate runs; the unit is a local branch
commit, so rollback is discarding/rewriting the branch pointer, and the QA gate re-runs from
a clean tree. Trigger to reopen: any failing gate, any later fixture run that flips the
pairing, or a mirror-vs-oracle divergence found in the next slice.
Residual risk: low. The strongest residual is the deferred glue (head-major reshape, norm
ops) outside this slice's scope.

verdict: overall PASS
