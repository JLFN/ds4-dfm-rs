QA report: feat/image — the P2 DiT primitive kernels (Qwen-Image-2.1, first P2 slice).
Independent falsification pass, rule 19. Nothing was committed or pushed; the only
repo file written is this report. Every probe edit was made in the working tree,
the gate was observed failing, and the header was then restored byte-exactly
(git checkout; sha256 re-verified = 3cb3ec3eb9d23f9a019a03eedc46d6b9d7f83efb963524a84466f6cb4152f3ba).
No tracked file is modified by this pass (git status: only the pre-existing
untracked graphify-rs-out/, run-bonsai-spark.sh, tests/test_qwen35_bonsai_rows).

Unit: commit e6099f631355379be0f94c6e9d01567181fe5fb3 "feat(image): add the P2 DiT
primitive kernels", branch feat/image, one commit ahead of the QA base
fork/feat/image = f074269ec1e75a884f264ca7fa55a53ca25a251f (verified: git rev-parse
fork/feat/image; git log --oneline f074269..HEAD lists only e6099f6). This report
was refreshed after the unit was amended in place from 93cfd0c; the amendment
delta 93cfd0c..e6099f6 touches exactly one file, docs/qwen-image-2.1-roadmap.md
(+8/-6), verified with git diff --name-only and --numstat. The code, the test, the
ABI and the Makefile are byte-identical between the two commits
(cuda/qwen_image_primitives.cuh sha256 3cb3ec3e...152f3ba in both trees; the
commit message is also byte-identical, diff of git show -s --format=%B is empty),
so the live gate runs and the four falsification probes below carry over
unchanged. Diff stat f074269..HEAD: +659/-3 across exactly the eight files below
(the docs change is the +2 net lines); no crates/ file is touched.

Refs and files inspected: e6099f6 (full message), 93cfd0c (pre-amendment tree),
f074269; cuda/qwen_image_primitives.cuh
(whole 155-line file, re-read after every probe), ds4_qwen_image_gpu.cuh (whole
103-line file), ds4_gpu.h:5138-5173 and :11, ds4_cuda.cu:48454, tests/test_qwen_image_primitives.c
(whole 327-line file), Makefile:740, :893-901, :103, :1962, tests/qa-gate.sh (whole file),
.gitignore:75, docs/qwen-image-2.1-roadmap.md:352-366 and :530, docs/qwen-image-2.1-recipe.md:80,
:187-188, :310-315.
Oracle: crates/ds4-core/src/qwen_image.rs:39-56 (DIT_NORM_EPS = 1e-6, hidden 4096,
intermediate 12288), crates/ds4-core/src/qwen_image/dit.rs:197-204 (layer_norm_row), :223-225
(silu), :488-512 (modulate), :669-704 (block: img_norm1/2, mod[0..3], img_mlp).
Reference (second oracle, local checkout): /tmp/sdref-build/src/model/diffusion/qwen_image_2_1.hpp:196-197
(img_norm1/img_norm2 = LayerNorm(hidden, 1e-6f, affine=false)), :208-219 (modulate),
:229-242 (fused chunk order and silu), :257 (norm_out), and
/tmp/sdref-build/src/model/common/ggml_block.hpp:739-786 (LayerNorm -> ggml_ext_layer_norm,
null weight).
Artifacts: cuda/qwen_image_primitives.cuh sha256 3cb3ec3e...152f3ba; rebuilt live this
pass: ds4_cuda.o, tests/test_qwen_image_primitives, ds4-server (forced relink).

Surfaces covered (each on its own line, exact string from the gate)

cuda/qwen_image_primitives.cuh
docs/qwen-image-2.1-roadmap.md
ds4_cuda.cu
ds4_gpu.h
ds4_gpu_qwen_image_layernorm_tensor
ds4_gpu_qwen_image_mlp_gated_fused_tensor
ds4_gpu_qwen_image_mlp_gated_tensor
ds4_gpu_qwen_image_modulate_tensor
ds4_qwen_image_gpu.cuh
.gitignore
Makefile
test-qwen-image-primitives
tests/test_qwen_image_primitives.c

Findings

1. Re-run on the committed tree, numbers reproduced:
- make test-qwen-image-primitives CUDA_ARCH=sm_89 -> all 10 cases ok, exit 0.
  Printed: layernorm rows=1 dim=4096 max_abs 7.451e-09 rel_rms 1.194e-09;
  rows=4224 max_abs 4.768e-07 rel_rms 5.853e-08; modulate 1/prefix0 plain and
  4224/prefix128 plain max_abs 0.000e+00 (bit-exact); modulate 129/prefix128 and
  4224/prefix128 gated+residual max_abs 4.768e-07 rel_rms 5.13-5.14e-08;
  mlp_gated rows=1 9.537e-07, rows=4224 1.907e-06; mlp_gated_fused rows=1
  3.815e-06, rows=4224 7.629e-06; rel_rms <= 5.992e-08 everywhere.
  The commit message's "max abs <= 7.6e-6 and rel RMS <= 6.0e-8" is the rounded
  reading of the observed maxima 7.629e-06 and 5.990e-08; the printed shape set
  (hidden 4096, intermediate 12288, 1 and 4224 joint tokens, plus the 129-token
  prefix-boundary modulate case) matches the message, as do "all cases ok", "the
  plain modulate cases bit-exact", and the stated bound 1e-4 abs / 1e-5 rel RMS
  (tests/test_qwen_image_primitives.c:35-36).
- make ds4-server CUDA_ARCH=sm_89: "up to date", exit 0. To make the link claim
  non-trivial I forced a real relink (make -W ds4_cuda.o ds4-server CUDA_ARCH=sm_89)
  -> exit 0, and the resulting binary exports all four symbols (nm: T
  ds4_gpu_qwen_image_layernorm_tensor at 0x3f27c50, modulate 0x3f27ee0,
  mlp_gated 0x3f28230, mlp_gated_fused 0x3f28490); ./ds4-server --help exits 0.
  The Rust host genuinely links the new object content.
- Test object: recompiled out of tree with the Makefile's exact command
  (cc -O3 -ffast-math -g -march=native -Wall -Wextra -std=c99 ...) -> exit 0, zero
  warnings; sha256 c97eb7e5a86b4be31919623cc413e06826d63dbfc159786a60b1454370d8e1ca
  is byte-identical to the committed tests/test_qwen_image_primitives.o.

2. THE CRITICAL CHECK — the host mirror vs the oracle. I diffed each element the
  task names; formula-level they agree:
- modulate row split: dit.rs:507-511 sends tokens prefix..len to row 0 and 0..prefix
  to row 1; the reference applies rows[0] to the slice from prefix_length and
  rows[1] to the prefix (qwen_image_2_1.hpp:213-217). Mirror
  tests/test_qwen_image_primitives.c:143 `row = (t < prefix) ? 1 : 0` and kernel
  cuda/qwen_image_primitives.cuh:127 `(token < prefix) ? 1u : 0u` agree, and the
  boundary is right: token == prefix is image-side (dit.rs `for token in prefix..`).
- gate: dit.rs:495-500 `gate ? v.tanh() : v + 1.0`; reference hpp:210
  `gate ? ggml_tanh(row) : ggml_scale_bias(row, 1.f, 1.f)` (= row + 1); mirror
  tests/...c:147 `gated ? tanh(p) : p + 1.0f`; kernel :128 `gated ? tanh_exp(p) : p + 1.0f`.
- eps: DIT_NORM_EPS = 1e-6 (qwen_image.rs:56) = kLayerNormEps 1e-6f (:51) = mirror
  1e-6 (tests/...c:129); the reference constructs every DiT norm with 1e-6f and
  affine=false (qwen_image_2_1.hpp:196-197, :257; ggml_block.hpp:774-786 passes a
  null weight to ggml_ext_layer_norm).
- gate/up: unfused oracle gate = img_mlp.gate_layer, up = img_mlp.proj, then
  up *= silu(gate) (dit.rs:693-697); mirror and kernel do the same. Fused chunk
  order gate = chunk 0, up = chunk 1 matches the reference
  qwen_image_2_1.hpp:229-240 (`parts = ggml_ext_chunk(gate_up, 2, 0); gate =
  parts[0]; h = parts[1]; h = h * silu(gate)`) and the recipe :187-188.
- silu: dit.rs:223-225 `x / (1 + exp(-x))`; mirror tests/...c:156-158 (double
  exp); kernel :59-61 (expf).
Divergences found are precision, not semantics: the mirror accumulates the
LayerNorm sum/var in double while dit.rs casts the mean to f32 and subtracts in
f32, and the mirror's tanh/exp are libm double against the kernel's fast-math
expf (documented in cuda/qwen_image_primitives.cuh:35-40). All are inside the
1e-4/1e-5 gate and the measured deviations in finding 1. No divergence between
mirror and oracle in the row split, the gate form, the eps, the chunk order or
the silu formula.

3. Falsification — one live edit per kernel, rebuilt at CUDA_ARCH=sm_89, observed
  failed, restored byte-exactly (sha256 back to 3cb3ec3e...152f3ba):
- swap the modulate rows (:127) -> all four modulate cases FAIL at max_abs
  2.702-4.477, rel_rms 0.755-0.927; the other six cases unchanged and ok; make
  exit 2.
- kLayerNormEps 1e-3 (:51) -> both layernorm cases FAIL, max_abs 6.523e-04 and
  7.142e-04, rel_rms 3.72e-04 and 3.75e-04, 6.5-7x past the 1e-4 abs bound; make
  exit 2.
- swap the unfused gate/up (:141) -> both mlp_gated cases FAIL at max_abs 7.239
  and 7.335, rel_rms ~0.79; both fused cases stay ok (separate kernel); make exit 2.
- swap the fused chunks (:150-153) -> both mlp_gated_fused cases FAIL at max_abs
  3.554e+01 and 3.581e+01, rel_rms ~0.97; both unfused cases stay ok; make exit 2.
After each restore the gate passes again with exactly the numbers of finding 1;
the last restored run is the one recorded there. The gate is non-vacuous for all
four entry points.

4. ABI: a mechanical extraction and comparison of the four declarations in
  ds4_gpu.h (:5143, :5151, :5161, :5170) against the four extern "C" definitions
  in ds4_qwen_image_gpu.cuh (:28, :44, :70, :90) reports MATCH on name, return type
  int, parameter count, order and types for all four. The declarations are inside
  the header's extern "C" block (ds4_gpu.h:10-11); nvcc enforces the same match
  because ds4_cuda.cu compiles the definitions after ds4_gpu.h.

5. Include and build wiring is minimal and correct:
- ds4_cuda.cu includes ds4_qwen_image_gpu.cuh exactly once (:48454, immediately
  after ds4_qwen35_gpu.cuh); no other translation unit includes it (repo-wide grep).
- Makefile:740 lists ds4_qwen_image_gpu.cuh and cuda/qwen_image_primitives.cuh as
  ds4_cuda.o prerequisites, proven live: the probe header edit triggered the nvcc
  recompile of ds4_cuda.cu (the ds4_cuda.cu(1258) warning line appears in the
  probe build output) and ds4_cuda.o's mtime follows the header's.
- The new test rules mirror the existing per-kernel CUDA tests: tests/...:893-894
  is the same shape as tests/test_solar_kda.o (Makefile:615-616); the link rule
  :896-897 uses $(DS4_CUDA_CORE_OBJS) + $(NVCC) like tests/test_solar_kda
  (Makefile:789-790); the target :900-901 runs the binary as the existing targets
  do; test-qwen-image-primitives is in the .PHONY list (:103) and in clean (:1962).
- .gitignore:75 adds /tests/test_qwen_image_primitives; git check-ignore -v
  tests/test_qwen_image_primitives exits 0 and names .gitignore:75.

6. Cheat audit: no hardcoded pass (compare()/g_failures at tests/...c:93-108, shown
  failable four times); every kernel is launched through its public entry point and
  read back (device_read after each launch); no skipped or dead case (main at
  :302-323 runs all 10 unconditionally); no unfireable assert; the tolerances are
  meaningful — the smallest perturbation probed (eps 1e-6 -> 1e-3) is caught 6-7x
  beyond the abs bound, and the semantic swaps 4-5 orders beyond; the data are
  splitmix64 spans (x +/-1.5-2, gate +/-6 so silu crosses both tails, layernorm rows
  carry per-row offsets), not degenerate. The mirror is independent host code, not
  a copy of the kernels.

7. Docs, on the amended tree (e6099f6): roadmap:352-366's P2 gate result and
  :530's tracker line match the measurements: "max absolute difference <= 7.6e-6
  and relative RMS <= 6.0e-8 against a stated bound of 1e-4 and 1e-5" (measured
  7.629e-06, 5.990e-08), "the plain modulate cases are bit-exact" (measured
  0.000e+00), "make ds4-server CUDA_ARCH=sm_89 still links" (forced relink,
  exit 0). The two items my previous pass noted are now closed:
- Residual attribution: the amended sentence reads "The 4224-row residual is the
  norm's F32 tree reduction against the oracle's double accumulation, plus the
  fused MLP's `expf` tails (7.6e-6, still about 13x inside the absolute bound)."
  That is exactly what I measured: the layernorm 4224-row deviation is 4.768e-07
  (the kernel's F32 block_sum against the mirror's double accumulation) and the
  overall maximum 7.629e-06 sits on mlp_gated_fused from the fast-math expf/silu
  tails; 1e-4 / 7.629e-6 = 13.1, so "about 13x" holds. The old sentence no longer
  reads as attributing the maximum to the norm; the wording note is FIXED.
- Fused-order caveat: it now reads "The fused MLP's chunk order (gate = chunk 0)
  was confirmed against the reference source (`qwen_image_2_1.hpp:229-240`), not
  only the recipe; the shipped fused artifact path is still not exercised by the
  CPU oracle", which records exactly the reference confirmation reported in
  finding 2 (hpp:229-240, `parts = ggml_ext_chunk(gate_up, 2, 0); gate =
  parts[0]`) and keeps the honest limitation about the shipped artifact.

Unverified items

- The date inside "Gate result (2026-10-03, first slice)" (roadmap:352) was not
  re-derived; the commit is dated 2026-10-04 01:24 +0200, so a gate run shortly
  before midnight is plausible, but I have no log of the author's run.
- The kernels are not yet wired into any DiT graph, so no end-to-end model path
  exercises them; this slice is the primitive API that the test drives, as the
  commit states.
- Production use of the fused [2n, rows] layout is not validated against a shipped
  fused weight file (there is none in the oracle), matching the commit's caveat.
- The tanh-vs-MUFU.TANH micro-measurement quoted in the commit message (7.8e-6 vs
  4.8e-7) was not re-measured; I only re-confirmed the 4.8e-7-scale gated
  deviation against the mirror.

verdict: overall PASS
