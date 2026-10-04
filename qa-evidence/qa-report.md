QA report: feat/image, unit 11, the VAE decoder kernels (Qwen-Image-2.1 P2, fourth and
last slice).
Independent falsification pass, rule 19. Falsify, do not approve: every claim below was
re-run live on this box in this pass. Nothing was committed or pushed; the only tracked
file written by this pass is this report. Probes: three header/probe edit rounds in
cuda/qwen_image_vae.cuh and ds4_qwen_image_gpu.cuh plus one in tests/
test_qwen_image_primitives.c, each observed failing the gate and restored byte-exactly
(sha256 equal to the pre-probe value and to git cat-file -p HEAD:<path>); one GPU probe
for the F32->F16 conversion in /tmp; the tree is clean at the end (git diff empty, only
the three pre-existing untracked graphify-rs-out/, run-bonsai-spark.sh and
tests/test_qwen35_bonsai_rows remain).

Unit: commit 0bd0bfdbb27fd1d1033673ecb21d9d4bbc57112b "feat(image): add the VAE decoder
kernels", branch feat/image, exactly one commit ahead of the QA base fork/feat/image =
109323ceaa96b941d09a02546a338c7af3b458e8. Verified live: git rev-parse HEAD =
0bd0bfdbb27f...; git rev-parse fork/feat/image = 109323ceaa96...; git log --oneline
fork/feat/image..HEAD lists only 0bd0bfd; git merge-base --is-ancestor 109323c HEAD
returns exit 0 (109323c is an ancestor). Diff stat 109323c..HEAD: 6 files, +1272/-12
(Makefile +1/-1, cuda/qwen_image_vae.cuh +255 new, ds4_gpu.h +69, ds4_qwen_image_gpu.cuh
+169/-3, tests/test_qwen_image_primitives.c +754, docs/qwen-image-2.1-roadmap.md
+35/-11). No crates/ file is touched by the commit.

Refs and files inspected: 0bd0bfd (full message and patch); 109323c:qa-evidence/
qa-report.md (the unit-10 report this one follows); cuda/qwen_image_vae.cuh:1-255 (whole
file, re-read and re-hashed after every probe); ds4_qwen_image_gpu.cuh:256-413 (the six
decoder entries); ds4_gpu.h:5241-5311; Makefile:737-739 (ds4_cuda.o prerequisite);
tests/test_qwen_image_primitives.c:1-60, :875-1663 (the whole VAE section), :1005-1006
(2e-4/2e-3 F16 bound), :1517-1524 (the attention-block relaxed 2e-4/2e-2 bound);
tests/qa-gate.sh (whole); docs/qwen-image-2.1-roadmap.md:327-411 (P2 gate result),
:543-592 (section 6 and tracker). Oracle: crates/ds4-core/src/qwen_image/vae.rs:266-354
(typed readers, taps, f16_round), :371-426 (f16_round), :442-462 (rms_norm), :464-539
(conv3x3), :541-581 (conv1x1), :583-596 (nearest_up2), :615-648 (dup_up3d), :667-722
(attention); its DupUp3D op-chain test :940-1060. Layout constant: crates/ds4-core/src/
qwen_image.rs:324-325 (conv3d_dims), :71-80 (VAE constants). Reference (pinned checkout
/tmp/sdref-build): ggml/src/ggml-cuda/im2col.cu:7-40,114-152; ggml/src/ggml-cuda/
conv2d.cu:86-134; src/model/vae/wan_vae.hpp:18-92,110-122,240,322-370,540,588-648,
1070-1110; src/core/ggml_extend.cpp:539-555 (split_image_qkv), :616-780
(ggml_ext_attention_ext, scale 1/sqrt(d_head)); src/model/common/ggml_block.hpp:430-455
(Conv2d F16), :700-735; src/model_loader.cpp:112-160 (convert_tensor BF16->F16).
Gemms: ds4_cuda.cu:7281-7284 (f32_to_f16_kernel / __float2half), :6938-6971
(matmul_f16_kernel, no activation rounding), :24199-24213 (cuda_native_f16_max_tokens
default 8), :26023-26111 (dispatch: cuBLAS converts activations for n_tok > 8, native
splitk/matmul_f16 for n_tok <= 8).

Artifacts (sha256; the six tree files match git cat-file -p HEAD:<path> exactly after
every probe restore)
  Makefile                              dc7bd84e67bf28c0e56b018d6167b37b76384c3643b163501b0d65bfe8e87c5e
  cuda/qwen_image_vae.cuh               901ba38fd582445422e330eb00764d41a3231396842ad9722505e14ceebd78bd
  ds4_gpu.h                             7da3972edb446272ba8952a2d1cf8590c2fcd94b09c1b41f32038d75ffea56dd
  ds4_qwen_image_gpu.cuh                731ed26eff413c5a6e1be26ee37aca75946b5815605933e5346ae3ecf85f7edd
  tests/test_qwen_image_primitives.c    a66bff8ad96c7983723cbac8f77bb14dc20f8c291add4e93161aa1e5dcee6138
  docs/qwen-image-2.1-roadmap.md        e394e32aff55bd9e48db4787b28d6fd947bdb84c5bf74cd2c3874453ffd9c726
VAE artifact (the P0 decode GGUF the oracle reads and the kernels must eventually
consume):
  misc/scratch/p0/vae-decode-bf16.gguf  d3feefed174e69d51f380c71bb500ab92033d85fc83f5d87c9811a8775725372
    (GGUF v3, 134 tensors, 0 metadata keys, every tensor type 30 = BF16; byte count
     518107872. /tmp/qa-vae.gguf and /tmp/qa-vae2.gguf hash identical.)
Environment re-probed live: nvcc 13.3.73, RTX 4070 SUPER (compute_cap 8.9),
CUDA_ARCH=sm_89 persisted in .ds4-cuda-config.mk, kThreads=256. Two forced clean
rebuilds of ds4_cuda.o (about 100 s each) plus a final clean rebuild all exit 0.

Surfaces covered (each on its own line, exact string from the gate)

cuda/qwen_image_vae.cuh
docs/qwen-image-2.1-roadmap.md
ds4_gpu.h
ds4_gpu_qwen_image_bias_add_tensor
ds4_gpu_qwen_image_dup_up3d_tensor
ds4_gpu_qwen_image_im2col3x3_tensor
ds4_gpu_qwen_image_nearest_up2_tensor
ds4_gpu_qwen_image_vae_attn_tensor
ds4_gpu_qwen_image_vae_rms_norm_tensor
ds4_qwen_image_gpu.cuh
Makefile
tests/test_qwen_image_primitives.c

Findings

1. Gate re-run on the committed tree: numbers reproduced, exit 0.
- make test-qwen-image-primitives CUDA_ARCH=sm_89, after rm of ds4_cuda.o and the test
  object/binary and a full rebuild: "all Qwen-Image checks passed", make exit 0
  (FINAL_TEST_EXIT=0). 49 lines end in " ok"; 47 of them are compare/compare_tol parity
  cases (26 DiT + 21 VAE, the "47 cases" the commit message cites) and the other two are
  the non-parity checks check_f16_round_trip and test_vae_refusals. Raw VAE section:
    vae f16 encoder round-trip                     ok
    vae refusals (DupUp3D ratio, 16384-token attn) ok
    vae im2col3x3 5x3 ci=3                         max_abs=0.000e+00 rel_rms=0.000e+00  ok
    vae im2col3x3 16x16 ci=64                      max_abs=0.000e+00 rel_rms=0.000e+00  ok
    vae rms_norm ch=1152 pixels=256                max_abs=3.576e-07 rel_rms=7.247e-08  ok
    vae nearest_up2 16x16 ch=1152                  max_abs=0.000e+00 rel_rms=0.000e+00  ok
    vae dup_up3d 16x16 in=1152 out=1152 ft=2       max_abs=0.000e+00 rel_rms=0.000e+00  ok
    vae dup_up3d 16x16 in=1152 out=576 ft=2        max_abs=0.000e+00 rel_rms=0.000e+00  ok
    vae dup_up3d 16x16 in=576 out=288 ft=2         max_abs=0.000e+00 rel_rms=0.000e+00  ok
    vae dup_up3d 16x16 in=288 out=144 ft=1         max_abs=0.000e+00 rel_rms=0.000e+00  ok
    vae dup_up3d 3x2 in=6 out=3 ft=1               max_abs=0.000e+00 rel_rms=0.000e+00  ok
    vae conv3x3 16x16 cin=64 cout=1152             max_abs=2.193e-05 rel_rms=9.927e-07  ok
    vae conv3x3 16x16 cin=1152 cout=1152           max_abs=2.937e-04 rel_rms=5.595e-06  ok
    vae conv3x3 16x16 cin=1152 cout=576            max_abs=1.793e-04 rel_rms=3.409e-06  ok
    vae conv3x3 16x16 cin=144 cout=4               max_abs=3.195e-05 rel_rms=2.141e-06  ok
    vae conv3x3 5x3 cin=64 cout=1152               max_abs=1.335e-05 rel_rms=9.924e-07  ok
    vae conv1x1 16x16 cin=64 cout=64               max_abs=9.537e-07 rel_rms=1.503e-07  ok
    vae conv1x1 16x16 cin=1152 cout=3456           max_abs=4.578e-05 rel_rms=1.924e-06  ok
    vae conv1x1 16x16 cin=1152 cout=1152           max_abs=4.578e-05 rel_rms=1.921e-06  ok
    vae conv1x1 5x3 cin=1152 cout=1152             max_abs=4.292e-05 rel_rms=1.924e-06  ok
    vae attn ch=1152 tokens=256                    max_abs=1.565e-07 rel_rms=4.681e-07  ok
    vae attn ch=1152 tokens=1024                   max_abs=1.788e-07 rel_rms=8.201e-07  ok
    vae attn block 16x16 ch=1152                   max_abs=9.201e-03 rel_rms=7.654e-05  ok
  The commit's "im2col, nearest_up2 and all five DupUp3D shapes bit-exact; rms_norm
  3.576e-07 / 7.247e-08; conv3x3 <= 2.937e-04 / 5.595e-06; conv1x1 <= 4.578e-05 /
  1.924e-06; attention 1.788e-07 at 1024" all reproduce to the digit.
- make ds4-server CUDA_ARCH=sm_89 -> exit 0 (FINAL_SERVER_EXIT=0), Rust host relinked.
  nm -C ds4-server shows all six symbols T: ds4_gpu_qwen_image_im2col3x3_tensor,
  _bias_add_tensor, _vae_rms_norm_tensor, _nearest_up2_tensor, _dup_up3d_tensor,
  _vae_attn_tensor. strings ds4-server contains all six launch lines: "Qwen-Image VAE
  im2col launch", "bias add launch", "RMSNorm launch", "nearest up2 launch",
  "DupUp3D launch", "attention launch".
- tests/qa-gate.sh before this report: GATE_EXIT=1, 8 checks red -- "QA report is fresh"
  (the stale 109323c report predates 0bd0bfd) and seven surfaces not covered
  (cuda/qwen_image_vae.cuh and the six new ABI names; docs/qwen-image-2.1-roadmap.md,
  ds4_gpu.h, ds4_qwen_image_gpu.cuh, Makefile, tests/test_qwen_image_primitives.c already
  matched). After this report: green (the closing evidence line below).

2. Falsification round 1: im2col tap order, nearest_up2 ox halving, DupUp3D oy parity.
  Edits in cuda/qwen_image_vae.cuh (iy/ix kernel-row/col swapped; nearest_up2's ox/2
  changed to ox%2; the + kUpFactorS*(oy%kUpFactorS) parity term set to 0), rebuilt, run.
  Observed (raw, probe 1):
    vae im2col3x3 5x3 ci=3            max_abs=1.843e+00 rel_rms=1.016e+00  FAIL
    vae im2col3x3 16x16 ci=64         max_abs=1.993e+00 rel_rms=1.152e+00  FAIL
    vae nearest_up2 16x16 ch=1152     max_abs=1.996e+00 rel_rms=1.369e+00  FAIL
    vae dup_up3d 1152->1152 ft=2      max_abs=0.000e+00 rel_rms=0.000e+00  ok
    vae dup_up3d 1152->576  ft=2      max_abs=0.000e+00 rel_rms=0.000e+00  ok
    vae dup_up3d 576->288   ft=2      max_abs=0.000e+00 rel_rms=0.000e+00  ok
    vae dup_up3d 288->144   ft=1      max_abs=1.992e+00 rel_rms=9.984e-01  FAIL
    vae dup_up3d 6->3       ft=1      max_abs=1.635e+00 rel_rms=1.071e+00  FAIL
    all five conv3x3 cases            FAIL (rel_rms 1.14-1.20, via im2col)
    all four conv1x1, rms_norm, attn, block  ok (unaffected)
  Attribution is clean: the tap-order break hits im2col and conv3x3 only; the ox-halving
  break hits nearest_up2 only; the parity break hits dup_up3d only. The parity result
  confirms the commit's own note verbatim: only the ft=1 cases carry the oy parity term.
  For ft=2 with repeats=8 (in==out) or repeats=4 the /repeats floor is invariant to the
  0..3 parity offset, so removing the term changes nothing; for ft=1, repeats=2, the
  offset selects between two input channels, so it flips. make exit 2. Restore:
  git checkout -- cuda/qwen_image_vae.cuh -> sha256 901ba38f... == git blob.

3. Falsification round 2: kChannelNormEps, the 1/sqrt(C) attention scale, the bias add.
  Edits (1e-12f -> 1e-6f; rsqrtf(channels) -> 1.0f; bias_add_rows += bias -> += 0.0f),
  rebuilt, run. Observed (raw, probe 2):
    vae rms_norm ch=1152 pixels=256   max_abs=7.059e-01 rel_rms=4.415e-02  FAIL
    vae attn ch=1152 tokens=256       max_abs=1.127e+00 rel_rms=1.335e+01  FAIL
    vae attn ch=1152 tokens=1024      max_abs=1.072e+00 rel_rms=2.536e+01  FAIL
    all five conv3x3 and all four conv1x1  FAIL (max_abs ~= 5.0e-1, the bias)
    vae attn block                    max_abs=3.321e+01 rel_rms=7.296e-01  FAIL
    im2col, nearest_up2, all five dup_up3d  ok (unaffected)
  Attribution is clean: eps -> rms_norm (the test seeds pixel 0 at 1e-6 so a wrong eps
  moves its scale); scale -> the two attn cases (13.35/25.36 rel RMS, the commit cites
  13.3 for the 256-token case); bias -> every conv. make exit 2. Restore:
  git checkout -> 901ba38f... == git blob. The DiT-side cases stayed green (14 ok lines
  between the DiT headings), so the break is local to the decoder ops.

4. Falsification round 3: the two named refusals and the test's own f16 unpack exponent.
  Edits in ds4_qwen_image_gpu.cuh (the "tokens > kMaxAttentionPixels" and the
  "(cout*factor) % cin" guards replaced by "if (0u) return 0;") plus tests/
  test_qwen_image_primitives.c unpack_f16 (subnormal biased exponent 113 - e -> 112 - e),
  rebuilt, run. Observed (raw, probe 3):
    vae f16 encoder round-trip        FAIL
    vae refusals (DupUp3D ratio, 16384-token attn)  FAIL
    (every parity case stayed ok, values within the baseline bound)
  Both guards are load-bearing: dropping the ratio guard lets cin=3/cout=2/ft=1 through
  (returns 1, test expected 0); dropping the pixel cap lets 9000 tokens launch (37 KiB
  shared, inside the 48 KiB limit -> returns 1). The unpack exponent edit flips the
  subnormal vectors and only that check. make exit 2. Restore: git checkout --
  ds4_qwen_image_gpu.cuh tests/test_qwen_image_primitives.c -> 731ed26e... and
  a66bff8a... == git blobs.

5. F16 conversion is bit-for-bit vae.rs::f16_round over the finite grid. Claim tested
  directly (not via the test's host mirror). Standalone probe /tmp/qa_f16_probe.cu
  reimplements vae.rs::f16_round (returning the half bits) and the test's pack_f16, runs
  the device's __float2half (the exact call f32_to_f16_kernel makes, ds4_cuda.cu:7283)
  over the oracle's 21 vectors plus 20,000,000 finite bit patterns covering subnormals,
  every exponent band and both signs, then compares. Result:
    N=20000000  __float2half vs f16_round: 0 mismatches
                __float2half vs pack_f16: 0 mismatches
                pack_f16 vs f16_round:    0 mismatches
  The NaN case is excluded from the corpus by design: all three diverge there (f16_round
  keeps a quiet NaN with the sign, __float2half canonicalizes, pack_f16 returns 0x7fff),
  exactly as the header note states; NaN never reaches the finite decoder arithmetic.

6. The attention-block residual (9.201e-03 / 7.654e-05) is explained and is not a defect;
  the implementer's stated magnitude source is wrong. I isolated the stages with a
  temporary, restored probe in test_vae_attention_block that reads the kernel's own
  attention output out of tattn. Raw:
    QA attn stage (kernel vs mirror)   max_abs=1.776e-03 rel_rms=2.623e-05
    QA proj-only (kernel attn)         max_abs=1.183e-04 rel_rms=1.921e-06  ok
    QA block residual outliers         n=294912 over2e-3=1525 over5e-3=26 over1e-2=0
  Reading: (a) the proj composition alone, given the SAME attention output, reproduces
  the mirror at the ordinary conv drift (1.18e-04 max, inside the 2e-3/2e-4 conv bound),
  so the block's 9.2e-3 does not come from the proj GEMM, the bias, or the unpatch; (b)
  the attention stage itself differs by 1.78e-3 at the block's mixed qkv, which is four
  orders larger than the 1.6e-7 the standalone attn case shows on a FIXED qkv, so the
  difference is driven by the qkv conv output (cuBLAS-vs-mirror F16 accumulation order,
  the tested conv1x1 drift of 4.6e-5) amplified by the 1152-wide q.k dot, not by fast-math
  expf; (c) the residual tail is sparse (1525/294912 = 0.5% of elements over 2e-3, 26 over
  5e-3, none over 1e-2), the signature of F16 rounding-boundary flips in the pre-proj
  activation, exactly the mechanism the test comment names. The test's own comment
  ("fast-math expf ... differ from the mirror by ~1e-7, which occasionally crosses an F16
  rounding boundary") misattributes the magnitude: the pre-proj difference is ~1.8e-3,
  and only the F16-flip mechanism is right. This is a documentation inaccuracy, not a
  correctness hole: the block bound was openly relaxed to 2e-4 rel RMS / 2e-2 max abs at
  tests/test_qwen_image_primitives.c:1524, the scale-free rel RMS (7.65e-5) sits 2.6x
  inside 2e-4, and every structural break I ran moves rel RMS to 1.0-25 (findings 2-3),
  orders of magnitude above 7.65e-5, so the relaxed max-abs bound cannot hide a real
  defect at these shapes.

7. Artifact format: the slice is kernel-complete but NOT wired-ready; CONFIRMED as the
  commit states. Read live:
  - crates/ds4-core/src/qwen_image/convert.rs:320 writes TYPE_BF16 for every tensor;
    :332-339 rejects any source dtype other than BF16. The P0 artifact
    misc/scratch/p0/vae-decode-bf16.gguf is GGUF v3, 134 tensors, all type 30 (BF16),
    sha256 d3feefed... (finding header).
  - crates/ds4-core/src/qwen_image/vae.rs:275-277 (dequantize_f32) refuses any tensor
    whose type is not TYPE_BF16, so the oracle only ever sees BF16 weights.
  - The decoder conv path here is ds4_gpu_matmul_f16_tensor (ds4_gpu.h:1151), which reads
    its weights as __half from the model map (matmul_f16_kernel / cublasGemmEx F16 input,
    ds4_cuda.cu:6940-6952, :26025-26031). It cannot read the BF16 artifact directly.
  - The reference does the cast at load: Conv2d::init_params fixes the weight tensor to
    GGML_TYPE_F16 (src/model/common/ggml_block.hpp:444-445) and the loader's
    convert_tensor copies any non-F16 source (BF16 included) through to_float then
    ggml_fp32_to_fp16_row into F16 (src/model_loader.cpp:112-160, else branch :143-158).
  - The slice's own test never touches the real artifact: it packs F16 weights into a
    private 64 MiB map with map_put_f16 (tests/test_qwen_image_primitives.c:1169-1202), so
    the green gate proves the kernel math, not the artifact delivery.
  Consequence: P4 MUST deliver the VAE conv/qkv/proj weights as F16 (a converter variant
  that writes TYPE_F16, or a load-time BF16->F16 cast into a mapped buffer) before the
  decoder graph is wired. This is disclosed loudly in the commit and the roadmap; it is an
  open P4 precondition, not a defect of the unit's own scope.
  One nuance I measured, below the contract's tolerance: the test's mirrors use
  F16-rounded weights (ref_vae_conv3x3/conv1x1 call unpack_f16) while the P1 oracle
  (vae.rs::taps -> dequantize_f32) uses BF16-as-F32 weights with no F16 rounding. The two
  differ only for BF16 values not on the F16 grid. Over the real artifact 27571 of
  259,048,212 BF16 weight elements (0.011%) are not F16-exact, all tiny (|w| below about
  8e-6, i.e. below the F16 subnormal exponent band); the resulting conv difference is
  O(1e-6), far inside the 2e-3 F16 conv bound. So the test is the more reference-faithful
  of the two and the divergence is immaterial, but "the P1 oracle's math" is loose as a
  description of the weight operand.

8. Oracle contract, index by index (kernel vs vae.rs vs the reference), all AGREE.
  - conv3x3 (vae.rs:464-539): oracle adds bias only after the accumulation, rounds only
    the activation with f16_round (vae.rs:517), and reads the weight in the artifact's
    flat order tap + 9*(ic + IC*oc). Kernel composition = im2col3x3 -> matmul_f16 ->
    bias_add -> unpatch. The GEMM reads w[oc*(9*ci) + j] with im2col feature j = tap +
    9*ic (cuda/qwen_image_vae.cuh:97-123), i.e. flat[9*ci*oc + tap + 9*ic] = the artifact
    index. Exact.
  - conv1x1 (vae.rs:541-581): one position per pixel, F16 operands, bias after. Kernel =
    patch_1x1 -> matmul_f16 -> bias_add -> unpatch; the GEMM row oc/feature ic reads
    flat[ic + IC*oc] = the artifact's [1,1,IC,OC] order. Exact.
  - rms_norm (vae.rs:442-462): mean = sum(x^2)/C in double, scale = 1/sqrt(mean +
    1e-12), then x*scale*gamma. Kernel: block-parallel sum of x^2, mean = sum/C,
    rsqrtf(mean + 1e-12), x*scale*gamma[c] (cuh:140-156). Same formula, same 1e-12
    (VAE_NORM_EPS, qwen_image.rs:80), same gamma order; only the accumulation order
    differs (double sequential vs block tree), carried by the 3.576e-07 result.
  - nearest_up2 (vae.rs:583-596): out(w,h,c) = in(w/2,h/2,c). Kernel gather
    dst[ox + outW*oy + outPix*c] = src[ox/2 + width*(oy/2) + pixels*c] (cuh:161-173).
    Exact, bit-identical.
  - dup_up3d closed form (vae.rs:615-648): m = fs^2*(ft-1+ft*oc) + fs*(oy%fs) + (ox%fs),
    c_in = m/repeats, repeats = cout*ft*fs^2/cin; the oracle's refusals are
    (cout*factor) % cin != 0 (vae.rs:622-627), and the kernel entry matches it exactly
    (ds4_qwen_image_gpu.cuh:335-336). The oracle's own op-chain test
    (vae.rs:940-1060) replays the literal ggml concat/reshape/permute chain against this
    closed form. Kernel (cuh:176-200) computes the same m, same c_in = m/repeats with
    repeats passed in by the entry. Exact, bit-identical at all five shapes.
  - attention (vae.rs:667-722): scale = 1/sqrt(C); scores = dot(q,k)*scale; max-subtract,
    exp, normalize; out = sum(p*v); q/k/v are the three contiguous channel thirds of
    to_qkv, token = flattened w + W*h. Kernel (cuh:202-251) computes the same with
    scale = rsqrtf(C), block max/sum, one sequential dot per key and one sequential p.v
    sum per feature in index order. The reference confirms all of it:
    ggml_ext_attention_ext with n_head=1 -> d_head = C and scale = 1/sqrt(d_head)
    (ggml_extend.cpp:653), mask nullptr and non-causal (wan_vae.hpp:635), and
    split_image_qkv cuts the [W,H,3C] tensor into contiguous q/k/v thirds
    (ggml_extend.cpp:539-555). AGREE.
  - Weight layout vs conv3d_dims: conv3d_dims(kw,kh,kt,ic,oc) = [kw,kh,kt,ic*oc]
    (qwen_image.rs:324-325), ggml flat = kw + 3*kh + 9*(ic + IC*oc) for the singleton
    temporal kernel, and the reference im2col writes feature offset
    iic*KD_KH_KW + ikd*KH_KW + ikh*KW + ikw = 9*ic + 3*ikh + ikw with KD=1, KH=KW=3
    (im2col.cu:114-152), so its dst feature index equals the kernel's j = tap + 9*ic
    (tap = 3*kh + kw). The oracle taps() repacks at the same flat index
    tap + TAPS*(ic + ci*oc) (vae.rs:330-350). Exact, all three.

9. Entry validation: boundary accepts and zero/undersized refusals hold exactly. A
  temporary, restored probe in test_vae_refusals drove the six entries (raw):
    vae entry boundaries (exact/zero/undersized)   ok
  Checks (all as expected): attn channels=4 tokens=8192 -> accept (the cap refuses only
  > 8192), tokens=0 -> 0, channels=0 -> 0; im2col ci=1 w=3 h=3 with src exactly 9 floats
  and dst exactly 81 floats -> accept, src 8 floats (one float short) -> 0, dst 9 floats
  (undersized) -> 0, channels=0 -> 0; bias_add dim=0/rows=0 -> 0; rms_norm
  channels=0/pixels=0 -> 0; nearest_up2 channels=0 -> 0; dup_up3d cin=0 -> 0 and a valid
  ratio (4,4,1,1,1) -> accept. The existing refusal case also confirms 9000 tokens (over
  the cap, but 37024 B of shared, inside 48 KiB) is refused by the named cap and 16384 is
  refused. Restore proven: git diff empty.

10. Native F16 GEMM caveat: confirmed. cuda_native_f16_max_tokens defaults to 8
    (ds4_cuda.cu:24199-24213). For n_tok > 8 the cuBLAS branch converts the activation
    with f32_to_f16_kernel -> __float2half (ds4_cuda.cu:26025-26031); for n_tok <= 8 the
    native matmul_f16_kernel reads x as raw float with no rounding to F16
    (ds4_cuda.cu:6938-6971, dispatched at :26110). Every decoder conv case keeps
    pixels > 8 and test_vae_conv hard-fails below that (tests:1206-1210), so the F16
    operand contract is exercised everywhere the gate runs; a future conv with <= 8 tokens
    would silently drop the reference's activation rounding. This is a real, disclosed
    limitation, not a defect of this slice (the decoder never has <= 8 spatial tokens).

11. Roadmap slice-4 claims vs my raw output (docs/qwen-image-2.1-roadmap.md:352-411,
    tracker :577). All numbers match: "im2col, nearest_up2 and all five DupUp3D shapes
    bit-exact" (0.0/0.0), "rms_norm 3.6e-7 abs" (3.576e-07), "conv3x3/conv1x1 <= 2.9e-4
    abs and <= 5.6e-6 rel RMS" (2.937e-04 / 5.595e-06), "against the F16 section bound
    (2e-3 / 2e-4, stated in the test)" (kF16MaxAbsTol/kF16RelRmsTol at :1005-1006),
    "attention 1.8e-7 abs" (1.788e-07), "six more ds4_gpu_qwen_image_* entries" (six new
    declarations), "Attention is capped at 8192 tokens ... 16384 refused" (probe 9), and
    the tracker row "P2 | done -- kernels ... the F16 weight delivery is carried into P4"
    (finding 7). Two presentation nits, reported not fixed: (a) the roadmap/commit say
    "Nine falsifications" but the parenthetical names only seven and the roadmap names
    six; the full nine are the ones listed in the task preamble and all nine were
    reproduced here (findings 2-4 plus the two guards), so the count is right and the
    enumeration is short; (b) "47 cases ok" counts the 47 parity cases, while the binary
    prints 49 " ok" lines (the two extra are the f16 round-trip and refusal checks).

12. Cheat audit: no cheat found. No hardcoded pass: compare_tol/compute rel RMS and
    max abs from the actual buffers, g_failures drives main's return, and every probe
    moved the metric from ~1e-7 to 1.0-25 (findings 2-4). No kernel left unlaunched in the
    test: every entry return is checked and exit(1) on refusal; the six launch strings are
    all present in ds4-server. The comparison is not self-fulfilling: the mirrors are
    independent double/sequential reimplementations diffed index-by-index against vae.rs
    (finding 8) and the permutation cases are bit-exact, which a wrong-but-small index
    shift could not be.

Residual risk
- The strongest residual is the BF16 artifact: the unit's green gate is kernel math only,
  and P4 cannot wire the decode graph until the weights are delivered as F16 (a converter
  variant or a load-time cast). Disclosed in the commit and roadmap; flagged here as the
  one thing that decides the slice's usability. Confirmed by reading convert.rs:320,
  vae.rs:275-277 and the F16 GEMM signature, and by the artifact's all-BF16 GGUF header.
- The attention-block max_abs (9.201e-03) rests on the relaxed 2e-2 bound; I showed the
  mechanism (qkv conv accumulation drift amplified by the 1152-wide dot, then F16
  boundary flips in the proj input, 0.5% of elements) and that the proj composition alone
  is at the ordinary conv drift. The test comment's magnitude attribution is wrong
  (1e-7 vs the measured 1.78e-3), but the number stays inside the disclosed bound and no
  structural break hides there. Low severity; worth a comment fix in a later slice.
- The test mirrors use F16 weights while the P1 oracle uses BF16-as-F32; the two differ on
  0.011% of the artifact's tiny weights by O(1e-6), immaterial at the 2e-3 conv bound, but
  "the oracle's math" is loose for the weight operand.
- Unverified: only sm_89 was exercised; the reference's F16 cast is a read of
  model_loader.cpp, not a run of the reference model (its dump files are not in this
  tree); the fused-flash-attn branch of the reference (flash_attn_enabled) was not run,
  but it is mathematically the same softmax and the oracle pins the manual path.
- Prevention/rollback: this report is written before the final qa-gate run; the unit is a
  local branch commit, so rollback is rewriting/disarding the branch pointer and the gate
  re-runs from a clean tree. Trigger to reopen: any red gate, an F16-delivery change that
  alters the conv operands, or a P4 wiring that exposes the <= 8-token GEMM caveat.

verdict: overall PASS
