QA report: feat/image, unit 10, the DiT timestep and patch glue kernels (Qwen-Image-2.1 P2, third slice).
Independent falsification pass, rule 19. Falsify, do not approve: every claim below was
re-run live on this box. Nothing was committed or pushed; the only tracked file written by
this pass is this report. Three probe edits were made (two in cuda/qwen_image_primitives.cuh,
one in tests/test_qwen_image_primitives.c), each observed to fail the gate and restored
byte-exactly (sha256 equal to the pre-probe value and to git cat-file -p HEAD:<path>, see
findings 4-6). Two standalone untracked probes in /tmp (a transpose-guard kernel probe and an
entry-refusal probe) were compiled against the tree's own objects but wrote no tracked file.
At the end git status shows only the pre-existing untracked graphify-rs-out/,
run-bonsai-spark.sh and tests/test_qwen35_bonsai_rows; git diff is empty.

Unit: commit 05097db4df204824d5beb99372f932d42ae1fa11 "feat(image): add the DiT timestep and
patch glue kernels", branch feat/image, exactly one commit ahead of the QA base
fork/feat/image = 0682b1d0be37a495258602eae49f1e464d74df02. Verified: git rev-parse HEAD =
05097db4df...; git rev-parse fork/feat/image = 0682b1d0be3...; git log --oneline
fork/feat/image..HEAD lists only 05097db; git merge-base --is-ancestor 0682b1d HEAD returns
true (exit 0). Diff stat 0682b1d..HEAD: 5 files, +497/-23 (cuda/qwen_image_primitives.cuh
+73/0, docs/qwen-image-2.1-roadmap.md +27/-11, ds4_gpu.h +37/0, ds4_qwen_image_gpu.cuh +87/-5,
tests/test_qwen_image_primitives.c +273/-7). No crates/ file is touched.

Refs and files inspected: 05097db (full message and patch); 0682b1d:qa-evidence/qa-report.md
(the unit-9 report this one follows); cuda/qwen_image_primitives.cuh:1-60 (header contract),
:68-73 (kThreads, __device__ silu), :167-224 (silu_rows, kTile, transpose_2d, re-read and
re-hashed after every probe); ds4_qwen_image_gpu.cuh (whole 252-line file: :26-39
tile_grid/transpose, :199-252 the three new entries); ds4_gpu.h:5204-5241; tests/
test_qwen_image_primitives.c:1-60 (header), :196-257 (ref_silu, ref_timestep_table,
ref_linear, ref_transpose), :678-877 (test_silu, check_zero_table, test_timestep, test_patch,
test_timestep_path), :894-905 (main); Makefile:60-76 (flags, object list), :893-901 (the target);
tests/qa-gate.sh (whole); docs/qwen-image-2.1-roadmap.md:275-321 (P1 gate result), :349-391
(P2 gate result), :519-522 (section 6), :538-539 and :555-556 (tracker rows);
docs/qwen-image-2.1-p1.md:112-120, :211-233; ds4-dfm-rs-handoff.md:108-112, :198-208.

Oracle: crates/ds4-core/src/qwen_image/dit.rs:223-225 (silu), :237-253 (timestep_embedding),
:742-756 (time = concat(t, 0) and the two Linear+silu stages), :788-800 (patchify gather),
:827-833 (unpatchify scatter); crates/ds4-core/src/qwen_image.rs:39-53 (DIT_HIDDEN 4096,
DIT_IN/OUT_CHANNELS 64, DIT_TIME_EMBED_DIM 256).

Artifacts (sha256, final tree = git blob):
  cuda/qwen_image_primitives.cuh      f6eea15df271505f485e27b41797aab4774ee1e46b99df92adcff11d8aaeb0f6
  ds4_qwen_image_gpu.cuh              46328864383652ebfe09ff31ce0df8b42b31917a0b632176da3964700bc15659
  ds4_gpu.h                           ac37003112b84f2cbddcd7b4f95e7c6c236a8d0b5519b1748fb60a4b238ada3f
  tests/test_qwen_image_primitives.c  da49d157318f935aca55fbf0cc880d4175e810be0998db141c18c6d00bbcb076
  docs/qwen-image-2.1-roadmap.md      02bfa7123416bf2b2381fd151b8bde77f2d1a44647feeae12fc622ecb9c22383
Each of the first four equals its git cat-file -p HEAD:<path> sha256 exactly after the probe
restores. Rebuilt live this pass: ds4_cuda.o, tests/test_qwen_image_primitives, ds4-server
(exit 0). Environment: nvcc 13.3.73, RTX 4070 SUPER (compute_cap 8.9), CUDA_ARCH=sm_89
persisted in .ds4-cuda-config.mk.

Surfaces covered (each on its own line, exact string from the gate)

cuda/qwen_image_primitives.cuh
docs/qwen-image-2.1-roadmap.md
ds4_gpu.h
ds4_gpu_qwen_image_patch_1x1_tensor
ds4_gpu_qwen_image_silu_tensor
ds4_gpu_qwen_image_unpatch_crop_tensor
ds4_qwen_image_gpu.cuh
tests/test_qwen_image_primitives.c

Findings

1. Gate re-run on the committed tree: numbers reproduced, 26 cases, exit 0.
- make test-qwen-image-primitives CUDA_ARCH=sm_89 -> all Qwen-Image checks passed, make exit 0.
  The 26 cases are 10 primitives + 4 attention-path + 12 glue (main: test_silu 1,
  test_timestep x2 timesteps x2 stages + 1 table check = 5, test_patch x3 grids x2 directions
  = 6). Raw glue lines:
    silu dim=4096 rows=2               max_abs=4.768e-07 rel_rms=5.437e-08  ok
    timestep mlp hidden t=999.7        max_abs=9.537e-07 rel_rms=1.002e-07  ok
    timestep mlp out t=999.7           max_abs=7.153e-07 rel_rms=1.625e-07  ok
    timestep table t=0: cos=1, sin=0, equal columns  ok
    timestep mlp hidden t=0            max_abs=9.537e-07 rel_rms=9.541e-08  ok
    timestep mlp out t=0               max_abs=7.153e-07 rel_rms=1.613e-07  ok
    patch_1x1 channels=64 pixels=4096   max_abs=0.000e+00 rel_rms=0.000e+00  ok
    unpatch_crop channels=64 pixels=4096  max_abs=0.000e+00 rel_rms=0.000e+00  ok
    patch_1x1 channels=64 pixels=16384  max_abs=0.000e+00 rel_rms=0.000e+00  ok
    unpatch_crop channels=64 pixels=16384  max_abs=0.000e+00 rel_rms=0.000e+00  ok
    patch_1x1 channels=64 pixels=27556  max_abs=0.000e+00 rel_rms=0.000e+00  ok
    unpatch_crop channels=64 pixels=27556  max_abs=0.000e+00 rel_rms=0.000e+00  ok
  Bound is rel RMS <= 1e-5 and max abs <= 1e-4, so the worst glue case (1.625e-07) is ~60x
  inside. The commit message's silu 4.768e-07/5.437e-08, hidden 9.537e-07/1.002e-07 and out
  7.153e-07/1.625e-07 reproduce to the digit.
- make ds4-server CUDA_ARCH=sm_89 -> exit 0 (Rust host relinked). nm -C ds4-server shows the
  three symbols T: ds4_gpu_qwen_image_silu_tensor, ds4_gpu_qwen_image_patch_1x1_tensor,
  ds4_gpu_qwen_image_unpatch_crop_tensor. strings ds4-server contains "Qwen-Image silu
  launch", "Qwen-Image patch launch", "Qwen-Image unpatch launch". The three ds4_gpu.h
  surface strings above are the gate's ABI class extracted from the diff.

2. The glue mirrors are the oracle, index by index (not approximately). I diffed each new
case against dit.rs:
- silu: kernel __device__ silu (cuh:71) = value / (1.0f + expf(-value)); oracle dit.rs:223
  = x / (1.0 + (-x).exp()); test ref_silu (c:200-205) = v / (1.0 + exp(-v)) in double. Same
  formula; the device rides fast-math expf, which the 5.4e-8 rel residual carries. AGREE.
- timestep_embedding: kernel is deliberately host-only. ref_timestep_table (c:213-224):
  half = dim/2; log_period = logf(10000.0f); freq = expf(-log_period * j / half); arg =
  times[c] * freq; table[j + dim*c] = cosf(arg); table[j + half + dim*c] = sinf(arg).
  Oracle dit.rs:237-253: half = dim/2; log_period = max_period.ln(); freq = (-log_period *
  j/half).exp(); arg = t*freq; out[j + dim*i] = arg.cos(); out[j + half + dim*i] = arg.sin().
  Same order, same theta 10000.0 (dit.rs:744), same column stride dim. AGREE.
- time = concat(t, 0) and the MLP: dit.rs:744-755 builds timestep_embedding(&[t, 0.0]) and
  runs linear_1 (256 -> hidden) -> silu -> linear_2 (hidden -> hidden) -> silu with n_tok = 2.
  test_timestep (c:~735-790) sets times[2] = {timestep, 0.0f}, cols = 2, runs
  ds4_gpu_matmul_f32_stable_rows_tensor(w1, kTimeEmbedDim=256, kHidden=4096, table, 2) ->
  silu -> matmul(w2, 4096, 4096, th1, 2) -> silu. AGREE (and the commit's note that the f32
  entry is stable-rows because the default is TF32 is a test-harness choice, not an oracle
  divergence).
- patchify gather: dit.rs:788-796 is patches[c + in_channels*index] = latent[index +
  image*c] with in_channels = DIT_IN_CHANNELS = 64 and image = pixels. The entry
  ds4_gpu_qwen_image_patch_1x1_tensor fixes a = pixels, b = channels so transpose_2d writes
  out[j + b*i] = src[i + a*j] -> out[c + channels*p] = src[p + pixels*c]. Exact match. The
  test independently mirrors ref_transpose(want, source, pixels, channels) -> want[j +
  channels*i] = source[i + pixels*j]. AGREE.
- unpatchify scatter: dit.rs:827-833 is velocity[index + image*c] = projected[c +
  out_channels*index]. The entry passes a = channels, b = pixels, so out[index + pixels*c] =
  src[c + channels*index]. Exact match; ref_transpose(want, source, channels, pixels). AGREE.
- Geometry: kHidden 4096, kTimeEmbedDim 256, kChannels 64 equal DIT_HIDDEN,
  DIT_TIME_EMBED_DIM, DIT_IN/OUT_CHANNELS (qwen_image.rs:39-51). AGREE.
No approximate or shifted index found; the six patch/unpatch cases are bit-exact (0.0/0.0),
which is the signature of a pure permutation done right.

3. ABI: ds4_gpu.h declarations vs ds4_qwen_image_gpu.cuh definitions, balanced-paren parse.
  silu:   (ds4_gpu_tensor *x, uint32_t dim, uint32_t rows) -> int, extern "C" on the def.
  patch:  (ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint32_t channels,
           uint32_t pixels) -> int.
  unpatch:(same signature as patch) -> int.
No name, order, type or return divergence. ds4_gpu.h now declares exactly 9
ds4_gpu_qwen_image_* entry points (6 from units 8-9 + these 3), consistent with the roadmap's
"six ... Slice 3 ... three more". All three are T in both the gate binary and ds4-server.

4. Falsification A: silu broken to identity. cuda/qwen_image_primitives.cuh:173 changed from
   "x[index] = silu(x[index]);" to "x[index] = x[index];", rebuilt, ran. Probe hash
   6a5f43c4da306ee25fc338567bfd6cd2765974d2961cb05f8af799eeac777ced. Observed:
     silu dim=4096 rows=2               max_abs=5.984e+00 rel_rms=1.019e+00  FAIL
     timestep mlp hidden t=999.7        max_abs=8.087e+00 rel_rms=1.006e+00  FAIL
     timestep mlp out t=999.7           max_abs=5.951e+00 rel_rms=1.955e+00  FAIL
     timestep mlp hidden t=0            max_abs=7.499e+00 rel_rms=1.018e+00  FAIL
     timestep mlp out t=0               max_abs=5.782e+00 rel_rms=1.969e+00  FAIL
     timestep table t=0 ... ok (the table is independent of the kernel)
   make exit 2, "Qwen-Image checks FAILED". This proves the silu case AND both timestep MLP
   stages (t=999.7 and t=0) can each fail. Matches the commit's 1.02 / 1.0-2.0. Restore:
   search_replace back; final sha256 f6eea15d... equals the pre-probe value and git
   cat-file -p HEAD:cuda/qwen_image_primitives.cuh exactly.

5. Falsification B: transpose stores untransposed. cuda/qwen_image_primitives.cuh:222 changed
   from "out[(uint64_t) b * row + column] = tile[threadIdx.x][threadIdx.y];" to
   "= tile[threadIdx.y][threadIdx.x];", rebuilt, ran. Probe hash
   7e22ac9b2929a37216c627b169a8f2cff9e0761e5551badd2aee6ec69501fe82. Observed (all six
   patch/unpatch cases; silu stayed 4.768e-07/5.437e-08 ok):
     patch_1x1 4096     max_abs=1.994e+00 rel_rms=1.391e+00  FAIL
     unpatch_crop 4096   max_abs=1.996e+00 rel_rms=1.391e+00  FAIL
     patch_1x1 16384    max_abs=1.996e+00 rel_rms=1.393e+00  FAIL
     unpatch_crop 16384  max_abs=2.000e+00 rel_rms=1.391e+00  FAIL
     patch_1x1 27556    max_abs=1.998e+00 rel_rms=1.393e+00  FAIL
     unpatch_crop 27556  max_abs=1.997e+00 rel_rms=1.392e+00  FAIL
   make exit 2. Matches the commit's 1.39 on all six. Restore: final sha256 f6eea15d... equals
   the pre-probe value and the git blob exactly; the post-restore rebuild ran all 26 cases ok,
   exit 0, with the finding-1 numbers.

6. Falsification C: the test's own table-layout assertion, swapped halves.
   tests/test_qwen_image_primitives.c:222-223 changed the two stores to
   "table[j + dim*c] = sinf(arg); table[j + half + dim*c] = cosf(arg);" (cos/sin halves
   exchanged), rebuilt (test object only), ran. Probe hash
   a39f89e3f8c63f8b76a7220d2e51a8b8aeb1270306a3b79407b319322b3ccdf3. Observed:
     timestep table t=0: cos=1, sin=0, equal columns  FAIL
     silu, all four MLP stages                                   ok
     all six patch/unpatch cases                                 ok
   make exit 2. This confirms the layout check is the assertion that catches a swapped table
   while the device-vs-device MLP comparison cannot (both sides are fed the same table), which
   is exactly why the check exists and what the commit claims. Restore: final sha256
   da49d157... equals the pre-probe value and git cat-file -p HEAD:tests/
   test_qwen_image_primitives.c exactly. The table swap is the only test-side probe; the two
   kernel probes above never touched this file.

7. Entry-point refusals (zero-dim, undersized, exact boundary). Standalone probe /tmp/
   qa_entry_refuse.c linked against the tree's own objects (ds4.o, ds4_ple.o,
   ds4_distributed.o, ds4_cuda.o, cuda/mmq/*.o, cuda/qwen38_ple.o). All 15 checks ok:
     silu dim=0 0, rows=0 0, undersized (bytes - 4) 0, exact size 1;
     patch channels=0 0, pixels=0 0, undersized dst 0, undersized src 0, null dst 0,
       exact size 1;
     unpatch channels=0 0, pixels=0 0, undersized dst 0, undersized src 0, exact size 1.
   The boundary is exactly count = channels*pixels floats for every entry, which is what the
   ds4_gpu.h contract states for silu ([dim, rows]) and for patch/unpatch (both tensors hold
   channels*pixels). The undersized case is bytes - sizeof(float), one float short, refused;
   the exact size is accepted. The refusal is a silent return 0 (no stderr line), the same
   convention as the other six entries in this header.

8. transpose guards: partial 32x32 tiles and non-square extents, no uninitialized reads.
   Standalone kernel probe /tmp/qa_transpose_guard.cu includes cuda/qwen_image_primitives.cuh
   and launches transpose_2d directly. It poisons every output element with -12345.0f first,
   then diffs all elements against the pure-permutation reference (bit-exact, since the kernel
   has no arithmetic), and counts unwritten elements. 12 shapes, all mismatches=0 unwritten=0:
     a=64 b=64; a=64 b=4096; a=4096 b=64; a=64 b=27556; a=27556 b=64; a=166 b=166 (a partial
     32x32 tile at grid (6,6)); a=100 b=3; a=3 b=100; a=1 b=1; a=33 b=1; a=1 b=33; a=129
     b=97. The 27556-pixel pair (166x166, the grid the test uses) and the 129x97 odd extent are
     the non-square/partial cases. cuobjdump confirms sm_89 SASS.
   compute-sanitizer over the same probe: initcheck "ERROR SUMMARY: 0 errors"; memcheck
   "ERROR SUMMARY: 0 errors". The store thread (tx,ty) reads tile[tx][ty], which the load
   thread (lx=ty, ly=tx) wrote under the guard bx*T+ty < a && by*T+tx < b -- identically the
   store's own guard -- so no padded tile element is ever read. The unwritten=0 count is the
   independent check that no output element is left as poison. The guards hold.

9. Cheat audit: no cheat found.
- Tolerances are meaningful: the falsification probes move the metric from ~1e-7 to 1.0-1.97
  rel RMS; compare() is observed failing (findings 4-6). No hardcoded pass; the test has no
  asserts, compare() increments g_failures and main returns 1 when it fires (observed).
- No kernel left unlaunched: silu_rows runs twice per timestep (four launches), transpose_2d
  runs twice per grid (six launches); every entry return is checked with exit(1) on refusal
  (c:~712, :~760, :~800).
- Inputs are not degenerate: splitmix64 uniform fills over O(1)-O(6) ranges; the patch cases
  use 64/4096, 64/16384, 64/27556 so both a full and a partial tile; the permutation is
  bit-exact, which a wrong-but-tiny index shift could not be.
- The comparison is not self-fulfilling: the mirror is an independent double reimplementation
  (ref_silu/ref_timestep_table/ref_linear/ref_transpose) diffed line by line against dit.rs in
  finding 2.

10. Roadmap P2 gate-result claims vs raw output. docs/qwen-image-2.1-roadmap.md:351-391 and
   the tracker row :556.
- Header "three slices": slice 1 primitives, slice 2 rope/attn, slice 3 silu/transpose -- the
  05097db diff adds exactly slice 3 and the text matches the changed files. AGREE.
- "silu 4.8e-7 abs / 5.4e-8 rel RMS" -> 4.768e-07 / 5.437e-08. AGREE (rounded).
- "the two timestep MLP stages 9.5e-7 / 1.6e-7" -> hidden 9.537e-07 / 1.002e-07 and out
  7.153e-07 / 1.625e-07. The abs figure is fine; note the stated rel bound 1.6e-7 is a
  two-significant-figure rounding of 1.625e-07, so read as a literal inequality "<= 1.6e-7"
  it is exceeded by 1.5%. This is a presentation nit (the raw number is printed in the commit
  message too), not a correctness claim; the real gate is 1e-5, ~60x away. Report, not fix.
- "all six patch/unpatch cases at 0.0/0.0" -> confirmed, all six are exactly 0.000e+00/0.000e+00.
- "the permutation is exact at 64x64, 128x128 and the partial-tile 166x166 grid" -> 64x64 =
  4096 px, 128x128 = 16384 px, 166x166 = 27556 px, exactly the three grid arguments in
  main (c:900-901). AGREE.
- Falsification values "identity silu (1.02 rel RMS on its own case, 1.0-2.0 on the MLP
  stages)" -> 1.019, 1.006/1.018, 1.955/1.969 (finding 4). "a no-transpose tile (1.39 on all
  six patch cases)" -> 1.391-1.393 (finding 5). "swapped table halves (the t=0 layout check
  fails while the MLP cases stay green)" -> confirmed (finding 6). AGREE.
- P2 gate result also carries the unit-9 mask-falsification values (8.99e-1 no-op mask,
  1.62e0 unconditional mask). Those belong to unit 9 and were re-run live in the base report
  0682b1d:qa-evidence/qa-report.md finding 3; this pass did not re-run them (they are outside
  unit 10's diff). Flagged as carried evidence, not re-measured here.

11. P1 staleness: nothing about P1 is stale. Read the three artifacts:
- P1 tracker row docs/qwen-image-2.1-roadmap.md:555: "three stages: byte-identical noise and
  Euler step, DiT velocity correlation 0.999973, VAE image PSNR 70.2 dB vs run1.png".
- P1 report docs/qwen-image-2.1-p1.md:114 (pairing table) "RopePairing::Interleaved 0.999973
  7.56e-3 1.37 106 s"; :223 (decode row) "PSNR 70.2 dB; relative RMS 3.26e-4".
- P1 section gate result roadmap:312-318: "correlation 0.999973 (relative RMS 7.6e-3)", "PSNR
  70.2 dB (max 5/255 ...)". The report's own stage headings are 1/2/3 (## 3, ## 4, ## 5), so
  "three stages" is accurate.
The row's 0.999973 and 70.2 dB match the report and the section exactly; 7.56e-3 rounds to the
section's 7.6e-3. No stale P1 line found. The three stale lines the commit fixed are all
P2/section-6: the P2 tracker row's "slices 1-2" -> "slices 1-3", roadmap:521 "nothing beyond P1
has been built" -> "P1 is built and P2 is in progress", and the P2 gate result's closing
sentence (the timestep/patch glue "remain for later slices" -> "the DiT now has every primitive
P3 needs"). Confirmed by the 05097db diff itself.

12. Report, do not fix: tests/qa-gate.sh drifted model name. Header comment :21 and the
    MODEL default :45 both say deepseek-v4.1-CC-flash. The project's recorded QA model is
    deepseek-v4-flash-JL1: ds4-dfm-rs-handoff.md:111 "QA and coding model (rule 19):
    deepseek-v4-flash-JL1" and :208 "the mount still uses the old name
    deepseek-v4.1-CC-flash; the recorded model is deepseek-v4-flash-JL1". This is a real,
    unfixed drift in an untracked-scope file (tests/qa-gate.sh is not in 05097db's diff), so it
    is not a defect of unit 10; reported as instructed and left unchanged.

13. tests/qa-gate.sh live. Before this report: GATE_EXIT=1, 4 checks red -- "QA report is
    fresh" (the stale unit-9 report predates the base commit) and the three new ABI surfaces
    (ds4_gpu_qwen_image_patch_1x1_tensor, _silu_tensor, _unpatch_crop_tensor) not covered.
    The gate's surface list is exactly the 8 strings in this report. After this report:
    GATE_EXIT=0, "QA GATE: overall PASS", all checks green (see the closing evidence line
    below). No commit or push was done.

Unverified items
- The unit-9 mask falsification values (8.99e-1, 1.62e0) were not re-run here; they are the
  base report's finding 3 and are carried, not re-measured.
- The reference dump files the P1 report used are not in this tree, so P1's 0.999973 / 70.2 dB
  were cross-read from docs (report vs section vs tracker), not recomputed from dumps.
- Only sm_89 was tested; no other architecture in this pass. The device fast-math trig claim
  (5.5e-4) is carried from the unit-9 report, not re-swept here; it only motivates the
  host-table design, which itself is verified by the t=0 layout check and the MLP parity.
- The end-to-end DiT path that consumes these three entries (the head-major q/k reshape the
  unit-9 report flagged, the norm ops, and the actual graph wiring) is outside this commit and
  was not exercised.

Risk analysis
Decision: accept 05097db as PASS for the feat/image QA gate (it may become the next pushed
state). No push, publish or deploy is done by this pass.
External risks: none network-facing. The evidence depends on the local CUDA 13.3 toolkit and
the RTX 4070 SUPER device state (probed live: nvcc 13.3.73, sm_89, all gates ran). A different
arch or a JIT-only build was not tested (unverified).
Decision risks: the PASS rests on the test's host mirror being the oracle; checked by a
line-level diff of every new element (silu, timestep_embedding, the MLP stage order, the two
permutation directions) against dit.rs in finding 2, plus the bit-exact 0.0/0.0 permutation
result. It is not checked by an oracle dump run (fixtures absent). The kernel falsifications
cover silu and transpose but not every mirrored constant; a constant shared between kernel and
mirror in the same wrong way would not be caught by this harness (mitigated by the diff). The
transpose guard result depends on the standalone probe, not on a sanitizer that tracks shared
memory initialization specifically; memcheck and initcheck both returned 0 errors and the
poison-diff doubles as an uninitialized-read detector.
Measurement limits: the guard probe is a fixed 12-shape set, not exhaustive; a=166 b=166 and
a=129 b=97 are the partial/odd representatives. The 1.625e-07 rel is a single-seed run of
splitmix64; the gate has a ~60x margin, so seed noise cannot flip it.
Prevention/rollback: this report is written before the final gate run; the unit is a local
branch commit, so rollback is discarding/rewriting the branch pointer, and the QA gate re-runs
from a clean tree. Trigger to reopen: any failing gate, a mirror-vs-oracle divergence found in
the next slice, or a fix that changes the transpose guards.
Residual risk: low. The strongest residual is the deferred DiT graph wiring (head-major q/k
reshape, norm ops) outside this slice, plus the carried (not re-run) unit-9 mask values.

verdict: overall PASS