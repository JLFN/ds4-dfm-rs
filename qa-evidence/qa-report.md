QA report: feat/image — the Qwen-Image-2.1 port reopen, P0 (identification and
layout contracts). Independent falsification pass, rule 19. /data/imagegen was
read read-only (files and read-only git); nothing was written and nothing was
started or stopped there. No push, no commit, no source edit; the only file
written in the repository is this report.

Verdict: PASS. The two defects the previous pass found are fixed and re-verified
against /data/imagegen, every remaining docs difference from the base is a
deliberate reopen edit and not a lost correction, all four required test suites
pass on this tip, and the shipped binary behaves as the roadmap claims.

Unit identification

Repository /data/ds4-dfm-rs, branch feat/image. HEAD 8bcb407 (docs, amended) on
fe25148 (code); base fork/feat/image = 9bdaf45. The branch was rewritten by a
concurrent session during the previous QA pass (reflog: base 9bdaf45; code
c6bdcee -> fe25148; docs 14eed3a -> 3ed060c -> 715d6b8 -> amended 8bcb407). The
delta 715d6b8..8bcb407 is docs-only (docs/qwen-image-2.1-plan.md,
docs/qwen-image-2.1-roadmap.md): the two fixes below. Working tree clean except
this report. Base ref fork/feat/image is still 9bdaf45, so tests/qa-gate.sh does
not short-circuit.

Artifacts, re-hashed here (all three match the expected values):
  DiT      /data/imagegen/models/diffusion_models/qwen-image-2.1-Q6_K.gguf
           sha256 a3a0d39bb03cda26302fc048b49d019baaea1c381cbda3994f6f2a7826344fb9
  VAE src  /data/imagegen/models/vae/qwen_image_2.1_vae_bf16.safetensors
           sha256 bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9
  VAE out  misc/scratch/p0/vae-decode-bf16.gguf
           sha256 d3feefed174e69d51f380c71bb500ab92033d85fc83f5d87c9811a8775725372

The two defects, and their resolution

DEFECT 1 (was blocking) — the false ceiling claim in docs/qwen-image-2.1-plan.md
appendix A.2. FIXED at 8bcb407.

Previous text (removed): "The bridge advertises a maximum edge (1152 in the
deployed configuration) ...". I re-verified that claim was false against
/data/imagegen (read-only):
  - crates/bridge/src/main.rs:973: `const MAX_EDGE: u32 = 1536;`; :976 enforces
    `if w > MAX_EDGE || h > MAX_EDGE`.
  - `grep -c max_edge imagegen.toml` = 0; the key does not exist.
  - crates/bridge/README.md:41-43: "width and height must be divisible by 32 and
    at most 1536 on either edge."
Current text (restored) now says the live value is not what the record says, that
`max_edge = 1152` was real in `imagegen.toml` at `d5fcfe0` and is gone at HEAD,
and that `const MAX_EDGE: u32 = 1536` (main.rs:973, README.md:41) governs now,
hardcoded above what the pinned layout can serve. This is TRUE on every point I
checked. The A.2 section is now byte-identical to the text base 9bdaf45 carried
(verified with `diff <(git show 9bdaf45:...A.2) <(current A.2)` -> identical),
and the closing requirement sentence "The live bridge does the opposite." is
back. The unit's own decision doc already stated 1536 correctly
(docs/qwen-image-2.1-decision.md:271-276); plan and decision now agree.

DEFECT 2 (minor) — the roadmap test count. FIXED at 8bcb407.
docs/qwen-image-2.1-roadmap.md now reads "13 module unit tests cover the contract
tables, the permute index math, the byte math and the placement rules (the
`--lib qwen_image` filter reports 15, two of them batch tests that match the
name)". Verified: the qwen_image module has 13 unit tests (9 in qwen_image.rs + 4
in convert.rs); `cargo test -p ds4-core --lib qwen_image` reports 15 because two
batch tests (batch::tests::continuous_admit_keeps_qwen_image_payload_alive and
batch::tests::qwen_image_probe_and_hash_hide_the_native_layout) contain the name.
The statement is now true.

Regression hunt: every remaining docs difference from 9bdaf45

`git diff 9bdaf45..HEAD -- docs/` touches five files. Each difference checked:

- docs/qwen-image-2.1-plan.md: status header rewritten to "proposal, being built"
  (deliberate reopen); the phase-S/decision "port closed" paragraph rewritten to
  "proceeds on ownership alone" (deliberate reopen); the section 3 preamble that
  framed the decision as the authority removed (deliberate reopen, the decision
  is superseded); section 11 build-order wording updated (deliberate reopen); A.2
  restored to the base text (the DEFECT 1 fix, true). No lost correction.
- docs/qwen-image-2.1-roadmap.md: full reopen rewrite. The removed content is the
  "closed" framing, the phase-closure table and the "decision done - stop"
  tracker row. The added content restores the phase bodies (P0-P8) as proposals,
  the phase map, the success criteria (status column dropped) and the P0 gate
  result. Every factual assertion in the added text was checked: the P0 gate
  result facts (297/229/68, 134, zero metadata keys, 132 decoder / 102 encoder,
  518 096 424 bytes, three artifact hashes, 62 convs at (1,3,3) and 14 at
  (1,1,1), the leading gamma spanning the input width, the eleven named refusals)
  are all confirmed true below; the phase-S figures (544 ms/step 18.2%, 4.855 ->
  2.945 s 1.65x, 5.5% host bubble, 8.0% offload, 7921 of 11894 MiB) match
  docs/qwen-image-2.1-phase-s.md and the decision. The stale figures a prior pass
  had removed stay removed (2977.7, 1033.5, 848.2, 407.0, 176.8, 1.9x, 1.912 are
  absent from all four port docs), and the live ones (1.910 s, 1.85x, 34.6/34.7,
  2992.5, 133.7, 112.7) are present and consistent. The tracker's S row is
  unchanged from the base and still records M1/M2/M3, matching phase-s's "M1 and
  M2 are settled in the roadmap's tracker".
- docs/qwen-image-2.1-decision.md: only the status header changed (marked
  "superseded 2026-10-01", original status kept below). The section 7 ceiling text
  and the section 2 checked-row (the earlier corrections) are untouched and still
  say the live bridge accepts 1536 and that max_edge = 1152 was d5fcfe0-only.
- docs/qwen-image-2.1-phase-s.md: header decision link dropped; the resolution
  paragraph reworded for the reopen. No numeric claim changed; the numbers are
  unchanged by the reopen.
- docs/README.md: the three image-generation index rows relabeled (plan
  "Proposal", roadmap "P0 built", decision "Superseded"). True.

Nothing in the diff is a lost correction and nothing is a false statement, apart
from the two non-blocking wording notes recorded under minor observations.

What was checked and how (independent verification)

1. Required test suites, run on 8bcb407 with the two artifact env vars set. All
   four pass:
     DS4_QWEN_IMAGE_DIT=... DS4_QWEN_IMAGE_VAE=... cargo test -p ds4-core --test qwen_image
       running 8 tests ... test result: ok. 8 passed; 0 failed
     cargo test -p ds4-core --lib qwen_image
       15 passed; 0 failed (13 module + 2 batch name matches)
     DS4_QWEN_IMAGE_DIT=... DS4_QWEN_IMAGE_VAE=... cargo test -p ds4-server --test image_cli
       running 4 tests ... test result: ok. 4 passed; 0 failed
     cargo test -p ds4-server --lib image_cli
       running 2 tests ... test result: ok. 2 passed; 0 failed

2. Independent artifact numbers, from my own GGUF/safetensors parsers written
   from the format spec in /tmp (not the unit's dump_gguf.py):
   - DiT GGUF: v3, 297 tensors, n_kv 0 (no metadata keys, so no
     general.architecture), Q6_K=229 BF16=68, sum of tensor bytes 5876533248,
     data_pos+maxend == filesize, no overlaps. Names/dims equal the contract:
     9 top-level + 9 per block x 32 blocks; Q6_K 5 + 7*32 = 229, BF16 4 + 2*32 =
     68.
   - VAE source safetensors: 238 BF16 tensors (132 decoder, 102 encoder, conv1,
     conv2); 76 5-D conv weights, 62 at (1,3,3) and 14 at (1,1,1).
   - Converted VAE GGUF: v3, 134 tensors, all BF16, 518096424 data bytes,
     dir_end 11405, data_pos 11424, data_pos+maxend == filesize, no overlaps;
     tensor set is exactly decoder.* + conv2.* (no encoder, no conv1); every dim
     equals my independent torch-to-ggml map.
   - Q6_K byte math: 210 bytes per 256-element block, so
     elements.div_ceil(256)*210; BF16 2/elem; F32 4/elem. Right.

3. Conv re-index math, falsified independently. A from-first-principles ggml
   row-major ne-index reference for the declared dims {kW,kH,kT,IC*OC} (conv3d)
   and {kW,kH,IC,OC} (conv2d) is bijective and equals the unit's permute_conv3d /
   permute_conv2d for every tested shape. The pinned reference source on this
   host agrees: ggml_extend.cpp ggml_ext_conv_3d derives OC = w->ne[3]/IC;
   /tmp/sdref/wan_vae.hpp:31 tests weight.ne[2]==1 && weight.ne[3]==IC*OC and
   :37-42 builds the weight ggml_new_tensor_4d(kW,kH,kT,IC*OC); conv2d {kW,kH,IC,
   OC} is the standard ggml_conv_2d layout. Rule proved correct.

4. Regeneration. The committed tool reproduces the recorded file byte-for-byte:
   `qwen-image-vae-gguf <src> /tmp/qa-vae.gguf` reports tensors=134
   bytes=518096424 dropped=104 source_tensors=238, and `cmp` against
   misc/scratch/p0/vae-decode-bf16.gguf is identical (same sha256).

5. Shipped binary. `make -n ds4-server` says "up to date" (the delta since my
   previous run is docs-only and the binary is newer than every source), so the
   existing ./ds4-server is current. Re-run:
   - clean `--check-config` exits 0, "qualified: dit_tensors=297 dit_q6_k=229
     dit_bf16=68 vae_tensors=134 refusals=11" with the deployed placement,
     stdout parses as valid JSON.
   - the same command plus all eleven AR controls exits 2 with eleven
     "error: <flag> is an autoregressive control: ..." lines and creates no
     directory at the given --kv-disk-dir.
   - naming artifacts without --check-config exits 2 with the P5 message.
   The concurrent ordering fix (image branch before the AR preconditions) is
   confirmed by the reachable --mtp refusal.

6. Error path: a malformed safetensors is refused at the contract check and
   leaves neither the output nor a .partial file.

Known pre-existing failure (not this unit's)

`cargo test -p ds4-server --lib` fails only in
cache_identity::tests::bounded_sidecar_and_ple, at
crates/ds4-server/src/cache_identity.rs:417 (assert_ne!(old, changed)). This unit
is not its cause:
  - the unit's diff does not touch crates/ds4-server/src/cache_identity.rs, and
    the file is byte-identical at 9bdaf45 and HEAD;
  - it fails when run alone, so the unit's new tests and parallelism are
    irrelevant;
  - it reproduces on a clean detached worktree at 9bdaf45;
  - the mechanism is coarse kernel file timestamps: the test rewrites part.bin
    with the same 4 bytes and asserts the stat snapshot changed, but an immediate
    equal-length rewrite leaves st_mtime_ns/st_ctime_ns unchanged (a python repro
    shows the timestamps differ only after a ~50 ms sleep). part.bin is a weights
    input with no content hash, so nothing else can differ. It is an environmental
    flaky test, pre-existing.

Minor observations (non-blocking)

- docs/qwen-image-2.1-roadmap.md section 6 says "nothing has been built", while
  the same document's header says "Nothing here is implemented except P0" and the
  tracker marks P0 done. Read strictly it is loose; read as scoped to the size
  estimates for P1-P8 (no model or kernel code has been built), it is true. The
  text is a deliberate reopen edit of the base's "nothing was built"; I judged it
  a wording nit, not a false claim about the deployment or the code.
- docs/qwen-image-2.1-plan.md:251-252 (unchanged from base) and the roadmap's P0
  work-item 2 say "this tree reads GGUF only - grep finds no safetensors reader".
  After this unit a safetensors header parser does exist in
  crates/ds4-core/src/qwen_image/convert.rs, but it is the offline converter, not
  a runtime reader, and the sentence describes the state that motivated the
  converter decision. Pre-existing, not introduced here, and not false of the
  runtime.

Surfaces covered (every surface qa-surfaces.txt lists)

crates/ds4-core/src/bin/qwen-image-vae-gguf.rs
crates/ds4-core/src/lib.rs
crates/ds4-core/src/qwen_image/convert.rs
crates/ds4-core/src/qwen_image.rs
crates/ds4-core/tests/qwen_image.rs
crates/ds4-server/src/bin/ds4-server-rs.rs
crates/ds4-server/src/image_cli.rs
crates/ds4-server/src/lib.rs
crates/ds4-server/tests/image_cli.rs
docs/qwen-image-2.1-decision.md
docs/qwen-image-2.1-phase-s.md
docs/qwen-image-2.1-plan.md
docs/qwen-image-2.1-roadmap.md
docs/README.md

New surfaces to be QA'd (deepseek-v4.1-CC-flash):
pub const ALL
pub const AR_CONTROLS
pub const DEFAULT_PLACEMENT
pub const DIT_AXES_DIM
pub const DIT_BF16_COUNT
pub const DIT_CONTEXT_DIM
pub const DIT_HEAD_DIM
pub const DIT_HEADS
pub const DIT_HIDDEN
pub const DIT_IN_CHANNELS
pub const DIT_INTERMEDIATE
pub const DIT_LAYERS
pub const DIT_MODULATION
pub const DIT_NORM_EPS
pub const DIT_OUT_CHANNELS
pub const DIT_Q6K_COUNT
pub const DIT_ROPE_THETA
pub const DIT_TENSOR_COUNT
pub const DIT_TIME_EMBED_DIM
pub const ENGINE
pub const TYPE_BF16
pub const TYPE_F32
pub const TYPE_Q6_K
pub const VAE_DEC_DIM
pub const VAE_DECODE_TENSOR_COUNT
pub const VAE_DIM_MULT
pub const VAE_LATENT_MEAN
pub const VAE_LATENT_STD
pub const VAE_NORM_EPS
pub const VAE_OUT_CHANNELS
pub const VAE_SCALE_FACTOR
pub const VAE_TEMPORAL_DOWNSAMPLE
pub const VAE_TEMPORAL_UPSAMPLE
pub const VAE_Z_DIM
pub enum ConvertError
pub enum ImageArtifactKind
pub enum ImageDevice
pub enum ImageEngine
pub enum ImageError
pub enum ImageLevel
pub enum ImageModule
pub enum ParamTier
pub fn ar_control
pub fn bytes
pub fn conv2d_dims
pub fn conv3d_dims
pub fn convert_vae
pub fn dit_contract
pub fn elements
pub fn find
pub fn has_errors
pub fn identify_dit
pub fn identify_vae
pub fn line
pub fn map_source_dims
pub fn matches
pub fn may_load
pub fn name
pub fn permute_conv2d
pub fn permute_conv3d
pub fn read_header
pub fn refuse_text_artifact
pub fn report
pub fn resolve_image_plan
pub fn to_json
pub fn token
pub fn type_count
pub fn vae_decode_contract
pub fn vae_decoder_dims
pub struct ArControl
pub struct ConvertReport
pub struct ImageArtifactId
pub struct ImageEffective
pub struct ImageIssue
pub struct ImagePlan
pub struct ImageQualified
pub struct ImageRequest
pub struct ImageRequested
pub struct ImageTensor
pub struct ModulePlacement
pub struct Safetensors
pub struct SourceTensor

Claims not verified, and whether the gap is acceptable

1. No GPU kernel was run; step time, resolution ceiling and VRAM peak are P6's
   ledger and the plan's qualified note says so. Acceptable at P0.
2. The phase-S figures are taken from the phase-S report and the decision (a
   prior unit's recorded run); I did not re-measure them. Acceptable: this unit
   changes no measurement and rerecords them consistently.
3. The conv permutation is proved against the declared ggml layout and the pinned
   reference's weight construction, not by executing a conv end to end.
   Acceptable at P0 (contracts only, no kernel).

verdict: overall PASS
