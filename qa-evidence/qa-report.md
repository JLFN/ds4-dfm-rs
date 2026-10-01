QA report: feat/image — the budget-derived placement rules (Qwen-Image-2.1 P0),
re-verification of the amended unit. Independent falsification pass, rule 19.
Unit: commit 5a5c339 "feat(image): derive the placement rules from the device
budget" (amended from c698939, message only), branch feat/image, one commit ahead
of fork/feat/image = c4c75c7; HEAD b95e6df is the QA evidence commit on top. Tree
identity checked with git: git rev-parse c698939^{tree} 5a5c339^{tree} both return
7c271a53567e826227329d9c9dfadd39ac7a30a7 and git diff c698939 5a5c339 is empty,
so the content is exactly what the previous pass verified. (c698939 and the
intermediate amend 901c3c0 are still reachable and share that tree.) The only
change between their messages is the 482 MiB sentence; see the findings below.
Refs inspected this pass: 5a5c339, c698939, the reflog, both commit messages, and
the tree diff. Refs inspected in the previous pass and still governing the tree:
f37e034, c4c75c7, the full diff c4c75c7..c698939, misc/scratch/p1/refdump/run1.log
(256x256), misc/scratch/phase-s/ref-run.log (1024x1024), the two real artifacts,
docs/qwen-image-2.1-plan.md and docs/qwen-image-2.1-roadmap.md. Nothing was
committed or pushed by me; the only file written is this report.

Surfaces covered (each on its own line, exact string from the gate)

crates/ds4-core/src/qwen_image.rs
crates/ds4-server/src/bin/ds4-server-rs.rs
crates/ds4-server/src/image_cli.rs
crates/ds4-server/tests/image_cli.rs
docs/qwen-image-2.1-plan.md
docs/qwen-image-2.1-roadmap.md

All six in the unit diff (283 insertions, 33 deletions across exactly those
files), and all six identical between c698939 and 5a5c339.

Findings from the previous pass, re-checked

D1 (VAE footprint sourced from the wrong artifact) FIXED.
FOOTPRINT_VAE_WEIGHTS_MIB is 495. The decode-only GGUF this engine loads
(misc/scratch/p0/vae-decode-bf16.gguf, sha256 d3feefed...5372) holds 518096424
bytes of BF16 tensor data = 494.095 MiB, so 495 is the round-up. The comment now
states this explicitly and distinguishes the reference's own 128-tensor VAE
figure (482.81 MiB, run1.log:244). All four weight/buffer constants are now
rounded up from their measured sources and each names its source:
  FOOTPRINT_TEXT_ENCODER_MIB  4303  <- 4302.32 / 4302.33 (run1.log:160, :195)
  FOOTPRINT_DIT_WEIGHTS_MIB   5605  <- 5876556448 bytes = 5604.32 MiB (artifact)
  FOOTPRINT_DIT_COMPUTE_MIB   2318  <- 2317.45 (phase-s ref-run.log:228)
  FOOTPRINT_VAE_WEIGHTS_MIB    495  <- 518096424 bytes = 494.10 MiB (GGUF)
  FOOTPRINT_VAE_DECODE_FLOOR_MIB 3973 (floor, unchanged)
  FOOTPRINT_DEFAULT_BUDGET_MIB 11894 (unchanged)
Every rounded value >= its source. Re-derived sums, all confirmed against the
string printed live by the rebuilt binary:
  te + DiT            = 4303 + 5605 + 2318              = 12226 MiB
  DiT + VAE           = 5605 + 2318 + 495 + 3973         = 12391 MiB
  all three           = 12226 + 495 + 3973               = 16694 MiB
16694 MiB = 16.30 GiB, and both 12226 and 12391 exceed the 11894 MiB default,
so the two refusals still stand at the default.

D2 (A.1 mislabelled the measurement resolution) FIXED.
docs/qwen-image-2.1-plan.md A.1 now reads "The three modules' measured footprints
total 16694 MiB (16.3 GiB) at 1024 square, the generation these rules were
measured on ... The DiT's compute buffer is the resolution-dependent term: 34.66
MiB at 256 square against the 2318 MiB used here, so a budget quoted at another
resolution needs that figure requoted." This is true: the 2318 MiB compute buffer
and the 3973 MiB floor come from the 1024-square reference run
(misc/scratch/phase-s/ref-run.log: width 1024, height 1024, generate_image
1024x1024, compute buffer 2317.45 MB at line 228), and the 256-square run reports
34.66 MB (misc/scratch/p1/refdump/run1.log: width 256, height 256, generate_image
256x256, line 233). 16694/1024 = 16.30, so "16.3 GiB" holds. The stale "at 256
square" phrase is gone (the only remaining "256 square" is the correct
resolution-dependent caveat).

Acceptance, live, on the rebuilt binary (from the previous pass; the tree is
unchanged, so these still hold)

  ./ds4-server --check-config --image-dit <dit> --image-vae <vae> \
      --image-placement te=cuda0:vram,diffusion=cuda0:vram,vae=cuda0:vram
  EXIT=2, both codes present, each naming the footprint and the budget:
    error: pinning te=cuda0,vram diffusion=cuda0:vram vae=cuda0:vram on cuda0
      needs about 16694 MiB and the budget is 11894 MiB: ... (image_te_vram_unsupported)
    error: ... same 16694/11894 ... (image_double_pin_unsupported)
  the same request with --max-vram 140
  EXIT=0, neither code present, artifacts identified (dit_tensors=297 dit_q6_k=229
  dit_bf16=68 vae_tensors=134, refusals=11). Matches the commit message.

Default behaviour unchanged (no --max-vram)
  - te=cuda0:vram (DiT at the default cuda0:vram): EXIT=2,
    image_te_vram_unsupported, "needs about 12226 MiB and the budget is 11894".
  - vae=cuda0:vram (DiT at the default cuda0:vram): EXIT=2,
    image_double_pin_unsupported, "needs about 12391 MiB and the budget is 11894".
The old pass set was {no te in VRAM} AND {not both DiT and VAE in VRAM}; every
such configuration pins 0, 4468 or 7923 MiB per device at the default budget,
all <= 11894, so no configuration that used to pass now fails. No regression.

Tests (from the previous pass; tree unchanged)
  cargo test -p ds4-core --lib qwen_image        -> 39 passed, 0 failed.
  cargo test -p ds4-server --test image_cli      ->  5 passed, 0 failed.
  cargo test -p ds4-server --lib image_cli       ->  3 passed, 0 failed.
  cargo test -p ds4-core --test tokenizer        -> FAILED,
    tokenizer_families_match_c_oracle (qwen35 specials family=12 vs the C oracle's
    11, crates/ds4-core/tests/tokenizer.rs:174). Pre-existing and untouched:
    git diff --name-only c4c75c7 5a5c339 lists no tokenizer path (the only crates
    file in the diff is crates/ds4-core/src/qwen_image.rs).
Commit trailer: "Unit: 6 complete" is the last body line of 5a5c339 (verified
this pass; unchanged by the amend).
Parsing on the fresh binary: --max-vram twelve / 0 / 12.5 all exit 2; the flag is
in the usage text ("... [--image-offload] [--max-vram GiB]"); the value reaches
the placement as MiB (12 GiB = 12288 admits te+DiT at 12226; 11 GiB refuses).

No false DGX/resolution claims. A.1 says the two measured rules are "measurements
of THAT card" (the 12 GB RTX 4070 SUPER) and that a DGX Spark's 128 GB "holds
resident" the 16.3 GiB stack - a capacity comparison, not a Spark measurement; no
Spark number exists. The roadmap P0 placement line states no resolution and no
DGX figure. (plan.md:111-112 is a pre-existing capacity statement, not from this
unit.)

Minor, non-blocking imprecisions (findings, not defects)

- The commit message previously called the reference's figure "the reference's
  own 482 MiB safetensors" (the safetensors file is 675509688 bytes = 644.38 MiB;
  482.81 MiB is the loaded params buffer, run1.log:244). FIXED by this amend:
  5a5c339 now reads "not the 482 MiB the reference's own 128-tensor scope
  reports", which is accurate. The message diff c698939..5a5c339 is confined to
  that sentence.
- Unchanged from the earlier pass and outside this amend: the new per-device,
  budget-based logic relaxes three configurations that the old code refused
  (te=cuda0:vram with the DiT on host; DiT and VAE pinned on different cuda
  devices; a VRAM tier on the Cpu device, te=cpu:vram / vae=cpu:vram). None is a
  configuration that used to pass, so none is a regression; recorded for the
  operator as the intended direction of the change.

Unverified items (findings, not passes)

- That the reference's params-buffer figure (482.81 MB) is meant to be
  byte-comparable to this engine's GGUF; I read both and the artifact sizes but
  did not reconcile the reference graph's 128 tensors against the converter's
  134. D1 is resolved either way because the constant now uses this engine's own
  artifact.
- I did not re-run the full serialized workspace suite; only the four gates the
  task names. The amend touched no test and no other crate.

verdict: overall PASS
