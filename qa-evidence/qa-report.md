QA report: feat/image — the Qwen-Image-2.1 stop decision (docs unit), pass 3

Third pass, on the second amendment. This pass verifies the two post-pass-2
corrections, confirms nothing else moved, and re-runs the gate. The substantive
checks from passes 1 and 2 are retained below and still stand. Independent
falsification pass, rule 19. The deployment at /data/imagegen was read read-only
(nothing written, nothing started or stopped there).

Unit identification

Repository /data/ds4-dfm-rs, branch feat/image.
HEAD 082335fdbbe6a1d98c01c691e98d56f5abce2998 ("docs(image): close the port
after pricing its motives"). Amendment history: 78fab2b (pass 1) -> d35aeff
(pass 2) -> 082335f (this pass). The diff d35aeff..082335f is
docs/qwen-image-2.1-decision.md only, 2 insertions / 2 deletions; the other four
documents and docs/README.md are unchanged across both amendments. The worktree
matches HEAD for docs/ (git diff HEAD -- docs/ is empty); the only modified file
is this report.
Base ref fork/feat/image = d92e5614544ddd77efb74a4619580aa9ffa41c88 (== d92e561).
merge-base with origin/main = 7b6448b9985ae81f9089ea68c19f3a936e44fdf7.
git diff --name-only d92e561..HEAD: docs/README.md,
docs/qwen-image-2.1-decision.md, docs/qwen-image-2.1-phase-s.md,
docs/qwen-image-2.1-plan.md, docs/qwen-image-2.1-roadmap.md.
No remote-writing git command was run; origin stays read-only.

Handoff record: the trailer "Unit: 1 complete" IS present on d92e561
(git log -1 --format=%B d92e561 ends with it), so the handoff's claim that the
unit marker is missing is stale. This has been true in every pass.

The two changes under test (d35aeff -> 082335f)

1. Section 7 item 3 grammar: the sentence now reads "The deployment's own record
   has 1152 square exceeding the 240 s read timeout at 40 steps
   (architecture.md:148-152); whether the flag moves that tier inside the timeout
   is a deployment measurement this unit did not make." The pass-2 bald phrasing
   ("has 1152 square do not fit") is gone.
2. Comparison table, last row: the unreachable-pool step time is now 1.910 s
   (was 1.912 s). The 53.5 s and -82.4 s cells are unchanged.

Check A — the two fixes are present and correct

git diff d35aeff 082335f shows exactly two changed lines, nothing else.

- Grammar: the new sentence is grammatical and states the true fact. It is
  consistent with the source it cites: /data/imagegen/docs/architecture.md
  lines 148-152 say "Requests at 1152 square do not fit open-grok's 240 s read
  timeout at 40 steps (275.88 s here)". 275.88 s exceeds 240 s, so "exceeding"
  is correct. The old string "do not fit" no longer appears in the decision;
  "exceeding the 240 s" appears once.
- Step time: 2.945 s (the --diffusion-fa step) minus the 1034.7 ms glue pool =
  1.9103 s -> 1.910 s. The new value is exactly consistent with the amended
  pool. 1.910 x 28 = 53.49 -> 53.5 s (unchanged) and 135.9 - 53.5 = 82.4 s
  (unchanged), so the row is now internally consistent. The old "1.912 s" no
  longer appears.

Check B — no other number moved

git diff d35aeff 082335f is the two lines above and nothing more; only
docs/qwen-image-2.1-decision.md changed between the two commits. In the amended
decision the pass-2 figures are all still present and unchanged: 34.6% (4x),
1034.7 (3x), 2992.5 (2x), 955.1, 993.6, 408.1, 133.7 (2x), 175.8, 112.7 (2x),
1.85x; and the stale pass-1 figures remain absent: 2977.7, 1033.5, 848.2, 407.0,
176.8, 1.9x, 1.912 s all count zero. The other four documents are byte-identical
to the versions pass 2 checked, so their cross-references (roadmap 34.7% /
1910 ms, phase-s 2992.5 / 133.7 / 112.7, README index lines 49-53) are unchanged
and still agree.

Re-confirmed by re-derivation (pass 2, unchanged this pass)

Command: python3 + sqlite3 over misc/scratch/phase-s/ref-fa-nsys.sqlite and
ref-nsys.sqlite (CUPTI_ACTIVITY_KIND_KERNEL joined to StringIds; end-start),
segmented at the six timestep_embedding_f32 launches.

- Steady-state step (the second of three): 2992.5 ms of kernel time.
- Per-class, matching the amended table exactly: op_mul 178.3, cpy_scalar 139.4,
  op_add 136.8, quantize 133.7, concat 112.7, SiLU 101.3, repeat 62.0, norms
  93.0, scale_f32 41.0, f16 copies 36.4; total 1034.7; 1034.7 / 2992.5 = 34.6%.
- Like-for-like: unfused 955.1 ms of 4697.7 ms against fused 993.6 ms (the
  unrounded sum is 993.7) = 3.88%, within 4%; extra unfused cpy_scalar 184.6 vs
  139.4 and scale_f32 720.2 confirmed.
- Fusion pools: 408.1 / 133.7 / 175.8 / 112.7 all re-derive.

Check 2 — the free-flag comparison (unchanged)

ref-run.log "2/3 - 4.87s/it | 3/3 - 4.84s/it", vae decode 76.93 s;
ref-fa.log "2/3 - 2.96s/it | 3/3 - 2.93s/it", vae decode 77.67 s; both
image_seq_len=4096; 4.855 / 2.945 = 1.65x. run-ref.sh and run-ref-fa.sh differ
only by --diffusion-fa, on the deployed placement. Unchanged by the amendment.
Verdict: holds.

Check 3 — the live re-run (pass-1 result stands)

GPU-free live re-run in pass 1: deployed flags 4.69/4.69 s, --diffusion-fa
2.90/2.92 s, ratio 1.61, CPU VAE decode 77.95/79.79 s, image_seq_len=4096 —
within a few percent of the recorded values. Outputs under
misc/scratch/phase-s/qa/. Not repeated this pass: the amendment is docs-only and
changes no measured figure. Verdict: holds (pass-1 live evidence).

Check 4 — ownership pricing (unchanged)

Pass 1 verified each item live against /data/imagegen; the amendment did not
touch section 3 or the deployment. Still holding: crates/bridge/src/main.rs is
1641 lines and is_alive() is try_wait-only (line 640); the alive-but-unresponsive
limitation is commented at 1421-1423; supervise_backend (679) reaps and respawns
with backoff; imagegen.toml backend_args is the two placement flags with no
--diffusion-fa, steps=28, no adapter configured, plain Q6_K; patches 0001/0002
are int8-convrot-only and the encoder file is deleted; docs/host-memory-guard.md
shows the guard is a job launcher so the hook-launched image stack is outside
it; nothing resident, ports 8787/8788/8899 closed, systemctl unit not-found.
Verdict: holds.

Check 5 — internal consistency of the five documents

Commands: python3 link resolver over the five files; grep -n for the shared
numbers and status lines; git diff d35aeff 082335f.

- Scope: only docs/qwen-image-2.1-decision.md changed in this amendment. All
  relative links in the five files resolve; docs/README.md lines 49-53 still
  match the status lines; no phase is presented as startable; the plan is still
  marked superseded; docs/qwen-image-2.1-roadmap.md line 132 (34.7% spike
  figure) and line 134 (1910 ms free-flag return) still agree with the decision,
  which reconciles the spike's 34.7% against its 34.6%.
- Numbers shared across documents (2992.5, 133.7, 112.7, 4.855/2.945/1.65,
  250-450 ms) agree.
- Both pass-2 residuals in this area are now fixed: the grammar slip (residual
  1) and the 1.912 s step time (residual 2).

Residual defects still open (none material):

1. Concat launches: the decision (section 4) says 324, phase-s section 5 says
   322, for the same class on the same step. The trace has 322
   concat_cont<...,(int)1> plus 2 concat_cont<...,(int)0>, so 324 is correct for
   "concat_cont" and phase-s undercounts by 2. Pre-existing phase-s text; the
   decision is right.
2. cpy_scalar is 139.4 in the decision and 139.5 in phase-s (unrounded 139.45) —
   half-rounding.

Verdict for check 5: holds, with the two non-material residuals above.

Gate

Command: bash tests/qa-gate.sh (base resolves to fork/feat/image = d92e561).
Output: all eight checks PASS, "QA GATE: overall PASS". Exit code 0.

Claims I could not verify, and whether the gap is acceptable

Gap table (claim / why unverified / acceptable)

  1. F1-F4 fusion savings, 250-450 ms/step (8-15%)
     why: no fused graph exists in this tree; the decision labels it an
     estimate and does not rest the verdict on it.  acceptable: yes.
  2. 28-step sampling-phase totals (135.9 / 82.5 / 70-76 / 53.5 s)
     why: extrapolations; only 3-step spikes were run.  acceptable: yes.
  3. An end-to-end 28-step 1024-square request
     why: not run; decision section 9 admits this.  acceptable: yes.
  4. concat_cont launches 324 (decision) against 322 (phase-s)
     why: the trace has 324; phase-s undercounts by 2.  acceptable: yes.
  5. Clock-locked re-run of any number
     why: no passwordless sudo on this host (nvidia-smi -lgc unavailable); the
     decision section 9 now states the clocks were unlocked.  acceptable: yes.
  6. The live re-run (check 3) was not repeated in passes 2 or 3
     why: the amendments are docs-only and change no measured figure; the
     pass-1 run against the same reference and artifacts stands.
     acceptable: yes.

Verdict

Both post-pass-2 corrections are present and correct: the section 7 item 3
sentence is now grammatical and matches its source, and the comparison table's
last-row step time (1.910 s) is now consistent with the 1034.7 ms pool it
subtracts. The diff d35aeff..082335f is exactly those two lines, so no other
figure moved; every pass-2 number is intact, the stale ones remain absent, and
the other four documents are unchanged. The gate passes with exit code 0. The
only residual defects are the two pre-existing phase-s text nits (concat
launches 322 vs 324; cpy_scalar 139.5 vs 139.4), neither of which affects the
STOP verdict or its evidence.

verdict: overall PASS
