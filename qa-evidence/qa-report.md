QA report: feat/image — the Qwen-Image-2.1 stop decision (docs unit), pass 5

Fifth pass. The pass-4 FAIL (a false git-history claim in docs/qwen-image-2.1-plan.md)
is fixed, and the fix is confirmed against the deployed repository's history.
Verdict: PASS. Independent falsification pass, rule 19. /data/imagegen was read
read-only (files and read-only git commands); nothing was written or
started/stopped there.

Unit identification

Repository /data/ds4-dfm-rs, branch feat/image.
HEAD 340b7210321d6484841165b2e0a9b2f10e6766a7 (the QA-evidence commit); the docs
commit under it is 082335fdbbe6a1d98c01c691e98d56f5abce2998. Amendment history:
78fab2b (pass 1) -> d35aeff (pass 2) -> 082335f (pass 3) -> pass 4 FAIL -> pass 5.
The working tree holds three uncommitted edits, in two documents, plus this report:
  docs/qwen-image-2.1-decision.md   (edits 2 and 3: section 7 item 3, section 2 row)
  docs/qwen-image-2.1-plan.md       (edit 1: appendix A.2)
  qa-evidence/qa-report.md          (this report)
git status shows exactly those files. docs/qwen-image-2.1-roadmap.md,
docs/qwen-image-2.1-phase-s.md and docs/README.md are unchanged from pass 4.
Base ref fork/feat/image = d92e5614544ddd77efb74a4619580aa9ffa41c88 for the
unit; note that the remote-tracking ref fork/feat/image currently points at HEAD
(340b721), so tests/qa-gate.sh short-circuits ("no unpushed commits yet", exit 0)
and neither gates nor contradicts this report. No remote-writing git command was
run.

The pass-4 defect and its fix

Pass 4 failed docs/qwen-image-2.1-plan.md appendix A.2 for asserting "the git
history shows it never was one: imagegen.toml has no max_edge key at any
revision". That was false: git show d5fcfe0:imagegen.toml has "max_edge = 1152"
(line 68), d5fcfe0 is an ancestor of the deployed HEAD 221d5a8, and
git log --full-history -Smax_edge HEAD -- imagegen.toml returns d5fcfe0; only the
default (history-simplified) git log -S was empty.

The fix is correct. The working-tree A.2 now says the record "describes an
earlier revision rather than the ship: the deployed repository did carry
max_edge = 1152 in imagegen.toml at d5fcfe0 (2026-09-22 11:29, 'bound the
bridge'), and at HEAD that key is gone. What governs now is const MAX_EDGE:
u32 = 1536 ... introduced earlier the same day by fb054be." Re-checked:
- d5fcfe0's subject is "feat(repo): measure the resolution headroom, port the
  verifier to Rust, bound the bridge"; its author date is 2026-09-22 11:29:33.
- fb054be is 2026-09-22 10:18:33, earlier the same day, and it introduced
  const MAX_EDGE: u32 = 1536 (its parent e0e9030 has no ceiling).
- At HEAD (221d5a8) imagegen.toml has no max_edge (grep count 0). Correct.
The false premise ("never one", "at any revision") is gone from the plan.

The same correction in docs/qwen-image-2.1-decision.md: section 7 item 3 now
says the record's configuration value "describes an older revision:
imagegen.toml did carry max_edge = 1152 at d5fcfe0, and at HEAD the key is gone
and the compiled 1536 governs (documented at crates/bridge/README.md:41)"; and
the section 2 what-was-checked row now names the check
"git log --full-history -Smax_edge -- imagegen.toml" and says "max_edge = 1152
was real in the toml at d5fcfe0 and is gone at HEAD, so the record describes an
older revision". Both are accurate.

Is "the compiled 1536 governs" defensible given d5fcfe0 existed?

Yes, and the documents scope it correctly. At d5fcfe0 the code READ the ceiling
from the config, not a constant: crates/bridge/src/main.rs at d5fcfe0 has
Config.max_edge (line 81), the file field (103), the loader
pick("IMAGEGEN_MAX_EDGE", file.max_edge, Some(DEFAULT_MAX_EDGE)) (190), and
check_edge_limit(w, h, max_edge) (470-471); DEFAULT_MAX_EDGE = 1152 (468); there
is no const MAX_EDGE. At HEAD (221d5a8) main.rs has only const MAX_EDGE: u32 =
1536 (973) and no max_edge field, no IMAGEGEN_MAX_EDGE. So the constant did not
govern at d5fcfe0, but it does govern at HEAD, which is exactly what the
documents now say ("at HEAD ... the compiled 1536 governs", "What governs now
is const MAX_EDGE ... introduced ... by fb054be"). The documents do not
over-state; if anything they omit that the config change lived on a branch whose
main.rs change was not carried into HEAD's mainline (fb054be's const is the one
that survived), which is a detail the claim does not need.

Residuals from pass 4

- The certainty objection is addressed. Pass 4 flagged "a request between 1153
  and 1536 is accepted and then fails inside the backend" as an inference (only
  1344 is measured to fail). The decision now says "the accepted range extends
  past the servable one, and sizes between 1152 and 1344 were not measured";
  the plan says the same. The unmeasured range is named and the categorical
  failure claim is gone. Confirmed.
- The gate short-circuit is now explicitly bounded (stated under Unit
  identification above): fork/feat/image == HEAD, so the gate reports ALL PASS
  without evaluating the dirty working-tree changes. Recorded, not hidden.

Prior passes, still holding

- Glue accounting re-derived from the traces at the steady-state step:
  1034.7 ms of 2992.5 ms = 34.6%, classes matching; like-for-like 955.1 against
  993.6 (within 4%); fusion pools 408.1 / 133.7 / 175.8 / 112.7.
- Free flag 4.855 -> 2.945 s, 1.65x; live re-run reproduced within a few percent.
- Ownership pricing verified live at /data/imagegen (1641-line bridge,
  try_wait-only is_alive, no auto-restart of a wedged backend, supervise_backend
  backoff, two-flag backend_args with no --diffusion-fa, steps=28, no adapter,
  plain Q6_K, both patches vestigial, the guard is a launcher, nothing resident).
- Operator-facing ceiling section 7 item 3 and the plan's A.2 no longer assert
  the configurable-1152 ceiling as live.

Internal consistency of the five documents, re-run this pass

- No document carries the retracted premise: grep for "never was", "at any
  revision", "no ... max_edge", "no such key" over the five files returns
  nothing.
- Every relative link in the five files resolves (0 missing).
- docs/README.md lines 49-53 still match the documents' status lines (decision
  "final / stop", phase-s "complete / no gain", plan "superseded", roadmap
  "closed").
- Shared numbers still agree: 34.7% (decision as the spike's figure, roadmap 132,
  phase-s 218/268) against the decision's recomputed 34.6%; 2992.5, 133.7 and
  112.7 appear in both the decision and docs/qwen-image-2.1-phase-s.md with the
  same values; 4.855 / 2.945 / 1.65 agree across decision, roadmap and phase-s.
- No stale figure reappeared: 2977.7, 1033.5, 848.2, 407.0, 176.8, 1.9x and
  1.912 are absent from all five files; the amended 1.910 s and 1.85x are
  present.

The five surfaces this unit changes: docs/qwen-image-2.1-decision.md (verdict
and the operator-facing resolution text, now correct), docs/qwen-image-2.1-plan.md
(the goal and appendices; A.2 was the pass-4 defect and is now correct),
docs/qwen-image-2.1-roadmap.md (phase tracker, unchanged and consistent),
docs/qwen-image-2.1-phase-s.md (the spike, unchanged; no ceiling figure), and
docs/README.md (the index, unchanged and matching the status lines).

Claims I could not verify, and whether the gap is acceptable

Gap table (claim / why unverified / acceptable)

  1. Which merge discarded the branch-side config main.rs change.
     why: verified only that the const line reaches HEAD (8c5b808 and bbd1bcf
     are const) and the config line (d5fcfe0, 8db5f08, c018d2e) does not; the
     documents make no claim about it.  acceptable: yes.
  2. F1-F4 fusion savings (250-450 ms/step).  why: no fused graph exists in this
     tree; labelled an estimate.  acceptable: yes.
  3. Clock-locked re-run of any number.  why: no passwordless sudo.  acceptable:
     yes.
  4. The live 3-step reference re-run was not repeated in passes 2-5.  why: the
     changes are docs-only; the pass-1 GPU run stands.  acceptable: yes.

Verdict

The pass-4 defect is fixed and the fix matches the deployed repository's git
history: max_edge = 1152 existed at d5fcfe0 and is gone at HEAD, and the compiled
const MAX_EDGE = 1536 is what governs at HEAD (main.rs:973; README.md:41). Both
edited documents, the roadmap, the phase-S report and the index are mutually
consistent, no false "never a configuration value" premise remains, the
unmeasured 1152-1344 range is now stated instead of a categorical failure claim,
and the gate short-circuit is recorded. The STOP verdict is untouched.

verdict: overall PASS
