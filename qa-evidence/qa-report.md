QA report: feat/image — phase-S spike for the Qwen-Image-2.1 port (docs unit)

Third QA pass (refresh). The five pass-2 residual nits are fixed; the commit was
then amended once more during this pass (82cea30 -> 4f549bf), a change I also
verified.

Unit identification

Repository /data/ds4-dfm-rs, branch feat/image.
HEAD at verification time: 4f549bf06e9f6c7f358eb93540165f1e2dda42df
  ("docs(image): record the phase-S spike results"), working tree clean.
The task named HEAD 82cea30; that commit was amended to 4f549bf mid-pass. The
82cea30 -> 4f549bf diff touches only docs/qwen-image-2.1-phase-s.md and adds:
  - the steady-state bullet now cites an independent re-run ("4.75-4.78 s and
    2.81-2.82 s") alongside the recorded "4.84-4.87 s" / "2.93-2.96 s", and says
    the ratio rather than the absolute seconds is the reproducible quantity;
  - the evidence-index cross-reference "section 8" is corrected to "section 9"
    (the evidence index is section 9; 8 is "what this does not prove");
  - a new evidence-index row for qa-evidence/qa-report.md;
  - section 5 "rest" 173 -> 173.2 (2992.5 - 2819.2 = 173.3; with the 2992.4 ms
    kernel total I measured, 173.2).
All four are accurate. The re-run pair quoted is exactly my pass-1 result
(4.75-4.78 / 2.81-2.82); the "1.65x" is the recorded-run ratio (4.855/2.945 =
1.649), while the re-run ratio is 1.693x, i.e. the ratio moved 2.7% against
4.4% for the FA absolute - so "the ratio is the more reproducible quantity" is
defensible, with the parenthetical naming the recorded value.

Correction history on this unit:
  44d669d  original phase-S docs commit
  7f87762  amended during pass 1 (M1 headroom arithmetic, M2 peak sentence)
  c4a5903  pass-2 correction round (M1 attention total, MMQ split, kernel
           counts, capture range, footprint attribution)
  82cea30  five pass-2 nits (scale_f32 geometry, 1.66x->1.65x, tracker cell,
           appendix A.1 rewrite, ceiling 2.9-3.1)
  4f549bf  amended during this pass (re-run citation, section 9 fix, QA-report
           evidence row, rest 173.2)
Base ref fork/feat/image = 1b15ba24db4596c15a422000a4e5b7c030c08c89 (unpushed).
merge-base with origin/main = 7b6448b9985ae81f9089ea68c19f3a936e44fdf7. Remotes
unchanged (origin read-only). No remote-writing git command was run.

The four gate surfaces, sha256 at the current working tree (= HEAD 4f549bf):
FILE docs/qwen-image-2.1-phase-s.md
  sha256 ed0a8bd252710b4b5d3d8822dd2b78da2a3f583b1008cb11a05fee7ba14df86c
  (this is the 4f549bf content; the earlier 82cea30 content was
   564e9a67d2313d6df0b2a1da8e439ea67ea63b6d8aa6ba150db30913ee7671c3)
FILE docs/qwen-image-2.1-plan.md
  sha256 067c0f318aadc054eb505824db572fc2cff181cfbcf5ee1dc9a52c4bf86fe5a9
FILE docs/qwen-image-2.1-roadmap.md
  sha256 210bde171c9af7f27b8a2480bb0e0553de5f19e931e6294fe530b194afd32acd
FILE docs/README.md
  sha256 940ccfd03f13e99a039ee88b938e492c6576acec625b160ba7d4688b2b41571e

What this pass fixed (the five pass-2 nits)

1. scale_f32 geometry in phase-s section 2 now reads "gridX 2101760 / 2104832,
   gridY=gridZ=1, block (256,1,1)"; the prose names the four grid shapes that
   disappear (gridX 11, 29, 2101760, 2104832, 32 launches each) and gives
   341.1 + 340.1 = 681.2 ms.
2. phase-s section 6 "1.66x" -> "1.65x", matching section 2.
3. roadmap tracker gate-result cell "M3 1.8-5.5%" -> "M3 <=5.5%".
4. plan appendix A.1 rewritten so the correct statement leads and the retracted
   "about 11 GB" is quoted inside the audit note rather than asserted.
5. launch-overhead ceiling "2.9-3.0" -> "2.9-3.1 ms/step (0.1%)".

New-value check (the only new arithmetic this round)

From the section-2 table: scale_f32 681.2 ms (128 launches, 5.322 ms/launch) +
soft_max_f32 706.3 + cutlass 256x64 520.6 + cutlass 64x64 483.6.
  341.1 + 340.1 = 681.2                       exact
  681.2 + 706.3 + 520.6 + 483.6 = 2391.7     exact
  681.2 / 128 = 5.3219 ms = 5.322            exact
  2391.7 / 4697.7 = 50.912% = 50.9%           holds
  2391.7 / 544.4 = 4.393x = 4.39x             holds
Against my own trace numbers (ref-nsys.sqlite steady step 2): scale_f32
grid(2101760,1,1) 341.1 ms / 32 launches and grid(2104832,1,1) 340.1 ms /
32 launches - the source of both addends, exact; soft_max_f32 706.2 (the doc's
706.3 is the trace average); cutlass 520.6 and 483.6 exact; my step-2 sum is
2391.6, i.e. 50.91%. The prose is also right that the default step runs 520
scale_f32 launches in eight distinct grid shapes (1, 11, 16, 29, 65680, 65776,
2101760, 2104832), that 392 remain under flash attention in four of them
(1, 16, 65680, 65776), and that exactly 32 launches each of gridX 11, 29,
2101760 and 2104832 (128 total) disappear.

No residual inconsistency: a sweep of the four documents finds no surviving
1.66x, ~3800, "1.8-5.5%", 2389.7 or 679.2; the 2.9-3.1 ceiling is consistent
with 0.650 us x 4544/4736 = 2.95/3.08 ms; and the "about 11 GB" / "11.5 GB"
strings appear only inside the correction notes that retract them, never as
assertions. The one rounding difference is the roadmap gate result's "2390
ms/step", a three-significant-figure round of 2391.7 (the same cell says 50.9%
and 1.65x, both consistent).

Claim-by-claim against the corrected documents

C1 footprint 5604.32 MB weights / 297 tensors + 2317.45 MB compute, not ~11 GB:
   REPRODUCED (pass 1; unchanged).
C2 steady step 4.855 s -> 2.945 s, 1.65x (recorded), on a 4697.7 -> 2992.5 ms
   kernel budget; re-run 4.765 -> 2.815 s: REPRODUCED (my pass-1 pair is now the
   one the doc cites; my ratio 1.69x vs the doc's 1.65x, noted).
C3 unfused attention 2391.7 ms / 50.9% (incl. 681.2 ms of 128 attention-path
   scale_f32 launches): REPRODUCED, arithmetic exact against my step-2 trace.
C4 flash attention 544.4 ms / 64 launches, 18.2%; 4.39x: REPRODUCED (unchanged).
C5 Q6_K MMQ 1309.8 ms default / 1410.9 ms FA, 448 launches both, 47.1% of the FA
   step, ~8% run/clock spread: REPRODUCED from both traces.
C6 Q6_K 72.0-80.8 TFLOP/s vs cuBLAS 70.4-77.3 (report 71.3-79.1 vs 69.5-74.0):
   REPRODUCED.
C7 4544 / 4736 kernels per step; launch ceiling 2.9-3.1 ms/step (0.1%):
   REPRODUCED (13632/3 and 14208/3; 0.650 us x counts = 2.95/3.08 ms).
C8 capture block-pass an unstable 3-55 ms/step bounded by the 172.7 ms (5.5%)
   host bubble: REPRODUCED (my runs 3-47 ms/step; bubble exact).
C9 footprint attributions to plan appendix A.1 and section 14; roadmap
   attribution retracted; A.1 now leads with the measured statement:
   REPRODUCED.
C10 whole-run totals 120.92 / 103.62 s marked not comparable (cold 13.13 s vs
   warm 1.46 s encoder): REPRODUCED / ADDRESSED.

Correction history retained from earlier passes

Pass 1 found six load-bearing claims reproduced and five defects: M1's unfused
attention under-count (the 128 scale_f32 launches), the "703.7 + 606.1 = 1410.9"
mixing two runs, "~3800 kernels/step", the un-pinned 55 ms/step capture figure,
and the false roadmap footprint attribution. Pass 2 verified all five addressed
and recorded five residual nits. This pass verifies the nits fixed, plus the
mid-pass amend above. The phase-S decision (no performance gain capturable by
the port; ownership-only) has never been in question in any pass.

Accepted risks / what this does not establish

- The docs commit was amended repeatedly during the QA passes (44d669d ->
  7f87762 -> c4a5903 -> 82cea30 -> 4f549bf). HEAD and the hashes above are the
   working-tree state at verification; a further amend would invalidate them.
   I confirmed 4f549bf stable for 20 s before this report.
- Clocks were not locked (no passwordless sudo); all numbers are same-session at
  an observed 2565-2805 MHz band. The 8% MMQ spread between the two traces and
  the absolute step times are not clock-locked; the report says so.
- No reference or harness was re-run this pass; the new arithmetic is an exact
  re-aggregation of the pass-1 trace numbers I measured (341.1, 340.1, 706.2,
  520.6, 483.6 ms).
- Harnesses are scratch and link prebuilt cuda/mmq/*.o older than the sources;
  the intervening commit adds PQ2_0 code only, so the measured Q6_K dense path
  is unchanged.
- Attention TFLOP/s remains derived arithmetic, not a counter reading.

Verdict

The third pass is a clean refresh. The five pass-2 nits are fixed, the new
attention total (681.2 + 706.3 + 520.6 + 483.6 = 2391.7 ms, 50.9%, 4.39x)
matches the section-2 table and my own trace numbers exactly, no figure in the
four documents is left inconsistent with it, and the late amend (re-run
citation, section 9 fix, evidence row, rest 173.2) is accurate.

verdict: overall PASS
