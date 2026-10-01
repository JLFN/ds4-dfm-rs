# Qwen-Image-2.1 native port — the decision

[Documentation index](README.md) | [Repository README](../README.md) |
[Plan](qwen-image-2.1-plan.md) | [Recipe](qwen-image-2.1-recipe.md) |
[Roadmap](qwen-image-2.1-roadmap.md) | [Phase S](qwen-image-2.1-phase-s.md)

Status: final. This document answers the question phase S left open, and it
closes the port. The [spike](qwen-image-2.1-phase-s.md) settled the performance
motive; this prices the ownership motive, prices the one unclaimed headroom the
spike measured, and decides the P0 boundary. Verdict: **stop — do not build the
port, keep the current deployment.**

Date: 2026-10-01. Host and reference: RTX 4070 SUPER (sm_89, 11894 MiB usable),
CUDA 13.3, driver 610.43.03. Reference `stable-diffusion.cpp` at `6dcb5bb`, the
pin of the [recipe](qwen-image-2.1-recipe.md) section 1; deployment
`/data/imagegen` at `221d5a8` with one uncommitted `imagegen.toml` edit.

## 1. Decision

**Stop. Do not start P0. Keep the deployment.**

The port was justified by two motives and both are now priced:

1. **Performance** — dead by measurement ([phase S](qwen-image-2.1-phase-s.md)
   section 7): M1 and M2 capture 0% through the port's own route, M3 captures at
   most 5.5% of a step.
2. **Ownership** — priced item by item in section 3. Two of the six claims in
   the plan's section 3 do not survive as losses at all, one is a capability
   rather than a loss, one survives only as a governance gap, and the strongest
   one (one process) removes a layer that the deployment already covers from
   outside. The whole case is worth less than the port costs.

The one measured headroom left, the 34.6% of a step spent in elementwise glue,
is priced in section 4. A realistic fusion pass captures 250-450 ms of a step;
the reference's own `--diffusion-fa` flag already returns 1910 ms of a step for
free on the same kernels. Fusion does not justify a six-phase engine, and P0
does not earn its place under an ownership-only motive (section 5).

The [roadmap](qwen-image-2.1-roadmap.md) now records every phase as closed with
its motive withdrawn, and section 8 lists the triggers that would reopen it.

## 2. What was checked in this session

Every claim below traces to one of these, run on 2026-10-01:

| Check | Command | Result |
| --- | --- | --- |
| Unit state | `git log --oneline 7b6448b..HEAD`, `git status` | `feat/image` at `d92e561`, clean, equal to `fork/feat/image`; 4 commits, none merged |
| Deployment live state | `ss -tlnp`, `ps -eo pid,cmd` | ports 8787/8788/8899 closed, no `sd-cli`/`sd-server` resident |
| Deployment unit | `systemctl --user is-enabled imagegen-bridge` | `not-found`; the hook launcher is the only start path |
| Deployment source | `/data/imagegen` git, `crates/bridge/src/main.rs`, `imagegen.toml` | `221d5a8` + one uncommitted `imagegen.toml` edit (documentation of `extra_lora`) |
| Glue accounting | re-derived from the phase-S trace, `ref-fa-nsys.sqlite`, segmented at the `timestep_embedding_f32` launches | 1034.7 ms of a 2992.5 ms steady-state step = **34.6%**, reproducing the spike's 34.7% within rounding |
| The free flag | `ref-fa.log` line 231, `run-ref-fa.sh` | steps 2 and 3 at **2.96 / 2.93 s** against **4.87 / 4.84 s** for the deployed flags; the invocation is the deployed placement plus `--diffusion-fa`, nothing else |
| The request budget | `ref-run.log`, `ref-fa.log` | CPU VAE decode **76.93 s** / 77.67 s, CPU text encode 13.13 s, `image_seq_len=4096` |
| The resolution ceiling | `MAX_EDGE` in `/data/imagegen/crates/bridge/src/main.rs`, its README, `git log --full-history -Smax_edge -- imagegen.toml`, and a tally of the bridge's own request log | the live bridge refuses nothing below **1536** on either edge; `max_edge = 1152` was real in the toml at `d5fcfe0` and is gone at HEAD, so the record describes an older revision; the five logged requests are all at 1024 long edge |

The deployment's own documents were read as the record of its design
(`/data/imagegen/docs/architecture.md`, `imagegen.toml` comments); where a claim
mattered to the decision it was re-checked against live code or live state, and
those two are named in the table above. Nothing was taken from memory. Figures
that come from the [phase-S report](qwen-image-2.1-phase-s.md) rather than from
this session's checks are cited to it rather than restated as current.

## 3. The ownership case, priced

The plan's section 3 lists six things the port buys. Each is restated below as a
loss the deployment suffers *today*, because a claim that names no loss is a
capability, not a motive.

| Claim (plan s3) | Loss today | Evidence | Verdict |
| --- | --- | --- | --- |
| One process instead of two | a 1641-line bridge to maintain, and two residual failure modes | `crates/bridge/src/main.rs:640,678-730,1411-1423` | survives, bounded |
| One memory policy | the image stack is outside the memory guard; placement is two flag pairs | `docs/host-memory-guard.md:1-24`; `imagegen.toml:110-113` | survives, governance only |
| One weight path | none: the reference loads the same GGUF, mmap-backed | `imagegen.toml:67`; phase-S M2 | does not survive |
| No third-party runtime | a pinned third-party binary and two patches, both now vestigial | `/data/imagegen/patches/0001,0002`; `architecture.md:196-206` | survives, reduced |
| Artifact control (re-quantization) | none measured; the speed motive it served is dead | phase-S M2 (MMQ at the cuBLAS limit) | does not survive |
| The tree's measurement discipline | none: this is a capability, not a current loss | — | not a motive |

**One process.** Two of the six do not survive at all and are dropped. The
strongest survivor is also the weakest motive, and the reason is in the live
source. The bridge spawns `sd-server` once and supervises it: `supervise_backend`
reaps a dead child and respawns it with exponential backoff, and `/health`
carries `backend_process_alive`, `backend_restarts` and `backend_restart_attempts`
(`main.rs:678-730`, `1117-1134`). The 8h51m incident the plan cites
(`architecture.md:206-212`) happened before that supervision existed and is the
reason it exists; today that failure class is contained.

What actually remains, read from the code rather than the plan:

- `is_alive()` is `try_wait() == Ok(None)` (`main.rs:640-645`) — process
  liveness only. A backend that is alive but stops answering HTTP is visible in
  `/health` as `backend false` and is **not** restarted; the bridge's own comment
  calls this a known limitation (`main.rs:1421-1423`).
- Readiness is an HTTP probe, so it passes as soon as the port binds, before
  weights load; the first request after a restart pays the full staging.

Merging removes the child, and with it the 1641-line bridge. It does
not remove the failure mode: an engine that wedges in-process is a wedged
process, and with no child there is nothing inside the process to restart it, so
the restart must come from outside — which is exactly what the deployment
already has, one layer further out (`ensure-imagegen-bridge` plus the user unit's
`Restart=on-failure`, `architecture.md:249-315`). The port trades a two-layer
stack for a one-layer stack; the layer it deletes is the one that currently
recovers the case the engine would inherit.

**One memory policy.** Real and unchanged by this session: the image stack is
launched by hooks, and the guard is a launcher (`tools/host_memory_guard.py`,
`--max-gib ... -- ./job`), so the bridge's ~11 GB backend RSS and the ~16 GiB
offload layout (`architecture.md:279-288`, `128-140`) are outside its admission
and its PSI trip floors. The loss is governance, not a defect: placement is
already measured and documented, and the measured constraints (the text encoder
must not be pinned; all-pinned fails at block 28 of 34; 1344 square does not fit
the pinned layout) are properties of the reference's own planner, which the port
would reimplement rather than discover.

**One weight path.** Dropped. The claim was that mmap or zero-copy loading
"applies unchanged"; it applies today, in the reference, to the same GGUF. Phase
S measured the Q6_K MMQ kernel at or above the cuBLAS dense-FP16 rate at the
DiT's shapes, so there is no measured delta for the tree's weight path to win
back either.

**No third-party runtime.** Reduced. The deployment ships a binary built from
`6dcb5bb` and two local patches. Both patches exist for the int8 convrot encoder
that was reverted on 2026-09-27 and deleted from disk; patch 0001's own note says
it "cannot regress the GGUF encoder", and the standing configuration uses the
GGUF encoder (`imagegen.toml:92`). So the maintenance burden is real but already
smaller than the plan assumed: the patches are carried, not needed, and dropping
them is a deployment change that does not require a port.

**Artifact control.** Dropped, and this is the one phase S killed outright. The
value claimed was re-quantizing the DiT "with this tree's own path, which is
where much of the text families' speed comes from". M2 measured the Q6_K kernel
at the tensor-core limit, so re-quantization has no speed at these shapes to
win, and the ownership value of the file alone is not a reason to build an
engine.

Net: one motive survives in full (memory governance), one survives in reduced
form (a restarter to stop maintaining), and the case that was supposed to carry
the port does not survive.

## 4. The fusion scope, priced

The [spike](qwen-image-2.1-phase-s.md) section 5 found 34.7% of a step in
elementwise glue and named it the only unclaimed headroom. No phase owned it.
Here it is priced, from the same trace.

Re-derived per step from `ref-fa-nsys.sqlite`, on the same steady-state step the
spike's section 5 reports (the second of the three in the trace, 2992.5 ms of
kernel time), by segmenting the kernel list at the six
`timestep_embedding_f32` launches:

| Class | Launches/step | ms/step | Share |
| --- | --- | --- | --- |
| Modulation multiply (`k_bin_bcast` `op_mul`) | 772 | 178.3 | 6.0% |
| Copies (`cpy_scalar`) | 516 | 139.4 | 4.7% |
| Modulation add (`k_bin_bcast` `op_add`) | 256 | 136.8 | 4.6% |
| Activation quantization (`quantize_mmq_q8_1`) | 450 | 133.7 | 4.5% |
| Segment concat (`concat_cont`) | 324 | 112.7 | 3.8% |
| SiLU-gated MLP (`unary_gated_op_kernel`) | 64 | 101.3 | 3.4% |
| Broadcast/repeat | 256 | 62.0 | 2.1% |
| Norms (`rms_norm_f32` 49.9 + `norm_f32` 43.1) | 260 | 93.0 | 3.1% |
| `scale_f32` | 392 | 41.0 | 1.4% |
| f32 → f16 copies (`cpy_scalar_contiguous`) | 320 | 36.4 | 1.2% |
| **total** | **3610** | **1034.7** | **34.6%** |

This reproduces the spike's 34.7% within rounding (its 65.3% / 34.7% split).
Over the same classes that belong to the graph rather than to the attention
path, the unfused trace's steady-state step carries 955.1 ms of them in 4697.7
ms, against 993.6 ms in the fused step: within 4%, so the glue's absolute cost
is a property of the graph and not of the attention path. What differs between
the two is where it sits — the unfused path pays its extra `cpy_scalar` copies
(184.5 ms against 139.4) and keeps its score scaling inside the attention total
(720.2 ms of `scale_f32`), which is why it is not counted here. The reference
also runs 3610 separate elementwise launches per step where a fused graph would
need a fraction of them.

The fusions the graph allows, sized from the classes above. The mechanism is
the same in each: a chain of *k* single-pass elementwise kernels over one
activation does 2*k* tensor traversals, a single fused pass does two, so a
3-op chain drops 6 traversals to 2 (−67%) and a 2-op chain drops 4 to 2 (−50%).

| Fusion | Merges | Measured pool | Ceiling saving |
| --- | --- | --- | --- |
| F1 norm + modulate | `norm_f32`/`rms_norm_f32` → `op_mul` → `op_add` in one pass | 408.1 ms | 200-270 ms |
| F2 quantize into the producer | the writer of the activation also emits the q8_1 MMQ input | 133.7 ms | 60-90 ms |
| F3 remove copies | allocate in the consumed layout and dtype instead of `cpy_scalar` / `cpy_scalar_contiguous` | 175.8 ms | up to 175.8 ms (a graph fix, not a kernel) |
| F4 fold the concat | segment assembly inside the writer of the segment | 112.7 ms | partial; not sized here |

Realistic total: **250-450 ms/step, or 8-15% of a step** — an estimate, not a
measurement, because it is priced from a trace of someone else's graph and this
tree has no graph to fuse. The upper bound, worth stating because it is
measured, is the whole pool: 1034.7 ms/step, 34.6%, if every elementwise kernel
could be removed, which no fusion pass achieves.

Put against the deployed workload (28 steps, `imagegen.toml:127`). The glue pool
is measured under `--diffusion-fa`, so rows 3 and 4 subtract from that step, not
from the deployed one:

| Option | Step time | Sampling phase, 28 steps | Evidence |
| --- | --- | --- | --- |
| Today (deployed flags) | 4.855 s | 135.9 s | `ref-run.log` steps at 4.87 / 4.84 s |
| `--diffusion-fa`, no code | 2.945 s | 82.5 s (**−53.4 s**) | `ref-fa.log` steps at 2.96 / 2.93 s |
| `--diffusion-fa` + realistic fusion (250-450 ms) | 2.50-2.71 s | 70-76 s (−7.0 to −12.6 s) | estimated, section 4 |
| `--diffusion-fa` + whole glue pool removed (unreachable) | 1.910 s | 53.5 s (−82.4 s) | measured pool, impossible to reach |

**Verdict: drop it.** One reference flag, requiring no code and reachable today,
returns 53.4 s of the sampling phase — more than four times the realistic fusion
gain, and 1.85x the entire glue pool that no fusion pass could remove
completely. Fusion is real work on a graph that does not exist (it is P3 at the
earliest), and its ceiling is below a switch the operator can flip. It is
recorded in the [roadmap](qwen-image-2.1-roadmap.md) as a P6 hypothesis if the
port is ever reopened, not carried as a motive.

## 5. The P0 boundary

Under an ownership-only motive, does P0 (engine kind, artifact formats, DiT and
VAE identification, bind plan, plan and refusal skeleton, placement model,
quantization pinning) still earn its place?

No. Every item in P0 exists to serve the port:

- The engine kind, the bind plan and the refusals are plumbing for an engine
  that would not exist.
- The VAE converter (safetensors → GGUF with a pinned layout) is the one
  standalone tool, and it earns its place only if something later has to read
  the VAE.
- Identification is needed because the DiT GGUF carries no
  `general.architecture` (plan appendix B.7); with no engine to route, there is
  nothing to route.

P0 is small, and that is the trap this unit exists to avoid: "small and next" is
not a reason to start six phases. If the port reopens, P0 is the first step and
the converter is the first artifact in it.

## 6. The alternative that was rejected

`continue-with-reduced-scope` was considered as the middle path and has no
natural boundary. The reduced scope would have to exclude P0 (nothing to
catalogue for), P1–P4 (an oracle no consumer uses), P5 (a serving surface no
engine backs) and P6 (nothing to optimize) — which is the whole roadmap. A
reduced scope that contains nothing is a stop, and calling it anything else
would leave the tracker carrying phases whose justification was deleted, which
is exactly what this unit was told not to do.

## 7. What this leaves for the operator

Not an action taken here: the deployment is another repository, and this unit
changes only documents in this tree.

1. **`--diffusion-fa` is unset and is the largest measured change available on
   this stack.** Deployed `backend_args` is the two placement flags only
   (`imagegen.toml:110-113`); adding the flag takes the step from 4.855 s to
   2.945 s (1.65x) on the reference's own kernels, measured on this host with
   everything else identical (`run-ref-fa.sh`). It is not a free numerics
   change: the two configurations produced different images (mean absolute
   difference 1.17 of 255, 93.2% of pixels differing by at least 1), so it is a
   quality decision as much as a speed one, and both belong to the operator.
2. **The two `sd.cpp` patches are vestigial**, now that the convrot encoder is
   reverted and deleted; the standing configuration uses the GGUF encoder that
   patch 0001 says it cannot affect. Dropping them is a deployment cleanup with
   no port attached.
3. **Resolution.** The live bridge accepts up to 1536 on either edge
   (`MAX_EDGE`, `crates/bridge/src/main.rs:973`), so a 1152-square request is
   servable today. The deployment's record calls that ceiling a configuration
   value of 1152, which describes an older revision: `imagegen.toml` did carry
   `max_edge = 1152` at `d5fcfe0`, and at HEAD the key is gone and the compiled
   1536 governs (documented at `crates/bridge/README.md:41`). Every request the
   bridge has actually logged is smaller: five of them through 2026-09-29, at
   768x1024 and 1024x672, 28 steps, base mode, from the aspect ratio the client
   sends. What was not measured here is the step time above 1024 under the
   deployed placement, and the record's "1152 square does not fit the 240 s read
   timeout" is a 40-step figure (`architecture.md:148-152`), not the deployed 28.
   One consequence is worth the operator's attention: the deployed pinned layout
   measures a ceiling below the bridge's — 1344 square fails during weight
   preparation (`architecture.md:126-140`) — so the accepted range extends past
   the servable one, and sizes between 1152 and 1344 were not measured.

## 8. Reopen triggers

The port reopens if, and only if, one of these becomes true, and each names the
evidence that would establish it:

1. **The reference stops serving the target.** `stable-diffusion.cpp` drops
   Qwen-Image-2.1, or a required layout cannot be expressed in its flags. The
   deployment is at a pin (`6dcb5bb`) and forks from there, so this is a drift
   risk, not a prediction.
2. **The deployment acquires a requirement the reference cannot meet** — for
   example a per-module tier, a device budget, or an admission rule that the
   bridge cannot impose from outside. The measured placement constraints
   (`architecture.md:64-108`) are the current candidate list.
3. **Ownership itself becomes a requirement**, stated as such: the tree must
   own every inference path on the host, with no measured benefit claimed. This
   is a legitimate reason and it is the operator's to give; it is the one motive
   this document cannot price away.
4. **The fusion hypothesis is measured to pay** on a real graph. That requires
   the graph, so it is a consequence of reopening, not a cause.

What would **not** reopen it: a lower step time in the reference (the port
captures none of it), a larger card (that removes the resolution ceiling without
a port), or a new image model whose reference support is good.

## 9. What this does not prove

- The fusion estimate is an estimate. It is priced from a trace of the
  reference's graph, and no fusion pass over this tree has been written or
  measured. Section 4 labels it as an estimate and the conclusion does not rest
  on it: the comparison is against a measured flag that needs no fusion at all.
- Section 4's per-class table is one steady-state step of a three-step run at
  one prompt and seed, on unlocked clocks. The classes are the same three-step
  run's; a different prompt changes the text length and the shapes slightly,
  and a longer run would average more steps into the figure.
- The ownership pricing is a cost-benefit judgement, not a measurement. The
  evidence for each item is named and checked, but the weighing is this
  document's, and trigger 3 in section 8 is the honest escape hatch from it.
- No end-to-end request was re-measured here. The request composition (CPU text
  encode, CPU VAE decode, sampling) is read from the phase-S logs; the
  deployment's own recorded figures were not reconciled against them.
- The deployment was read at `221d5a8` with one uncommitted edit; a later change
  to `imagegen.toml` or the bridge can move any item in section 3.

## 10. Evidence index

| Path | What it is |
| --- | --- |
| `misc/scratch/phase-s/ref-fa-nsys.sqlite` | the trace the per-class table is re-derived from |
| `misc/scratch/phase-s/ref-run.log`, `ref-fa.log` | step times, CPU VAE decode, `image_seq_len` |
| `misc/scratch/phase-s/run-ref.sh`, `run-ref-fa.sh` | the two invocations, deployed placement plus the flag |
| `misc/scratch/phase-s/artifact-hashes.txt` | artifact SHA-256 values |
| `/data/imagegen/crates/bridge/src/main.rs` | supervision, readiness and `/health` (lines cited inline) |
| `/data/imagegen/imagegen.toml` | deployed paths, placement, steps, adapter state |
| `/data/imagegen/docs/architecture.md` | the deployment's measured record (date-stamped, not live state) |
| `/data/imagegen/patches/0001,0002` | the two patches and the encoder they were for |
| `docs/qwen-image-2.1-phase-s.md` | the spike and its three measurements |
| `docs/host-memory-guard.md` | the guard's admission model and its launch scope |
