# Handoff — FP8 and FP4 tensor cores for DS4 V4.1 Flash 1017 on GB10

Written 2026-11-22. Scope: **ds41 only**. The study is
`docs/fp8-tensor-core-research-2026-11-22.md`; the phase detail with edit sites is
`github.com/JLFN/vq-tools` `docs/handoff-fp8-phases.md` at `dev` @ `42e9e43`.
This file is the entry point and it is specific to this family.

## The finding that matters

ds41's sparse attention kernel is not an FP8 kernel and cannot become one
cheaply. It computes both products with `mma.sync` **BF16**
(`cuda/ds41_attn_mma.cuh:151-153`, `ds4_fa_mma`), 21-24 percent of a
65,536-token prefill block, 13.5 ms per block-layer. Against the 107.8 TFLOPS
BF16 peak its arithmetic is **11.8 percent**, which agrees with the record's own
"about 13 TFLOPS". An earlier draft of the study said 0.4 percent and was wrong
by 32x: it divided a whole-prefill layer's FLOPs by a per-block time.

But its compressed KV is **already FP4 with E4M3 scales** — `DS4_V41_CKV_NIB`
scales decoded one per lane, four-bit values unpacked per lane
(`cuda/ds41_attn_mma.cuh:47-52`, `:58-80`). The kernel therefore decodes FP4 to
BF16 in shared memory and then multiplies in BF16.

That looked like the whole opportunity, and the measurement says it is not. The
toolkit's `bin/attn-bench` times the shipped kernel with no model loaded and
separates the two key sources at identical MMA work per tile: the FP4 pool path
costs **1.2 to 1.3x** the F32 window path per tile at matched tile counts, and
the per-launch fixed cost is about 2.1 ms of the 12.5 ms launch. The gather and decode are the cost; the MMA is not.
The lever is the decode path — vectorising the unpack and overlapping it with
the MMAs — not a change of tensor instruction.

Two constraints follow, and both are recorded in the study. The stored scale is
one per 16 elements along `head_dim`, which is QK's reduction axis but PV's
output axis, so the block-scaled FP4 form cannot consume the stored pair for PV
without a re-quantisation. And the compressed pool does not exist on every
layer: the artifact's metadata reads 0 for layers 0-1, 2 for layers 2-19 and 1
for layers 20-39, so layers 0 and 1 attend to the 128-key window alone.

## Where the artefacts live

| Thing | Where | State |
|---|---|---|
| Study | `docs/fp8-tensor-core-research-2026-11-22.md` | Committed here; also on vq-tools `dev` as `docs/fp8-tensor-cores.md` |
| The attention bench | vq-tools `c/gpu/attn-bench.cu` → `bin/attn-bench`, `docs/attn-bench.md` | `dev` @ `9a3290b`; built and run on the GB10, cross-checked from a fresh clone |
| Phases, edit sites | vq-tools `docs/handoff-fp8-phases.md` | Pushed, `dev` @ `42e9e43` |
| The MMA rate probe | vq-tools `c/gpu/mma-rate.cu` → `bin/mma-rate` | Pushed, built, run |
| hVQ fitter (cited by line) | vq-tools `crates/vq-calib/reference/hvq/` | Pushed, sha256 recorded |
| Runtime repo | `~/ds4-dfm-rs` on the Spark, `main` @ `c57543fb` | Untouched |
| Docs index | `docs/README.md` | Updated with these findings |

The fork rule holds: nothing goes to the upstream runtime repository. The runtime
repo here is `github.com/JLFN/ds4-dfm-rs`; the toolkit is
`github.com/JLFN/vq-tools`, private.

## The host, verified today

`leandro@192.168.1.91`, hostname `gx10-e1d2`. GB10, capability 12.1, 48 SMs,
121.6 GiB unified memory, CUDA 13.0 at `/usr/local/cuda`, max clock 2,418 MHz.
Server **stopped**; 113 GB free.

- Start `bash ~/start-deepseek41.sh`; stop `bash ~/stop-deepseek41.sh`; status is
  the installer's `status` action under `~/dgx-recipe/scripts/`.
- Base, Engram tables, vision sidecar and the verification record are present and
  matching. No download needed.
- `~/VQ_tools` there predates the `vq_hvq.cu` commit and cannot fetch (no
  credentials). Deliver over scp; do not update that checkout without asking.

Three things silently invalidate a measurement: `nvidia-clocks` is inactive (the
driver boosts to 2,366-2,411 MHz against a 2,200 MHz policy — record the clock,
never compare absolutes across states); `nvidia-smi` memory queries read N/A on
GB10 (use `free -g`); a resident 113.6 GB server leaves ~10 GB free and competes
for the same memory system.

## Phases, in ranked order

### Phase 1 — block-scaled MMA rate: DONE, PASSED

    plain     mma.sync m16n8k32 e4m3 -> f32        246.1 TFLOPS (1.676 ms)
    scaled    mma.sync kind::mxf8f6f4 block_scale
              scale_vec::1X e4m3 -> f32 ue8m0      246.1 TFLOPS (1.676 ms)
              ratio 100.0 percent, idle GPU, 2,366 MHz

A second sample with the server resident read 99.3 percent; both kept. It
establishes that a per-32 hardware scale is free. It does not establish a
throughput argument, because the promotion it removes is itself measured free.

### Phase 2 — sparse attention: MEASURED on the reference host

**The decode is the cost, and it is measured rather than guessed.** The toolkit's
`bin/attn-bench` compiles this kernel's own source behind five host stubs, so it
runs with no model loaded, in a few hundred MiB, and can be captured with `ncu`
under the one-GPU-job rule. On the reference host (GB10, 2,048 rows, 2,470 MHz):

    shipped shape (128 window + 512 topk, 40 tiles)   12.47 ms/kernel
    record for the class per block-layer              13.5 ms

    matched tiles      F32 window     FP4 pool     pool / window
        8 tiles          3.159 ms      3.775 ms        1.20x
       16 tiles          4.483 ms      5.487 ms        1.22x
    marginal 8->16     0.1655 ms/tile 0.2140 ms/tile   1.29x

Two results, and both change the plan.

1. **The bench reproduces the record.** 12.47 ms against 13.5 ms means this
   kernel is 92 percent of the recorded attention time; only 8 percent is the
   surrounding prep and epilogue. The class is one kernel, not a group.
2. **The compressed pool costs 20 to 29 percent more per tile than the F32
   window at equal MMA work.** That excess is the gather and decode — the
   288-byte row, one scale per lane, two shuffles per element — so the decode is
   the lever, not the tensor instruction.

The launch is also sublinear in tiles: the first 4 tiles cost 3.0 ms against a
0.9 ms marginal, about 2.1 ms of fixed cost in a 2,048-row launch. An
instruction change cannot touch that, and it is a further reason to put the
pipeline structure first.

One method note, because an earlier run of this bench got it wrong: do not
compare averages at unequal tile counts. The per-launch fixed cost is divided
over a different denominator, and at 8 tiles against 32 the pool looked
*cheaper* than the window on the same host. Match the count or take the slope.

Three things the measurement settles:

1. The FP4 instruction does **not** remove the decode cost, because the decode
   is what the matched-tile excess measures and the instruction only removes the
   widening that follows it. Ranking it first was wrong.
2. The rate is 11.8 percent of the BF16 peak, matching the record's 13 TFLOPS;
   the study's earlier 0.4 percent was a block-convention error.
3. Softmax is serialised on 16 threads of 256 between full-CTA barriers
   (`cuda/ds41_attn_mma.cuh:253-261`), and the decode kernel already has the
   warp-parallel form (`:122-135`), so the fix is a port and is bit-equal.

Remaining question for the capture: the split between the decode's shared-memory
traffic and its shuffle/issue cost, which decides whether vectorising the unpack
or overlapping it with the MMAs is the better first move.

Gate for any of it: this family is documented as **not** bit-equal to the scalar
kernel — "the gate is the quality scale (NLL/PPL), not cmp"
(`cuda/ds41_attn_mma.cuh:13-16`).

### Phase 3 — the compressed-pool decode: vectorise and overlap

The measured target. Vectorise the nibble unpack and overlapping it with the
MMAs. Two of the three sources of pooled keys are already ruled out on paper:
FLUTE's bank duplication needs shared memory the kernel has spent, and a fixed
lane rotation is a bijection that leaves a random gather unchanged.

### Phase 3b — block-scaled FP4 QK, only for QK, only after Phase 3

Demoted from the top of the list, and narrowed. The stored scale is one per 16
elements along `head_dim`: QK's reduction axis, which the instruction can use,
but PV's *output* axis, which it cannot without a re-quantisation. So at most
half the class is reachable, P and the F32 window would have to become E2M1, and
the measured cost is elsewhere besides.

This is the item with a numerics risk, so it goes through the family's own
quality gate over the release's 26 cases, against the original. Two named
hazards: the accumulator layout differs from the operand's, and P is BF16 today,
so enabling it is a second, separable change.

### Phase 4 — E8M0 per-32 activation scales on the routed experts

Byte-neutral by arithmetic: one E8M0 byte per 32 elements of K is 0.03125 bytes
per element, identical to the F32 scale per 128 it replaces. Ranked below
attention because the class is already within a quarter of its DRAM stream bound,
so the ceiling is small; and because its own rejected-variant record says removing
the per-stage promotion was measured **even**.

Edit sites: `cuda/minimax_mma8w.cuh:115,455`, `cuda/glm_vq_mma8.cuh:29,412`,
constant at `:15`, ds41 launch at `cuda/ds41_vq_prefill_fp8.cuh:202`. Promotion
at `cuda/glm_vq_mma8.cuh:288,328-334`, `cuda/beam_mma8w.cuh:515-529`,
`cuda/minimax_mma8w.cuh:363-380`.

Gate: teacher-forced NLL and top-1 over the 26 cases and 58,777 positions against
the original, floor from the chunk-size switch. If prefill does not move, record
the negative and stop.

### Phase 5 — the codeword read width

Split each 8-byte codeword read into rotated 4-byte halves. The only gather
variant the study leaves standing, because it changes wavefront count rather than
collision distribution. Sites `cuda/minimax_mma8w.cuh:345-348`, residual `:355`.
Gate: bit-identical on `make test-minimax-mma8w`, then real-route replay.

### Phase 6 — decode, only after a stall profile

Verify is at ~77 percent of its width-corrected DRAM floor, so headroom is 12-18
ms of a 51 ms pass. One capture first; if the pass is within a few percent of its
floor, record that and stop.

| Signal | Cause | Lever |
|---|---|---|
| `long_scoreboard` high, DRAM low | latency, low occupancy | staging admitting 2 CTAs/SM |
| `mio_throttle` high | lookup issue | fewer wavefronts; needs book space |
| `barrier` high | producer ring | pipeline structure |

## Settled — do not redo

- The BF16 and FP8 peaks: 107.8 and 215.6 TFLOPS at 2,190 MHz, and the FP4
  block-scaled form at 509.5 at ~2.5 GHz (`docs/gb10-hardware.md:90-100`).
- ds41's attention runs at 12.7 TFLOPS, 11.8 percent of the BF16 peak, and the
  record's "about 13 TFLOPS" is right. The block is 2,048 rows through layers
  0-19; the study's 0.4 percent was a 32x convention error.
- The compressed-pool path costs 1.2 to 1.3x the F32 window path per tile at
  matched tile counts and equal MMA work, measured with `attn-bench`, so the
  decode is the cost and the MMA is not.
- The kernel alone is 12.47 ms of the record's 13.5 ms per block-layer, so the
  class is one kernel and only 8 percent is the prep and epilogue around it.
- The block-scaled FP4 instruction cannot serve PV on the stored pool, because
  its scale runs along `head_dim`, which is PV's output axis.
- ds41's index scores are an exact INT8 problem on s8 tensor cores, 841 → 50 ms
  at 59K. Round 1, not a lever.
- The per-stage promotion removal: measured even at 180 registers.
- Any fixed permutation of the book: a bijection, simulated twice, no change.
- The grouped-versus-per-pair decode dispatch: already chosen.
- Any bit-width change, codebook refit, or size-for-quality trade. The artifact is
  frozen at 113,556,639,424 bytes.
- The hVQ converter as a performance target.

## Open discrepancies

1. **The hVQ gain's kernel independence.** Follows from the refit formula
   (`tools/vq_calib/cuda/ds41_hvq.cu:171-180`) but not cross-checked against a
   refit run. The E8M0 experiment depends on it.
2. **The decode's internal split.** `attn-bench` measures the compressed path at
   about 1.2 to 1.3x the window path per tile but does not say how much of that is
   shared-memory wavefronts versus shuffle and issue. The capture settles it.

## Next action

Run the bench on the Spark first; it needs no model, so the resident server can
stay up and the one-GPU-job rule is respected:

    DS41_SRC=~/ds4-dfm-rs ./build.sh && ./bin/attn-bench --rows 2048 \
        --keys 512,0 --window 0,128 --reps 9 --clock-mhz <live SM clock>

Then capture the same binary, one launch:

    ncu --clock-control none --set full --launch-count 1 \
        bin/attn-bench --rows 2048 --keys 512 --once

Read tensor-pipe active, the shared-memory and LSU pipes, and the stall
breakdown. That split decides whether Phase 3 vectorises the unpack or overlaps
it with the MMAs; nothing else is waiting on it.
