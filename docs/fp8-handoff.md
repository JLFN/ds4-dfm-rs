# Handoff — FP8 and FP4 tensor cores for DS4 V4.1 Flash 1017 on GB10

Written 2026-11-22. Scope: **ds41 only**. The study is
`docs/fp8-tensor-core-research-2026-11-22.md`; the phase detail with edit sites is
`github.com/JLFN/vq-tools` `docs/handoff-fp8-phases.md` at `dev` @ `42e9e43`.
This file is the entry point and it is specific to this family.

## The finding that matters

ds41's sparse attention kernel is not an FP8 kernel and cannot become one
cheaply. It computes both products with `mma.sync` **BF16**
(`cuda/ds41_attn_mma.cuh:151-153`, `ds4_fa_mma`), 21-24 percent of a
65,536-token prefill block, 13.5-13.7 ms per layer. Against the 107.8 TFLOPS
BF16 peak its arithmetic is 0.4 percent.

But its compressed KV is **already FP4 with E4M3 scales** — `DS4_V41_CKV_NIB`
scales decoded one per lane, four-bit values unpacked per lane
(`cuda/ds41_attn_mma.cuh:47-52`, `:58-80`). The kernel therefore decodes FP4 to
BF16 in shared memory and then multiplies in BF16. The block-scaled FP4 tensor
instruction (`kind::mxf4nvf4`, `mma.sync m16n8k64`, E2M1 operands with an E4M3
scale per 16) consumes exactly that stored pair natively, with no decode.

That is the whole opportunity, and it is the opposite of what an FP8 study would
predict: the storage is already low precision and the arithmetic is not. It is
also the one place in this engine where an instruction change could reach a
23.8-percent class without a re-quantisation, because the bytes on disk already
have the format the instruction wants.

## Where the artefacts live

| Thing | Where | State |
|---|---|---|
| Study | `docs/fp8-tensor-core-research-2026-11-22.md` | Uncommitted here; on vq-tools `dev` as `docs/fp8-tensor-cores.md` |
| Phases, edit sites | vq-tools `docs/handoff-fp8-phases.md` | Pushed, `dev` @ `42e9e43` |
| The MMA rate probe | vq-tools `c/gpu/mma-rate.cu` → `bin/mma-rate` | Pushed, built, run |
| hVQ fitter (cited by line) | vq-tools `crates/vq-calib/reference/hvq/` | Pushed, sha256 recorded |
| Runtime repo | `~/ds4-dfm-rs` on the Spark, `main` @ `c57543fb` | Untouched |
| Docs index | `docs/README.md` | Modified, uncommitted |

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

### Phase 2 — sparse attention: the largest number, and a format question

**Start here.** Cost: one `ncu` GPU Speed Of Light capture on an idle box, no
code change.

The kernel is `ds4_sparse_attn_mma_kernel` (`cuda/ds41_attn_mma.cuh:189`),
BF16 MMA, `__launch_bounds__(256)`, 26 KB of shared memory. Its items are
gathered from the FP4 compressed pool plus the FP8/F32 window, decoded to BF16
into `ks`, then multiplied.

Four questions the capture answers:

1. How much of the round is the decode versus the MMA? The gather resolves each
   element with two shuffles (`cuda/ds41_attn_mma.cuh:47-52`); if that dominates,
   the FP4 instruction removes it outright and the win is large.
2. Is softmax overlapping the MMAs? FP8/FP4 double the matmul rate and leave the
   exponential unit unchanged, so a serialised kernel idles the tensor cores by
   construction ([2407.08608](https://arxiv.org/abs/2407.08608) [page 1-2, 5-6]).
3. Does P get quantised? No: pass 2 recomputes S, exp's it and writes the tile as
   `__float2bfloat16` (`cuda/ds41_attn_mma.cuh:589`), so the PV product is BF16
   too. Making QK FP4
   without PV leaves half the class at the old rate.
4. Does the accumulator layout match the operand layout for the FP4 form?
   SageAttention3 found it does not and permuted the P tile instead of shuffling
   at run time ([2505.11594](https://arxiv.org/abs/2505.11594) [page 5]).

Gate: this family is documented as **not** bit-equal to the scalar kernel — "the
gate is the quality scale (NLL/PPL), not cmp"
(`cuda/ds41_attn_mma.cuh:13-16`). Expect a large idle tensor-pipe fraction; if
the capture disagrees, re-derive the arithmetic before touching code.

### Phase 3 — QK and PV on the block-scaled FP4 instruction

Only if Phase 2 says the decode or the MMA issue dominates. The stored form is
already what `kind::mxf4nvf4` consumes, so the change is a layout and scale-plumbing
change, not a re-quantisation.

This is the item with a numerics risk, so it goes through the family's own
quality gate over the release's 26 cases, against the original. Two named
hazards: the accumulator layout differ from the operand's, and P is BF16 today,
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
- ds41's index scores are an exact INT8 problem on s8 tensor cores, 841 → 50 ms
  at 59K. Round 1, not a lever.
- The per-stage promotion removal: measured even at 180 registers.
- Any fixed permutation of the book: a bijection, simulated twice, no change.
- The grouped-versus-per-pair decode dispatch: already chosen.
- Any bit-width change, codebook refit, or size-for-quality trade. The artifact is
  frozen at 113,556,639,424 bytes.
- The hVQ converter as a performance target.

## Open discrepancies

1. **The attention rate.** The repo records "about 13 TFLOPS"; backing it out of
   13.5 ms gives 0.4. Both are far below the 107.8 BF16 peak, and Phase 2 settles
   which is real.
2. **The hVQ gain's kernel independence.** Follows from the refit formula
   (`tools/vq_calib/cuda/ds41_hvq.cu:171-180`) but not cross-checked against a
   refit run. Phase 4 depends on it.

## Next action

One capture, on the idle box:

    ncu --clock-control none --set full --launch-count 1 \
        --kernel-name ds4_sparse_attn_mma_kernel <the prefill command>

Read tensor-pipe active, the shared-memory and LSU pipes, and the stall
breakdown against the four questions. Everything else waits on that number, and
it decides whether Phase 3 is worth its numerics risk.
