# FP8 tensor cores: what the literature says, and what it means for this engine

Research note, 2026-11-22. Documentation only. Scope: FP8 arithmetic and its
scaling, read from the literature and checked against our kernels. Nothing here
is adopted, and no kernel, harness or probe source changed. The single
measurement recorded is a compiler check, reproduced verbatim in section 2, so
anyone can re-run it with two commands.

Findings in one paragraph. Asked where this engine actually multiplies on tensor
cores, and at what fraction of the peak, the answer is not the experts: they are
29.5 percent of a prefill block and already within a quarter of their DRAM
stream bound, so no arithmetic change helps them. The class that computes on
tensor cores and does it badly is sparse attention, 23.8 percent of the block at
11.8 percent of its own peak. Two things about it are specific to this
family. Its arithmetic is BF16, not FP8, while its compressed KV is already FP4
with E4M3 scales — so the stored bytes are in the form the block-scaled FP4
tensor instruction takes natively, and the kernel currently spends that
advantage widening them to BF16. And the literature names the structural reason
a kernel like it idles its tensor cores: it computes QK, waits for softmax, then
computes PV, while FP8 or FP4 doubles matmul throughput and leaves the
exponential unit where it was. Separately, the cost of an FP8 scale lands on
CUDA cores rather than tensor cores, and the SM120 family can move it back into
the instruction through a block-scaled warp MMA this engine does not use; a
compiler check confirms the instruction and a measured run puts it at 99-100
percent of the plain form's rate, so its case rests on instruction count, not
throughput. Enabling FP8 on the attention product has a precondition: casting
the softmax output to E4M3 collapses it under attention sinks unless the key
walk is reversed, and every attention kernel here walks keys forward. Finally,
hVQ composes with all of this at no cost, because its row gain is applied after
the reduction and is fit from quantities that do not depend on the kernel; that
composition is section 9.

hVQ is our format, not an FP8 arithmetic scheme: it refits the indices and row
gains of an existing iVQ codebook under a Hessian metric
(`tools/vq_calib/convert_layer_ds41_hvq.py:1-20`, `tools/vq_calib/ds41_hvq.py:82-90`).
It changes which codebook entry a weight points at; the codebook stays E4M3 and
no FP8 instruction changes. Section 8 covers it on its own terms and section 9
covers how it combines with an FP8 kernel change. The rest of this note is about
FP8 arithmetic, which is a different question.

Nineteen works, all read first-hand (abstracts and method pages, cited by page):

| Work | Id | Used for |
|---|---|---|
| FP8 Formats for Deep Learning | [2209.05433](https://arxiv.org/abs/2209.05433) | E4M3/E5M2 definitions and the range/precision split |
| Microscaling Data Formats for Deep Learning | [2310.10537](https://arxiv.org/abs/2310.10537) | MX block layout, E8M0 scales, scale axis rules |
| MOSS: FP8 LLM training with microscaling | [2511.05811](https://arxiv.org/abs/2511.05811) | Two-level scaling, and the measured cost of per-group dequantisation |
| FP8 Quantization: The Power of the Exponent | [2208.09225](https://arxiv.org/abs/2208.09225) | Why E4M3 beats INT8 for PTQ |
| P-Cast Precision in FP8 Attention | [2606.06521](https://arxiv.org/abs/2606.06521) | P-cast failure modes, S = 2^8, reverse KV iteration |
| Towards Fully FP8 GEMM LLM Training at Scale (FOG) | [2505.20524](https://arxiv.org/abs/2505.20524) | Where activation outliers come from, and why attention is quantised last |
| TWEO: Transformers Without Extreme Outliers | [2511.23225](https://arxiv.org/abs/2511.23225) | Outliers are structural, not data-driven |
| LiquidGEMM | [2509.01229](https://arxiv.org/abs/2509.01229) | The dequantisation instruction budget, and the pipeline that meets it |
| FireQ | [2505.20839](https://arxiv.org/abs/2505.20839) | Post-RoPE key quantisation with FP8 scales; per-tensor scaling against underflow |
| FP8-RL | [2601.18150](https://arxiv.org/abs/2601.18150) | Blockwise FP8 W8A8, per-step KV scale recalibration, excluded layers |
| SonicMoE | [2512.14080](https://arxiv.org/abs/2512.14080) | Padding waste in grouped GEMM |
| TMA-Adaptive FP8 Grouped GEMM | [2508.16584](https://arxiv.org/abs/2508.16584) | Removing the padding requirement entirely |
| Fast Matrix Multiplication in fp8 | [2609.26077](https://arxiv.org/abs/2609.26077) | Block-scaled E4M3 error model, coefficient geometry |
| AQLM | [2401.06118](https://arxiv.org/abs/2401.06118) | Multi-codebook quantisation, the general form of our hVQ step |
| FLUTE | [2407.10960](https://arxiv.org/abs/2407.10960) | Vectorised and bank-duplicated shared-memory lookup tables for LUT-quantised weights |
| VPTQ | [2409.17066](https://arxiv.org/abs/2409.17066) | Vector codebooks over weight groups, the closest published artifact to hVQ |
| HARP | [2605.29843](https://arxiv.org/abs/2605.29843) | Learned rotations versus fixed Hadamard for outlier control |
| FlashAttention-3 | [2407.08608](https://arxiv.org/abs/2407.08608) | Warp-specialization, GEMM-softmax overlap, and why an FP8 attention kernel idles its tensor cores |
| SageAttention3 | [2505.11594](https://arxiv.org/abs/2505.11594) | FP4 attention: two-level P scaling, 1x16 groups, and the accumulator-layout permutation |

A note on what this note is not: an earlier draft of it sampled one third-party
engine's code path for leads. That material is gone. Every claim below now rests
either on a peer-reviewed or preprint FP8 source, on PTX ISA 9.5, or on a file in
this repository.

## 1. The scaling cost is the whole FP8 kernel problem

The formats are settled: E4M3 for forward (3 mantissa bits, 448 max), E5M2 where
range matters more ([2209.05433](https://arxiv.org/abs/2209.05433)), and E4M3 is
the better post-training choice than INT8 across a wide network set, with the
exponent-bit count driven by outlier severity
([2208.09225](https://arxiv.org/abs/2208.09225)). None of that decides a kernel.

What decides a kernel is where the scale is applied. MOSS measures the problem
directly: on an H100 the FP32 CUDA-core peak is 1.6 percent of the FP8 tensor
peak, so dequantising one partial sum in a per-group GEMM costs the equivalent of
about 60 tensor-core MACs, and a per-group scheme (COAT) loses to a per-tensor one
(Transformer Engine) despite being numerically better
([2511.05811](https://arxiv.org/abs/2511.05811) [page 2-3]). Their fix is
two-level: an FP32 global factor per tensor, and a power-of-two E8M0 factor per 32
elements, so the main loop stays on the tensor cores and dequantisation moves to
the epilogue.

That is the same conclusion our own records reached from measurement rather than
from the model: the GLM FP8 decode dot is a byte-count win, not a tensor-core win
(`docs/glm53-vq-fp8-2026-10-16.md:17`, `docs/gb10-hardware.md:117`), and the
expert kernels that do reach the tensor pipe are limited by the epilogue, the
shared-memory operand path and the weight stream, not by MMA issue
(`cuda/minimax_mma8w.cuh:38-56`).

Two corollaries worth stating because they prune work:

- Fine-grained scaling is not free, and the price is paid on CUDA cores. A change
  that adds a scale granularity must remove an instruction per element somewhere
  else, or it loses even when it is more accurate.
- LiquidGEMM's cost model makes the bar explicit: to overlap dequantisation with
  the weight stream, the cost must fall under roughly 0.5 instructions per
  dequantised element, against QServe's dozens
  ([2509.01229](https://arxiv.org/abs/2509.01229) [page 4-5, 7-8]). Their answer
  is two arithmetic instructions per four elements (IMAD plus XOR), a packed
  layout so one `LDS.128` feeds two MMAs, and a single-producer multi-consumer
  pipeline with no dequantisation warp group of its own.

## 2. The hardware now applies the fine scale for free, and we do not use it

The two-level scheme in MOSS was written in Triton because Hopper has no
microscaling tensor instruction. `sm_120a` and `sm_121a` do, and a compiler check
in this session confirms it (section 2):

    mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X
        .f32.e4m3.e4m3.f32.ue8m0
        {d0..d3}, {a0..a3}, {b0,b1}, {d0..d3},
        scaleA, {byte_id_a, thread_id_a}, scaleB, {byte_id_b, thread_id_b};

`nvcc -arch=sm_121a -cubin` emits `QMMA.SF.16832.F32.E4M3.E4M3.E8`; the plain
form we ship emits `QMMA.16832.F32.E4M3.E4M3` with no scale operand. The same
form compiles for `sm_120a` and fails for `sm_89`, so a throughput measurement
needs the GB10; the host this note was written on is a 4070 SUPER at capability
8.9.

Three ISA facts the design depends on, from PTX ISA 9.5:

- `.stype` is `.ue8m0` only and `.scale_vec::1X` is the only granularity, so the
  block is 32 along K and the scale is a power of two. That is exactly the MX
  level-2 factor MOSS stores as E8M0.
- `scale_A` is M x 1 and `scale_B` is 1 x N, and the selector tuple must be
  braced; a flat operand list is rejected by ptxas. The byte id may be a
  register, so the selector is not limited to immediates.
- It computes `(A * scale_A) * (B * scale_B) + C` with FP32 accumulation and an
  unspecified order, so adopting it is a value change.

Why this matters here specifically. Our expert kernels already use a power-of-two
scale on the attention side of ds41 — `v41_act_quant_kernel` computes
`s = 2^ceil(log2(amax/448))` per block of at most 32 elements
(`cuda/ds41_attn.cuh:173-190`), which is MXFP8's level-2 factor with the same
block size ([2310.10537](https://arxiv.org/abs/2310.10537) [page 3]). The expert
prefill arm does not: it uses one full-precision `amax / 448` per (row, 128)
(`cuda/glm_vq_mma8.cuh:11,29`, `cuda/ds41_vq_prefill_fp8.cuh:14`), and then
promotes each 128-k stage with four FMAs per output pair
(`cuda/glm_vq_mma8.cuh:288,328-334`; `cuda/beam_mma8w.cuh:515-529`;
`cuda/minimax_mma8w.cuh:363-380`).

So the proposal is narrow and testable: make the expert activation scale MX-like
(E8M0 per 32, plus an FP32 row factor to keep the range that a bare power of two
wastes), then let the instruction apply it and drop the per-stage promotion. The
numerical expectation is neutral to better, because per-32 groups follow the
dynamic range across K more closely than per-128, and a power-of-two scale is
exact in binary32 by the same argument the P-cast paper proves for S = 2^k
([2606.06521](https://arxiv.org/abs/2606.06521) [page 5]).

Only the activation side can move. The weight side carries a per-row f16 gain
applied after the dot, measured faster than per-weight rounding
(`docs/glm53-vq.md`, "Apply the VQ gain after the dot"), and a gain that is not a
power of two cannot be `scale_A`. In the Beam and MiniMax orientation the weights
are the A operand (`cuda/beam_mma8w.cuh:9-18`), so their `scale_A` stays 1.0.

Outcome, measured 2026-11-22 on GB10: the form holds the plain form's rate at
100.0 percent — 246.1 TFLOPS for both forms, same binary, same run, on an idle
GPU at a 2,366 MHz boost clock (an earlier run with a server resident measured
99.3 percent). The absolute figures are not comparable to the recorded
215.6 TFLOPS because the clock policy service was inactive, so the ratio is the
result and the level is not. What follows is that the hardware scale is
effectively free here, so the case for this instruction rests on instruction
count and register pressure, never on MMA throughput.

Line of enquiry as written, with item 1 now answered:

1. ~~Rate of this instruction against the plain form~~ **Answered: 99.3
   percent.** The harness reference is `E4M3_F32` beside `E2M1_NVF4`, reported
   against 215.6 TFLOPS at 2,190 MHz (`docs/gb10-hardware.md:96`).
2. If it holds, the ds41 FP8 expert arm: E8M0 per 32 on the activation side,
   gated on the existing teacher-forced NLL and KL against the chunk-size floor
   (`docs/deepseek41-optimization-2026-11-12.md`, floor KL 0.0154).
3. Only then the GLM, Beam and MiniMax arms, which need their own long-context
   gate because their per-128 scale changes too.

### The compiler check, reproduced

The check is one file and two commands. It was run on 2026-11-22 with CUDA 13.3.73
and PTX ISA 9.5, and it needs no GPU for the compile step.

    #include <cstdint>

    __device__ void e4m3_bs(float d[4], const uint32_t a[4], const uint32_t b[2],
                            uint32_t sa, uint32_t sb) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4"
                     ".block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
            "%10, {%11, %12}, %13, {%14, %15};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
              "r"(sa), "h"((uint16_t)0), "h"((uint16_t)0),
              "r"(sb), "h"((uint16_t)0), "h"((uint16_t)0));
    }

    __device__ void e4m3_plain(float d[4], const uint32_t a[4], const uint32_t b[2]) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }

    __global__ void probe_kernel(float *out) {
        const uint32_t a[4] = {0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
        const uint32_t b[2] = {0x38383838u, 0x38383838u};
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        e4m3_bs(d, a, b, 127u, 127u);      /* ue8m0 127 = 2^0 */
        e4m3_plain(d, a, b);
        out[threadIdx.x] = d[0] + d[1] + d[2] + d[3];
    }

    nvcc -arch=sm_121a -cubin -o /tmp/p.cubin probe.cu
    cuobjdump -sass /tmp/p.cubin | grep QMMA

Observed SASS on `sm_121a` and `sm_120a` (registers differ with allocation; the
opcodes are the point):

    QMMA.SF.16832.F32.E4M3.E4M3.E8 R4, R8, R2.reuse, RZ, R4, R4, URZ ;
    QMMA.16832.F32.E4M3.E4M3       R4, R8, R2, R4 ;

Three failure modes worth recording, because each costs a compile cycle to
rediscover:

- A flat operand list without the braces is rejected: "Arguments mismatch for
  instruction 'mma'", and a bare selector reads as "Vector expected for argument
  5".
- The type string order is `...row.col.kind::mxf8f6f4.block_scale...f32.e4m3.e4m3`
  with `.kind` before `.block_scale`; the `.f32` destination type comes after the
  layout qualifiers, not before them.
- `sm_89` rejects the form outright (five ptxas errors for the two-function
  snippet above, thirteen for a four-function variant), so a workstation check
  cannot stand in for the GB10.

The mixed `.f32.e2m1.e4m3` variant emits `QMMA.SF.16832.F32.E2M1.E4M3.E8`: the
same instruction with a 4-bit A operand packed in an 8-bit container, 2 bits of
padding above and below [PTX ISA 9.5, 9.7.16.4]. It is the only mixed 4-bit by
8-bit tensor product this part offers per warp, so a 4-bit-weight, 8-bit-
activation kernel must go through it rather than through a separate FP4 path.

## 3. Where this engine actually calculates on FP8 tensor cores, and why it is slow

An earlier version of this note was organised around "what can we change in hVQ",
which is the wrong axis: hVQ is a storage format whose kernels do no FP8
arithmetic beyond feeding a codeword to the MMA. The axis that matters is where
the engine multiplies, and at what fraction of the peak that instruction reaches. Asked that
way, the answer is not the experts.

From the 2026-11-12 records on a 65,536-token prefill
(`docs/deepseek41-optimization-2026-11-12.md:210-215`,
`docs/deepseek41-optimization-2026-11-12-dense.md:172-188`):

| Kernel class | ms per block | Share | Rate |
|---|---:|---:|---|
| Routed experts (FP8 kernels) | 334.9 | 29.5% | 75% of the DRAM stream bound |
| Sparse attention | 270.4 | 23.8% | 13.5 ms per block-layer; window keys constant, pool keys per layer from the artifact |
| q4_K projections | 244.5 | 21.5% | fused path, F32→BF16 conversions around it |
| mHC | 171.0 | 15.1% | DRAM-bound, reads the hyper state twice |
| Index scores and top-k | 31.3 | 2.8% | already fixed by round 1 |

So the experts — the class every FP8 kernel in this note targets — are at 29.5
percent of the block but within a quarter of their bandwidth floor. The second
largest class, sparse attention at 23.8 percent, is the one that computes on
tensor cores and does it badly.

**How badly, measured honestly.** The record states "about 13 TFLOPS" for sparse
attention, and an earlier draft of this section contradicted it by a factor of
32. The error is worth recording because it inverted the phase ranking, and a
tool now prevents it (`bin/attn-bench`, `docs/attn-bench.md`).

The figure is per **block**, and the block is 2,048 rows through layers 0-19,
not a whole-prefill layer. The record's own words are "per 2,048-row block of
layers 0-19" (`docs/deepseek41-optimization-2026-11-12-dense.md:43`), and the
same document's independent numbers confirm the convention: "663 GFLOP per
layer (13.3 TFLOP per block)" (`:54`), which is 663 GFLOP x 20 layers. Against
that block,

    2,048 rows x 64 heads x 640 keys x 512 dims x 2 products x 2 FLOP
      = 0.1718 TFLOP per layer, and 0.1718 TFLOP / 13.5 ms = 12.7 TFLOPS

which is 11.8 percent of the 107.8 TFLOPS BF16 peak and reproduces the record's
"about 13 TFLOPS". The earlier 0.4 came from dividing a whole 65,536-token
layer's 5.5 TFLOP by the per-block 13.5 ms, mixing two conventions; 65,536/2,048
is exactly 32, and 0.4 x 32 = 12.8.

One caveat on the 640, and it is a correction to an earlier draft of this
section as well: 640 keys is the shape of most layers, not of all of them. The
artifact's own metadata (`deepseek4.attention.compress_ratios`, 40 entries in
the release file) reads zero for layers 0-1, two for layers 2-19, and one for
layers 20-39. A ratio of zero means no compressed pool, so layers 0 and 1
attend to the 128-key sliding window alone; the attention sum over layers 0-19
therefore covers 2 window-only layers and 18 at 640 keys. The rate above is
quoted for the 640-key shape and is not diluted by that, but any per-layer
budget that assumes a uniform 640 is wrong by those two layers.

The corrected rate changes the conclusion. The class is not two orders of
magnitude below its peak: it is at 12 percent, which is a normal
tensor-core-attention number and leaves real headroom. What it does not settle
is where that 88 percent goes, and the phase ranking below depends on that
split, which is the counterfactual `attn-bench` performs.

### The correction that matters: ds41's attention is BF16, and its KV is already FP4

Section 3 above was first written as if sparse attention were an FP8 kernel that
needed to become one. It is the reverse, and the difference changes the phase.

`ds4_fa_mma` issues `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
(`cuda/ds41_attn_mma.cuh:151-153`), and the kernel `ds4_sparse_attn_mma_kernel`
(`:189`) uses it for both products, QK and PV. So the class runs against the
**107.8 TFLOPS BF16 peak**, not 215.6, and at 13.5 ms per block-layer its
arithmetic is 11.8 percent of that.

What is already low precision is the storage, and this is the DS41-specific fact
that no generic FP8 survey would surface: the compressed KV is **FP4 with E4M3
scales**. The gather decodes 32 scales, one per lane, and four-bit values, one
per lane, then resolves each element with two shuffles
(`cuda/ds41_attn_mma.cuh:47-52`, `:58-80`; the comment records 2.4 → 1.4 ms per
step at 12k for the earlier per-element form). So the kernel takes FP4 in, widens
it to BF16 in shared memory, and multiplies in BF16.

The block-scaled FP4 tensor instruction consumes exactly the stored pair:
`mma.sync ... kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1`
with a `ue4m3` scale, which the tree already links at
`tools/gb10_peaks.cu:50-56` and measures at 509.5 TFLOPS at ~2.5 GHz
(`docs/gb10-hardware.md:98`). No decode to BF16 and no widening: the operand
arrives in the form the MMA wants.

That is the largest and most specific opportunity in this note, and it is not an
FP8 one. It is also why the phase order below puts attention first: the class is
23.8 percent of the block and 11.8 percent of its own peak. What the measurement
then changed is *which* part of it to attack: section 9.4's `attn-bench` run
puts the compressed pool at 1.2 to 1.3x the F32 window path per tile at matched
tile counts, so the decode is the cost, not the instruction.

**The precedent that this is fixable, in this tree.** Gemma's full-layer E4M3 KV
had the same shape — `g4_attn_fp8_kernel`, one warp per (row, head), a scalar
loop over every visible key — and was 90.5 percent of GPU time at 32,768 keys and
95.0 percent at 65,536. Moving it to tensor cores with the hardware
`e4m3x2 → f16x2` conversion took prefill at 65,536 from 47.2 to 460.3 tok/s, a
9.75 times improvement, and decode at 32,768 from 0.50 to 19.27
(`docs/gemma4-vq-optimization-2026-11-05.md:121-150`). ds41's sparse attention is
already on tensor cores — `cuda/ds41_attn_mma.cuh` — so the same 10x is not
available. What is available is the fraction of the peak that a tensor-core
attention kernel can reach once it is written properly.

### What the literature says that fraction is

Three works, in order of how directly they apply.

**FlashAttention-3** states the structural reason and the cost of ignoring it:
FlashAttention-2 reached 35 percent utilisation on H100 against 80-90 percent for
optimised GEMM, because it "makes no explicit use of asynchrony and low-precision
in its design". Its three fixes are warp-specialization with a circular shared
buffer, ping-pong scheduling that overlaps one warpgroup's softmax with another's
GEMM, and FP8 with block quantization plus incoherent processing. It measures: FP16
at up to 740 TFLOPS (75 percent utilisation), FP8 "close to 1.2 PFLOP/s", and
FP8 2.6 times more accurate than per-tensor FP8 attention on outlier features
([2407.08608](https://arxiv.org/abs/2407.08608) [page 1-2, 5-6]).

The arithmetic inside that is the part worth stealing, and it is architecture
independent. On H100, 989 TFLOPS of FP16 matmul against 3.9 TFLOPS of special
functions means the softmax exponential can consume about 50 percent of the cycle
even in FP16; at FP8 "the matmul throughput doubles but the exponential throughput
stays the same". So a hand-written FP8 attention kernel that computes QK, waits,
computes softmax, then computes PV is leaving half the tensor cores idle by
construction, and no amount of MMA tuning fixes it. The fix is structural overlap,
not a better instruction.

**SageAttention3** is the FP4 successor and the most complete account of the
attention-specific hazards ([2505.11594](https://arxiv.org/abs/2505.11594)
[page 1-5]):

- The attention map P is concentrated in [0, 1], so quantising it directly forces
  the scale factors into a narrow range — and the hardware requires those scales
  in FP8, which loses accuracy. Their fix is two-level: normalise each row to
  [0, 448/6] first, then microscale, which "maximizes the E4M3 range utilization
  for s_P".
- Constrain the quantisation group to 1x16 rather than per-tensor or per-channel,
  "to contain outlier effects within each block". That is the same granularity
  argument as section 1 of this note, and it chose NVFP4 over MXFP4 because "the
  accuracy of NVFP4 is much higher than that of MXFP4 in attention quantization"
  — with MXFP4's scale in E8M0, the format our block-scaled FP8 instruction uses.
- The FP32 accumulator layout differs from the operand A register layout, so
  matching them by thread shuffle costs more than permuting the P tile's columns
  and rearranging K to compensate, fused into the quantisation kernel.
- The 1x16 max for the scale needs elements spread across four threads, so they
  fuse the quantisation with online softmax and reuse the running row max,
  removing 50 percent of the redundant shuffles and max operations for about 10
  percent of whole-kernel speed.

**The P-cast paper** is section 4 of this note: at 8-bit the sink-induced collapse
of P on a forward key walk is a first-order effect, and the reverse iteration plus
S = 256 makes the quantised P behave like the exact one
([2606.06521](https://arxiv.org/abs/2606.06521) [page 1-5]).

### What that implies for ds41's sparse attention, concretely

The class is 21 to 24 percent of a prefill block. If a tensor-core attention
kernel reached even a quarter of the FP8 peak on its blocked products, the class
would shrink by more than an order of magnitude, and that is the largest single
number available anywhere in this note. The experts cannot offer that: they are
within a quarter of their DRAM bound, so no arithmetic change helps them.

Four requirements follow from the sources, and each is checkable in our kernel:

1. **Is softmax overlapped with the MMAs, or serialised between them?** The
   kernel's structure at `cuda/ds41_attn_mma.cuh` is a two-pass split with
   `DS4_ATTN_MMA_WARPS = 8`, "pass 1/2 split K, the O step splits the output
   dim". Whether a warpgroup's exponential runs while another's MMA is in flight
   is the first thing to establish, because it decides the ceiling before any
   tuning.
2. **Is P quantised at all today?** No: ds41 writes P in BF16
   (`cuda/ds41_attn_mma.cuh:581`). So the blocked product is BF16, not FP8, and
   the 215.6 TFLOPS figure was never reachable by this kernel. Enabling FP8 PV is
   section 4's lever, and it needs the reverse walk first.
3. **What is the scale granularity on Q and K?** ds41's attention side already
   uses a power of two per block of at most 32 dims
   (`cuda/ds41_attn.cuh:173-190`), which is MXFP8's level-2 factor. SageAttention3
   found NVFP4's E4M3 scale materially better than MXFP4's E8M0 for attention —
   so before folding that scale into the block-scaled instruction, note that the
   one source that compared them in attention chose the finer scale. Section 2's
   proposal is sound for the experts, where the operands are wide; it is not
   established for P.
4. **Does the accumulator layout match the operand layout?** If not, SageAttention3
   says permute the P columns and rearrange K in the quantisation kernel rather
   than shuffling at run time.

The measurement that settles all four: an `ncu` GPU Speed Of Light capture on the
attention kernel at a 65,536-token prefill, reporting tensor pipe active, the
shared-memory and LSU pipes, and the stall reasons. That is one capture on the
idle box and it turns this section from argument into a target.

## 4. P-cast: our next attention step must reverse its key walk

This is the most directly actionable paper in the set, and it applies to a change
we have already listed as a next lever.

Casting the softmax output P to E4M3 before the PV product gives only 8
representable values per binade, a relative precision of 12.5 percent, so two
implementation details that are invisible in higher precision become first-order
([2606.06521](https://arxiv.org/abs/2606.06521) [page 1-4]):

- **A forward key walk collapses P.** Attention sinks make the first block's
  logit the running maximum, which pushes every later probability toward zero. A
  P value underflows to zero when `p * S < 2^-10`, and with scale S the fraction of
  non-sink values that vanish is `Phi(Delta + delta_k - 6.93 - ln S)`. At sink
  strength 7 and S = 1 that is 94.8 percent of non-sink values; at S = 256 it is
  0.3 percent. Reverse iteration, so the sink block is processed last, removes the
  effect with a zero-underflow guarantee when combined with S = 256.
- **S should be 256, not 448.** 256 is bit-exact in binary32 and sits on the lower
  envelope of the worst-case quantisation step; FlashAttention-3 and 4 use it,
  while FlashInfer and TensorRT-LLM use 448. Measured MSE improves by 3 to 10
  times at moderate sink strength, and the two fixes saturate rather than stack.

Our state today: no kernel casts P to E4M3. GLM's flash kernel rounds P to BF16
for the PV MMA (`cuda/glm_vq_flash.cuh:20`), ds41's scalar path does the same
(`cuda/ds41_attn.cuh:9`), ds41's tensor path writes P in BF16
(`cuda/ds41_attn_mma.cuh:581`), and MiMo's prefill8 keeps V and PV as they were
(`docs/mimo2-vq-optimization-2026-10-31.md:52-56`). Every one of those kernels
walks its key tiles from the oldest or first retained key toward the newest:
`cuda/ds41_attn.cuh:319`, `cuda/ds41_attn_mma.cuh:222,421,568,581`,
`cuda/gemma4_attn_fast.cuh:59`, `cuda/mimo2_prefill.cuh:171`, and GLM's
`for (uint32_t tile = first; tile < last; tile++)` at `cuda/glm_vq_flash.cuh:160`.

And our ds41 attention has a sink: `cuda/ds41_attn.cuh:9` records "sink in the
denominator only". So the failure mode this paper quantifies is live for us, not
hypothetical: the day someone enables E4M3 P on a forward walk, most non-sink
probabilities of a strong-sink layer vanish, and the loss shows up as attention
drift rather than as an error.

The requirement for the already-listed lever ("P x V in E4M3",
`docs/mimo2-vq-optimization-2026-10-31.md:135`) is therefore two coupled changes:
reverse the KV tile walk and fix S = 256. Neither is expensive; both are
invisible at BF16 and mandatory at E4M3. Agreeing with the paper, the same
analysis is what makes the accepted E4M3 Q results
(`docs/mimo2-vq-optimization-2026-10-31.md:41-100`) safe to extend: Q and K are
quantised before the softmax, so they do not collapse anything, whereas P is
quantised after it.

## 5. Picking the FP8 format, and where attention is quantised last

FOG is the clearest account of why attention is the last FP8 conversion: its
authors get FP8 into every GEMM of the block, including attention, only after
changing the architecture — removing pre-normalisation, freezing the QK RMSNorm
gains, and adding post-normalisation — and they measure up to 43 percent
throughput at 8B with quality matched to BF16
([2505.20524](https://arxiv.org/abs/2505.20524) [page 1-4]). Their diagnostic is
kurtosis, used to predict divergence from short runs.

TWEO goes further and argues the outlier is structural rather than data-driven:
extreme outliers come from colinearity in the weight matrices, and a single loss
term takes them from over 10,000 to under 20, which then makes plain W8A8
per-tensor static quantisation viable ([2511.23225](https://arxiv.org/abs/2511.23225)).

For us these are artifact properties, not serving decisions: our weights are
quantised offline by converters, and the activation outliers of a given family
are what they are. What transfers is the ordering — weight and GEMM quantisation
first, attention last, with an outlier measurement rather than an assumption
behind each step — which is the order our own campaign already follows (expert
and index kernels before attention, `docs/deepseek41-optimization-2026-11-12.md`).

One boundary the MX paper states and our format must respect: the scale axis is
the reduction dimension, and transposing an MX tensor is not the same as
transposing and then quantising, so a transposed copy has to be stored and
quantised separately ([2310.10537](https://arxiv.org/abs/2310.10537) [page 3]).
Our per-row gains and per-(row, 128) activation scales are laid out for the
reduction order the kernel uses; a change of that order is a re-quantisation, not
a relabelling.

## 6. What the FP8 KV cache literature adds, and where it disagrees with us

Two papers give a policy for FP8 attention caches, and both are relevant because
our E4M3 KV work is currently row-scaled and quantised at write time.

FireQ quantises keys and values to INT4 with FP8 (E4M3) scales and queries to
FP8, and the part that transfers is the geometry rather than the bit width: it
quantises keys **after** RoPE, per token, because pre-RoPE quantisation needs a
dequantise-quantise round trip around the rotation
([2505.20839](https://arxiv.org/abs/2505.20839) [page 4]). It then splits the key
channels into two classes and smooths each on its own side of the rotation:
channel-wise RoPE scaling after it for the channels whose distribution the
rotation mixes, and a RoPE-preserving normalisation before it for the regular
pairs, both folded into the weights offline. Their result is that INT4 keys with
FP8 scales stay accurate, which is a stronger claim than our E4M3 rows need.

FP8-RL takes the inference-side view and adds the detail that matters for a
serving engine: the KV scale is **recalibrated per step** rather than fixed at
write time, which is what removes the long-context capacity loss from a stale
scale, and its W8A8 comes with a mismatch correction (token-level truncated
importance sampling, clip at 2) because a quantised policy is not the policy the
sampler assumes ([2601.18150](https://arxiv.org/abs/2601.18150) [page 2-4]). The
same paper excludes the LM head and the norms from quantisation.

Where this disagrees with us is worth stating plainly. Our position, measured
twice, is that FP8 KV pays only when the bytes saved exceed the dequantisation
cost: MiMo's E4M3 full-layer KV bought 5.4 percent mean prefill and 10.9 percent
at 64K (`docs/mimo2-vq-optimization-2026-10-31.md:41-100`), Gemma's E4M3 KV on
tensor cores was decisive because the alternative was a scalar per-key loop
(`docs/gemma4-vq-optimization-2026-11-05.md:121-150`), and the recipe log reports
FP8 KV losing 2.4 percent at 16K once the larger cache cancels the saved
dequantisation. The literature's framing, a fixed capacity win at long context,
holds only when the scale is recalibrated and the dequantisation is free in the
instruction; both of those are engineering conditions, not format properties.

Our own quantization shape is already close to the MX level-2 factor on the
attention side: `v41_act_quant_kernel` uses a power-of-two scale over a block of
at most 32 along the reduced dimension with `s = 2^ceil(log2(amax/448))`
(`cuda/ds41_attn.cuh:173-190`), which is the MXFP8 layout without the standard's
name ([2310.10537](https://arxiv.org/abs/2310.10537) [page 3]). Two things in the
literature follow from that: the transpose rule of section 5 means a K-major
layout and a V-major layout cannot share one quantised copy, and a per-step
recalibration needs the scale to be a live device value rather than a captured
constant, which is the same constraint our captured-decode rules already impose
on positional state.

## 7. Grouped-expert tails are an FP8-shaped cost, and our data has them

Two papers address the same problem: grouped FP8 GEMM over experts wastes work on
padding, and both remove the padding rather than tune around it.
TMA-Adaptive FP8 Grouped GEMM uses a descriptor pool to handle every residual row
case and measures 1.7 to 20.4 percent faster with up to 23.8 percent less memory
([2508.16584](https://arxiv.org/abs/2508.16584)); SonicMoE's "token rounding"
targets the same waste and reports 1.86 times compute throughput over a BF16
grouped kernel ([2512.14080](https://arxiv.org/abs/2512.14080)).

Our kernels take the tail differently, by staging padded n8 tiles and giving
small expert items their own tile. The measured distribution says the tail is not
rare: in MiniMax's routed prefill, 29 percent of work items carry 16 or fewer
pairs, and a unit streams its whole 120 KB index block regardless of its token
count, so units under about 46 tokens are bound by ring latency rather than by
arithmetic (`docs/minimax-vq-optimization-2026-10-25.md:64`, `cuda/minimax_mma8w.cuh`
"Measured limits"). That is a statement about *our* routes, and it is the same
phenomenon the two papers attack.

This is the second-ranked probe for that reason: it needs no new instruction and
no numerical change, only a smaller tile (or a merge of adjacent small items) for
the short tail. Gate it on the existing parity test
(`make test-minimax-mma8w`, which holds the warp-specialized kernels to
`glmvq_mma8_kernel` bit for bit) and on real-route replay rather than synthetic
routes, since uniform-random routes understate the skew
(`docs/gb10-hardware.md`, Beam's route-replay note).

## 8. hVQ is a quantisation-assignment result, not an FP8 arithmetic one

Worth stating precisely, because the two get conflated:

- The artifact's codebook is E4M3 and the kernels consume those bytes unchanged
  (`cuda/ds41_vq_prefill_fp8.cuh:1-30`). No FP8 instruction changed when hVQ
  landed.
- What hVQ changes is the assignment: each 8-column block's index is refitted with
  error feedback under `H_t = D^1/2 ((1-t) I + t R) D^1/2 + damp * mean(D) I`,
  where `R` is the expert's input correlation, with rare experts shrunk toward the
  layer mean (`tools/vq_calib/convert_layer_ds41_hvq.py:1-20`,
  `tools/vq_calib/ds41_hvq.py:82-90`). `t = 0` reproduces the iVQ conversion
  exactly, which is what makes it a controlled experiment.
- The result is the point: plain weight error rose 21.5 percent while the
  activation-weighted output error fell 51.0 percent, and generated-position KL
  fell 17.0 percent (`docs/deepseek41-hvq-evaluation-2026-11-11.md`).

That is the metric-mismatch principle in its cleanest form: a codebook assignment
minimised in the Frobenius norm is the wrong objective, and the operative metric
is the one the next layer actually applies. AQLM makes the same argument from the
multi-codebook side, learning additive codebooks and optimising them jointly per
block rather than per weight ([2401.06118](https://arxiv.org/abs/2401.06118));
hVQ is the single-codebook, fixed-book analogue with a learned metric. HARP is
the rotation-based version of the same idea for low-bit quantisation, replacing a
fixed Hadamard transform with a learned structured one
([2605.29843](https://arxiv.org/abs/2605.29843)).

Two consequences for FP8 work specifically. First, an hVQ-style refit and an FP8
kernel change are independent axes and should be measured separately: one moves
which codeword, the other moves how the codeword is multiplied. Second, because
the codebook is E4M3 and shared, a refit that improves the activation-weighted
error also improves what the FP8 kernel computes — the gain is realised in the
existing arithmetic, which is why hVQ needed no kernel work at all.

The other literature lesson that applies to our conversion pipeline is the
transpose rule from section 5: any weighting, rotation or refit that changes the
reduction order must be followed by a re-quantisation of the transposed copy, and
our `R`-based metric is already defined in the input-column order the kernel
reduces over.

## 9. Combining hVQ with the block-scaled FP8 instruction

The two meet at one line of arithmetic, and the composition is decided by where
hVQ's row gain is applied.

### 9.1 Which part of an hVQ layer is FP8 arithmetic at all

hVQ stores a weight as one 12- or 13-bit index per eight elements, into a shared
64 KiB book of E4M3 codewords, plus one F16 gain per row. The kernel's job each
step is: read the index, fetch the 8-byte codeword it names, and hand that
codeword to the tensor core as an operand. Only the last part is FP8 arithmetic,
and it is the part that is *not* the bottleneck.

That distinction has a direct consequence for this whole note. The MOSS and
LiquidGEMM argument of section 1 — dequantisation on CUDA cores costs more than
the multiply — applies to hVQ in a stronger form than to any FP8 artifact, because
hVQ's decode does no arithmetic dequantisation whatsoever. A codeword is already a
legal `mma.m16n8k32.e4m3` operand as stored (`cuda/minimax_mma8w.cuh:6-9`, "a book
vector is the m16n8k32 B fragment of its weight row, so one LDS.64 per (row, k32)
feeds the MMAs"). So hVQ's cost is a **shared-memory gather plus an index decode**,
not a conversion, and the levers are different from the ones the FP8 literature
offers.

The index decode is cheap and already minimal. `glmvq_mma8_code` reads a 12-bit
window as a 64-bit load shifted by the codeword position (`cuda/glm_vq_mma8.cuh:86-90`),
and the 13th bit arrives as one plane bit per codeword (`:96-102`, `:92-94`). Its
whole cost is a shift and a mask per codeword, on the integer pipe, off the
critical path.

### 9.2 The measured bottleneck, and why my earlier proposal does not address it

The kernel's own header records the profile, and it is unusually specific
(`cuda/minimax_mma8w.cuh:38-56`, measured with `tests/test_minimax_mma8w` at 2,048
rows with random routes):

- gate takes 4.8 ms with weights streamed, 3.6 ms with every expert resident in L2,
  and **2.3 ms with the producers' copies removed** — which is 134 TFLOPS, 53
  percent of the E4M3 peak. So the tensor cores alone can reach 53 percent; the
  copies take the rest.
- `ncu`, two slots: **tensor pipe 38-44 percent, shared-memory pipe 48 percent, and
  42 percent of the shared-memory load wavefronts are book-gather bank conflicts**.
  The consumers spend their wait on the `full` barrier.
- Rejected variants, each with a number: 128-pair items over 64-k stages (+16-19
  percent, slower), separate index and input barriers (+16 percent on R13),
  accumulators rescaled per stage (even), and — this one matters most here — "a
  k-outer loop with one activation scale per row and no per-stage promotion (180
  registers, even)".

That last entry is a direct empirical refutation of a proposal in an earlier draft
of this note's plan: removing the per-stage promotion and moving to one scale per
row was tried, cost 180 registers, and came out even. The promotion is not the
bottleneck. Section 2's proposal to fold the activation scale into the instruction
is a different change (same promotion count, hardware-applied scale), so it is not
refuted — but its expected gain is now bounded by what the promotion actually
costs, which this measurement says is roughly nothing at these shapes. The honest
ranking follows from that: the block-scaled instruction's gain would come from
instruction count and register pressure, not from removing FMAs.

### 9.3 Where the shared-memory pipe actually goes, and what would relieve it

The book gather is one `LDS.64` per (row, k32) and it conflicts 42 percent of the
time. Two published techniques attack exactly this, and neither is in the kernel
today.

FLUTE, built for lookup-table quantised weights, names the problem the same way:
shared-memory lookups are non-uniform access, and it "vectorizes" the table by
building an entry for every *pair* of indices so one load serves two values, then
**duplicates** the table across banks so threads stop colliding. For a 4-bit table
that is 256 entries of 32-bit values, "slightly more than 1 KB" against the 48 to
163 KB of shared memory available, and it removes the 8-way conflict a naive layout
produces ([2407.10960](https://arxiv.org/abs/2407.10960) [page 5]). The mechanism is
a bank-level fix, not an algorithmic one.

VPTQ is the closest thing in the literature to our own artifact: a codebook of
centroids over groups of weights, with residual and outlier extensions, reported at
1.6 to 1.8 times the inference throughput of the prior state of the art at the same
bit rate ([2409.17066](https://arxiv.org/abs/2409.17066) [page 1-3]). Its
contribution is mostly in how the codebook is fitted — channel-independent
second-order optimisation — which is the hVQ side of the problem and not the kernel
side, but it is worth reading as a check on whether our fitting choice is the best
available.

Applying FLUTE's two ideas to our kernel, and one that does not transfer:

- **Table duplication is the mechanism, and it does not fit.** FLUTE duplicates
  its table across banks precisely so threads stop colliding. Our book is 64 KiB of
  a 99 KiB budget, so a second full copy is impossible, and a partial copy only
  helps if the duplicated region is the one being read — which for a random index it
  is not.
- **A fixed rotation does not work, and this was checked.** An earlier draft
  proposed rotating each lane's codeword address by analogy with the latent-tile
  rotation in `cuda/glm_vq_flash.cuh:60-64`. That analogy fails: rotation breaks a
  *deterministic stride*, and a random-index gather has none. A fixed permutation of
  the book is a bijection, so the collision count over uniform indices is unchanged
  in distribution; a direct simulation of 32 random 8-byte codeword reads gives the
  same conflict count before and after the rotation. Do not spend a round on it.
- **The one part that might still hold is the read width.** Two lanes' 8-byte reads
  at the same bank pair serialize, and scattering each lane's 8 bytes into two
  4-byte halves at rotated offsets was worth 6 percent, bit-identical, in MiniMax's
  R13 expert gate (`docs/gb10-hardware.md`, "L1 bank conflicts also hit
  cp.async.ca"). That is a copy-side measurement and it is about *how many*
  wavefronts one read costs, not about avoiding collisions, so it may transfer; it is
  the only one of the three that is still worth a probe.

The conclusion for the prefill kernel is therefore narrower than an earlier draft of
this section claimed. The conflicts are real and measured at 42 percent of load
wavefronts, but the published cure needs shared memory the artifact's book spends,
and the cheap rotation is not a cure. What remains available is the producer-copy
side and the read-width question above.

### 9.4 What the FP8 instruction can and cannot do for hVQ

The honest summary, and the answer to the question this section exists for:

- The tensor-core side is already at 53 percent of peak in isolation, so the
  ceiling on any pure-MMA change is that much, and the kernel runs at 38-44 percent
  with copies in play. Headroom for an instruction change is therefore real but
  second-order behind the shared-memory pipe.
- The block-scaled instruction removes a promotion whose cost is measured as
  roughly zero ("even"), and it adds a scale operand the kernel must stage. Unless
  the peak harness shows it also *issues* better than the plain form, it is not the
  lever for hVQ.
- The levers that match the measured bottleneck are: relieve the 42-percent
  book-gather conflicts, and get the producers' copies off the shared-memory pipe.
  The first has a published, low-cost mechanism (FLUTE's vectorisation plus bank
  rotation, or our own XOR rotation which costs no memory). The second is why the
  kernel is at 38-44 percent instead of 53, and the header's own note that "with
  every expert in L2, gate 6.6 / 9.1 ms" bounds what a copy fix can recover.

This reorders section 10's list, and section 10 is written to match.

**The same conclusion for ds41's attention, and this time measured.** The
argument above is about hVQ's expert kernels; `attn-bench` (`docs/attn-bench.md`
in the toolkit) puts it on attention directly. It times the shipped
`ds4_sparse_attn_mma_kernel` with the FP4 compressed pool and with the F32
window. Measured on the reference host (GB10, 48 SMs, 2,048 rows, 2,470 MHz,
scored against 118.8 TFLOPS at the device maximum):

    matched tiles        F32 window      FP4 pool      pool / window
        8 tiles           3.159 ms       3.775 ms         1.20x
       16 tiles           4.483 ms       5.487 ms         1.22x
    marginal 8->16      0.1655 ms/tile  0.2140 ms/tile    1.29x

Every tile does the same MMA work — 16 keys by 512 dims by 64 heads — so the
20 to 29 percent excess is the gather and decode: the 288-byte FP4 row, one
scale per lane, two shuffles per element. The MMA is not what makes the
compressed path expensive, and its instruction is not the lever.

The method matters and an earlier draft of this section got it wrong by
comparing averages at unequal tile counts, where a fixed per-launch cost (about
2.1 ms of a 2,048-row launch) is divided over a different denominator. Comparing
8 tiles against 32 reported the pool as cheaper than the window on the same
host, which is an artifact. Match the tile count, or take the slope.

The launch is also markedly sublinear in tiles: 32 pool tiles take 10.4 ms where
the 0.30 ms/tile marginal rate predicts 9.6 from the 8-tile point, and the first
4 tiles cost 3.0 ms against a 0.9 ms marginal. That fixed cost is what an
instruction change cannot touch, and it is why the ranking below puts the
pipeline structure before the operand format.

**The bench reproduces the record, which is the check that matters.** At 2,048
rows and the shipped 640-key shape the bench reports **12.47 ms** for the
kernel alone; the record gives 13.5 ms for the class per block-layer. So this
kernel is 92 percent of the recorded attention time, and the remaining 8 percent
is the prep and epilogue launches around it. That closes the accounting: the
study's earlier claim that a large share of the class belonged to something
other than this kernel was speculation, and it was mostly wrong.

That also bounds what the block-scaled FP4 instruction can return here. It
removes a widening whose cost is now measured as a share of a tile, not the
whole tile, and it cannot consume the compressed pool unmodified anyway: the
stored scale is one per 16 elements along `head_dim`, which is the reduction
axis of QK but the *output* axis of PV, and an MX block-scaled operand cannot be
transposed across its scale axis without a re-quantisation. PV would need a
separate scaling scheme even if QK did not.

### 9.5 Tokens per second: the decode side is a different problem

The question above is about decode throughput, and the prefill analysis of 8.2 to
8.4 does not answer it. Decode has its own arithmetic, and it points somewhere
else.

**The budget, corrected.** From the shape table (`docs/deepseek41-flash.md:112-113`):
40 layers, 384 routed experts with 6 used per token, expert FF 2304, hidden 5120.
An earlier draft of this section multiplied the *single-row* expert traffic by 40
and called the result the round's bound. That was wrong, and the error matters,
so it is recorded rather than quietly fixed: a verify pass runs 1 + D rows (2 to 6
in the records), and each of those rows routes to 6 of 384 experts *independently*.
Four independent draws hit

    384 * (1 - (378/384)^4) = 23.4 distinct experts

so the pass reads about four times the single row's expert bytes, not one times.
The port's own measurement agrees: the per-pair kernel costs 0.356 ms per layer at
n=1 and 1.283 at n=4, a factor of 3.6 (`cuda/ds41_vq_group.cuh:5-8`), and 1.283 x
40 = 51.3 ms, which is exactly the recorded verify time.

So the correct budgets are:

    one row     1.70 + 0.71 + 1.92 = 4.33 GB  -> 18.4 ms  (54 tok/s)
    4 rows, 23 experts  6.64 + 0.71 + 1.92 = 9.27 GB  -> 39.5 ms
    3 rows, 18 experts  5.10 + 0.71 + 1.92 = 7.73 GB  -> 32.9 ms

**Verify is at 77 percent of its bound, not 36 percent.** The 51.3 ms recorded
against the 39.5 ms four-row floor leaves about 19 percent of headroom, and perhaps
30 percent if the average width is closer to 3 rows. This is the single most
important correction in the note: it moves decode from "a large obvious win" to "a
small contested one", and it is why section 9 ranks decode below prefill.

**What that leaves.** The expert bytes are all but accounted for. Three things can
still explain the residual, in the order a profile is likely to rank them:

- DRAM latency rather than bandwidth: 23 scattered one-row experts per layer give
  few in-flight cache lines, and the 64 KiB book limits occupancy.
- Fixed non-MoE cost per round: attention, the indexer's top-512, router, norms,
  the shared expert and the head, which the 4.33 GB budget bundles as a single
  "dense" term but which does not shrink with width.
- The per-pair kernel's 15.7 wavefronts per round when m = 1 dominates, since
  at 1.6 bits per weight there are five times more lookups per DRAM byte than at
  8 bits: 32 indices arrive every ~23 SM-cycles while the lookups cost ~15.7.

The one view that separates them is `ncu`'s GPU Speed Of Light beside its warp
stall breakdown: high `long_scoreboard` with low DRAM utilisation means latency;
high `mio_throttle` or `short_scoreboard` with high LSU pipe means lookup issue;
high `barrier` means producer copies. Anything unpinned by that view stays
unverified.

**Where the gap lives.** Not in the experts: they are 1.70 of 4.33 GB, and the
prefill arms already reach 75 percent of the stream bound
(`docs/deepseek41-optimization-2026-11-12.md:121-122`). The gap is that decode and
verify rows "keep their kernels" — the FP8 tensor-core arms of section 9 are
prefill-only, and verify still runs the BF16 register-direct path, which rebuilds
each BF16 fragment from the E4M3 book. That is the exact conversion the FP8 arm
removed, and it is why the same weights cost 27.3 ms instead of 15.3 per layer at
wide shapes (`docs/deepseek41-optimization-2026-11-12.md:112-116`).

So the tokens-per-second lever is **the verify width's byte efficiency, not a
missing dispatch**. The port already has the machinery an earlier draft of this
section proposed: `v41_vq_group.cuh` is a grouped fused-MoE kernel for exactly this
shape, and its header states the disease and the cure in the same terms section 9.2
uses. A verify batch through the per-pair kernels costs 0.356 ms/layer at n=1 and
1.283 at n=4 — 3.6 times — because "the weight bytes do not amortize at all", and
`ncu` puts the cost in L1TEX wavefronts: 11.7 codebook lookups plus 4 activation
plus 1.5 bitstream per round, paid `m` times when `m` tokens pick the same expert.
The grouped kernel reads the bitstream once, decodes each codeword once, and pays
`11.7 + 4m` wavefronts per round against `15.7m`
(`cuda/ds41_vq_group.cuh:5-15`). That is the same insight as section 9.3, solved.

And the port has already measured the limit of it, which is what matters here: a
"mixed dispatch" sending *every* group through the grouped kernel measured 94.0 ms
against 87.5 for per-pair on a k=3 verify batch, "because most groups hold one
token, and they are slower in the 16-warp/high-register shape". So m=1 groups stay
on the per-pair kernel and only m>=2 uses the grouped one
(`cuda/ds41_vq_group.cuh:127-133`). The grouped path is already the best of both at
the current routing.

What that leaves is the honest conclusion for tokens per second:

1. **The verify width is already specialised, and what remains is the per-token
   lookup, not the weight stream.** With 6 of 384 experts routing independently per
   token, a 4-row verify batch lands about 23 distinct experts, so only a few of
   them have `m >= 2` and the grouped kernel's amortisation fires rarely. Lifting
   the *routing* overlap (an expert union, or a longer draft so more rows collide)
   is the only way to raise `m`, and it is not a kernel change: the grouped kernel
   already takes every collision it can, and an expert-union kernel measured "under
   10 percent" on this host (`AGENTS.md:285-288`).
2. **The FP8 matmul instruction is not the lever here.** At 2-6 rows the products
   are nowhere near the FP8 peak, the pass is within a quarter of its DRAM floor,
   and the grouped kernel's own accounting is wavefronts of codebook lookup rather
   than MMA issue. This is section 9.4's conclusion reached from the decode side,
   and the corrected roofline above strengthens it: there is no large slack for an
   instruction change to recover.
3. **A fixed permutation cannot fix a data-dependent gather.** An earlier draft of
   this note proposed rotating each lane's codeword address, by analogy with the
   latent-tile rotation in `cuda/glm_vq_flash.cuh:60-64`. That analogy does not
   hold, and a check settles it: rotating through a fixed permutation of the book is
   a bijection, so the collision *count* over random indices is unchanged (a direct
   simulation of 32 random 8-byte codeword reads gives 2 conflicts before and 2
   after). Stride rotation works because it breaks a deterministic pattern; a
   random-index gather has no pattern to break. FLUTE's real mechanism is *bank
   duplication*, which needs the space our 64 KiB book does not have.
4. **The remaining measurable question** is the ~12 to 18 ms between the corrected
   floor and the recorded verify time. The three candidate causes are DRAM latency
   at low occupancy, fixed non-MoE cost, and per-pair lookup issue, and the `ncu`
   view named above separates them. That is the first decode step, and it is
   deliberately placed behind the prefill items in section 9.

**A second lever that needs no kernel.** The round is 59.1 ms and 2.7 tokens per
round means roughly 1 + 1.7 accepted drafts or better scheduling of them. Draft
acceptance is the multiplier on every byte read, so it is a tokens-per-second lever
with no numerical cost at all, and the repo measures it as larger than several
kernel rounds (`docs/deepseek41-optimization-2026-11-12.md:174-186`; section 4's
acceptance trap). It is not an FP8 lever and is noted here only so the kernel work
is not measured without it.

**What to measure, concretely.** Two numbers decide the decode campaign, and the
first is already known: the ratio of recorded verify time to the width-corrected
DRAM floor, which is about 77 percent at four rows. The second is the stall
breakdown from an `ncu` GPU Speed Of Light capture on the expert kernel at the
production verify width, read as above. If `long_scoreboard` dominates, the lever is
occupancy and in-flight bytes, not instruction selection; if `mio_throttle`
dominates, the lever is fewer lookup wavefronts per round, which is FLUTE's table
duplication and needs space this artifact does not have; if `barrier` dominates, the
lever is the producer ring. Chapter and verse for the last of those already exists in
`cuda/minimax_mma8w.cuh:38-56` for prefill.

### 9.6 How hVQ's gain composes with an instruction change

**The gain is a post-reduction scalar.** hVQ fits one F16 multiplier per weight
row, and the kernel applies it to the finished accumulator rather than to any
weight:
`glmvq_round(glmvq_mul(acc[m][j][side * 2 + e], gain[m][side]))`
(`cuda/glm_vq_mma8.cuh:383`, again at `:445` for the second mode). In the
warp-specialized kernels the same thing happens one step earlier, as a float
multiply of the per-stage partial sum (`cuda/beam_mma8w.cuh:301,544`;
`cuda/minimax_mma8w.cuh:314,377-380`). `glmvq_mul` is `mul.rn.f32`, deliberately,
so fast-math cannot flush a legal scale (`cuda/glm_vq.cuh:56-61`), and
`glmvq_round` is a round to BF16 (`cuda/glm_vq_ops.cuh:18-20`).

**The instruction is pre-reduction, per block.** From section 2,
`(A * scale_A) * (B * scale_B)` applies exactly one `ue8m0` factor per 32 elements
of K before the product, and the accumulator sees `C` unchanged.

**The composition, stated precisely.** The two factors act on different sides of
the reduction, so ordering them is exact and not merely close: a fitted gain `g`
applied after the dot gives `bf16(g * dot)`, and a power-of-two activation scale
`s` applied before it gives `bf16(g * (dot/s) * s)`. For a power of two the two
paths are bit-identical, because the exactness of a `2^k` multiply in binary32 is
what makes `dot/s * s == dot` exactly ([2606.06521](https://arxiv.org/abs/2606.06521)
[page 5], proving the same identity for the P-cast scale). The one thing that does
not carry is the accumulator's own reduction order: folding the scale into the
instruction can change what the tensor core computes, so `dot` may be a different
real number. That changes the realised error, not the fitted `g`, which is why
point 2 below is about measuring rather than refitting.

Two things must not be confused here. The commuting argument says the gain and a
power-of-two activation scale may be applied in either order; it does **not** say
the change is bit-identical end to end. The current pipeline applies `amax / 448`
per (row, 128) in F32 after each stage, and an E8M0 scale per 32 in the
instruction is a different decomposition of the same product: finer groups, and a
power-of-two instead of a full-precision factor. The numbers will move, and that
movement is exactly what section 2's gate measures.

Three consequences follow, and they are the reason this combination is worth
writing down rather than discovering later.

1. **hVQ's gain cannot move into `scale_A`.** It is a fitted real per row
   (`g = w H_t c^T / c H_t c^T`, `tools/vq_calib/ds41_hvq.py:41`), not a power of
   two, so it is not representable as `ue8m0`. It must stay in the epilogue. What
   the instruction can absorb is only the *activation* side. A design that tries
   to fold the refit gain into the MMA's scale operand is arithmetically wrong,
   not merely lossy.

2. **The gain transfers across a kernel change, and that is worth knowing.**
   `hvq_refit_kernel` computes `g = (sum_c h_c w_c c_c + noff) / (sum_c h_c c_c^2 + doff)`
   with `h = diag(H)`, `noff = w H_off c^T` and `doff = c H_off c^T`
   (`tools/vq_calib/cuda/ds41_hvq.cu:171-180`; the fit is the least-squares row
   gain under H for fixed indices, `tools/vq_calib/ds41_hvq.py:41`). Those inputs
   are the source weight, the chosen codeword and the metric — none of them is a
   property of the kernel. So the gain is kernel-independent and does **not** need
   refitting when only the reduction order or the stage boundary changes. What
   changes is the realised error, which is measured, not refitted. This is the
   opposite of the usual intuition about fitted parameters, and it is what makes
   the combination cheap: one artifact, two kernels.

3. **The two gains are additive in the quality budget, not multiplicative.** The
   FP8 expert arm already costs something: rounds 1–3 moved logits by KL 0.0199
   against the 0.0154 chunk-size floor, with 96.99 percent top-1
   (`docs/deepseek41-optimization-2026-11-12.md`). hVQ recovered KL 0.050 paired
   against iVQ on generated positions, 95 percent interval -0.062 to -0.038, plus
   2.1 points of top-1 (`docs/deepseek41-hvq-evaluation-2026-11-11.md`). The
   hVQ recovery is banked in the artifact, and the E8M0 activation rounding is
   spent on top of it as a new rounding of the *other* operand. The combined gate
   is therefore not "does hVQ still help" but "is the total distance from the
   original still inside the band", measured once against the original rather
   than pairwise against either arm.

**What actually has to change in the payload.** Nothing, for the first version.
The DQVL v3 row is a header, one F16 gain per row, the index block and (on
13-bit books) the plane bits (`cuda/ds41_vq_prefill_fp8.cuh:32-35`), and the
item record carries `gain` as a separate pointer
(`cuda/ds41_vq_prefill_fp8.cuh:192`). An `ue8m0` activation scale is a property of
the *activation*, produced per forward pass, so it enters the kernel as a new
staged array beside the existing F32 activation scale
(`cuda/ds41_vq_prefill_fp8.cuh:14`, `cuda/glm_vq_mma8.cuh:11`) and leaves the
artifact untouched. That is the cheapest possible composition: no
re-quantisation of any expert, no new blob version, and `DS4_DS41_VQ_FP8=0` still
restores the BF16 arm.

Worth stating because it is a choice, not a necessity: hVQ could instead have been
made the trigger for a *new* artifact whose codebook is re-fit so that the
8-column blocks align with 32-element scale groups. That would gain nothing
numerically, because the hVQ fit is against the source weight and the fixed
codebook, and it would cost a full re-conversion of forty layers at roughly 200
seconds each (`docs/deepseek41-hvq-evaluation-2026-11-11.md`, per-layer conversion
times). The activation-scale change belongs in the kernel.

**Where a real conflict exists.** The R13+8 layers, which carry a second
residual codeword whose two MMAs accumulate into one register
(`cuda/minimax_mma8w.cuh:26-27`, "W = gain * (C1 + C2)"). ds41's arm passes a
null residual slot (`cuda/ds41_vq_prefill_fp8.cuh:192`), so it is unaffected
today. But if a residual stage is ever added to a block-scaled kernel, the two
codewords must share one `scale_A`: they are two halves of one weight, and
scaling them differently would scale the second half by the wrong power of two.
The rule to write down now is that the residual codeword inherits the layer
codeword's scale, never its own.

**The measurement that settles it.** One arm, three points, on a fixed prompt set:
the artifact as shipped with the BF16 expert arm, the same artifact with the FP8
arm, and the same artifact with the FP8 arm plus E8M0 per-32 activation scales.
The middle point already exists (`docs/deepseek41-optimization-2026-11-12.md`).
The third is the new one, and its gate is paired NLL and top-1 against the
original's logits over the hVQ evaluation's 26 cases and 58,777 positions, with
the chunk-size floor beside it — the protocol that record already defines, so no
new harness and no new fixture are needed.

**Expected shape of the result, stated so it can be wrong.** The honest expectation
after section 9.2 must be lowered from an earlier draft of this paragraph. Prefill
may improve a little, from instruction count and register pressure, but not from
removing the promotion: the kernel's own note records that removing per-stage
promotion entirely ("a k-outer loop with one activation scale per row and no
per-stage promotion") cost 180 registers and came out **even**
(`cuda/minimax_mma8w.cuh:52-54`). Quality should be neutral to slightly better,
because per-32 groups track the dynamic range across K more closely than per-128.
Decode should not move, since the token loop is byte-bound at 233 to 237 GB/s
(`docs/gb10-hardware.md:117`) and no decode kernel changes. The real prefill
headroom is the shared-memory pipe: 42 percent of its load wavefronts are
book-gather conflicts, the tensor pipe is at 38-44 percent while reaching 53 percent
with the copies removed (`cuda/minimax_mma8w.cuh:38-56`), which is where section
8.3's relief belongs.

## 10. What this changes in the plan

The artifact is frozen, and that fixes the ranking before any experiment runs. The
release is 113,556,639,424 bytes, 105.76 GiB of the 121.6 GiB the Spark's unified
memory provides, with 10.5 GiB left available with the server resident
(https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-hVQ-GGUF, and
`docs/deepseek41-hvq-evaluation-2026-11-11.md:16`). The bit allocation stays as it
is — 13-bit on layers 0-13, 12-bit on 14-39 — and hVQ's quality stays as it
measured. So no proposal may grow the artifact, spend quality, or change a bit
width; anything that does is out whatever its speed.

Ranked by what the measurement says, not by what the note was originally
organised around. Section 3 moves sparse attention to the top: it is 21 to 24
percent of a prefill block, it is the class that computes on tensor cores, and it
runs at 12 percent of the BF16 peak. The experts are 29.5 percent but within a
quarter of their bandwidth floor, so no arithmetic change helps them; the items
that target them are ranked below for that reason, not because they are cheap.

1. **Sparse attention, measured and then profiled.** `bin/attn-bench` in the
   toolkit times the shipped kernel with no model loaded, which is what the
   engine's own safety table allows and what the resident server leaves room for.
   It already answers the largest question: the compressed-pool path costs 1.2 to
   1.3x the F32 window path per tile at matched tile counts, so the gather and
   decode are the cost and the MMA is not (section 9.4). It also reproduces the
   record: 12.47 ms of the 13.5 ms per block-layer. The remaining split — barriers
   against shared-memory traffic against the 16-thread softmax — needs the `ncu`
   Speed Of Light capture, which the same tool supports with `--once`. This is
   first because it is the largest class, it costs one capture rather than a
   kernel, and it settles the rate: the note's earlier 0.4 was a convention error
   and the record's 13 TFLOPS is right.

2. **Attention: the serialised softmax, which is also the cheapest fix.** Three
   of every four warps idle while 16 threads run the online softmax serially
   (`cuda/ds41_attn_mma.cuh:253-261`), between full-CTA barriers that no other
   work covers. The decode kernel already has the warp-parallel form
   (`ds4_attn_mma_stats`, `:122-135`), so this is a port rather than an
   invention, and the sum keeps its `k = 0..15` order, which makes it bit-equal.
   Beyond that, FlashAttention-3's arithmetic applies: matmul throughput doubles
   at FP8 while the exponential unit's does not, so a kernel that computes QK,
   waits, computes softmax, then computes PV idles the tensor cores by
   construction ([2407.08608](https://arxiv.org/abs/2407.08608) [page 1-2, 5-6]).
   Gate on teacher-forced NLL, not cmp: this family is documented as not
   bit-equal to the scalar kernel and its gate is the quality scale.

3. **The compressed-pool decode, which is now the measured target.** The excess
   above is one scale byte per lane, 32 nibble reads per lane and two shuffles
   per element, paid on the pool's tiles; it is also where the 2.1 ms fixed cost
   of a launch sits, which no operand change touches. Vectorising that unpack and
   overlapping it with the MMAs addresses the cost the measurement actually
   found. This item did not exist in the earlier ranking, which put an
   instruction change here instead; the measurement moved it.

4. **Block-scaled FP4 QK, only after the two above and only for QK.** Its
   precondition is now explicit: the stored scale is one per 16 elements along
   `head_dim`, which is the reduction axis of QK but the output axis of PV, so
   PV cannot consume the stored pair without a re-quantisation (section 9.4). It
   also prices P and the F32 window as operands that would have to become E2M1.
   Ranked below the decode work because the decode is the measured cost and this
   is the inferred one.

5. **Book-gather bank-conflict relief, by the one route still open.** The kernel
   spends 48 percent of its shared-memory pipe on load wavefronts of which 42
   percent are book-gather conflicts, and reaches 53 percent of the E4M3 peak once
   the copies are removed (`cuda/minimax_mma8w.cuh:38-56`). Two of the three routes
   are closed and were checked: FLUTE's bank duplication needs shared memory the
   64 KiB book has already spent, and a fixed lane rotation is a bijection that
   leaves a random gather's collision count unchanged (section 9.3, with the
   simulation). What remains is the read *width*: splitting each 8-byte codeword
   read into rotated 4-byte halves, which was 6 percent bit-identical in the R13
   expert gate. Probe that alone; it is cheap and it is the only version of this
   item with evidence behind it.
6. **Producer-copy relief on the same pipe.** The gap between 53 percent (copies
   removed) and 38-44 percent (copies present) is the producers' work competing for
   the same shared-memory bandwidth. The header bounds the prize: with every expert
   resident in L2 the gate runs 6.6 ms against 7.1 at 4,096 rows
   (`cuda/minimax_mma8w.cuh:44-48`). This is the larger of the two and the harder,
   because it is a pipeline-structure change rather than a bank fix.
7. **Block-scaled E4M3 rate on the existing peak harness** (`tools/gb10_peaks.cu`,
   beside its `E4M3_F32` probe). One run on the GB10; it gates the instruction arm
   below. Already established here: the form assembles for `sm_120a` and `sm_121a`
   and emits QMMA.SF, and its rate is now measured (Phase 1 of the handoff: 100.0
   percent of the plain form). It sits here rather than near the top because
   section 3 bounds its gain: the promotion it removes is measured as even.
8. **ds41 FP8 experts with E8M0 per-32 activation scales on the hVQ artifact**, if
   the rate holds. Held to instruction count and register pressure only, since the
   FMA removal is refuted (section 3, the Gemma note and the rejected-variant
   record). It is **byte-neutral by construction**: one
   E8M0 byte per 32 elements of K is 1/32 bytes per element, and the F32 scale per
   128 it replaces is 4/128, the same number. The artifact does not change either,
   since the gain is kernel-independent and the codebook untouched.

        current: 4 bytes / 128 elements = 0.031250 bytes per element
        E8M0:    1 byte  /  32 elements = 0.031250 bytes per element

   That equality is the reason the change is allowed at all. A finer scale (per-16,
   or a full-precision factor per 32) would grow the activation stage and is
   refused by the memory constraint; a coarser one (per-64) would save space and
   spend quality and is refused by the quality constraint. The block-scaled
   instruction's native format is exactly the size-neutral point between them.
9. **Grouped-expert tail tile.** No numerical change and no memory change; the
   parity test and real-route replay already exist. Backed by our own 29-percent
   small-item measurement and by the two padding papers. It shares the kernel with
   items 1 and 2, so it is measured in the same runs.
10. **P-cast with reverse KV iteration and S = 256**, as a precondition of the
   already-listed E4M3 PV lever rather than as a separate experiment.
11. **cuBLASLt algorithm enumeration at the prefill shapes**, bits held fixed. Not
   from FP8 literature, but it touches the largest named class in a ds41 prefill
   block, costs nothing numerically, and allocates nothing beyond the existing
   32 MiB workspace.
12. **Decode, last and only after a profile.** Section 8.5 corrects the earlier
   roofline and finds verify already at about 77 percent of its width-corrected DRAM
   floor, so the remaining decode headroom is roughly 12 to 18 ms of a 51 ms pass,
   not the large figure an earlier draft implied. The first step is an `ncu` stall
   breakdown, not a code change, and if the ceiling looks low after it, prefill is
   the better target — which is why this is the last item.

**What the constraints rule out, stated so it is not proposed again.** The artifact
is fixed: 13-bit indices on layers 0-13, 12-bit on 14-39, hVQ's quality, 113.56 GB.
Nothing in this note changes a bit width, and nothing proposes a re-conversion.
Re-fitting the codebooks at load, as AQLM does, would need a working set
proportional to the model and cannot fit beside a resident 105.76 GiB artifact.
Bit width is the quality dial and it stays where it is; runtime speed is the only
axis this artifact has left, which is why every item above is a kernel change that
leaves the file byte-identical.

Explicitly not worth pursuing on this evidence: adding a scale granularity finer
than per-32 along K (MOSS measures the CUDA-core price, and the byte arithmetic
above refuses it), casting P to E4M3 on a forward walk (section 4), rotating a
random-index book gather through a fixed permutation (section 9.3, checked and
refuted), re-converting the artifact to align 8-column codebook blocks with
32-element scale groups (section 9, "what actually has to change in the payload"),
and any format change whose only argument is a smaller cache footprint, since our own
measurements show that case is decided by the dequantisation cost and not by the
bytes.

## 11. Verification status

Measured in this session, with the snippet and commands in section 2: the
block-scaled FP8 form assembles for `sm_120a` and `sm_121a` and emits
`QMMA.SF`; the same source fails for `sm_89`; the selector tuple must be braced;
the byte id accepts a register; PTX ISA 9.5 Table 44-45 and section 9.7.16.3.

Read first-hand and quoted above: the abstracts and method pages of the nineteen
works in the table, each cited by page. Two of them are cited only for their
policy rather than their numbers: FireQ's post-RoPE key quantisation and FP8-RL's
per-step KV recalibration, both in section 5.

Read from this repository for section 9, not measured: where the hVQ gain is
applied (`cuda/glm_vq_mma8.cuh:383,445`, `cuda/beam_mma8w.cuh:301,544`,
`cuda/minimax_mma8w.cuh:314,377-380`), what `glmvq_mul` and `glmvq_round` do
(`cuda/glm_vq.cuh:56-61`, `cuda/glm_vq_ops.cuh:18-20`), the refit formula
(`tools/vq_calib/cuda/ds41_hvq.cu:171-180`), the payload geometry
(`cuda/ds41_vq_prefill_fp8.cuh:32-35,192`), the index decode
(`cuda/glm_vq_mma8.cuh:86-102`), and the kernel's own measured profile and rejected
variants (`cuda/minimax_mma8w.cuh:38-56`). Two claims in this note turn on reading
our own code rather than on a paper, and both deserve a cross-check before anyone
relies on them: that the hVQ gain is kernel-independent, and that the per-stage
promotion is expensive enough to matter. The second is the weaker of the two,
because the kernel header records the opposite ("even"), and section 9.2 and the
plan are written to that reading.

Measured since the first draft, and now part of the record: the block-scaled form's
rate on GB10 relative to the plain form (99.3 percent, section 2). Still not
measured: any numerical effect of the proposals; whether the combined arm's distance
from the original stays inside the band; the tail-item share on our own current
routes (the 29 percent figure is MiniMax's recorded measurement, not a fresh one);
and the FP8 KV capacity claims, which come from engines whose dequantisation is
free in the instruction.

Read from the release record rather than measured for section 9: the artifact size
113,556,639,424 bytes and the 10.5 GiB headroom with the server resident, both from
the model card at
https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-hVQ-GGUF; the per-layer
expert parameter count from its 384 experts of `w1`/`w3` [2304, 5120] and `w2`
[5120, 2304] (`tools/vq_calib/convert_layer_ds41.py:44`), used only to check that
the size arithmetic in section 9 holds.

Reviewed by a second model, and it found a real error. The decode roofline in
section 9.5 originally applied the single-row expert traffic to a 1 + D verify pass;
the reviewer pointed out that independent routing makes a four-row pass touch about
23 distinct experts, so the pass reads roughly four times the bytes and the pass is
nearer its floor than the draft claimed. That correction is applied above and it is
the reason decode now ranks last. The reviewer also argued that a fixed permutation
of a random-index book cannot reduce gather conflicts; the simulation in section 9.3
confirms it, and the proposal is withdrawn. Both changes are the reviewer's, not
mine, and both went in only after the arithmetic was checked here.

One thing this note is not: it is not about the hVQ converter. An earlier answer
treated that conversion path as a speed target and it is not one, because hVQ
exists precisely so the model fits the Spark at a quality it measured, and the
converter's cost is paid once at build time. A later draft then floated dropping
the 13-bit layers to 12-bit; that is withdrawn too. The bit allocation and the
quality are fixed by the release, so the only proposals that survive here are
kernel changes that leave the file byte-identical.

Not measured, and named so it is not mistaken for measured: the 4.33 GB per-round
decode budget in section 9.5 is arithmetic from the shape table and the artifact's
bit rate, not a trace; the 31-percent share of 235 GB/s follows from that budget and
the recorded 59.1 ms round, and an `nsys` export is what would confirm it. The
grouped decode kernel's own numbers (0.356 ms/layer at n=1, 1.283 at n=4, 94.0
against 87.5 ms for mixed dispatch) are the port's recorded measurements
(`cuda/ds41_vq_group.cuh:5-15,127-133`), not fresh ones.

Scope, restated once because it is easy to drift from: this note is about FP8
tensor-core *speed*. Every proposal in it is a kernel change measured against the
existing artifact, gated by the KL and top-1 protocol the hVQ record already
defines. Nothing in it changes a bit width, re-fits a codebook, or trades quality
for bytes.
