# The DiT's elementwise pair: a staged layout, bit for bit

One evaluation of the fast arm at job A's shape (768x768, 107 frames,
`H3_DIT_SOL=0.9`) spends 2,044 ms of its 12,346 ms of kernels in two
elementwise kernels that between them move 1,012 MB: `h3d_adaln_kernel`
(1,013 ms, 102 calls of 9.9) and `h3d_qkv_rope_kernel` (1,031 ms, 50 calls of
20.6), both at about a sixth of their bandwidth floors
(`denoise-attribution-2026-10-28.md`). Both were verbatim ports of the
reference's own spelling (`h3_gpu_cuda.cu:2656-2687`, `:3079-3136`), so the
reference is slow here too.

This record is the attribution, the staged kernels that replace the two, and
the proof that they compute the reference's arithmetic bit for bit: the
50-block forward hash and the render's MP4 sha256 are the ones the previous
spelling produced.

## 1. Why the two kernels were slow

Neither kernel is arithmetic-bound; each is a reduction that reads far more
than it needs, and the rest of the block waits for it.

- **AdaLN**, one block per row of 5,376 elements: thread 0 owns the RMS sum,
  so one lane issues 5,376 dependent FMAs fed by 2-byte loads while the other
  255 threads of the block sit at the barrier. The block's critical path is
  one lane's memory latency, and the bytes it wants arrive 2 at a time.
  Measured at the shape: 9.743 ms per call on the Spark, 6.09 on the
  workstation, against 1.6-1.9 ms for the same kernel with the sum removed and
  the elementwise pass left alone. The sum is 80% of the call.
- **QKV/RoPE**, one block per (row, head) of 128 dimensions: every one of the
  block's 128 threads walks the whole 128-element Q and the 128-element K
  vector for the two sums, because every thread needs both scalars. The block
  therefore issues 512 broadcasts of the same 512 bytes, and a broadcast load
  moves 4 useful bytes per instruction. Measured: 20.224 ms per call on the
  Spark against 8.8 ms with the sums replaced by constants.

Neither is the reference's own trade to make: it ships the same two kernels
unchanged, so a port that stages them beats the reference stage it is
measured against without waiting for an artifact.

## 2. The design

Both kernels keep the reference's own accumulation order - the sum is a serial
chain in index order and its rounding is part of the output - and move only
where the operands come from:

- **AdaLN** copies the row into shared memory with the block's 256 threads in
  16-byte units (fully coalesced, 2.6 copies per thread), and thread 0 walks
  the copy in the same index order. The elementwise pass then reads the copy
  as well, so the row is read from DRAM once instead of twice.
- **QKV/RoPE** stages the head's Q and K in shared memory the same way (one
  copy per thread for a 128-dimension head), writes V during the staging -
  it is a copy of `qkv`'s third block, untouched by the rotation - and lets
  one lane walk both staged vectors in order. The epilogue reads the copy.

Neither is a numerical change: the staged bytes are the same bits the kernel
read from global memory, and the chain consumes them in the same sequence, so
every intermediate rounding is the reference's.

`H3_DIT_ELEMENTWISE=reference` restores the previous spelling for an A/B in
one binary. A row wider than the 8,192-element staging, or a head dimension
above 256, falls back to it automatically. Both switches are read once per
process.

## 3. What it is worth (Spark, busy 2398-2405 MHz)

`tests/h3/elementwise_bm` is the reproducer: the shipped kernels, verbatim
copies of the previous ones, a reduction-free control and the candidate
designs, each timed at the denoise shape (18,799 rows, width 5,376, 56 heads
of 128, rope 48) and hashed. Ten calls per arm, one process.

| AdaLN arm | ms/call | GB/s |
|---|---:|---:|
| reference spelling (`H3_DIT_ELEMENTWISE=reference`) | 9.743 | 59.4 |
| staged (shipped) | 2.029 | 285.0 |
| staged, repeated at the end of the run | 2.041 | 283.3 |
| elementwise pass alone (no sum) | 1.895 | 305.1 |
| block reduction of the same sum (order change) | 1.887 | 306.4 |
| 16-element batched loads, no staging | 4.968 | 116.4 |
| 64-element batched loads, no staging | 3.249 | 178.0 |

| QKV/RoPE arm | ms/call | GB/s |
|---|---:|---:|
| reference spelling | 20.224 | 76.3 |
| staged (shipped) | 8.793 | 175.4 |
| elementwise pass alone (no sum) | 8.824 | 174.8 |
| block reduction of the same sum (order change) | 8.639 | 178.5 |

Bytes counted: 578 MB for AdaLN (input, output and both modulation rows it
reads; 404 MB of that is DRAM, the modulation rows are L2-resident) and
1,542 MB for QKV/RoPE (the 3 x rows x heads x head_dim triple read and
written).

Three readings:

1. **The sum is now free.** AdaLN's staged call is 0.18 ms above its
   reduction-free control, QKV's is 0.03 ms *below* its own (the staged
   block's access pattern is better than the control's). Both kernels are
   now bound by the elementwise pass, which is the point.
2. **Batching loads without staging does not close it.** The same chain with
   16 and with 64 elements in flight per iteration was measured first:
   9.743 -> 4.968 -> 3.249 ms on the Spark, and on the workstation 6.09 ->
   6.13 -> 3.14. Widening one lane's in-flight window helps, but the operands
   still come from DRAM under the chain; staging the row is what reaches
   2.03 ms.
3. **The rounding is not worth trading.** Replacing the serial chain with a
   parallel block reduction is the cheapest possible rewrite and it buys
   nothing here: QKV's 8.639 ms is 1.8% under the staged kernel's 8.793, and
   AdaLN's 1.887 is 0.14 ms under the staged 2.029 - the exact chain costs a
   seventh of a millisecond. For that, both would change their outputs:
   200,544 of AdaLN's 101,063,424 elements (0.198%, worst 33,935 bf16 steps)
   and 2,765 of QKV's 134,751,232 (0.002%). At equal speed the bit-exact
   design wins.

Per evaluation at job A's shape the two kernels cost 1,014 + 1,037 ms before
and 219 + 440 ms after - 102 and 50 calls - over 50 blocks.

## 4. Correctness

Two independent checks, both required, both green:

- **The reproducer**: every arm's output is hashed and must equal the
  reference copy's. AdaLN: 101,063,424 elements, `f02c856ca699b4e2` in all
  four bit-exact arms. QKV/RoPE: 134,751,232 elements over Q ^ K,
  `fc495989e8ac7218` in all three. The reduction arms are the only ones that
  differ, by design.
- **The model**: `tests/h3_dit_gate` at 768x768, 107 frames, `H3_DIT_SOL=0.9`,
  one binary, both modes in sequence. The forward after one evaluation prints
  `FORWARD video=b0150d3a0eebf5a8 audio=2643467e773bfc9b
  video_edge=2.421875000/-2.906250000 audio_edge=1.203125000/0.480468750` in
  both modes, and after the whole sampler
  (`--denoise --steps 2`) `video=81d36bee9bca3b16 audio=a6ba900150f7aaa1` and
  (`--denoise --steps 4`) `video=28f2945b59591454 audio=274a33b5345b6155`,
  again in both modes with identical edges. Those hashes cover all 50 blocks
  and every step of the sampler, and they were also identical in the four
  nsys captures.
- **The render**: the fast arm's job A render (NVFP4 prompt encoder, four
  evaluations, seed 42) writes
  `5b5e735c1ebcbf2e99abd3a69f210affd0844406c473bdedca707cabeb224876`,
  the same MP4 the steady state recorded before the change
  (`steady-state-2026-10-28.md`), so the video VAE decode, the audio VAE and
  the mux are untouched.

## 5. Speed end to end

### In the model: the per-kernel table

Two captures per mode (`--denoise`, two and four evaluations, differenced and
halved), same host, same session, one binary, at job A's shape. The reference
column is the record's own table reproduced (its cutlass figure is 7% lower,
7,102 against 7,606, most likely a different cuBLASLt pick for the
projections; every port kernel matches within 2%).

| kernel (calls/eval) | reference ms/eval | staged ms/eval |
|---|---:|---:|
| `cutlass_80_tensorop_bf16_s16816gemm` (150) | 7,606.1 | 7,601.2 |
| `h3d_sol_forward_kernel` (50) | 1,326.8 | 1,303.5 |
| `h3d_qkv_rope_kernel` (50) | 1,036.9 | 439.7 |
| `h3d_adaln_kernel` (102) | 1,013.8 | 219.3 |
| `nvjet_sm121_tst_mma_128x240x64` (50) | 808.5 | 845.2 |
| `h3d_swiglu_kernel` (50) | 383.9 | 383.0 |
| `h3d_gate_kernel` (100) | 298.8 | 301.7 |
| `h3d_sol_vsa_kernel` (50) | 214.8 | 215.2 |
| `h3d_sol_pool_kernel` (50) | 127.4 | 126.5 |
| `h3d_sol_threshold_kernel` (50) | 65.6 | 68.5 |
| **total** | **12,892.3** | **11,510.3** |

1,382 ms of a 12,892 ms evaluation. Per call the two kernels now cost 2.15 and
8.79 ms against 9.94 and 20.74, and the QKV figure is the reproducer's own to
the third digit.

### The walls

| measure | before | after |
|---|---:|---:|
| Spark, AdaLN per call | 9.743 ms | 2.029 ms |
| Spark, QKV/RoPE per call | 20.224 ms | 8.793 ms |
| gate, `--denoise --steps 2`, `DENOISE` | 25.134 s | 22.339 s |
| gate, `--denoise --steps 4`, `DENOISE` | 50.982 s | 45.396 s |
| gate, one evaluation (`--steps 4`, no sampler) | 12.769 s | 11.177 s |
| render, `DENOISE steps=4` | 49.906 / 50.550 s | 45.466 s |
| render, MP4 sha256 | `5b5e735c...` | `5b5e735c...` |

The sampler's four-evaluation run falls by 5.59 s and its two-evaluation run
by 2.80 s, which is 1.40 s per evaluation - the kernel table's 1.382 s. The
render's `DENOISE` falls by 4.4-5.1 s against the steady state's two recorded
renders, and its MP4 is byte-identical. The workstation (RTX 4070 SUPER) shows
the same shape: AdaLN 6.09 -> 1.28 ms, QKV/RoPE 16.55 -> 3.96 ms per call.

## 6. What is left

- **QKV/RoPE is still 30% off its floor**: 8.79 ms against 6.2 ms for the
  1,542 MB it moves at this host's ~250 GB/s. That is the epilogue's store
  pattern and the V copy, not the sum.
- **AdaLN is at 80% of its floor**: 2.03 ms against 1.6 ms of DRAM traffic
  plus the modulation rows.
- **The bigger lever is the projection before QKV/RoPE**: `qkv` is written by
  the qkv projection (R x 5376 x 21504) and read back here; fusing the two
  would remove 808 MB of write and 808 MB of read per call. That needs the
  projection to be ours, not cuBLASLt's, and it belongs with the projection
  work, not here.

## 7. Method

`tests/h3/elementwise_bm` is the vehicle: it includes the kernel translation
unit, so it links the shipped entry points and its own copies of them at once,
and it prints a hash per arm. The nsys pair (two and four evaluations,
differenced and halved) is the same method as the attribution record's; the
captures are kept on the Spark as `~/lev1-{staged,reference}-{2,4}.nsys-rep`
with their sqlite exports.

Two traps this unit hit, worth writing down:

- **The gate runs one evaluation unless `--denoise` is passed.** `--steps N`
  only sets the schedule's sigmas; the sampler's N evaluations come from
  `--denoise`. A "pair" captured without it is the same single evaluation
  twice, and the differencing then returns noise. The check that catches it is
  the port's own kernel counts: one evaluation is 102 `h3d_adaln_kernel` calls
  and 52 `h3d_qkv_rope_kernel` calls, so a capture that does not show a
  multiple of those is not the capture the table needs.
- **A single-lane chain is not fixed by wider loads.** The first two
  candidates here (16- and 64-element batches) were implemented and measured
  before the staged design, and neither reaches it; the row has to leave
  global memory for the chain to stop waiting on it.
