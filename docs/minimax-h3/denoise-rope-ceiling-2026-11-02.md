# The denoise step's rope is already at its ceiling, and the 1,890 launches were the decoder's (2026-11-02)

Unit: handoff section 3, step 0 and the bounded reproducer. The handoff entered
this unit with two arithmetic-neutral levers named from
`step-profile-2026-11-01.md`: `h3d_qkv_rope_kernel` at 112.7 ms a call, 51 calls
a step, and `h3d_attention_mma_f32_kernel` launched 1,890 times inside a denoise
step. Re-measured on `bdf515ec` with the delivery's own arms, **neither premise
survives**: the rope is 17.3 ms a call and 0.883 s of a 38.36 s step, and the
f32 attention launches are the video VAE decoder's — the denoise step makes
none of them.

**The verdict first.**

| figure | handoff | measured on `bdf515ec` |
|---|---:|---:|
| `h3d_qkv_rope_kernel` per call | 112.7 ms | **17.32 ms** (bench 17.676) |
| launches a step | 51 | 51 (`102` instances over two steps) |
| its share of the step | 6.6 s (17%) | **0.883 s (2.3%)** |
| its distance from its own traffic floor | unstated | **2.75 ms a call** (17.676 against 14.926) |
| `h3d_attention_mma_f32_kernel` in a denoise step | 1,890 | **0** |

So the rope is 6.5x cheaper than the record said and the second lever does not
exist. Both are worth closing rather than spending an occupancy experiment on:
the whole class the handoff wanted to attack is 0.9 s of a 38.4 s step, and its
ceiling is 0.14 s.

## 1. Step 0: the same render, both limbs, one commit

Two renders of one job — `1344x768`, `--frames 121`, `--steps 2`, seed 42,
`H3_DIT_GEMM=fp8 --vae-gemm bf16i --vae-attention tensor`, delivery checkpoint
`MiniMax-H3-Turbo-v4` — one plain for the stopwatch and one under
`nsys profile --trace=cuda`, kernel table from
`nsys stats -r cuda_gpu_kern_sum`:

| run | `TEXT` | `LOAD` | `DENOISE` | s/step | `FRAMES` | wall | clip md5 |
|---|---:|---:|---:|---:|---:|---:|---|
| plain | 28.305 | 28.620 | 76.729 | **38.36** | 47.691 | 188 | `5f03feac1c394b78…` |
| profiled | — | — | 77.559 | 38.78 | 52.010 | — | `5f03feac1c394b78…` |

The two clips are byte-identical, and the profiler's 1.1% on the step is not the
26% `step-profile-2026-11-01.md` reports. The denoise figure agrees with
`fp8-delivery-2026-11-02.md` (38.93 s/step) to 1.5%.

The step, per class, from the same table (instances over two steps, then the
per-step figure):

| class | instances | total ms | per call | per step |
|---|---:|---:|---:|---:|
| `h3d_attention_mma_pipe_kernel<8,1>` | 102 | 52,480.4 | **514.5 ms** | **26.24 s** |
| `nvjet_sm121_qqtst_mma_128x256x128` | 300 | 13,730.4 | 45.8 ms | 6.87 s |
| `h3d_qkv_rope_kernel` | 102 | 1,766.7 | 17.3 ms | 0.88 s |
| `h3d_swiglu_kernel` (BF16) | 102 | 1,588.4 | 15.6 ms | 0.79 s |
| `nvjet_sm121_qqtst_mma_192x176x128` | 100 | 1,442.8 | 14.4 ms | 0.72 s |
| `h3d_fp8_amax_kernel` | 600 | 2,550.4 | 4.25 ms | 1.28 s |
| `h3d_fp8_quantize_kernel` | 600 | 2,243.4 | 3.74 ms | 1.12 s |
| `h3d_gate_kernel` | 200 | 1,218.4 | 6.09 ms | 0.61 s |
| `h3d_adaln_kernel` | 204 | 873.4 | 4.28 ms | 0.44 s |

Attention is **68.4%** of the step, matching the 64% the width record measured.
Everything else in the table sums to **12.7 s** of the step, of which 6.9 s is
the one big GEMM (`128x256`, 3 launches a layer); the rest of the render's GPU
time (`h3d_attention_mma_f32_kernel` 3,780, `h3d_uncast_kernel` 15,120,
`h3d_swiglu_f32_kernel` 3,780, `cutlass` 3,780, `nvjet` 128x208 7,560) belongs
to the decoder, one launch per tile-block. `h3d_rms_norm_f32_kernel` (7,560) is
the one row that cannot be attributed from a whole-render table because both
limbs call it, so it is left out of the per-step column.

## 2. The rope, against its own floor

`tests/h3/dit_rope_bm` (new, `make CUDA_ARCH=sm_121 tests/h3/dit_rope_bm`) runs
the model's own entry point, the reference spelling it replaced, two
rows-per-block shapes that keep the sum's order, and a floor that does the same
six streams and no arithmetic. Every variant is hashed on Q/K/V:

| variant | S = 37,722 | GB/s | S = 18,432 | GB/s | `fnv` |
|---|---:|---:|---:|---:|---|
| shipped (staged) | **17.676 ms** | 183.6 | **8.607 ms** | 184.2 | `727cdc5793b11282` |
| reference (re-read) | 40.627 | 79.9 | 20.094 | 78.9 | same |
| rows4 | 18.238 | 177.9 | 8.891 | 178.3 | same |
| rows8 | 18.406 | 176.3 | 9.076 | 174.7 | same |
| floor (no math) | 14.926 | 217.4 | 7.292 | 217.4 | differs by construction |

Three things follow.

- **The staged arm is bit-identical to the reference** at both shapes, which is
  the kernel's own comment confirmed by an independent hash rather than by
  inspection. That is also why no arithmetic moved to get here.
- **The kernel is at 84% of its traffic floor and 67% of the device's 273 GB/s.**
  The remaining 2.75 ms a call is the serial RMS sum and the rotation, and it is
  the only headroom there is: 51 calls a step caps the whole lever at 0.14 s.
- **Wider blocks do not help.** 4 and 8 rows per block amortize the two barriers
  over more work and land 3-5% *behind* the shipped shape at both shapes, so the
  barrier pair was never the cost. The floor is flat at 217.4 GB/s at both
  shapes, so the limit is the six streams themselves, not the launch geometry.

## 3. The 1,890 launches are the decoder's, and the count says so

`h3d_sdpa_f32` (`native/h3/h3_dit_gpu.cu:4568`) routes to
`h3d_attention_mma_f32_kernel` only `if (head_dim == 64)` (`:4583`), and 64 is
the *video VAE decoder's* head dimension
(`crates/ds4-core/src/minimax_h3/vae_video.rs:64`: `HEAD_DIM = 64`);
the DiT's is 128 and goes through `h3d_sdpa` (`dit.rs:2848`). The count agrees:
this render holds **3,780**
instances, which is exactly the decode plan's `105 tiles x 36 blocks`, and
1,890 is that number divided by the two denoise steps the render ran — the
arithmetic the old record used to produce a per-step figure from a whole-render
count. The decode record already prices the class at 6.50 s a clip (6.61 s here).

There is no batching defect to fix. One launch per tile-block is the decoder's
design, and the decode unit has already taken everything its schedule had.

## 4. Why the old figure was 6.5x this one

`step-profile-2026-11-01.md` was a real capture, so the number has a cause, and
the bench rules out the two obvious ones: the shipped spelling is 17.7 ms and the
reference spelling is 40.6 ms at the same shape, neither of them 112.7. The
kernel has not changed since `6752b7b3` (2026-10-29, "Give the DiT's elementwise
pair a staged layout"), which is *before* that profile. So the profile's binary
was not this commit's binary — the same trap the decode unit hit when a gate
binary turned out to be hours stale. Recorded as the cause it most likely is,
not as a fact: the trace cannot name its own binary.

The guardrail this unit adds is therefore narrow and usable: **a per-call figure
from a profile is only as old as the binary that produced it.** Priced at the
bench, this class is 0.9 s a step; priced from the record, it was 6.6 s, and six
of those seven seconds were spent before the unit started.

## 5. What is left in the denoise, and it is all numerics

With the rope closed at its floor and the width record's arms closed by
registers, the arithmetic-neutral budget of the denoise step is spent:

- **Attention, 26.24 s of 38.36 s.** 514.5 ms a call, 51 calls. The width is at
  its optimum (`attention-widths-2026-11-02.md`); the lever left is E4M3 K/V on
  the block-scale MMA, which `fp8-attention-2026-11-02.md` already measured at
  329.8 ms a call with a 1.51e-4 deviation — a numerics decision with a quality
  gate, not a scheduling one.
- **The FP8 arm's own preparation, 2.40 s a step.** 600 `amax` and 600
  `quantize` launches over two steps, 4.25 and 3.74 ms each. Fusing the two
  passes is arithmetic-adjacent: it changes no value only if the fused pass keeps
  `amax`'s exact order.
- **The three decode numerics levers** parked in `vae-decode-levers-2026-11-02.md`
  section 6 (two cast passes 10.35 s, the F32 elementwise chain 7.83 s, BF16 P/V
  attention 6.50 s) stay parked for the same reason.

So the next unit is not a scheduling unit. It is either the E4M3 attention arm
with its quality gate, or the queued two-stage delivery phase, which does not
depend on this one's outcome.

## 6. Reproducing

```
git log --oneline -1                     # bdf515ec
export H3_DIT_GEMM=fp8 H3_VAE_GEMM=bf16i H3_VAE_ATTENTION=tensor

# the two renders of section 1 (this one under nsys)
./target/release/h3-rs -d /mnt/h3-artifacts/MiniMax-H3-Turbo-v4 \
    -p "$(cat ~/ds4-test/denoise-unit/prompt.txt)" \
    --width 1344 --height 768 --frames 121 --steps 2 --seed 42 \
    --crf 15 --preset slow -o ~/ds4-test/denoise-unit/plain.mp4
nsys profile --trace=cuda -o /tmp/step0 ./target/release/h3-rs ...same job...
nsys stats -r cuda_gpu_kern_sum --format csv /tmp/step0.nsys-rep \
    > ~/ds4-test/denoise-unit/kern.csv
python3 ~/ds4-test/kern-table.py ~/ds4-test/denoise-unit/kern.csv

# the rope, both shapes
make CUDA_ARCH=sm_121 tests/h3/dit_rope_bm
tests/h3/dit_rope_bm 37722 56 5
tests/h3/dit_rope_bm 18432 56 5
```

Instruments on the Spark: `~/ds4-test/denoise-step0.sh` (the two renders),
`~/ds4-test/kern-table.py` (the CSV table), `~/ds4-test/denoise-unit/` (logs,
prompt, both clips, `kern.csv`, `step0.nsys-rep`), `~/ds4-test/dit-rope-bm.log`
and `~/ds4-test/dit-rope-bm-18432.log`.
