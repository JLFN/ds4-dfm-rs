# The distilled stage-2 LoRA, fused into the reader (unit 11a)

Date: 2026-10-30. Branch `feat/sol-h3`, commit `c053cea7` (this record's commit
follows it). Reference pins as unit 10: `/data/ltx2-pin/packages`
(Lightricks/LTX-2 `d1511477` = tag v1.2.0), the release runtime
`/data/Sol-h3-Spark` @ `6e4a440`, and the artifacts under
`/mnt/h3-artifacts/LTX-2.5/` on the Spark.

Units 8 to 10 denoise with the plain `dev` transformer. The release does not:
`runtime/stage2_ops/models.py:76` builds the refiner's stage with
`LoraPathStrengthAndSDOps(paths["refiner_lora"], 0.8, LTXV_LORA_COMFY_RENAMING_MAP)`
and `:77-79` passes `loras=(lora,)` to `DiffusionStage.from_checkpoint`, so every
weight the release denoises with is the dev transformer plus the distilled
LoRA's delta. A three-interval `SIGMAS` schedule *is* the distilled route, so
the port has been running that schedule on weights it was not distilled for.
This unit lands the fusion and its gate; the fused evaluation is unit 11b.

## 1. What landed

- `crates/ds4-core/src/minimax_h3/ltx_refiner.rs`: `Lora` (open, the key
  lookup, the fusion, the counts) and one hook at the end of `Reader::pull`,
  which is the single funnel every weight the refiner loads passes through.
  `Refiner::open` and `Edges::open` take `Option<&Lora>`.
- `crates/ds4-core/src/minimax_h3/ltx_stage2.rs`: `Stage2::open` passes it to
  both.
- `crates/ds4-core/examples/h3_ltx_refiner_gate.rs`: `--lora FILE`,
  `--lora-strength F` on every mode, a `LORA` report line, and `--lora-probe
  DIR`, the fusion's own gate. `--identity` with `--lora` is refused, and
  `head_check` fuses the projection it transcribes (both explained below).
- No new device kernel: the fusion is `h3d_linear` plus `h3d_add`, both
  existing.

## 2. The reference, read first-hand

`loader/fuse_loras.py` is the whole of it, read at the pin:

- **Which keys meet.** `fuse_lora_weights` (`:119-151`) iterates
  `_affected_weight_keys` (`:183-186`), the set of `.weight` keys the LoRA
  names, and for each gets `_products_for_sd_key` (`:189-208`), which reads
  `<prefix>.lora_A.weight` and `<prefix>.lora_B.weight` where `<prefix>` is the
  model key without `.weight`. The two sides meet because each dropped one
  leading segment earlier: `LTXV_MODEL_COMFY_RENAMING_MAP`
  (`model_configurator.py:222-226`) strips `model.diffusion_model.` and
  `LTXV_LORA_COMFY_RENAMING_MAP` (`sd_ops.py:135-137`) strips
  `diffusion_model.`, so the LoRA keeps the segment the model does not.
- **The scale.** `aggregate_lora_products` (`:99-116`) materializes the first
  product as `matmul(B * strength, A).to(bf16)` and adds the rest with
  `addmm_(..., alpha=strength)`. The `B * strength` multiply is the subtle
  step: `mul` dispatches on the BF16 common dtype (`BinaryMulKernel.cu`,
  `mul_kernel_cuda`) into `opmath_symmetric_gpu_kernel_with_scalars`, whose
  `AUnaryFunctor` holds the scalar at the functor's own F32 precision - the
  comment there is "NB: scalar is stored in higher precision!" (`Loops.cuh`) -
  and whose functor takes F32 arguments. So the product is `0.8f32 * b` in F32
  with one rounding to BF16, not `bf16(0.8) * b`.
- **The product.** `A` is `[rank, in]`, `B` is `[out, rank]`, and the GEMM is
  BF16 in and BF16 out with F32 accumulation. torch's BF16 `mm` reaches
  `cublasGemmEx` with compute type `CUDA_R_32F` (`aten/src/ATen/cuda/CUDABlas.cpp:1218`,
  read at pytorch `main`; the release pins torch 2.13.0+cu132,
  `requirements/stage2-observed.txt:3`) and cuBLAS's default math mode, the
  reduction option defaulting to `AllowReducedPrecisionWithSplitK`
  (`aten/src/ATen/Context.h:533-534`).
- **The add.** `_bf16_fuse` (`:61-68`) is `deltas.add_(weight)`, a BF16 add
  with one rounding, then `.to(weight.dtype)`; every weight this LoRA names is
  BF16 in the checkpoint, so that cast is a no-op.
- **The alpha/rank division does not exist here.** The file's own metadata has
  `lora_rank` 450 and `lora_alpha` 450, and nothing in `ltx-core` reads
  `lora_alpha`; the only scale in the loader is `strength`. 450/450 is 1.0.

Two facts about the files, both read off their own headers this session:

- The LoRA carries 3320 tensors, i.e. **1660 modules**. Per block, 34: five
  projections (`to_q`, `to_k`, `to_v`, `to_out.0`, `to_gate_logits`) for each
  of the six attentions, plus the two feed-forwards' two linears each
  (34 x 48 = 1632). Outside the blocks, 28: eight AdaLN groups' three linears
  each (`adaln_single`, `audio_adaln_single`, `prompt_adaln_single`,
  `audio_prompt_adaln_single`, `av_ca_a2v_gate`, `av_ca_v2a_gate`,
  `av_ca_video_scale_shift`, `av_ca_audio_scale_shift`), plus `patchify_proj`,
  `audio_patchify_proj`, `proj_out` and `audio_proj_out`.
- The transformer is 4059 BF16 tensors and 290 F32 ones, and all 290 F32 are
  `scale_shift_table`s: no `.weight` in the checkpoint is F32, so the fusion is
  BF16-on-BF16 everywhere and the F32 branch of `_bf16_fuse` never applies.
- The release's own descriptor (`configs/checkpoints.json:83-94`) names the
  file `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` from
  `Lightricks/LTX-2.5` at revision `5e6e7101`, 8,899,889,568 bytes, sha256
  `86370bbf...a9bbf5`. The Spark's copy is that size (verified over the sftp
  mount); the digest is re-checked in 11b, on the Spark.

## 3. The port's shape

The hook goes at the end of `Reader::pull`, because that is where every weight
the refiner loads arrives - biases, norms, tables and all - with its bytes in
hand and about to reach the device. `Registry`-style name mapping lives in
`Lora::factors`: strip `model.` and `.weight` off the model key, then look up
`diffusion_model.<rest>.lora_A.weight` and `.lora_B.weight`. A key the file
does not name returns `None` and nothing happens; a key it names with the wrong
shapes is an error, not a skip.

The fusion itself runs on the device, in three ops:

```
A [rank, cols]  --host-->  A^T [cols, rank]  --upload-->
B [rows, rank]  --host-->  bf16(B * 0.8)     --upload-->   h3d_linear  -->  delta [rows, cols]  --h3d_add-->  W
```

- `A` is transposed **on the host**, while its bytes are in hand and it is a
  few megabytes: the reference's `B @ A` contracts A's first axis, and the
  port's `h3d_linear` contracts its operands' last, so the two GEMMs have to
  be given the same contraction as a layout, not as a transpose flag.
- `B * 0.8` is a `f32` multiply rounded once to BF16 on the host, which is
  exactly the arm §2 cites.
- `h3d_linear` (cuBLASLt, `CUBLAS_COMPUTE_32F`, BF16 output) is the GEMM, and
  `h3d_add` is the `deltas.add_(weight)`: it widens both operands, adds in F32
  and rounds once, which is the same function.

**Why the device and not the host.** The host would be simpler to reason about
but it is a ~25 TFLOP F32 GEMM per refiner load: the release fuses 1660
modules, and the two video feed-forward projections alone are 4,096 x 16,384 at
rank 450. The device path costs one cuBLASLt call per weight, and the reference
fuses on the device too.

**The one deviation.** `torch.matmul`'s reduction order is cuBLAS's, this
port's is cuBLASLt's, and both are an F32 accumulation of exact BF16 products
rounded once into BF16. A merged weight can therefore differ from the
reference's by one ulp of the *delta*, and no reimplementation can avoid that.
It is named here rather than hidden. The reference's own math mode defaults to
allowing reduced-precision reduction with split-k (§2), which for a
large-output, small-K GEMM like these should not be selected, but this port
does not claim to have observed which kernel cuBLAS picks for the reference's
call.

## 4. The gate: the fusion against its own transcription

`--lora-probe DIR` writes a LoRA at the release's rank (450) and the graph's
own shapes - `transformer_blocks.0.attn1.to_q` at 4,096 x 4,096 and
`transformer_blocks.0.audio_ff.net.2` at 2,048 x 8,192, one square and one
non-square target - and then fuses one synthetic weight per target, comparing
both the delta and the fused weight against a host transcription of the same
expression.

The transcription is **exact by construction, on purpose**. The factors are
drawn from a three-fractional-bit alphabet (`(-6..=6)/4`), and only three k
rows are nonzero (`0`, `1` and `449`, so a truncated k range or an off-by-one
cannot hide). Every product of two such values therefore has a 16-bit
significand and every partial sum stays inside 24: the accumulation is exact in
any order, in F32 and in F64 alike. What is left is the two roundings the
reference has, which the transcription performs with the same rule the device
does. A wrong contraction, a dropped strength, an add in the wrong place or a
truncating round is then an element that differs, not a tolerance to argue
about.

```
LORA-PROBE file=/tmp/lora-probe.safetensors targets=2 rank=450 strength=0.8
LORA-PROBE transformer_blocks.0.attn1.to_q rows=4096 cols=4096 delta_differ=0 fused_differ=0 pass
LORA-PROBE transformer_blocks.0.audio_ff.net.2 rows=2048 cols=8192 delta_differ=0 fused_differ=0 pass
LORA-PROBE-CONTROL transformer_blocks.0.attn1.to_q strength=1 differ=16567732 pass
LORA-PROBE-CONTROL transformer_blocks.0.attn1.to_q other-axis differ=16567755 pass
LORA-PROBE-CONTROL transformer_blocks.0.audio_ff.net.2 strength=1 differ=16531802 pass
LORA-PROBE-CONTROL transformer_blocks.0.audio_ff.net.2 other-axis differ=16532161 pass
LORA-PROBE-UNTOUCHED transformer_blocks.1.attn1.to_q fused=false differ=0 pass
LORA-PROBE-MODULES targets=2 fused=2 bytes=16588800
```

33,554,432 elements of delta and the same count of fused weight came back
bit-identical to the transcription. The two controls are what makes that
number mean something: transcribed at strength 1 the same factors land on
16,567,732 (square) and 16,531,802 (non-square) different elements, and read on
A's other axis on 16,567,755 and 16,532,161 - so the comparison is sensitive to
both the strength and the contraction it claims to test. The last line is the
mapping's negative: the same class one block over, which the file does not
carry, fuses nothing and leaves the tensor untouched.

## 5. The integration, over the real files

The probe prices the fusion; the reader hook is priced by running the real
LoRA against the real shard. Both artifacts were read over the workstation's
sftp mount of the Spark - `--transformer .../ltx-2.5-22b-dev-transformer-bf16.safetensors`,
`--lora .../ltx-2.5-22b-distilled-lora-450-bf16.safetensors` - so the numbers
below are the workstation's device (sm_89) and a network file system, not the
Spark's.

```
$ h3_ltx_refiner_gate --transformer $TX --blocks 1 --rows 8 --audio-rows 4 --seed 20251030 --lora $LORA
REFINER blocks=1 video_rows=8 audio_rows=4 tensors=84 bytes=773546368 load=8.797
LORA strength=0.8 targets=1660 fused=34 bytes=181825536
```

`targets=1660` is the file's module count from §2, and `fused=34` is exactly
the per-block set - so the mapping met every module the block loads, and the
report's 34 class lines name them (`attn1.to_q` ... `audio_ff.net.2`). The
edges mode, which also loads the head and the eight AdaLN groups, reports
`fused=62` - the block's 34 plus the 28 outside it - and `LORA-CLASS proj_out 1`
is one of them. A full 48-layer stage load therefore fuses all 1660 modules.
Nothing was skipped silently, and a run that names a LoRA and fuses nothing
exits 3 with a line saying so.

The same block, same seed, without and with the LoRA:

| state | without | with |
|---|---|---|
| `block0-video` | fnv `675a49762e2b7fa1` rms 35.193165 | fnv `a2783329b264d2e9` rms 35.042427 |
| `block0-audio` | fnv `e0291f69032c2722` rms 41.390873 | fnv `bc009b93c86bdc0e` rms 41.604156 |

`--compare` puts the difference at rel_rms 6.512e-2 (cosine 0.997898, max_abs
88) on the video state and 3.020e-2 (cosine 0.999555, max_abs 14) on the audio
one; through the edges mode, `edges-block0-video` 8.315e-2 (cosine 0.998290),
`edges-head-video` 6.056e-2 (0.999576) and `edges-head-audio` 6.220e-2
(0.998475). Those are a *structural* confirmation that the fused weights reach
the graph - one block over the gate's own synthetic input, whose magnitudes are
in the hundreds; they are not a quality figure for the model.

Two instruments had to be adjusted, and both are corrections rather than
concessions:

- `head_check` transcribes the head's projection from the shard's raw bytes,
  so with a LoRA on it compared the device's fused head against an unfused
  weight and failed (`head_audio` rel_rms 2.874e-2). It now fuses the
  projection it uploads, through the same `Lora::fuse` the reader uses; the
  fusion is priced by `--lora-probe`, not there, and the check goes back to
  being what it was: the head's own graph. Both heads then pass (rel_rms
  1.595e-3 and 1.327e-3).
- `--identity` zeroes the residual projections (`to_out.0`, the feed-forwards)
  and requires the state back bit for bit. A LoRA writes a delta into exactly
  those projections, so the two cannot hold at once: the combination is
  refused with that reason instead of failing in the state compare.

## 6. What it does not prove

- **The reduction order**, as §3 states: the port's merged weight can differ
  from the reference's by one ulp of the delta, and the port has not observed
  which cuBLAS kernel the reference's own call selects. What is measured is
  that the port computes the expression it claims to, on the shapes the file
  has.
- **Which weight the reference ends up with** needs the reference to run, and
  it does not run on either host. The fused *evaluation* (11b) is the next
  closest thing: the port's states move, and by how much.
- **The delta's own size per class** is not in this record. It is a host-side
  measurement over the real factors (a row sample per class against the
  weight's own RMS) and belongs with 11b's numbers.
- The workstation's `load=8.797` against `7.352` without the LoRA is not the
  fusion's cost: it is a network mount reading 181,825,536 bytes of factors,
  and the two runs are single samples. 11b measures the load on the Spark,
  where the shard is local, and reports the fusion's share.

## 7. Reproduction

```
# the fusion against its own transcription, no shard, no reference file
tests/h3_ltx_refiner_gate --lora-probe /tmp

# the reader hook over the real files (paths on the Spark, reachable from the
# workstation through the sftp mount)
TX=/mnt/h3-artifacts/LTX-2.5/ltx-2.5-22b-dev-transformer-bf16.safetensors
LORA=/mnt/h3-artifacts/LTX-2.5/ltx-2.5-22b-distilled-lora-450-bf16.safetensors
tests/h3_ltx_refiner_gate --transformer $TX --blocks 1 --rows 8 --audio-rows 4 \
    --seed 20251030 --lora $LORA --dump /tmp/lora-fused
tests/h3_ltx_refiner_gate --compare /tmp/lora-none /tmp/lora-fused

# the regression guards, unchanged and without the LoRA
tests/h3_ltx_refiner_gate --probe
cargo test -p ds4-core --lib          # 761 pass, 7 ignored
```

## 8. Owed to 11b

The fused evaluation on the Spark: `--identity` and the `--edges` guard
unchanged (no LoRA), then the stage-2 evaluation on the fused weights with unit
10's seed, latent and context, so the difference is the weights alone; the
merge's own cost with a local shard; and the per-class delta RMS.
