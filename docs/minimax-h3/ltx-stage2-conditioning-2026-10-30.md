# The stage-2 evaluation on real conditioning, and the inert attention (unit 10)

Date: 2026-10-30. Branch `feat/sol-h3`, commits `88e8f8e5` (the video
self-attention diagnostic), `ab6e522b` (the adapter gate's `--dump-latent`) and
`75788757` (the residual write-back and its guard). Reference pins as unit 9:
`/data/ltx2-pin/packages` (Lightricks/LTX-2 `d1511477` = tag v1.2.0) and the
release runtime `/data/Sol-h3-Spark` @ `6e4a440`.

Unit 9 landed the release's Sol-Attn policy, measured it at 3.4x, and found
every state it produced bit-identical to the dense run it was compared against.
This unit was to explain that: run the ported producers so the evaluation reads
real conditioning, and price the video self-attention's own term against the
state's BF16 resolution. Pricing it found something else. The term was not
below the resolution - it was never applied. `self_and_text` computed the
self-attention and cross-attention residuals into a scratch buffer that nothing
read, so both attentions had been dead in the refiner since unit 6, and every
gate in the campaign was a self-consistency comparison that could not see it.

## 1. What landed

- `crates/ds4-core/src/minimax_h3/ltx_refiner.rs`: `Diagnostic`, a switch with
  no counterpart in the release, and the write-back that makes `self_and_text`'s
  running sum reach the modality state.
- `crates/ds4-core/src/minimax_h3/ltx_stage2.rs`: `Stage2::set_diagnostic`.
- `crates/ds4-core/examples/h3_ltx_refiner_gate.rs`: `--no-self-attention` on
  the stage and edges modes, the `STAGE-DIAGNOSTIC` and `DIAGNOSTIC` lines, the
  edges mode's two-arm guard, and a `rms=` field on every `STAGE` line.
- `crates/ds4-core/examples/h3_ltx_adapter_gate.rs`: `--dump-latent FILE`.
- No new device kernel: the diagnostic is two host-side lines around one
  `h3d_copy`, and the fix is one more.

## 2. The instrument

`BasicAVTransformerBlock.forward`'s video path is `attn1` and then
`post_sa_function` (`transformer.py:278-285`), and `PytorchPostSAFunction`
(`ops.py:72-84`) is `x_fma = x + y * gate` (`:81`). The port's `self_and_text`
does the same in device ops and then normalizes the sum for the
cross-attention (`transformer.py:283-289`).

`Diagnostic::NoSelfAttention` replaces the self-attention's `mul` and `add` with
one `h3d_copy`, leaving the residual as `state` itself: the term is absent
rather than scaled, which is the same arithmetic as a block whose
`attn1.to_out.0` weight and bias are zero. Two arms with identical inputs and
seed then isolate what the term contributes - provided the term reaches the
state at all, which is the assumption this unit tested.

It does not leave the block otherwise unchanged: with the term gone
`post_normed` is the norm of `state` rather than of `state + term`, so the
cross-attention reads a different operand. The measured quantity is "the block
with its video self-attention removed" against "the block as released".

## 3. What the first pass measured, and the finding

Dense, sparse and term-dropped arms over 48 layers, on the real conditioning
(section 5), the same seed and the same latent:

| arm | step 0 | step 1 | step 2 | denoise |
|---|---:|---:|---:|---:|
| dense | 125.725 s | 126.791 s | 127.794 s | 380.333 s |
| sparse (`--sol`) | 43.376 s | 37.960 s | 31.250 s | 112.607 s |
| dense, term dropped | 128.059 s | 127.652 s | 127.028 s | 382.761 s |

All three arms produced the same four state hashes - `4da997b8948111eb`,
`8c76d6206fed46e1`, `5a26fb341c743ae5`, `63490108be352505` - and `--compare`
reported `differ=0` on all six dumps of both pairs. A one-layer pair over the
same latent, where no later layer can mask anything, was identical as well
(`denoise=9.433`).

That is not a resolution statement. The term is not small: block 0's AdaLN gate
row, `scale_shift_table[2] + modulation[:, 2, :]`, is `rms 0.157` at the step's
sigma (the table read from the checkpoint, F32, row 2 rms 0.0997; the
modulation from the gate's own `--edges --signals` dump), and
`attn1.to_out.0.weight` is a full `[4096, 4096]` matrix of std 0.0315 with no
zero. A state that does not move when a live term is removed means the term is
not being applied.

## 4. The write-back

`self_and_text` ended at `gpu.add(&m.crossed, &m.residual, &m.gated)`. In the
reference that line rebinds the modality state - `vx, vx_normed =
self.post_sa_function(...)` (`transformer.py:285`) and `vx = vx +
self._apply_text_cross_attention(...)` (`:289-292`) - and the block's AV step
and feed-forward both read `vx`. In the port the sum landed in `m.crossed`,
which had **no reader anywhere in the crate**: `feed_forward` and `av_direction`
read the modality tensor, which still held the block's input. The self-attention
and the cross-attention were therefore dead code, and had been since unit 6.

The fix is one `h3d_copy` per modality per block:

    gpu.copy(state, 0, &m.crossed, 0, (rows * dim) as usize)

so the running sum is the modality state at every point the reference rebinds
it, which is what the AV operands, the AV residuals and the feed-forward read.

The guard is the `--edges` mode's `DIAGNOSTIC` arm: it runs the released graph
and the term-dropped graph from the same input and **fails** when the two agree.
Every gate this campaign built before it compares the port against itself, and
none of them can see a value that never arrives; this one asserts an inequality
that only a live term can satisfy.

Measured after the fix, on the Spark:

    EDGES tensors=59 bytes=854954496 video_rows=8 ... load=0.213
    STAGE edges-block0-video fnv=8478e5c4a68c5bfd high=14.500000 low=-161.000000 rms=2.705516
    DIAGNOSTIC NoSelfAttention values=32768 differ=32610 rel_rms=1.235e0 pass

and the same pair at one layer over the real latent and context, where no later
layer can mask anything:

    released        stage-step0 fnv=49e07733b20de464 rms=0.433066
    term dropped    stage-step0 fnv=85f1d86b583b54a1 rms=0.622933
    COMPARE stage-step0 ... differ=2057929 rel_rms=5.600e-1 cosine=0.841234
    COMPARE stage-step1 ... differ=2062264 rel_rms=8.675e-1 cosine=0.497446
    COMPARE stage-step2 ... differ=2062276 rel_rms=1.107e0 cosine=0.130585

The block's own output moved with it. The same `--edges --rows 8` run reports
`edges-block0-video rms=5.199` before the fix (that is the feed-forward alone,
with the attention discarded) against `rms=2.706` after it, and the one-layer
released arm's first state is `rms=0.687485` before against `rms=0.433066`
after - with the pre-fix pair's step-0 hash `2419cfa2e6a2a7be` identical on
both arms. At one layer the video self-attention changes the state by more than
the state itself.

## 5. The producers, run for real

### The context is real conditioning

`h3_gemma_connector_gate --gemma <int8 gemma> --connector <bf16 dev
transformer> --prompt 'A golden retriever running on a beach at sunset.'
--dump DIR --cache FILE` is the whole chain for a real prompt: the
checkpoint's own tokenizer (1024 wide, bos 2, 1013 padding - the same three
numbers unit 4 recorded), the Gemma4 tower's 48 layers, the feature extractor
and both 8-block connectors. `tensors=262 bytes=6344455168 load=4.490
run=0.477` on the Spark. `DIR/video.bf16` is `[1024, 4096]`, which is exactly
what the refiner's `--context` reads, and `--cache` writes the port's own
stage-2 cache for the same prompt.

Two checks tie the evaluation to that artifact: `video.bf16`'s sha256 is
`6b44ac18c910e0d353814e56383163032425a17f76c744d1ed6de8930b9ba96b`, identical
to the video context inside `prompt-cache10.pt`, and
`h3_prompt_cache_gate --inspect prompt-cache10.pt --prompt '<the prompt>'`
reads it back through the port's own container (`PROMPT-CACHE-OK`,
fingerprint `79ca00d1…`, `CONTEXT video shape=1,1024,4096 dtype=torch.bfloat16`).

The artifact is the release's own BF16 deviation (unit 4's §2): the INT8 gemma
file is the recipe's, and the connector comes from the BF16 dev transformer
because the recipe's `dev` INT8 connector file is not in the store. A
BF16-derived cache is not interchangeable with the INT8 one; section 9 carries
that.

### The latent is a draw, run through the ported path

`h3_ltx_adapter_gate --dump-latent FILE` writes the port's own
`refiner_input`: the adapter's normalized LTX latent cropped to the refiner's
16 frames (`runtime/stage2.py:76`, `latent[:, :, :16]`) as the raw BF16
`[channels, rows]` file `--latent` reads. On the Spark it reproduces the
reference's own float32 vector to `h3_upscaled 6.668e-3`, `aligned_h3
1.775e-3`, `adapter_output 1.208e-2`, `refiner_input 1.212e-2` relative RMS -
the bf16 arm's precision statement against that vector, not a structural gate
(the f32 arm is, and the workstation's run of it is 2.962e-6, 5.805e-9,
6.064e-6, 6.139e-6).

The dumped file decodes to `mean 0.005386, sd 0.869310, range [-6.375,
8.1875], no nan` where the reference's own float32 vector is `mean 0.005379,
sd 0.869607, range [-6.388, 8.181]`, so the crop and the bf16 storage are the
whole of the difference.

What it ran on is the golden's `h3_input`, which
`tools/minimax_h3/ltx/latent_oracle.py:175-176` builds as
`torch.randn(1, *h3_shape)` under a fixed seed: a draw in the shape and scale
the node's own normalization expects, not a latent that any video produced.
The latent the refiner evaluated is therefore the release's own definition of
a refiner input, with everything the ported upscaler and adapter do to it, and
without the structure a real H3 generation would put there. Section 9 carries
that.

## 6. The evaluation after the fix

Same latent, same real context, same seed `20251030`, 48 layers, on the Spark:

| arm | step 0 | step 1 | step 2 | denoise | wall |
|---|---:|---:|---:|---:|---:|
| dense | 126.206 s | 127.332 s | 128.067 s | 381.627 s | 389 s |
| sparse (`--sol`) | 41.767 s | 37.362 s | 34.011 s | 113.163 s | 120 s |
| dense, term dropped | 128.128 s | 127.239 s | 126.990 s | 382.380 s | 390 s |

The sparse arm is **3.372x** the dense one, essentially the pre-fix 3.377x: the
policy's cost is the same, and the fix's own cost is one `h3d_copy` of 132 MB
per block per step (48 copies, 6.3 GB a step), which lands inside the
run-to-run spread of the dense arm (its three pre-fix steps spread 2.07 s
against the 0.4 s difference here). Peak memory is unchanged at 44 GiB used,
77 GiB available, and the process's own resident set is 645 MB.

The states are the graph's now, and the two comparisons are the point:

    dense vs sparse   step0 differ=1794342 rel_rms=1.576e-2 cosine=0.999876 max_abs=0.1406
                      step1 differ=2002602 rel_rms=7.013e-2 cosine=0.997629 max_abs=0.4414
                      step2 differ=2043080 rel_rms=1.900e-1 cosine=0.983461 max_abs=0.9844
    dense vs term-off step0 differ=2059476 rel_rms=4.709e-1 cosine=0.981841 max_abs=1.3828
                      step1 differ=2063015 rel_rms=2.778e0 cosine=0.278525 max_abs=3.0688
                      step2 differ=2062529 rel_rms=1.778e0 cosine=0.081205 max_abs=3.9375

with `stage-noise` and `stage-state0` at `differ=0` in both pairs, which is the
control that the two arms did see the same input.

So unit 9's finding is resolved in the opposite direction from its hypothesis:
the Sol-Attn policy is not below the state's BF16 resolution, it is a *visible*
approximation once its output reaches the state - 1.6% of the state at step 0,
7.0% at step 1, 19.0% at step 2, cosine to dense 0.99988 down to 0.98346. The
release ships this schedule and accepts that loss; what was wrong was the port,
not the policy. Removing the whole video self-attention instead moves the state
by 47% to 278% (cosine 0.982 down to 0.081), so the term is the larger part of
what each block does, not a correction.

The route shares moved with the state (0.2292 / 0.1835 / 0.1479 against the
pre-fix 0.2025 / 0.1484 / 0.1220), and the release's accounting still holds at
every step: `sol_calls=47 dense_calls=1`.

The dense states are `step0 997fbdebb2df78bb rms=0.743450`,
`step1 13f7b0c9cb09ec6f rms=0.522007`, `step2 8b43989aecdb52c7 rms=0.476979`
and `latent 30f1fbe7a6bd5e9b rms=0.476979`; the sparse arm's are
`1626ab4f754afc09`, `de0c851b7307faf9`, `6a22f0871ebbb565` and
`1b6382b9e5cb7dd1`; the term-dropped arm's are `823eee6db78bb40e`,
`cba4719caa45bd26`, `73209179e1a4e708` and `4fbf3021c86584f4`. Changing the
graph moved the dense states well away from the pre-fix ones
(`4da997b8948111eb`, `8c76d6206fed46e1`, `5a26fb341c743ae5`,
`63490108be352505`), which is itself the clearest statement of how much of the
refiner the write-back restores.

The identity invariant still holds after the fix: `--identity` over one block
with the audio half present zeros 15 tensors and reports `differ=0` on both
modalities (`IDENTITY identity-video ... pass`, `identity-audio ... pass`,
`LTX-REFINER-IDENTITY-OK`), so the write-back leaves the residual path
bit-exact when every added term is zero, and the audio path is exercised with
it.

## 7. What it does not prove

- **The refiner's values are not the release's.** `runtime/stage2_ops/models.py:
  76-79` builds the refiner's stage with `LoraPathStrengthAndSDOps(paths
  ["refiner_lora"], 0.8, LTXV_LORA_COMFY_RENAMING_MAP)` and passes
  `loras=(lora,)` to `DiffusionStage.from_checkpoint`, so the release denoises
  the distilled, LoRA-merged weights. The port denoises the plain dev
  transformer. A three-interval `SIGMAS` schedule is the distilled route, so
  this is a named deviation, not a rounding difference, and it is what unit 11
  is for.
- **No golden for the refiner's values.** Every number above compares the port
  against itself (the two arms, the two hosts, the release's call accounting).
  Nothing here says the latent, the context or the sampler produce the
  reference's pixels.
- **The AV path was not re-measured.** The write-back changes what the A2V/V2A
  operands and residuals see, so every block and edge hash recorded in units 6
  to 8 is superseded; those gates must be re-run before their hashes are quoted
  again. This unit ran the video-only stage.
- **The latent is a draw** (section 5), and the Sol-Attn policy's quality loss
  is now real - the policy is a deliberate approximation and this unit does not
  characterize what it costs in pixels.
- **The pre-fix numbers are the bug's evidence, not measurements of the
  release's graph.**

## 8. Reproduction

Workstation (sm_89):

    make CUDA_ARCH=sm_89 tests/h3_ltx_refiner_gate tests/h3_ltx_adapter_gate \
        tests/h3_prompt_cache_gate
    ./tests/h3_ltx_refiner_gate --probe
    ./tests/h3_ltx_adapter_gate <golden> <upscaler.safetensors> <adapter-dir> --arm f32
    ./tests/h3_ltx_adapter_gate <golden> <upscaler.safetensors> <adapter-dir> \
        --arm bf16 --dump-latent /tmp/refiner-latent10.bf16

Spark (sm_121): `~/ds4-test/unit10-spark.sh` (the first pass, whose arms are
section 3's evidence) and `~/ds4-test/unit10b-spark.sh` (the fix's guard, the
one-layer pair and section 6's arms). The one-layer pair on its own, 10 s an
arm:

    ./tests/h3_ltx_refiner_gate --stage --transformer <dev transformer> --layers 1 \
        --seed 20251030 --latent refiner-latent10.bf16 \
        --context context10/video.bf16 --dump l1off
    ./tests/h3_ltx_refiner_gate --stage --transformer <dev transformer> --layers 1 \
        --seed 20251030 --latent refiner-latent10.bf16 \
        --context context10/video.bf16 --no-self-attention --dump l1no
    ./tests/h3_ltx_refiner_gate --compare l1off l1no

The latent's statistics, with no interpreter (the file is raw BF16,
little-endian, so each 16-bit unit is an f32's top half):

    od -An -tu2 -v refiner-latent10.bf16 | awk '{for(i=1;i<=NF;i++){v=$i;
      s=(v>=32768)?-1:1; e=int(v/128)%256; m=v%128; if(e==0)val=0;
      else if(e==255){nan++; continue} else val=s*(1+m/128)*2^(e-127);
      n++; sa+=val; sb+=val*val}} END{printf "mean=%.6f std=%.6f\n",
      sa/n, sqrt(sb/n-(sa/n)^2)}'

## 9. Owed decisions

Unchanged from units 8 to 9, plus one:

1. The decoder's route (`F32` 110.4 s, `bf16` 30.4 s, `bf16i` 24.4 s on job A).
2. The recipe's `dev` INT8 connector file (sha256 `2edbdb44…caf92b`), so a
   cache built from the BF16 connector is not the release's.
3. A real H3 latent for `--latent`: a stage-1 generation on this host, or a
   dump from the reference.
4. **New:** the distilled LoRA at strength 0.8. Until it lands, every stage-2
   value is the port's dev transformer, not the release's refiner.

