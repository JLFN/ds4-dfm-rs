# The fused evaluation on the Spark (unit 11b)

Date: 2026-10-30. Branch `feat/sol-h3`, commits `e7ce3725` (the delta
measurement) and this record's own commit. Unit 11a is
`docs/minimax-h3/ltx-lora-fusion-2026-10-30.md`: the fusion, its gate and the
key mapping. Reference pins as unit 10.

Unit 11a proved the fusion computes the reference's expression and that it
reaches the graph. This unit is the evaluation: the same run unit 10 made, on
the weights the release actually denoises with, so the difference between the
two is the weights and nothing else.

## 1. The artifacts

Both files are on the Spark at `/mnt/h3-artifacts/LTX-2.5/`, and the numbers
below are the Spark's own (a local NVMe, the model loaded in memory):

| file | bytes | digest |
|---|---|---|
| `ltx-2.5-22b-dev-transformer-bf16.safetensors` | 42,018,190,584 | (per unit 8) |
| `ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | 8,899,889,568 | sha256 `86370bbf79a9eb4edaa158907e2b48a5188fe4c5dc8ce30c7eb8f2f131a9bbf5` |

That digest is the one the release's own descriptor states for the file
(`configs/checkpoints.json:83-94`), so the adapter this unit fuses is the
adapter the release names, not a lookalike.

## 2. The guards, and that nothing else moved

Everything below ran `make CUDA_ARCH=sm_121` on the Spark's clone at
`c438a908`.

- **The kernel probe set is unchanged.** All 31 `PROBE` lines pass and no line
  in the run says `FAIL`, with the same numbers as unit 10
  (`sol_forward_tau1` still `differ=1505114 rel_rms=2.145e-3`, `randn` still
  `fnv=8cb73e5f217fe213`), which is the point: this unit touches weights, not
  the graph.
- **`--identity` still passes, without the LoRA**: 15 tensors zeroed, both
  modalities `differ=0`, `LTX-REFINER-IDENTITY-OK`. A merge touches every
  linear in the block, so this had to be re-run rather than assumed.
- **The `--edges` guard is unchanged without the LoRA** and passes with it:
  the released arm's block-0 state is `fnv=8478e5c4a68c5bfd` and
  `DIAGNOSTIC NoSelfAttention differ=32610 rel_rms=1.235e0 pass` - byte for
  byte unit 10's numbers, so 11a's reader hook left the unfused graph alone.
  With `--lora` the same guard reports `differ=32583 rel_rms=1.219e0 pass`
  (released arm `fnv=08024c697937c950`), i.e. the two attentions are still the
  live term they were fixed to be in unit 10, on the fused weights too.
- **The fusion's own gate passes on sm_121 exactly as on sm_89**: the same
  `delta_differ=0 fused_differ=0` on both targets, the same control counts
  (16,567,732 / 16,567,755 / 16,531,802 / 16,532,161) and the same
  `LORA-PROBE-UNTOUCHED ... differ=0`. The synthetic factors make every sum
  exact, so this gate is architecture-independent by construction, and it is
  the one place the port's cuBLASLt result can be compared to a transcription
  bit for bit.

## 3. The merge's cost

The fusion is a load-time operation on 1660 modules: 8,899,889,568 bytes of
extra reads and one GEMM per module.

| load | without the LoRA | with it | difference |
|---|---|---|---|
| 48 layers | 5.985 s (unit 10) | 13.447 s | +7.462 s |
| 1 block | 0.154 s | 0.509 s | +0.355 s |

So the merge costs about 7.5 s on a 6 s load, on a warm page cache. It is paid
once per refiner load, it needs no extra disk, and the fused weights are the
release's own arithmetic rather than a derived artifact that would have to be
carried and re-validated.

## 4. What the weights do

**The block moves.** The same one-block run with and without the LoRA, seed
`20251030`:

| state | without | with |
|---|---|---|
| `block0-video` | fnv `95929ad387521ec3` rms 35.200760 | fnv `3daae05d1ff4f189` rms 35.042267 |
| `block0-audio` | fnv `fefff4d806124fcd` rms 41.387573 | fnv `47b7ed0f7c6dc28c` rms 41.611214 |

`--compare` puts it at rel_rms 6.534e-2 (cosine 0.997885, max_abs 88) on the
video state and 3.038e-2 (cosine 0.999550, max_abs 14) on the audio one, which
is four times the policy's own effect at the same frontier (unit 10 measured
1.576e-2 for Sol-Attn at step 0).

**Where it comes from.** `--lora-deltas` reads each merged delta and its weight
back and reports the RMS per class. Over block 0's 34 classes:

| class | delta RMS | delta max | weight RMS | ratio |
|---|---|---|---|---|
| `attn1.to_q` | 5.484e-4 | 3.342e-3 | 1.679e-2 | 3.3% |
| `attn1.to_k` | 5.958e-4 | 3.326e-3 | 1.883e-2 | 3.2% |
| `attn1.to_v` | 5.577e-4 | 3.372e-3 | 2.618e-2 | 2.1% |
| `attn1.to_out.0` | 5.869e-4 | 3.494e-3 | 3.145e-2 | 1.9% |
| `ff.net.0.proj` | 4.067e-4 | 3.357e-3 | 1.518e-2 | 2.7% |
| `audio_ff.net.2` | 5.917e-4 | 3.616e-3 | 3.906e-2 | 1.5% |
| `audio_to_video_attn.to_k` | 1.650e-5 | 5.836e-4 | 2.374e-2 | 0.07% |
| `audio_to_video_attn.to_q` | 1.855e-5 | 8.392e-4 | 2.748e-2 | 0.07% |

The range over all 34 classes is 1.3e-5 to 6.7e-4 RMS against weights of
5.6e-3 to 5.6e-2, i.e. **0.07% to 3.3% of the weight the delta lands on**.
That is a real change to the model and not a rounding: it is one to two orders
above a BF16 ulp of the weight, and it is why the state moves by 6.5%.

This also bounds the deviation 11a named. The reduction order can move the
delta by a few F32 ulps, i.e. ~1e-5 relative of a delta that is itself at most
3.3% of the weight: the merged weight then differs from the reference's only
where the F32 sum lands within that window of a BF16 boundary, and where it
does the change is one ulp of that weight.

## 5. The evaluation

Unit 10's own seed, latent and context, 48 layers, on the fused weights:

- **dense**: 125.751 / 126.866 / 127.884 s per step, denoise 380.522 s, wall
  395 s. The plain dev transformer was 126.206 / 127.332 / 128.067 (381.627),
  wall 389 - so the fused weights cost nothing in speed.
- **sparse** (`--sol`): 41.412 / 37.281 / 33.725 s, denoise 112.453 s, wall
  127 s against the plain dev's 41.767 / 37.362 / 34.011 (113.163), wall 120.
  **3.39x**, `sol_calls=47 dense_calls=1` every step.
- The routing barely moves: `exact_share` 0.2304 / 0.1807 / 0.1436 against
  0.2292 / 0.1835 / 0.1479.
- `LORA ... fused=1660 bytes=8899362816`: every module in the file, and
  essentially every byte of it.
- Memory is unchanged: peak 44 GiB of 121 in both runs, 853,400 kB maximum RSS
  - the deltas are transient, one per weight.

| state | plain dev (unit 10) | fused |
|---|---|---|
| `step0` | `997fbdebb2df78bb` rms 0.743450 | `c3a608993bf76f7f` rms 0.761350 |
| `step1` | `13f7b0c9cb09ec6f` rms 0.522007 | `ac3237f152f23d29` rms 0.620069 |
| `step2` | `8b43989aecdb52c7` rms 0.476979 | `08540329a57e505f` rms 0.811042 |
| `latent` | `30f1fbe7a6bd5e9b` | `b5f27d2d6d7a2e73` |
| `latent` (sparse) | `1b6382b9e5cb7dd1` | `f40259f5a9c8e9bc` |

The differences, with `stage-noise` and `stage-state0` `differ=0` as the
control in every pair:

| pair | step0 | step1 | step2 / latent |
|---|---|---|---|
| plain dev vs fused, dense | 7.830e-2 | 3.484e-1 | 6.321e-1 |
| plain dev vs fused, sparse | 8.300e-2 | 3.556e-1 | 6.392e-1 |
| fused dense vs fused sparse | 6.990e-2 | 3.199e-1 | 5.853e-1 |

(cosine at the latent: 0.804573 dense, 0.811372 sparse, 0.825126 for the policy
on the fused weights.)

Two things follow that the campaign has to take seriously:

- **The weights matter more than the policy.** The distilled LoRA moves the
  latent by rel_rms 0.632 against the 0.190 the Sol-Attn policy moves it (unit
  10, same frontier). Unit 9's question - is the sparse route acceptable - was
  asked on weights the model was not distilled for.
- **On the fused weights the policy's own effect is larger**: 6.990e-2 at step
  0 against 1.576e-2, and 0.585 at the latent against 0.190. The distilled
  weights amplify the sparse/dense divergence, so the policy's cost, whatever
  it is, is not the one unit 9 measured on the plain dev transformer.

What the evaluation does not say is anything about *quality*. The latent is
still a draw through the ported upscaler and adapter (unit 10, §5), not a real
image latent, and the reference's own output is not available to compare
against; what this unit establishes is that the port now denoises with the
weights the release names, and how far they move the states.

## 6. What it does not prove

- **That the merged weights are the release's, bit for bit.** §4 bounds the
  only step that cannot be copied exactly, and the bound is small, but the
  reference did not run here.
- **Any quality claim.** See above: the input latent is synthetic, and the
  output path (the video VAE decoder) is not ported yet, so there is not even
  a frame to look at.
- **The per-class deltas across the model.** They were measured on block 0's
  34 classes, over the workstation's network mount, because `--lora-deltas`
  costs two device reads per weight. The other 1626 modules were fused but not
  priced individually.
- **That the 48-layer fused run is deterministic across loads.** Each stage run
  is one sample; the plain-dev numbers it is compared against are unit 10's,
  taken on the same host and the same clock policy but in a different session.

## 7. Reproduction

```
# on the Spark, in ~/ds4-sol-h3 at c438a908 (the script also builds)
bash ~/ds4-test/unit11b-spark.sh          # the whole unit, log at ~/ds4-test/unit11b.log

# the per-class deltas, one block (cheap, and it works from the workstation
# over the sftp mount of the artifacts)
tests/h3_ltx_refiner_gate --transformer $TX --blocks 1 --rows 8 --audio-rows 4 \
    --seed 20251030 --lora $LORA --lora-deltas
```

## 8. What this changes for the campaign

- Every state hash from units 6 to 10 is superseded *again* for anything that
  claims to be the release's stage-2 output: units 6 to 8 were superseded by
  unit 10's write-back, and all of them by the weights.
- The stage-2 evaluation now runs the release's arithmetic end to end, on real
  conditioning, with a real adapter - and still starts from a synthetic latent
  and stops at a latent, with no decoder behind it.
