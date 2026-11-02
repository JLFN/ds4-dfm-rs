# The decode is a byte budget: 2.61 GB of casts at 251 GB/s, and a wider transaction changes nothing (2026-11-02)

Unit: handoff section 3, first arm. The three numerical levers the 10-31 record
priced (`vae-decode-2026-10-31.md` section 6) all assume the passes they touch
are *instruction*-bound: "the two cast passes, removed by a BF16 activation
chain", "the F32 elementwise chain, halved in bytes", "attention with BF16 P/V
operands". This arm asked the cheaper question first - can the passes be made
faster without changing one bit? - and the answer is **no, and now we know why**:
they are already at memory bandwidth, so only *deleting* a pass can pay.

**The verdict first.** The four passes (`h3d_cast_kernel`, `h3d_uncast_kernel`,
`h3d_swiglu_f32_kernel`, `h3d_scale_add_f32_kernel`, plus the RMS norm's
write-back) were rewritten to move four elements a thread in 16-byte
transactions, with every expression, every `if (bias)` and every reduction order
left exactly as it was. The frames hash stays `387f816c77ee986f` - the arm is
bit-identical, as designed - and the decode is **46.524 / 45.865 s** against a
control of **46.231 / 45.581 s**. Nothing. The change was reverted; the
arithmetic below is what the unit bought.

| pass | bytes a block call | over the decode | time | rate |
|---|---:|---:|---:|---:|
| the two cast passes | 689.6 MB | 2.61 GB | 10.37 s | 251 GB/s |
| the F32 elementwise chain | 505.9 MB | 1.91 GB | 7.96 s | 240 GB/s |
| the video RoPE | 137.9 MB | 0.52 GB | 3.47 s | 150 GB/s |

Spark `gx10-e1d2`, one binary, sm_121a, CUDA 13.0. The latent is the gate's own
deterministic draw at the product canvas (37x48x84 -> 1344x768, tile 320, 5x3
tiles, 7 chunks, 124 frames), the arms are the delivery's (`H3_VAE_GEMM=bf16i`,
`H3_VAE_ATTENTION=tensor`), and the harness is `tests/h3_vae_gate --latent
37,48,84 --repeat 3`: 3,780 block calls of 2,805 rows each, 15,116 narrows and
15,116 widens.

## 1. The control, and the pin

```
RUN 0 seconds=46.231 ... hash=387f816c77ee986f first=0.287888765 last=0.349735707
RUN 1 seconds=45.581 ... hash=387f816c77ee986f
```

`387f816c77ee986f` is the hash `vae-decode-levers-2026-11-02.md` pinned, so the
decode has moved by 0.5 s of clock state and not at all in value since. This is
the denominator for the unit, and `--expect 387f816c77ee986f` is what every arm
below had to reproduce.

## 2. The arm, and the geometry that kills it

The four-element rewrite is the obvious one: consecutive threads read consecutive
F32 words, so a warp already issues 128-byte transactions; eight-byte BF16 stores
from a warp were two 64-byte halves and became one 256-byte pair. If the pass had
been issue-bound - one thread per element, 1.44 M threads a call - the rewrite
would have been worth up to 4x. It was worth 0.0 s, and the capture says why:

| kernel | calls | grid | elements a call | bytes a call |
|---|---:|---|---:|---:|
| `h3d_cast_kernel` | 15,116 | 5,610 x 256 (x4) or 22,440 x 256 (x4) | 5.74 M or 23.0 M | 34.5 or 138.0 MB |
| `h3d_uncast_kernel` | 15,116 | 5,610 / 16,830 / 44,880 x 256 (x4) | 5.74 / 17.2 / 46.0 M | 34.5 / 103.4 / 275.7 MB |
| `h3d_swiglu_f32_kernel` | 3,779 | 8 x 2,805 x 256 (x4) | 23.0 M | 276.0 MB |
| `h3d_scale_add_f32_kernel` | 7,558 | 2 x 2,805 x 256 (x4) | 5.74 M | 69.1 MB |
| `h3d_rms_norm_f32_kernel` | 7,558 | 2,805 x 256 (x4) | 5.74 M | 46.0 MB |

The grids are the *arm's* (the x4 is the rewrite); the element counts are the
shipped kernel's own shape, 2,805 rows a call. Summed per block call: narrow
w1/qkv/out at 2,048 columns and w2 at 8,192 (241.5 MB of F32 in and BF16 out),
then widen w1 at 16,384, qkv at 6,144, w2 and out at 2,048 (448.1 MB of BF16 in
and F32 out). 689.6 MB a call, 2.61 GB over the 3,780 calls, in 10.37 s of a
46.2 s window: **251 GB/s**. The elementwise chain is the same story at 240 GB/s
and the RoPE at 150.

So the pass is not short of transactions; it is short of bandwidth. Four
elements a thread cannot help a copy that is already moving six bytes an element
at the rate the package can deliver them, and the 5.0 GB those three passes move
over a decode is 3.3x the 1.54 GB of F32 frames they exist to produce.

## 3. What this changes in the plan

The three numerical levers survive, but their mechanism is not "make the pass
cheaper" - it is "move fewer bytes", and the number each one is worth follows
from the table above rather than from the 10-31 projection:

| change | bytes removed | worth |
|---|---:|---:|
| a BF16 activation chain between the kernels (the casts disappear) | 2.61 GB | -10.4 s |
| the same chain at half the width for the elementwise pass | 0.96 GB | -4.0 s |
| the same chain writing Q/K/V and the RoPE output at half the width | 0.26 GB | -1.5 s |
| attention staging BF16 K/V (it already computes in BF16) | 3.8 TB of L2 | -3.2 s |

That is the ~19.6 s `vae-decode-2026-10-31.md` section 6 projected, arrived at
from measured traffic instead of from a projection: a decode toward **27 s**
from 46.2 s. None of those four is schedule-only, so each one is an arm, and the
delivered clip is the judge in the way the handoff's section 3 acceptance states.

Two of them are the same change as seen from two ends - once the block's
activations are BF16 there is nothing left to cast and the attention's operands
are already the type it wants - which is why the next unit is one arm with four
parts and not four units.

## 4. What the reference has to say

`/data/Rapid-H3` solved the same decode on the same box from the other side
(`custom_nodes/rapid_h3/vae_fast.py`, its levels 1-4, `docs/kernels.md` section
"Video VAE"). Three of its findings transfer and one is a warning:

- **It reached 18.2 s at 124 frames and 18.4-19.5 s at 141, and its lossless
  trims are the same shape as ours**: a fused w1 SwiGLU epilogue (its 78.8 ->
  70.9 ms a block) and q/k RMS+RoPE fused into the attention's quantizer (-1.0 s
  a decode, bit-identical). Both delete a read-back; neither makes a kernel
  faster.
- **Its tile batching is already ComfyUI's, and bigger batches were slower**
  (27 s at 7-14 tiles against 20 s), so batching is not a lever for us either -
  our port already carries a whole temporal chunk's rows in one call (2,805),
  which is the same idea at a different granularity.
- **Its bits are not ours**: int8 `convrot` codes, fp16 `h`, 42.7 dB PSNR against
  its own fp16. It can suggest structure and cannot judge our clips.
- The warning: its decode profile is where ours is - GEMMs at ~80% of peak,
  attention and passes to taste - so nothing in its number is a kernel we have
  failed to write.

## 5. Traps this unit paid for

- **Two decodes at once inflate everything by ~1.9x.** The nsys capture was
  taken while the arm's third repeat was still on the GPU: window 87.608 s, busy
  84.794 s, and that repeat itself read 89.341 s against 45.9. The *composition*
  survives (it is the same 72,531 kernels at the same ratios) and the absolute
  numbers do not. A profile and a timing run are two separate sessions.
- **A wider transaction is not a faster pass.** Coalesced scalar reads already
  fill a 128-byte line per warp; four elements a thread saves instructions the
  copy was not waiting on.
- **The 10-31 table's "class" column is the decision, not the size.** Every one
  of those levers is numerical because the only way to stop paying for a pass is
  to stop moving its bytes, i.e. to change what the next kernel reads.
- **`make tests/h3_vae_gate` needs `CUDA_ARCH=sm_121`** or nvcc falls back to its
  own default arch and ptxas rejects the tree's `.bf16` MMAs ("requires .target
  sm_80 or higher").
