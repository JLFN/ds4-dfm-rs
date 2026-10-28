# Sol-H3-Spark: what it is, and what of it we can use

Recon of NVIDIA's H3 pipeline for one DGX Spark, and the portability question
behind branch `feat/sol-h3`. Everything here is read from the source at
`/data/Sol-h3-Spark` (NVlabs/Sana, branch `sol-engine`, commit `6e4a440`,
blobless sparse checkout whose cone is `models/minimax_h3/Sol-H3-Spark`,
`models/minimax_h3/Sol-H3`, `models/minimax_h3/GB200`,
`models/minimax_h3/super_acceleration` and `techniques/sparse_backends`) and
from the project page and the package's own README, `docs/setup.md` and
`docs/validation.md`. Nothing here has been run.

## What it is

A two-stage video-and-audio pipeline for one GB10, not a kernel and not a
variant of the H3 engine this repo ports:

1. Draft: MiniMax-H3 at 672x384x124 with W8A8 FP8 weights, a four-step
   FastH3 VSA DataFree LoRA, and VSA attention at 90% video sparsity in
   64-token blocks, the selected blocks run by cuDNN's BSA backend
   (`runtime/stage1_ops/vsa.py`).
2. Transfer: the LBH learned H3 x2 latent upscaler (normalized H3 latent
   24x37x24x42 to 24x37x48x84), then a learned H3-to-LTX latent adapter
   (to 128x17x24x42). No pixel-space decode and re-encode between stages
   (`runtime/stage2_ops/h3_upscale.py`, `runtime/latent_transfer.py`).
3. Refine: LTX-2.5 dev in BF16 with a distilled LoRA at strength 0.8, three
   updates against the frozen H3 audio latent, Triton Sol-Attn with per-step
   thresholds 1 / 1.25 / 1.5 and a dense first layer
   (`runtime/stage2_ops/sol.py`).
4. Decode: the Conv video VAE, chunked H.264 and AAC mux. H3's own video VAE
   decoder is bypassed entirely.

Their headline, from the project page and the package README: 1344x768,
121 frames, 5 s, 24 FPS, 56.17 s hot end to end on one Spark, split as Qwen
2.4%, H3 four-step generation 35.2%, upscaler plus adapter 3.8%, LTX three-step
refinement 44.0%, LTX VAE decode 12.5%, mux 1.3%, overhead 0.9%. The page's own
ladder is 374 s for the four-step LoRA alone, 153 s with quantization, 56 s with
the whole engine, which is 6.7x.

## Why that number is not comparable to ours

Our yardstick job is 768x768, 107 frames, two steps, 277.019 s with the tensor
DiT attention (21.0 s of it the VAE decode in the tensor arm). Their job is
1344x768, 121 frames, seven model evaluations in two model families, with FP8
and NVFP4 weights, sparse attention, a distilled four-step LoRA and a different
refiner and decoder. The output, the model set and the architecture all differ.
No ratio between the two numbers means anything, and this file does not compute
one.

Their per-stage split is the useful part: their H3 stage is ~19.8 s for four
384p updates, their LTX refiner is ~24.7 s for three 768p updates, and their
decode is ~7.0 s. Our denoise at 20 steps is ~403 s at 768p with one model. So
the gap is dominated by step count, draft resolution and model choice, not by a
kernel.

## The five techniques, and what each needs

| Technique | Mechanism, cited | Needs | Portable to this engine |
|---|---|---|---|
| Sol-Attn | query-block threshold over light QK scores, selected blocks exact, skipped blocks corrected from pooled K/V, one online-softmax pass, no retraining (`techniques/sparse_backends/sol_attn`, `_vendor/flash_attn/cute/block_sparsity.py`) | nothing beyond the algorithm; their kernels are Python/Triton/CuTe | as an algorithm, yes; their code is the reference |
| VSA draft attention | 90% video sparsity, 64-token blocks, cuDNN BSA for selected blocks, 52 compiled full-graph regions | cuDNN frontend 1.27 and the FastH3 VSA LoRA | the policy yes, cuDNN BSA no (Phase 5 forbids cuDNN) |
| Latent transfer | upscaler then adapter, no inter-stage VAE (page: "24 s saved", VAE handoffs 2 to 0) | the LBH upscaler checkpoint (691 MB, no declared license) and the public H3-to-LTX adapter | yes, as new kernels plus weights |
| Cached refiner prompt | one content-agnostic prompt encoded once offline through INT8 Gemma and the connector; Gemma never resident | only meaningful with the LTX refiner | no, without the LTX stage |
| Resident + quantized | NVFP4 Qwen, FP8 H3, low-bit video components, 116.9 GiB of the 119.68 GiB pool, no per-request construction | quantized checkpoints, or our own quantizer | FP8 DiT yes; NVFP4 Qwen is the text encoder path |

Two things this repo can take from the source without new weights, and they are
the two biggest levers on our own denoise:

1. Few-step distilled LoRA. Their Stage1 is four updates because of a distilled
   LoRA (FastVideo FastH3 VSA DataFree, 5.34 GB, pinned revision), not because
   of a scheduler trick. Our default is 20 steps. Four steps would be a 5x on
   the denoise, and it needs LoRA load/merge in our host plus that adapter. The
   cost is a published recipe change, not an invention.
2. Sparse attention as a policy. Our DiT attention is dense and O(n^2); on the
   20-step 768p job it is ~192 s. Sol-Attn's claim is 3.1x on the kernel versus
   FA4, on-the-fly and training-free. Porting the selection-and-correction
   algorithm into our CUDA attention kernel is self-contained work, and it is
   the only lever here that needs no new weights.

## Constraints that have to be settled before any of it ships

- cuDNN. Their Stage1 draft attention runs on cuDNN BSA. This campaign's Phase 5
  removed cuDNN by operator decision and says not to propose it. Porting their
  VSA stage as-is reopens that decision; porting the *policy* into our own
  kernel does not.
- Their runtime is Python and needs three separate environments (Qwen in a
  Docker image, Stage1 with patched FastVideo and its kernel, Stage2 with LTX
  and its compiled `all2all_cpp`), plus FA4 with its own CuTe, ComfyUI 0.30.0,
  and pinned gated weights (LTX-2.5). A clean-environment install is not
  validated by them either (`docs/validation.md`: "Installation from public
  dependency pins in a clean runtime" is listed as required before release
  promotion).
- Licensing. LTX-2.5 is gated; the LBH upscaler's weights "do not declare an
  explicit license" (`docs/setup.md`); the H3 LoRA weights inherit the H3
  community licence. Nothing here is redistributable as-is.
- The H3 revision they pin, `9bfb6693f2cf6de171db46d1aa586f67d773a1da`, differs
  from the revision this repo's README records, `42ed227e`. Reconciling that is
  a precondition for any weight-level work.

## What "as an option" can mean, in the only three shapes that are honest

A. Reference only. Read their runtime and kernels as the reference
   implementation for sparse-attention selection and for the latent-transfer
   geometry, and implement our own in Rust/CUDA. No new environments, no new
   dependencies. This is what rule 0.2 requires anyway: the reference is read
   first, and the port mirrors it with citations.
B. Port specific techniques into this engine, one at a time, each with its own
   gate: sparse attention policy (no weights), few-step LoRA plus FP8 (weights
   and a quantizer), latent transfer (weights and new kernels). Each changes
   the released output and needs the render gate and the still-missing noise
   floor.
C. Run their pipeline as a separate engine and expose it beside ours. This is
   an integration project: the three environments, the gated weights, a cuDNN
   reversal, and a second output contract. It is not a branch of this repo's
   kernel work.

## Phases on branch `feat/sol-h3`

Phase A, read the reference (no GPU). Extract the Sol-Attn selection and
correction algorithm from `techniques/sparse_backends/sol_attn` (the `sm120`,
`sm100` kernels, `triton_ref` and `_vendor/flash_attn/cute/block_sparsity.py`)
and the Stage1 VSA policy from `runtime/stage1_ops/vsa.py` into a spec with
file:line citations: block size, threshold rule, sink rule, the pooled-K/V
correction, and what makes it training-free. Deliverable: a section in this
file, not code.

Phase B, cost the sparse diversion in our kernel. Read our
`h3d_attention_mma_kernel` against that spec and state what a
selection-and-correction path would add per tile (extra QK pass, block masks,
the correction term) and which of our gates it would move. Deliverable: a
measurement plan and, if it looks sound, a reproducer in the shape of
`tests/h3/attn_bm.cu`.

Phase C, decide with the operator which lever to implement first, from the
three in B above, with the numbers from Phase B and the LoRA/FP8 options
costed. No implementation before that decision.

Phase D, implement the chosen lever with its own gate, in the campaign's
existing shape: a reproducer, a bit-identity or deviation measurement, a render
gate, and a record in `docs/minimax-h3/`.

What would falsify this plan: if the sparse-attention gain they report depends
on the FastH3 VSA LoRA being loaded, then the policy alone will not reproduce
it and the lever becomes "distilled LoRA plus sparsity" - a weights-level
decision, not a kernel one. Their Stage1 loads both, and `vsa.py` refuses a
non-VSA policy without the matching profile, so this is the first thing Phase A
must establish from the source.
