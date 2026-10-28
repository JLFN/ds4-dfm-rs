# sol — NVIDIA's Sol-H3 pipeline, as a separate engine

This directory holds the operator's Sol-H3 route: NVIDIA's two-stage
draft-and-refine pipeline for MiniMax-H3 on one DGX Spark, pinned to a commit
and run from here as its own stack.

It shares no code with this repository's engine. Nothing under `crates/`,
`native/`, `tests/` or the Makefile is built, wrapped or replaced by it, and
nothing here enters `make h3-rs`. The Rust/CUDA path stays the default and
keeps its own gates; this tree is a second engine with its own environment,
its own weights and its own output contract.

Reference reading is in
[../docs/minimax-h3/sol-h3-spark-recon-2026-10-28.md](../docs/minimax-h3/sol-h3-spark-recon-2026-10-28.md):
what the pipeline is, its own reported split, why its headline number is not
comparable to this engine's, and the four constraints (cuDNN, the three Python
environments, gated and unlicensed weights, the H3 revision mismatch).

## What it is, in one paragraph

Stage 1 generates a 672x384 draft with MiniMax-H3 in W8A8 FP8 and a four-step
distilled LoRA, using VSA attention at 90% video sparsity whose selected blocks
run on cuDNN BSA. The draft latent is upscaled by a learned H3 x2 upscaler and
mapped into LTX latent space by a learned adapter, so no pixels are decoded
between stages. Stage 2 refines with LTX-2.5 and three Sol-Attn steps, then
decodes with the LTX Conv VAE. NVIDIA reports 56.17 s hot end to end at
1344x768, 121 frames, with 35.2% of it in the H3 stage and 44.0% in the LTX
refiner.

## What this tree is not

- Not a dependency of the engine, and not on any gate path.
- Not a faster route to the same frame. It produces a 768p result through a
  different model family and decoder, so this repository's render gate and its
  "matches the reference" guarantee describe the other path, not this one.
- Not redistributable as a bundle: LTX-2.5 is gated, the H3 upscaler's weights
  declare no licence, and ComfyUI is GPL-3.0. Nothing here is vendored into
  this repository; `prepare.sh` fetches third-party sources into a gitignored
  directory.

## Layout

    sol/
      README.md        this file
      prepare.sh       pin and fetch the upstream sources, build the venv, pin cuDNN
      run.sh           run one request through the upstream entry point
      upstream/        third-party checkout, fetched at a pinned commit (ignored)
      venvs/           Python environments (ignored)
      checkpoints/     weights (ignored)
      outputs/         generated media and per-request receipts (ignored)
      paths.json       the upstream path manifest (ignored; written by the operator)

## Requirements, and what is actually needed

- Linux aarch64, one GB10/SM121 GPU, and its full memory. NVIDIA reports a
  resident footprint of 116.9 GiB of the Spark's 119.68 GiB pool, so nothing
  else may share the GPU.
- CUDA 13.x. The engine host has 13.0.88; the CUDA 13 build of cuDNN runs
  beside it without changing the toolkit (NVIDIA's cuDNN support matrix lists
  CUDA 13.0 among the supported toolkits for the CUDA 13.x package).
- cuDNN is needed only here, for Stage 1's BSA backend. The engine's own
  kernels do not use it. `prepare.sh` pins the version NVIDIA's Spark pipeline
  pins (`nvidia-cudnn-cu13==9.20.0.48`, frontend `1.27.0`), which they
  validated on this host; the current cuDNN release documents a newer minimum
  driver than this Spark runs, so do not float the version.
- Weights, downloaded separately and never committed: the H3 snapshot, the
  FastH3 VSA LoRA, the LTX-2.5 stack with its LoRA, the H3-to-LTX adapter, the
  H3 upscaler, and the offline INT8 Gemma and connector set for the cached
  prompt context. NVIDIA's `download_checkpoints.py --plan` lists them with
  pinned revisions and sha256s.

## Prepare

    ./sol/prepare.sh

It is idempotent and does only what is mechanical: pin and fetch the upstream
sources at the commit this tree was written against, create the venv, and
install the pinned cuDNN packages into it. It then prints the steps that
cannot be automated here, which are the ones that need the operator:

1. Accept the LTX-2.5 access terms and `hf auth login` on this host.
2. Download the weights (`download_checkpoints.py --output-dir checkpoints
   --include-offline`).
3. Build the three upstream environments as their `docs/setup.md` describes
   (Stage 1 with patched FastVideo and `fastvideo-kernel`; Stage 2 with LTX and
   its `all2all_cpp`; Qwen in its Docker image).
4. Write `paths.json` with `prepare.py` and build the offline prompt cache with
   `runtime.cache_builder`.

None of those four has been run from this repository. Upstream's own validation
records that a clean environment install is not yet validated by them either
(`docs/validation.md`).

## Run

    ./sol/run.sh "a red fox walks through a sunlit meadow" 42 outputs/fox

The first request of a session is not warm; upstream's timing contract counts
the first formal request and excludes only its explicit warmup. Run one GPU job
at a time, and prefer the repository's host memory guard
(`tools/host_memory_guard.py`) around it: this pipeline is designed to fill the
pool.

## Pinning

`prepare.sh` records `SOL_UPSTREAM_COMMIT` and fetches exactly that commit
rather than the branch tip, so a later upstream commit cannot change what this
directory runs without an explicit edit and a commit here. The H3 checkpoint
revision NVIDIA pins (`9bfb6693...`) differs from the one this repository's H3
port records (`42ed227e...`); that mismatch is noted in the recon and is a
precondition for any weight-level comparison between the two engines.
