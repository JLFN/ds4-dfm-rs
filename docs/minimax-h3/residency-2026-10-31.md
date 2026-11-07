# Residency: the second request pays neither the prompt nor the load (2026-10-31)

Phase 2 of `docs/minimax-h3/plan-2026-10-31.md`. The resident engine already
existed: `ds4_core::minimax_h3::engine::EngineHandle` keeps the prompt
embedding and the prepared DiT across requests (`engine.rs:1-14`, the caches at
`:234-238`), owns a thread because the device handles are `!Send`, and reports
per-phase seconds with `text_hit` and `prepared_hit` (`:73-86`,
`:688-700`). `crates/ds4-server/src/h3_engine.rs` already wraps it as the
`Renderer` trait. What did not exist was a measurement of the reuse.

## 1. The gate

`crates/ds4-core/examples/h3_engine_gate.rs`, built as `tests/h3_engine_gate`
(`H3_GATES` in the Makefile, with its own object line: it needs the vision
object because the engine reaches the vision encoder through `ds4-sys`). The
checkpoint is named by the environment and nothing else - `H3_MODEL`, or
`H3_ARTIFACTS` plus `H3_MODEL_NAME`; unset or absent, the gate exits 3, which
`make test-h3` reads as a missing artifact.

One `EngineHandle` renders the same request twice. The second render must hit
both caches and both renders must produce the same media hashes, because
residency is worth nothing if it changes the clip. Canvas 512x512, 22 frames
(`5 + 17k`), 4 steps, seed 42, and the adopted stage 1 (`H3_DIT_GEMM=fp8`,
`H3_DIT_SOL=0.9`, NVFP4 prompt encoder); decoder `Tensor`/`Reduced` from the
request, not from the environment.

## 2. Measured, two runs

| phase | run 1 req 1 | run 1 req 2 | run 2 req 1 | run 2 req 2 |
|---|---:|---:|---:|---:|
| text | 12.309 (miss) | 0.000 (hit) | 12.313 (miss) | 0.000 (hit) |
| load | 8.777 (miss) | 0.000 (hit) | 8.190 (miss) | 0.000 (hit) |
| denoise | 2.693 | 2.687 | 2.706 | 2.720 |
| audio | 0.634 | 0.644 | 0.630 | 0.637 |
| video decode | 12.467 | 12.467 | 12.441 | 12.479 |
| mux | 0.253 | 0.245 | 0.237 | 0.247 |
| wall | 37.196 | 16.107 | 36.579 | 16.146 |

Media identical in both requests and both runs: video `c63efab051142d5d`,
audio `617d9a1dd4c19425`, frames `c1e6ad50eda648cc`. The gate prints
`RESULT PASS residency`.

## 3. What it means

Residency removes the two largest fixed costs: 21.1 s of a 37.2 s first
request at this canvas, 57%. At the product canvas those same two phases
measured 15.696 s (187-token prompt) and 28.974 s (cold load), so the product
clip's 133.6 s becomes about 89 s. That is an estimate until the
product-shaped run exists.

With both fixed costs gone the VAE decode is 12.5 s of a 16.1 s warm request,
77%. Phase 3 is now argued by measurement and not by the page's 12.5% share:
the untiled decoder is the whole cost of a resident request at this canvas.

## 4. Open

- The gate is a probe, not the product. `h3-rs` still renders one clip per
  process and does not drive the engine. The operator's CLI-only constraint
  makes a resident CLI mode (N requests in one process) the product face, not
  the HTTP server.
- A product-shaped run through the engine (672x384x124 into the 1344x768
  upscale and decode) replaces the 89 s estimate with a measurement, and its
  hashes say whether the engine path equals the one-shot CLI path.
- The decoder at this canvas is 12.4-12.5 s for 22 frames of 512x512; Phase 3
  applies to it as much as to the release canvas.
