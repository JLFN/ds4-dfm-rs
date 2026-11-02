# The Gemma4 encoder path leaves the tree, 2026-11-02

The H3 render encodes its prompt with Qwen3-VL (`minimax_h3::text`,
`text_forward`, `native/h3/h3_text_gpu.cu`). The Gemma4 tower, its AV
connector and its tokenizer were ported for the LTX-2.5 recipe, and no H3
route reached any of them: `engine`, `render`, `jobs`, `media`,
`latent_upscale`, `prompt_cache` and the CLI name none of the three. Their
only callers were the two gates and the two benches that were removed with
them.

## What left

| Artefact | Lines deleted |
|---|---|
| `minimax_h3/gemma.rs`, `gemma_connector.rs`, `gemma_token.rs` | 2,596 |
| `examples/h3_gemma_gate.rs`, `h3_gemma_connector_gate.rs` | 2,477 |
| `tests/h3/gemma_tower_bm.cu`, `int8_convrot_bm.cu`, `test_int8_convrot.rs` | 1,734 |
| `native/h3/h3_dit_gpu.cu` (1,201) + `.h` (279) | 1,480 |
| `crates/ds4-sys/src/h3_dit.rs` (externs, wrappers, plan structs) | 982 |
| Eight `docs/minimax-h3/gemma*` records | 1,428 |
| Module, Cargo, Makefile and index lines | 34 |

Commit `70466247`, 23 files, 10,731 deletions and 19 insertions - the
insertions are the dtype renumbering and the two rewritten comments. The
removed code stays in history at `10a9484a`.

The INT8 convrot linear went with the tower, because the tower was its only
consumer - the header's own comment said so. `H3D_I8` and `H3D_I32` left the
ABI enum with it and `H3D_F16` moved from 6 to 4; nothing persists these
codes, and `h3d_alloc` still refuses a dtype it does not know.

Three things stayed on purpose:

- `h3d_rms_norm`'s optional weight. The DiT calls the weighted form, and the
  reference's `with_scale=False` builds carry none, so the branch is reachable
  without the tower.
- The FP8 and F16 dtype routes, which are the DiT's and the resizer's.
- `prompt_cache`. Its `gemma_int8_linears` field is the reference's own
  recipe fingerprint - a golden value to hash and compare, not this port's
  tower - and its gate is separate from the connector's.

## The evidence

Built on the Spark at `70466247`: `make -j8 CUDA_ARCH=sm_121` for `h3-rs` and
the five gates that survived the cut (`text`, `dit`, `prompt_cache`, `engine`,
`upscale`), exit 0 (`~/ds4-test/gemma-removal-build.log`).

The DiT gate reproduces the pinned hashes exactly, on the same prompt and
shape the attention record used (`~/ds4-test/gate-ab.sh`):

| | recorded | at `70466247` |
|---|---|---|
| `TEXT fnv` | e389ad74da16deb8 | **e389ad74da16deb8** |
| `LAYOUT positions` | b80b23d417768272 | **b80b23d417768272** |
| `FORWARD video=` | cb7bbf34526896b8 | **cb7bbf34526896b8** |
| `FORWARD audio=` | e10227686629592c | **e10227686629592c** |
| `video_edge` | 1.148437500 / -1.640625000 | 1.148437500 / -1.640625000 |

`h3_engine_gate` passes its own two-render check: `RESULT PASS residency`,
`MEDIA identical=true`, video `88ea05b50c4f1131`, audio `3945a4b4ecd9b28c`,
frames `1ad7a65c4d74fdec` (`~/ds4-test/gemma-removal-engine.log`).

So this is a deletion with no numerics in it, at the same shapes the records
pinned before it.

## Reproducing

    make -j8 CUDA_ARCH=sm_121 h3-rs tests/h3_engine_gate tests/h3_dit_gate
    M=/mnt/h3-artifacts/MiniMax-H3-FastH3-4step/FL2VA
    ./tests/h3_dit_gate "$M/text_encoder" "$M/tokenizer/tokenizer.json" \
        "$M/transformer" "A corgi in a chef hat flipping a pancake"
    H3_MODEL=/mnt/h3-artifacts/MiniMax-H3-FastH3-4step ./tests/h3_engine_gate

The first command prints the four hashes above; the second prints
`RESULT PASS residency`.
