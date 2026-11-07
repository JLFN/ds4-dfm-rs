# The server's steady state: 88.99 s per render

Job A throughout: 768x768, 107 frames, seed 42, the prompt "a red fox walking
through deep snow at sunrise, cinematic", the fast arm (derived checkpoint
`MiniMax-H3-FastH3-4step`, four steps, `H3_DIT_SOL=0.9`, the fast decoder
`H3_VAE_ATTENTION=tensor H3_VAE_GEMM=bf16`), one binary, `sm_121a`, 2.4-2.5 GHz
(the host has no clock policy). `h3-server-rs` from `ab87a1e7` on the Spark,
which is the commit the CLI arms below were measured on too.

The campaign had no two-render number: every recorded wall paid `TEXT` and
`LOAD` again, and `LOAD` moves 9 to 52 s with the host's page cache alone, which
made the walls incomparable across rows. This is that measurement. It also
re-ranks the plan: with both cold phases served from cache, the two terms that
decide the wall are `DENOISE` and `FRAMES`.

## 1. What was measured

Two renders per process, two processes, one per encoder arm: `H3_TEXT_NVFP4`
naming the reference's NVFP4-AWQ artifact, then unset for the released BF16
shards. The engine holds the prompt embedding and the prepared DiT
(`crates/ds4-core/src/minimax_h3/engine.rs:154-158`), so the second render in a
process should pay neither phase. Harness `~/phaseA.sh` (md5
`9cb042bd05ca8342ed785ad770098ef8` on both sides); the request body is the
contract's square 4-second job (`target.short_edge` 768, `aspect_ratio` 1:1,
`duration_seconds` 4), which `frames_from_seconds` resolves to 107.

The host had just finished the NVFP4 arm when the BF16 arm started, so the
second arm's DiT read was partly cold: 42.956 s of `LOAD` against the other
arm's 9.359 s, with no code difference. That is the campaign's known
`LOAD`-moves-with-cache-memory effect, and it is why the pair to compare is the
two renders inside one process, not the two arms' first renders.

| arm | render | wall | `TEXT` | `LOAD` | `DENOISE` | `AUDIO` | `FRAMES` | `MUX` | MP4 sha256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| NVFP4 | 1 | 112.228 s | 10.619 | 9.359 | 49.906 | 2.585 | 37.478 | 0.674 | `5b5e735c...` |
| NVFP4 | 2 | **88.988 s** | 0.001 (hit) | 0.000 (hit) | 50.550 | 2.646 | 34.146 | 0.600 | `5b5e735c...` |
| BF16 | 1 | 165.143 s | 29.069 | 42.956 | 50.148 | 2.758 | 38.271 | 0.685 | `093d6fcd...` |
| BF16 | 2 | **93.088 s** | 0.002 (hit) | 0.000 (hit) | 50.475 | 2.660 | 38.240 | 0.580 | `093d6fcd...` |

The phase lines are the server's own (`crates/ds4-server/src/h3_engine.rs:108`,
`RenderReport::summary`, `engine.rs:688-703`); the wall is the client's
POST-to-download, so it exceeds the phase sum by 1.1-1.6 s. Both arms report
`frames=107 evaluated=4` and `scale=1.0000`.

## 2. What it establishes

- **The second render pays neither cold phase.** `TEXT` falls to 1-2 ms and
  `LOAD` to 0.000 s, both flagged as hits. The steady state is 88.99 s on the
  quantized arm and 93.09 s on the released one.
- **The server path reproduces the CLI render byte for byte.** Each arm's two
  renders are the same file, and that file is its CLI arm's recorded MP4:
  `5b5e735c...` for NVFP4 and `093d6fcd...` for BF16
  (`nvfp4-encoder-2026-10-28.md:116-118`). The two arms differ from each other
  because the quantized encoder gives a different prompt embedding, which is the
  change under test in that record, not a server-versus-CLI difference.
- **The encoder choice no longer decides the wall.** At steady state the two
  arms sit 4.1 s apart, and that is `FRAMES`' own spread (34.146 against 38.240
  on the same binary and the same flags), not `TEXT`: the phase that the encoder
  switch moves is the one both arms now skip.
- **The cold phases are 20.0 s of the quantized arm's first render and 72.0 s of
  the released one's**, the difference being the 62 GiB BF16 encoder read
  evicting the DiT's pages. Both are paid once per process, not once per render.

## 3. Where the steady-state wall is

| term | NVFP4 render 2 | BF16 render 2 | share |
|---|---:|---:|---:|
| `DENOISE` | 50.550 | 50.475 | 57% |
| `FRAMES` (video VAE decode) | 34.146 | 38.240 | 39% |
| `AUDIO` | 2.646 | 2.660 | 3% |
| `MUX` | 0.600 | 0.580 | 1% |

`DENOISE` is four evaluations, 12.6 s each across 50 blocks and 18,799 rows.
`FRAMES` decodes 107 frames through the tensor-core, BF16-GEMM decoder
(`vae-options-2026-10-28.md`). The 56 s the recon attributes to the reference's
project page is not a ratio over either number: it is a different canvas, frame
count, step count, model set and decoder, and the checkout does not claim it
(`sol-port-plan-2026-10-28.md:41-46`). Against our own split, both remaining
terms are per-evaluation and per-frame work, so the next optimization is not
cold-read work.

## 4. Limits

- One render pair per arm on one host state: the walls are single runs, not
  repeat ranges. `FRAMES` moved 4.1 s between two renders with identical inputs,
  which bounds what a wall difference of that size can decide here.
- The audio path and the mux are unexamined; `AUDIO`'s 2.6 s is taken as
  recorded, not attributed.
- Nothing here says the fast arm's output equals the reference's: the arm and
  its quality instrument are the merge record's (`lora-merge-2026-10-28.md`,
  detail energy 3.1893 BF16 / 3.4338 NVFP4 on this job).
