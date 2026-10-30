# P0: the LTX attention layout - the port fed `h3d_sdpa` operands its contract does not accept

Unit 5b. It was not planned. It came out of the first read of the refiner's
pinned block for unit 6 (LTX's `PytorchAttention`), which is the read that
exposed it: the port's dense attention for the LTX path had been reading
token-major operands as if they were head-major since unit 4, so **every value
the AV connector produced was wrong**, and the cache unit 5 wrote carries
those values. This record states the defect, the measurement that proves it,
the fix, and what it invalidates.

## 1. What was wrong: three facts, each read first-hand

1. **`h3d_sdpa` reads its operands head-major.** The tiled kernel's query base
   is `((size_t)head * sequence + row) * HEAD_DIM`
   (`native/h3/h3_dit_gpu.cu:3047`, the `h3d_attention_tiled_kernel` body), and
   the key/value staging uses the same `[head][sequence][dim]` stride
   (`:3063`). The header says it outright: "head-major input, row-major output"
   (`native/h3/h3_dit_gpu.h:567-573`). The port's own test pins it: the
   `value` tensor in `tests/h3/test_native.cu:270-271` is written
   `source[(head * rows + row) * width + dim]` and the expected output
   `expected[(row * heads + head) * width + dim]`. That convention is the H3
   engine's, and `h3d_qkv_rope` is what writes it: the kernel's own store is
   `((size_t)head * sequence + row) * head_dim` (`:2837`) and the qkv it reads
   is `[row][head][3][dim]` (`:2801-2803`).
2. **The LTX path has no such writer.** `to_q`/`to_k`/`to_v` are plain
   `torch.nn.Linear`s (`attention.py:508-510`), so their output is
   token-major, and `PytorchAttention.__call__` reads it that way:
   `q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in
   (q, k, v))` (`attention.py:87`), SDPA, then `.transpose(1, 2).reshape(b, -1,
   heads * dim)` (`:101`). Confirmed for the second attention backend as well
   (`MPSSdpaAttention`, `:133-135`), so this is the reference's convention and
   not a backend quirk.
3. **`gemma_connector` handed the token-major projections straight to
   `h3d_sdpa`.** `Block::apply` runs `to_q`/`to_k`/`to_v` into
   `scratch.{query,key,value}` (`gemma_connector.rs:508-512`), normalizes them
   per row over the whole inner dim (`:516-533`), rotates them in place with
   `h3d_ltx_rope`, whose own index is row-major
   (`((size_t)row * heads + head) * head_dim`, `h3_dit_gpu.cu:1988`), and then
   called `sdpa` with those same buffers (`:552-563`).

So for each block, each head, each row, the attention gathered its keys and
values from **another head's rows**: for head `h` and row `r` the kernel read
flat offset `(h * rows + r) * head_dim`, which under the real
`(row, head, dim)` layout is row `h * heads + r / head_dim`'s head
`r % head_dim`. The result is finite and plausible-looking, which is why
nothing failed loudly.

## 2. The measurement

The connector gate's `--probe` mode now transcribes
`PytorchAttention.__call__` on the host (`host_attention`,
`h3_gemma_connector_gate.rs`) and feeds it token-major operands through the
same call the graph makes. Against the old `gpu.sdpa`:

    PROBE attention_video values=32768 differ=32768 rel_rms=1.413e0 verdict=residual FAIL
    PROBE attention_audio values=16384 differ=16384 rel_rms=1.419e0 verdict=residual FAIL

Every element differed, at 1.41 relative RMS - 70x the gate's 2e-2 ceiling,
i.e. the output is uncorrelated with the correct attention rather than merely
imprecise. With `h3d_ltx_attention`, on both hosts, the same probe reads:

| host | `attention_video` | `attention_audio` |
|---|---|---|
| sm_89 (RTX 4070 SUPER) | `differ=32768 rel_rms=1.691e-3` | `differ=16384 rel_rms=1.679e-3` |
| sm_121 (GB10) | `differ=32768 rel_rms=1.691e-3` | `differ=16384 rel_rms=1.679e-3` |

Bit-identical residuals across the two hosts, because both sides of that
number are this port's own arithmetic. The residue is below the BF16 ulp
(`2^-8` = 3.9e-3), so most elements agree to the last bit and the differing
ones are one ulp apart: the device's shuffle-tree reduction against the host's
sequential sum.

This is the decisive form of the bug report: not "the layouts disagree"
(a code-read argument) but "the call the graph makes disagrees with the
reference's own expression, and here is the number".

## 3. What it changed, and what it did not

The fix changes **exactly** the two stages downstream of the attention. The
extractor has no attention, so its stages must not move, and they do not:

| stage | before | after (sm_89) | after (sm_121) |
|---|---|---|---|
| `scales` | `f2231100c4002dff` | `f2231100c4002dff` | `f2231100c4002dff` |
| `normed` | `9629e928b5d6488f` | `9629e928b5d6488f` | `9629e928b5d6488f` |
| `video_features` | `3fd024002cd20995` | `3fd024002cd20995` | `45af76b34d01d99b` |
| `audio_features` | `4a88cdca0a6595fa` | `4a88cdca0a6595fa` | `9446ac95a33e3478` |
| `video` | `4cff505ef3e46d68` | `12a96c2aa9a100c9` | `51496cc78b9dcfb3` |
| `audio` | `b9d09900a73cd156` | `b3bf2a37fdff8f18` | `81434dd95e5a0f9b` |

128 synthetic states, seed 1. The last two rows are the fix; the
`video_features`/`audio_features` split across hosts is unit 4's known BF16
linear non-stability, unchanged.

The prompt path (Spark, 1024 rows, `'A golden retriever running on a beach at
sunset.'`), the same run the unit-5 record quotes:

| stage | before | after |
|---|---|---|
| `scales` | `fb74790f4038ca20` | `fb74790f4038ca20` |
| `normed` | `bf20f75e05b3705a` | `bf20f75e05b3705a` |
| `video_features` | `41183d7bfb7cceb3` | `41183d7bfb7cceb3` |
| `audio_features` | `9aed0d4298a554fc` | `9aed0d4298a554fc` |
| `video` | `dc2e26ddc23e46ab` | `7ad2eebda1b86a97` |
| `audio` | `76457d4d47d3785e` | `6398937749953938` |

**The unit-5 cache is superseded.** Its two contexts were written from the
wrong values: video `sha256 91816f40...` is now `6b44ac18...` and audio
`5e1a67b8...` is now `f412760a...` (files read back with
`h3_prompt_cache_gate --inspect`). The fingerprint is unchanged -
`79ca00d1518977c098075f01ee57e3650b731d63a055c6cab83aa45a76a364e2` is a
function of the prompt alone, so unit 5's container and contract work is not
in question; only the payload's values were. The superseded file is kept on
the Spark as `~/ds4-test/prompt-cache-buggy.pt`, and the corrected one is
`~/ds4-test/prompt-cache.pt`.

Everything else unit 5 landed stands: the container, the fingerprint path, the
twelve validation rules, and the gate are untouched by this fix and their
evidence still holds.

## 4. The fix

`h3d_ltx_attention` (`native/h3/h3_dit_gpu.{h,cu}`, wrapper
`ds4_sys::h3_dit::Gpu::ltx_attention`) is
`h3d_attention_tiled_kernel` with two changes:

- the operand addresses are token-major, `((size_t)row * heads + head) *
  HEAD_DIM` for the query and `((size_t)source * heads + head) * HEAD_DIM` for
  the keys and values, which is `PytorchAttention.__call__`'s own reading of
  `(b, T, heads, dim)`;
- the query and key row counts are independent arguments.

The second is not incidental to this fix: the refiner's A2V and V2A
cross-attentions are rectangular (16,128 video queries against 126 audio
keys, and the reverse), and `h3d_sdpa`'s single `sequence` cannot express
them. Nothing else differs - the online softmax, the `KEYS`-row staging and
the output addressing are the ones the port already had, so the LTX path and
the H3 path now differ only in their operand layout, which is a fact about
each engine.

`gemma_connector` calls it, and the connector gate's probe covers both
geometries (128 and 64 wide) so the convention cannot silently regress.

Named deviation, measured: the reference runs SDPA, whose reduction order is
torch's and whose backend on the Spark's sm_121 is picked by
`torch.backends.cuda` (`_select_primary_attention`, `attention.py:270-289`);
this port runs its own tiled kernel. The gate above measures that difference
at 1.7e-3, below the BF16 ulp.

## 5. The lesson, and the corrections it forces

**A probe set that omits an op cannot catch that op.** Unit 4's `--probe`
transcribed five of the connector's operations - `stack_scale`, `interleave`,
`gather_rows`, `register_fill`, `rope`, `per_head_gate` - and every one of
them reproduced the port's reduction order bit for bit, which read as strong
evidence. The one operation it did not transcribe is the one whose kernel
carried a convention from a *different engine*. The gate was only as good as
its coverage, and its coverage was chosen by "what did this unit add", not by
"where can the port's assumptions diverge from the reference's".

**A cross-host comparison cannot catch a shared assumption.** Both hosts
agreed on every stage they were compared on, to the element, which unit 4
recorded as the finding of the round. They agreed because they ran the same
wrong kernel. The two-host discipline catches architecture-dependent
arithmetic; it cannot catch a layout the port chose for both hosts at once.

**Corrections to earlier records.** They are superseded in exactly one respect
each, and this is the authoritative version of both tables:

- `gemma-connector-2026-10-30.md` §5's stage hashes and §6's cross-host table
  carry the pre-fix values for `video` and `audio` (and for the prompt path).
  Its §6 conclusion stands in shape - the BF16 linear is not bit-stable across
  hosts - with the corrected hashes above.
- `prompt-cache-2026-10-30.md`'s two context sha256 values are the superseded
  ones; everything it says about the container, the fingerprint and the
  validation rules is unaffected.

**A claim this record withdraws.** The handoff's "the conditioning chain now
exists end to end, as a file" was true of the plumbing and false of the
values: from unit 4 until this fix the chain computed wrong contexts. Nothing
downstream of it had been built, which is the only reason this was cheap to
find; had unit 6's refiner been built against a cache, the cache would have
been the thing blamed.

## 6. Evidence and reproduction

Workstation (sm_89, CUDA 13.3) and Spark (sm_121, CUDA 13.0), commit
`4f8f4ee4`:

    rm -f native/h3/*.o tests/h3_gemma_connector_gate
    make CUDA_ARCH=sm_89 tests/h3_gemma_connector_gate    # sm_121 on the Spark
    tests/h3_gemma_connector_gate --probe

The failing form of the same probe (the old `gpu.sdpa` call) is reproducible
by reverting `gemma_connector.rs:552` and the probe's `ltx_attention` call to
`sdpa`; it is the state commit `3fc7d261` is in.

Workstation connector run, over the gvfs mount (49 synthetic states cannot
run without both shards; 128 rows, seed 1, `load=51.049 run=0.091`,
`GEMMA-CONNECTOR-OK`). Spark: `--rows 128 --seed 1` at its own disk speed
(`load=1.231 run=0.232`), then the prompt path
(`load=1.376 run=0.685`), which needs the INT8 Gemma file because the tower's
reader requires its `comfy_quant` keys:

    tests/h3_gemma_connector_gate \
      --gemma /mnt/h3-artifacts/Gemma4/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors \
      --connector /mnt/h3-artifacts/LTX-2.5/ltx-2.5-22b-dev-transformer-bf16.safetensors \
      --prompt 'A golden retriever running on a beach at sunset.' \
      --cache ~/ds4-test/prompt-cache.pt

`cargo test -p ds4-core --lib`: 746 passed, 7 ignored. OSV on the lockfile:
140 packages, 0 findings (2026-10-30).

One artifact fact read for unit 6 while writing this, because the plan had
assumed the opposite: `keyframes_abs_pos_embedding` is **not** zero. The
tensor's own 8192 bytes (`data_offsets [1576162048, 1576170240]` in the
transformer shard, read directly with `tail -c` and inspected with `od -An
-tx2`) hold 1,398 distinct 16-bit words and only 15 NUL bytes, so the keyframe
add is a real term on the first latent frame's 1,008 tokens, not the no-op the
initialization comment (`model.py:217-218`) suggests.
