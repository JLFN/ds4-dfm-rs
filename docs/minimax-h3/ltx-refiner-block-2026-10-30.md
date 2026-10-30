# Unit 6: the LTX-2.5 refiner's block, ported and gated

The stage-2 refiner's transformer body - the piece the whole graph is 48 copies
of - is now in the port, on the shard's own weights, with the block's residual
structure pinned separately from its arithmetic. What is NOT in this unit is
named in section 6: the preprocessing, the output head, the sampler and the Sol
policy.

Source of truth read first-hand this unit: `/data/ltx2-pin/packages` (the pin
the release runs, `d151147788a9284cca791edc6ce898007e727fe6` = tag v1.2.0),
`ltx-core/src/ltx_core/model/transformer/`: `transformer.py`, `ops.py`,
`attention.py`, `feed_forward.py`, `gelu_approx.py`, `adaln.py`,
`transformer_args.py`, `modality.py`. Geometry from the shard's own
`__metadata__.config.transformer`, not from the constructor.

## 1. What landed

| file | what |
|---|---|
| `crates/ds4-core/src/minimax_h3/ltx_refiner.rs` | the block, the 48-block loop, the scratch, the loader |
| `crates/ds4-core/examples/h3_ltx_refiner_gate.rs` | the gate's four modes |
| `native/h3/h3_dit_gpu.{cu,h}` | seven kernels (section 3) |
| `crates/ds4-sys/src/h3_dit.rs` | the seven safe wrappers |
| `Makefile`, `crates/ds4-core/Cargo.toml` | the gate target and its `h3-gates` example entry |

Commit `9293fc67` on `feat/sol-h3`, pushed to `fork`.

## 2. The block, in the reference's own order

`BasicAVTransformerBlock` (`transformer.py:87-190`), replicated 48 times with no
per-layer variation (`model.py:407-417`). One block is 84 tensors: six attention
modules x 12, `ff` 2 bias-free, `audio_ff` 4 biased, six F32 tables. The
forward (`transformer.py:253-417`):

1. video self-attention, rows 0..3 of the 9-row table (`:271-285`);
2. video text cross-attention, rows 6..8 on the query and the 2-row prompt table
   on the context, **no RoPE** (`:288-298`, `:239-252`);
3. audio self-attention and audio text cross-attention, the same two steps at
   width 2048 (`:300-326`);
4. A2V then V2A, both from the pre-A2V state (`:328-397`), the 5-row
   per-direction table with rows 0,1 for A2V and 2,3 for V2A and the gate on row
   4 (`:201-220`);
5. the two feed-forwards, rows 3..5, `mult` 4 (`:399-414`).

Four facts the port must not lose, each of which a "reasonable" simplification
would break:

- **Torch rounds BF16 elementwise ops one at a time.** `opmath_type<BFloat16>`
  is `float`, so each op computes in FP32 and rounds once, and every expression
  in the source is a chain of separate ops. `rms_norm(x) * (1 + scale) + shift`
  is four roundings, not one, and that chain is what `h3d_ada_zero` implements.
- **A2V and V2A both read the state before either mutates it**
  (`transformer.py:331-334`). Because `ada_zero` is pure, the port forms all
  four modulated operands before running either attention: the reference's
  function without its two 132 MB copies.
- **The AdaLN tables are F32 in the checkpoint** and the reference casts them to
  the timestep's BF16 before adding (`transformer.py:195-197`), so `Reader::table`
  reads F32 and rounds once at load; the block's add runs on BF16 operands.
- **The per-head gate reads the module's own input**, not the context, and is
  applied *before* `to_out`, which is therefore ungated (`attention.py:514`,
  `:576-582`), and the text cross-attention's own result is gated by the query
  modulation's third value (`transformer.py:447`).

## 3. The seven kernels

Each is the reference's own expression in the reference's own rounding order.

| kernel | reference | roundings |
|---|---|---|
| `h3d_ada_add` | `get_ada_values` (`transformer.py:190-200`) | one add, one rounding |
| `h3d_modulate` | `x * (1 + scale) + shift` (`:445-446`, `model.py:487-488`) | `1+scale`, the product, the sum |
| `h3d_mul_rows` | `attn_out * gate_out` (`:362`, `:447`) | one product |
| `h3d_ada_zero` | `rms_norm(x) * (1 + scale) + shift` (`ops.py:50-58`) | the norm, `1+scale`, the product, the sum |
| `h3d_layer_norm` | `LayerNorm(elementwise_affine=False)` (`model.py:231`) | one, after a biased-variance reduction |
| `h3d_ltx_rope_pair` | `apply_rotary_emb` on two tables (`ops.py:22-36`) | two per half, as `addcmul_` does |
| `h3d_channel_transpose` | the patchifier at patch size 1 (`patchifiers.py:27-64`) | none, a permutation |

`h3d_ada_add` exists because the modulation is `[rows, coefficient, width]` -
the shape `timestep.reshape(batch, T, num_ada_params, -1)` produces - so one
coefficient row is strided and cannot be handed to `h3d_add`.

Reused unchanged: `h3d_linear`, `h3d_gelu_tanh`, `h3d_add`, `h3d_mul`,
`h3d_rms_norm` (its weight covers `q_norm`/`k_norm`), `h3d_ltx_rope`,
`h3d_per_head_gate`, `h3d_ltx_attention` (all six attentions, the AV pair using
its independent query/key row counts).

## 4. The measurements

The gate has four modes; every figure below is a command in section 7.

### 4.1 Probes: all eleven bit-identical on both hosts

The `--probe` mode transcribes each new kernel **and** the two broadcast cases
of `modulate`/`mul_rows` on the host in the kernel's own rounding order. Eleven
probes, `differ=0`, `rel_rms=0.000e0` on both hosts - including `ada_zero` and
`layer_norm`, whose reference ends in `rsqrtf`: at these magnitudes the
approximation does not move a rounded value. A wiring fault would move every
element by two orders more.

### 4.2 The block on the shard's own weights

| host | rows | state | fnv1a | high / low | block time |
|---|---|---|---|---|---|
| sm_89 | 8 / 4 | video | `ea9a5781399daede` | 458 / -212 | 0.058 s |
| sm_89 | 8 / 4 | audio | `8444306cc7515bdb` | 540 / -728 | 0.058 s |
| sm_121 | 8 / 4 | video | `f95c2450ea674251` | 458 / -212 | 0.175 s |
| sm_121 | 8 / 4 | audio | `dee23161773fc3d8` | 540 / -728 | 0.175 s |

The hashes differ across hosts and the extremes do not: `h3d_linear` is
cuBLASLt, whose accumulation order is architecture-specific (unit 5's finding),
while the state's extrema are set by the large-magnitude path. The 8-row hash is
reproducible on its own host across rebuilds.

Load: **84 tensors, 773,546,368 bytes per block** - 0.16 s from the Spark's own
disk, 6.1-8.2 s over the workstation's gvfs mount. 48 blocks are 37.1 GB, which
is why the gate has `--blocks` and the workstation runs one.

Two blocks chained on the same scratch: block 1's state hashes differ from block
0's on both hosts (`b17d26090d24fa61` / `12333cb535a714d4` on the video), which is
the loop's own check that the second block sees the first one's output, and
`--blocks 2` reproduces `--blocks 1` block 0 hash for hash.

### 4.3 The residual structure, exactly

`--identity` writes a shard whose six `to_out` projections and two
feed-forward outputs are zero (15 tensors per block) and requires the block to
return its input **bit for bit**: `x + y*gate` with `y = to_out(0) = 0` is `x`
exactly, and the same holds for both feed-forwards. Both hosts, both modalities:
`differ=0`. This is the check no probe can make - it is the only test of the
residual *wiring* (which buffer each term is added into, that the AV pair does
not write into the state it reads, that no branch is doubled).

### 4.4 The release's own geometry

`--rows 16128 --audio-rows 126` is the pipeline's real token counts.

| host | video fnv1a | high / low | block time |
|---|---|---|---|
| sm_89 | `4dbfb03078a82ff4` | 4832 / -4928 | 1.877 s |
| sm_121 | `dcb9686f096644a2` | 4832 / -4928 | 2.785 s |

Row scaling on the same host attributes the block: 4,032 rows 0.242 s, 8,064
0.641 s, 16,128 1.877 s. A two-term fit gives `0.0411 us*rows + 0.00467
us*rows^2`, i.e. the dense attention is **~65%** of the block at the release
geometry and the projections and feed-forwards the rest - so the optimization
priority for this stage is the attention path (and the Sol policy that removes
90% of it for 47 of 48 layers), not the GEMMs.

For the campaign's cost model: 1.88 s per block on the workstation and 2.79 s on
the Spark, dense at every layer, is 90 s and 134 s per evaluation against the
release's 8.2 s (24.71 s for three updates, with the Sol policy on GB10). The
gap is the whole of unit 7's and 8's work, and it is where the release's own
45%-of-peak figure says it should be.

### 4.5 The cross-host residual

The two hosts' dumps compared element-wise (the workstation's run against the
Spark's dump):

| rows | state | differ / values | rel_rms | ceiling |
|---|---|---|---|---|
| 8 | video | 178 / 32,768 | 1.040e-3 | 2e-2 |
| 8 | audio | 586 / 8,192 | 9.705e-4 | 2e-2 |
| 16,128 | video | 4,318,350 / 66,060,288 | 8.709e-4 | 2e-2 |
| 16,128 | audio | 12,508 / 258,048 | 7.488e-4 | 2e-2 |

One part in a thousand, on about 6% of the elements - the BF16 ulp of the
largest terms, which is what two different cuBLASLt accumulation orders must
differ by. This is the campaign's first cross-host *value* gate on this family
and it says the port's own arithmetic is host-independent to within the format;
everything above the linears is bit-identical (section 4.1).

## 5. What this unit does not prove

- **The gate's `Signals` are synthetic.** The modulation, gates, contexts and
  positional tables come from the gate's LCG, not from the reference, so the
  state's *values* above are not the pipeline's (they grow across blocks because
  the synthetic shifts do). What the run proves is the block's structure, its
  consumption of those tensors and its kernel set.
- **No golden dump exists yet.** Every gate this campaign has is a host
  transcription of the reference's own expression, which cannot catch a shared
  misreading of the reference's layout - that is exactly what unit 5b was. The
  owed decision stands: a dump from the pinned stage-2 environment is the only
  gate that can.
- **`--identity`'s exactness has one caveat**: `bf16(-0.0 + 0.0)` is `+0.0`, so
  a state holding a negative zero would fail the bit comparison. The LCG produces
  none, and the reference has the same property.
- **The block has never run on the real 48-layer configuration**; only up to two
  blocks, plus one block at the real row geometry. Loading all 48 needs 37.1 GB
  and a resident model, which is the pipeline unit's job.

## 6. What is still not ported of the refiner

- the preprocessing: patchify (the transpose kernel is in and gated), positions
  and the RoPE tables, the AdaLN MLPs, the keyframe term, the time-axis cross
  `pe`;
- the output head (the affine-free LayerNorm and the transpose are in and gated,
  the model-level table and `proj_out` are not);
- the sampler, the noiser and the state creation;
- the Sol-Attn dispatch policy and its backend.

## 7. Evidence and reproduction

```
# workstation, sm_89
make CUDA_ARCH=sm_89 tests/h3_ltx_refiner_gate
./tests/h3_ltx_refiner_gate --probe --rows 8 --seed 7
M=/run/user/1000/gvfs/sftp:host=192.168.1.91,user=leandro/mnt/h3-artifacts/LTX-2.5/ltx-2.5-22b-dev-transformer-bf16.safetensors
./tests/h3_ltx_refiner_gate --transformer $M --blocks 1 --rows 8 --audio-rows 4 --seed 7 --dump /tmp/refiner-a
./tests/h3_ltx_refiner_gate --transformer $M --identity /tmp/refiner-zero.safetensors --blocks 1 --rows 8 --audio-rows 4 --seed 7
./tests/h3_ltx_refiner_gate --transformer $M --blocks 1 --rows 16128 --audio-rows 126 --seed 7 --dump /tmp/refiner-full
./tests/h3_ltx_refiner_gate --transformer $M --blocks 1 --rows 8 --audio-rows 4 --seed 7 --residual /tmp/refiner-spark

# Spark, sm_121 (script kept at ~/ds4-test/unit6-spark.sh, log unit6-spark.log)
cd ~/ds4-sol-h3 && rm -f native/h3/*.o && make CUDA_ARCH=sm_121 tests/h3_ltx_refiner_gate
MODEL=/mnt/h3-artifacts/LTX-2.5/ltx-2.5-22b-dev-transformer-bf16.safetensors
./tests/h3_ltx_refiner_gate --probe --rows 8 --seed 7
./tests/h3_ltx_refiner_gate --transformer $MODEL --blocks 1 --rows 8 --audio-rows 4 --seed 7 --dump ~/ds4-test/refiner-spark

# hygiene
cargo test -p ds4-core --lib 3<&-      # 750 passed; 0 failed; 7 ignored
osv-mcp map_dependencies /data/ds4-dfm-rs   # 140 packages, 0 findings
```

Scratch on the workstation: `/tmp/refiner-a`, `/tmp/refiner-zero.safetensors`
(773,595,963 B, the zeroed block-0 shard), `/tmp/refiner-full`,
`/tmp/refiner-spark`, `/tmp/refiner-full-spark`. On the Spark:
`~/ds4-test/refiner-spark`, `~/ds4-test/refiner-full-spark`,
`~/ds4-test/refiner-zero.safetensors`, `~/ds4-test/unit6-spark.{sh,log}`,
`~/ds4-test/b6.bundle`.
