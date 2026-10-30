# The LTX-2.5 Sol-Attn policy (unit 9)

Date: 2026-10-30. Branch `feat/sol-h3`, commits `c8771502` (the policy),
`8986d253` (the gate's dump comparison), `35280e19` (the release's accounting)
and `4b39f639` (an unused local).
Reference pin `/data/ltx2-pin/packages` (Lightricks/LTX-2 `d1511477` = tag
v1.2.0); release runtime `/data/Sol-h3-Spark` @ `6e4a440`, its Spark profile at
`models/minimax_h3/Sol-H3-Spark/`.

Unit 8 ran the release's whole stage-2 route end to end, but every one of the
48 blocks ran dense attention, so a step cost 126 s. What the release does
instead is not a kernel the port was missing, it is a policy it had not ported.
This unit is that policy.

## 1. What landed

- `crates/ds4-core/src/minimax_h3/ltx_refiner.rs`: `TAU_SCHEDULE`,
  `SOL_LAYERS`, `SOL_HEAD_DIM`, `SOL_FIRST_LAYER`, and `Sol`, the policy
  handle. `Refiner::apply`/`apply_one`, `Block::apply`, `self_and_text` and
  `Attention::run` carry `Option<&Sol<'_>>`, so one binary runs both arms.
  `Refiner::apply` and `Stage2::step` return the release's
  `(sol_calls, dense_calls)`.
- `native/h3/h3_dit_gpu.{cu,h}`: `h3d_sol_ltx_pool_kernel`,
  `h3d_sol_ltx_threshold_kernel`, `h3d_sol_ltx_forward_kernel` and the two
  entry points `h3d_sol_ltx_search`/`h3d_sol_ltx_forward`, all token-major.
- `crates/ds4-sys/src/h3_dit.rs`: `sol_ltx_search`, `sol_ltx_forward`,
  `sol_mask`.
- The gate: `--sol` on the stage mode, `probe_sol`, and the `--compare` mode.

## 2. The policy, read from the release

`models/minimax_h3/super_acceleration/stage2/sol_attention.py` (297 lines) is
the whole policy, and it is three constants and one redirect:

| fact | source |
|---|---|
| `STAGE2_TAUS = (1.0, 1.25, 1.5)`, one tau per denoise step | `:18` |
| `STAGE2_TRANSFORMER_LAYERS = 48`, enforced | `:19`, `:78-82` |
| layer 0 stays dense, layers 1..47 route | `:4-5`, `:114-117` |
| only `attn1` (video self-attention) is redirected | `:3-4` |
| the step advances on arrival at layer 1, one dense call per step | `:152-166` |
| the cross-attention stays dense | `:205-209` |
| the route is `_run_sol_attn_bthd(tau=STAGE2_TAUS[step], thresh_type="diag", kv_splits="auto", dense_fn=...)` | `:180-193` |
| a dense control exists and is instrumented the same way | `:223-296` |

The rule the kernel applies is the released `sol_attn`, unchanged from what the
port already runs for H3's stage 1:

- `kc` = block mean of K, `vc` = block **sum** of V, both BF16, into 64-token
  blocks (`sol_attn/preprocess.py:280-322`, `sol_attn/triton_ref/preprocess.py:34-68`);
- per head, `kc_mean`/`kc_var_diag` over the blocks (`sol_attn/preprocess.py:96-135`);
- per (query block, head) threshold
  `q_bar . kc_mean * log2_scale + tau * sqrt(max(q_bar^2 . kc_var * log2_scale^2, 0) + 1e-6)`
  with `q_bar` the query block's mean row (`:143-186`);
- `exact = col_mean > threshold or |q_block - block| <= 1`, the diagonal band
  unconditional (`triton_ref/fwd.py:98-99`);
- a group with no exact block leaves the running maximum untouched
  (`fwd.py:267-275`), an exact block is walked key by key, an approximated one
  is folded as `o += exp2(score) @ vc` with `denominator += exp2(score) * length`
  (`fwd.py:121-131`).

### Which backend is the reference

Two variants exist and they are not interchangeable. The Spark release
**enforces the Triton backend** — `runtime/stage2_ops/models.py:100-103`
raises unless `_actual_sol_backend() == "triton"` — while the GB200 variant
requires `cute_sm100` (`refiner_encoder_ablation_single_gpu.py:480-481`).
`sol_attn/interface.py:72-91` selects CuTe only for a capability in its own
table (`(8,9)`, `(9,0)`, `(10,0)`, `(12,0)`) and Triton otherwise, and the
Spark builds for `sm_121a` (section 8), which is not in that
table. **The reference for this port is therefore the Triton kernel**
(`sol_attn/triton_ref/fwd.py`); the CuTe files are not.

### The release's own accounting

`_checked_attention_stats` (`refiner_encoder_ablation_single_gpu.py:767-793`)
asserts, for the `sol` backend, `completed_steps = len(sigmas) - 1 = 3`,
`dense_calls = 3` and `sol_calls = 47 * 3 = 141` — one dense video
self-attention per step against 47 that route. That is the policy's shape, and
it needs no golden. The port reproduces it and the gate checks it (`c8771502`,
`35280e19`).

### What the other engine's H3 policy is not

`models/minimax_h3/GB200/sol_attn_h3.py` calls the same released kernel, with
`tau=1.0` fixed and the packed `[text | conditioning video | audio | target
video]` prefix marked as an exact KV **sink**
(`sink_start`/`sink_tokens`, `interface.py:134-142`). The port's existing
`h3d_sol_search`/`h3d_sol_forward` are that policy, head-major. The LTX-2.5
stage-2 policy has **no sink** — `attn1`'s sequence is video rows only — and a
three-value tau schedule instead of one. Same kernel, different policy.

## 3. The port's choice: the same rule, the refiner's layout

The rule above was already in the port, head-major, for H3's stage 1, and its
header says so: `h3d_sol_search`'s doc comment names it "the tau policy ... the
release applies to the joint refiner". What the refiner cannot use is the
*layout*: its attention is `[rows][heads][128]` (`Attention::run`), while the
H3 pair is `[heads][rows][128]`. Transposing every operand per layer would cost
more than the attention it feeds — 47 layers of 16,128 x 32 x 128 BF16, six
transposes a layer, ~37 GB of traffic per step at ~235 GB/s.

So the rule is re-expressed row-major and the two layout-agnostic steps are
reused verbatim:

| step | head-major (H3) | token-major (LTX-2.5) |
|---|---|---|
| block summaries | `h3d_sol_pool_kernel` | `h3d_sol_ltx_pool_kernel` |
| kc statistics | `h3d_sol_stats_kernel` | reused verbatim |
| query centroid + threshold | `h3d_sol_threshold_kernel` | `h3d_sol_ltx_threshold_kernel` |
| route (mask, count) | `h3d_sol_route_kernel` | reused verbatim |
| sparse forward | `h3d_sol_forward_kernel` | `h3d_sol_ltx_forward_kernel` |

`h3d_sol_stats_kernel` and `h3d_sol_route_kernel` read only `kc` (as
`[head][block][128]`) and `query_bar` (as `[q_block][head][128]`), which both
layouts produce identically, so they are the same code. `h3d_sol_search` and
`h3d_sol_forward` are untouched.

Named deviations of the token-major pair: row-major against the reference's
`[B,T,H,D]`; natural-log softmax where `fwd.py` tracks the maximum in log2
units, which is the port's dense kernel's own unit, so a tau that forces every
block exact reproduces `h3d_ltx_attention`'s arithmetic exactly; the
accumulation orders the port fixes where Triton's trees are not reproducible
here.

## 4. The kernels

`h3d_sol_ltx_forward_kernel` is the shape of `h3d_ltx_attention_kernel` — one
CTA per (16-row query tile, head), one warp per `H3D_ATTENTION_ROWS` rows,
each row's 128-wide dot split over the warp's lanes — with the KV loop
replaced by a per-64-token-block decision read from the plan's mask:

```
for block in 0 .. blocks:
    exact = (mask[q_block][head][block/32] >> (block%32)) & 1
    if not exact:                       # one fold for the whole block
        score = q_row . kc[block]       # block mean of K
        o += exp(score) * vc[block]     # block SUM of V
        denominator += exp(score) * key_rows
    else:                               # the dense walk, restricted
        for each 8-key chunk of the block:
            score = q_row . k
            o += exp(score) * v
            denominator += exp(score)
```

The block decision is the plan's mask, so the forward adds no policy of its
own: the diagonal band, the tau and the sink range (empty here) were all
decided in `h3d_sol_ltx_search`. `h3d_ltx_attention_kernel` is unchanged and
stays the control.

Where the policy reaches the graph: `Refiner::apply` filters the policy to
blocks whose shard layer index is at least `SOL_FIRST_LAYER`, so `Block::apply`
only passes it to `self_and_text` for the **video** self-attention; the audio
self-attention, both cross-attentions and the AV pair get `None` and stay
dense. `Attention::run` branches on the `Option` and requires
`head_dim == 128` and `q_rows == kv_rows`, which is what makes it a
self-attention.

## 5. Measurements

### 5.1 The probes, on both hosts

`./tests/h3_ltx_refiner_gate --probe` runs 30 probes; the 7 new Sol ones pass
identically on the workstation's RTX 4070 SUPER (sm_89) and the Spark's GB10
(sm_121):

```
PROBE sol_forward_tau-1000000000 values=8388608 differ=0 rel_rms=0.000e0 verdict=exact pass
PROBE sol_forward_tau0           values=8388608 differ=1116023 rel_rms=1.765e-3 verdict=residual pass
PROBE sol_forward_tau1           values=8388608 differ=1505114 rel_rms=2.145e-3 verdict=residual pass
PROBE sol_forward_tau1.25        values=8388608 differ=1549022 rel_rms=2.190e-3 verdict=residual pass
PROBE sol_forward_tau1.5         values=8388608 differ=1576555 rel_rms=2.219e-3 verdict=residual pass
PROBE sol_shares exact_share=[-1000000000:1.0000, 0:0.4222, 1:0.1693, 1.25:0.1386, 1.5:0.1192]
PROBE sol_fold values=8388608 differ=128 rel_rms=4.140e-7 verdict=fold pass
```

Three facts, none of which needs a golden:

1. **All-exact is the dense walk.** `tau = -1e9` drives every block above
   threshold, and the sparse forward then agrees with `h3d_ltx_attention`
   **bit for bit** over 8,388,608 values. The mask, the summaries, the
   block decision and the walk are wired correctly; the write-back and the
   online softmax are the dense kernel's own.
2. **The band is always exact and tau is monotone.** The probe checks
   `|q_block - block| <= 1` for every off-diagonal pair at five taus and
   reports the realized exact share per tau; the share falls strictly with
   tau, which is the routing-comparison contract (`preprocess.py:186`).
3. **The fold is the arithmetic it claims to be.** With every 64-token block
   of K and V constant the block mean *is* the key and the block sum is 64 of
   them, so the approximated path computes the same real number as the exact
   one and only F32 summation order differs: `rel_rms = 4.14e-7`,
   `differ = 128` of 8.4 M.

### 5.2 The three-interval evaluation, on the Spark

48 layers, release geometry (16,128 video rows, 1,024 context rows), synthetic
latent and context, seed 20251030 — the same run as unit 8's dense evaluation,
with `--sol` added and nothing else changed.

| | unit 8, dense | unit 9, Sol-Attn (run 1 / run 2) |
|---|---|---|
| step 0 | 126.022 s | 42.518 / 42.688 s |
| step 1 | 127.106 s | 37.391 / 37.620 s |
| step 2 | 127.995 s | 30.854 / 31.077 s |
| denoise | 381.143 s | 110.784 / 111.433 s |
| wall | 389 s | 119 / 118 s |
| load | 6.710 s | 6.638 / 5.852 s |
| memory peak | 44 GiB of 121 | 44 GiB of 121 |

Two runs, 0.6% apart on the denoise, 3.44x and 3.42x against the dense run. The
exact share per step falls with tau exactly as the schedule says, and the
counters are the release's:

```
STAGE-SOL step=0 tau=1    exact_share=0.2025 sol_calls=47 dense_calls=1
STAGE-SOL step=1 tau=1.25 exact_share=0.1484 sol_calls=47 dense_calls=1
STAGE-SOL step=2 tau=1.5  exact_share=0.1221 sol_calls=47 dense_calls=1
```

`sol_calls=47 dense_calls=1` per step is the release's own accounting
(`refiner_encoder_ablation_single_gpu.py:771-780`); the gate fails a 48-layer
run that reports anything else. (The first launch of the second run backgrounded
a compound shell command and never started; the run above is the one that ran.)

Memory is flat: the plan is `kc`/`vc` at `heads x blocks x 128` BF16 (4.1 MB),
the mask `heads x blocks x 8` u32 (258 KB), and the routing scratch is
per-(block, head) — nothing scales with rows x keys, which is the failure mode
the family contract warns about.

### 5.3 The state hashes, and what they do not show

Every state hash of the sparse run equals the dense run's:

```
stage-noise   2eac5cb8bb2aa957   (identical: the noise is drawn before either path)
stage-state0  a7991fc33188f068
stage-step0   2f0eb3015a958e30
stage-step1   d22edcb9bf77ac7d
stage-step2   03a31966cef75544
stage-latent  4266f9e3f93be058
```

and `--compare stage9-dump stage8b-dump` reports all six dumps bit-identical:

```
COMPARE stage-latent.bf16 values=2064384 differ=0 rel_rms=0.000e0 cosine=1.000000 max_abs=0.0000
COMPARE stage-noise.bf16  values=2064384 differ=0 rel_rms=0.000e0 cosine=1.000000 max_abs=0.0000
COMPARE stage-state0.bf16 values=2064384 differ=0 rel_rms=0.000e0 cosine=1.000000 max_abs=0.0000
COMPARE stage-step0.bf16  values=2064384 differ=0 rel_rms=0.000e0 cosine=1.000000 max_abs=0.0000
COMPARE stage-step1.bf16  values=2064384 differ=0 rel_rms=0.000e0 cosine=1.000000 max_abs=0.0000
COMPARE stage-step2.bf16  values=2064384 differ=0 rel_rms=0.000e0 cosine=1.000000 max_abs=0.0000
```

This is **not** evidence that the policy is correct, and it is not evidence
that it did nothing. The step time (42.7 s against 126.0 s), the realized exact
share (20%, 15%, 12%) and the counters (`sol_calls=47`) all say the routing ran
and that ~80-88% of the KV blocks were folded. What the identity says is that
the change is below the state's resolution here.

The natural explanation is BF16 rounding, and it is **consistent but weak**:
the probe measures the attention output moving 2.1e-3 relative at `tau=1`, and
the state is about 4.5 in magnitude, whose ulp is 0.03125 — so a delta is lost
when `|gate * attention| * 2.1e-3 < 0.0156`, i.e. whenever the video
self-attention's gated contribution is under ~7.4, which a residual of 4.5
would make extraordinary to exceed. The bound is therefore so loose that the
observation carries no information about the contribution's actual size, and
nothing here measures it. Section 6.

## 6. What it does not prove

- **Every value gate is a host transcription of the reference's own
  expression**, so a shared misreading survives it. The Sol-Attn rule here is
  the port's *existing* H3 transcription, which was gated against the
  reference's raw-row order (`tests/h3/test_native.cu`) and its cost measured
  (`tests/h3/sol_route_bm`); this unit re-expresses it row-major and does not
  re-gate it against a second source.
- **The identical state hashes are unexplained by measurement.** BF16 rounding
  is the only hypothesis consistent with the numbers and its bound is loose
  enough to be nearly vacuous (5.3): no probe or dump reads the video
  self-attention's contribution magnitude, or its gate's, against the state's.
  The identity is consistent with an inert attention, with a contribution much
  smaller than the state's ulp, and with a contribution barely under it.
- **The routing's *quality* is measured on random operands**, not on the
  model's own. The release's acceptance is paired NLL / top-1 over many
  positions and the final latent against the dense run; both need a real input
  latent and a real context (section 9), and the state-level comparison has no
  resolving power here (5.3).
- **No golden.** The release's own stage-2 output for this geometry has not
  been produced, so nothing in this unit is compared against the release's
  numbers, only against the port's own dense control.

## 7. What is not ported

The cute_sm100 variant (the port targets the Spark, which the release pins to
Triton), `kv_splits > 1` (SM90 only, `interface.py:113-119`), the `exact`
threshold type (`thresh_type` is `"diag"` in the release's policy, and only
that is implemented), the fused MMDiT path with a text KV sink (H3's, already
ported head-major), Morton ordering (`install_wan_morton_forward`, which the
H3 policy's own docstring explains H3 does not need), and the latent
upsampler, decoder, audio unpatchify, keyframe slots and LoRA merge that unit 8
also left out.

## 8. Reproduction

```
# workstation (sm_89)
make CUDA_ARCH=sm_89 tests/h3_ltx_refiner_gate
./tests/h3_ltx_refiner_gate --probe

# Spark (sm_121)
cd ~/ds4-sol-h3
export PATH=$HOME/.cargo/bin:$PATH
rm -f native/h3/*.o tests/h3_ltx_refiner_gate
make CUDA_ARCH=sm_121 tests/h3_ltx_refiner_gate
./tests/h3_ltx_refiner_gate --probe
./tests/h3_ltx_refiner_gate --stage --transformer \
  /mnt/h3-artifacts/LTX-2.5/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  --layers 48 --seed 20251030 --sol --dump ~/ds4-test/stage9-dump
./tests/h3_ltx_refiner_gate --compare ~/ds4-test/stage9-dump ~/ds4-test/stage8b-dump
```

`--compare` is reporting only: it prints the residual, the cosine and the
largest absolute difference per shared dump file, has no ceiling and returns 0.
`--residual` keeps its 2e-2 ceiling, which is right for BF16 arithmetic noise
and wrong for a policy change.

## 9. Owed decisions

1. **A real input latent and context** (the upsampler and the connector), without
   which the policy's quality is a property of random operands only.
2. **A measurement, not an inference, of why the states do not move**: the
   video self-attention contribution and its gate against the state's
   resolution.
3. **The golden dump** for units 2b-9, unchanged: every gate in this campaign is
   a transcription of the reference's own expression.
4. The written cache's `torch.load` acceptance (unit 5).
5. The INT8 connector file: `distilled` or `dev`?
6. The decoder's route (F32 110.4 s, `bf16` 30.4 s, `bf16i` 24.4 s).
