# DeepSeek V4.1 Flash on ds4-dfm-rs — port plan

[Repository README](../README.md) | [Model families](ds4-dfm-model-families.md) |
[Performance](performance.md) | [Prefill/decode playbook](prefill-decode-optimization-playbook.md)

Status: proposal, not started. Branch: `ds41` (off `dev` 7b6448b).

Resume protocol: this document is the state carrier. There is no handoff file and
no rotation to a fresh session; the session compacts (/compact) and continues, so
every step below must be executable from this text alone.

Authored 2026-10-08. The port has one authority and it is now a real source tree,
not a decompilation: the working C/CUDA engine at `/data/YoungAi` (clone of
JLFN/YoungAi, branch `restructure`, commit 3946dbc). Citations marked (a) were
collected by read-only inventory agents in this session and carry file:line for
re-checking; citations marked (v) were verified first-hand.

## 1. Goal

**ds4-dfm-rs serves the published DeepSeek V4.1 Flash vq8 artifact natively, with
the serving behaviour this tree already gives every other family: prefill, KV
reuse, its own sampling defaults, its own DSML tool path, and the four API
contracts.**

The port is a port of capability, not a rescue: the C engine already runs this
model, measured (artifact `README.md` §1). What is missing is the whole V4.1
substrate in the Rust host.

| Id | Success criterion | Proof |
| --- | --- | --- |
| G1 | The artifact loads: the GGUF's 40-44 tensor types and the VQ blobs are parsed and the tensor inventory matches the engine's accepted inventory | the inventory diff, both engines |
| G2 | Numerically faithful: tokens and logits match the C engine's golden set on fixed prompts at temperature 0 | the golden-set comparison, per position |
| G3 | Served through the existing API surface: chat, completions, responses, Anthropic | one served request per contract |
| G4 | Tool calls work end to end, no translating shim | a tool call issued, executed, fed back |
| G5 | Speculative decode runs: MTP towers load and the verify/rollback keeps the committed cache at the accepted-prefix boundary | the losslessness gate, cache-frontier |
| G6 | Sidecars work: a domain sidecar changes the output as the engine's does, and a mismatched posttrain pair is refused | the sidecar A/B and the `base.fnv` refusal |
| G7 | Performance ledger, measured not assumed | the ledger, same prompts as §2 |

Non-goals for v1: no Metal (V4.1 is CUDA-only in the reference, README §7),
no distributed inference, no quantizer or solver work (the sidecar and
post-training solvers are the C toolchain's job), no re-quantization.

## 2. The artifact, as measured

Read over the gvfs mount of the Spark
(`/run/user/1000/gvfs/sftp:host=192.168.1.91,user=leandro/home/leandro/youngai/model`) (v).

| item | value |
| --- | --- |
| base | `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf`, 113,556,639,424 bytes, 40 parts |
| base sha256 | `9469cfa9ff47c5b9f1e2bbf31b4226ef11b4de9ddb92a0c7641c4a9b623e4654` (`SHA256SUMS`; every part also hashed) |
| sidecars | five domain dirs `…-grrb-{vqfin41_vqhalf_a_n8192,code_fit_n15360,law_fit_n15360,med_fit_n15360,sci_fit_n15360}-engine`, each `gr_Lnn.bin` (1,044,492 B) + `rb_Lnn.bin` + `manifest.txt` |
| post-train | `posttrain-experimental-20260924/`: `gr_L39.bin` + `base.fnv` (20 B) |
| engram | not in the artifact: two official shards, read in place via `--engram-dir`. Present on the Spark at `/home/leandro/youngai/deepseek-engram`: `model-00047-of-00048.safetensors` 101,535,150,936 B and `model-00048-of-00048.safetensors` 101,537,926,640 B (190 GB, measured over sftp 2026-10-08) |
| bins | `bin/ds4`, `bin/ds4-server`, built on the Spark with `make cuda-spark` |

Engine invocation the port must be able to reproduce (artifact README §3, §7):

    ./bin/ds4-server --cuda -m <base>.gguf --zchain <sidecar_dir> \
        --engram-dir <shards_dir> --mem-budget-mb 110000 [--posttrain <dir>] [--no-dspark]

Endpoints `/v1/chat/completions`, `/v1/completions`, `/v1/responses`,
`/v1/messages`; request-omitted sampling follows the model card (temperature 1.0,
top_p 1.0, no min_p); thinking is off unless requested.

## 3. Method: derive from the engine, cite it, gate on the golden set

The reference implementation is `/data/YoungAi` at 3946dbc. Nothing is
re-derived from a spec or from memory: every format, index width, slot address
and kernel geometry is read from that tree and cited in the code, the evidence
file and the commit message (the tree's own rule 0.2 doctrine).

The acceptance instrument is the C engine's captured behaviour on the Spark:
for a fixed prompt set at temperature 0, the token stream and the logits. The
port consumes that capture and nothing else as its correctness oracle. Until it
exists, G2 has no gate and "it works" is an impression.

Fidelity to the Rust host's standards (its `AGENTS.md`): common serving surface
across families, requested/effective/qualified limits reported, correctness
proof AND speed proof per optimization commit, and the inference boundary
preserved (host -> ds4-core -> ds4-sys -> native).

## 4. What the Rust host already has (v, (a))

- Family identity and shapes: `ModelFamily::DeepSeek4 = 0`, `SHAPE_FLASH` 43
  layers, `SHAPE_PRO` 61 (`crates/ds4-core/src/shape.rs:20-33`, `:435-492`,
  `:494-551`).
- The DeepSeek validator and its required metadata: `validate_deepseek`
  (`crates/ds4-core/src/validate.rs:317-456`), including
  `hyper_connection.count`, `hyper_connection.sinkhorn_iterations`,
  `hyper_connection.epsilon`, `compress_ratios`, `swiglu_clamp_exp`,
  `attention.indexer.*`.
- Weight binding with the unified expert naming that DeepSeek uses
  (`bind.rs:656-700`, `layout.rs:2383`), and the DSpark sibling bind slots in
  the native boundary (`native/bridge/ds4_host_load.h:323-331`).
- DSpark device kernels: `dspark_capture_mean`, `dspark_gather_concat`,
  `dspark_markov_step` (`ds4_cuda.cu:16308`, `:16322`, `:19621`; launchers
  `:21815-21833`), mirrored in `metal/dsv4_misc.metal:1330-1391`. These are
  the V4-era drafter's kernels; the V4.1 artifact needs the three-tower MTP
  drafter instead (inventory §8), so this is a re-use question, not a port.
- Serving caps for the family, including `MtpKind::DeepSeek` and
  `SpecLane::Bank` (`crates/ds4-core/src/serving.rs:886-904`), and the
  DeepSeek DSML tool path (`crates/ds4-server/src/tools.rs:20-31`, `:260+`).

So the base V4 architecture is not the work. The work is the V4.1 container
format and its subsystems.

The function-level inventory of the C engine's V4.1 surface (host and device,
with citations) lives in [the inventory](deepseek41-port-inventory.md) and is
the work list this plan is built from. It is being completed subsystem by
subsystem (VQ, engram and sidecars, MTP draft, core and serving).

## 5. Gap inventory

"Absent" = zero grep hits in the Rust tree; the C reference is cited per row.

| id | gap | C reference (source of truth) |
| --- | --- | --- |
| G1 | GGUF tensor types 40-44 (`go1b` 256/34, `go2b` 256/68, `vqblob` 1/1, `fp4x32` 32/17, `fp8_32x32` 1024/1025) unknown: the host table is 31 entries covering ids 0-30 plus `pq2_0`=142 (`crates/ds4-core/src/tensors.rs:55-96`) (v) | `src/core/core_gguf.c:89-104`; engine enum `ds4_internal.h:177-181` (a) |
| G2 | VQ expert decode absent. `DQVL` blob header, v2 `DQVQ` per-matrix payload (own codebook) and v3 `DQV3` (per-layer codebook + 12-bit main stream + 1-bit plane, E4M3 codewords, per-row f16 gain) | `vq_fmt.h:1-20`, decode `vq_fmt.h:57`/`:91` (v); device geometry `src/cuda/cuda_vq_row.inc.cu:19-47`; dispatch by (version, bit width) `cuda_vq_decode_launch.inc.cu:21-30` (a) |
| G3 | MTP draft towers absent: 3 towers reusing the layer struct, per-expert `fp4x32` or a single `ffn_exps_vq.blob`, five shared heads | bind `src/core/core_bind_v41.c:101-145`; tower types `core_draft_tower_types.h`; forward `core_v41_draft.c:1-11` (a) |
| G3b | The expert tensors are not the ones this host expects: the artifact carries one `blk.L.ffn_exps_vq.blob` per layer and binds no `ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps` at all, while the Rust bind plan requires those three (`bind.rs:693`, `layout.rs:2672-2692`) | `core_bind_v41.c:41` (a), inventory §1 |
| G4 | Speculative verify/rollback for V4.1 absent. Note the engine has TWO drafters: the V4-era DSpark (`dspark.*`, `cuda_dspark.inc.cu`) which this host already mirrors, and the V4.1 three-tower MTP (`mtp.*`, `core_v41_draft.c`) which is the one this artifact needs | `v41_spec_snapshot` `src/core/core_v41_forward.c:241`, `v41_spec_rollback` `:267`; round drive `core_v41_api.c:322`, `:365`; tower contract and the two rollback cautions: inventory §8 (a) |
| G5 | Engram absent: two tables, 264 B rows (256 B fp8 e4m3 + 8 B ue8m0 scale), rolling 4-gram hash, O_DIRECT per-row reads | `src/core/core_v41_engram.c:6`, `:21`, `:30`, `:63`; metadata `core_validate_v41.c:118-154`; flag `--engram-dir` `src/cli/cli_opts.c:269`, `--v41-no-engram` `:371` (a) |
| G6 | Domain sidecars absent: `gr_Lnn.bin` per-row gains, `rb_Lnn.bin` router bias, `amp_Lnn.bin`, the `base.fnv` pair fingerprint, `--zchain`, `--posttrain` | `src/core/core_v41_amp.c:41`, `:88`, `:125-147`, `:160-186`, `:227`; fingerprint `src/common/ds4_gr_fnv.h`; flags `cli_opts.c:258-268` (a) |
| G7 | V4.1 metadata is not identified: `deepseek4.attention.kv_source_layers` and `index_source_layers` are read nowhere, and there is no V4.1 variant/shape branch | V4.1 metadata load `core_validate_v41.c:46-154`; shape `core_shape_select.c:81-116`, `:211` (a) |
| G8 | V4.1 serving surface absent: no `--zchain`/`--posttrain`/`--engram-dir`/`--no-dspark` on the Rust server, and request-omitted sampling has no family default (the host is request-driven) | server flags `src/server/server_config.c:314-329`; defaults `server_msgs.c:162-164` from `ds4.h:54-56` (a) |
| G9 | `manifest.txt` is documentation only in the C engine too (no reader); the Rust port must not invent one | grep for "manifest" in `src/`: no reader (a) |

## 6. Build order and gates

Each phase is gated on a comparison, never on inspection. The Spark is the only
machine the reference was tested on (sm_121, CUDA 13, 128 GB unified), so P4
onward run there; all Spark work goes through the owner's ssh session and asks
before any state-changing command.

| phase | work | gate |
| --- | --- | --- |
| P0 | Freeze inputs: verify the assembled artifact hash (the 40 part files are deleted after assembly, so `SHA256SUMS`' part lines cannot resolve); capture the C engine's golden set with `tests/capture_ds41_golden.sh`, with the engram tables as the primary instrument and `NO_ENGRAM=1` as the fixture variant; record the artifact's accepted tensor inventory | hashes re-verify; the golden set exists as files with its own MANIFEST |
| P0 | DONE 2026-10-08 | see 6.5 |
| P1 | Tensor types 40-44 in `tensors.rs` and the VQ decode as a Rust oracle (CPU), codebook geometry read from the blob, not assumed | decoded weights byte-match `ds4vq_dequant_f32` on fixed tensors of both blob versions (v2 and v3) |
| P2 | Loaders: engram metadata and table open, sidecar `gr`/`rb` reader, `base.fnv` check (parse-only, not applied) | the tensor/key inventory matches the engine's; a mismatched posttrain pair is refused |
| P3 | Routing and bind: a `Variant::V41` shape, the V4.1 keys, and the engram/sidecar state wired into session state | the loader accepts the artifact and the inventory diff is empty |
| P4 | CUDA: VQ MoE decode (mirror `v41_vq_open` geometry), the fp8_32x32 skeleton path, and the engram read path | G2 on device: logits match the golden set |
| P5 | MTP towers and DSpark verify/rollback | byte-identical greedy output with drafting on and off at N=1; cache-frontier gate at N>1. The engine's own cautions: the main-hidden ring trim in rollback, and the compressor pending-row snapshot taken before the shift |
| P6 | Serving: the flags, the sampling profile for this family, the four API contracts, the DSML tool path | a served tool call, end to end, no shim |
| P7 | Performance ledger: prefill and decode on the §2 prompts, cache reuse demonstrated | the ledger, published with its method |

P0 and P1 can start without further owner decisions.

## 6.1 Where the work hooks into the Rust host

From the call-graph index (`callgraph-mcp`, project ds4-dfm-rs, queried
2026-10-08):

- `validate_file` is the per-family dispatch: it calls `validate_deepseek` and
  its ten siblings (`crates/ds4-core/src/validate.rs`). A V4.1 arm extends this
  dispatch.
- `validate_deepseek` has 15 call sites at depth 2 from `open_impl`,
  `validate_gguf`, `probe_model_artifact`, `check_metadata`, `dump_validate`
  and `host_catalog_uses_exact_contract`; any signature change to it is felt by
  all of them.
- `bind_names` has 14 call sites from `resolve`, `matches_published_tensors`,
  `bind_plan_requires_every_published_tensor`, `layout_covers_bind_catalog` and
  the Flash-inventory tests; the V4.1 bind path must keep those catalogs in
  agreement (layout vs bind vs published tensors).

## 6.3 What stays native, what moves to Rust

The dividing rule is the repo's own boundary rule (`AGENTS.md`: keep new
host/control-plane code in Rust; existing CUDA/MMQ stays native; the inference
boundary host -> ds4-core -> ds4-sys -> native is preserved). Applied to this
port, per subsystem:

Moves to Rust (host side: parsing, validation, policy, orchestration, I/O):

- The GGUF type table entries 40-44 and their byte geometry
  (`src/core/core_gguf.c:89-104`): done in `crates/ds4-core/src/tensors.rs`.
- The VQ container and payload parse, both blob versions
  (`vq_fmt.h`, `src/cuda/cuda_vq_row.inc.cu:19-49` geometry): done in
  `crates/ds4-core/src/vq.rs`. This is the oracle the loader validates with; it
  is not the execution path.
- Metadata validation and identification (`core_validate_v41.c:46-154`,
  `core_shape_select.c:81-116`) into `validate.rs`, `identify.rs`, `shape.rs`.
- Tensor naming and layout (`core_bind_v41.c:41-145`) into `bind.rs` and
  `layout.rs`.
- Sidecar formats and policy: `gr_Lnn.bin`, `rb_Lnn.bin`, `amp_Lnn.bin`, the
  `base.fnv` fingerprint check (`core_v41_amp.c:41-227`, `ds4_gr_fnv.h`) are
  file formats and admission policy, so they belong to `ds4-core`.
- Engram metadata parse, table open, row addressing and the prefetch policy
  (`core_validate_v41.c:118-154`, `core_v41_engram.c` open/pool parts): host
  I/O in Rust. The wkv matmul and its gate stay native.
- Serving: the family's sampling profile, the `--zchain`/`--posttrain`/
  `--engram-dir`/`--no-dspark` flags and their defaults
  (`server_config.c:314-329`, `server_msgs.c:162-164`), the V4.1 request
  branch, and the DSML/thinking handling (`server_generate_v41.c`).
- The speculative round loop's policy: when to draft, the acceptance rule, the
  commit boundary, and the snapshot/rollback orchestration
  (`core_v41_api.c:322,365`, `core_v41_forward.c:241,267`) - the host drives
  it, the device state itself stays native.

Stays native (never re-implemented in Rust):

- Every CUDA kernel: the VQ decode/GEMV/prefill/MMA family
  (`src/cuda/cuda_vq_*.inc.cu`), the V4.1 attention, indexer, hyper-connection,
  fp4/q4k and sampler kernels (`cuda_v41_*.inc.cu`), the draft and DSpark
  kernels (`cuda_v41_draft.inc.cu`, `cuda_dspark.inc.cu`). They stay in the
  single-TU native backend behind the existing ABI.
- The kernel-internal geometry: block loads, the swizzle, the shfl index
  distribution, register budgets, the 12-bit/13-bit block arithmetic - the
  measured micro-optimizations are device facts, not host policy.
- The device-side sampling kernel and the verify forward's device state.
- The C engine itself stays where it is as the behaviour oracle and the
  reference for every kernel; nothing in it is deleted by this port.

Decision test for anything not listed: if it runs per token on the device, it
stays native; if it runs once at load, per request, or is policy, validation or
file-format work, it moves to Rust.

## 6.4 Test strategy: the Spark, through the owner's tmux session

Owner waiver, recorded 2026-10-08: the rule 19 QA-tester gate does not apply
to this workstream; the agent tests its own work and reports the evidence.

All builds and tests for this port run on the DGX Spark (gx10-e1d2, sm_121,
CUDA 13, 128 GB unified), which is the only machine the reference engine was
tested on and the only one holding the artifact. The Rust host must therefore
be present on the Spark (clone or rsync of the `ds41` branch) — see O5.

Working rules for the Spark, non-negotiable:

- Drive it only through the owner's ssh tmux pane, or an askpass session; there
  is no standing ssh key (owner rule, 2026-10-03).
- Read-only by default. Any state-changing command (build, write, stop a
  process) is asked for explicitly, per connection.
- Tests are launched from the pane and their output is captured from the pane;
  long runs are left visible rather than backgrounded.
- The memory budget is the guard: the engine refuses to start above
  `--mem-budget-mb 110000`, so a test run must not hold another large model.

## 6.5 P0 evidence (2026-10-08, the Spark)

Frozen inputs, all measured:

| item | value |
| --- | --- |
| engine binary | `/home/leandro/youngai/model/bin/ds4`, sha256 `c03be9b4ac5de52b197b3740ba6ec60001b7454b7ab5d0f111cda3f3fee8a7f5` |
| artifact | sha256 `9469cfa9ff47c5b9f1e2bbf31b4226ef11b4de9ddb92a0c7641c4a9b623e4654`, recomputed by the capture script and equal to the `SHA256SUMS` assembled line |
| engram | `--engram-dir /home/leandro/youngai/deepseek-engram`, two shards, 190 GB |
| golden set | `/home/leandro/youngai/model/golden/`, MANIFEST with every hash |

Segment PPL of the five prompts (with engram): p1 112.0845 (S=8), p2 24.9582
(S=11), p3 7.2809 (S=7), p4 14.3524 (S=18), p5 876.7182 (S=7, Chinese). The
no-engram variant of p1 differs (104.3867), which is the expected effect of the
subsystem and the reason the primary capture runs with the tables.

Discovery worth the phase order: the score path writes a per-layer trace next
to the logits — `x_L<nn>.bin` (the layer input, S x 5120 f32), `y_L<nn>.bin`,
`hce_L01/L14.bin` and `erows_L01/L14.txt` for the engram layers. That is a
layer-by-layer gate instrument, not just an end-of-model one, and the port
should use it from P3 onward: a wrong layer shows up at its own layer index
instead of being smeared into the final logits.

## 7. Numerics contract

- Device arithmetic and every format detail follow the C engine's code, and the
  PTX/SASS it compiles to where the two disagree.
- The VQ gain application and the E4M3 codeword interpretation follow
  `cuda_vq_row.inc.cu`, not a plausible reading of the format description.
- Greedy (temperature 0) comparisons are the gate. Sampled comparisons are
  reported as distributions, because the Rust host samples with its own profile
  and the streams diverge after the first non-greedy step.
- Where the Rust host's existing DeepSeek kernels and the engine disagree
  numerically, the engine's captured behaviour wins for this family, and the
  deviation is recorded rather than smoothed over.

## 8. Risks

| risk | evidence | mitigation |
| --- | --- | --- |
| The golden set never lands, so P4+ has no gate | none captured yet; P0 is the first step | P1-P3 proceed regardless and say so |
| Scope is large: four subsystems (VQ, engram, MTP, sidecars) | §5, each with its own kernels | phase gates; the base V4 family already works, so nothing existing regresses |
| The host memory budget: 113.6 GB base + sidecar, 203 GB engram on SSD | artifact README §3 | engram is optional by design (`--v41-no-engram` exists in the reference) |
| Two engines, one model: the Rust host is not the C engine and will not be bit-identical everywhere | P4 gate is the first real test | gate on logits/tokens at temperature 0 and record deviations |
| Spark access: no standing key (owner rule), state-changing commands need per-connection approval | memory 2026-10-03 | drive the Spark only through the owner's ssh tmux pane or askpass, read-only by default |
| The port quietly becomes a reimplementation of the engine | this is the doctrine's whole point | cite or refuse; no invented mechanism |

## 9. Rollback

The port is additive: a V4.1 variant/keys loader plus new device paths; no
existing family's code path changes. Rollback is reverting or deleting the
branch. No migration, no data, no protocol change.

## 10. Open questions for the owner

- O1 Does v1 ship without engram (the 203 GB shards) and without speculative
  decode? Both are disableable in the reference, so both are phasable.
- O2 Sampling: the reference ships the model card's recipe as the default
  (temperature 1.0, top_p 1.0, speculative on). Does this tree adopt that
  per-family profile, or keep request-driven sampling only?
- O3 Where does the port run? The reference is tested on the Spark alone; this
  host has no DS4 GGUF and no 110 GB of free VRAM-class memory.
- O4 Is parity with the C engine the goal (G2 as written, expensive), or
  capability plus speed with a looser numerical tolerance?
- O5 How does the `ds41` branch reach the Spark: clone the fork there (needs a
  credential for a private repo), or rsync the working tree over the existing
  gvfs/ssh path? The Spark has no Cargo workspace for this tree yet.
