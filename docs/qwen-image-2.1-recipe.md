# Qwen-Image-2.1 port recipe — reference index and op mapping

[Documentation index](README.md) | [Repository README](../README.md) |
[Plan](qwen-image-2.1-plan.md)

This is the working reference for the port described in the
[plan](qwen-image-2.1-plan.md). It records the reference pin, the file-and-line
index, the model geometry and the ds4-side op mapping.

The reference is a C++ project with its own ggml fork. It is read, never built
or linked: every formula below is translated into this tree's own layers — C for
the native engine, nvcc C++ for the CUDA units, Rust for the host — and nothing
from `stable-diffusion.cpp` is compiled, linked or shipped.

Rule for using it: the reference implementation is the authority. Where this
document states a number or a layout it was read from the pinned source at the
cited line; where it says *read at port time*, the detail is in the reference
and must be read there, not inferred. Nothing here is a substitute for that
read.

## 1. Reference pin

| Field | Value |
| --- | --- |
| Project | `stable-diffusion.cpp` (`leejet/stable-diffusion.cpp`) |
| Commit | `6dcb5bbd4278aa8f6d851f7515e87555f8e757b7` |
| Version string | `master-890-74988b2-4-g6dcb5bb` |
| Its ggml | submodule pinned at `4bf5f6000653b7881d00963cd6ddb665ccd62a8d`; the deployed build uses the patched in-tree ggml |
| Local binaries (patched) | `/data/imagegen/bin/sd-cli`, `/data/imagegen/bin/sd-server` |

The tree was not on disk when this was written; the files below were fetched
from the raw GitHub URLs at the pinned commit. The binaries' version string was
read from `sd-cli --help`.

Model artifacts already on this host (gitignored, never committed):

```
/data/imagegen/models/diffusion_models/qwen-image-2.1-Q6_K.gguf
/data/imagegen/models/vae/qwen_image_2.1_vae_bf16.safetensors
/data/imagegen/models/text_encoders/Qwen3VL-8B-Instruct-Q4_K_M.gguf

The diffusion GGUF is the only one this plan targets; the other diffusion files
on disk are not part of it. The text encoder's own quantization is a property of
that file.
```

## 2. Reference file index

| Path | Lines | What it owns |
| --- | --- | --- |
| `src/model/diffusion/qwen_image_2_1.hpp` | 384 | The 2.1 config, layout/segments, attention, block, model and runner. **The port follows this file.** |
| `src/model/diffusion/qwen_image.hpp` | 815 | The 1.0 DiT. Used only for the shared `TimestepEmbedding` and for contrast. |
| `src/model/vae/wan_vae.hpp` | 1566 | The causal 3D VAE, all blocks, the 2.1 configuration and the latent statistics. |
| `src/model/common/rope.hpp` | 1172 | `embed_nd`, `apply_rope`, `gen_qwen_image_ids` (1.0), the attention wrapper. |
| `src/model/diffusion/dit.hpp` | 180 | `patchify`, `patchify_3d`, `unpatchify_and_crop`. |
| `src/model/diffusion/flux.hpp` | 1758 | `Flux::modulate`, used by the 1.0 block. |
| `src/runtime/denoiser.hpp` | 3061 | Schedulers, the Euler samplers, CFG handling. |
| `src/model/common/block.hpp`, `ggml_block.hpp` | — | `Linear`, `LayerNorm`, `RMSNorm`, `Conv2d`, `FeedForward`, weight plumbing. |
| `src/core/ggml_extend.h/.cpp` | — | `ggml_ext_conv_3d`, `ggml_ext_attention_ext`, `ggml_ext_timestep_embedding`, pad/slice/permute helpers. |
| `src/pipeline/diffusion_engine.cpp` | 133 KB | The loop: latents, CFG, scheduler, VAE calls. **Read at port time.** |
| `src/conditioning/conditioner.hpp` | 178 KB | Qwen3-VL conditioning. **Read at port time.** |
| `src/tokenizers/qwen2_tokenizer.cpp` | 3.2 KB | The tokenizer wrapper. |

Line numbers below are against the pinned commit.

## 3. Geometry

### 3.1 DiT

From `QwenImage21Config` (`qwen_image_2_1.hpp:8-52`), with `num_layers` and
`intermediate_size` detected from the weights:

| Field | Value | Source |
| --- | --- | --- |
| layers | 32 | `qwen_image_2_1.hpp:14`, confirmed by detection at `:33-47` |
| hidden | 4096 | `:10`, detected from `img_in.weight` at `:24-27` |
| context dim | 4096 | `:11`, detected from `txt_in.in_layer.weight` at `:28-31` |
| head dim | 128 | `:12`, detected from `norm_q.weight` at `:32` |
| heads | 32 | derived: `hidden_size / head_dim` (`:161`) |
| intermediate | 12288 | `:13`, with a `fused_mlp` flag when `img_mlp.gate_up.weight` exists (`:34-40`) |
| in channels | 64 | `:9`, detected from `img_in.weight` |
| out channels | 64 | `:9`, detected from `proj_out.weight` |
| axes dim | `{16, 56, 56}` | `:16` |

Three facts that contradict a reading of the 1.0 file:

1. There is no spatial patchify in the 2.1 path. The image goes through
   `DiT::patchify(image, 1, 1)` (`qwen_image_2_1.hpp:287`), so in-channels 64 is
   the raw latent channel count, not a patch-folded size.
2. There are no `add_q/k/v_proj` and no `to_add_out`. Text and image are one
   token stream; the attention splits it by segment
   (`qwen_image_2_1.hpp:160-186`).
3. The timestep conditioning is `concat(t, 0)` (`:266`), and every modulation
   parameter is two rows: row 0 for the image part, row 1 (the zero row) for the
   text prefix (`:200-215`).

### 3.2 VAE

From the `VERSION_QWEN_IMAGE_2_1` branch (`wan_vae.hpp:1072-1090`):

| Field | Value | Source |
| --- | --- | --- |
| decoder dim (`dec_dim`) | 144 | `wan_vae.hpp:1074` |
| latent channels (`z_dim`) | 64 | `:1075` |
| output channels (`input_channels`) | 4 | `:1076` |
| `dim_mult` | `{1, 2, 4, 8, 8}` | `:1077` |
| temporal upsample | `{true, true, true, false}` | `:1086` |
| temporal downsample | `{false, true, true, true}` | `:1087` |
| `decode_only` | default true | `wan_vae.hpp:1027` |

The latent statistics table for 64 channels is at `wan_vae.hpp:1366-1378`; the
two conversions are at `:1381-1389`:

```
vae -> diffusion :  (latents - mean) * scale_factor / std
diffusion -> vae :  latents * std / scale_factor + mean
```

`mean` and `std` are the two 64-value tables at those lines. `scale_factor` is a
VAE-runner member; read its value at port time rather than assuming 1.

### 3.3 Sampler

Euler with classifier-free guidance, flow schedule (`denoiser.hpp`):

- `flux_time_shift(mu, sigma, t) = exp(mu) / (exp(mu) + (1/t - 1)^sigma)` (`:725`).
- `mu` is a linear map over the image token count, anchored at 256 tokens for
  `base_shift` and 4096 tokens for `max_shift` (`:732-758`).
- The grid is `t = 1 - i/n`, `i` in `0..=n`, with the last sigma forced to 0
  (`:769-783`).
- The Euler step and the CFG variants are `sample_euler_cfg_pp` (`:2718`) and
  `sample_euler_ancestral_cfg_pp` (`:2738`), dispatched at `:3051`.

The 2.1 runner itself asserts batch size 1 (`qwen_image_2_1.hpp:325-329`), so
guidance is assembled by the pipeline, not inside the runner. Read
`src/pipeline/diffusion_engine.cpp` at port time for how cond and uncond are
issued.

## 4. DiT dataflow

Forward, from `QwenImage21Model::forward` (`qwen_image_2_1.hpp:266-303`):

```
time       = concat(timestep, zeros_like(timestep))                 # 2 rows
time       = timestep_embedding(time, 256, theta=10000, scale=1)   # [256, 2]
time       = TimestepEmbedding(256 -> 4096, no bias)(time)         # [4096, 2]
time       = silu(time)
modulation = Linear(4096 -> 4*4096, no bias)(time)                 # [16384, 2]
mod          = chunk(modulation, 4, dim 0)                         # 4 x [4096, 2]

text  = txt_in(context)                 # ZeroCenterRMSNorm -> Linear -> GELU -> Linear
joint = concat over segments of:
            text[context_start : context_start+len]     for a text segment
            img_in(patchify_1x1(image))                 for an image segment
for i in 0..32:  joint = block_i(joint, mod, pe, layout, masks)
joint = joint[prefix_length :]           # image tokens only
scale = norm_out.linear(chunk(time, 2, dim 1)[0])
joint = LayerNorm(joint) * scale
joint = proj_out(joint)                  # -> 64 channels
out   = unpatchify_and_crop(joint, 1, 1)
```

Details to read at port time, with the lines:

- `TimestepEmbedding` itself (`qwen_image.hpp:74-96`): `linear_1`, `silu`,
  `linear_2`.
- `QwenImage21TextProjection` (`qwen_image_2_1.hpp:136-149`): a
  `QwenImage21ZeroCenterRMSNorm` (`:122-133`, eps 1e-6), then
  `Linear(4096, 4096, no bias)`, then GELU, then `Linear(4096, 4096, no bias)`.
- `modulate` (`:200-215`): chunks the parameter on its second axis into two rows,
  applies row 0 to the slice after `prefix_length` and row 1 to the prefix,
  multiplies, and applies `tanh` when `gate` is set. The exact effect of the
  `ggml_scale_bias(x, 1.f, 1.f)` calls inside it (`:206`, `:290`) must be read
  from `ggml_extend.h` rather than assumed.
- The last `joint = LayerNorm(joint) * scale` line uses `ggml_scale_bias` on the
  scale tensor (`:291`); same rule.

Block, from `QwenImage21TransformerBlock::forward` (`qwen_image_2_1.hpp:216-240`):

```
h = LayerNorm(img_norm1)(x)
h = modulate(h, mod[0])                    # no gate
h = attention(h, pe, segments, masks)
x = x + modulate(h, mod[1], gate=true)     # tanh-gated residual
h = LayerNorm(img_norm2)(x)
h = modulate(h, mod[2])
gate_up = Linear(gate_up)(h)               # or proj + gate_layer when not fused
gate, h = chunk(gate_up, 2, dim 0)
h = h * silu(gate)
h = Linear(img_mlp.out)(h)
x = x + modulate(h, mod[3], gate=true)
```

Attention, from `QwenImage21Attention::forward` (`qwen_image_2_1.hpp:160-186`):

```
q,k,v = to_q/to_k/to_v(x)                  # Linear, no bias, all four erased
q = RMSNorm(head_dim, 1e-6)(q);  k = RMSNorm(head_dim, 1e-6)(k)
q = apply_rope(q, pe);  k = apply_rope(k, pe)      # rope_interleaved defaults true
for each segment:
    sq = q[segment.start : segment.end]
    sk = k[0 : segment.end]
    sv = v[0 : segment.end]
    out = attention_ext(sq, sk, sv, heads, masks[i], skip_reshape, flash)
result = concat(out)
return to_out_0(result)                    # Linear, no bias
```

Segments, masks and positions (`qwen_image_2_1.hpp:72-121` and `:340-357`):

- A text segment gets a causal mask: `mask[k, q] = -inf` for `k > q`, with shape
  `[segment.end, segment.len]` (`:346-353`).
- An image segment gets no mask (`:355`), so image tokens attend to the whole
  text prefix and to every image token — bidirectional.
- Positions come from `QwenImage21Layout::build`, not from
  `Rope::gen_qwen_image_ids`. Text tokens take a monotonic scalar repeated on all
  three axes; image tokens take a `(t, h, w)` triple with a centered spatial id.
  Read the builder for the exact constants; `gen_qwen_image_ids`
  (`rope.hpp:569-591`) belongs to the 1.0 runner (`qwen_image.hpp:638`) and must
  not be transplanted.

RoPE:

- `pe = Rope::embed_nd(positions, 1, 10000.f, axes_dim)` (`qwen_image_2_1.hpp:335`),
  producing a tensor shaped `[2, 2, head_dim/2, L]` (`:361`).
- `embed_nd` (`rope.hpp:191-250`) applies one rope per axis with that axis's
  width (`16`, `56`, `56`) and theta 10000, then concatenates.
- `apply_rope` (`rope.hpp:1110-1113`) takes `pe` as `[L, d_head/2, 2, 2]` laid
  out `[[cos, -sin], [sin, cos]]`, and `rope_interleaved` defaults to `true`.
  This is the adjacent-pair convention; it is one of the two conventions that
  cannot be settled by reading alone and needs a probe against the reference.

## 5. VAE dataflow

Decoder structure (`wan_vae.hpp:832-900`):

```
conv1    = CausalConv3d(z_dim -> dims[0], kernel {3,3,3}, pad {1,1,1})
middle   = ResidualBlock, AttentionBlock, ResidualBlock   (dims[0] -> dims[0])
upsample = Up_ResidualBlock chain over dim_mult
head     = conv -> out_channels
```

`dims` is built as `{dim_mult.back() * dim}` plus `dim * dim_mult[i]` in reverse
(`:851-854`).

Block contracts:

- `CausalConv3d` (`wan_vae.hpp:18-92`): weight is 4-D F16 `[kW, kH, kT, IC*OC]`
  (`:33-38`); spatial padding is symmetric, temporal padding is left-only
  `lp2 = 2*p_t`, `rp2 = 0` (`:69-82`), which is the causality; an optional
  `cache_x` is concatenated on the temporal axis for streaming decode
  (`:78-81`); the convolution itself is single-group
  (`ggml_ext_conv_3d(..., false, ...)` at `:85`). Exports may store a Conv3d
  weight with a singleton temporal kernel, in which case the temporal kernel and
  padding collapse to 1 and 0 (`:31-35`).
- `RMS_norm` (`:93-125`): permute to put channels at `ne[0]`, `ggml_rms_norm`
  with eps `1e-12` (`:118`), scale by the block weight, permute back.
- `Resample` (`:144-274`): 2x nearest upscale (`ggml_upscale(..., 2,
  GGML_SCALE_MODE_NEAREST)` at `:238-240`) with the temporal handling around it.
- `ResidualBlock` (`:373-464`), `Down_ResidualBlock` (`:465-525`),
  `Up_ResidualBlock` (`:526-587`).
- `AttentionBlock` (`:588-648`): single head, non-causal
  (`ggml_ext_attention_ext(..., 1, nullptr, false, flash)` at `:635`).

## 6. ds4 op mapping

Left column is the need; middle is what already exists in this tree; right is
the action. Line numbers are in `ds4_gpu.h` unless stated.

| Need | Existing | Action |
| --- | --- | --- |
| Quantized GEMM | `matmul_q8_0_tensor:884`, `matmul_f16_tensor:1151`, `matmul_bf16_tensor:1163`, `matmul_f32_tensor:1268`, `cuda/mmq` | reuse |
| Per-head RMSNorm | `head_rms_norm_tensor:1847`, `rms_norm_plain_tensor:1304` | reuse for `norm_q`/`norm_k` |
| Weighted RMSNorm | `rms_norm_weight_rows_tensor:1326` | reuse for the text projection |
| SiLU gate | `swiglu_tensor:2669` | adapt: the VAE and the DiT gate are a plain `a * silu(b)`, not a packed SwiGLU |
| GELU | `mimo2_gelu:5100`, `qwen4exp_vision_bias_gelu_tensor` | adapt to a standalone elementwise op |
| Attention with a mask | `attention_prefill_masked_mixed_heads_tensor:2561` | adapt for the text segment; the image segment is unmasked, `attention_prefill_raw_heads_tensor:2395` is the closer base |
| Sequence concat | `concat_rows_tensor:3426` | reuse |
| Residual add | `add_tensor:3136` | reuse |
| Tensor, view, alloc | `tensor_view:45`, `tensor_alloc:43`, `tensor_read:77`, `tensor_write:76` | reuse |
| RoPE | `rope_tail_tensor:1977`, `qwen4exp_mrope_tensor` | new variant: 3 axes, adjacent pair, per-axis widths |
| LayerNorm without affine | — | new |
| Modulation (chunk + row select + tanh gate + fma) | — | new |
| Sinusoidal timestep embedding | — | new (host-side acceptable) |
| patchify/unpatchify at 1x1 | — | new glue over tensor views |
| VAE causal Conv3d | 1-D only: `ds4_gpu_mimo2_conv1d`, `qwen4_conv_stream_tensor`, `inkling_sconv` | new |
| VAE channel RMSNorm | — | new |
| 2x nearest upscale | — | new |
| VAE spatial attention | — | new (small; a tiled kernel is likely simpler than adapting the LLM kernel) |
| GroupNorm | — | not needed for this VAE; do not build it speculatively |
| Upsample/downsample blocks | — | new, composites of the above |

Verified absent (grep across `ds4.c`, `ds4_cuda.cu`, `ds4_metal.m`,
`metal/*.metal`, `cuda/*.cuh`): `conv2d`, `conv3d`, `upsample`, `groupnorm`,
`pixel_shuffle`. The only 2-D spatial op in the tree is the bilinear position
resize at `cuda/step37_vision.cuh:32`.

## 7. Kernel inventory (CUDA)

New file `cuda/qwen_image_primitives.cuh`, included through
`ds4_qwen_image_gpu.cuh`, in the style of the existing `cuda/*_primitives.cuh`
headers: a header
comment that states the math, a namespace, small named device helpers, and a
comment beside anything whose layout is not obvious.

| Kernel | Contract | Test |
| --- | --- | --- |
| `qwen_image_modulate` | split a `[hidden, 2]` parameter on its second axis, select the row by token range, multiply, xor `tanh` gate, add residual | vs the CPU oracle at one block |
| `qwen_image_layernorm` | affine-free LayerNorm, eps 1e-6, over the hidden axis | unit |
| `qwen_image_rope3d` | 3-axis, widths 16/56/56, theta 10000, adjacent pair, `pe` as `[L, 64, 2, 2]` | probe vs the reference dump |
| `qwen_image_attn_segment` | per-segment attention with an optional causal mask, head dim 128, 32 heads | unit + one full block vs the oracle |
| `qwen_image_mlp_gated` | `Linear -> chunk -> a * silu(b) -> Linear`, fused and unfused weight layouts | unit |
| `qwen_image_timestep_embed` | 256-wide sinusoidal, theta 10000, then the 256 -> 4096 MLP | unit vs the oracle |
| `qwen_image_patch_1x1` / `unpatch_crop` | reshape and crop for a 1x1 patch | unit |
| `qwen_image_conv3d_causal` | weight `[kW,kH,kT,IC*OC]` F16, symmetric spatial pad, left-only temporal pad, stride, dilation, optional temporal cache, single group | unit vs a CPU reference convolution |
| `qwen_image_vae_rms` | channel-wise RMSNorm, eps 1e-12, with a block weight | unit |
| `qwen_image_upsample2x` | nearest 2x with the temporal handling of `Resample` | unit |
| `qwen_image_vae_attn` | single-head, non-causal spatial attention | unit |
| `qwen_image_vae_conv_block` | `CausalConv3d` + `RMS_norm` + SiLU, the repeated composite | unit |
| `qwen_image_euler_step` | the Euler update and the CFG combine, or host-side | unit |

The heavy GEMMs and the elementwise glue reuse the existing ops listed in
section 6; this inventory is only what does not exist.

## 8. Rust host work

Following the ownership split in
[FFI_CONTRACT.md](rust-migration/FFI_CONTRACT.md) and
[ARCHITECTURE.md](rust-migration/ARCHITECTURE.md): Rust owns identification,
validation, the tensor bind plan, the request and lifecycle; native owns memory,
graphs and numerics.

1. Identification and layout. A module beside `identify.rs` that recognizes the
   DiT and VAE GGUFs (architecture key, tensor names, dims) and produces the
   expected-name table and the bind plan. The pattern to mirror is the one the
   text families use: a variant value, a `SHAPE_*` constant, an `expected_*`
   layout, a `validate_*` and a binder — but expressed as an engine kind rather
   than a `ModelFamily`.
2. Engine kind. Image is not a `ModelFamily`; it is a sibling engine at the
   `Model` level (`crates/ds4-core/src/lib.rs:561,902`). Design that split before
   writing kernels.
3. Plan and quote. Requested, effective and qualified limits, with the AR-only
   flags refused by name rather than silently accepted
   ([serving-contract.md](serving-contract.md)).
4. A crate for the pipeline: config parsing, the sampler loop, conditioning
   input, PNG output, and the HTTP surface.

## 9. ABI additions

New symbols in `native/bridge/ds4_bridge.h`, following the existing shape
(opaque handle, `int` return, `char *err, size_t errlen`, one `_free` per
`_open`; see `ds4_bridge_model_open` at `ds4_bridge.h:172`):

```
ds4_bridge_image_open(...)        -> opaque handle
ds4_bridge_image_step(...)        -> one DiT evaluation
ds4_bridge_image_decode(...)      -> VAE decode to pixels
ds4_bridge_image_plan(...)        -> requested/effective/qualified
ds4_bridge_image_free(...)
```

No `CUstream`, device pointer or MMQ descriptor crosses this boundary.

## 10. Harness and fixtures

The reference binary can dump its own intermediates, which is what makes a
staged port possible. Confirm the exact switches against the pinned source
before relying on them; the ones the deployed build exposes include the token
ids, the conditioning tensor, the per-step velocity and the initial noise.

The committed evidence for this port is: artifact hashes, the per-stage numbers
(noise and ids byte-identical, conditioning correlation, one DiT evaluation
correlation and relative RMS, final image statistics), the timing split between
sampling and VAE decode, and the images themselves. Scratch harnesses stay
outside the tree.

## 11. Open items

1. The exact `ggml_scale_bias(x, 1.f, 1.f)` semantics in `modulate` and in the
   final norm scale.
2. Whether `apply_rope`'s adjacent-pair convention is the one the 2.1 weights
   were trained with, or whether the 2.1 path needs a different pairing. Settle
   it with a probe against a reference dump, not by reading.
3. `scale_factor`'s value for this VAE.
4. How the pipeline batches cond and uncond, given the runner's batch-1 assert.
5. The conditioner's prompt template and the hidden-state slice point.
6. Whether the VAE attention should reuse an adapted prefill kernel or get a
   dedicated tiled kernel.
