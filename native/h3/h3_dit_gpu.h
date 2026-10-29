/*
 * h3d_* — the BF16 primitives the FL2VA DiT needs.
 *
 * Additive MiniMax-H3 family: new names, no other family's kernel touched, and
 * nothing here is reachable from another family's path. The reference is
 * `/data/h3.c` at `eae5d54`, `h3_gpu_cuda.cu` (`h3_gpu.h` for the signatures);
 * each function names the reference lines it mirrors.
 *
 * The host loop lives in Rust (`minimax_h3::dit`); this file owns device work
 * only. The block graph is:
 *
 *   hidden --adaln(norm1, slots 0/1)--> mod_attention
 *          --linear(qkv) + qk-norm + RoPE--> q,k,v  (head-major)
 *          --full attention--> heads --linear(out)--> attention_output
 *          --gate(residual hidden, slot 2) + adaln(norm2, slots 3/4)--> mod_mlp
 *          --linear(fc1) + swiglu + linear(fc2)--> mlp_output
 *          --gate(slot 5)--> hidden
 *
 * Arithmetic order is the reference's and is copied expression for expression,
 * including leaving the fused multiply-add decisions to the same compiler: the
 * E5 gate hashes the BF16 output, so a different reduction tree or an explicit
 * `fmaf` where the reference relies on contraction changes the last bit.
 *
 * Deviation, named: the reference carries F32, BF16, U32 and int8 tensors in
 * one `h3_gpu_tensor`; this ABI keeps the two dtypes the BF16 path uses and
 * refuses the int8/fused-NAX variants (CUDA returns 0 from
 * `h3_gpu_has_int8_mlp`/`h3_gpu_has_nax_mlp`, `h3_gpu_cuda.cu:332-334`, so the
 * reference's own CUDA path is the one implemented here).
 */
#ifndef H3D_DIT_GPU_H
#define H3D_DIT_GPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* `H3D_FP8` is E4M3, one byte per element, and exists only for the FP8
 * projection route below. `h3d_is` still matches a dtype exactly, so no
 * existing op can be handed one by accident. */
enum h3d_dtype { H3D_BF16 = 0, H3D_F32 = 1, H3D_U32 = 2, H3D_FP8 = 3 };

/* Sol-Attn's fixed shapes (`techniques/sparse_backends/sol_attn/preprocess.py`
 * `:11-13` in NVIDIA's Sana `sol-engine` tree: BLOCK_SIZE 64, HEAD_DIM 128;
 * `triton_ref/fwd.py:23` GROUP 32, the KV blocks one route mask word covers). */
enum { H3D_SOL_BLOCK = 64, H3D_SOL_HEAD_DIM = 128, H3D_SOL_GROUP = 32 };

typedef struct h3d_context h3d_context;
typedef struct h3d_tensor h3d_tensor;

/* The search step's per-step workspace. Every shape is documented and checked
 * before a kernel runs, because an undersized mask reads as an empty one.
 *
 *   kc[head][block][128]               bf16 block mean of K
 *   vc[head][block][128]               bf16 block sum of V
 *   mask[q_block][head][word]          u32  exact-KV-block bits for that word
 *   count[q_block][head]               u32  exact blocks, sinks included
 *   threshold[q_block][head]           f32  the routing threshold
 *   query_bar[q_block][head][128]      f32  the query block's centroid
 *   kc_mean[head][128]                 f32  block-mean statistics of kc
 *   kc_var[head][128]                  f32  their diagonal variance
 *
 * `kc` and `vc` are the block summaries the reference derates to BF16 before
 * anything reads them (`preprocess.py:287-291`): a context that keeps them F32
 * would route a different set. */
struct h3d_sol_plan {
    h3d_tensor *kc;
    h3d_tensor *vc;
    h3d_tensor *mask;
    h3d_tensor *count;
    h3d_tensor *threshold;
    h3d_tensor *query_bar;
    h3d_tensor *kc_mean;
    h3d_tensor *kc_var;
};

h3d_context *h3d_create(char *error, size_t error_size);
void h3d_destroy(h3d_context *context);

/* Attention path for this context. `scalar` nonzero selects the byte-exact
 * scalar kernel, the only path that reproduces the reference; zero is the
 * tensor-core default. Overrides H3_DIT_ATTENTION and H3_DIT_ATTENTION_MMA. */
void h3d_set_attention(h3d_context *context, int scalar);

/* One video VAE decoder option per call, for benchmarking an arm on one binary.
 * Names and values: "attention" (scalar | tensor, the F32 attention kernel),
 * "sdpa_order" (heads | rows, the F32 attention CTA order), "gemm" (fp32 |
 * tf32 | bf16, the decoder's F32 matmul compute mode). Every default is the
 * reference path, so a context that is never set is unchanged, and all three
 * are read from H3_VAE_ATTENTION, H3_VAE_SDPA_ORDER and H3_VAE_GEMM at create.
 * Only the decoder's own entry points read them: the video VAE encoder and the
 * audio VAE keep the reference's F32 matmul and separate bias add, which their
 * bit-exact gates compare against. Returns 0 for an unknown name or value, so a
 * typo is refused rather than ignored. */
int h3d_set_vae_option(h3d_context *context, const char *name,
                       const char *value);

h3d_tensor *h3d_alloc(h3d_context *context, int dtype, size_t elements,
                      char *error, size_t error_size);
void h3d_free(h3d_tensor *tensor);

/* Whole-tensor upload, then a range upload for the per-step latents. */
int h3d_upload(h3d_context *context, h3d_tensor *tensor, const void *host,
               char *error, size_t error_size);
int h3d_upload_at(h3d_context *context, h3d_tensor *tensor, size_t element,
                  const void *host, size_t elements, char *error,
                  size_t error_size);
int h3d_read(h3d_context *context, void *host, const h3d_tensor *tensor,
             char *error, size_t error_size);

/* `h3_gpu_linear_bf16`/`h3_gpu_linear_f32` (`:832-870`) through `h3_linear_lt`
 * (`:733-779`) and `h3_linear_bias` (`:783-804`). `bias` may be NULL; the
 * bias add happens after the matmul rounds to the output dtype. */
int h3d_linear(h3d_context *context, h3d_tensor *output,
               const h3d_tensor *input, const h3d_tensor *weight,
               const h3d_tensor *bias, uint32_t rows, uint32_t input_dim,
               uint32_t output_dim, int dtype, char *error, size_t error_size);

/* The release's own route for the DiT's projection linears, W8A8 tensorwise
 * E4M3. `runtime/stage1.py:126-134` tags the layer names
 * `runtime/stage1_ops/lookup.py:15-16` selects and calls
 * `convert_model_to_fp8`; the arithmetic here is FastVideo's
 * `fastvideo/layers/quantization/fp8_config.py` at the revision
 * `configs/dependencies.json` pins (`3d8ac9d1`). `h3d_fp8_scale` writes a
 * one-element F32 scale for a BF16 tensor, `h3d_fp8_quantize` rounds a BF16
 * tensor to E4M3 through it, and `h3d_linear_e4m3` is the matmul
 * `torch._scaled_mm` reaches: E4M3 x E4M3, F32 accumulate, BF16 output, with
 * the two tensorwise scales folded into the matmul's alpha on the device. */
int h3d_fp8_scale(h3d_context *context, h3d_tensor *scale,
                  const h3d_tensor *input, char *error, size_t error_size);
int h3d_fp8_quantize(h3d_context *context, h3d_tensor *output,
                     const h3d_tensor *scale, const h3d_tensor *input,
                     size_t elements, char *error, size_t error_size);
int h3d_linear_e4m3(h3d_context *context, h3d_tensor *output,
                    const h3d_tensor *input, const h3d_tensor *input_scale,
                    const h3d_tensor *weight, const h3d_tensor *weight_scale,
                    uint32_t rows, uint32_t input_dim, uint32_t output_dim,
                    char *error, size_t error_size);

/* The same route at the reference's `granularity="channel"`: one scale per row
 * of a row-major `[rows, columns]` BF16 tensor, so a weight's rows are its
 * output channels and an activation's are its tokens (`fp8_config.py:63-68`,
 * `:225-229`). cuBLASLt will not take the two vectors - the outer-vector scale
 * mode is `NOT_SUPPORTED` for E4M3 on this architecture and CUDA version
 * (`tests/h3/fp8_gemm_bm` modes 4-8) - so the matmul runs unscaled
 * (`h3d_linear_e4m3_raw`, F32 output) and `h3d_fp8_scale_epilogue` applies
 * both: `out[i][j] = bf16(acc[i][j] * row[i] * col[j])`, the reference's own
 * `(A @ W^T) * scale_a * scale_b` order on the F32 accumulator. */
int h3d_fp8_scale_rows(h3d_context *context, h3d_tensor *scales,
                       const h3d_tensor *input, uint32_t rows,
                       uint32_t columns, char *error, size_t error_size);
int h3d_fp8_quantize_rows(h3d_context *context, h3d_tensor *output,
                          const h3d_tensor *scales, const h3d_tensor *input,
                          uint32_t rows, uint32_t columns, char *error,
                          size_t error_size);
int h3d_linear_e4m3_raw(h3d_context *context, h3d_tensor *output,
                        const h3d_tensor *input, const h3d_tensor *weight,
                        uint32_t rows, uint32_t input_dim, uint32_t output_dim,
                        char *error, size_t error_size);
int h3d_fp8_scale_epilogue(h3d_context *context, h3d_tensor *output,
                           const h3d_tensor *accumulator,
                           const h3d_tensor *row_scale,
                           const h3d_tensor *column_scale, uint32_t rows,
                           uint32_t columns, char *error, size_t error_size);

/* The video VAE decoder's F32 linear: `h3d_linear` at `H3D_F32` with the
 * decoder's own options applied (`h3d_set_vae_option`). Separate from
 * `h3d_linear` on purpose, because the options must not reach the video VAE
 * encoder or the audio VAE, whose gates are bit-exact against the reference. */
int h3d_vae_linear(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *input, const h3d_tensor *weight,
                   const h3d_tensor *bias, uint32_t rows, uint32_t input_dim,
                   uint32_t output_dim, char *error, size_t error_size);

/* `h3_gpu_patch_linear_bf16_offset` (`:872-906`) and `h3_patch_convert_kernel`
 * (`:808-829`): an F32 matmul with F32 weights, bias added in F32, one BF16
 * rounding at the end. */
int h3d_patch_linear(h3d_context *context, h3d_tensor *output,
                     size_t output_element, const h3d_tensor *input,
                     size_t input_element, const h3d_tensor *weight,
                     const h3d_tensor *bias, uint32_t rows,
                     uint32_t input_dim, uint32_t output_dim, char *error,
                     size_t error_size);

/* `h3_gpu_patch_linear_bf16_map` (`:932-970`) with `h3_patch_scatter_kernel`
 * (`:917-930`): the same projection scattered to the rows a multi-segment
 * layout assigns it, which is what a keyframe or reference layout needs. */
int h3d_patch_linear_map(h3d_context *context, h3d_tensor *output,
                         const h3d_tensor *input, const h3d_tensor *weight,
                         const h3d_tensor *bias, const h3d_tensor *row_map,
                         uint32_t output_rows, uint32_t rows,
                         uint32_t input_dim, uint32_t output_dim, char *error,
                         size_t error_size);

/* `h3_gpu_adaln_bf16_offset` (`:2737-2755`) with `h3_adaln_kernel`
 * (`:2656-2687`): one row per block, thread 0 owns the RMS sum. */
int h3d_adaln(h3d_context *context, h3d_tensor *output,
              const h3d_tensor *input, size_t input_element,
              const h3d_tensor *norm_weight, const h3d_tensor *modulation,
              const h3d_tensor *row_map, uint32_t rows, uint32_t width,
              uint32_t slots, uint32_t shift_slot, uint32_t scale_slot,
              float epsilon, char *error, size_t error_size);

/* `h3_gpu_gate_bf16` (`:2803-2809`) with `h3_gate_kernel` (`:2689-2702`). */
int h3d_gate(h3d_context *context, h3d_tensor *output,
             const h3d_tensor *residual, const h3d_tensor *branch,
             const h3d_tensor *modulation, const h3d_tensor *row_map,
             uint32_t rows, uint32_t width, uint32_t slots,
             uint32_t gate_slot, char *error, size_t error_size);

/* `h3_gpu_token_pool_bf16` (`h3_gpu_cuda.cu:3297-3349`) with
 * `h3_token_pool_kernel` (`:3271-3295`): snapshot the source rows this pass
 * replaces, average each pair, and store the pooled video rows in the
 * baseline buffer the expansion later subtracts. A pair whose two rows are the
 * same (every non-video row, and an unpaired last column) passes through
 * unchanged. */
int h3d_token_pool(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *input, size_t input_element,
                   h3d_tensor *original, size_t original_element,
                   h3d_tensor *baseline, size_t baseline_element,
                   const h3d_tensor *baseline_indices,
                   const h3d_tensor *pairs, uint32_t input_rows,
                   uint32_t rows, uint32_t baseline_rows, uint32_t width,
                   char *error, size_t error_size);

/* `h3_gpu_token_expand_delta_bf16` (`:3353-3383`) with
 * `h3_token_expand_kernel` (`:3328-3351`): every row takes its parent's
 * reduced value as a delta over the pooled baseline, scaled and added to the
 * snapshot; rows before `prefix_rows` and rows with no baseline index take the
 * reduced value directly. */
int h3d_token_expand(h3d_context *context, h3d_tensor *output,
                     const h3d_tensor *original, size_t original_element,
                     const h3d_tensor *reduced, const h3d_tensor *baseline,
                     size_t baseline_element,
                     const h3d_tensor *baseline_indices,
                     const h3d_tensor *parents, uint32_t rows,
                     uint32_t reduced_rows, uint32_t baseline_rows,
                     uint32_t width, uint32_t prefix_rows, float scale,
                     char *error, size_t error_size);

/* `h3_gpu_rms_norm_bf16` (`:2512-2530`) with `h3_rms_norm_bf16_kernel`
 * (`:2462-2489`). */
int h3d_rms_norm(h3d_context *context, h3d_tensor *output,
                 const h3d_tensor *input, const h3d_tensor *weight,
                 uint32_t rows, uint32_t width, float epsilon, char *error,
                 size_t error_size);

/* `h3_gpu_swiglu_bf16` (`:2879-2883`) with `h3_swiglu_kernel` (`:2841-2855`). */
int h3d_swiglu(h3d_context *context, h3d_tensor *output,
               const h3d_tensor *fused, uint32_t rows, uint32_t width,
               char *error, size_t error_size);

/* `h3_gpu_grouped_qkv_rope_bf16` (`:3241-3249`) with `h3_qkv_rope_kernel`
 * (`:3079-3136`): the checkpoint's [head, q/k/v, dim] rows, Q/K head RMS norm,
 * then the pair rotation over the first `2 * rope_half` dimensions. */
int h3d_qkv_rope(h3d_context *context, h3d_tensor *query, h3d_tensor *key,
                 h3d_tensor *value, const h3d_tensor *qkv,
                 const h3d_tensor *q_norm, const h3d_tensor *k_norm,
                 const h3d_tensor *rope_cos, const h3d_tensor *rope_sin,
                 uint32_t sequence, uint32_t heads, uint32_t head_dim,
                 uint32_t rope_half, float epsilon, char *error,
                 size_t error_size);

/* `h3_gpu_sdpa_bf16` (`:1667-1672`) taking the CUDA tiled dispatch
 * (`:1620-1631`) with `h3_attention_tiled_bf16_kernel` (`:1418-1487`):
 * one block per (head, 8 rows), head-major input, row-major output. */
int h3d_sdpa(h3d_context *context, h3d_tensor *output,
             const h3d_tensor *query, const h3d_tensor *key,
             const h3d_tensor *value, uint32_t sequence, uint32_t heads,
             uint32_t head_dim, float scale, char *error, size_t error_size);

/* `h3_gpu_add_bf16` (`:2373-2376`) with `h3_binary_bf16_kernel`
 * (`:2347-2357`), `h3_gpu_cast_f32_to_bf16` (`:2287-2296`), and the SiLU
 * branches of `h3_gpu_silu_f32`/`h3_gpu_silu_bf16` (`:2276-2306`, `:2321-2329`)
 * the timestep embedding needs. */
int h3d_add(h3d_context *context, h3d_tensor *output,
            const h3d_tensor *left, const h3d_tensor *right, size_t elements,
            char *error, size_t error_size);
/* `h3_gpu_sub_bf16` (`h3_gpu_cuda.cu:2378-2382`): the core-reuse residual
 * (`h3_dit.c:2306-2312`). */
int h3d_sub(h3d_context *context, h3d_tensor *output,
            const h3d_tensor *left, const h3d_tensor *right, size_t elements,
            char *error, size_t error_size);
int h3d_cast_f32_to_bf16(h3d_context *context, h3d_tensor *output,
                         const h3d_tensor *input, size_t elements, char *error,
                         size_t error_size);
int h3d_silu_f32(h3d_context *context, h3d_tensor *output,
                 const h3d_tensor *input, size_t elements, char *error,
                 size_t error_size);
int h3d_silu_bf16(h3d_context *context, h3d_tensor *output,
                  const h3d_tensor *input, size_t elements, char *error,
                  size_t error_size);

/* `h3_gpu_copy_bf16` (`:690-696`) via `h3_copy` (`:672-688`). */
int h3d_copy(h3d_context *context, h3d_tensor *destination,
             size_t destination_element, const h3d_tensor *source,
             size_t source_element, size_t elements, char *error,
             size_t error_size);

/* Wait for every queued op; returns 0 and fills `error` on failure. */
int h3d_sync(h3d_context *context, char *error, size_t error_size);

/* ---- F32 primitives the AudioVAE decoder needs (`h3_audio_vae.c`) ---- */

/* `h3_gpu_copy_f32` (`:697-703`). */
int h3d_copy_f32(h3d_context *context, h3d_tensor *destination,
                 size_t destination_element, const h3d_tensor *source,
                 size_t source_element, size_t elements, char *error,
                 size_t error_size);

/* `h3_gpu_conv1d_f32`/`h3_gpu_conv1d_stride_f32` (`:1757-1794`) with
 * `h3_conv1d_kernel` (`:1681-1706`). Layout: [batch, time, channels]. */
int h3d_conv1d(h3d_context *context, h3d_tensor *output,
               const h3d_tensor *input, const h3d_tensor *weight,
               const h3d_tensor *bias, uint32_t batch, uint32_t length,
               uint32_t input_channels, uint32_t output_channels,
               uint32_t kernel, uint32_t stride, uint32_t padding,
               uint32_t dilation, char *error, size_t error_size);

/* `h3_gpu_conv_transpose1d_f32` (`:1796-1840`) with
 * `h3_conv_transpose1d_kernel` (`:1714-1743`). */
int h3d_conv_transpose1d(h3d_context *context, h3d_tensor *output,
                         const h3d_tensor *input, const h3d_tensor *weight,
                         const h3d_tensor *bias, uint32_t batch,
                         uint32_t length, uint32_t input_channels,
                         uint32_t output_channels, uint32_t kernel,
                         uint32_t stride, uint32_t padding, char *error,
                         size_t error_size);

/* `h3_gpu_weight_norm_f32` (`:2976-2992`) with `h3_weight_norm_kernel`
 * (`:2958-2974`): one row per thread, the same fmaf square sum and rsqrtf. */
int h3d_weight_norm(h3d_context *context, h3d_tensor *output,
                    const h3d_tensor *vector, const h3d_tensor *magnitude,
                    uint32_t outer, uint32_t inner, char *error,
                    size_t error_size);

/* `h3_gpu_alias_free_snake_f32` (`:1943-1961`) with
 * `h3_alias_free_snake_kernel` (`:1907-1933`). */
int h3d_alias_free_snake(h3d_context *context, h3d_tensor *output,
                         const h3d_tensor *input,
                         const h3d_tensor *alpha_log,
                         const h3d_tensor *beta_log,
                         const h3d_tensor *upsample_filter,
                         const h3d_tensor *downsample_filter, uint32_t batch,
                         uint32_t length, uint32_t channels, char *error,
                         size_t error_size);

/* `h3_gpu_add_scaled_f32` (`:2397-2411`) with `h3_add_scaled_kernel`
 * (`:2385-2391`) and `h3_gpu_clip_f32` (`:2309-2317`), whose clip is the
 * `H3_CLIP` branch of `h3_unary_f32_kernel` (`:2246-2274`). */
int h3d_add_scaled(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *left, const h3d_tensor *right,
                   float left_scale, float right_scale, size_t elements,
                   char *error, size_t error_size);
int h3d_clip(h3d_context *context, h3d_tensor *output,
             const h3d_tensor *input, size_t elements, float minimum,
             float maximum, char *error, size_t error_size);

/* ---- F32 primitives the VideoVAE decoder needs (`h3_video_vae.c`) ---- */

/* `h3_gpu_sdpa_f32` (`:1651-1657`) through `h3_attention_dispatch`
 * (`:1562-1649`): one batch, non-causal, Q/K/V head-major, output row-major.
 * `head_dim == 64` takes `h3_attention_tiled_f32_kernel<64>` (`:1489-1547`),
 * the video VAE's shape; any other width takes the generic
 * `h3_attention_kernel<float>` (`:1337-1416`), as the reference dispatches
 * (`:1623-1632`). */
int h3d_sdpa_f32(h3d_context *context, h3d_tensor *output,
                 const h3d_tensor *query, const h3d_tensor *key,
                 const h3d_tensor *value, uint32_t sequence, uint32_t heads,
                 uint32_t head_dim, float scale, char *error,
                 size_t error_size);

/* `h3_gpu_rms_norm_f32` (`:2496-2510`) with `h3_rms_norm_f32_kernel`
 * (`:2438-2460`): 256 threads per row, fmaf square sum, rsqrtf. */
int h3d_rms_norm_f32(h3d_context *context, h3d_tensor *output,
                     const h3d_tensor *input, const h3d_tensor *weight,
                     uint32_t rows, uint32_t width, float epsilon, char *error,
                     size_t error_size);

/* `h3_gpu_layer_norm_f32` (`:2601-2617`) with `h3_layer_norm_f32_kernel`
 * (`:2529-2561`): the mean is a plain running sum, the variance an fmaf sum,
 * and the epilogue keeps the reference's multiply order. */
int h3d_layer_norm_f32(h3d_context *context, h3d_tensor *output,
                       const h3d_tensor *input, const h3d_tensor *weight,
                       const h3d_tensor *bias, uint32_t rows, uint32_t width,
                       float epsilon, char *error, size_t error_size);

/* `h3_gpu_swiglu_f32` (`:2874-2877`) with `h3_swiglu_kernel` (`:2841-2852`):
 * gate and up halves of each fused row, `gate / (1 + expf(-gate)) * up`.
 * `bias` is the producing linear's own bias or NULL: adding it on load is the
 * same F32 add `h3_linear_bias_kernel<float>` makes, and `h3d_vae_linear` is
 * then called without one. */
int h3d_swiglu_f32(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *fused, const h3d_tensor *bias,
                   uint32_t rows, uint32_t width, char *error,
                   size_t error_size);

/* `h3_gpu_scale_add_f32` (`:2895-2910`) with `h3_scale_add_kernel`
 * (`:2885-2893`): `residual + branch * scale`, one multiply, no fma. `bias`,
 * when given, is added to `branch` on load, before the multiply. */
int h3d_scale_add_f32(h3d_context *context, h3d_tensor *output,
                      const h3d_tensor *residual, const h3d_tensor *branch,
                      const h3d_tensor *bias, const h3d_tensor *scale,
                      uint32_t rows, uint32_t width, char *error,
                      size_t error_size);

/* `h3_gpu_video_qkv_rope_f32` (`:3190-3205`) with `h3_qkv_rope_kernel`
 * (`:3078-3136`) at `grouped = 1`, `normalize = 1`, `weighted = 0`: the
 * checkpoint's [head][q,k,v][dim] rows, Q/K RMS over the head with no learned
 * weight, then the pair rotation. This is the video VAE variant, not the
 * DiT's. `bias`, when given, is the QKV linear's own, added on load. */
int h3d_video_qkv_rope_f32(h3d_context *context, h3d_tensor *query,
                           h3d_tensor *key, h3d_tensor *value,
                           const h3d_tensor *qkv, const h3d_tensor *bias,
                           const h3d_tensor *rope_cos,
                           const h3d_tensor *rope_sin, uint32_t sequence,
                           uint32_t heads, uint32_t head_dim,
                           uint32_t rope_half, float epsilon, char *error,
                           size_t error_size);

/* ---- F32 primitives the VideoVAE encoder needs (`h3_video_encoder.c`) ---- */

/* `h3_gpu_vae_encoder_pad_f32` (`h3_gpu_cuda.cu:2002-2043`) with
 * `h3_vae_pad_kernel` (`:1974-2000`): `depth_front` zeros in front, then a
 * single-reflection spatial pad (`h3_reflect`, `:1969-1973`). The temporal
 * axis is never reflected: the encoder pads time with zeros only. */
int h3d_vae_encoder_pad(h3d_context *context, h3d_tensor *output,
                        const h3d_tensor *input, uint32_t batch,
                        uint32_t depth, uint32_t height, uint32_t width,
                        uint32_t channels, uint32_t depth_front,
                        uint32_t height_before, uint32_t height_after,
                        uint32_t width_before, uint32_t width_after,
                        char *error, size_t error_size);

/* `h3_gpu_conv3d_f32` (`:2075-2127`) with `h3_conv3d_kernel` (`:2034-2073`):
 * one thread per output element, bias seeded before the first tap, then the
 * reference's fmaf order (input channel, kt, ky, kx). Weights are
 * `[output_channel, input_channel, kt, ky, kx]`. */
int h3d_conv3d(h3d_context *context, h3d_tensor *output,
               const h3d_tensor *input, const h3d_tensor *weight,
               const h3d_tensor *bias, uint32_t batch, uint32_t depth,
               uint32_t height, uint32_t width, uint32_t input_channels,
               uint32_t output_channels, uint32_t kernel_depth,
               uint32_t kernel_height, uint32_t kernel_width,
               uint32_t stride_depth, uint32_t stride_height,
               uint32_t stride_width, char *error, size_t error_size);

/* `h3_gpu_vae_encoder_group_norm_silu_f32` (`:2153-2178`) with
 * `h3_group_norm_silu_kernel` (`:2114-2151`): one thread per
 * (temporal, group) row, spatial-major accumulation inside the group, a plain
 * running sum for the mean, an fmaf variance, then `value / (1 + expf(-value))`
 * with the per-channel weight and bias applied first. */
int h3d_group_norm_silu(h3d_context *context, h3d_tensor *output,
                        const h3d_tensor *input, const h3d_tensor *weight,
                        const h3d_tensor *bias, uint32_t batch,
                        uint32_t depth, uint32_t height, uint32_t width,
                        uint32_t channels, uint32_t groups, float epsilon,
                        char *error, size_t error_size);

/* `h3_gpu_geglu_f32` (`h3_gpu_cuda.cu:2922-2933`) with `h3_geglu_kernel`
 * (`:2912-2920`): tanh GELU of the gate times the linear half. */
int h3d_geglu_f32(h3d_context *context, h3d_tensor *output,
                  const h3d_tensor *gate, const h3d_tensor *linear,
                  size_t elements, char *error, size_t error_size);

/* `h3_gpu_snake1d_f32` (`h3_gpu_cuda.cu:2945-2960`) with `h3_snake_kernel`
 * (`:2935-2943`): plain `x + sin(a x)^2 / (a + 1e-9)`, no filters. */
int h3d_snake1d_f32(h3d_context *context, h3d_tensor *output,
                    const h3d_tensor *input, const h3d_tensor *alpha,
                    uint32_t batch, uint32_t length, uint32_t channels,
                    char *error, size_t error_size);

/* `h3_gpu_audio_qkv_split_f32` (`h3_gpu_cuda.cu:1850-1888`) with
 * `h3_audio_qkv_split_kernel` (`:1829-1848`). */
int h3d_audio_qkv_split_f32(h3d_context *context, h3d_tensor *query,
                            h3d_tensor *key, h3d_tensor *value,
                            const h3d_tensor *qkv, const h3d_tensor *q_bias,
                            const h3d_tensor *k_bias,
                            const h3d_tensor *v_bias, uint32_t batch,
                            uint32_t length, uint32_t heads,
                            uint32_t head_dim, char *error,
                            size_t error_size);

/* `h3_gpu_audio_attention_pool_f32` (`h3_gpu_cuda.cu:1890-1926`) with
 * `h3_audio_pool_kernel` (`:1874-1893`). */
int h3d_audio_attention_pool_f32(h3d_context *context, h3d_tensor *output,
                                 const h3d_tensor *attended, uint32_t batch,
                                 uint32_t length, uint32_t heads,
                                 uint32_t head_dim, uint32_t output_dim,
                                 char *error, size_t error_size);

/* `h3_gpu_sdpa_causal_f32` (`h3_gpu_cuda.cu:1659-1667`) with the F32 branch
 * of `h3_attention_kernel` (`:1337-1419`), head-major input, row-major out. */
int h3d_sdpa_causal_f32(h3d_context *context, h3d_tensor *output,
                        const h3d_tensor *query, const h3d_tensor *key,
                        const h3d_tensor *value, uint32_t batch,
                        uint32_t sequence, uint32_t heads, uint32_t head_dim,
                        float scale, char *error, size_t error_size);

/* -------------------------------------------------------------------------
 * Sol-Attn's search step: which KV blocks attend exactly and which are
 * approximated from their summaries. Not on any production path yet; the
 * consumer is the sparse forward this port has not built.
 *
 * Ported from NVIDIA Sana `sol-engine` @ `6e4a440`, `sol_attn/`:
 *   preprocess.py:280-322   kc = block mean of K, vc = block sum of V (BF16)
 *   preprocess.py:96-135    kc_mean/kc_var_diag per head
 *   preprocess.py:143-186   the diag threshold: q centroid, mean + TAU*std
 *   triton_ref/fwd.py:250-266  col_mean = column mean of the route logits
 *   selector.py:132-142     exact = col_mean > threshold OR |q-n| <= 1
 *   sm120/mainloop.py:430-470  the same rule in the SM120 mainloop
 *   interface.py:134-142    the sink block range, forced exact
 *
 * Deviations, named: head-major `[heads][sequence][head_dim]` tensors with
 * batch 1 (the port's layout for every H3 attention primitive, and the
 * reference's [B,T,H,D] transposed); `query_bar` stored F32 so the routed
 * block list depends only on the reference's own BF16 rounding of kc/vc; the
 * accumulation orders are fixed by this port where the reference's Triton
 * trees are not reproducible here, and `tests/h3/sol_route_bm` measures what
 * the choice costs.
 *
 * `h3d_sol_search` routes from the query block's centroid, which is the
 * reference's column mean: sum_q (q.kc) / len = (sum_q q / len).kc, the same
 * number with 64x fewer multiplies, and both sides of its comparison then
 * come from one F32 centroid. The reference's own raw-row order is the oracle
 * `tests/h3/test_native.cu` checks this against, and `tests/h3/sol_route_bm`
 * reports what the substitution moves on the job's shape. */
int h3d_sol_search(h3d_context *context, const struct h3d_sol_plan *plan,
                   const h3d_tensor *query, const h3d_tensor *key,
                   const h3d_tensor *value, uint32_t sequence, uint32_t heads,
                   float scale, float tau, uint32_t sink_start,
                   uint32_t sink_tokens, char *error, size_t error_size);
/* Number of `mask` words per query block: ceil(ceil(sequence/64) / 32). */
uint32_t h3d_sol_words(uint32_t sequence);

/* VSA, the reference's own selection rule for the H3 stage, emitting the mask
 * the sparse forward consumes. Read from the FastVideo module the reference
 * hashes and refuses to run without
 * (`fastvideo/attention/backends/video_sparse_attn_h3.py`, sha256
 * `b7c6d4e4...dc0fab9` = the reference's pinned `VSA_SOURCE_SHA`):
 *
 *   tile summary = the tile MEAN of Q and of K, F32               (`:380-383`)
 *   scores = q_pool @ k_pool^T / sqrt(head_dim)                     (`:629`)
 *   k_vid  = max(1, min(ceil((1 - sparsity) * video_tiles), video_tiles))
 *                                            (`video_sparse_attn.py:161-163`)
 *   the video columns compete for the k_vid largest scores, every prefix
 *   column is selected, every prefix query row is dense, and a sparsity of 0
 *   selects everything                                      (`:387-410`)
 *
 * `prefix_blocks` is this engine's own text-and-audio span, rounded UP to
 * whole 64-token blocks: a block that holds a prefix row is treated as prefix,
 * which is the conservative reading of a span the reference keeps tile-aligned.
 * The rule is undefined when every block is prefix (its own `topk` would be
 * asked for more columns than exist), and `prefix_blocks >= blocks` is refused.
 *
 * It reuses the tau path's kernels - the block pooling, the kc statistics and
 * the threshold kernel that produces `query_bar` - so the plan must carry the
 * same buffers, and the `threshold` it fills is not read by this policy. Only
 * the selection kernel is new. `h3d_sol_search`'s doc comment applies to every
 * other deviation this shares with it. */
int h3d_sol_vsa(h3d_context *context, const struct h3d_sol_plan *plan,
                const h3d_tensor *query, const h3d_tensor *key,
                const h3d_tensor *value, uint32_t sequence, uint32_t heads,
                float scale, double sparsity, uint32_t prefix_blocks,
                char *error, size_t error_size);

/* The sparse forward the mask was built for: Sol-Attn's attention with the
 * exact/approximated split `h3d_sol_search` routed. One CTA per (query block,
 * head), four warps of 16 query rows each. Per group of 32 KV blocks the
 * group's approximated blocks are folded into the running softmax from the
 * pooled summaries in one pass, then its exact blocks are walked in increasing
 * block order, one 64-key block at a time:
 *
 *   s = q @ kc[n]^T * scale*log2(e)                 (`triton_ref/fwd.py:88-108`)
 *   approximated: p = exp2(s - m); o += p @ vc;
 *     row_sum += p * block_length    (`fwd.py:109-131`, vc is a SUM,
 *                                     `preprocess.py:82`)
 *   exact:        p = exp2(q @ k^T * scale*log2(e) - m); o += p @ v;
 *     row_sum += sum(p)                            (`fwd.py:132-170`)
 *   output = o / row_sum, BF16                                (`fwd.py:171-173`)
 *
 * An approximated block is folded, never skipped, and a group with no
 * approximated block leaves the running maximum untouched (`fwd.py:256-262`).
 * The mask is the plan's, so the sink range and the diagonal band arrive
 * already set and this function adds no policy of its own: it reads `kc`, `vc`
 * and `mask` and nothing else. Call `h3d_sol_search` for the same plan first.
 *
 * Deviations, named: BF16 `P` for both products, as the reference casts too
 * (`fwd.py:129,164`); one maximum per 64-key exact block; the port's `exp2`,
 * F32 accumulators and reduction trees against Triton's. `tests/h3/test_native.cu`
 * gates the walk against the reference's own rule. */
int h3d_sol_forward(h3d_context *context, h3d_tensor *output,
                    const struct h3d_sol_plan *plan, const h3d_tensor *query,
                    const h3d_tensor *key, const h3d_tensor *value,
                    uint32_t sequence, uint32_t heads, float scale, char *error,
                    size_t error_size);

/* ------------------------------------------------------------------------
 * h3b_* — the H3-to-LTX latent path (`minimax_h3::ltx_adapter`).
 *
 * The profile's `stage2_ops/h3_upscale.py` (the pinned author node's
 * `LatentResizer3D` wrapped in its own N/D) and `stage2_ops/h3_ltx_adapter/`
 * (the alignment and the 22-block factorized Conv3D adapter), ported. These
 * models have no C reference: the authority is that Python, read at
 * `/data/Sol-h3-Spark` and at the node's own pin `d7c01b90`.
 *
 * Every op takes `dtype` (`H3D_BF16` or `H3D_F32`). The reference runs both
 * models in bf16; the f32 arm exists so the port can be gated tightly against
 * an f32 reference vector. Activations are channel-major
 * `[channel][time][height][width]` and weights are PyTorch's own
 * `[out][in][kd][kh][kw]`, so no weight is ever transformed.
 * ------------------------------------------------------------------------ */

/* Zero-pad the temporal and spatial axes (`Conv3d(padding=...)`). */
int h3b_pad3d(h3d_context *context, h3d_tensor *output,
              const h3d_tensor *input, int dtype, uint32_t channels,
              uint32_t depth, uint32_t height, uint32_t width, uint32_t pad_t,
              uint32_t pad_h, uint32_t pad_w, char *error, size_t error_size);

/* The convolution's im2col: `(depth, height, width)` interior positions of the
 * padded tensor, starting at `t_begin`, into a `depth*height*width` by
 * `channels*kd*kh*kw` matrix. */
int h3b_im2col(h3d_context *context, h3d_tensor *output,
               const h3d_tensor *padded, int dtype, uint32_t channels,
               uint32_t pad_depth, uint32_t pad_height, uint32_t pad_width,
               uint32_t kd, uint32_t kh, uint32_t kw, uint32_t depth,
               uint32_t height, uint32_t width, uint32_t t_begin, char *error,
               size_t error_size);

/* The im2col-free convolution: `h3b_planar` writes the padded activation in
 * channel-last form once per kernel column tap, `h3b_weight_pack` writes the
 * weights in the pass decomposition's own order, and `h3b_matmul_planar` runs
 * one tap as a dense matmul whose A operand is a window of the planar buffer —
 * leading dimension `input_dim`, the window at `a_offset` elements into
 * `planar`, the result at `c_offset` elements into `output` — accumulating into
 * the same f32 `output` when `accumulate` is set. The A view is only valid for a
 * same-shape kernel, which `h3b_planar` checks: `out_width + kw - 1` must equal
 * the padded width. */
int h3b_planar(h3d_context *context, h3d_tensor *output,
               const h3d_tensor *input, int dtype, uint32_t channels,
               uint32_t depth, uint32_t height, uint32_t width, uint32_t kw,
               uint32_t pad_t, uint32_t pad_h, uint32_t pad_w,
               uint32_t out_width, char *error, size_t error_size);

int h3b_weight_pack(h3d_context *context, h3d_tensor *output,
                    const h3d_tensor *weight, int dtype, uint32_t out_channels,
                    uint32_t in_channels, uint32_t taps, uint32_t kw,
                    char *error, size_t error_size);

int h3b_matmul_planar(h3d_context *context, h3d_tensor *output,
                      const h3d_tensor *planar, size_t a_offset,
                      const h3d_tensor *weight, size_t b_offset,
                      size_t c_offset, uint32_t rows, uint32_t input_dim,
                      uint32_t output_dim, int accumulate, int dtype,
                      char *error, size_t error_size);

/* The two matmuls, both into an f32 accumulator: `h3b_matmul` for the im2col
 * form (`left` is `rows x input_dim`, `right` is `output_dim x input_dim`,
 * both row-major) and `h3b_matmul_channels` for a 1x1 convolution with no
 * copy at all (`input` is `input_dim x spatial` channel-major, `weight` is
 * `output_dim x input_dim`). */
int h3b_matmul(h3d_context *context, h3d_tensor *output,
               const h3d_tensor *left, const h3d_tensor *right, uint32_t rows,
               uint32_t input_dim, uint32_t output_dim, int dtype, char *error,
               size_t error_size);

int h3b_matmul_channels(h3d_context *context, h3d_tensor *output,
                        const h3d_tensor *input, const h3d_tensor *weight,
                        uint32_t spatial, uint32_t input_dim,
                        uint32_t output_dim, int dtype, char *error,
                        size_t error_size);

/* The convolution's tail: `(spatial, channels)` f32 into channel-major `T`,
 * bias added in f32, one rounding. `row_stride` and `row_offset` place the
 * tile inside a larger channel-major tensor, which is how the convolution
 * runs in temporal slabs. */
int h3b_epilogue(h3d_context *context, h3d_tensor *output,
                 const h3d_tensor *matrix, const h3d_tensor *bias, int dtype,
                 uint32_t channels, uint32_t spatial, size_t row_stride,
                 size_t row_offset, char *error, size_t error_size);

/* `nn.GroupNorm`, which for a 5-D input reduces over the whole
 * `(channels/groups, time, height, width)` volume per group, with the
 * ResBlock epilogues folded in: `embedding` non-NULL applies
 * `norm * (1 + scale) + shift` from its two halves, `silu` applies SiLU after.
 * `partial`, `mean` and `rstd` are f32 scratch the caller sizes: `partial`
 * needs `groups * (channels/groups) * ceil(rows*spatial/8192) * 2` elements and
 * `mean`/`rstd` need `groups` each. */
int h3b_norm(h3d_context *context, h3d_tensor *output, const h3d_tensor *input,
             const h3d_tensor *weight, const h3d_tensor *bias,
             const h3d_tensor *embedding, h3d_tensor *partial,
             h3d_tensor *mean, h3d_tensor *rstd, int dtype, uint32_t channels,
             uint32_t groups, uint32_t rows, uint32_t spatial, float epsilon,
             int silu, char *error, size_t error_size);

int h3b_silu(h3d_context *context, h3d_tensor *output, const h3d_tensor *input,
             int dtype, size_t elements, char *error, size_t error_size);

int h3b_add(h3d_context *context, h3d_tensor *output, const h3d_tensor *left,
            const h3d_tensor *right, int dtype, size_t elements, char *error,
            size_t error_size);

/* A depthwise `(kernel, 1, 1)` convolution with `groups == channels` and
 * symmetric temporal padding. */
int h3b_depthwise_temporal(h3d_context *context, h3d_tensor *output,
                           const h3d_tensor *input, const h3d_tensor *weight,
                           const h3d_tensor *bias, int dtype, uint32_t channels,
                           uint32_t rows, uint32_t spatial, uint32_t kernel,
                           uint32_t pad, char *error, size_t error_size);

/* `F.interpolate(..., mode="trilinear", align_corners=False)`. */
int h3b_resize3d(h3d_context *context, h3d_tensor *output,
                 const h3d_tensor *input, int dtype, uint32_t channels,
                 uint32_t rows, uint32_t height, uint32_t width,
                 uint32_t out_rows, uint32_t out_height, uint32_t out_width,
                 char *error, size_t error_size);

/* Take target frame `index[target*slots + slot]` of the source
 * (`0xffffffff` writes zero): the nearest pack and the replicate padding. */
int h3b_time_gather(h3d_context *context, h3d_tensor *output,
                    const h3d_tensor *input, const h3d_tensor *index, int dtype,
                    uint32_t in_channels, uint32_t source_frames,
                    uint32_t target_frames, uint32_t spatial, uint32_t slots,
                    char *error, size_t error_size);

/* `a + w * (b - a)` between two source frames. */
int h3b_time_lerp(h3d_context *context, h3d_tensor *output,
                  const h3d_tensor *input, const h3d_tensor *left,
                  const h3d_tensor *right, const h3d_tensor *weight, int dtype,
                  uint32_t channels, uint32_t source_frames,
                  uint32_t target_frames, uint32_t spatial, char *error,
                  size_t error_size);

/* `F.pixel_unshuffle(x, 2)` over `(height, width)`, giving four times the
 * channels. */
int h3b_unshuffle2(h3d_context *context, h3d_tensor *output,
                   const h3d_tensor *input, int dtype, uint32_t in_channels,
                   uint32_t rows, uint32_t height, uint32_t width, char *error,
                   size_t error_size);

/* The chunked upscaler's `out_full += segment * weight` and its
 * `out_full / weight_full.clamp(min=1e-8)`; the accumulator is f32 and holds
 * every output frame. */
int h3b_accumulate(h3d_context *context, h3d_tensor *accumulator,
                   const h3d_tensor *segment, const h3d_tensor *weight,
                   int dtype, uint32_t channels, uint32_t frames,
                   uint32_t segment_frames, uint32_t spatial, uint32_t in_frames,
                   uint32_t in_begin, uint32_t out_begin, char *error,
                   size_t error_size);

int h3b_blend_finish(h3d_context *context, h3d_tensor *output,
                     const h3d_tensor *accumulator, const h3d_tensor *weight,
                     int dtype, uint32_t channels, uint32_t frames,
                     uint32_t spatial, char *error, size_t error_size);

/* `value * silu(gate)` over the two halves of a `2*half` channel tensor. */
int h3b_gate_mul(h3d_context *context, h3d_tensor *output,
                 const h3d_tensor *input, int dtype, uint32_t half,
                 size_t plane, char *error, size_t error_size);

/* The node's `N` and `D`: `(x - center) / scale` and `x * scale + center`,
 * applied per channel. */
int h3b_scale(h3d_context *context, h3d_tensor *output, const h3d_tensor *input,
              const h3d_tensor *center, const h3d_tensor *scale, int dtype,
              uint32_t channels, size_t plane, int denormalize, char *error,
              size_t error_size);

/* A plain copy into `output` at `out_offset` elements from `input` at
 * `in_offset`: `torch.cat` over the channel axis. */
int h3b_copy_offset(h3d_context *context, h3d_tensor *output,
                    const h3d_tensor *input, int dtype, size_t out_offset,
                    size_t in_offset, size_t elements, char *error,
                    size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
