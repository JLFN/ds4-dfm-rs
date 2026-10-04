/* Included by the CUDA backend.  Qwen-Image-2.1 entry points: the DiT
 * primitives in cuda/qwen_image_primitives.cuh, the DiT position/attention path
 * in cuda/qwen_image_attn.cuh, and the VAE decoder primitives in
 * cuda/qwen_image_vae.cuh, validated and launched on the ds4_gpu tensor
 * surface.  Together with the host-built sinusoidal timestep table and the
 * tree's existing GEMMs, that is every op the DiT graph of P3 needs; the VAE
 * graph of P4 composes the decoder primitives with the same GEMMs. */

#include "cuda/qwen_image_attn.cuh"
#include "cuda/qwen_image_vae.cuh"

namespace qwen_image_gpu {

static bool tensor(const ds4_gpu_tensor *t, uint64_t bytes) {
    return t && t->ptr && bytes <= t->bytes;
}

static int launched(const char *what) {
    return cuda_ok(cudaGetLastError(), what);
}

/* tokens on x, feature blocks on y: matches the kernel indexing and keeps a
 * prefill's thousands of token rows off the 65535 blockIdx.y limit. */
static dim3 grid(uint32_t tokens, uint32_t features) {
    return dim3(tokens, (features + qwen_image_cuda::kThreads - 1u) /
                                qwen_image_cuda::kThreads);
}

/* transposes tile their own way: x walks the source's feature axis, y its
 * rows, and a = the feature extent (transpose_2d's first dimension). */
static dim3 tile_grid(uint32_t a, uint32_t b) {
    return dim3((a + qwen_image_cuda::kTile - 1u) / qwen_image_cuda::kTile,
                (b + qwen_image_cuda::kTile - 1u) / qwen_image_cuda::kTile);
}

static void transpose(ds4_gpu_tensor *dst, const ds4_gpu_tensor *src,
                      uint32_t a, uint32_t b) {
    qwen_image_cuda::transpose_2d<<<tile_grid(a, b),
            dim3(qwen_image_cuda::kTile, qwen_image_cuda::kTile), 0,
            cuda_decode_stream()>>>((float *) dst->ptr,
                                    (const float *) src->ptr, a, b);
}

} // namespace qwen_image_gpu

/* In-place affine-free LayerNorm over the feature axis, eps 1e-6. */
extern "C" int ds4_gpu_qwen_image_layernorm_tensor(
        ds4_gpu_tensor *x, uint32_t dim, uint32_t rows) {
    if (dim == 0 || rows == 0) return 0;
    if (!qwen_image_gpu::tensor(x, (uint64_t) dim * rows * sizeof(float))) {
        return 0;
    }
    float *buf = (float *) x->ptr;
    qwen_image_cuda::layernorm_rows<<<rows, qwen_image_cuda::kThreads,
            qwen_image_cuda::kThreads * sizeof(float), cuda_decode_stream()>>>(
            buf, buf, dim);
    return qwen_image_gpu::launched("Qwen-Image LayerNorm launch");
}

/* out = residual + x * (gated ? tanh(param[row]) : param[row] + 1), with the
 * parameter row chosen by the token's side of the prefix.  residual == NULL
 * skips the add; the result overwrites x.  param is [hidden, 2]. */
extern "C" int ds4_gpu_qwen_image_modulate_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *param,
        ds4_gpu_tensor       *residual,
        uint32_t              hidden,
        uint32_t              tokens,
        uint32_t              prefix,
        uint32_t              gated) {
    if (hidden == 0 || tokens == 0 || prefix > tokens || gated > 1) return 0;
    const uint64_t count = (uint64_t) hidden * tokens;
    if (!qwen_image_gpu::tensor(x, count * sizeof(float))) return 0;
    if (!qwen_image_gpu::tensor(param, (uint64_t) hidden * 2u * sizeof(float))) {
        return 0;
    }
    if (residual && !qwen_image_gpu::tensor(residual, count * sizeof(float))) {
        return 0;
    }
    qwen_image_cuda::modulate_rows<<<qwen_image_gpu::grid(tokens, hidden),
            qwen_image_cuda::kThreads, 0, cuda_decode_stream()>>>(
            (float *) x->ptr, (const float *) param->ptr,
            residual ? (const float *) residual->ptr : NULL, hidden, prefix,
            gated);
    return qwen_image_gpu::launched("Qwen-Image modulate launch");
}

/* out = up * silu(gate) for separate [n, rows] projections. */
extern "C" int ds4_gpu_qwen_image_mlp_gated_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        uint32_t              n,
        uint32_t              rows) {
    if (n == 0 || rows == 0) return 0;
    const uint64_t count = (uint64_t) n * rows;
    if (!qwen_image_gpu::tensor(out, count * sizeof(float))) return 0;
    if (!qwen_image_gpu::tensor(gate, count * sizeof(float))) return 0;
    if (!qwen_image_gpu::tensor(up, count * sizeof(float))) return 0;
    qwen_image_cuda::mlp_gated_rows<<<qwen_image_gpu::grid(rows, n),
            qwen_image_cuda::kThreads, 0, cuda_decode_stream()>>>(
            (float *) out->ptr, (const float *) gate->ptr,
            (const float *) up->ptr, n);
    return qwen_image_gpu::launched("Qwen-Image gated MLP launch");
}

/* Same op from the fused [2n, rows] projection: gate in feature chunk 0, up
 * in chunk 1. */
extern "C" int ds4_gpu_qwen_image_mlp_gated_fused_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *fused,
        uint32_t              n,
        uint32_t              rows) {
    if (n == 0 || rows == 0) return 0;
    const uint64_t count = (uint64_t) n * rows;
    if (!qwen_image_gpu::tensor(out, count * sizeof(float))) return 0;
    if (!qwen_image_gpu::tensor(fused, 2u * count * sizeof(float))) return 0;
    qwen_image_cuda::mlp_gated_fused_rows<<<qwen_image_gpu::grid(rows, n),
            qwen_image_cuda::kThreads, 0, cuda_decode_stream()>>>(
            (float *) out->ptr, (const float *) fused->ptr, n);
    return qwen_image_gpu::launched("Qwen-Image fused gated MLP launch");
}

/* In place 3-axis rope over one [n_head][tokens][head_dim] q or k, from a
 * prebuilt [tokens, head_dim/2, 2, 2] table (oracle::rope_table). */
extern "C" int ds4_gpu_qwen_image_rope3d_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *pe,
        uint32_t              tokens,
        uint32_t              n_head,
        uint32_t              head_dim) {
    if (tokens == 0 || n_head == 0 || head_dim == 0 || head_dim % 2u != 0) {
        return 0;
    }
    const uint64_t count = (uint64_t) n_head * tokens * head_dim;
    if (!qwen_image_gpu::tensor(x, count * sizeof(float))) return 0;
    if (!qwen_image_gpu::tensor(pe, (uint64_t) tokens * head_dim * 2u * sizeof(float))) {
        return 0;
    }
    const uint32_t rows = n_head * tokens;
    const uint32_t pairs = head_dim / 2u;
    const uint64_t threads = (uint64_t) rows * pairs;
    if (threads > 0xffffffffull) return 0; /* the kernel indexes threads in u32 */
    qwen_image_cuda::rope3d_rows<<<
            (uint32_t)((threads + qwen_image_cuda::kThreads - 1u) /
                       qwen_image_cuda::kThreads),
            qwen_image_cuda::kThreads, 0, cuda_decode_stream()>>>(
            (float *) x->ptr, (const float *) pe->ptr, rows, tokens, pairs);
    return qwen_image_gpu::launched("Qwen-Image rope3d launch");
}

/* One attention segment: queries [start, end) attend to keys [0, end).
 * causal != 0 masks key > query (the text prefix); an image segment passes 0
 * and stays bidirectional. */
extern "C" int ds4_gpu_qwen_image_attn_segment_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        uint32_t              tokens,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              start,
        uint32_t              end,
        uint32_t              causal) {
    if (tokens == 0 || n_head == 0 || head_dim == 0) return 0;
    if (start >= end || end > tokens) return 0;
    /* dit.rs masks only the text prefix, which starts the sequence; any
     * other masked segment would need a different mask indexing. */
    if (causal > 1u || (causal && start != 0u)) return 0;
    if (end > qwen_image_cuda::kMaxSegmentKeys) return 0;

    const uint64_t count = (uint64_t) tokens * n_head * head_dim;
    if (!qwen_image_gpu::tensor(out, count * sizeof(float))) return 0;
    if (!qwen_image_gpu::tensor(v, count * sizeof(float))) return 0;
    if (!qwen_image_gpu::tensor(q, count * sizeof(float))) return 0;
    if (!qwen_image_gpu::tensor(k, count * sizeof(float))) return 0;

    const dim3 grid(end - start, n_head);
    const uint32_t shared =
            (end + qwen_image_cuda::kThreads) * (uint32_t) sizeof(float);
    qwen_image_cuda::attn_segment_rows<<<grid, qwen_image_cuda::kThreads,
            shared, cuda_decode_stream()>>>(
            (float *) out->ptr, (const float *) q->ptr, (const float *) k->ptr,
            (const float *) v->ptr, tokens, n_head, head_dim, start, end,
            causal);
    return qwen_image_gpu::launched("Qwen-Image segment attention launch");
}

/* In-place x = x / (1 + exp(-x)) over a feature-fastest [dim, rows] tensor:
 * the two activations of the timestep path's 256 -> 4096 MLP (dit.rs:
 * linear_1 -> silu -> linear_2 -> silu).
 *
 * The 256-wide sinusoidal table that feeds it is built on the host and
 * uploaded, exactly as the RoPE `pe` table is, because under this tree's
 * --use_fast_math the device cosf/sinf measure 5.5e-4 max abs against libm
 * over [0, 4224] rad (unit 9), five times the 1e-4 gate.  The table is 512
 * floats per denoise step and the host already owns the timestep, so no
 * kernel exists for it.  Contract (dit.rs::timestep_embedding): per column,
 * feature j < 128 is cos(t * 10000^(-j/128)) and feature j >= 128 is the same
 * angle's sin; column 0 is the flow timestep, column 1 is zero. */
extern "C" int ds4_gpu_qwen_image_silu_tensor(
        ds4_gpu_tensor *x, uint32_t dim, uint32_t rows) {
    if (dim == 0 || rows == 0) return 0;
    if (!qwen_image_gpu::tensor(x, (uint64_t) dim * rows * sizeof(float))) {
        return 0;
    }
    qwen_image_cuda::silu_rows<<<qwen_image_gpu::grid(rows, dim),
            qwen_image_cuda::kThreads, 0, cuda_decode_stream()>>>(
            (float *) x->ptr, dim);
    return qwen_image_gpu::launched("Qwen-Image silu launch");
}

/* The 1x1 patchify of dit.rs::forward: the latent's channel-slowest layout is
 * gathered into the feature-fastest token matrix the img_in GEMM consumes.
 * Drop the two layouts onto one element index p = the latent's pixel:
 *
 *      latent [pixels, channels]  src[p + pixels*c]     c = channel, p = pixel
 *      tokens [channels, pixels]  dst[c + channels*p]
 *
 * so this is transpose_2d with a = pixels and b = channels, no arithmetic. */
extern "C" int ds4_gpu_qwen_image_patch_1x1_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *src,
        uint32_t              channels,
        uint32_t              pixels) {
    if (channels == 0 || pixels == 0) return 0;
    const uint64_t count = (uint64_t) channels * pixels;
    if (!qwen_image_gpu::tensor(dst, count * sizeof(float))) return 0;
    if (!qwen_image_gpu::tensor(src, count * sizeof(float))) return 0;
    qwen_image_gpu::transpose(dst, src, pixels, channels);
    return qwen_image_gpu::launched("Qwen-Image patch launch");
}

/* The inverse: proj_out's channel-fastest [channels, pixels] back into the
 * latent's own [pixels, channels], transpose_2d with the roles swapped.
 *
 * The reference calls this unpatchify_and_crop(x, H, W, 1, 1) and at a 1x1
 * patch its crop is empty for every H and W: pad_h = (1 - H%1)%1 = 0 and
 * pad_w = 0, so unpatchify steps h = H, w = W and both trailing
 * ggml_ext_slice calls ask for the full extent (dit.hpp:94-104).  The crop
 * therefore has no parameters and needs no code of its own; the image tokens
 * the DiT emits are already the caller's own grid. */
extern "C" int ds4_gpu_qwen_image_unpatch_crop_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *src,
        uint32_t              channels,
        uint32_t              pixels) {
    if (channels == 0 || pixels == 0) return 0;
    const uint64_t count = (uint64_t) channels * pixels;
    if (!qwen_image_gpu::tensor(dst, count * sizeof(float))) return 0;
    if (!qwen_image_gpu::tensor(src, count * sizeof(float))) return 0;
    qwen_image_gpu::transpose(dst, src, channels, pixels);
    return qwen_image_gpu::launched("Qwen-Image unpatch launch");
}

/* ---------------------------------------------------------------------------
 * The VAE decoder primitives (cuda/qwen_image_vae.cuh).  Planes are
 * channel-slowest [pixels, channels]; the feature-fastest token matrix the
 * GEMMs consume is built by patch_1x1/unpatch_crop above.
 * ------------------------------------------------------------------------- */

/* The 3x3 conv's im2col: plane [pixels, IC] -> patches [9*IC, pixels],
 * feature j = tap + 9*ic, zero padding at the borders.  The result is the
 * activation for ds4_gpu_matmul_f16_tensor at the weight's own map offset. */
extern "C" int ds4_gpu_qwen_image_im2col3x3_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *src,
        uint32_t              channels,
        uint32_t              width,
        uint32_t              height) {
    if (channels == 0 || width == 0 || height == 0) return 0;
    const uint64_t pixels = (uint64_t) width * height;
    if (!qwen_image_gpu::tensor(src, pixels * channels * sizeof(float))) {
        return 0;
    }
    if (pixels > UINT64_MAX / (qwen_image_cuda::kConvTaps * (uint64_t) channels)) {
        return 0;
    }
    const uint64_t patches = qwen_image_cuda::kConvTaps * channels * pixels;
    if (!qwen_image_gpu::tensor(dst, patches * sizeof(float))) return 0;

    const uint32_t blocks = (uint32_t)((patches + qwen_image_cuda::kThreads - 1u) /
                                       qwen_image_cuda::kThreads);
    qwen_image_cuda::im2col3x3_kernel<<<blocks, qwen_image_cuda::kThreads, 0,
            cuda_decode_stream()>>>((float *) dst->ptr,
                                    (const float *) src->ptr, channels, width,
                                    height);
    return qwen_image_gpu::launched("Qwen-Image VAE im2col launch");
}

/* Per-output-channel bias add on a feature-fastest [dim, rows] tensor: the
 * oracle adds the bias only after the conv accumulation (vae.rs:517-520). */
extern "C" int ds4_gpu_qwen_image_bias_add_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *bias,
        uint32_t              dim,
        uint32_t              rows) {
    if (dim == 0 || rows == 0) return 0;
    if (!qwen_image_gpu::tensor(out, (uint64_t) dim * rows * sizeof(float))) {
        return 0;
    }
    if (!qwen_image_gpu::tensor(bias, (uint64_t) dim * sizeof(float))) return 0;
    qwen_image_cuda::bias_add_rows<<<qwen_image_gpu::grid(rows, dim),
            qwen_image_cuda::kThreads, 0, cuda_decode_stream()>>>(
            (float *) out->ptr, (const float *) bias->ptr, dim);
    return qwen_image_gpu::launched("Qwen-Image VAE bias add launch");
}

/* Channel-wise RMSNorm, eps 1e-12, per-channel gamma, in place on a plane. */
extern "C" int ds4_gpu_qwen_image_vae_rms_norm_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *gamma,
        uint32_t              channels,
        uint32_t              pixels) {
    if (channels == 0 || pixels == 0) return 0;
    if (!qwen_image_gpu::tensor(x, (uint64_t) channels * pixels * sizeof(float))) {
        return 0;
    }
    if (!qwen_image_gpu::tensor(gamma, (uint64_t) channels * sizeof(float))) {
        return 0;
    }
    qwen_image_cuda::vae_rms_norm<<<pixels, qwen_image_cuda::kThreads,
            qwen_image_cuda::kThreads * sizeof(float), cuda_decode_stream()>>>(
            (float *) x->ptr, (const float *) gamma->ptr, channels, pixels);
    return qwen_image_gpu::launched("Qwen-Image VAE RMSNorm launch");
}

/* Nearest 2x spatial upsample of a plane. */
extern "C" int ds4_gpu_qwen_image_nearest_up2_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *src,
        uint32_t              channels,
        uint32_t              width,
        uint32_t              height) {
    if (channels == 0 || width == 0 || height == 0) return 0;
    const uint64_t pixels = (uint64_t) width * height;
    if (!qwen_image_gpu::tensor(src, pixels * channels * sizeof(float))) {
        return 0;
    }
    const uint64_t out_pixels = 4u * pixels;
    if (!qwen_image_gpu::tensor(dst, out_pixels * channels * sizeof(float))) {
        return 0;
    }
    const dim3 grid((uint32_t)((out_pixels + qwen_image_cuda::kThreads - 1u) /
                               qwen_image_cuda::kThreads),
                    channels);
    qwen_image_cuda::nearest_up2_gather<<<grid, qwen_image_cuda::kThreads, 0,
            cuda_decode_stream()>>>((float *) dst->ptr,
                                    (const float *) src->ptr, width, height);
    return qwen_image_gpu::launched("Qwen-Image VAE nearest up2 launch");
}

/* DupUp3D at one frame: a factor_s = 2 nearest gather whose channel grouping
 * is set by the temporal factor.  Refused when out*factor % in != 0, where the
 * reference asserts (wan_vae.hpp:336). */
extern "C" int ds4_gpu_qwen_image_dup_up3d_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *src,
        uint32_t              cin,
        uint32_t              cout,
        uint32_t              width,
        uint32_t              height,
        uint32_t              factor_t) {
    if (cin == 0 || cout == 0 || width == 0 || height == 0 || factor_t == 0) {
        return 0;
    }
    const uint64_t factor = (uint64_t) qwen_image_cuda::kUpFactorS *
                            qwen_image_cuda::kUpFactorS * factor_t;
    if (((uint64_t) cout * factor) % cin != 0) return 0;
    const uint32_t repeats = (uint32_t)(((uint64_t) cout * factor) / cin);

    const uint64_t pixels = (uint64_t) width * height;
    if (!qwen_image_gpu::tensor(src, pixels * cin * sizeof(float))) return 0;
    const uint64_t out_pixels = 4u * pixels;
    if (!qwen_image_gpu::tensor(dst, out_pixels * cout * sizeof(float))) {
        return 0;
    }
    const dim3 grid((uint32_t)((out_pixels + qwen_image_cuda::kThreads - 1u) /
                               qwen_image_cuda::kThreads),
                    cout);
    qwen_image_cuda::dup_up3d_gather<<<grid, qwen_image_cuda::kThreads, 0,
            cuda_decode_stream()>>>((float *) dst->ptr,
                                    (const float *) src->ptr, width, height,
                                    factor_t, repeats);
    return qwen_image_gpu::launched("Qwen-Image VAE DupUp3D launch");
}

/* One head of the AttentionBlock's attention over all tokens.  qkv is the
 * feature-fastest [3*channels, tokens] output of the to_qkv conv (q, k, v in
 * that channel order); out is feature-fastest [channels, tokens].  Refused
 * above kMaxAttentionPixels (the score vector's shared staging). */
extern "C" int ds4_gpu_qwen_image_vae_attn_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *qkv,
        uint32_t              channels,
        uint32_t              tokens) {
    if (channels == 0 || tokens == 0) return 0;
    if (tokens > qwen_image_cuda::kMaxAttentionPixels) return 0;
    if (!qwen_image_gpu::tensor(qkv, (uint64_t) 3u * channels * tokens * sizeof(float))) {
        return 0;
    }
    if (!qwen_image_gpu::tensor(out, (uint64_t) channels * tokens * sizeof(float))) {
        return 0;
    }
    const uint32_t shared =
            (tokens + qwen_image_cuda::kThreads) * (uint32_t) sizeof(float);
    qwen_image_cuda::vae_attn_head<<<tokens, qwen_image_cuda::kThreads, shared,
            cuda_decode_stream()>>>((float *) out->ptr,
                                    (const float *) qkv->ptr, channels, tokens);
    return qwen_image_gpu::launched("Qwen-Image VAE attention launch");
}

